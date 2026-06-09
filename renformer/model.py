import jax.numpy as jnp
from flax import linen as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al. 2022).

    Normalises over the time axis (axis=1) per channel with learnable
    affine parameters gamma (scale) and beta (shift), both initialised to
    the identity (gamma=1, beta=0) so training starts from plain z-score.

    Forward:  x_norm = gamma * (x - mean) / std + beta
    Inverse:  x_raw  = (x_norm - beta) / gamma * std + mean

    Returns (x_norm, mean, std, gamma, beta) so the parent module can
    denormalise outputs without accessing submodule internals.
    """
    num_features: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        gamma = self.param('gamma', nn.initializers.ones,  (1, 1, self.num_features))
        beta  = self.param('beta',  nn.initializers.zeros, (1, 1, self.num_features))
        mean  = jnp.mean(x, axis=1, keepdims=True)          # (B, 1, F)
        std   = jnp.std(x,  axis=1, keepdims=True) + self.eps
        x_norm = (x - mean) / std * gamma + beta
        return x_norm, mean, std, gamma, beta


class PositionalEncoding(nn.Module):
    max_len: int
    d_model: int

    @nn.compact
    def __call__(self, x):
        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * -(jnp.log(10000.0) / self.d_model)
        )
        pe = jnp.zeros((self.max_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return x + pe[None, : x.shape[1], :]


class TransformerBlock(nn.Module):
    """Post-LayerNorm Transformer block (matches paper Eq. 2–4)."""
    d_model: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train: bool):
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
        )(x, deterministic=not train)
        x = nn.LayerNorm()(x + attn)

        y = nn.Dense(self.mlp_dim)(x)
        y = nn.gelu(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(self.d_model)(y)
        x = nn.LayerNorm()(x + y)
        return x


class TransformerEncoder(nn.Module):
    d_model: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    dropout_rate: float
    max_len: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(self.d_model)(x)
        x = PositionalEncoding(self.max_len, self.d_model)(x)
        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.mlp_dim,
                self.dropout_rate,
            )(x, train)
        return x


class SkipTransformerEncoder(nn.Module):
    """TransformerEncoder with a macro skip-connection from input to output.

    After the linear projection and positional encoding, the representation
    x_0 is saved.  The N transformer blocks are applied as usual, then x_0 is
    added back before a final LayerNorm.  This gives gradients a direct path
    to the projection layer, helping very deep stacks and reducing the risk
    of the attention layers being ignored early in training.
    """
    d_model: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    dropout_rate: float
    max_len: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(self.d_model)(x)
        x = PositionalEncoding(self.max_len, self.d_model)(x)
        skip = x
        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.mlp_dim,
                self.dropout_rate,
            )(x, train)
        return nn.LayerNorm()(x + skip)


class TimeSeriesTransformer(nn.Module):
    """
    REnFormer: global Transformer for probabilistic multi-site forecasting.

    Input  shape: (batch, seq_len, in_features)
        channels 0..out_features-1  = target power series
        channels out_features..     = exogenous/calendar features (optional)
    Output: (mean, log_std) each of shape (batch, horizon, out_features),
            always in raw (denormalised) units when instance_norm=True.

    RevIN is applied only to the power channels; exogenous features are
    passed through unchanged so their bounded [-1,1] encoding is preserved.
    """
    d_model: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    dropout_rate: float
    max_len: int
    horizon: int
    out_features: int = 1
    in_features: int = 1
    instance_norm: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        if self.instance_norm:
            power = x[..., :self.out_features]                      # (B, T, out_features)
            power_norm, mu, sd, gamma, beta = RevIN(self.out_features)(power)
            x = jnp.concatenate([power_norm, x[..., self.out_features:]], axis=-1)

        enc = TransformerEncoder(
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.mlp_dim,
            self.dropout_rate,
            self.max_len,
        )(x, train)

        summary = enc[:, -1, :]                                     # last-token pooling
        out_dim = self.horizon * self.out_features
        mean    = nn.Dense(out_dim)(summary).reshape(-1, self.horizon, self.out_features)
        log_std = nn.Dense(out_dim)(summary).reshape(-1, self.horizon, self.out_features)

        if self.instance_norm:
            _eps = 1e-5  # numerical stability for RevIN inverse — must stay constant
            # Invert affine then instance norm:  x = (x_norm - beta) / gamma * sd + mu
            mean    = (mean - beta) / (jnp.abs(gamma) + _eps) * sd + mu
            # sigma_raw = sigma_norm / |gamma| * sd  →  log_std shifts accordingly
            log_std = log_std + jnp.log(sd) - jnp.log(jnp.abs(gamma) + _eps)

        return mean, log_std


class TimeSeriesTransformerSkip(nn.Module):
    """REnFormer with a macro skip-connection across the entire encoder stack.

    Identical to TimeSeriesTransformer except the encoder uses
    SkipTransformerEncoder: the projected+PE representation is added back
    to the encoder output before the forecasting heads.  All other
    hyperparameters and the RevIN logic are unchanged, making this a
    drop-in replacement for ablation studies.
    """
    d_model: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    dropout_rate: float
    max_len: int
    horizon: int
    out_features: int = 1
    in_features: int = 1
    instance_norm: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        if self.instance_norm:
            power = x[..., :1]
            mu = jnp.mean(power, axis=1, keepdims=True)
            sd = jnp.std(power, axis=1, keepdims=True) + 1e-5
            x = jnp.concatenate([(power - mu) / sd, x[..., 1:]], axis=-1)

        enc = SkipTransformerEncoder(
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.mlp_dim,
            self.dropout_rate,
            self.max_len,
        )(x, train)

        summary = enc[:, -1, :]
        out_dim = self.horizon * self.out_features
        mean    = nn.Dense(out_dim)(summary).reshape(-1, self.horizon, self.out_features)
        log_std = nn.Dense(out_dim)(summary).reshape(-1, self.horizon, self.out_features)

        if self.instance_norm:
            mean    = mean * sd + mu
            log_std = log_std + jnp.log(sd)
        return mean, log_std
