import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np


class PositionalEncoding(nn.Module):
    max_len: int
    d_model: int

    @nn.compact
    def __call__(self, x):
        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * -(jnp.log(10000.0) / self.d_model)
        )
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
        y = nn.Dense(self.d_model)(y)   # output dim must match d_model for residual
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


class TimeSeriesTransformer(nn.Module):
    """
    REnFormer: global Transformer for probabilistic multi-site forecasting.

    Input  shape: (batch, seq_len, in_features)
    Output: (mean, log_std) each of shape (batch, horizon, out_features)
    """
    d_model: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    dropout_rate: float
    max_len: int
    horizon: int
    out_features: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        enc = TransformerEncoder(
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.mlp_dim,
            self.dropout_rate,
            self.max_len,
        )(x, train)

        summary = enc[:, -1, :]   # last-token pooling (batch, d_model)

        out_dim = self.horizon * self.out_features
        mean = nn.Dense(out_dim)(summary).reshape(-1, self.horizon, self.out_features)
        log_std = nn.Dense(out_dim)(summary).reshape(-1, self.horizon, self.out_features)
        return mean, log_std
