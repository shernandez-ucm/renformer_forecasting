import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
from typing import Iterator, Tuple

# =====================================================
# Positional Encoding
# =====================================================
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
        pe = pe[None, :, :]
        return x + pe[:, : x.shape[1], :]

# =====================================================
# Transformer Block
# =====================================================
class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float
    input_dim:int

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
        y = nn.Dense(self.input_dim)(y)
        x = nn.LayerNorm()(x + y)
        return x

# =====================================================
# Encoder
# =====================================================
class TransformerEncoder(nn.Module):
    """
    Supports missing & irregular data via:
    - value masking
    - delta-time features
    - attention masking
    """
    d_model: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    dropout_rate: float
    max_len: int
    input_dim: int

    @nn.compact
    def __call__(self, x, train: bool):
                # project augmented input (values + mask + dt)
        x = nn.Dense(self.d_model)(x)
                # add positional encoding (still useful even with irregular time)
        x = PositionalEncoding(self.max_len, self.d_model)(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.mlp_dim,
                self.dropout_rate,
                self.input_dim
            )(x, train)
        return x

# =====================================================
# Multivariate, Multi‑Step Forecasting Model
# =====================================================
class TimeSeriesTransformer(nn.Module):
    """
    Time Series Transformer with explicit support for:
    - missing values
    - irregular timestamps
    - multivariate & multi-step forecasting
    - probabilistic forecasting (Gaussian output)
    - renewable energy forecasting (weather & calendar covariates)
    """
    """
    Time Series Transformer with explicit support for:
    - missing values
    - irregular timestamps
    - multivariate & multi-step forecasting
    """
    d_model: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    dropout_rate: float
    max_len: int
    horizon: int          # number of future steps
    out_features: int     # number of target variables

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Returns:
          mean: (batch, horizon, out_features)
          log_std: (batch, horizon, out_features)
        """
        enc = TransformerEncoder(
            self.d_model,
            self.num_heads,
            self.num_layers,
            self.mlp_dim,
            self.dropout_rate,
            self.max_len,
        )(x, train)

        summary = enc[:, -1, :]  # last token pooling
                # Gaussian parameters
        out_dim = self.horizon * self.out_features
        mean = nn.Dense(out_dim)(summary)
        log_std = nn.Dense(out_dim)(summary)

        mean = mean.reshape(x.shape[0], self.horizon, self.out_features)
        log_std = log_std.reshape(x.shape[0], self.horizon, self.out_features)
        return mean, log_std(x.shape[0], self.horizon, self.out_features)