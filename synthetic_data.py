from renformer.train import  train
from renformer.model import TimeSeriesTransformer
import pandas as pd
import numpy as np
import jax.numpy as jnp 

# =====================================================
# Example Usage: Solar + Wind, 96h Horizon
# =====================================================
# Configuration tailored for joint solar + wind forecasting
# - Hourly resolution
# - 96-hour (4-day) horizon
# - Probabilistic outputs (Gaussian)

# =====================================================
def generate_data(n_samples = 2000,seq_len = 168,in_features = 2,horizon = 96,out_features = 2):
  
    # Energy observations (NaNs allowed for outages)
    X = np.random.randn(n_samples, seq_len, in_features).astype(np.float32)
    Y = np.random.randn(n_samples, horizon, out_features).astype(np.float32)
    return X, Y

def generate_model_and_train(X, Y,seq_len = 168,horizon = 96,out_features = 2):
    model = TimeSeriesTransformer(
        # Larger model for 96h horizon + weather-driven dynamics(
        d_model=128,
        num_heads=4,
        num_layers=4,
        mlp_dim=256,
        dropout_rate=0.1,
        max_len=seq_len,
        horizon=horizon,
        out_features=out_features,
    )

    trained_params = train(
        model,
        X,
        Y,
        epochs=10,
        batch_size=64,
        lr=3e-4,
    )
    return model, trained_params
