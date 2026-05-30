"""
Baselines for the REnFormer paper (Table 2):
  - Persistence
  - Per-site MLP  (256–128, ReLU, fitted independently per site)
  - Per-site LSTM (128 hidden units, fitted independently per site)

Per-site models are trained on individual-site sliding windows extracted
from the shared SENDataset.  Use max_sites to cap the number of sites
evaluated (useful for smoke-tests).
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from typing import Sequence


BATCH  = 64
EPOCHS = 20
LR     = 1e-3


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persistence_forecast(X: np.ndarray, horizon: int) -> np.ndarray:
    """
    Repeat the last observed value for every horizon step.
    X shape: (N, lookback, features).  Output: (N, horizon, features).
    """
    last = X[:, -1:, :]                             # (N, 1, F)
    return np.broadcast_to(last, (X.shape[0], horizon, X.shape[2])).copy()


# ---------------------------------------------------------------------------
# Per-site MLP
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    hidden: Sequence[int]
    horizon: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        return nn.Dense(self.horizon)(x)[:, :, None]   # (B, H, 1)


def _fit_site_model(model, X_site, Y_site, epochs=EPOCHS, lr=LR, seed=0):
    """Generic JAX training loop for a single-site Flax model."""
    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, X_site[:2])
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        def loss_fn(p):
            return jnp.mean((model.apply(p, x) - y) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_state

    n = X_site.shape[0]
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, BATCH):
            idx = perm[start: start + BATCH]
            params, opt_state = step(
                params, opt_state,
                jnp.array(X_site[idx]),
                jnp.array(Y_site[idx]),
            )
    return params


def run_per_site_mlp(train_ds, val_ds, test_ds, horizon=24, max_sites=None):
    """
    Train one MLP per site; return concatenated predictions on test_ds.

    Returns: (Y_true_raw, Y_pred_mu) arrays shaped (N_test_windows, H, 1).
    """
    n_sites = train_ds.S if max_sites is None else min(max_sites, train_ds.S)
    all_true, all_pred = [], []

    for s in range(n_sites):
        # Extract this site's training windows
        t_idx = np.arange(train_ds.n_windows)
        s_arr = np.full_like(t_idx, s)
        X_tr, Y_norm_tr, _, _, _ = train_ds._fetch(s_arr, t_idx)
        Y_tr = Y_norm_tr[:, :, 0]   # (N, H)  — predict normalised target

        model = _MLP(hidden=(256, 128), horizon=horizon)
        params = _fit_site_model(model, X_tr, Y_tr)

        # Evaluate on test windows for this site
        t_idx_te = np.arange(test_ds.n_windows)
        s_arr_te = np.full_like(t_idx_te, s)
        X_te, _, Y_raw_te, _, _ = test_ds._fetch(s_arr_te, t_idx_te)

        pred = np.array(model.apply(params, jnp.array(X_te)))   # (N, H, 1)
        all_true.append(Y_raw_te)
        all_pred.append(pred)

        if (s + 1) % 50 == 0:
            print(f"  MLP: fitted {s + 1}/{n_sites} sites")

    return np.concatenate(all_true), np.concatenate(all_pred)


# ---------------------------------------------------------------------------
# Per-site LSTM (GRU cell — simpler carry, same expressive power for this task)
# ---------------------------------------------------------------------------

class _LSTMForecast(nn.Module):
    hidden: int
    horizon: int

    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_len, features)
        cell = nn.LSTMCell(self.hidden)
        # initialize_carry expects input_shape = shape of one time step
        carry = cell.initialize_carry(
            jax.random.PRNGKey(0), x[:, 0, :].shape
        )
        for t in range(x.shape[1]):
            carry, _ = cell(carry, x[:, t, :])
        h = carry[1]                             # LSTM carry is (c, h)
        return nn.Dense(self.horizon)(h)[:, :, None]   # (B, H, 1)


def run_per_site_lstm(train_ds, test_ds, horizon=24, hidden=128, max_sites=None):
    """
    Train one LSTM per site; return concatenated predictions on test_ds.

    Returns: (Y_true_raw, Y_pred_mu) arrays shaped (N_test_windows, H, 1).
    """
    n_sites = train_ds.S if max_sites is None else min(max_sites, train_ds.S)
    all_true, all_pred = [], []

    for s in range(n_sites):
        t_idx = np.arange(train_ds.n_windows)
        s_arr = np.full_like(t_idx, s)
        X_tr, Y_norm_tr, _, _, _ = train_ds._fetch(s_arr, t_idx)
        Y_tr = Y_norm_tr[:, :, 0]

        model = _LSTMForecast(hidden=hidden, horizon=horizon)
        params = _fit_site_model(model, X_tr, Y_tr)

        t_idx_te = np.arange(test_ds.n_windows)
        s_arr_te = np.full_like(t_idx_te, s)
        X_te, _, Y_raw_te, _, _ = test_ds._fetch(s_arr_te, t_idx_te)

        pred = np.array(model.apply(params, jnp.array(X_te)))
        all_true.append(Y_raw_te)
        all_pred.append(pred)

        if (s + 1) % 50 == 0:
            print(f"  LSTM: fitted {s + 1}/{n_sites} sites")

    return np.concatenate(all_true), np.concatenate(all_pred)
