import jax
import jax.numpy as jnp
import optax
import numpy as np

EPSILON_MW = 0.1   # intermittency threshold (raw MW, paper Section 3)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def masked_gaussian_nll(mean, log_std, target, mask):
    """
    Masked NLL (Eq. 1 in paper). Only penalises timesteps where mask > 0
    (i.e. raw generation > epsilon, meaning non-trivial active generation).
    """
    var = jnp.exp(2.0 * log_std)
    nll = 0.5 * ((target - mean) ** 2 / var + 2.0 * log_std + jnp.log(2 * jnp.pi))
    return (nll * mask).sum() / (mask.sum() + 1e-6)


def mse_loss(mean, log_std, target, mask=None):
    """Unmasked MSE — used for the REnFormer-MSE ablation."""
    return jnp.mean((target - mean) ** 2)


# ---------------------------------------------------------------------------
# JIT-compiled step factories (close over model / optimizer so jit can trace)
# ---------------------------------------------------------------------------

def make_train_step(model, optimizer, loss_fn=masked_gaussian_nll):
    @jax.jit
    def train_step(params, opt_state, x, y_norm, mask, rng):
        def _loss(params):
            mean, log_std = model.apply(params, x, train=True, rngs={"dropout": rng})
            return loss_fn(mean, log_std, y_norm, mask)
        loss, grads = jax.value_and_grad(_loss)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), new_opt_state, loss
    return train_step


def make_eval_step(model):
    @jax.jit
    def eval_step(params, x):
        return model.apply(params, x, train=False)
    return eval_step


# ---------------------------------------------------------------------------
# Dataset — lazy sliding-window sampler (avoids materialising all windows)
# ---------------------------------------------------------------------------

class SENDataset:
    """
    Lazy sliding-window dataset.  Stores (S, T) arrays in memory and samples
    (site, time) pairs on the fly — avoids the ~14 GB materialization of all
    windows.
    """
    def __init__(self, df_norm, df_raw, lookback=168, horizon=24):
        # (S, T) layout for fast per-site slicing
        self.arr_norm = df_norm.values.T.astype(np.float32)
        self.arr_raw  = df_raw.values.T.astype(np.float32)
        self.site_names = df_norm.columns.tolist()
        self.lookback = lookback
        self.horizon  = horizon
        self.S, self.T = self.arr_norm.shape
        # number of valid start positions (one window = lookback + horizon)
        self.n_windows = self.T - lookback - horizon + 1

    def __len__(self):
        return self.S * self.n_windows

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        s_idx = rng.integers(0, self.S, size=batch_size)
        t_idx = rng.integers(0, self.n_windows, size=batch_size)
        return self._fetch(s_idx, t_idx)

    def sequential_batches(self, batch_size: int = 512):
        """Yields all windows in (site-major, time-minor) order for evaluation."""
        all_s = np.repeat(np.arange(self.S), self.n_windows)
        all_t = np.tile(np.arange(self.n_windows), self.S)
        for start in range(0, len(all_s), batch_size):
            sl = slice(start, start + batch_size)
            yield self._fetch(all_s[sl], all_t[sl])

    def _fetch(self, s_idx, t_idx):
        X = np.stack([
            self.arr_norm[s, t: t + self.lookback]
            for s, t in zip(s_idx, t_idx)
        ])[:, :, np.newaxis]                              # (B, L, 1)

        Y_norm = np.stack([
            self.arr_norm[s, t + self.lookback: t + self.lookback + self.horizon]
            for s, t in zip(s_idx, t_idx)
        ])[:, :, np.newaxis]                              # (B, H, 1)

        Y_raw = np.stack([
            self.arr_raw[s, t + self.lookback: t + self.lookback + self.horizon]
            for s, t in zip(s_idx, t_idx)
        ])[:, :, np.newaxis]

        mask = (Y_raw > EPSILON_MW).astype(np.float32)
        return (
            X.astype(np.float32),
            Y_norm.astype(np.float32),
            Y_raw.astype(np.float32),
            mask,
            s_idx,
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model,
    train_ds: SENDataset,
    val_ds: SENDataset,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    steps_per_epoch: int = 1000,
    seed: int = 0,
    loss_fn=masked_gaussian_nll,
):
    rng_np  = np.random.default_rng(seed)
    rng_jax = jax.random.PRNGKey(seed)

    # Initialise with a dummy batch
    dummy_x = jnp.zeros((1, model.max_len, model.out_features))
    params  = model.init(rng_jax, dummy_x, train=False)

    schedule  = optax.cosine_decay_schedule(lr, decay_steps=epochs * steps_per_epoch)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    train_step = make_train_step(model, optimizer, loss_fn)
    eval_step  = make_eval_step(model)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # --- train ---
        train_losses = []
        for _ in range(steps_per_epoch):
            x_b, y_b, _, mask_b, _ = train_ds.sample_batch(batch_size, rng_np)
            rng_jax, step_rng = jax.random.split(rng_jax)
            params, opt_state, loss = train_step(
                params, opt_state,
                jnp.array(x_b), jnp.array(y_b), jnp.array(mask_b), step_rng,
            )
            train_losses.append(float(loss))

        # --- validate (one pass over val dataset) ---
        val_losses = []
        for x_b, y_b, _, mask_b, _ in val_ds.sequential_batches(batch_size * 4):
            mean, log_std = eval_step(params, jnp.array(x_b))
            val_loss = masked_gaussian_nll(mean, log_std, jnp.array(y_b), jnp.array(mask_b))
            val_losses.append(float(val_loss))

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | train NLL {t_loss:.4f} | val NLL {v_loss:.4f}")

    return params, history
