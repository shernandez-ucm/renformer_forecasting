import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd

from renformer.sen_data import calendar_features

EPSILON_MW = 0.1   # intermittency threshold (raw MW, paper Section 3)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def masked_gaussian_nll(mean, log_std, target, mask):
    """
    Masked NLL (Eq. 1 in paper). Only penalises timesteps where mask > 0
    (i.e. raw generation > epsilon, meaning non-trivial active generation).
    """
    log_std = jnp.clip(log_std, -10.0, 10.0)   # prevent exp overflow / division by ~0
    var = jnp.exp(2.0 * log_std)
    nll = 0.5 * ((target - mean) ** 2 / var + 2.0 * log_std + jnp.log(2 * jnp.pi))
    return (nll * mask).sum() / (mask.sum() + 1e-6)


def mse_loss(mean, _log_std, target, _mask=None):
    """Unmasked MSE — used for the REnFormer-MSE ablation."""
    return jnp.mean((target - mean) ** 2)


# ---------------------------------------------------------------------------
# JIT-compiled step factories (close over model / optimizer so jit can trace)
# ---------------------------------------------------------------------------

def make_train_step(model, optimizer, loss_fn=masked_gaussian_nll):
    @jax.jit
    def train_step(params, opt_state, x, y_target, mask, rng):
        def _loss(params):
            mean, log_std = model.apply(params, x, train=True, rngs={"dropout": rng})
            return loss_fn(mean, log_std, y_target, mask)
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
    def __init__(self, df_norm, df_raw, lookback=168, horizon=24,
                 times=None, add_time_features=False, raw_input=False):
        # (S, T) layout for fast per-site slicing
        self.arr_norm = df_norm.values.T.astype(np.float32)
        self.arr_raw  = df_raw.values.T.astype(np.float32)
        self.site_names = df_norm.columns.tolist()
        self.lookback = lookback
        self.horizon  = horizon
        self.S, self.T = self.arr_norm.shape
        # number of valid start positions (one window = lookback + horizon)
        self.n_windows = self.T - lookback - horizon + 1
        self.index     = df_norm.index   # DatetimeIndex preserved for serialization
        self.raw_input = raw_input

        # Power channel fed to the model: raw (let the model's RevIN normalize)
        # or the pre-computed global z-score (legacy behaviour).
        self.arr_pow = self.arr_raw if raw_input else self.arr_norm

        # Optional calendar features, shared across all sites: (T, 4)
        if add_time_features:
            if times is None:
                raise ValueError("add_time_features=True requires `times` (DatetimeIndex)")
            self.tf = calendar_features(times)
        else:
            self.tf = None

    # -----------------------------------------------------------------------
    # Parquet persistence
    # -----------------------------------------------------------------------

    def save(self, path) -> None:
        """Serialize this split to a directory (norm.parquet, raw.parquet, meta.json)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.arr_norm.T, index=self.index, columns=self.site_names).to_parquet(
            path / "norm.parquet"
        )
        pd.DataFrame(self.arr_raw.T, index=self.index, columns=self.site_names).to_parquet(
            path / "raw.parquet"
        )
        (path / "meta.json").write_text(json.dumps({
            "lookback":          self.lookback,
            "horizon":           self.horizon,
            "add_time_features": self.tf is not None,
            "raw_input":         self.raw_input,
        }))

    @classmethod
    def from_parquet(cls, path, **kwargs):
        """Reconstruct a SENDataset from a directory written by .save()."""
        path    = Path(path)
        df_norm = pd.read_parquet(path / "norm.parquet")
        df_raw  = pd.read_parquet(path / "raw.parquet")
        meta    = json.loads((path / "meta.json").read_text())
        meta.update(kwargs)   # caller may override any field
        times   = df_norm.index if isinstance(df_norm.index, pd.DatetimeIndex) else None
        return cls(
            df_norm, df_raw,
            lookback=meta["lookback"],
            horizon=meta["horizon"],
            times=times,
            add_time_features=meta["add_time_features"],
            raw_input=meta["raw_input"],
        )

    def __len__(self):
        return self.S * self.n_windows

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        s_idx = rng.integers(0, self.S, size=batch_size)
        t_idx = rng.integers(0, self.n_windows, size=batch_size)
        return self._fetch(s_idx, t_idx)

    def sequential_batches(self, batch_size: int = 512, stride: int = 1):
        """Yields windows in (site-major, time-minor) order for evaluation.

        stride=1  → all overlapping windows (default, training-set coverage).
        stride=H  → non-overlapping windows; use when comparing against models
                    evaluated on non-overlapping windows (e.g. TimesFM zero-shot).
        """
        t_indices = np.arange(0, self.n_windows, stride)
        all_s = np.repeat(np.arange(self.S), len(t_indices))
        all_t = np.tile(t_indices, self.S)
        for start in range(0, len(all_s), batch_size):
            sl = slice(start, start + batch_size)
            yield self._fetch(all_s[sl], all_t[sl])

    def _fetch(self, s_idx, t_idx):
        Xp = np.stack([
            self.arr_pow[s, t: t + self.lookback]
            for s, t in zip(s_idx, t_idx)
        ])                                                # (B, L)
        if self.tf is not None:
            Xt = np.stack([self.tf[t: t + self.lookback] for t in t_idx])  # (B, L, 4)
            X = np.concatenate([Xp[:, :, np.newaxis], Xt], axis=-1)        # (B, L, 1+4)
        else:
            X = Xp[:, :, np.newaxis]                       # (B, L, 1)

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
    train_target: str = "norm",
):
    rng_np  = np.random.default_rng(seed)
    rng_jax = jax.random.PRNGKey(seed)

    # Initialise with a dummy batch (input channel count = in_features)
    in_feat = getattr(model, "in_features", model.out_features)
    dummy_x = jnp.zeros((1, model.max_len, in_feat))
    params  = model.init(rng_jax, dummy_x, train=False)

    use_raw = train_target == "raw"   # supervise in raw MW (RevIN model) vs z-scored

    schedule  = optax.cosine_decay_schedule(lr, decay_steps=epochs * steps_per_epoch)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)

    train_step = make_train_step(model, optimizer, loss_fn)
    eval_step  = make_eval_step(model)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # --- train ---
        train_losses = []
        for _ in range(steps_per_epoch):
            x_b, y_b, y_raw_b, mask_b, _ = train_ds.sample_batch(batch_size, rng_np)
            target = y_raw_b if use_raw else y_b
            rng_jax, step_rng = jax.random.split(rng_jax)
            params, opt_state, loss = train_step(
                params, opt_state,
                jnp.array(x_b), jnp.array(target), jnp.array(mask_b), step_rng,
            )
            train_losses.append(float(loss))

        # --- validate (one pass over val dataset) ---
        val_losses = []
        for x_b, y_b, y_raw_b, mask_b, _ in val_ds.sequential_batches(batch_size * 4):
            target = y_raw_b if use_raw else y_b
            mean, log_std = eval_step(params, jnp.array(x_b))
            val_loss = loss_fn(mean, log_std, jnp.array(target), jnp.array(mask_b))
            val_losses.append(float(val_loss))

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        print(f"Epoch {epoch:03d}/{epochs} | train NLL {t_loss:.4f} | val NLL {v_loss:.4f}")

    return params, history


# ---------------------------------------------------------------------------
# Orbax checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(params, directory, step: int = 0) -> None:
    """
    Save Flax params to an Orbax CheckpointManager directory.

    max_to_keep=1 keeps only the latest checkpoint, overwriting the previous
    one on each call. Increment step to retain multiple checkpoints.
    """
    import orbax.checkpoint as ocp
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mngr = ocp.CheckpointManager(
        str(directory),
        item_names=("params",),
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )
    mngr.save(step, args=ocp.args.Composite(params=ocp.args.PyTreeSave(params)))
    mngr.wait_until_finished()
    mngr.close()


def load_checkpoint(params_like, directory):
    """
    Restore Flax params from the latest checkpoint in directory.

    params_like must be a pytree with the same structure and dtypes as the
    saved params — typically produced by model.init with a dummy batch.
    """
    import orbax.checkpoint as ocp
    mngr = ocp.CheckpointManager(
        str(Path(directory).resolve()),
        item_names=("params",),
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )
    step = mngr.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoint found in {directory}")
    restored = mngr.restore(
        step,
        args=ocp.args.Composite(params=ocp.args.PyTreeRestore(params_like)),
    )
    mngr.close()
    return restored["params"]


def checkpoint_exists(directory) -> bool:
    """Return True if a valid Orbax checkpoint is present in directory."""
    import orbax.checkpoint as ocp
    try:
        mngr = ocp.CheckpointManager(
            str(Path(directory).resolve()),
            item_names=("params",),
        )
        exists = mngr.latest_step() is not None
        mngr.close()
        return exists
    except Exception:
        return False
