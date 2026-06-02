"""
Fine-tune TimesFM 2.5 (PyTorch backend) on Chilean-grid solar generation and
compare against the zero-shot baseline from ``forecast_solar.py``.

Why PyTorch (not the Flax backend used in forecast_solar.py): fine-tuning needs
gradients, and only the torch module exposes a differentiable forward pass.

How the fine-tuning works
-------------------------
For a 24 h horizon the model never autoregresses: ``num_decode_steps =
(24 - 1) // 128 = 0``, so the whole forecast comes from the *prefill* step — a
single forward over the patched context. ``train_forward`` below reproduces that
exact inference path (outer instance-norm, per-patch RevIN running stats,
continuous-quantile head, positivity clamp) but *with* autograd, so we can put a
supervised loss on the 24 h forecast in real MW units.

Evaluation mirrors ``forecast_solar.py``'s ForecastConfig (flip invariance,
continuous quantile head, etc.) for BOTH the zero-shot and fine-tuned models, so
the comparison is apples-to-apples. We report:
  1. The 2026-05-29 day forecast (same target as forecast_solar.py), and
  2. A leakage-free rolling backtest over the last BACKTEST_DAYS complete days.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import timesfm
from timesfm.torch.util import revin, update_running_stats

# ── Config ───────────────────────────────────────────────────────────────────

DATA_PATH = "data/Descarga_Generación_Real_2026-05-29_18-57-56.csv"
CONTEXT   = 192   # 8 days (multiple of patch size 32 → no padding needed)
HORIZON   = 24    # 1 day

EVAL_DAY      = pd.Timestamp("2026-05-29 00:00")  # partial actuals (matches forecast_solar.py)
BACKTEST_DAYS = 14    # complete held-out days used as the robust test set
TRAIN_DAYS    = 365   # history (before the backtest period) to draw windows from
TRAIN_STRIDE  = 24    # one training window per day

EPOCHS          = 100
BATCH_SIZE      = 16
LR              = 1e-4
UNFREEZE_LAST_N = 2     # train output heads + this many final transformer layers (0 = heads only)
GRAD_CLIP       = 1.0
SEED            = 0

OUT_PLOT    = "data/finetune_solar_compare.png"
OUT_METRICS = "data/finetune_solar_metrics.csv"
OUT_WEIGHTS = "data/finetuned_solar"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 1. Load, filter to solar, aggregate hourly, and fill to a complete grid ───

df = pd.read_csv(DATA_PATH, sep=";", decimal=",", encoding="utf-8-sig")
solar_df = df[df["Tipo"] == "Solar"]

hora_cols = [f"Hora {i}" for i in range(1, 25)]
long = solar_df.melt(id_vars=["Fecha"], value_vars=hora_cols,
                     var_name="hora_str", value_name="mw")
long["hour"] = long["hora_str"].str.extract(r"(\d+)").astype(int) - 1
long["timestamp"] = pd.to_datetime(long["Fecha"]) + pd.to_timedelta(long["hour"], unit="h")

ts = long.groupby("timestamp")["mw"].sum().sort_index()
full_idx = pd.date_range(ts.index[0], ts.index[-1], freq="h")
ts = ts.reindex(full_idx).interpolate("linear").fillna(0.0)

vals = ts.values.astype(np.float32)
pos  = {t: i for i, t in enumerate(ts.index)}

def window(end_pos):
    """Return (context[CONTEXT], target[HORIZON]) ending right before end_pos."""
    return (vals[end_pos - CONTEXT:end_pos],
            vals[end_pos:end_pos + HORIZON])

# ── 2. Build train / backtest / eval splits (no leakage) ──────────────────────
# Backtest origins: the last BACKTEST_DAYS complete days ending 2026-05-28.
# Training windows: targets must finish before the earliest backtest forecast.

last_complete_day = EVAL_DAY - pd.Timedelta(days=1)            # 2026-05-28 00:00
backtest_starts = [last_complete_day - pd.Timedelta(days=k)
                   for k in range(BACKTEST_DAYS - 1, -1, -1)]  # chronological
cutoff_pos = pos[backtest_starts[0]]                           # earliest backtest forecast start

last_train_pos  = cutoff_pos - HORIZON                         # target fully before cutoff
first_train_pos = max(CONTEXT, cutoff_pos - TRAIN_DAYS * 24)
train_origins   = list(range(first_train_pos, last_train_pos + 1, TRAIN_STRIDE))

X_tr = np.stack([window(o)[0] for o in train_origins])         # (Ntr, CONTEXT)
Y_tr = np.stack([window(o)[1] for o in train_origins])         # (Ntr, HORIZON)

bt_ctx = [window(pos[d])[0] for d in backtest_starts]
bt_tgt = np.stack([window(pos[d])[1] for d in backtest_starts])  # (BACKTEST_DAYS, HORIZON)

# Eval-day forecast (partial actuals, up to the last non-zero exported hour)
ep = pos[EVAL_DAY]
eval_ctx = vals[ep - CONTEXT:ep]
eval_actual_full = ts.loc[EVAL_DAY:EVAL_DAY + pd.Timedelta(hours=HORIZON - 1)]
nz = eval_actual_full[eval_actual_full > 0]
eval_actual = eval_actual_full[eval_actual_full.index <= nz.index.max()].values
n_eval = len(eval_actual)

print(f"Training windows: {len(train_origins)}  "
      f"({ts.index[first_train_pos]} → {ts.index[last_train_pos]} targets)")
print(f"Backtest days:    {len(backtest_starts)}  "
      f"({backtest_starts[0].date()} → {backtest_starts[-1].date()})")
print(f"Eval day:         {EVAL_DAY.date()}  ({n_eval} h of actuals available)")

# ── 3. Load model (eager, so weights stay trainable) ──────────────────────────

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    torch_compile=False,   # avoid the torch.compile wrapper; keep the module trainable
)
mod = model.model          # the nn.Module
device = mod.device

EVAL_CONFIG = timesfm.ForecastConfig(
    max_context=CONTEXT,
    max_horizon=HORIZON,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=True,
    fix_quantile_crossing=True,
)
model.compile(EVAL_CONFIG)

def evaluate(contexts, targets):
    """Point MAE per series for a batch of (context, target) pairs."""
    pt, _ = model.forecast(horizon=HORIZON,
                           inputs=[c.astype(np.float32) for c in contexts])
    pt = np.asarray(pt)[:, :HORIZON]
    return np.abs(pt - targets).mean(axis=1), pt

# ── 4. Zero-shot baseline (== forecast_solar.py approach, reproduced here) ─────

zs_bt_mae, zs_bt_pt = evaluate(bt_ctx, bt_tgt)
_, zs_eval_pt = evaluate([eval_ctx], np.zeros((1, HORIZON)))
zs_eval_pt = zs_eval_pt[0]
zs_eval_mae = np.abs(zs_eval_pt[:n_eval] - eval_actual).mean()

print(f"\nZero-shot  backtest MAE: {zs_bt_mae.mean():7.1f} ± {zs_bt_mae.std():5.1f} MW")
print(f"Zero-shot  eval-day MAE: {zs_eval_mae:7.1f} MW")

# ── 5. Differentiable prefill forward (mirrors decode for horizon ≤ 128) ───────

def train_forward(values, masks, horizon=HORIZON):
    """Replicates TimesFM_2p5_200M_torch._compiled_decode (prefill only, no AR,
    no flip invariance) with gradients. Returns (B, horizon, q) in MW units."""
    B, L = values.shape
    p, o, q, os, nx = mod.p, mod.o, mod.q, mod.os, mod.x

    is_pos = torch.all(values >= 0, dim=-1, keepdim=True)
    mu = values.mean(-1, keepdim=True)
    sigma = values.std(-1, keepdim=True)
    inp = revin(values, mu, sigma, reverse=False)

    n_patches = L // p
    pin = inp.reshape(B, n_patches, p)
    pmask = masks.reshape(B, n_patches, p)

    n = torch.zeros(B, device=values.device)
    rmu = torch.zeros(B, device=values.device)
    rsig = torch.zeros(B, device=values.device)
    cmu, csig = [], []
    for i in range(n_patches):
        (n, rmu, rsig), _ = update_running_stats(n, rmu, rsig, pin[:, i], pmask[:, i])
        cmu.append(rmu)
        csig.append(rsig)
    context_mu = torch.stack(cmu, dim=1)
    context_sigma = torch.stack(csig, dim=1)

    normed = revin(pin, context_mu, context_sigma, reverse=False)
    normed = torch.where(pmask, torch.zeros_like(normed), normed)
    (_, _, out_ts, out_qs), _ = mod(normed, pmask, [None] * nx)

    renormed = revin(out_ts, context_mu, context_sigma, reverse=True).reshape(B, n_patches, o, q)
    renormed_qs = revin(out_qs, context_mu, context_sigma, reverse=True).reshape(B, n_patches, os, q)[:, -1]

    full = renormed[:, -1].clone()                       # (B, o, q) — prefill last patch
    for qi in [1, 2, 3, 4, 6, 7, 8, 9]:                  # continuous quantile head
        full[:, :, qi] = renormed_qs[:, :o, qi] - renormed_qs[:, :o, 5] + full[:, :, 5]
    full = full[:, :horizon, :]
    full = revin(full, mu, sigma, reverse=True)          # undo outer instance-norm
    full = torch.where(is_pos[..., None], torch.clamp(full, min=0.0), full)
    return full

def loss_fn(full, target):
    """Point Huber (median ch5) + mean MSE (ch0) + pinball over q10..q90."""
    median, mean_ch = full[..., 5], full[..., 0]
    loss = F.smooth_l1_loss(median, target) + F.mse_loss(mean_ch, target)
    pin = 0.0
    for i in range(1, 10):                               # channels 1..9 = quantiles 0.1..0.9
        level = i * 0.1
        err = target - full[..., i]
        pin = pin + torch.maximum(level * err, (level - 1.0) * err).mean()
    return loss + pin / 9.0

# ── 6. Fine-tune (freeze backbone; train output heads + last N layers) ────────

for prm in mod.parameters():
    prm.requires_grad_(False)
trainable = list(mod.output_projection_point.parameters()) + \
            list(mod.output_projection_quantiles.parameters())
for layer in mod.stacked_xf[len(mod.stacked_xf) - UNFREEZE_LAST_N:]:
    trainable += list(layer.parameters())
for prm in trainable:
    prm.requires_grad_(True)

n_trainable = sum(p.numel() for p in trainable)
n_total = sum(p.numel() for p in mod.parameters())
print(f"\nFine-tuning {n_trainable:,} / {n_total:,} params "
      f"(heads + last {UNFREEZE_LAST_N} layers) on {device} ...")

opt = torch.optim.Adam(trainable, lr=LR)
Xt = torch.from_numpy(X_tr).to(device)
Yt = torch.from_numpy(Y_tr).to(device)
masks0 = torch.zeros_like(Xt, dtype=torch.bool)

mod.train()
for epoch in range(EPOCHS):
    perm = torch.randperm(len(Xt))
    epoch_loss, n_batches = 0.0, 0
    for s in range(0, len(perm), BATCH_SIZE):
        b = perm[s:s + BATCH_SIZE]
        opt.zero_grad()
        out = train_forward(Xt[b], masks0[b])
        loss = loss_fn(out, Yt[b])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        opt.step()
        epoch_loss += loss.item()
        n_batches += 1
    print(f"  epoch {epoch + 1}/{EPOCHS}  loss = {epoch_loss / n_batches:.4f}")
mod.eval()

# ── 7. Fine-tuned evaluation (same compiled pipeline reads the updated weights) ─

ft_bt_mae, ft_bt_pt = evaluate(bt_ctx, bt_tgt)
_, ft_eval_pt = evaluate([eval_ctx], np.zeros((1, HORIZON)))
ft_eval_pt = ft_eval_pt[0]
ft_eval_mae = np.abs(ft_eval_pt[:n_eval] - eval_actual).mean()

model.save_pretrained(OUT_WEIGHTS)

# ── 8. Compare ────────────────────────────────────────────────────────────────

def improvement(zs, ft):
    return 100.0 * (zs - ft) / zs if zs else float("nan")

print("\n" + "=" * 62)
print(f"{'Metric':<26}{'Zero-shot':>12}{'Fine-tuned':>12}{'Δ%':>10}")
print("-" * 62)
print(f"{'Backtest MAE (mean MW)':<26}{zs_bt_mae.mean():>12.1f}"
      f"{ft_bt_mae.mean():>12.1f}{improvement(zs_bt_mae.mean(), ft_bt_mae.mean()):>9.1f}%")
print(f"{'Backtest MAE (std MW)':<26}{zs_bt_mae.std():>12.1f}{ft_bt_mae.std():>12.1f}{'':>10}")
print(f"{'Eval-day MAE (MW)':<26}{zs_eval_mae:>12.1f}"
      f"{ft_eval_mae:>12.1f}{improvement(zs_eval_mae, ft_eval_mae):>9.1f}%")
print("=" * 62)

metrics = pd.DataFrame({
    "day": [d.date() for d in backtest_starts],
    "zeroshot_mae": zs_bt_mae,
    "finetuned_mae": ft_bt_mae,
})
metrics.to_csv(OUT_METRICS, index=False)
print(f"Per-day backtest metrics → {OUT_METRICS}")
print(f"Fine-tuned weights        → {OUT_WEIGHTS}/")

# ── 9. Plot ────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))

hrs = np.arange(HORIZON)
ax1.plot(hrs, zs_eval_pt, color="steelblue", lw=2, label="Zero-shot")
ax1.plot(hrs, ft_eval_pt, color="goldenrod", lw=2, label="Fine-tuned")
ax1.plot(np.arange(n_eval), eval_actual, color="black", lw=1.6,
         ls="--", label="Actuals (partial)")
ax1.set_title(f"Solar 24 h forecast — {EVAL_DAY.date()} "
              f"(zero-shot MAE {zs_eval_mae:.0f} → fine-tuned {ft_eval_mae:.0f} MW)")
ax1.set_xlabel("Hour of day")
ax1.set_ylabel("Solar generation (MW)")
ax1.legend()

x = np.arange(len(backtest_starts))
ax2.bar(x - 0.2, zs_bt_mae, width=0.4, color="steelblue", label="Zero-shot")
ax2.bar(x + 0.2, ft_bt_mae, width=0.4, color="goldenrod", label="Fine-tuned")
ax2.set_xticks(x)
ax2.set_xticklabels([d.strftime("%m-%d") for d in backtest_starts], rotation=45, ha="right")
ax2.set_title(f"Per-day backtest MAE over {BACKTEST_DAYS} held-out days "
              f"(mean {zs_bt_mae.mean():.0f} → {ft_bt_mae.mean():.0f} MW)")
ax2.set_ylabel("MAE (MW)")
ax2.legend()

fig.tight_layout()
fig.savefig(OUT_PLOT, dpi=150)
print(f"Comparison plot           → {OUT_PLOT}")
