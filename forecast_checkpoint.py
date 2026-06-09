"""
forecast_checkpoint.py — Load a saved REnFormer checkpoint and produce a
probabilistic forecast for the next HORIZON hours for every site, using the
last LOOKBACK hours of the dataset as context.

Usage
-----
python forecast_checkpoint.py \\
    --csv  data/Descarga_Generación_Real_2026-05-29_18-57-56.csv \\
    --checkpoint checkpoints/renformer_params.pkl \\
    [--out  forecasts/renformer_forecast.csv] \\
    [--batch_size 128]

Output columns (CSV / printed table):
    timestamp, site, mean_mw, std_mw, q10_mw, q90_mw
plus one row per hour for the grid-level aggregate (site = "__total__").
"""

import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from renformer.model import TimeSeriesTransformer
from renformer.train import make_eval_step
from renformer.sen_data import prepare_sen_dataset

# Must match the checkpoint's training config (paper Table 1).
HPARAMS = dict(
    d_model=128,
    num_heads=4,
    num_layers=4,
    mlp_dim=256,
    dropout_rate=0.1,
    max_len=168,
    horizon=24,
    out_features=1,
    in_features=1,
)

LOOKBACK = 168
HORIZON  = 24

# Gaussian quantile multipliers for 80 % interval (z_{0.10}, z_{0.90})
_Z10 = -1.2815516
_Z90 =  1.2815516


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def run(args):
    # ------------------------------------------------------------------
    # 1. Data — run the full pipeline to get the raw MW series.
    # ------------------------------------------------------------------
    (train_raw, train_norm), \
    (val_raw,   val_norm), \
    (test_raw,  test_norm), \
    _norm_stats = prepare_sen_dataset(args.csv)

    # Full raw-MW series in chronological order. The model is trained with
    # raw_input=True (RevIN normalises per window), so it expects raw MW.
    full_raw = pd.concat([train_raw, val_raw, test_raw])  # (T, S)
    sites = full_raw.columns.tolist()
    S = len(sites)

    last_ts = full_raw.index[-1]
    context_raw = full_raw.iloc[-LOOKBACK:]               # (LOOKBACK, S)

    print(f"\nContext window : {context_raw.index[0]}  →  {last_ts}")
    print(f"Forecast window: {last_ts + pd.Timedelta(hours=1)}  →  "
          f"{last_ts + pd.Timedelta(hours=HORIZON)}")
    print(f"Sites          : {S}")

    # ------------------------------------------------------------------
    # 2. Build input batch  (S, LOOKBACK, 1)
    # ------------------------------------------------------------------
    # arr[s, t, 0] = raw MW generation for site s at time t
    X = context_raw.values.T[:, :, np.newaxis].astype(np.float32)

    # ------------------------------------------------------------------
    # 3. Load model and checkpoint
    # ------------------------------------------------------------------
    model  = TimeSeriesTransformer(**HPARAMS)
    params = load_checkpoint(args.checkpoint)

    eval_step = make_eval_step(model)

    # ------------------------------------------------------------------
    # 4. Inference in batches  →  (S, H, 1) mean / std, already in MW
    #    (RevIN inverse inside the model maps back to input units)
    # ------------------------------------------------------------------
    mu_list, sigma_list = [], []
    for start in range(0, S, args.batch_size):
        x_b = jnp.array(X[start: start + args.batch_size])
        mean_b, log_std_b = eval_step(params, x_b)
        mu_list.append(np.array(mean_b))
        sigma_list.append(np.array(jnp.exp(log_std_b)))

    mean_mw = np.concatenate(mu_list)     # (S, H, 1)
    std_mw  = np.concatenate(sigma_list)  # (S, H, 1)
    mean_mw = np.clip(mean_mw, 0.0, None)  # solar generation is non-negative

    # ------------------------------------------------------------------
    # 5. Build per-site forecast DataFrame
    # ------------------------------------------------------------------
    forecast_idx = pd.date_range(
        last_ts + pd.Timedelta(hours=1), periods=HORIZON, freq="h"
    )

    rows = []
    for s_idx, site in enumerate(sites):
        mu_s    = mean_mw[s_idx, :, 0]
        sigma_s = std_mw[s_idx, :, 0]
        for h in range(HORIZON):
            rows.append({
                "timestamp": forecast_idx[h],
                "site":      site,
                "mean_mw":   float(mu_s[h]),
                "std_mw":    float(sigma_s[h]),
                "q10_mw":    float(max(0.0, mu_s[h] + _Z10 * sigma_s[h])),
                "q90_mw":    float(mu_s[h] + _Z90 * sigma_s[h]),
            })

    # Grid-level aggregate (independent Gaussians → sum means, RSS stds)
    total_mean  = mean_mw[:, :, 0].sum(axis=0)                          # (H,)
    total_std   = np.sqrt((std_mw[:, :, 0] ** 2).sum(axis=0))           # (H,)
    for h in range(HORIZON):
        rows.append({
            "timestamp": forecast_idx[h],
            "site":      "__total__",
            "mean_mw":   float(total_mean[h]),
            "std_mw":    float(total_std[h]),
            "q10_mw":    float(max(0.0, total_mean[h] + _Z10 * total_std[h])),
            "q90_mw":    float(total_mean[h] + _Z90 * total_std[h]),
        })

    fc = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 6. Print aggregate summary
    # ------------------------------------------------------------------
    total_fc = fc[fc["site"] == "__total__"].set_index("timestamp")
    peak_h   = total_fc["mean_mw"].idxmax()
    print(f"\n{'Hour':<22} {'Mean (MW)':>10} {'Std (MW)':>10} {'80% interval':>22}")
    print("-" * 68)
    for ts, row in total_fc.iterrows():
        marker = " ← peak" if ts == peak_h else ""
        print(f"{str(ts):<22} {row['mean_mw']:>10.1f} {row['std_mw']:>10.1f} "
              f"  [{row['q10_mw']:>7.1f}, {row['q90_mw']:>7.1f}]{marker}")
    print(f"\nPeak forecast : {total_fc.loc[peak_h, 'mean_mw']:.1f} MW  at {peak_h}")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fc.to_csv(args.out, index=False)
        print(f"\nForecast saved → {args.out}")

    return fc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         required=True,
                   help="Path to SEN Chile CSV")
    p.add_argument("--checkpoint",  default="checkpoints/renformer_params.pkl",
                   help="Path to saved params pickle (default: checkpoints/renformer_params.pkl)")
    p.add_argument("--out",         default=None,
                   help="Optional CSV path to save per-site forecast rows")
    p.add_argument("--batch_size",  type=int, default=128,
                   help="Sites per inference batch (default: 128)")
    args = p.parse_args()
    run(args)
