"""
compare_models.py — Compare REnFormer (from checkpoint) vs TimesFM 2.5 (zero-shot)
on SEN Chile data.

Evaluation:
  • REnFormer: all overlapping sliding-window predictions over the test set
  • TimesFM: non-overlapping 24 h windows over the test period (per-site univariate)
  • Active-mask metrics (MW > 0.1): MAE, RMSE, CRPS
  • Efficiency: parameter count and pure inference time (JIT/compile warm-up
    excluded), reported per model alongside ms per site-window forecast

Usage
-----
python compare_models.py --csv data/Descarga_Generación_Real_2026-05-29_18-57-56.csv
python compare_models.py --csv <path> --cache_dir data/       # reuse parquet cache
python compare_models.py --csv <path> --checkpoint_dir checkpoint
python compare_models.py --csv <path> --max_sites 50          # limit sites for speed
python compare_models.py --csv <path> --skip_timesfm
python compare_models.py --csv <path> --skip_renformer
"""
import argparse
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from compare_models_skip import (
    add_efficiency,
    collect_renformer_skip_predictions,
    compute_metrics,
    count_params,
    print_table,
    run_timesfm_zero_shot,
)
from renformer.model import TimeSeriesTransformer
from renformer.train import SENDataset, make_eval_step, load_checkpoint, checkpoint_exists
from renformer.sen_data import (
    prepare_sen_dataset,
    load_prepared_dataset,
    prepared_dataset_exists,
    save_prepared_dataset,
)

warnings.filterwarnings("ignore", category=UserWarning)

LOOKBACK  = 168   # must match REnFormer
HORIZON   = 24

HPARAMS = dict(
    d_model=128,
    num_heads=4,
    num_layers=4,
    mlp_dim=256,
    dropout_rate=0.1,
    max_len=LOOKBACK,
    horizon=HORIZON,
    out_features=1,
)

# ──────────────────────────────────────────────────────────────────────────────
# REnFormer checkpoint evaluation
# ──────────────────────────────────────────────────────────────────────────────

def collect_renformer_predictions(test_norm, test_raw, checkpoint_dir, max_sites=None):
    """
    Load REnFormer from checkpoint and run all sliding-window predictions over
    the test set (same protocol as run_experiment.py collect_predictions).

    Returns (y_true_mw, y_pred_mw, y_sigma_mw, stats) where the first three are
    flat 1-D arrays in MW units and stats is
    {"params": int, "infer_s": float, "n_fc": int} — n_fc counts site-window
    forecasts; infer_s covers only eval_step calls, JIT compilation excluded.

    The model is trained on raw-MW input (raw_input=True, RevIN handles
    per-window scale) and supervised against raw MW (train_target="raw"),
    so its mean/log_std outputs are already in MW — no de-normalization needed.
    """
    if max_sites is not None:
        cols      = test_norm.columns[:max_sites]
        test_norm = test_norm[cols]
        test_raw  = test_raw[cols]

    model   = TimeSeriesTransformer(**HPARAMS)
    test_ds = SENDataset(test_norm, test_raw, LOOKBACK, HORIZON, raw_input=True)

    in_feat     = getattr(model, "in_features", model.out_features)
    dummy_x     = jnp.zeros((1, LOOKBACK, in_feat))
    params_like = model.init(jax.random.PRNGKey(0), dummy_x, train=False)
    params      = load_checkpoint(params_like, checkpoint_dir)
    n_params    = count_params(params)

    eval_step = make_eval_step(model)
    mu_list, sigma_list, y_raw_list = [], [], []
    infer_s, n_fc, seen_shapes = 0.0, 0, set()

    n_nonoverlap = len(range(0, test_ds.n_windows, HORIZON))
    print(f"  REnFormer: evaluating {n_nonoverlap:,} non-overlapping windows …", flush=True)
    print(f"  REnFormer: {n_params:,} parameters", flush=True)
    for x_b, _, y_raw_b, _, _ in test_ds.sequential_batches(512, stride=HORIZON):
        x_d = jnp.array(x_b)
        if x_b.shape not in seen_shapes:              # untimed warm-up per batch shape
            jax.block_until_ready(eval_step(params, x_d))
            seen_shapes.add(x_b.shape)
        t0 = time.perf_counter()
        mean, log_std = eval_step(params, x_d)
        jax.block_until_ready((mean, log_std))
        infer_s += time.perf_counter() - t0
        n_fc    += x_b.shape[0]
        mu_list.append(np.array(mean))
        sigma_list.append(np.array(jnp.exp(log_std)))
        y_raw_list.append(y_raw_b)

    y_true  = np.concatenate(y_raw_list).reshape(-1)
    y_pred  = np.concatenate(mu_list).reshape(-1)
    y_sigma = np.concatenate(sigma_list).reshape(-1)
    stats   = {"params": n_params, "infer_s": infer_s, "n_fc": n_fc}
    return y_true, y_pred, y_sigma, stats

# TimesFM zero-shot evaluation, metrics, and table printing are shared with
# compare_models_skip.py (run_timesfm_zero_shot, compute_metrics,
# add_efficiency, print_table — imported above).


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(args):
    # ── 1. Load data ──────────────────────────────────────────────────────────
    if args.cache_dir and prepared_dataset_exists(args.cache_dir):
        print(f"Loading preprocessed dataset from cache: {args.cache_dir}")
        result = load_prepared_dataset(args.cache_dir)
    else:
        result = prepare_sen_dataset(args.csv)
        if args.cache_dir:
            save_prepared_dataset(result, args.cache_dir)

    (_, _), (val_raw, _), (test_raw, test_norm), _norm_stats = result

    n_sites = test_raw.shape[1] if args.max_sites is None else min(args.max_sites, test_raw.shape[1])
    print(f"\nSites used : {n_sites}")
    print(f"Test period: {test_raw.index[0].date()} → {test_raw.index[-1].date()}")

    results = {}

    # ── 2. REnFormer from checkpoint ──────────────────────────────────────────
    if not args.skip_renformer:
        if checkpoint_exists(args.checkpoint_dir):
            print(f"\n─── REnFormer (checkpoint: {args.checkpoint_dir}) ───────────────────")
            y_true, y_pred, y_sigma, stats = collect_renformer_predictions(
                test_norm, test_raw,
                checkpoint_dir=args.checkpoint_dir,
                max_sites=args.max_sites,
            )
            results["REnFormer (checkpoint)"] = add_efficiency(
                compute_metrics(y_true, y_pred, y_sigma), stats)
        else:
            print(f"\nNo REnFormer checkpoint found at '{args.checkpoint_dir}' — skipping.")

    # ── 3. TimesFM zero-shot ──────────────────────────────────────────────────
    if not args.skip_timesfm:
        print("\n─── TimesFM 2.5 (zero-shot) ─────────────────────────────────────────")
        y_true, y_pred, y_sigma, stats = run_timesfm_zero_shot(
            val_raw, test_raw, max_sites=args.max_sites
        )
        results["TimesFM 2.5 (zero-shot)"] = add_efficiency(
            compute_metrics(y_true, y_pred, y_sigma), stats)

    # ── 2. REnFormer-Skip from checkpoint ─────────────────────────────────────
    if not args.skip_renformer:
        if checkpoint_exists(args.checkpoint_dir):
            print(f"\n─── REnFormer-Skip (checkpoint: {args.checkpoint_dir}) ───────────────────")
            y_true, y_pred, y_sigma, stats = collect_renformer_skip_predictions(
                test_norm, test_raw,
                checkpoint_dir=args.checkpoint_dir,
                max_sites=args.max_sites,
            )
            results["REnFormer-Skip (checkpoint)"] = add_efficiency(
                compute_metrics(y_true, y_pred, y_sigma), stats)
        else:
            print(f"\nNo REnFormer-Skip checkpoint found at '{args.checkpoint_dir}' — skipping.")

    # ── 4. Summary table ──────────────────────────────────────────────────────
    print_table(results, header="═" * 65 + "\nFINAL COMPARISON\n" + "═" * 65)

    if args.out:
        import json
        out_path = Path(args.out)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare REnFormer vs TimesFM on SEN Chile.")
    p.add_argument("--csv",              required=True,  help="Path to SEN Chile CSV")
    p.add_argument("--cache_dir",        default=None,   help="Parquet cache directory (shared with run_experiment.py)")
    p.add_argument("--checkpoint_dir",   default="checkpoints", help="Orbax checkpoint directory for REnFormer (default: checkpoints)")
    p.add_argument("--max_sites",        type=int,       default=None,   help="Restrict to first N sites (faster smoke tests)")
    p.add_argument("--skip_timesfm",     action="store_true", help="Skip TimesFM zero-shot evaluation")
    p.add_argument("--skip_renformer",   action="store_true", help="Skip REnFormer checkpoint evaluation")
    p.add_argument("--out",              default=None,   help="Write JSON results to this file")
    args = p.parse_args()

    run(args)
