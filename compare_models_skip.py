"""
compare_models_skip.py — Compare REnFormer-Skip (from checkpoint) vs TimesFM 2.5 (zero-shot)
on SEN Chile data.

Evaluation:
  • REnFormer-Skip: all overlapping sliding-window predictions over the test set
  • TimesFM: non-overlapping 24 h windows over the test period (per-site univariate)
  • Active-mask metrics (MW > 0.1): MAE, RMSE, CRPS
  • Efficiency: parameter count and pure inference time (JIT/compile warm-up
    excluded), reported per model alongside ms per site-window forecast

Usage
-----
python compare_models_skip.py --csv data/Descarga_Generación_Real_2026-05-29_18-57-56.csv
python compare_models_skip.py --csv <path> --cache_dir data/       # reuse parquet cache
python compare_models_skip.py --csv <path> --checkpoint_dir checkpoint_skip
python compare_models_skip.py --csv <path> --max_sites 50          # limit sites for speed
python compare_models_skip.py --csv <path> --skip_timesfm
python compare_models_skip.py --csv <path> --skip_renformer
"""
import argparse
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from renformer.model import TimeSeriesTransformerSkip
from renformer.train import SENDataset, make_eval_step, load_checkpoint, checkpoint_exists
from renformer.sen_data import (
    prepare_sen_dataset,
    load_prepared_dataset,
    prepared_dataset_exists,
    save_prepared_dataset,
)
from renformer.metrics import EPSILON_MW, mae, rmse, crps_gaussian

warnings.filterwarnings("ignore", category=UserWarning)

LOOKBACK  = 168   # must match REnFormer-Skip
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
# Efficiency helpers
# ──────────────────────────────────────────────────────────────────────────────

def count_params(params) -> int:
    """Total number of scalar parameters in a JAX/Flax pytree."""
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)
                   if hasattr(x, "size")))


def count_timesfm_params(tfm):
    """Parameter count of a TimesFM 2.5 flax model (None if introspection fails)."""
    try:
        from flax import nnx
        state = nnx.state(tfm.model, nnx.Param)
        return count_params(state)
    except Exception as exc:                          # noqa: BLE001
        print(f"  (could not count TimesFM params: {exc})")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# REnFormer-Skip checkpoint evaluation
# ──────────────────────────────────────────────────────────────────────────────

def collect_renformer_skip_predictions(test_norm, test_raw, checkpoint_dir, max_sites=None):
    """
    Load REnFormer-Skip from checkpoint and run all sliding-window predictions over
    the test set (same protocol as run_experiment_skip.py collect_predictions).

    Returns (y_true_mw, y_pred_mw, y_sigma_mw, stats) where the first three are
    flat 1-D arrays in MW units and stats is
    {"params": int, "infer_s": float, "n_fc": int} — n_fc counts site-window
    forecasts; infer_s covers only eval_step calls, JIT compilation excluded.

    The model is trained on raw-MW input (raw_input=True, instance norm handles
    per-window scale) and supervised against raw MW (train_target="raw"),
    so its mean/log_std outputs are already in MW — no de-normalization needed.
    """
    if max_sites is not None:
        cols      = test_norm.columns[:max_sites]
        test_norm = test_norm[cols]
        test_raw  = test_raw[cols]

    model   = TimeSeriesTransformerSkip(**HPARAMS)
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
    print(f"  REnFormer-Skip: evaluating {n_nonoverlap:,} non-overlapping windows …", flush=True)
    print(f"  REnFormer-Skip: {n_params:,} parameters", flush=True)
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


# ──────────────────────────────────────────────────────────────────────────────
# TimesFM zero-shot evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_timesfm_zero_shot(val_raw, test_raw, max_sites=None):
    """
    Evaluate TimesFM 2.5 (Flax, zero-shot) on non-overlapping 24h test windows,
    forecasting each site as an independent univariate series.

    val_raw is prepended so that early test windows (which need lookback before
    the test start date) have sufficient context.

    Returns (y_true_mw, y_pred_mw, y_sigma_mw, stats) where the first three are
    flat 1-D arrays in MW units and stats is
    {"params": int | None, "infer_s": float, "n_fc": int} — n_fc counts
    site-window forecasts; infer_s covers only tfm.forecast calls, with one
    untimed warm-up call to exclude compilation.

    Sigma is estimated via OLS fit to the nine output quantiles (same approach
    as forecast_solar.py).
    """
    import timesfm

    if max_sites is not None:
        cols     = test_raw.columns[:max_sites]
        val_raw  = val_raw[cols]
        test_raw = test_raw[cols]

    n_sites = test_raw.shape[1]
    val_len = len(val_raw)

    # (S, T_val + T_test) — row per site, column per hour
    ctx_arr = np.concatenate([val_raw.values, test_raw.values], axis=0).T.astype(np.float32)

    tfm = timesfm.TimesFM_2p5_200M_flax.from_pretrained("google/timesfm-2.5-200m-flax")
    tfm.compile(timesfm.ForecastConfig(
        max_context=LOOKBACK,
        max_horizon=HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    ))

    # Φ⁻¹(level) for levels 0.1…0.9 — used to estimate Gaussian sigma via OLS
    z_inv = np.array([-1.2815516, -0.8416212, -0.5244005, -0.2533471, 0.0,
                       0.2533471,  0.5244005,  0.8416212,  1.2815516])

    n_test        = len(test_raw)
    # Start at LOOKBACK so the first TimesFM target window aligns with REnFormer-Skip's
    # first target window (test[LOOKBACK:LOOKBACK+HORIZON]).  Windows before
    # LOOKBACK cannot be matched by REnFormer-Skip anyway (needs full test context).
    window_starts = list(range(LOOKBACK, n_test - HORIZON + 1, HORIZON))
    print(f"  TimesFM zero-shot: {n_sites} sites × {len(window_starts)} windows …", flush=True)

    n_params = count_timesfm_params(tfm)
    if n_params is not None:
        print(f"  TimesFM: {n_params:,} parameters", flush=True)

    mu_list, sigma_list, y_raw_list = [], [], []
    infer_s, n_fc, warmed = 0.0, 0, False

    for i, win_rel in enumerate(window_starts):
        ctx_end   = val_len + win_rel           # absolute index (end of context)
        ctx_start = ctx_end - LOOKBACK
        if ctx_start < 0:
            continue                            # not enough history; skip

        ctx = ctx_arr[:, ctx_start:ctx_end]           # (S, LOOKBACK)
        tgt = ctx_arr[:, ctx_end:ctx_end + HORIZON]   # (S, HORIZON)

        inputs = [ctx[s] for s in range(n_sites)]
        if not warmed:                          # untimed warm-up: exclude compilation
            tfm.forecast(horizon=HORIZON, inputs=inputs)
            warmed = True
        t0 = time.perf_counter()
        point_fc, quant_fc = tfm.forecast(
            horizon=HORIZON,
            inputs=inputs,
        )
        infer_s += time.perf_counter() - t0
        n_fc    += n_sites
        # point_fc: (S, HORIZON)  quant_fc: (S, HORIZON, 10)

        q_vals = quant_fc[:, :, 1:10]                         # (S, H, 9)
        sigma  = np.abs((q_vals @ z_inv) / (z_inv @ z_inv))  # (S, H)

        mu_list.append(point_fc[:, :HORIZON].reshape(-1))
        sigma_list.append(sigma.reshape(-1))
        y_raw_list.append(tgt.reshape(-1))

        if (i + 1) % 50 == 0:
            print(f"    … {i+1}/{len(window_starts)} windows done", flush=True)

    stats = {"params": n_params, "infer_s": infer_s, "n_fc": n_fc}
    return (
        np.concatenate(y_raw_list),
        np.concatenate(mu_list),
        np.concatenate(sigma_list),
        stats,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_sigma: np.ndarray = None) -> dict:
    """MAE, RMSE (and CRPS if sigma is provided), active-mask applied."""
    mask = (y_true > EPSILON_MW)
    results = {
        "MAE" : mae(y_true, y_pred, mask),
        "RMSE": rmse(y_true, y_pred, mask),
    }
    if y_sigma is not None:
        results["CRPS"] = crps_gaussian(y_true, y_pred, y_sigma, mask)
    return results


def add_efficiency(metrics: dict, stats: dict) -> dict:
    """Merge param count and inference-time stats into a metrics dict."""
    metrics["Params"]      = stats["params"]
    metrics["InferTime_s"] = stats["infer_s"]
    metrics["ms_per_fc"]   = 1000.0 * stats["infer_s"] / max(stats["n_fc"], 1)
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Print utilities
# ──────────────────────────────────────────────────────────────────────────────

def print_table(results: dict, header: str = ""):
    if header:
        print(f"\n{header}")
    row_fmt  = (f"{'Method':<30} {'MAE':>10} {'RMSE':>10} {'CRPS':>12} "
                f"{'Params':>10} {'Time(s)':>10} {'ms/fc':>9}")
    sep      = "─" * len(row_fmt)
    print(row_fmt)
    print(sep)
    for name, m in results.items():
        mae_s  = f"{m['MAE']:.4f}"  if "MAE"  in m else "——"
        rmse_s = f"{m['RMSE']:.4f}" if "RMSE" in m else "——"
        crps_s = f"{m['CRPS']:.4f}" if "CRPS" in m else "——"
        par_s  = f"{m['Params']/1e6:.2f}M"   if m.get("Params") is not None else "——"
        tim_s  = f"{m['InferTime_s']:.2f}"   if "InferTime_s" in m else "——"
        ms_s   = f"{m['ms_per_fc']:.2f}"     if "ms_per_fc"   in m else "——"
        print(f"{name:<30} {mae_s:>10} {rmse_s:>10} {crps_s:>12} "
              f"{par_s:>10} {tim_s:>10} {ms_s:>9}")


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

    # ── 3. TimesFM zero-shot ──────────────────────────────────────────────────
    if not args.skip_timesfm:
        print("\n─── TimesFM 2.5 (zero-shot) ─────────────────────────────────────────")
        y_true, y_pred, y_sigma, stats = run_timesfm_zero_shot(
            val_raw, test_raw, max_sites=args.max_sites
        )
        results["TimesFM 2.5 (zero-shot)"] = add_efficiency(
            compute_metrics(y_true, y_pred, y_sigma), stats)

    # ── 4. Summary table ──────────────────────────────────────────────────────
    print_table(results, header="═" * 65 + "\nFINAL COMPARISON\n" + "═" * 65)

    if args.out:
        import json
        out_path = Path(args.out)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare REnFormer-Skip vs TimesFM on SEN Chile.")
    p.add_argument("--csv",              required=True,  help="Path to SEN Chile CSV")
    p.add_argument("--cache_dir",        default=None,   help="Parquet cache directory (shared with run_experiment_skip.py)")
    p.add_argument("--checkpoint_dir",   default="checkpoint_skip", help="Orbax checkpoint directory for REnFormer-Skip (default: checkpoint_skip)")
    p.add_argument("--max_sites",        type=int,       default=None,   help="Restrict to first N sites (faster smoke tests)")
    p.add_argument("--skip_timesfm",     action="store_true", help="Skip TimesFM zero-shot evaluation")
    p.add_argument("--skip_renformer",   action="store_true", help="Skip REnFormer-Skip checkpoint evaluation")
    p.add_argument("--out",              default=None,   help="Write JSON results to this file")
    args = p.parse_args()

    run(args)
