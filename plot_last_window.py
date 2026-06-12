"""
Plot the last-window 24 h forecast for one site from three models:
    • REnFormer       (base checkpoint)
    • REnFormer-Skip  (skip checkpoint)
    • TimesFM 2.5     (zero-shot)

Style mirrors docs/Unknown.png: three stacked panels sharing the site name as
title, black ground-truth line, blue mean forecast, light-blue ±1σ band,
LaTeX-serif font and grid. The x-axis is the 24 h forecast horizon.

The models are trained on raw-MW input (raw_input=True), so their mean/log_std
outputs are already in MW and need no de-normalisation. TimesFM σ is recovered
from its output quantiles via the same OLS fit used in compare_models_skip.py.

Usage
-----
python plot_last_window.py --cache_dir data \
    --renformer_dir checkpoint --skip_dir checkpoint_skip
python plot_last_window.py --cache_dir data --site "El Romero"
python plot_last_window.py --cache_dir data --csv data/data.csv   # no cache
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])
import jax
import jax.numpy as jnp

from renformer.model import TimeSeriesTransformer, TimeSeriesTransformerSkip
from renformer.train import make_eval_step, load_checkpoint, checkpoint_exists
from renformer.sen_data import (
    prepare_sen_dataset,
    load_prepared_dataset,
    prepared_dataset_exists,
    save_prepared_dataset,
)

warnings.filterwarnings("ignore", category=UserWarning)

LOOKBACK = 168
HORIZON  = 24

HPARAMS = dict(
    d_model=128, num_heads=4, num_layers=4, mlp_dim=256,
    dropout_rate=0.1, max_len=LOOKBACK, horizon=HORIZON, out_features=1,
)

# Φ⁻¹(level) for levels 0.1…0.9 — recover Gaussian σ from TimesFM quantiles (OLS).
Z_INV = np.array([-1.2815516, -0.8416212, -0.5244005, -0.2533471, 0.0,
                   0.2533471,  0.5244005,  0.8416212,  1.2815516])


# ──────────────────────────────────────────────────────────────────────────────
# Per-model single-window forecast (returns mean (H,) and sigma (H,) in MW)
# ──────────────────────────────────────────────────────────────────────────────

def renformer_forecast(model_cls, checkpoint_dir, context):
    """context: (LOOKBACK,) raw MW → (mean, sigma) each (HORIZON,) in MW."""
    model = model_cls(**HPARAMS)
    x = jnp.asarray(context, dtype=jnp.float32).reshape(1, LOOKBACK, 1)
    params_like = model.init(jax.random.PRNGKey(0), x, train=False)
    params = load_checkpoint(params_like, checkpoint_dir)
    mean, log_std = make_eval_step(model)(params, x)
    return np.asarray(mean).reshape(-1), np.asarray(jnp.exp(log_std)).reshape(-1)


def timesfm_forecast(context):
    """context: (LOOKBACK,) raw MW → (mean, sigma) each (HORIZON,) in MW."""
    import timesfm

    tfm = timesfm.TimesFM_2p5_200M_flax.from_pretrained("google/timesfm-2.5-200m-flax")
    tfm.compile(timesfm.ForecastConfig(
        max_context=LOOKBACK, max_horizon=HORIZON,
        normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=True, infer_is_positive=True,
        fix_quantile_crossing=True,
    ))
    point_fc, quant_fc = tfm.forecast(
        horizon=HORIZON, inputs=[context.astype(np.float32)]
    )
    mean = np.asarray(point_fc)[0, :HORIZON]
    q = np.asarray(quant_fc)[0, :HORIZON, 1:10]            # (H, 9)
    sigma = np.abs((q @ Z_INV) / (Z_INV @ Z_INV))          # (H,)
    return mean, sigma


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_panels(times, truth, forecasts, site, out):
    """forecasts: list of (label, mean, sigma) → stacked panels (Unknown.png style)."""
    blue, band = "#1f5fa8", "#aac8e8"

    fig, axes = plt.subplots(len(forecasts), 1, figsize=(8, 2.6 * len(forecasts)),
                             sharex=True)
    if len(forecasts) == 1:
        axes = [axes]

    for ax, (label, mean, sigma) in zip(axes, forecasts):
        ax.fill_between(times, mean - sigma, mean + sigma, color=band, zorder=1)
        ax.plot(times, mean, color=blue, lw=1.8, zorder=2, label=label)
        ax.plot(times, truth, color="black", lw=1.5, zorder=3, label="actual")
        ax.set_ylabel("Generation (MW)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate(rotation=0, ha="center")
    axes[0].set_title(site)
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"Saved figure → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/data.csv")
    ap.add_argument("--cache_dir", default="data")
    ap.add_argument("--renformer_dir", default="checkpoint",
                    help="REnFormer (base) Orbax checkpoint dir")
    ap.add_argument("--skip_dir", default="checkpoint_skip",
                    help="REnFormer-Skip Orbax checkpoint dir")
    ap.add_argument("--site", default=None,
                    help="Site name (default: highest-output site in the test split)")
    ap.add_argument("--skip_timesfm", action="store_true")
    ap.add_argument("--out", default="figures/last_window_forecast.png")
    args = ap.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.cache_dir and prepared_dataset_exists(args.cache_dir):
        print(f"Loading cached dataset from {args.cache_dir}")
        result = load_prepared_dataset(args.cache_dir)
    else:
        result = prepare_sen_dataset(args.csv)
        if args.cache_dir:
            save_prepared_dataset(result, args.cache_dir)
    (_, _), (_, _), (test_raw, _), _ = result

    site = args.site or test_raw.mean().idxmax()
    if site not in test_raw.columns:
        raise SystemExit(f"Site '{site}' not found. Examples: {list(test_raw.columns[:5])}")
    print(f"Site: {site}")

    # Second-to-last full window: shift back one HORIZON from the tail so the
    # target is the penultimate 24 h block [LOOKBACK context | HORIZON target].
    series = test_raw[site].to_numpy(dtype=np.float32)
    win = series[-(LOOKBACK + 2 * HORIZON):-HORIZON]
    context, truth = win[:LOOKBACK], win[LOOKBACK:]
    times = test_raw.index[-2 * HORIZON:-HORIZON]
    print(f"Forecast window: {times[0]} → {times[-1]}")

    # ── Run models ─────────────────────────────────────────────────────────────
    forecasts = []
    if checkpoint_exists(args.renformer_dir):
        m, s = renformer_forecast(TimeSeriesTransformer, args.renformer_dir, context)
        forecasts.append(("REnFormer", m, s))
    else:
        print(f"  (no REnFormer checkpoint at '{args.renformer_dir}' — skipping)")

    if checkpoint_exists(args.skip_dir):
        m, s = renformer_forecast(TimeSeriesTransformerSkip, args.skip_dir, context)
        forecasts.append(("REnFormer-Skip", m, s))
    else:
        print(f"  (no REnFormer-Skip checkpoint at '{args.skip_dir}' — skipping)")

    if not args.skip_timesfm:
        m, s = timesfm_forecast(context)
        forecasts.append(("TimesFM 2.5", m, s))

    if not forecasts:
        raise SystemExit("No models available to plot.")

    plot_panels(times, truth, forecasts, site, args.out)


if __name__ == "__main__":
    main()
