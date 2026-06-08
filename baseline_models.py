"""Baseline forecasting models for time series analysis.

Sections
--------
1. Statistical baselines (pandas/numpy) — arithmetic mean, rolling mean,
   seasonal naive; operate on single-site DataFrames.
2. SEN Chile neural baselines (JAX/Flax) — Persistence, per-site MLP,
   per-site LSTM; import from renformer.baselines and compare against a
   saved REnFormer checkpoint.

Run section 2 as a script:
    python baseline_models.py --csv <path> [--cache_dir data/] [--max_sites N]
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Statistical baselines (single-site, DataFrame-level)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_arithmetic_mean_model(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    target: str,
    window_size: int,
) -> pd.DataFrame:
    """Arithmetic mean baseline model. There exists a single constant level.
    Compute one mean from history. Predict this constant mean for all future
    timestamps.

    Args:
        train_set (pd.DataFrame): Training dataset
        test_set (pd.DataFrame): Test dataset
        target (str): Target column name
        window_size (int): Number of past observations to use for the mean

    Returns:
        pd.DataFrame: DataFrame with actual and predicted values
    """
    mean_temp = train_set.tail(window_size).mean()
    mean_pred = np.full(shape=len(test_set), fill_value=mean_temp)
    arithmetic_mean_baseline_df = pd.DataFrame()
    arithmetic_mean_baseline_df[target] = test_set[target]
    arithmetic_mean_baseline_df["arithmetic_mean_pred"] = mean_pred
    return arithmetic_mean_baseline_df


def baseline_rolling_mean_model(
    data: pd.DataFrame, test_set: pd.DataFrame, target: str, window_size: int
) -> pd.DataFrame:
    """Rolling mean baseline model.The level slowly changes over time.
    At each time step, recompute the mean of the past `window_size`
    observations.

    Args:
        data (pd.DataFrame): Full dataset (train + test)
        target (str): Target column name
        window_size (int): Number of past observations to use for the rolling mean
    Returns:
        pd.DataFrame: DataFrame with actual and predicted values
    """
    pred_all = data.shift(1).rolling(window_size).mean()

    y_true = test_set[target]
    y_pred = pred_all.loc[y_true.index].dropna()
    y_true = y_true.loc[y_pred.index]

    rolling_mean_baseline_df = pd.DataFrame()
    rolling_mean_baseline_df[target] = y_true
    rolling_mean_baseline_df["rolling_mean_pred"] = y_pred
    return rolling_mean_baseline_df


def baseline_model_seasonal_naive(
    test_set: pd.DataFrame, target: str, period: int
) -> pd.DataFrame:
    """Create a seasonal naive persistence model with a specified period.

    Args:
        test_set (pd.DataFrame): Test dataset
        target (str): Target column name
        period (int): Seasonal period (in number of time steps)

    Returns:
        pd.DataFrame: DataFrame with actual and predicted values
    """
    baseline_df = pd.DataFrame()
    shifted_column = test_set[target].shift(period)
    baseline_df[target] = test_set[target]
    baseline_df[f"H+{period}"] = shifted_column
    baseline_df.dropna(inplace=True)
    return baseline_df


def baseline_arithmetic_mean_for_multihorizon(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    target: str,
    window_size: int,
    forecast_days: int,
    timestamps_per_day: int,
) -> pd.DataFrame:
    """
    Arithmetic mean baseline model for multi-horizon forecasting.

    Computes a single arithmetic mean from the last window_size observations
    in the training set and uses this constant value as prediction for all
    forecast horizons.

    Args:
        train_set (pd.DataFrame): Training dataset
        test_set (pd.DataFrame): Test dataset
        target (str): Target column name
        window_size (int): Number of past observations to use for the mean
        forecast_days (int): Number of days to forecast
        timestamps_per_day (int): Number of timestamps per day

    Returns:
        DataFrame with actual values and predictions for each horizon
        (H+24, H+48, etc.)
    """
    arithmetic_mean_baseline_df = pd.DataFrame()
    arithmetic_mean_baseline_df[target] = test_set[target]

    for i in range(1, forecast_days + 1):
        period = i * timestamps_per_day
        mean_temp = train_set.tail(window_size).mean()
        mean_pred = np.full(shape=len(test_set), fill_value=mean_temp)
        arithmetic_mean_baseline_df[target] = test_set[target]
        arithmetic_mean_baseline_df[f"H+{period}"] = mean_pred
    return arithmetic_mean_baseline_df


def baseline_rolling_mean_for_multihorizon(
    data: pd.DataFrame,
    test_set: pd.DataFrame,
    target: str,
    window_size: int,
    forecast_days: int,
    timestamps_per_day: int,
) -> pd.DataFrame:
    """Rolling mean baseline model for multi-horizon forecasting.

    Computes a rolling mean of the past window_size observations (using shift(1)
    to prevent data leakage) at each time step. The level slowly adapts over time
    as new observations become available. The rolling mean represents the forecast level
    available at time t.

    Args:
        data: Full dataset (train + test)
        target: Target column name
        window_size: Number of past observations for rolling mean calculation
        forecast_days: Number of days to forecast
        timestamps_per_day: Number of timestamps per day (e.g., 24 for hourly)

    Returns:
        DataFrame with actual values and predictions for each horizon
        (H+24, H+48, etc.)
    """
    rolling_mean_baseline_df = pd.DataFrame()

    rolling_mean = data.shift(1).rolling(window_size).mean()
    rolling_mean_baseline_df[target] = test_set[target]

    for i in range(1, forecast_days + 1):
        period = i * timestamps_per_day
        rolling_mean_baseline_df[f"H+{period}"] = rolling_mean.shift(-period)
    rolling_mean_baseline_df.dropna(inplace=True)
    return rolling_mean_baseline_df


def baseline_model_seasonal_naive_1h_for_multihorizon(
    test_set: pd.DataFrame,
    target: str,
    forecast_days: int,
    timestamps_per_day: int,
) -> pd.DataFrame:
    """Hourly persistence baseline model for multi-horizon forecasting.

    Uses the value from 1 hour ago as prediction for each forecast horizon.
    This is the simplest form of persistence model where each prediction
    is based on the observation from exactly 1 time step back.

    Args:
        test_set: Test dataset
        target: Target column name
        forecast_days: Number of days to forecast
        timestamps_per_day: Number of timestamps per day (e.g., 24 for hourly)

    Returns:
        DataFrame with actual values and predictions for each horizon
        (H+24, H+48, etc.)
    """
    baseline_df = pd.DataFrame()
    baseline_df[target] = test_set[target]
    lag1 = test_set[target].shift(1)
    for i in range(1, forecast_days + 1):
        period = i * timestamps_per_day
        baseline_df[f"H+{period}"] = lag1.shift(period)

    baseline_df.dropna(inplace=True)
    return baseline_df


def baseline_model_seasonal_naive_24h_for_multihorizon(
    test_set: pd.DataFrame,
    target: str,
    forecast_days: int,
    timestamps_per_day: int,
) -> pd.DataFrame:
    """Daily seasonal naive baseline model for multi-horizon forecasting.

    Uses the value from exactly 24 hours ago (same time yesterday) as
    prediction for each forecast horizon. This captures daily seasonality
    patterns by assuming that values repeat with a 24-hour cycle.

    Args:
        test_set: Test dataset
        target: Target column name
        forecast_days: Number of days to forecast
        timestamps_per_day: Number of timestamps per day (e.g., 24 for hourly)

    Returns:
        DataFrame with actual values and predictions for each horizon
        (H+24, H+48, etc.)
    """
    baseline_df = pd.DataFrame()
    baseline_df[target] = test_set[target]
    for i in range(1, forecast_days + 1):
        period = i * timestamps_per_day
        shifted_column = test_set[target].shift(period)
        baseline_df[f"H+{period}"] = shifted_column

    baseline_df.dropna(inplace=True)
    return baseline_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEN Chile neural baselines (JAX/Flax, multi-site sliding-window)
# ─────────────────────────────────────────────────────────────────────────────

LOOKBACK = 168   # must match run_experiment.py
HORIZON  = 24


def run_sen_baselines(train_ds, val_ds, test_ds, grand_mean, grand_std):
    """
    Run Persistence, per-site MLP, and per-site LSTM on SEN test windows.

    All methods evaluate on exactly the sites present in the supplied datasets.
    To restrict evaluation, slice the DataFrames to the desired sites before
    constructing the SENDataset objects (see __main__).

    All predictions are in z-scored space; evaluate() converts to MW via
    grand_mean / grand_std (averages of per-site normalisation stats).

    Returns a dict  {method_label: metrics_dict}  ready for print_results_table.
    """
    from renformer.baselines import persistence_forecast, run_per_site_mlp, run_per_site_lstm
    from renformer.metrics import evaluate

    n = test_ds.S
    results = {}

    # Persistence — repeat the last z-scored observation for every horizon step
    print(f"\n--- Persistence [{n} sites] ---")
    y_true_list, y_pers_list = [], []
    for x_b, _, y_raw_b, _, _ in test_ds.sequential_batches(512):
        y_pers = persistence_forecast(x_b[..., :1], HORIZON)  # power channel only
        y_true_list.append(y_raw_b)
        y_pers_list.append(y_pers)
    results["Persistence      [single-site]"] = evaluate(
        np.concatenate(y_true_list),
        np.concatenate(y_pers_list),
        denorm_mean=grand_mean,
        denorm_std=grand_std,
    )

    # Per-site MLP — independent 256→128→H model per site
    print(f"\n--- Per-site MLP [{n} sites] ---")
    mlp_true, mlp_pred = run_per_site_mlp(
        train_ds, val_ds, test_ds, horizon=HORIZON,
    )
    results["MLP (per-site)   [single-site]"] = evaluate(
        mlp_true, mlp_pred,
        denorm_mean=grand_mean, denorm_std=grand_std,
    )

    # Per-site LSTM — independent 128-hidden LSTM per site
    print(f"\n--- Per-site LSTM [{n} sites] ---")
    lstm_true, lstm_pred = run_per_site_lstm(
        train_ds, test_ds, horizon=HORIZON,
    )
    results["LSTM (per-site)  [single-site]"] = evaluate(
        lstm_true, lstm_pred,
        denorm_mean=grand_mean, denorm_std=grand_std,
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import jax
    import jax.numpy as jnp

    from renformer.model import TimeSeriesTransformer
    from renformer.train import SENDataset, make_eval_step, load_checkpoint, checkpoint_exists
    from renformer.sen_data import (
        prepare_sen_dataset,
        save_prepared_dataset, load_prepared_dataset, prepared_dataset_exists,
    )
    from renformer.metrics import evaluate, print_results_table

    HPARAMS = dict(
        d_model=128, num_heads=4, num_layers=4, mlp_dim=256,
        dropout_rate=0.1, max_len=LOOKBACK, horizon=HORIZON, out_features=1,
    )

    p = argparse.ArgumentParser(
        description="Evaluate SEN baselines and optionally compare with REnFormer."
    )
    p.add_argument("--csv",            required=True,  help="Path to SEN Chile CSV")
    p.add_argument("--cache_dir",      default=None,   help="Parquet cache directory")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="REnFormer checkpoint directory (compared if present)")
    p.add_argument("--max_sites",      type=int, default=None,
                   help="Restrict all methods to the first N sites")
    args = p.parse_args()

    # Load data
    if args.cache_dir and prepared_dataset_exists(args.cache_dir):
        print(f"Loading preprocessed dataset from cache: {args.cache_dir}")
        result = load_prepared_dataset(args.cache_dir)
    else:
        result = prepare_sen_dataset(args.csv)
        if args.cache_dir:
            save_prepared_dataset(result, args.cache_dir)

    (train_raw, train_norm), (val_raw, val_norm), (test_raw, test_norm), norm_stats = result

    # Restrict every split and norm_stats to the same N sites so that all
    # methods (Persistence, MLP, LSTM, REnFormer) are evaluated on identical data.
    if args.max_sites is not None:
        cols       = train_norm.columns[:args.max_sites]
        train_norm = train_norm[cols]
        train_raw  = train_raw[cols]
        val_norm   = val_norm[cols]
        val_raw    = val_raw[cols]
        test_norm  = test_norm[cols]
        test_raw   = test_raw[cols]
        norm_stats = {
            "mean":  norm_stats["mean"].loc[cols],
            "std":   norm_stats["std"].loc[cols],
            "sites": cols.tolist(),
        }

    train_ds = SENDataset(train_norm, train_raw, LOOKBACK, HORIZON)
    val_ds   = SENDataset(val_norm,   val_raw,   LOOKBACK, HORIZON)
    test_ds  = SENDataset(test_norm,  test_raw,  LOOKBACK, HORIZON)

    grand_mean = float(norm_stats["mean"].values.mean())
    grand_std  = float(norm_stats["std"].values.mean())

    n_sites = test_ds.S
    print(f"\nSites: {n_sites}  |  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    # Run baselines
    results = run_sen_baselines(train_ds, val_ds, test_ds, grand_mean, grand_std)

    # Optionally include REnFormer from checkpoint for side-by-side comparison
    if checkpoint_exists(args.checkpoint_dir):
        print(f"\n--- REnFormer from checkpoint: {args.checkpoint_dir} [{n_sites} sites] ---")
        model     = TimeSeriesTransformer(**HPARAMS)
        dummy_x   = jnp.zeros((1, LOOKBACK, model.in_features))
        params    = load_checkpoint(
            model.init(jax.random.PRNGKey(0), dummy_x, train=False),
            args.checkpoint_dir,
        )
        eval_step = make_eval_step(model)

        mu_list, sigma_list, y_raw_list = [], [], []
        for x_b, _, y_raw_b, _, _ in test_ds.sequential_batches(512):
            mean, log_std = eval_step(params, jnp.array(x_b))
            mu_list.append(np.array(mean))
            sigma_list.append(np.array(jnp.exp(log_std)))
            y_raw_list.append(y_raw_b)

        renformer_metrics = evaluate(
            np.concatenate(y_raw_list),
            np.concatenate(mu_list),
            np.concatenate(sigma_list),
            denorm_mean=grand_mean, denorm_std=grand_std,
        )
        all_results = {"REnFormer        [multi-site]": renformer_metrics, **results}
    else:
        all_results = results

    print(f"\n{'='*55}")
    print("Multi-site vs Single-site comparison")
    print(f"{'='*55}")
    print_results_table(all_results)
