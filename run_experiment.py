"""
run_experiment.py — End-to-end reproduction of the REnFormer paper methodology.

Usage
-----
python run_experiment.py --csv data/Descarga_Generación_Real_2026-05-29_18-57-56.csv

Optional flags:
  --epochs      int   (default 50)
  --batch_size  int   (default 64)
  --steps_ep    int   gradient steps per epoch (default 1000)
  --max_sites   int   cap per-site baselines to N sites (default: all)
  --skip_baselines    skip slow per-site MLP / LSTM
  --ablation          also run REnFormer-MSE ablation
"""
import argparse
import numpy as np
import jax
import jax.numpy as jnp

from renformer.model import TimeSeriesTransformer
from renformer.train import SENDataset, train, make_eval_step, mse_loss, masked_gaussian_nll
from renformer.sen_data import prepare_sen_dataset
from renformer.metrics import evaluate, print_results_table
from renformer.baselines import persistence_forecast, run_per_site_mlp, run_per_site_lstm

# Paper Table 1 hyperparameters
HPARAMS = dict(
    d_model=128,
    num_heads=4,
    num_layers=4,
    mlp_dim=256,
    dropout_rate=0.1,
    max_len=168,   # 7-day lookback
    horizon=24,
    out_features=1,
)

LOOKBACK = 168
HORIZON  = 24


def collect_predictions(params, model, dataset: SENDataset, batch_size=512):
    """Run the model over all windows in `dataset` and collect arrays."""
    eval_step = make_eval_step(model)
    mu_list, sigma_list, y_raw_list = [], [], []

    for x_b, y_norm_b, y_raw_b, _, _ in dataset.sequential_batches(batch_size):
        mean, log_std = eval_step(params, jnp.array(x_b))
        mu_list.append(np.array(mean))
        sigma_list.append(np.array(jnp.exp(log_std)))
        y_raw_list.append(y_raw_b)

    return (
        np.concatenate(y_raw_list),    # (N, H, 1) raw MW — for masking
        np.concatenate(mu_list),       # (N, H, 1) normalised predictions
        np.concatenate(sigma_list),    # (N, H, 1) normalised std
    )


def run(args):
    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    (train_raw, train_norm), \
    (val_raw,   val_norm), \
    (test_raw,  test_norm), \
    norm_stats = prepare_sen_dataset(args.csv)

    train_ds = SENDataset(train_norm, train_raw, LOOKBACK, HORIZON)
    val_ds   = SENDataset(val_norm,   val_raw,   LOOKBACK, HORIZON)
    test_ds  = SENDataset(test_norm,  test_raw,  LOOKBACK, HORIZON)

    print(f"\nDataset sizes  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    results = {}

    # ------------------------------------------------------------------
    # 2. Persistence baseline
    # ------------------------------------------------------------------
    print("\n--- Persistence ---")
    y_true_all, mu_all, sigma_all = [], [], []
    for x_b, _, y_raw_b, _, _ in test_ds.sequential_batches(512):
        pred = persistence_forecast(x_b, HORIZON)
        y_true_all.append(y_raw_b)
        mu_all.append(pred)
    y_true_all  = np.concatenate(y_true_all)
    mu_all      = np.concatenate(mu_all)
    # Persistence predicts in normalised space; denorm for MW metrics
    site_std_arr = norm_stats["std"].values                   # (n_sites,)
    site_mean_arr = norm_stats["mean"].values

    # For global metrics we use the grand mean / std of all sites' normalisation params
    # (exact per-window denorm requires tracking site index — see comments below)
    grand_mean = float(site_mean_arr.mean())
    grand_std  = float(site_std_arr.mean())
    results["Persistence"] = evaluate(y_true_all, mu_all, denorm_mean=grand_mean, denorm_std=grand_std)
    print("  ", results["Persistence"])

    # ------------------------------------------------------------------
    # 3. Per-site baselines  (optional, slow)
    # ------------------------------------------------------------------
    if not args.skip_baselines:
        n = args.max_sites
        print(f"\n--- Per-site MLP (max_sites={n or 'all'}) ---")
        y_true_mlp, y_pred_mlp = run_per_site_mlp(train_ds, val_ds, test_ds, HORIZON, max_sites=n)
        results["Per-site MLP"] = evaluate(y_true_mlp, y_pred_mlp, denorm_mean=grand_mean, denorm_std=grand_std)
        print("  ", results["Per-site MLP"])

        print(f"\n--- Per-site LSTM (max_sites={n or 'all'}) ---")
        y_true_lstm, y_pred_lstm = run_per_site_lstm(train_ds, test_ds, HORIZON, max_sites=n)
        results["Per-site LSTM"] = evaluate(y_true_lstm, y_pred_lstm, denorm_mean=grand_mean, denorm_std=grand_std)
        print("  ", results["Per-site LSTM"])

    # ------------------------------------------------------------------
    # 4. REnFormer-MSE ablation  (optional)
    # ------------------------------------------------------------------
    if args.ablation:
        print("\n--- REnFormer-MSE (ablation) ---")
        model_mse = TimeSeriesTransformer(**HPARAMS)
        params_mse, _ = train(
            model_mse, train_ds, val_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_ep,
            lr=3e-4,
            loss_fn=mse_loss,
        )
        y_true_mse, mu_mse, _ = collect_predictions(params_mse, model_mse, test_ds)
        results["REnFormer-MSE"] = evaluate(y_true_mse, mu_mse, denorm_mean=grand_mean, denorm_std=grand_std)
        print("  ", results["REnFormer-MSE"])

    # ------------------------------------------------------------------
    # 5. REnFormer (full model)
    # ------------------------------------------------------------------
    print("\n--- REnFormer (masked Gaussian NLL) ---")
    model = TimeSeriesTransformer(**HPARAMS)
    params, history = train(
        model, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_ep,
        lr=3e-4,
        loss_fn=masked_gaussian_nll,
    )
    y_true_rf, mu_rf, sigma_rf = collect_predictions(params, model, test_ds)
    results["REnFormer"] = evaluate(
        y_true_rf, mu_rf, sigma_rf,
        denorm_mean=grand_mean, denorm_std=grand_std,
    )
    print("  ", results["REnFormer"])

    # ------------------------------------------------------------------
    # 6. Results table
    # ------------------------------------------------------------------
    print("\n=== Test Results (non-trivial timesteps, MW scale) ===")
    print_results_table(results)

    return params, results, history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",            required=True, help="Path to SEN Chile CSV")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--steps_ep",       type=int,   default=1000,
                   help="Gradient steps per epoch (controls compute budget)")
    p.add_argument("--max_sites",      type=int,   default=None,
                   help="Cap per-site baselines to N sites")
    p.add_argument("--skip_baselines", action="store_true",
                   help="Skip per-site MLP/LSTM (faster smoke test)")
    p.add_argument("--ablation",       action="store_true",
                   help="Also train REnFormer-MSE ablation")
    args = p.parse_args()
    run(args)
