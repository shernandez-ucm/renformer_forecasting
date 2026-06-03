"""
run_experiment.py — Train REnFormer on SEN Chile data.

Usage
-----
python run_experiment.py --csv data/Descarga_Generación_Real_2026-05-29_18-57-56.csv

Optional flags:
  --epochs      int   (default 50)
  --batch_size  int   (default 64)
  --steps_ep    int   gradient steps per epoch (default 1000)
"""
import argparse
import json
import os
import pickle
import numpy as np
import jax.numpy as jnp

from renformer.model import TimeSeriesTransformer
from renformer.train import SENDataset, train, make_eval_step, masked_gaussian_nll
from renformer.sen_data import prepare_sen_dataset
from renformer.metrics import evaluate, print_results_table

LOOKBACK = 168   # 7-day context window
HORIZON  = 24    # 24-hour forecast horizon

# Paper Table 1 hyperparameters
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


def _save_checkpoint(checkpoint_dir: str, name: str, params, history: dict):
    os.makedirs(checkpoint_dir, exist_ok=True)
    params_path  = os.path.join(checkpoint_dir, f"{name}_params.pkl")
    history_path = os.path.join(checkpoint_dir, f"{name}_history.json")
    # Convert JAX arrays → numpy so the pickle is portable across JAX versions.
    np_params = jax.tree_util.tree_map(np.array, params)
    with open(params_path, "wb") as f:
        pickle.dump(np_params, f)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Checkpoint saved → {params_path}, {history_path}")


def collect_predictions(params, model, dataset: SENDataset, batch_size=512):
    eval_step = make_eval_step(model)
    mu_list, sigma_list, y_raw_list = [], [], []

    for x_b, _, y_raw_b, _, _ in dataset.sequential_batches(batch_size):
        mean, log_std = eval_step(params, jnp.array(x_b))
        mu_list.append(np.array(mean))
        sigma_list.append(np.array(jnp.exp(log_std)))
        y_raw_list.append(y_raw_b)

    return (
        np.concatenate(y_raw_list),
        np.concatenate(mu_list),
        np.concatenate(sigma_list),
    )


def last_window_metrics(params, model, test_ds: SENDataset):
    """
    Evaluate the model on the final LOOKBACK window of the test set.
    Uses the last valid start position across all sites.
    RevIN denormalises outputs to raw MW, so no external denorm stats needed.
    """
    eval_step = make_eval_step(model)
    last_t     = test_ds.n_windows - 1
    all_s      = np.arange(test_ds.S)
    last_t_arr = np.full(test_ds.S, last_t, dtype=np.int64)

    X, _, Y_raw, _, _ = test_ds._fetch(all_s, last_t_arr)
    mean, log_std = eval_step(params, jnp.array(X))
    sigma = np.array(jnp.exp(log_std))

    # Model outputs are in raw MW — evaluate directly with no linear rescaling.
    return evaluate(Y_raw, np.array(mean), sigma)


def run(args):
    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    (train_raw, train_norm), \
    (val_raw,   val_norm), \
    (test_raw,  test_norm), \
    _ = prepare_sen_dataset(args.csv)

    # raw_input=True: feed raw MW to the model so RevIN performs a clean
    # per-window normalisation from scratch (no prior global z-score).
    train_ds = SENDataset(train_norm, train_raw, LOOKBACK, HORIZON, raw_input=True)
    val_ds   = SENDataset(val_norm,   val_raw,   LOOKBACK, HORIZON, raw_input=True)
    test_ds  = SENDataset(test_norm,  test_raw,  LOOKBACK, HORIZON, raw_input=True)

    print(f"\nDataset sizes  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    # ------------------------------------------------------------------
<<<<<<< Updated upstream
    # 2. Train REnFormer (metrics reported every epoch)
=======
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
        params_mse, history_mse = train(
            model_mse, train_ds, val_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_ep,
            lr=3e-4,
            loss_fn=mse_loss,
        )
        _save_checkpoint(args.checkpoint_dir, "renformer_mse", params_mse, history_mse)
        y_true_mse, mu_mse, _ = collect_predictions(params_mse, model_mse, test_ds)
        results["REnFormer-MSE"] = evaluate(y_true_mse, mu_mse, denorm_mean=grand_mean, denorm_std=grand_std)
        print("  ", results["REnFormer-MSE"])

    # ------------------------------------------------------------------
    # 5. REnFormer (full model)
>>>>>>> Stashed changes
    # ------------------------------------------------------------------
    print("\n--- REnFormer (masked Gaussian NLL) ---")
    model = TimeSeriesTransformer(**HPARAMS)
    # train_target="raw": supervise in raw MW so the loss is in the same
    # space as RevIN's denormalised outputs.
    params, history = train(
        model, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_ep,
        lr=3e-4,
        loss_fn=masked_gaussian_nll,
        train_target="raw",
    )
<<<<<<< Updated upstream
=======
    _save_checkpoint(args.checkpoint_dir, "renformer", params, history)
    y_true_rf, mu_rf, sigma_rf = collect_predictions(params, model, test_ds)
    results["REnFormer"] = evaluate(
        y_true_rf, mu_rf, sigma_rf,
        denorm_mean=grand_mean, denorm_std=grand_std,
    )
    print("  ", results["REnFormer"])
>>>>>>> Stashed changes

    # ------------------------------------------------------------------
    # 3. Full test-set evaluation
    # ------------------------------------------------------------------
    # RevIN denormalises outputs to raw MW — evaluate directly.
    print("\n--- Test-set evaluation ---")
    y_true, mu, sigma = collect_predictions(params, model, test_ds)
    test_metrics = evaluate(y_true, mu, sigma)
    print_results_table({"REnFormer (test)": test_metrics})

    # ------------------------------------------------------------------
    # 4. Last-window performance (final LOOKBACK hours → HORIZON forecast)
    # ------------------------------------------------------------------
    print(f"\n--- Last {LOOKBACK}-hour window → {HORIZON}-hour forecast ---")
    last_metrics = last_window_metrics(params, model, test_ds)
    print_results_table({"REnFormer (last window)": last_metrics})

    return params, history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",            required=True, help="Path to SEN Chile CSV")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--steps_ep",       type=int,   default=1000,
                   help="Gradient steps per epoch (controls compute budget)")
    p.add_argument("--max_sites",      type=int,   default=None,
                   help="Cap per-site baselines to N sites")
    p.add_argument("--skip_baselines",  action="store_true",
                   help="Skip per-site MLP/LSTM (faster smoke test)")
    p.add_argument("--ablation",        action="store_true",
                   help="Also train REnFormer-MSE ablation")
    p.add_argument("--checkpoint_dir",  default="checkpoints",
                   help="Directory to write params (.pkl) and history (.json) (default: checkpoints)")
    args = p.parse_args()
    run(args)
