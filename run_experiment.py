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
    # 2. Train REnFormer (metrics reported every epoch)
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
    p.add_argument("--csv",        required=True, help="Path to SEN Chile CSV")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps_ep",   type=int, default=1000,
                   help="Gradient steps per epoch")
    args = p.parse_args()
    run(args)
