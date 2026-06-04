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
import jax
import jax.numpy as jnp

from renformer.model import TimeSeriesTransformer
from renformer.train import (
    SENDataset, train, make_eval_step, masked_gaussian_nll,
    save_checkpoint, load_checkpoint, checkpoint_exists,
)
from renformer.sen_data import (
    prepare_sen_dataset,
    save_prepared_dataset, load_prepared_dataset, prepared_dataset_exists,
)
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


def last_window_metrics(params, model, test_ds: SENDataset, grand_mean, grand_std):
    """
    Evaluate the model on the final LOOKBACK window of the test set.
    Uses the last valid start position across all sites.
    """
    eval_step = make_eval_step(model)
    last_t = test_ds.n_windows - 1
    all_s   = np.arange(test_ds.S)
    last_t_arr = np.full(test_ds.S, last_t, dtype=np.int64)

    X, _, Y_raw, _, _ = test_ds._fetch(all_s, last_t_arr)
    mean, log_std = eval_step(params, jnp.array(X))
    sigma = np.array(jnp.exp(log_std))

    return evaluate(
        Y_raw, np.array(mean), sigma,
        denorm_mean=grand_mean, denorm_std=grand_std,
    )


def run(args):
    # ------------------------------------------------------------------
    # 1. Data  (load from parquet cache when available)
    # ------------------------------------------------------------------
    if args.cache_dir and prepared_dataset_exists(args.cache_dir):
        print(f"Loading preprocessed dataset from cache: {args.cache_dir}")
        result = load_prepared_dataset(args.cache_dir)
    else:
        result = prepare_sen_dataset(args.csv)
        if args.cache_dir:
            save_prepared_dataset(result, args.cache_dir)

    (train_raw, train_norm), \
    (val_raw,   val_norm), \
    (test_raw,  test_norm), \
    norm_stats = result

    train_ds = SENDataset(train_norm, train_raw, LOOKBACK, HORIZON)
    val_ds   = SENDataset(val_norm,   val_raw,   LOOKBACK, HORIZON)
    test_ds  = SENDataset(test_norm,  test_raw,  LOOKBACK, HORIZON)

    print(f"\nDataset sizes  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    grand_mean = float(norm_stats["mean"].values.mean())
    grand_std  = float(norm_stats["std"].values.mean())

    # ------------------------------------------------------------------
    # 2. Train REnFormer  (or restore from checkpoint with --resume)
    # ------------------------------------------------------------------
    model = TimeSeriesTransformer(**HPARAMS)

    if args.resume and checkpoint_exists(args.checkpoint_dir):
        print(f"\n--- Restoring params from {args.checkpoint_dir} (skipping training) ---")
        in_feat     = getattr(model, "in_features", model.out_features)
        dummy_x     = jnp.zeros((1, LOOKBACK, in_feat))
        params_like = model.init(jax.random.PRNGKey(0), dummy_x, train=False)
        params      = load_checkpoint(params_like, args.checkpoint_dir)
        history     = {}
    else:
        print("\n--- REnFormer (masked Gaussian NLL) ---")
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
        save_checkpoint(params, args.checkpoint_dir)
        print(f"Checkpoint saved → {args.checkpoint_dir}/")

    # ------------------------------------------------------------------
    # 3. Full test-set evaluation
    # ------------------------------------------------------------------
    print("\n--- Test-set evaluation ---")
    y_true, mu, sigma = collect_predictions(params, model, test_ds)
    test_metrics = evaluate(y_true, mu, sigma, denorm_mean=grand_mean, denorm_std=grand_std)
    print_results_table({"REnFormer (test)": test_metrics})

    # ------------------------------------------------------------------
    # 4. Last-window performance (final LOOKBACK hours → HORIZON forecast)
    # ------------------------------------------------------------------
    print(f"\n--- Last {LOOKBACK}-hour window → {HORIZON}-hour forecast ---")
    last_metrics = last_window_metrics(params, model, test_ds, grand_mean, grand_std)
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
    p.add_argument("--cache_dir",       default=None,
                   help="Directory to cache/restore preprocessed parquet splits")
    p.add_argument("--checkpoint_dir",  default="checkpoints",
                   help="Directory for Orbax params checkpoint (default: checkpoints)")
    p.add_argument("--resume",          action="store_true",
                   help="Load params from --checkpoint_dir and skip training")
    args = p.parse_args()
    run(args)
