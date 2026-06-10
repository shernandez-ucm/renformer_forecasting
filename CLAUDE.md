# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**REnFormer** is a global Transformer for probabilistic multi-site solar generation forecasting, trained on Chile's SEN grid data. It outputs Gaussian `(mean, log_std)` predictions evaluated with masked NLL (penalising only non-trivial generation timesteps, raw MW > 0.1).

The repo also includes **TimesFM 2.5** zero-shot and fine-tuning scripts for comparison.

## Environment

Python 3.12 with a local virtualenv at `env/`. Activate before running anything:

```bash
source env/bin/activate
```

Key dependencies: `jax==0.10.1`, `jaxlib==0.10.1` (CUDA 12), `flax==0.12.7`, `optax==0.2.8`, `numpy`, `pandas`, `scipy`, `timesfm`, `orbax-checkpoint`.

There is no test suite, linter config, or packaging — scripts are run directly from the repo root. `python synthetic_data.py` is the quick sanity check.

## Running the Code

The main SEN data file is `data/data.csv`. A pre-built parquet cache lives in `data/` so you can skip re-parsing the CSV entirely with `--cache_dir data/`.

**Full REnFormer experiment** (real SEN Chile data):
```bash
python run_experiment.py --csv data/data.csv
python run_experiment.py --csv data/data.csv --cache_dir data/  # use parquet cache, skip CSV parse
python run_experiment.py --csv <path> --resume                  # skip training, load checkpoint
```
Checkpoints are saved to `checkpoints/` by default (override with `--checkpoint_dir`). The `--ablation` flag is declared but currently a no-op (never read by `run()`).

**Checkpoint directory naming is inconsistent across scripts** — note plural vs singular: the training scripts default to `checkpoints/` / `checkpoints_skip/`, but `compare_models_skip.py` defaults to `checkpoint_skip` and the trained checkpoints in this working copy live in `checkpoint/` (gitignored) and `checkpoint_skip/`. Pass `--checkpoint_dir` explicitly to be safe.

**REnFormer-Skip experiment** (macro skip-connection variant):
```bash
python run_experiment_skip.py --csv data/data.csv
python run_experiment_skip.py --csv <path> --cache_dir data/   # reuse parquet cache
python run_experiment_skip.py --csv <path> --resume            # load from checkpoints_skip/
```
Checkpoints are saved to `checkpoints_skip/` by default (override with `--checkpoint_dir`).

**Produce per-site forecast from a saved checkpoint**:
```bash
python forecast_checkpoint.py \
    --csv  data/data.csv \
    --checkpoint checkpoints/renformer_params.pkl \
    --out  forecasts/renformer_forecast.csv
```
Outputs columns `timestamp, site, mean_mw, std_mw, q10_mw, q90_mw` plus a `__total__` aggregate row per hour. **Caveat:** this script loads params with `pickle.load` (default `checkpoints/renformer_params.pkl`), but the training pipeline saves Orbax checkpoints and nothing in the repo writes that pickle — the params must be exported to a pickle separately before this script works.

**REnFormer + REnFormer-Skip vs TimesFM 2.5 zero-shot comparison** (loads existing checkpoints, no retraining):
```bash
python compare_models.py --csv <path>
python compare_models.py --csv <path> --cache_dir data/          # reuse parquet cache
python compare_models.py --csv <path> --checkpoint_dir checkpoint
python compare_models.py --csv <path> --max_sites 50             # limit sites for speed
python compare_models.py --csv <path> --skip_timesfm             # REnFormer variants only
python compare_models.py --csv <path> --skip_renformer           # TimesFM only
python compare_models.py --csv <path> --out results.json         # save metrics as JSON
```
Note: it restores **both** the base and Skip models from the same `--checkpoint_dir` (default `checkpoints`).

**Neural baselines vs REnFormer checkpoint** (Persistence, per-site MLP, per-site LSTM from `renformer/baselines.py`):
```bash
python baseline_models.py --csv <path> [--cache_dir data/] [--max_sites N]
```

**REnFormer-Skip vs TimesFM comparison** (loads existing checkpoint, no retraining):
```bash
python compare_models_skip.py --csv <path>
python compare_models_skip.py --csv <path> --cache_dir data/       # reuse parquet cache
python compare_models_skip.py --csv <path> --checkpoint_dir checkpoint_skip
python compare_models_skip.py --csv <path> --max_sites 50          # limit sites for speed
python compare_models_skip.py --csv <path> --skip_timesfm          # REnFormer-Skip only
python compare_models_skip.py --csv <path> --skip_renformer        # TimesFM only
python compare_models_skip.py --csv <path> --out results.json      # save metrics as JSON
```

**Site-scaling sweep** (REnFormer-Skip vs TimesFM across increasing `--max_sites`):
```bash
./run_scaling_skip.sh                  # sweep max_sites 10, 60, 110, … 291
STEP=10 ./run_scaling_skip.sh          # finer sweep
CHECKPOINT_DIR=checkpoint_skip OUT_DIR=results_scaling ./run_scaling_skip.sh
```
Wraps `compare_models_skip.py`, writing one `sites_<N>.json` + `sites_<N>.log` per run to `results_scaling/`. Existing non-empty JSONs are skipped (delete to re-run), so the sweep is resumable. Configured via env vars: `CSV`, `CACHE_DIR`, `CHECKPOINT_DIR` (default `checkpoint_skip`), `OUT_DIR`, `STEP` (default 50 — the header comment saying 10 is stale), `MIN_SITES`, `MAX_SITES` (291 = all sites, always included as the final point). It activates the venv itself.

**TimesFM zero-shot forecasts** (no training needed):
```bash
python forecast_example.py   # all-generation 24 h forecast (total MW, Flax backend)
python forecast_solar.py     # solar-only 24 h forecast (Flax backend, quantile output)
```

**TimesFM fine-tuning** (PyTorch backend; freezes backbone, trains heads + last N layers):
```bash
python finetune_solar.py     # fine-tune on Chilean solar; compares vs zero-shot baseline
```

**Quick sanity check** (no SEN CSV needed — uses random data):
```bash
python synthetic_data.py
```

**EDA**:
```bash
python eda_generacion.py
```

## Architecture

### Module layout

```
renformer/
  model.py      — TransformerBlock → TransformerEncoder/SkipTransformerEncoder → TimeSeriesTransformer/TimeSeriesTransformerSkip
  train.py      — loss functions, JIT step factories, SENDataset, Orbax checkpointing, training loop
  sen_data.py   — SEN Chile CSV loader, site-matrix builder, chronological split, normalization, parquet cache
  metrics.py    — MAE, RMSE, CRPS (active-mask aware)
  baselines.py  — Persistence, per-site MLP, per-site LSTM (all in JAX/Flax)
  data_utils.py — Monash .tsf parser (legacy; not used by run_experiment.py)
run_experiment.py      — end-to-end paper reproduction script; checkpoints to checkpoints/
run_experiment_skip.py — REnFormer-Skip variant (macro encoder skip-connection); checkpoints to checkpoints_skip/
forecast_checkpoint.py — load saved params and produce per-site probabilistic forecast CSV
compare_models.py      — REnFormer + REnFormer-Skip (from checkpoint) vs TimesFM 2.5 zero-shot; imports from compare_models_skip.py
compare_models_skip.py — REnFormer-Skip vs TimesFM 2.5 comparison (loads from checkpoint_skip by default)
run_scaling_skip.sh    — resumable sweep of compare_models_skip.py over --max_sites; JSON+log per point → results_scaling/
forecast_example.py    — TimesFM 2.5 zero-shot all-generation forecast
forecast_solar.py      — TimesFM 2.5 zero-shot solar-only forecast
finetune_solar.py      — TimesFM 2.5 fine-tuning (PyTorch) on solar data
synthetic_data.py      — generates random (X, Y) tensors and runs a short training loop; validates the stack without real data
baseline_models.py     — section 1: pandas/numpy statistical baselines (arithmetic mean, rolling mean, seasonal naive; single-site, DataFrame-level); section 2 (runnable as a script): JAX/Flax Persistence/MLP/LSTM from renformer/baselines.py compared against a saved REnFormer checkpoint
paper/                 — LaTeX source (renformer_paper.tex) and bibliography (renformer.bib)
```

### Model (`model.py`)

Two model variants share the same hyperparameters and RevIN/head design:

**`TimeSeriesTransformer`** (base):
1. Applies **RevIN** with learnable affine parameters (gamma, beta) on the power channel(s); exogenous channels (calendar features) are left unchanged.
2. `TransformerEncoder`: linear projection → sinusoidal PE → N × post-LayerNorm `TransformerBlock` (self-attention + MLP with GELU).
3. **Last-token pooling** → two parallel `Dense` heads for `mean` and `log_std`, each reshaped to `(batch, horizon, out_features)`.
4. Outputs are de-normalised to raw MW units via the RevIN inverse before returning.

**`TimeSeriesTransformerSkip`** (ablation, `run_experiment_skip.py`):
- Replaces `TransformerEncoder` with `SkipTransformerEncoder`, which saves the projected+PE representation (`skip = x`) before the N blocks and adds it back after: `LayerNorm(x + skip)`.
- Uses a simpler instance norm (plain mean/std, no learnable gamma/beta) instead of full RevIN.
- All other hyperparameters are identical, making it a drop-in ablation.

Paper hyperparameters: `d_model=128, num_heads=4, num_layers=4, mlp_dim=256, dropout_rate=0.1, max_len=168` (7-day lookback), `horizon=24`.

### Data pipeline (`sen_data.py`)

Source: CEN "Descarga Generación Real" semicolon-delimited CSV with 24 hour-columns per row.

Pipeline: `load_sen_csv` → `build_site_matrix` (pivot to `(time × site)`, filter Solar type, drop sites with > 10% missing, clip negatives) → `chronological_split` (train < 2024-01-01, val < 2025-01-01, test ≥ 2025-01-01) → `normalize_per_site` (z-score with training stats).

`prepare_sen_dataset` wraps the full pipeline. Results can be persisted with `save_prepared_dataset` / `load_prepared_dataset` (parquet files) to avoid re-parsing. A pre-built cache already exists in `data/` (`train_raw.parquet`, `val_raw.parquet`, `test_raw.parquet`, `train_norm.parquet`, `val_norm.parquet`, `test_norm.parquet`, `norm_stats.parquet`), so `--cache_dir data/` works immediately without re-parsing the CSV.

### Dataset & training (`train.py`)

`SENDataset` is a **lazy sliding-window** sampler over `(S, T)` arrays — avoids materialising ~14 GB of all windows. It supports random `sample_batch` (training) and ordered `sequential_batches` (evaluation). Optional calendar features (`sin/cos` hour + day-of-year) can be appended as extra input channels.

`make_train_step` / `make_eval_step` return `@jax.jit`-compiled functions closed over the model and optimizer. The training loop uses cosine-decay Adam with `train_target="raw"` (supervise in MW, not z-score, when RevIN is active).

The experiment scripts build datasets with `raw_input=True`: the model receives **raw MW** input and RevIN/instance-norm handles per-window scale (mirroring TimesFM's `normalize_inputs=True`), so per-site capacity information is preserved and model outputs are directly in MW. Consequently all evaluations use `evaluate()` / `compute_metrics()` **without** de-normalization arguments — predictions are never rescaled. Checkpoints trained under the older z-scored-input configuration are incompatible (they load, but produce wrong-scale outputs) and must be retrained.

Loss: `masked_gaussian_nll` — only penalises timesteps where raw generation > `EPSILON_MW = 0.1`.

Checkpoints use Orbax (`save_checkpoint` / `load_checkpoint`) with `max_to_keep=1`.

### TimesFM scripts

`forecast_example.py` and `forecast_solar.py` use the **Flax** backend (`TimesFM_2p5_200M_flax`) for zero-shot inference. `finetune_solar.py` uses the **PyTorch** backend (`TimesFM_2p5_200M_torch`) because only the torch module exposes a differentiable forward pass; it freezes the backbone and trains output heads + the last `UNFREEZE_LAST_N` transformer layers.

### Evaluation

`metrics.evaluate()` computes MAE, RMSE, and optionally CRPS, all restricted to the active mask (`y_raw > 0.1 MW`). With `raw_input=True` training the model's predictions are already in MW, so `evaluate()` is called without `denorm_mean` / `denorm_std` (those arguments only apply to models that predict in z-score space).
