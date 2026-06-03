# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**REnFormer** is a global Transformer for probabilistic multi-site solar generation forecasting, trained on Chile's SEN grid data. It outputs Gaussian `(mean, log_std)` predictions evaluated with masked NLL (penalising only non-trivial generation timesteps, raw MW > 0.1).

## Environment

Python 3.12 with a local virtualenv at `env/`. Activate before running anything:

```bash
source env/bin/activate
```

Key dependencies: `jax==0.10.1`, `jaxlib==0.10.1` (CUDA 12), `flax==0.12.7`, `optax==0.2.8`, `numpy`, `pandas`, `scipy` (optional; pure-numpy CRPS fallback exists), `matplotlib`, `seaborn`.

TimesFM scripts only: `timesfm` (Flax backend for `forecast_solar.py`; PyTorch backend for `finetune_solar.py`).

## Running Experiments

**Full experiment** (real SEN Chile data):
```bash
python run_experiment.py --csv data/Descarga_Generación_Real_2026-05-29_18-57-56.csv
python run_experiment.py --csv <path> --skip_baselines          # faster, skips MLP/LSTM
python run_experiment.py --csv <path> --ablation                # also trains MSE variant
python run_experiment.py --csv <path> --checkpoint_dir out/ckpt # custom checkpoint dir
```

Note: `synthetic_data.py` uses the old `train()` API and is currently broken — `run_experiment.py` is the authoritative entry point.

**Forecast from checkpoint** (uses the last LOOKBACK hours of the dataset as context):
```bash
python forecast_checkpoint.py --csv <path> --checkpoint checkpoints/renformer_params.pkl
python forecast_checkpoint.py --csv <path> --checkpoint <pkl> --out forecasts/fc.csv
```

**TimesFM comparisons**:
```bash
python forecast_solar.py    # zero-shot 24 h forecast (Flax backend)
python finetune_solar.py    # fine-tune TimesFM on Chilean solar (PyTorch backend)
```

**EDA**:
```bash
python eda_generacion.py    # saves 9 figures to docs/eda/
```

## Architecture

### Module layout

```
renformer/
  model.py      — TransformerBlock → TransformerEncoder → TimeSeriesTransformer
  train.py      — loss functions, JIT step factories, SENDataset, training loop
  sen_data.py   — SEN Chile CSV loader, site-matrix builder, chronological split, normalization
  metrics.py    — MAE, RMSE, CRPS (active-mask aware)
  baselines.py  — Persistence, per-site MLP, per-site LSTM (all in JAX/Flax)
  data_utils.py — Monash .tsf parser (legacy; not used by run_experiment.py)
run_experiment.py      — end-to-end paper reproduction script
forecast_checkpoint.py — inference from a saved checkpoint
```

### Model (`model.py`)

`TimeSeriesTransformer` takes `(batch, seq_len, in_features)`:
1. Optionally applies **RevIN** (reversible instance normalisation) on channel 0 (power); exogenous channels (calendar features) are left unchanged.
2. `TransformerEncoder`: linear projection → sinusoidal PE → N × post-LayerNorm `TransformerBlock` (self-attention + MLP with GELU).
3. **Last-token pooling** → two parallel `Dense` heads for `mean` and `log_std`, each reshaped to `(batch, horizon, out_features)`.
4. If RevIN is on, outputs are de-normalised back to the same space as the input before returning.

Paper hyperparameters: `d_model=128, num_heads=4, num_layers=4, mlp_dim=256, dropout_rate=0.1, max_len=168` (7-day lookback), `horizon=24`.

### Data pipeline (`sen_data.py`)

Source: CEN "Descarga Generación Real" semicolon-delimited CSV with 24 hour-columns per row.

Pipeline: `load_sen_csv` → `build_site_matrix` (pivot to `(time × site)`, drop sites with > 10 % missing, clip negatives) → `chronological_split` (train < 2024-01-01, val < 2025-01-01, test ≥ 2025-01-01) → `normalize_per_site` (z-score with training stats only).

### Dataset & training (`train.py`)

`SENDataset` is a **lazy sliding-window** sampler over `(S, T)` arrays — avoids materialising ~14 GB of all windows. It supports random `sample_batch` (training) and ordered `sequential_batches` (evaluation). Optional calendar features (`sin/cos` hour + day-of-year) can be appended as extra input channels.

`make_train_step` / `make_eval_step` return `@jax.jit`-compiled closures over the model and optimizer. The training loop uses cosine-decay Adam.

Loss: `masked_gaussian_nll` — only penalises timesteps where raw generation > `EPSILON_MW = 0.1 MW`.

### Checkpointing

`run_experiment.py` saves two files per training run to `--checkpoint_dir` (default `checkpoints/`):
- `<name>_params.pkl` — params dict with all arrays converted to numpy (portable across JAX versions)
- `<name>_history.json` — per-epoch train/val NLL lists

### Evaluation

`metrics.evaluate()` computes MAE, RMSE, and optionally CRPS, all restricted to the active mask (`y_raw > 0.1 MW`). Pass `denorm_mean` / `denorm_std` to get MW-scale metrics matching the paper; omit for normalised-space metrics.
