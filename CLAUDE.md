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

Key dependencies: `jax==0.10.1`, `jaxlib==0.10.1` (CUDA 12), `flax==0.12.7`, `optax==0.2.8`, `numpy`, `pandas`, `scipy`, `timesfm`, `neuralforecast`, `orbax-checkpoint`.

## Running the Code

**Full REnFormer experiment** (real SEN Chile data):
```bash
python run_experiment.py --csv data/Descarga_Generación_Real_2026-05-29_18-57-56.csv
python run_experiment.py --csv <path> --skip_baselines          # faster, skips MLP/LSTM
python run_experiment.py --csv <path> --ablation                # also trains MSE variant
python run_experiment.py --csv <path> --cache_dir data/cache/   # cache preprocessed parquet
python run_experiment.py --csv <path> --resume                  # skip training, load checkpoint
```
Checkpoints are saved to `checkpoints/` by default (override with `--checkpoint_dir`).

**NeuralForecast comparison** (TSMixer, DeepAR, TFT, Autoformer vs REnFormer):
```bash
python compare_models.py --csv <path>
python compare_models.py --csv <path> --models deepar tft       # subset of models
python compare_models.py --csv <path> --zero_shot               # no SEN training
python compare_models.py --csv <path> --max_sites 50            # limit sites for speed
python compare_models.py --cache_dir data/cache/ --csv <path>   # reuse parquet cache
```

**TimesFM zero-shot forecasts** (no training needed):
```bash
python forecast_example.py   # all-generation 24 h forecast (total MW, Flax backend)
python forecast_solar.py     # solar-only 24 h forecast (Flax backend, quantile output)
```

**TimesFM fine-tuning** (PyTorch backend; freezes backbone, trains heads + last N layers):
```bash
python finetune_solar.py     # fine-tune on Chilean solar; compares vs zero-shot baseline
```

**EDA**:
```bash
python eda_generacion.py
```

## Architecture

### Module layout

```
renformer/
  model.py      — TransformerBlock → TransformerEncoder → TimeSeriesTransformer
  train.py      — loss functions, JIT step factories, SENDataset, Orbax checkpointing, training loop
  sen_data.py   — SEN Chile CSV loader, site-matrix builder, chronological split, normalization, parquet cache
  metrics.py    — MAE, RMSE, CRPS (active-mask aware)
  baselines.py  — Persistence, per-site MLP, per-site LSTM (all in JAX/Flax)
  data_utils.py — Monash .tsf parser (legacy; not used by run_experiment.py)
run_experiment.py  — end-to-end paper reproduction script
compare_models.py  — NeuralForecast benchmark (TSMixer, DeepAR, TFT, Autoformer)
forecast_example.py — TimesFM 2.5 zero-shot all-generation forecast
forecast_solar.py   — TimesFM 2.5 zero-shot solar-only forecast
finetune_solar.py   — TimesFM 2.5 fine-tuning (PyTorch) on solar data
```

### Model (`model.py`)

`TimeSeriesTransformer` takes `(batch, seq_len, in_features)`:
1. Optionally applies **RevIN** (reversible instance normalisation) on channel 0 (power); exogenous channels (calendar features) are left unchanged.
2. `TransformerEncoder`: linear projection → sinusoidal PE → N × post-LayerNorm `TransformerBlock` (self-attention + MLP with GELU).
3. **Last-token pooling** → two parallel `Dense` heads for `mean` and `log_std`, each reshaped to `(batch, horizon, out_features)`.
4. If RevIN is on, outputs are de-normalised back to raw units before returning.

Paper hyperparameters: `d_model=128, num_heads=4, num_layers=4, mlp_dim=256, dropout_rate=0.1, max_len=168` (7-day lookback), `horizon=24`.

### Data pipeline (`sen_data.py`)

Source: CEN "Descarga Generación Real" semicolon-delimited CSV with 24 hour-columns per row.

Pipeline: `load_sen_csv` → `build_site_matrix` (pivot to `(time × site)`, filter Solar type, drop sites with > 10% missing, clip negatives) → `chronological_split` (train < 2024-01-01, val < 2025-01-01, test ≥ 2025-01-01) → `normalize_per_site` (z-score with training stats).

`prepare_sen_dataset` wraps the full pipeline. Results can be persisted with `save_prepared_dataset` / `load_prepared_dataset` (parquet files) to avoid re-parsing.

### Dataset & training (`train.py`)

`SENDataset` is a **lazy sliding-window** sampler over `(S, T)` arrays — avoids materialising ~14 GB of all windows. It supports random `sample_batch` (training) and ordered `sequential_batches` (evaluation). Optional calendar features (`sin/cos` hour + day-of-year) can be appended as extra input channels.

`make_train_step` / `make_eval_step` return `@jax.jit`-compiled functions closed over the model and optimizer. The training loop uses cosine-decay Adam with `train_target="raw"` (supervise in MW, not z-score, when RevIN is active).

Loss: `masked_gaussian_nll` — only penalises timesteps where raw generation > `EPSILON_MW = 0.1`.

Checkpoints use Orbax (`save_checkpoint` / `load_checkpoint`) with `max_to_keep=1`.

### TimesFM scripts

`forecast_example.py` and `forecast_solar.py` use the **Flax** backend (`TimesFM_2p5_200M_flax`) for zero-shot inference. `finetune_solar.py` uses the **PyTorch** backend (`TimesFM_2p5_200M_torch`) because only the torch module exposes a differentiable forward pass; it freezes the backbone and trains output heads + the last `UNFREEZE_LAST_N` transformer layers.

### Evaluation

`metrics.evaluate()` computes MAE, RMSE, and optionally CRPS, all restricted to the active mask (`y_raw > 0.1 MW`). Pass `denorm_mean` / `denorm_std` to get MW-scale metrics matching the paper; omit them for normalised-space metrics.
