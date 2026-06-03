# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

<<<<<<< Updated upstream
**REnFormer** is a global Transformer for probabilistic multi-site solar generation forecasting, trained on Chile's SEN grid data. It outputs Gaussian `(mean, log_std)` predictions evaluated with masked NLL (penalising only non-trivial generation timesteps, raw MW > 0.1).

## Environment

Python 3.12 with a local virtualenv at `env/`. Activate before running anything:

```bash
source env/bin/activate
```

Key dependencies: `jax==0.10.1`, `jaxlib==0.10.1` (CUDA 12), `flax==0.12.7`, `optax==0.2.8`, `numpy`, `pandas`, `scipy`.
=======
**REnFormer** is a Transformer-based probabilistic forecasting model for renewable energy (solar), trained on Chilean grid data (SEN). Implemented in JAX + Flax (linen) with Optax.
>>>>>>> Stashed changes

## Running Experiments

<<<<<<< Updated upstream
**Full experiment** (real SEN Chile data):
```bash
python run_experiment.py --csv docs/Descarga_Generación_Real_2026-05-29_18-57-56.csv
python run_experiment.py --csv <path> --skip_baselines   # faster, skips MLP/LSTM
python run_experiment.py --csv <path> --ablation         # also trains MSE variant
```
=======
### Full paper reproduction
>>>>>>> Stashed changes

**Synthetic smoke test** (no data needed, but note `synthetic_data.py` uses the old `train()` API and is currently broken — see `run_experiment.py` as the authoritative entry point):
```bash
python run_experiment.py --csv data/Descarga_Generación_Real_2026-05-29_18-57-56.csv
```

<<<<<<< Updated upstream
**EDA**:
```bash
python eda_generacion.py
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
run_experiment.py — end-to-end paper reproduction script
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

Pipeline: `load_sen_csv` → `build_site_matrix` (pivot to `(time × site)`, drop sites with > 10% missing, clip negatives) → `chronological_split` (train < 2024-01-01, val < 2025-01-01, test ≥ 2025-01-01) → `normalize_per_site` (z-score with training stats).

### Dataset & training (`train.py`)

`SENDataset` is a **lazy sliding-window** sampler over `(S, T)` arrays — avoids materialising ~14 GB of all windows. It supports random `sample_batch` (training) and ordered `sequential_batches` (evaluation). Optional calendar features (`sin/cos` hour + day-of-year) can be appended as extra input channels.

`make_train_step` / `make_eval_step` return `@jax.jit`-compiled functions closed over the model and optimizer. The training loop uses cosine-decay Adam.

Loss: `masked_gaussian_nll` — only penalises timesteps where raw generation > `EPSILON_MW = 0.1`.

### Evaluation

`metrics.evaluate()` computes MAE, RMSE, and optionally CRPS, all restricted to the active mask (`y_raw > 0.1 MW`). Pass `denorm_mean` / `denorm_std` to get MW-scale metrics matching the paper; omit them for normalised-space metrics.
=======
Key flags:
- `--epochs 50 --batch_size 64 --steps_ep 1000` (paper defaults)
- `--skip_baselines` — skip slow per-site MLP/LSTM (saves hours)
- `--ablation` — also train the REnFormer-MSE variant
- `--max_sites N` — cap per-site baselines for smoke tests

### TimesFM comparisons

```bash
# Zero-shot solar forecast (Flax backend)
python forecast_solar.py

# Fine-tune TimesFM on Chilean solar data (PyTorch backend)
python finetune_solar.py
```

### EDA

```bash
python eda_generacion.py   # saves plots to docs/eda/
```

## Architecture

### `renformer/` package

- **`model.py`** — `PositionalEncoding` → `TransformerBlock` (post-LN) → `TransformerEncoder` → `TimeSeriesTransformer`
- **`train.py`** — loss functions, `SENDataset`, JIT-compiled step factories, training loop
- **`sen_data.py`** — SEN CSV loader, site-matrix builder, chronological split, per-site z-score
- **`metrics.py`** — `mae`, `rmse`, `crps_gaussian` (Gneiting & Raftery closed form), `evaluate`, `print_results_table`
- **`baselines.py`** — Persistence, per-site MLP, per-site LSTM

### Model (`TimeSeriesTransformer`)

Input `(batch, seq_len, in_features)` where channel 0 is the target power series and channels 1+ are optional exogenous (calendar) features.

**RevIN (Reversible Instance Norm):** when `instance_norm=True` (default), each window is centred/scaled by its own mean/std before encoding; the Gaussian head outputs are mapped back to raw MW units. This handles per-window level shifts without global z-score.

Output: `(mean, log_std)` each `(batch, horizon, out_features)`.

### Data pipeline (`SENDataset`)

Stores `(S, T)` site × time arrays in memory and samples `(site, time)` pairs on the fly — avoids materialising all ~14 GB of windows. Key methods:
- `sample_batch(batch_size, rng)` — random sampling for training
- `sequential_batches(batch_size)` — full deterministic pass for evaluation

**Splits (paper Section 5.1):** Train 2021–2023, Val 2024, Test 2025–present.

### Loss

`masked_gaussian_nll` (Eq. 1): diagonal Gaussian NLL computed only on timesteps where raw generation > `EPSILON_MW = 0.1` MW (non-trivial active generation). This avoids trivially-correct nighttime zeros inflating the metric.

### Hyperparameters (paper Table 1)

`d_model=128, num_heads=4, num_layers=4, mlp_dim=256, dropout=0.1, lookback=168h (7 days), horizon=24h`

## Dependencies

Core: `jax`, `flax`, `optax`, `numpy`, `pandas`  
Metrics: `scipy` (optional; pure-numpy fallback for CRPS if absent)  
EDA / plotting: `matplotlib`, `seaborn`  
TimesFM scripts: `timesfm` (flax backend for `forecast_solar.py`; torch backend for `finetune_solar.py`)

No `requirements.txt` exists. Activate the local venv with `source env/bin/activate`.
>>>>>>> Stashed changes
