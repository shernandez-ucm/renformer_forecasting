# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**renformer** is a Transformer-based probabilistic time series forecasting model for renewable energy (solar + wind), implemented in JAX + Flax (linen) with Optax for optimization.

## Running the Code

Run the end-to-end synthetic-data smoke test (no real data needed):

```bash
python synthetic_data.py
```

To use real `.tsf` data:

```python
from renformer.data_utils import convert_tsf_to_dataframe
df, frequency, horizon, _, _ = convert_tsf_to_dataframe("data/solar_4_seconds_dataset.tsf")
```

No `requirements.txt` exists yet. Core dependencies: `jax`, `flax`, `optax`, `numpy`, `pandas`.

## Architecture

```
renformer/
  model.py       — PositionalEncoding → TransformerBlock → TransformerEncoder → TimeSeriesTransformer
  data_utils.py  — .tsf parser, train/val split, lag-window batch builders
  train.py       — Gaussian NLL loss, JIT-compiled train_step, training loop
synthetic_data.py — end-to-end example with random arrays
data/            — .tsf files (solar and wind at 4-second resolution)
```

### Model

`TimeSeriesTransformer` (top-level Flax module):
1. Runs `TransformerEncoder`: linear projection → sinusoidal PE → N × `TransformerBlock`
2. Pools the **last encoder token** as the sequence summary
3. Projects to `(horizon × out_features)` for both `mean` and `log_std` → probabilistic Gaussian output

Input shape: `(batch, seq_len, in_features)`. Missing values handled via a value mask passed into `model.apply`.

### Data utilities

- `convert_tsf_to_dataframe` — parses Monash `.tsf` format into a Pandas DataFrame
- `train_test_split(data, split_fraction, feature_keys)` — z-scores using train-split statistics only
- `create_batch` / `create_batch_multistep` — sliding-window (X, y) pairs via `pd.concat` + `.shift`

### Training

- Loss: diagonal Gaussian NLL (`gaussian_nll`)
- Optimizer: Adam via Optax
- `train_step` is `@jax.jit`-compiled; takes a `rng` key for dropout

## Known Issues

- `model.py:139`: `return mean, log_std(x.shape[0], ...)` incorrectly calls `log_std` as a function — should be `return mean, log_std` (already reshaped on lines 136–138).
- `train.py`: `create_batch` is called as `create_batch(X_train, Y_train, batch_size)` but the function signature is `create_batch(data, lag, future)` — signatures are incompatible; training loop needs a proper mini-batch iterator.
- `TransformerEncoder` passes `self.input_dim` to `TransformerBlock` but has no `input_dim` field defined.
