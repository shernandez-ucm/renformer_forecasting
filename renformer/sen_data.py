"""
SEN Chile data loading and preprocessing pipeline.

Source: CEN "Descarga Generación Real" CSV
Format: semicolon-delimited, comma-decimal, wide (24 hour columns per row/day).
"""
import pandas as pd
import numpy as np

# Chronological split boundaries (paper Section 5.1)
TRAIN_END = "2024-01-01"
VAL_END   = "2025-01-01"
# Test: 2025-01-01 → end of file (~May 2026)

MAX_MISSING_FRAC = 0.10   # drop sites with > 10 % missing hours


def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Cyclical hour-of-day and day-of-year features, shape (T, 4):
        [sin(hour), cos(hour), sin(doy), cos(doy)]

    Solar generation is driven by the diurnal and annual cycles; feeding these
    as exogenous channels lets the model anchor its forecast to clock time
    rather than only to position within the context window.
    """
    hour = index.hour.to_numpy()
    doy  = index.dayofyear.to_numpy()
    return np.stack([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    ], axis=-1).astype(np.float32)


def load_sen_csv(path: str) -> pd.DataFrame:
    """
    Load the wide-format CEN CSV and return a long-format DataFrame with
    columns: [timestamp, Llave, Tipo, generation_mw].
    """
    raw = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",   # handles UTF-8 BOM
        dtype=str,
        low_memory=False,
    )

    id_cols   = ["Llave", "Tipo", "Fecha"]
    hour_cols = [c for c in raw.columns if c.startswith("Hora")]

    df = raw[id_cols + hour_cols].copy()
    df_long = df.melt(
        id_vars=id_cols,
        value_vars=hour_cols,
        var_name="hora_col",
        value_name="generation_mw",
    )

    # Parse numeric — some cells may be "-" or empty
    df_long["generation_mw"] = pd.to_numeric(
        df_long["generation_mw"].str.replace(",", "."), errors="coerce"
    )

    # Build UTC-naive hourly timestamps
    hour_offset = df_long["hora_col"].str.extract(r"(\d+)$").astype(int)[0] - 1
    df_long["timestamp"] = (
        pd.to_datetime(df_long["Fecha"])
        + pd.to_timedelta(hour_offset, unit="h")
    )

    return df_long[["timestamp", "Llave", "Tipo", "generation_mw"]]


def build_site_matrix(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to Solar plants, pivot to (timestamp × site), enforce a complete
    hourly index, and drop sites with > MAX_MISSING_FRAC missing hours.

    Returns a DataFrame with shape (n_hours, n_sites), values in MW ≥ 0.
    """
    solar = df_long[df_long["Tipo"].str.strip() == "Solar"].copy()

    # Aggregate in case of duplicate (timestamp, site) entries
    solar = (
        solar.groupby(["timestamp", "Llave"])["generation_mw"]
        .sum()
        .reset_index()
    )

    pivot = solar.pivot(index="timestamp", columns="Llave", values="generation_mw")
    pivot = pivot.sort_index()

    # Reindex to a gapless hourly grid
    full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq="h")
    pivot = pivot.reindex(full_idx)

    # Drop sites that exceed the missing-hours budget
    missing_frac = pivot.isna().mean()
    pivot = pivot.loc[:, missing_frac <= MAX_MISSING_FRAC]

    # Clip to [0, ∞) — BESS-like negative values shouldn't appear for solar;
    # small negatives from sensor noise are zeroed.
    pivot = pivot.clip(lower=0.0).fillna(0.0)

    return pivot


def chronological_split(pivot: pd.DataFrame):
    """Return (train, val, test) DataFrames using paper's 70/10/20 cut."""
    train = pivot[pivot.index < TRAIN_END]
    val   = pivot[(pivot.index >= TRAIN_END) & (pivot.index < VAL_END)]
    test  = pivot[pivot.index >= VAL_END]
    return train, val, test


def normalize_per_site(train, val, test):
    """
    Z-score each site column using training-split statistics only
    (Dimitriadis et al. 2025 convention).

    Returns: train_norm, val_norm, test_norm, site_mean, site_std
    """
    site_mean = train.mean()
    site_std  = train.std().replace(0, 1.0)   # avoid /0 for constant sites
    train_norm = (train - site_mean) / site_std
    val_norm   = (val   - site_mean) / site_std
    test_norm  = (test  - site_mean) / site_std
    return train_norm, val_norm, test_norm, site_mean, site_std


def prepare_sen_dataset(csv_path: str):
    """
    Full preprocessing pipeline.

    Returns
    -------
    train_raw, val_raw, test_raw : pd.DataFrame  (hours × sites, raw MW)
    train_norm, val_norm, test_norm : pd.DataFrame  (normalized)
    norm_stats : dict with keys 'mean', 'std', 'sites'
    """
    print("Loading SEN CSV …")
    df_long = load_sen_csv(csv_path)

    print("Building site matrix …")
    pivot = build_site_matrix(df_long)
    n_sites = pivot.shape[1]
    n_hours = pivot.shape[0]
    n_obs   = int((pivot > 0).sum().sum())
    intermt = float((pivot <= 0.1).mean().mean()) * 100
    print(f"  Sites : {n_sites}")
    print(f"  Hours : {n_hours:,}  ({n_hours / 8760:.1f} yr)")
    print(f"  Obs > 0 : {n_obs:,}")
    print(f"  Intermittent fraction (≤ 0.1 MW): {intermt:.1f}%")

    train_raw, val_raw, test_raw = chronological_split(pivot)
    print(f"  Train {train_raw.index[0].date()} → {train_raw.index[-1].date()}  "
          f"({len(train_raw):,} h)")
    print(f"  Val   {val_raw.index[0].date()} → {val_raw.index[-1].date()}  "
          f"({len(val_raw):,} h)")
    print(f"  Test  {test_raw.index[0].date()} → {test_raw.index[-1].date()}  "
          f"({len(test_raw):,} h)")

    train_norm, val_norm, test_norm, site_mean, site_std = normalize_per_site(
        train_raw, val_raw, test_raw
    )

    norm_stats = {
        "mean":  site_mean,
        "std":   site_std,
        "sites": pivot.columns.tolist(),
    }
    return (
        (train_raw, train_norm),
        (val_raw,   val_norm),
        (test_raw,  test_norm),
        norm_stats,
    )
