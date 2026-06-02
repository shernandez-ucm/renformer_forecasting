"""
Zero-shot solar generation forecast using TimesFM 2.5 (Flax/JAX backend).

Data: Chilean grid – solar plants only (all PFV / PMGD PFV plants)
      aggregated to total solar MW per hour.
Context: last LOOKBACK hours ending at 2026-05-28 23:00
Forecast: next HORIZON hours (2026-05-29 00:00 – 23:00)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import timesfm

DATA_PATH = "data/Descarga_Generación_Real_2026-05-29_18-57-56.csv"
LOOKBACK = 168   # 7 days × 24 h
HORIZON  = 24    # 1 day

# ── 1. Load, filter to solar, and aggregate by hour ──────────────────────────

df = pd.read_csv(DATA_PATH, sep=";", decimal=",", encoding="utf-8-sig")

solar_df = df[df["Tipo"] == "Solar"]

hora_cols = [f"Hora {i}" for i in range(1, 25)]
long = solar_df.melt(id_vars=["Fecha"], value_vars=hora_cols,
                     var_name="hora_str", value_name="mw")
long["hour"] = long["hora_str"].str.extract(r"(\d+)").astype(int) - 1
long["timestamp"] = pd.to_datetime(long["Fecha"]) + pd.to_timedelta(long["hour"], unit="h")

ts = long.groupby("timestamp")["mw"].sum().sort_index()

# ── 2. Extract context and actuals ───────────────────────────────────────────

context_end    = pd.Timestamp("2026-05-28 23:00")
context_start  = context_end - pd.Timedelta(hours=LOOKBACK - 1)
forecast_start = context_end + pd.Timedelta(hours=1)
forecast_end   = forecast_start + pd.Timedelta(hours=HORIZON - 1)

context_ts = ts.loc[context_start:context_end]
# Actuals for May 29 up to the last non-zero hour (file exported mid-day)
actuals_ts = ts.loc[forecast_start:forecast_end]
actuals_ts = actuals_ts[actuals_ts.index <= actuals_ts[actuals_ts > 0].index.max()]

print(f"Context:  {context_ts.index[0]}  →  {context_ts.index[-1]}  ({len(context_ts)} hours)")
print(f"Forecast: {forecast_start}  →  {forecast_end}  ({HORIZON} hours)")
print(f"Actuals available: {len(actuals_ts)} hours  "
      f"(peak actual = {actuals_ts.max():.1f} MW)")

# ── 3. Load model and compile ────────────────────────────────────────────────

model = timesfm.TimesFM_2p5_200M_flax.from_pretrained(
    "google/timesfm-2.5-200m-flax"
)
model.compile(
    timesfm.ForecastConfig(
        max_context=LOOKBACK,
        max_horizon=HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,   # solar is always ≥ 0
        fix_quantile_crossing=True,
    )
)

# ── 4. Forecast ──────────────────────────────────────────────────────────────

point_fc, quant_fc = model.forecast(
    horizon=HORIZON,
    inputs=[context_ts.values.astype(np.float32)],
)
point = point_fc[0]
q10   = quant_fc[0, :, 1]
q90   = quant_fc[0, :, 9]

# ── 5. Per-horizon predictive statistics ─────────────────────────────────────
# quant_fc[0, :, 0] is the predictive mean; indices 1..9 are q10..q90.
# Estimate a Gaussian-implied std per horizon hour by least-squares fitting the
# nine quantiles against the standard-normal quantile (z) for each level.

z = np.array([   # Φ⁻¹(level) for levels 0.1 … 0.9 (hardcoded, no scipy dependency)
    -1.2815516, -0.8416212, -0.5244005, -0.2533471, 0.0,
     0.2533471,  0.5244005,  0.8416212,  1.2815516,
])

mean_h = quant_fc[0, :, 0]          # predictive mean per horizon hour
q_vals = quant_fc[0, :, 1:10]       # (HORIZON, 9) quantile values
std_h  = np.abs((q_vals @ z) / (z @ z))   # OLS slope vs. z = implied σ per hour

stats = pd.DataFrame(
    {"mean_mw": mean_h, "std_mw": std_h},
    index=pd.date_range(forecast_start, periods=HORIZON, freq="h"),
)
stats.index.name = "timestamp"

print("\nPer-horizon predictive statistics:")
print(stats.round(1).to_string())

stats_path = "data/forecast_solar_stats.csv"
stats.to_csv(stats_path)
print(f"Stats saved → {stats_path}")

# ── 6. Plot ──────────────────────────────────────────────────────────────────

forecast_idx = pd.date_range(forecast_start, periods=HORIZON, freq="h")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(context_ts.index, context_ts.values, color="steelblue",
        linewidth=1.2, label="Context – solar (168 h)")
ax.plot(forecast_idx, point, color="goldenrod",
        linewidth=2.0, label="Forecast (24 h)")
ax.fill_between(forecast_idx, q10, q90,
                color="goldenrod", alpha=0.25, label="80% prediction interval")
ax.plot(actuals_ts.index, actuals_ts.values, color="black",
        linewidth=1.4, linestyle="--", label="Actuals (partial)")

ax.axvline(context_end, color="gray", linestyle=":", linewidth=1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.set_ylabel("Solar generation (MW)")
ax.set_title("TimesFM 2.5 – Chilean Grid: Solar-only 24-hour Forecast")
ax.legend()
fig.tight_layout()

out_path = "data/forecast_solar.png"
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved → {out_path}")

# ── 7. Summary ───────────────────────────────────────────────────────────────

print(f"\nForecast peak:   {point.max():.1f} MW  at hour {point.argmax()}")
print(f"Actual peak:     {actuals_ts.max():.1f} MW  at hour {actuals_ts.idxmax().hour}")

n = len(actuals_ts)
mae = np.abs(point[:n] - actuals_ts.values).mean()
print(f"MAE (first {n} h): {mae:.1f} MW")
