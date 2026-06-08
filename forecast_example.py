"""
Zero-shot hourly energy generation forecast using TimesFM 2.5 (Flax/JAX backend).

Data: Chilean grid real generation (all plants, all technologies)
      aggregated to total MW per hour.
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
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
DATA_PATH = "data/data.csv"
LOOKBACK = 168   # 7 days × 24 h
HORIZON  = 24    # 1 day

# ── 1. Load and reshape ──────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH, sep=";", decimal=",", encoding="utf-8-sig")

hora_cols = [f"Hora {i}" for i in range(1, 25)]
long = df.melt(id_vars=["Fecha"], value_vars=hora_cols,
               var_name="hora_str", value_name="mw")
long["hour"] = long["hora_str"].str.extract(r"(\d+)").astype(int) - 1
long["timestamp"] = pd.to_datetime(long["Fecha"]) + pd.to_timedelta(long["hour"], unit="h")

ts = long.groupby("timestamp")["mw"].sum().sort_index()

# Drop trailing zeros from the last incomplete day (file exported mid-day)
last_valid = ts[ts > 0].index.max()
ts = ts[:last_valid]

# ── 2. Extract context and actuals ───────────────────────────────────────────

context_end   = pd.Timestamp("2026-05-28 23:00")
context_start = context_end - pd.Timedelta(hours=LOOKBACK - 1)
forecast_start = context_end + pd.Timedelta(hours=1)
forecast_end   = forecast_start + pd.Timedelta(hours=HORIZON - 1)

context_ts = ts.loc[context_start:context_end]
actuals_ts = ts.loc[forecast_start:forecast_end]   # partial actuals for comparison

print(f"Context:  {context_ts.index[0]}  →  {context_ts.index[-1]}  ({len(context_ts)} hours)")
print(f"Forecast: {forecast_start}  →  {forecast_end}  ({HORIZON} hours)")
print(f"Actuals available: {len(actuals_ts)} hours")

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
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# ── 4. Forecast ──────────────────────────────────────────────────────────────

point_fc, quant_fc = model.forecast(
    horizon=HORIZON,
    inputs=[context_ts.values.astype(np.float32)],
)
# point_fc: (1, HORIZON)  quant_fc: (1, HORIZON, 10)
point = point_fc[0]          # shape (HORIZON,)
q10   = quant_fc[0, :, 1]   # 10th percentile
q90   = quant_fc[0, :, 9]   # 90th percentile

# ── 5. Plot ──────────────────────────────────────────────────────────────────

forecast_idx = pd.date_range(forecast_start, periods=HORIZON, freq="h")

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(context_ts.index, context_ts.values, color="steelblue",
        linewidth=1.2, label="Context (168 h)")
ax.plot(forecast_idx, point, color="tomato",
        linewidth=1.8, label="Forecast (24 h)")
ax.fill_between(forecast_idx, q10, q90,
                color="tomato", alpha=0.18, label="80% prediction interval")
if len(actuals_ts) > 0:
    ax.plot(actuals_ts.index, actuals_ts.values, color="black",
            linewidth=1.2, linestyle="--", label="Actuals (partial)")

ax.axvline(context_end, color="gray", linestyle=":", linewidth=1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.set_ylabel("Total generation (MW)")
ax.set_title("TimesFM 2.5 – Chilean Grid: 24-hour Generation Forecast")
ax.legend()
fig.tight_layout()

out_path = "data/forecast_output.png"
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved → {out_path}")

# ── 6. Summary stats ─────────────────────────────────────────────────────────

print(f"\nForecast summary (MW):")
print(f"  min  = {point.min():.1f}")
print(f"  mean = {point.mean():.1f}")
print(f"  max  = {point.max():.1f}")
if len(actuals_ts) > 0:
    n = len(actuals_ts)
    mae = np.abs(point[:n] - actuals_ts.values).mean()
    print(f"\nMAE vs. {n} available actuals: {mae:.1f} MW")
