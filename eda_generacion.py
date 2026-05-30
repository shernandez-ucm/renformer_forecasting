"""
Exploratory Data Analysis — Descarga Generación Real (SEN Chile, 2021-2026)

Wide format: one row = one plant × one day, columns Hora 1..Hora 24 (MW).
Script melts to long format, then analyses generation by technology type.
Figures are saved to docs/eda/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="tab10")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_FILE = "data/Descarga_Generación_Real_2026-05-29_18-57-56.csv"
FIG_DIR = "docs/eda"
os.makedirs(FIG_DIR, exist_ok=True)

HORA_COLS = [f"Hora {h}" for h in range(1, 25)]
RENEWABLE_TYPES = ["Solar", "Eólicas", "Hidroeléctricas", "Bess", "Geotérmica"]
TYPE_COLORS = {
    "Solar": "#f4a61d",
    "Eólicas": "#4e9fd4",
    "Hidroeléctricas": "#2ca02c",
    "Termoeléctricas": "#d62728",
    "Bess": "#9467bd",
    "Geotérmica": "#8c564b",
}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading data...")
df_raw = pd.read_csv(
    DATA_FILE,
    sep=";",
    encoding="utf-8-sig",
    decimal=",",
    dtype={"Año": "int16"},
    parse_dates=["Fecha"],
)
df_raw[HORA_COLS] = df_raw[HORA_COLS].apply(pd.to_numeric, errors="coerce")

print(f"  Rows: {len(df_raw):,}  |  Date range: {df_raw['Fecha'].min().date()} → {df_raw['Fecha'].max().date()}")
print(f"  Tipos: {sorted(df_raw['Tipo'].unique())}")
print(f"  Centrales: {df_raw['Central'].nunique():,}")

# ---------------------------------------------------------------------------
# Melt to long format: (Central, Fecha, hora, MW)
# ---------------------------------------------------------------------------
print("Melting to long format...")
df = df_raw.melt(
    id_vars=["Fecha", "Tipo", "Central", "Subtipo", "Año"],
    value_vars=HORA_COLS,
    var_name="hora_label",
    value_name="MW",
)
df["hora"] = df["hora_label"].str.extract(r"(\d+)").astype("int8")
df["datetime"] = df["Fecha"] + pd.to_timedelta(df["hora"] - 1, unit="h")
df = df.drop(columns="hora_label")


# ---------------------------------------------------------------------------
# 1. Overall dataset summary
# ---------------------------------------------------------------------------
print("\n--- Dataset summary by Tipo ---")
summary = (
    df.groupby("Tipo")["MW"]
    .agg(["count", "mean", "std", "min", "max"])
    .rename(columns={"count": "obs", "mean": "mean_MW", "std": "std_MW", "min": "min_MW", "max": "max_MW"})
    .round(2)
)
summary["centrales"] = df.groupby("Tipo")["Central"].nunique()
print(summary.to_string())


# ---------------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------------
missing_pct = df_raw[HORA_COLS].isnull().mean() * 100
missing_by_tipo = (
    df.groupby("Tipo")["MW"].apply(lambda x: x.isnull().mean() * 100).round(3)
)
print("\n--- Missing hourly values by Tipo (%) ---")
print(missing_by_tipo.to_string())

fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(range(1, 25), missing_pct.values, color="#555")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Missing (%)")
ax.set_title("Missing values per hour across all plants")
ax.set_xticks(range(1, 25))
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/01_missing_by_hour.png", dpi=120)
plt.close()


# ---------------------------------------------------------------------------
# 3. Total daily generation by Tipo (GWh)
# ---------------------------------------------------------------------------
daily_tipo = (
    df.groupby(["Fecha", "Tipo"])["MW"]
    .sum()
    .reset_index()
)
daily_tipo["GWh"] = daily_tipo["MW"] / 1000

fig, ax = plt.subplots(figsize=(14, 5))
for tipo, grp in daily_tipo.groupby("Tipo"):
    ax.plot(grp["Fecha"], grp["GWh"].rolling(7, center=True).mean(),
            label=tipo, color=TYPE_COLORS.get(tipo), lw=1.5, alpha=0.9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax.set_xlabel("Date")
ax.set_ylabel("GWh/day (7-day rolling mean)")
ax.set_title("Daily Generation by Technology (SEN Chile 2021–2026)")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/02_daily_generation_by_tipo.png", dpi=120)
plt.close()


# ---------------------------------------------------------------------------
# 4. Monthly generation mix (stacked area, GWh)
# ---------------------------------------------------------------------------
monthly = (
    df.groupby([pd.Grouper(key="Fecha", freq="ME"), "Tipo"])["MW"]
    .sum()
    .div(1000)
    .reset_index()
    .rename(columns={"Fecha": "Mes", "MW": "GWh"})
)
# BESS has negative values (charging) — exclude from stacked area, plot separately
monthly_wide = monthly.pivot_table(index="Mes", columns="Tipo", values="GWh", fill_value=0)
plot_tipos = [c for c in monthly_wide.columns if c != "Bess"]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
monthly_wide[plot_tipos].plot.area(
    ax=axes[0],
    color=[TYPE_COLORS.get(c, "#aaa") for c in plot_tipos],
    alpha=0.85,
)
axes[0].set_ylabel("GWh")
axes[0].set_title("Monthly Generation Mix (stacked, excl. BESS)")
axes[0].legend(loc="upper left", fontsize=9)

if "Bess" in monthly_wide.columns:
    axes[1].bar(monthly_wide.index, monthly_wide["Bess"], color=TYPE_COLORS["Bess"], width=20, alpha=0.85)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("GWh")
    axes[1].set_title("BESS net generation (negative = net charging)")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
axes[1].set_xlabel("Month")
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/03_monthly_generation_mix.png", dpi=120)
plt.close()


# ---------------------------------------------------------------------------
# 5. Diurnal profiles: Solar and Wind by season
# ---------------------------------------------------------------------------
df["month"] = df["Fecha"].dt.month
df["season"] = df["month"].map({
    12: "Summer", 1: "Summer", 2: "Summer",
    3: "Autumn",  4: "Autumn",  5: "Autumn",
    6: "Winter",  7: "Winter",  8: "Winter",
    9: "Spring", 10: "Spring", 11: "Spring",
})
season_order = ["Summer", "Autumn", "Winter", "Spring"]
season_colors = {"Summer": "#e67e22", "Autumn": "#e74c3c", "Winter": "#2980b9", "Spring": "#27ae60"}

for tipo in ["Solar", "Eólicas"]:
    diurnal = (
        df[df["Tipo"] == tipo]
        .groupby(["hora", "season"])["MW"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    for season in season_order:
        grp = diurnal[diurnal["season"] == season]
        ax.plot(grp["hora"], grp["MW"], label=season, color=season_colors[season], lw=2, marker="o", ms=3)
    ax.set_xticks(range(1, 25))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean MW (fleet total)")
    ax.set_title(f"Diurnal generation profile — {tipo}")
    ax.legend()
    plt.tight_layout()
    fname = tipo.lower().replace("é", "e")
    fig.savefig(f"{FIG_DIR}/04_diurnal_{fname}.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 6. Year-over-year installed capacity proxy (peak generation)
# ---------------------------------------------------------------------------
peak = (
    df.groupby(["Año", "Tipo"])["MW"]
    .quantile(0.99)
    .reset_index()
    .rename(columns={"MW": "P99_MW"})
)

fig, ax = plt.subplots(figsize=(10, 4))
for tipo, grp in peak.groupby("Tipo"):
    ax.plot(grp["Año"], grp["P99_MW"], label=tipo, color=TYPE_COLORS.get(tipo), marker="o", lw=2)
ax.set_xlabel("Year")
ax.set_ylabel("99th-percentile daily total MW")
ax.set_title("Fleet peak generation proxy (P99) by year")
ax.legend()
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/05_peak_proxy_by_year.png", dpi=120)
plt.close()


# ---------------------------------------------------------------------------
# 7. Solar vs Wind correlation (daily totals)
# ---------------------------------------------------------------------------
daily_pivot = daily_tipo.pivot_table(index="Fecha", columns="Tipo", values="GWh", fill_value=np.nan)

if "Solar" in daily_pivot.columns and "Eólicas" in daily_pivot.columns:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(daily_pivot["Solar"], daily_pivot["Eólicas"], alpha=0.25, s=5, color="#555")
    ax.set_xlabel("Solar generation (GWh/day)")
    ax.set_ylabel("Wind generation (GWh/day)")
    ax.set_title("Solar vs Wind daily generation")
    corr = daily_pivot[["Solar", "Eólicas"]].corr().iloc[0, 1]
    ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_solar_vs_wind_scatter.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 8. Distribution of hourly generation per renewable type
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(RENEWABLE_TYPES), figsize=(16, 4), sharey=False)
for ax, tipo in zip(axes, RENEWABLE_TYPES):
    data = df.loc[(df["Tipo"] == tipo) & df["MW"].notna(), "MW"]
    ax.hist(data, bins=60, color=TYPE_COLORS.get(tipo, "#aaa"), edgecolor="none", density=True)
    ax.set_title(tipo, fontsize=10)
    ax.set_xlabel("MW")
    ax.set_ylabel("Density" if ax == axes[0] else "")
fig.suptitle("Hourly generation distribution by renewable type", y=1.02)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/07_hourly_distribution_renewables.png", dpi=120, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# 9. Top 10 solar and wind plants by mean generation
# ---------------------------------------------------------------------------
for tipo in ["Solar", "Eólicas"]:
    top = (
        df[df["Tipo"] == tipo]
        .groupby("Central")["MW"]
        .mean()
        .nlargest(10)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top["Central"][::-1], top["MW"][::-1], color=TYPE_COLORS.get(tipo))
    ax.set_xlabel("Mean hourly MW")
    ax.set_title(f"Top 10 {tipo} plants by mean hourly generation")
    plt.tight_layout()
    fname = tipo.lower().replace("é", "e")
    fig.savefig(f"{FIG_DIR}/08_top10_{fname}.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 10. Heatmap: mean solar generation by month × hour
# ---------------------------------------------------------------------------
solar_heatmap = (
    df[df["Tipo"] == "Solar"]
    .groupby(["month", "hora"])["MW"]
    .mean()
    .unstack("hora")
)
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    solar_heatmap,
    ax=ax,
    cmap="YlOrRd",
    linewidths=0.3,
    cbar_kws={"label": "Mean MW"},
    yticklabels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Month")
ax.set_title("Solar fleet — mean hourly generation by month × hour")
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/09_solar_heatmap_month_hour.png", dpi=120)
plt.close()

print(f"\nAll figures saved to {FIG_DIR}/")
