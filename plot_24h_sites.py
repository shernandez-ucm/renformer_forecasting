"""
Plot 24-hour solar generation profiles for 3 sites across 2 seasons.

Chile is in the southern hemisphere, so summer ≈ January and winter ≈ July.
Reads the pre-built parquet cache in data/ (raw MW, time × site) rather than
re-parsing the 2.2M-row CSV. Falls back to parsing data/data.csv if the cache
is absent.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])
from renformer.sen_data import load_sen_csv, build_site_matrix

# Southern-hemisphere seasons: pick a representative clear-ish day per season.
SEASONS = {
    "Summer (Jan)": "2023-01-15",
    "Winter (Jul)": "2023-07-15",
}


def load_matrix(csv: str, cache_dir: str) -> pd.DataFrame:
    """Return a (time × solar-site) MW matrix, preferring the parquet cache."""
    cache = Path(cache_dir) / "train_raw.parquet"
    if cache.exists():
        print(f"Loading cached site matrix → {cache}")
        return pd.read_parquet(cache)
    print(f"Parsing {csv} (no cache found) …")
    return build_site_matrix(load_sen_csv(csv))


def pick_sites(mat: pd.DataFrame, n: int, lowest: bool = False) -> list[str]:
    """Pick n solar sites by mean generation.

    lowest=False → highest-output sites; lowest=True → smallest-output sites,
    excluding essentially-dead sites (mean ≤ 0.1 MW) so the profiles are
    still meaningful.
    """
    means = mat.mean()
    if lowest:
        means = means[means > 0.1].sort_values()
        return means.head(n).index.tolist()
    return means.sort_values(ascending=False).head(n).index.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/data.csv")
    ap.add_argument("--cache_dir", default="data")
    ap.add_argument("--n_sites", type=int, default=3)
    ap.add_argument("--lowest", action="store_true",
                    help="pick lowest-output sites instead of highest")
    ap.add_argument("--out", default="figures/solar_24h_sites_seasons.png")
    args = ap.parse_args()

    mat = load_matrix(args.csv, args.cache_dir)
    sites = pick_sites(mat, args.n_sites, lowest=args.lowest)
    print(f"Selected sites: {sites}")

    fig, axes = plt.subplots(
        1, len(SEASONS), figsize=(6 * len(SEASONS), 4.5), sharey=True
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(sites)))

    for ax, (season, day) in zip(axes, SEASONS.items()):
        day_slice = mat.loc[day]  # 24 hourly rows for that date
        hours = day_slice.index.hour
        for site, c in zip(sites, colors):
            ax.plot(hours, day_slice[site].to_numpy(), marker="o", ms=3,
                    color=c, label=site)
        ax.set_title(f"{season} — {day}")
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Solar generation (MW)")
    axes[-1].legend(title="Site", fontsize=8)

    rank = "lowest-output" if args.lowest else "highest-output"
    fig.suptitle(f"24-hour solar generation — {len(sites)} {rank} sites, "
                 "2 seasons (SEN Chile)")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved figure → {out}")


if __name__ == "__main__":
    main()
