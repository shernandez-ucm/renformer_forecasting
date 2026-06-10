#!/usr/bin/env bash
# run_scaling_skip.sh — sweep compare_models_skip.py over --max_sites 10 … 291.
#
# Runs REnFormer-Skip (checkpoint) vs TimesFM 2.5 zero-shot for max_sites
# 10, 10+STEP, … and finally 291 (all sites), saving one JSON per run to
# OUT_DIR plus a full console log.
#
# Usage
# -----
#   ./run_scaling_skip.sh
#   STEP=50 ./run_scaling_skip.sh                       # coarser sweep
#   CSV=path/to.csv CHECKPOINT_DIR=checkpoint_skip ./run_scaling_skip.sh
#
# Overridable environment variables (defaults in parentheses):
#   CSV            (data/data.csv)      SEN Chile CSV
#   CACHE_DIR      (data/)              parquet cache directory
#   CHECKPOINT_DIR (checkpoint_skip)    Orbax checkpoint for REnFormer-Skip
#   OUT_DIR        (results_scaling)    where JSON results + logs are written
#   STEP           (10)                 max_sites increment
#   MIN_SITES      (10)                 first max_sites value
#   MAX_SITES      (291)                last max_sites value (always included)

set -uo pipefail

CSV="${CSV:-data/data.csv}"
CACHE_DIR="${CACHE_DIR:-data/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoint_skip}"
OUT_DIR="${OUT_DIR:-results_scaling}"
STEP="${STEP:-10}"
MIN_SITES="${MIN_SITES:-10}"
MAX_SITES="${MAX_SITES:-291}"

cd "$(dirname "$0")"
source env/bin/activate
mkdir -p "$OUT_DIR"

# 10, 10+STEP, … plus the full site count as the final point
sites_list=$(seq "$MIN_SITES" "$STEP" "$MAX_SITES")
if [ "$(echo "$sites_list" | tail -1)" -ne "$MAX_SITES" ]; then
    sites_list="$sites_list $MAX_SITES"
fi

echo "Sweeping max_sites: $(echo $sites_list | tr '\n' ' ')"
echo "Results → $OUT_DIR/"

failed=()
for n in $sites_list; do
    out_json="$OUT_DIR/sites_${n}.json"
    log_file="$OUT_DIR/sites_${n}.log"

    if [ -s "$out_json" ]; then
        echo "── max_sites=$n: $out_json exists, skipping (delete it to re-run)"
        continue
    fi

    echo
    echo "══════════════════════════════════════════════════════════════"
    echo "── max_sites=$n  ($(date '+%F %T'))"
    echo "══════════════════════════════════════════════════════════════"

    if ! python compare_models_skip.py \
            --csv "$CSV" \
            --cache_dir "$CACHE_DIR" \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --max_sites "$n" \
            --out "$out_json" 2>&1 | tee "$log_file"; then
        echo "!! max_sites=$n FAILED (see $log_file)"
        failed+=("$n")
    fi
done

echo
if [ "${#failed[@]}" -gt 0 ]; then
    echo "Done with failures at max_sites: ${failed[*]}"
    exit 1
fi
echo "Sweep complete — JSON results in $OUT_DIR/"
