#!/usr/bin/env bash
# Rebuild every number and figure in the catalogue paper.
#   bash paper/reproduce.sh [workdir]
# Needs a Copernicus CDS key in ~/.cdsapirc for the download step. Years
# already on disk, and settings already run, are skipped.
set -euo pipefail
WD=${1:-hwcat-data}
HERE=$(cd "$(dirname "$0")" && pwd)

hwcat download --years 1991-2026 --workdir "$WD"
# Every setting for every summer, leave-one-out, matched by shared cell-days,
# against the reference (50,000 km2, 90th percentile, 15-day window with the
# seasonal cycle removed, 3 days, European countries only). Also writes the
# per-summer tables behind paper Figure 4.
hwcat robustness --years 1991-2026 --workdir "$WD"
python -m hwcat.review --workdir "$WD"
for y in 2003 2022 2026; do
  cp "$WD/catalogues/heatwave_events_europe_${y}_a50k_p90_w15_sc.csv" \
     "$HERE/results/heatwave_events_europe_$y.csv"
  cp "$WD/robustness/catalogue_sensitivity_europe_$y.csv" "$HERE/results/"
done
cp "$WD/robustness/catalogue_stability_allyears.csv" \
   "$WD/robustness/review_checks.csv" "$HERE/results/"
python "$HERE/figures.py" "$WD"
cd "$HERE" && latexmk -pdf catalogue_paper.tex
