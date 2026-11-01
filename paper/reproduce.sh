#!/usr/bin/env bash
# Rebuild every number and figure in the catalogue paper.
#   bash paper/reproduce.sh [workdir]
# Needs a Copernicus CDS key in ~/.cdsapirc for the download step. Years
# already on disk, and settings already run by this version, are skipped.
set -euo pipefail
WD=${1:-hwcat-data}
HERE=$(cd "$(dirname "$0")" && pwd)

hwcat download --years 1991-2026 --workdir "$WD"
hwcat download --region 84 -141 41.5 -52 --region-name canada \
  --years 1991-2026 --workdir "$WD/canada"
# Every setting for every summer, leave-one-out, matched by shared cell-days,
# against the reference (50,000 km2, 90th percentile, 15-day window with the
# seasonal cycle removed, 3 days, the region's own countries). Also writes
# the per-summer tables behind paper Figure 4.
hwcat robustness --years 1991-2026 --workdir "$WD"
# Canada, the second region the lead rule is tested on.
hwcat robustness --years 1991-2026 --workdir "$WD/canada" --region-name canada
python -m hwcat.review --workdir "$WD" --validate "$WD/canada"
for y in 2003 2022 2026; do
  cp "$WD/catalogues/heatwave_events_europe_${y}_a50k_p90_w15_sc.csv" \
     "$HERE/results/heatwave_events_europe_$y.csv"
  cp "$WD/robustness/catalogue_sensitivity_europe_$y.csv" "$HERE/results/"
done
cp "$WD/robustness/catalogue_stability_allyears.csv" \
   "$WD"/robustness/review_*.csv "$HERE/results/"
python -m hwcat.review --workdir "$WD/canada" --region-name canada
hwcat consensus --years 1991-2026 --workdir "$WD" --skip-runs \
  --validate "$WD/canada" --validate-region canada
cp "$WD"/robustness/consensus_*.csv "$WD/canada/robustness/consensus_canada_setups.csv" \
   "$HERE/results/"
for f in catalogue_stability_allyears review_checks review_settings; do
  cp "$WD/canada/robustness/$f.csv" "$HERE/results/canada_$f.csv"
done
python "$HERE/edge_share.py" "$WD" europe
python "$HERE/edge_share.py" "$WD/canada" canada
python "$HERE/write_numbers.py"
python "$HERE/figures.py" "$WD"
python "$HERE/detector_figures.py" "$WD"
cd "$HERE" && latexmk -pdf catalogue_paper.tex
