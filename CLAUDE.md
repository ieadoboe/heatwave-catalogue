# Project conventions

## What lives here

`hwcat`, the gridded heat-wave detector and its robustness check, and the
catalogue paper in `paper/`. The NeuralGCM storyline work is a separate
repository (`heatwave-initial-conditions`), which installs `hwcat` at a pinned
tag. A change here reaches it only when that pin moves.

## Outputs

Everything a run writes goes under the working folder (`--workdir`, default
`./hwcat-data`, git-ignored); see `src/hwcat/workdir.py` for the layout. Never
write outputs inside `src/`. The tables and figures the paper uses are copied
into `paper/results/` and `paper/figures/` by `paper/reproduce.sh`, and those
copies are tracked.

## Plotting

Use the project style: `from hwcat.plots import mpl_apply; mpl_apply()`
(importing `hwcat.plots` applies it). Do not redefine rcParams inline; extend
`MPL_CONFIG` in `src/hwcat/plots.py` for a new global setting.

## Tests

Plain scripts, no pytest: `for t in tests/test_*.py; do .venv/bin/python $t; done`.
Each prints `ok` or `ALL CHECKS PASSED`. `tests/test_catalogue.py` compares a
2003 run byte for byte against reference outputs in
`/Users/isaacadoboe/Developer/hwcat-reference`, frozen before the code moved
here; it is the guard against a refactor silently changing results.

## Heat-wave detection over a grid

`hwcat/detect.py` implements Perkins and Alexander CTX90pct on a grid and turns
the resulting cell-days into discrete events by connected-component labelling
in (time, lat, lon).

Order matters and is not obvious: each day is filtered for spatial extent
BEFORE the space-time labelling. A 90th-percentile threshold puts about a tenth
of all cell-days above it by construction, so scattered exceedances are
everywhere; labelling first lets them bridge unrelated blobs through
single-cell chains. In testing that turned a planted 5-day event into a 12-day
one. `tests/test_detect.py` plants known events and asserts the durations, so
run it after touching the detection code.

## Matching events across settings

Event ids do not carry across runs (the labeller numbers blobs in scan order).
Compare runs with `hwcat.match.match_by_overlap`, which pairs events by the
cell-days they share. The older `match_events` pairs by dates and centroids;
it is kept only for the 3-summer sweep behind paper Figure 4, because it can
report a small event that vanished as having merged into a big neighbour. On
August 2003 that inflated the merge count from 3 settings to 10.

## Gotchas

The threshold cache name encodes everything that shapes a threshold: region,
baseline, percentile, window, any left-out year and the month span. The month
span matters because the window makes thresholds near the ends of the season
depend on which months the baseline holds. A new input to the threshold goes
into `catalogue.threshold_name`.

`paper/results/heatwave_events_europe_2026.csv` and paper Figures 1 and 3 were
made on 2026-10-07. The 2026 ERA5 file was updated on 2026-10-10, so a fresh
`paper/figures.py` run redraws Figures 1 and 3 differently until the 2026
catalogue is rebuilt (`paper/reproduce.sh` does both together).
