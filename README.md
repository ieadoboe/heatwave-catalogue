# hwcat: gridded heat-wave catalogues and how robust their rankings are

`hwcat` builds a catalogue of heat waves from ERA5 daily maximum temperature on a
grid, using the CTX90pct index of Perkins and Alexander (2013), and ranks the
events by severity. It then checks how much that ranking depends on the
detector's own settings: the percentile, the minimum area, and the window used
to build each day's threshold. It is the code behind the paper in `paper/`,
which asks whether the events a catalogue calls the worst stay the worst when
those settings change.

## Install

```bash
pip install "hwcat[download] @ git+https://github.com/ieadoboe/heatwave-catalogue@v0.1.0"
```

Downloading needs a Copernicus Climate Data Store key in `~/.cdsapirc`
([how to get one](https://cds.climate.copernicus.eu/how-to-api)). Detection
needs only numpy, pandas, xarray, dask, scipy, netCDF4 and matplotlib: no GPU.

## Quickstart: one past summer

```bash
hwcat download --years 1991-2020 2003       # the baseline, plus the summer to study
hwcat detect --year 2003 --leave-one-out    # the ranked catalogue and a map
hwcat robustness --years 2003               # every setting, and what moves
```

- `download` writes one file per year to `hwcat-data/era5/`. Years already on
  disk are skipped, so an interrupted download resumes.
- `detect` writes the catalogue to `hwcat-data/catalogues/` and a map to
  `hwcat-data/maps/`. `--leave-one-out` judges a summer inside 1991-2020
  against the other 29 years, so it never helps set its own threshold.
- `robustness` reruns the detector under 18 settings (minimum area 25,000,
  50,000 or 100,000 km²; 90th or 95th percentile; 7-, 15- or 31-day window),
  matches every event across them by the grid cells and days they share, and
  writes a per-summer summary to `hwcat-data/robustness/`.

## Another region

```bash
hwcat download --region 50 -130 25 -60 --region-name usa --years 1991-2020 2021
hwcat detect --year 2021 --region-name usa
```

`--region` is north, west, south, east in degrees. The land mask and climate
classes come from a global Köppen-Geiger map that ships with the package.
Country borders are drawn only on maps of Europe.

## The working folder

Everything `hwcat` writes goes under `--workdir` (default `./hwcat-data`):

```
era5/<region>_tmax_<year>.nc
thresholds/heatwave_threshold_<region>_...nc
catalogues/heatwave_events_<region>_<year><tag>.csv
catalogues/labels/heatwave_labels_<region>_<year><tag>.npz
maps/heatwave_<region>_<year><tag>.pdf
robustness/
```

## Reproducing the paper

```bash
bash paper/reproduce.sh
```

It downloads 1991-2026, rebuilds every catalogue and robustness table, copies
the tables into `paper/results/`, redraws the figures and compiles the
manuscript. The manuscript needs the AMS LaTeX class files, which the AMS site
does not serve to scripts; `paper/catalogue_paper.tex` says where to get them.

## Data and method

- ERA5: Hersbach et al. (2020), Q. J. R. Meteorol. Soc. 146, 1999-2049.
- Köppen-Geiger classes: Beck et al. (2023), Sci. Data 10, 724.
- CTX90pct: Perkins and Alexander (2013), J. Climate 26, 4500-4517.
