"""One HTML page to browse and replay every catalogued event.

    python -m hwcat.viewer --workdir hwcat-data

Packs every robustness run (year x area x percentile x window) that has both
an event CSV and a label file into <workdir>/viewer/heatwave_viewer.html. The
page needs no server: each run's footprints are gzipped into the page and
unpacked in the browser only when that run is picked.
"""
import argparse
import base64
import gzip
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from hwcat.detect import land_mask_for
from hwcat import zones
from hwcat.robustness import AREAS_KM2, PERCENTILES, WINDOWS, tag_for
from hwcat.workdir import Workdir

COLUMNS = ["rank", "event_id", "start", "peak", "end", "duration_days",
           "peak_area_km2", "max_anom_C", "peak_tmax_C", "centroid_lat",
           "centroid_lon", "zone"]


def b64gz(raw: bytes) -> str:
    return base64.b64encode(gzip.compress(raw, 9)).decode()


def pack_run(csv: Path, npz: Path) -> dict | None:
    events = pd.read_csv(csv)
    if events.empty:
        return None
    f = np.load(npz)
    t, eid = f["t"].astype("<i2"), f["event_id"].astype("<i4")
    # Day 0 of the label grid, from any event's first cell-day and its start
    # date. Taken from the run itself so a later ERA5 refresh cannot shift it.
    first = events.iloc[0]
    t0 = (pd.Timestamp(first.start)
          - pd.Timedelta(days=int(t[eid == first.event_id].min())))
    # event_id first so the Int32 view in the page starts 4-byte aligned.
    blob = (eid.tobytes() + t.tobytes() + f["i"].astype("<i2").tobytes()
            + f["j"].astype("<i2").tobytes())
    return {"t0": t0.strftime("%Y-%m-%d"), "n": int(t.size),
            "events": events[COLUMNS].round(2).values.tolist(),
            "cells": b64gz(blob)}


def country_grid(lat: np.ndarray, lon: np.ndarray, land: np.ndarray):
    """Country index per cell of this grid, the names it indexes, and
    whether each is European.

    Nearest 0.1 degree cell, as land_mask_for does. A coastal land cell whose
    centre falls just offshore of the Natural Earth outline takes the nearest
    country, so every land cell an event can cover has one."""
    f = np.load(Path(__file__).parent / "data" / "countries_europe_0p1.npz")
    i = np.abs(f["lat"][:, None] - lat[None, :]).argmin(axis=0)
    j = np.abs(f["lon"][:, None] - lon[None, :]).argmin(axis=0)
    grid = f["country"][np.ix_(i, j)]
    _, (ni, nj) = ndimage.distance_transform_edt(grid == 0, return_indices=True)
    grid = np.where(land & (grid == 0), grid[ni, nj], grid)
    # Natural Earth files Cyprus under Asia; it is an EU member, so it counts.
    european = [str(c) == "Europe" or str(n) == "Cyprus"
                for n, c in zip(f["names"], f["continents"])]
    return grid.astype(np.uint8), [str(n) for n in f["names"]], european


def zone_grid(lat: np.ndarray, lon: np.ndarray):
    """Koppen main-group name index per cell (0 = ocean), sampled as
    land_mask_for samples the land mask, and the names it indexes."""
    classes, klat, klon, codes = zones.koppen_grid()
    i = np.abs(klat[:, None] - lat[None, :]).argmin(axis=0)
    j = np.abs(klon[:, None] - (((lon + 180.0) % 360.0) - 180.0)[None, :]).argmin(axis=0)
    names = [""] + list(zones.GROUP_NAMES.values())
    of_class = np.array([names.index(zones.GROUP_NAMES[str(c)[0]])
                         if str(c)[0] in zones.GROUP_NAMES else 0 for c in codes])
    grid = of_class[classes[np.ix_(i, j)]]
    return grid.astype(np.uint8), names


def build(workdir: Workdir) -> Path:
    grid = xr.open_dataset(sorted(workdir.era5.glob(
        f"{workdir.region}_tmax_*.nc"))[0])
    lat, lon = grid["latitude"].values, grid["longitude"].values
    land = land_mask_for(lat, lon)
    countries, names, european = country_grid(lat, lon, land)
    zone_cells, zone_names = zone_grid(lat, lon)

    runs, years = {}, set()
    for csv in sorted(workdir.catalogues.glob(
            f"heatwave_events_{workdir.region}_*.csv")):
        year = int(csv.stem.split("_")[3])
        for area, pct, win in product(AREAS_KM2, PERCENTILES, WINDOWS):
            tag = tag_for(area, pct, win)
            if csv.name != workdir.catalogue_file(year, tag).name:
                continue
            npz = workdir.labels_file(year, tag)
            if npz.exists() and (run := pack_run(csv, npz)):
                runs[f"{year}{tag}"] = run
                years.add(year)

    data = {"lat0": float(lat[0]), "dlat": float(lat[1] - lat[0]),
            "lon0": float(lon[0]), "dlon": float(lon[1] - lon[0]),
            "ny": int(lat.size), "nx": int(lon.size),
            "land": b64gz(np.packbits(land).tobytes()),
            "country": b64gz(countries.tobytes()), "countries": names, "european": european,
            "zone": b64gz(zone_cells.tobytes()), "zones": zone_names,
            "borders": json.loads((Path(__file__).parent / "data"
                                   / "borders_europe_ne50m.json").read_text()),
            "years": sorted(years), "columns": COLUMNS,
            "areas": [int(a / 1000) for a in AREAS_KM2],
            "pcts": [int(round(p * 100)) for p in PERCENTILES],
            "windows": list(WINDOWS), "runs": runs}
    out = workdir.root / "viewer" / f"heatwave_viewer_{workdir.region}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    page = (Path(__file__).with_name("viewer.html").read_text()
            .replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    out.write_text(page)
    print(f"{len(runs)} runs, {len(years)} summers -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="python -m hwcat.viewer")
    p.add_argument("--workdir", type=Path, default=Path("hwcat-data"))
    p.add_argument("--region-name", default="europe")
    a = p.parse_args()
    build(Workdir(a.workdir, region=a.region_name))
