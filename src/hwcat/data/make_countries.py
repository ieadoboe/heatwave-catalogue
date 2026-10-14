"""Rebuild countries_europe_0p1.npz from Natural Earth 1:50m countries.

    curl -LO https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_50m_admin_0_countries.geojson
    python src/hwcat/data/make_countries.py ne_50m_admin_0_countries.geojson

A 0.1 degree grid over the Europe box, each cell holding the index of the
country containing its centre (0 = none), so a lookup is an array read.
Each country keeps its Natural Earth continent, which the viewer uses to
tell European countries from the rest of the box.
Natural Earth is public domain.
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon

LAT = np.round(np.arange(72.0, 34.95, -0.1), 2)
LON = np.round(np.arange(-25.0, 45.05, 0.1), 2)


def rasterize(geojson: Path) -> tuple[np.ndarray, list[str], list[str]]:
    lon2, lat2 = np.meshgrid(LON, LAT)
    points = np.column_stack([lon2.ravel(), lat2.ravel()])
    grid = np.zeros(points.shape[0], dtype=np.uint8)
    names, continents = [""], [""]
    for f in json.loads(geojson.read_text())["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        hit = np.zeros(points.shape[0], dtype=bool)
        for rings in polys:
            ring = np.asarray(rings[0])
            lo, hi = ring.min(axis=0), ring.max(axis=0)
            box = ((points >= lo) & (points <= hi)).all(axis=1)
            if not box.any():
                continue
            inside = Polygon(ring).contains_points(points[box])
            for hole in rings[1:]:
                inside &= ~Polygon(np.asarray(hole)).contains_points(points[box])
            hit[np.flatnonzero(box)[inside]] = True
        if hit.any():
            names.append(f["properties"]["NAME"])
            continents.append(f["properties"]["CONTINENT"])
            grid[hit] = len(names) - 1
    return grid.reshape(lat2.shape), names, continents


if __name__ == "__main__":
    grid, names, continents = rasterize(Path(sys.argv[1]))
    out = Path(__file__).with_name("countries_europe_0p1.npz")
    np.savez_compressed(out, country=grid, lat=LAT, lon=LON, names=np.array(names),
                        continents=np.array(continents))
    print(f"{len(names) - 1} countries -> {out} ({out.stat().st_size / 1e3:.0f} kB)")
