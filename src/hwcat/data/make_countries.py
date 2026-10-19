"""Rebuild countries_<region>_0p1.npz and borders_<region>_ne50m.json from
Natural Earth 1:50m countries.

    curl -LO https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_50m_admin_0_countries.geojson
    python src/hwcat/data/make_countries.py ne_50m_admin_0_countries.geojson canada

A 0.1 degree grid over the region's box, each cell holding the index of the
country containing its centre (0 = none), so a lookup is an array read.
Each country keeps its Natural Earth continent, which the viewer uses to
tell the region's own countries from the rest of the box. The borders file
holds the outer ring of every polygon that reaches into the box, for the
viewer to draw. Natural Earth is public domain.
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon

# North, south, west, east, as passed to hwcat download --region.
BOXES = {"europe": (72.0, 35.0, -25.0, 45.0), "canada": (84.0, 41.5, -141.0, -52.0)}


def grid_for(region: str) -> tuple[np.ndarray, np.ndarray]:
    n, s, w, e = BOXES[region]
    return (np.round(np.arange(n, s - 0.05, -0.1), 2),
            np.round(np.arange(w, e + 0.05, 0.1), 2))


def rasterize(geojson: Path, lat: np.ndarray, lon: np.ndarray
              ) -> tuple[np.ndarray, list[str], list[str]]:
    lon2, lat2 = np.meshgrid(lon, lat)
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


def borders(geojson: Path, region: str) -> list[list[list[float]]]:
    """Outer rings of every polygon with a point in the box, to 0.01 degree."""
    n, s, w, e = BOXES[region]
    rings = []
    for f in json.loads(geojson.read_text())["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for rings_ in polys:
            ring = np.round(np.asarray(rings_[0]), 2)
            inside = ((ring[:, 0] >= w) & (ring[:, 0] <= e)
                      & (ring[:, 1] >= s) & (ring[:, 1] <= n))
            if inside.any():
                rings.append(ring.tolist())
    return rings


if __name__ == "__main__":
    geojson, region = Path(sys.argv[1]), sys.argv[2]
    lat, lon = grid_for(region)
    grid, names, continents = rasterize(geojson, lat, lon)
    out = Path(__file__).with_name(f"countries_{region}_0p1.npz")
    np.savez_compressed(out, country=grid, lat=lat, lon=lon, names=np.array(names),
                        continents=np.array(continents))
    print(f"{len(names) - 1} countries -> {out} ({out.stat().st_size / 1e3:.0f} kB)")
    out = Path(__file__).with_name(f"borders_{region}_ne50m.json")
    out.write_text(json.dumps(borders(geojson, region), separators=(",", ":")))
    print(f"borders -> {out} ({out.stat().st_size / 1e3:.0f} kB)")
