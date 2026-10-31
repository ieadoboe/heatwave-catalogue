"""Write paper/results/edge_share_<region>.csv: the share of each reference
leader's area-days on cells where the domain cuts events off (the outer ring
of the box and cells next to excluded land, as detect.cut_cells defines
them). touches_edge flags any contact; this says how much.

    python paper/edge_share.py hwcat-data europe
    python paper/edge_share.py hwcat-data/canada canada
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hwcat import detect as D
from hwcat.catalogue import domain_mask
from hwcat.robustness import REFERENCE, tag_of
from hwcat.workdir import Workdir

HERE = Path(__file__).resolve().parent


def main(root: str, region: str) -> None:
    wd = Workdir(Path(root), region=region)
    tag = tag_of(REFERENCE)
    rows = []
    for year in range(1991, 2027):
        tx = D.open_daily_tmax([wd.era5_file(year)])
        lat, lon = tx.latitude.values, tx.longitude.values
        land = D.land_mask_for(lat, lon)
        cut = D.cut_cells(land, domain_mask(lat, lon, land, region))
        area = D.cell_area_km2(lat, lon)
        cat = pd.read_csv(wd.catalogue_file(year, tag))
        z = np.load(wd.labels_file(year, tag))
        k = z["event_id"] == cat.iloc[0].event_id
        i, j = z["i"][k], z["j"][k]
        a = area[i, j]
        rows.append({"year": year,
                     "share_pct": round(100 * a[cut[i, j]].sum() / a.sum(), 2)})
    out = HERE / "results" / f"edge_share_{region}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
