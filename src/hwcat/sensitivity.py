"""How much of one summer's catalogue survives other detection settings.

    python -m hwcat.sensitivity --year 2026 --workdir hwcat-data

Runs the catalogue for every combination of minimum area (25k, 50k, 100k km2)
and percentile (90th, 95th), then matches the top events of the reference
catalogue into each run by overlapping dates and the nearest centroid. This
is the sweep behind paper Figure 4; robustness.py is the exact, all-summers
version.

Output: <workdir>/robustness/catalogue_sensitivity_<region>_<year>.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hwcat import catalogue
from hwcat.match import match_events
from hwcat.workdir import Workdir

AREAS_KM2 = (25_000, 50_000, 100_000)
PERCENTILES = (0.90, 0.95)


def tag_for(area_km2: float, pct: float) -> str:
    return f"_a{int(area_km2 / 1000)}k_p{int(round(pct * 100))}"


def sweep(year: int, workdir: Workdir, top: int = 6) -> pd.DataFrame:
    """Every area x percentile setting for one summer, matched to the reference."""

    dates = ["start", "peak", "end"]
    ref_path = workdir.catalogue_file(year, "")
    if not ref_path.exists():
        catalogue.run(year, workdir, plot=False)
    ref = pd.read_csv(ref_path, parse_dates=dates).head(top)

    rows = []
    for pct in PERCENTILES:
        for area in AREAS_KM2:
            tag = tag_for(area, pct)
            path = workdir.catalogue_file(year, tag)
            if not path.exists():
                print(f"\n=== min-area {area:,} km2, percentile {pct:.0%} ===")
                catalogue.run(year, workdir, percentile=pct, min_area=area,
                              plot=False, tag=tag)
            if not path.exists():
                print(f"skipping {path.name}: not found")
                continue
            run = pd.read_csv(path, parse_dates=dates)
            hit = match_events(ref, run)
            for (i, e), j in zip(ref.iterrows(), hit):
                m = run.loc[j] if j >= 0 else None
                rows.append({
                    "min_area_km2": area, "percentile": pct,
                    "n_events": len(run), "ref_rank": int(e["rank"]),
                    "survives": j >= 0,
                    "rank": int(m["rank"]) if m is not None else np.nan,
                    "start": m.start.date() if m is not None else pd.NaT,
                    "end": m.end.date() if m is not None else pd.NaT,
                    "duration_days": m.duration_days if m is not None else np.nan,
                    "peak_area_km2": m.peak_area_km2 if m is not None else np.nan,
                    "severity_C_km2_day": (m.severity_C_km2_day
                                           if m is not None else np.nan),
                })

    table = pd.DataFrame(rows)
    workdir.robustness.mkdir(parents=True, exist_ok=True)
    out = workdir.robustness / f"catalogue_sensitivity_{workdir.region}_{year}.csv"
    table.to_csv(out, index=False)
    print(f"\n-> {out}\n")

    counts = table.groupby(["percentile", "min_area_km2"]).n_events.first()
    print("events per setting:\n" + counts.to_string() + "\n")
    label = table.apply(
        lambda r: (f"{int(r['rank'])} ({int(r.duration_days)} d)"
                   if r.survives else "-"), axis=1)
    pivot = (table.assign(cell=label)
             .pivot(index="ref_rank", columns=["percentile", "min_area_km2"],
                    values="cell"))
    print("reference rank -> rank (duration) under each setting:\n"
          + pivot.to_string())
    return table


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="3-summer sweep behind paper Figure 4")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--workdir", type=Path, default=Path("hwcat-data"))
    a = p.parse_args()
    sweep(a.year, Workdir(a.workdir))
