"""The hwcat command: download, detect, robustness."""
import argparse
import sys
from pathlib import Path

from hwcat.workdir import Workdir

EUROPE = [72.0, -25.0, 35.0, 45.0]


def parse_years(tokens: list[str]) -> list[int]:
    """'1991-2026', '2003' or '2003 2022' into a list of years."""
    years = []
    for t in tokens:
        if "-" in t:
            a, b = (int(x) for x in t.split("-"))
            if b < a:
                raise ValueError(f"year range {t} runs backwards")
            years += list(range(a, b + 1))
        else:
            years.append(int(t))
    return years


def parse_region(values: list[float]) -> list[float]:
    n, w, s, e = values
    if n <= s:
        raise ValueError(f"region north {n} must be above south {s}")
    return [n, w, s, e]


def _common(p):
    p.add_argument("--workdir", type=Path, default=Path("hwcat-data"),
                   help="Folder for downloads and outputs (default ./hwcat-data)")
    p.add_argument("--region-name", default="europe",
                   help="Prefix for file names (default europe)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hwcat", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="ERA5 daily maxima for a region")
    _common(d)
    d.add_argument("--years", nargs="+", required=True)
    d.add_argument("--region", nargs=4, type=float, default=EUROPE,
                   metavar=("N", "W", "S", "E"))
    d.add_argument("--overwrite", action="store_true")

    t = sub.add_parser("detect", help="One catalogue for one year")
    _common(t)
    t.add_argument("--year", type=int, required=True)
    t.add_argument("--percentile", type=float, default=0.90)
    t.add_argument("--min-days", type=int, default=3)
    t.add_argument("--min-area", type=float, default=50_000.0)
    t.add_argument("--window", type=int, default=15)
    t.add_argument("--leave-one-out", action="store_true")
    t.add_argument("--save-labels", action="store_true")
    t.add_argument("--no-plot", action="store_true")
    t.add_argument("--tag", default=None)

    r = sub.add_parser("robustness", help="Every setting for every summer")
    _common(r)
    r.add_argument("--years", nargs="+", required=True)
    r.add_argument("--top", type=int, default=6)
    r.add_argument("--jobs", type=int, default=3)

    a = parser.parse_args(argv)
    wd = Workdir(a.workdir, region=a.region_name)
    try:
        if a.command == "download":
            from hwcat.download import download
            download(parse_years(a.years), wd, parse_region(a.region),
                     overwrite=a.overwrite)
        elif a.command == "detect":
            from hwcat.catalogue import run
            run(a.year, wd, percentile=a.percentile, min_days=a.min_days,
                min_area=a.min_area, window=a.window,
                leave_one_out=a.leave_one_out, save_labels=a.save_labels,
                plot=not a.no_plot, tag=a.tag)
        else:
            from hwcat.robustness import sweep
            sweep(parse_years(a.years), wd, top=a.top, jobs=a.jobs)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"hwcat {a.command}: {e}")
