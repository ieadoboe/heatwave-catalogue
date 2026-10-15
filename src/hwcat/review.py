"""Reviewer checks on the all-summers runs, from outputs already on disk.

    python -m hwcat.review --workdir hwcat-data

Four questions about "one change of setting moves the leader":
  margin    was the leader far ahead of number 2, or a near-tie?
  metric    does ranking by peak exceedance or by duration hold up better
            than ranking by severity?
  stricter  does a top-3 event drop out of the top 6, not just swap places?
  baseline  are summers inside 1991-2020 more or less stable than later ones?

Only single-setting changes from the reference count, as in the headline.
Needs the catalogues and label files that `hwcat robustness` wrote.

Outputs, under <workdir>/robustness/:
  review_summers.csv   one row per summer and metric
  review.pdf
"""

import argparse
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from hwcat.match import footprint, match_by_overlap
from hwcat.robustness import (AREAS_KM2, BASELINE, PERCENTILES, REFERENCE,
                              WINDOWS, changed_parameter, tag_for)
from hwcat.workdir import Workdir

METRICS = {"severity": "severity_C_km2_day",
           "peak exceedance": "max_anom_C",
           "duration": "duration_days"}


def metric_rank(cat: pd.DataFrame, col: str) -> np.ndarray:
    """1 = largest. Ties go to the more severe event, so duration, which
    ties often, still has a single leader."""
    order = cat.sort_values([col, "severity_C_km2_day"], ascending=False,
                            kind="stable").index
    return pd.Series(np.arange(1, len(cat) + 1), index=order) \
        .reindex(cat.index).to_numpy()


def review_year(year: int, workdir: Workdir, top: int = 6) -> list[dict]:
    def load(setting):
        tag = tag_for(*setting)
        path = workdir.catalogue_file(year, tag)
        if not path.exists():
            return None
        return pd.read_csv(path), footprint(workdir.labels_file(year, tag))

    loaded = load(REFERENCE)
    if loaded is None:
        return []
    ref, ref_fp = loaded

    runs = []          # (parameter, run catalogue, match of every reference event)
    for setting in product(AREAS_KM2, PERCENTILES, WINDOWS):
        param = changed_parameter(*setting)
        got = None if param is None else load(setting)
        if got is not None:
            run, run_fp = got
            runs.append((param, run, match_by_overlap(ref, ref_fp, run, run_fp)))

    sev = np.sort(ref.severity_C_km2_day.to_numpy())[::-1]
    margin = 1 - sev[1] / sev[0] if len(sev) > 1 else np.nan

    rows = []
    for metric, col in METRICS.items():
        ref_rank = metric_rank(ref, col)
        leader = int(np.flatnonzero(ref_rank == 1)[0])
        top3 = np.flatnonzero(ref_rank <= 3)
        moved_by, out_by, n_moving = set(), set(), 0
        for param, run, m in runs:
            run_rank = metric_rank(run, col) if len(run) else np.array([])

            def new_rank(i):
                j = m.run_index.iloc[i]
                return run_rank[j] if j >= 0 else np.inf   # vanished

            if new_rank(leader) != 1:
                moved_by.add(param)
                n_moving += 1
            if any(new_rank(i) > top for i in top3):
                out_by.add(param)
        rows.append({
            "year": year, "metric": metric,
            "in_baseline": BASELINE[0] <= year <= BASELINE[1],
            "n_events": len(ref), "leader_margin": margin,
            "n_settings": len(runs),
            "leader_moved": bool(moved_by),
            "leader_moved_by": ",".join(sorted(moved_by)),
            "settings_moving_leader": n_moving,
            "top3_left_top6": bool(out_by),
        })
    return rows


def plot(table: pd.DataFrame, path: Path) -> None:
    from hwcat.plots import mpl_apply
    mpl_apply()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    sev = table[table.metric == "severity"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    colour = np.where(sev.leader_moved, "#B5470F", "#1D4E89")
    ax1.scatter(100 * sev.leader_margin, sev.settings_moving_leader,
                c=colour, s=30, alpha=0.85)
    for _, r in sev[sev.year.isin([2003, 2022, 2026])].iterrows():
        ax1.annotate(str(r.year), (100 * r.leader_margin,
                                   r.settings_moving_leader),
                     xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax1.set_xlabel(r"leader's severity lead over number 2 (\%)")
    ax1.set_ylabel("single changes that move the leader")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title("orange: leader moved in at least one", fontsize=9)

    names = list(METRICS)
    n = table.groupby("metric").year.nunique().reindex(names)
    moved = table.groupby("metric").leader_moved.sum().reindex(names)
    out = table.groupby("metric").top3_left_top6.sum().reindex(names)
    x = np.arange(len(names))
    ax2.bar(x - 0.2, moved, 0.4, color="#B5470F", label="leader moved")
    ax2.bar(x + 0.2, out, 0.4, color="#1D4E89", label="a top-3 event left the top 6")
    ax2.set_xticks(x, names)
    ax2.set_ylabel(f"summers (of {int(n.max())})")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title("ranking by", fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"-> {path}")


def review(workdir: Workdir, top: int = 6) -> pd.DataFrame:
    pattern = re.compile(rf"heatwave_events_{workdir.region}_(\d{{4}})"
                         rf"{re.escape(tag_for(*REFERENCE))}\.csv$")
    years = sorted(int(m.group(1)) for f in workdir.catalogues.iterdir()
                   if (m := pattern.match(f.name)))
    table = pd.DataFrame([row for y in years for row in review_year(y, workdir, top)])
    out = workdir.robustness / "review_summers.csv"
    table.to_csv(out, index=False)
    print(f"-> {out}\n")

    sev = table[table.metric == "severity"]
    print(f"{len(sev)} summers")
    print("\nleader moved / a top-3 event left the top 6, by ranking metric:")
    print(table.groupby("metric")[["leader_moved", "top3_left_top6"]]
          .sum().reindex(list(METRICS)).to_string())
    print("\nseverity leader's lead over number 2, by outcome (median %):")
    print((100 * sev.groupby("leader_moved").leader_margin.median()).round(1)
          .to_string())
    from scipy.stats import spearmanr
    rho, p = spearmanr(sev.leader_margin, sev.settings_moving_leader)
    print(f"Spearman, lead vs single changes moving the leader: "
          f"rho={rho:.2f} p={p:.2g}")
    print("\nseverity leader moved, inside vs after the baseline:")
    print(sev.groupby("in_baseline").leader_moved.agg(["sum", "count"]).to_string())
    plot(table, workdir.robustness / "review.pdf")
    return table


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workdir", type=Path, default=Path("hwcat-data"))
    ap.add_argument("--region-name", default="europe")
    a = ap.parse_args(argv)
    review(Workdir(a.workdir, a.region_name))


if __name__ == "__main__":
    main()
