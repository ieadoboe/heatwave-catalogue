"""How stable is the top of the catalogue, summer by summer?

    hwcat robustness --years 1991-2026 --workdir hwcat-data

For every summer, runs the catalogue under every combination of minimum area
(25k, 50k, 100k km2), percentile (90th, 95th) and threshold window (7, 15, 31
days), always with a leave-one-out baseline: a summer inside 1991-2020 is
judged against the other 29 years. The reference setting is 50k km2, 90th
percentile, 15 days. Every reference event is matched into every other
setting by the cell-days it shares (hwcat.match.match_by_overlap), and each
summer is summarized by how many of its top events change rank, which
parameter moved them, merges, and how crowded its leading events were.

Leave-one-out thresholds are about 36 MB each and only one summer uses them,
so each is deleted once that summer's runs finish.

Outputs, under <workdir>/robustness/:
  stability_<region>_<year>.csv               per setting, per top event
  catalogue_stability_allyears.csv            one row per summer
  (merge_settings counts settings in which two reference events became one)
  catalogue_stability_allyears.pdf
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from hwcat.match import footprint, match_by_overlap
from hwcat.sensitivity import tag_for as _area_pct_tag
from hwcat.workdir import Workdir

AREAS_KM2 = (25_000, 50_000, 100_000)
PERCENTILES = (0.90, 0.95)
WINDOWS = (7, 15, 31)
REFERENCE = (50_000, 0.90, 15)
BASELINE = (1991, 2020)
DATES = ["start", "peak", "end"]




def tag_for(area_km2: float, pct: float, window: int) -> str:
    return _area_pct_tag(area_km2, pct) + f"_w{window}"






def per_year_table(workdir: Workdir, year: int) -> Path:
    """Per setting, per top event, for one summer. Named apart from the
    3-summer sweep's catalogue_sensitivity_<region>_<year>.csv, which shares
    the folder."""
    return workdir.robustness / f"stability_{workdir.region}_{year}.csv"


def changed_parameter(area, pct, window) -> str | None:
    """Which single parameter this setting changes from the reference, or
    None when it changes more than one (or none)."""
    diff = [name for name, v, r in zip(("area", "percentile", "window"),
                                       (area, pct, window), REFERENCE)
            if v != r]
    return diff[0] if len(diff) == 1 else None


def crowding(top: pd.DataFrame, max_gap_days: int = 3,
             max_dist_deg: float = 10.0) -> int:
    """Pairs of leading events that sit close in both time and place.

    Close in time: they overlap, or one starts within max_gap_days of the
    other ending. Close in place: centroids within max_dist_deg, longitude
    scaled by latitude as in the matcher.

    ponytail: dates and centroids only, not footprint overlap. Two events on
    either side of a large one can count as close without sharing a cell.
    Use the label grids if this measure ends up carrying the result."""
    n = 0
    for (_, a), (_, b) in combinations(top.iterrows(), 2):
        gap = (max(a.start, b.start) - min(a.end, b.end)).days
        dlat = a.centroid_lat - b.centroid_lat
        dlon = (a.centroid_lon - b.centroid_lon) * np.cos(
            np.radians(0.5 * (a.centroid_lat + b.centroid_lat)))
        if gap <= max_gap_days and np.hypot(dlat, dlon) <= max_dist_deg:
            n += 1
    return n


def run_year(year: int, workdir: Workdir, skip_runs: bool = False) -> None:
    """Every setting for one summer, then drop its leave-one-out thresholds."""
    for area, pct, window in product(AREAS_KM2, PERCENTILES, WINDOWS):
        tag = tag_for(area, pct, window)
        done = (workdir.catalogue_file(year, tag).exists()
                and workdir.labels_file(year, tag).exists())
        if skip_runs or done:
            continue
        subprocess.run(
            [sys.executable, "-m", "hwcat", "detect", "--year", str(year),
             "--min-area", str(area), "--percentile", str(pct),
             "--window", str(window), "--leave-one-out", "--no-plot",
             "--save-labels", "--workdir", str(workdir.root),
             "--region-name", workdir.region, "--tag", tag],
            check=True, stdout=subprocess.DEVNULL)
    for f in workdir.thresholds.glob(
            f"heatwave_threshold_{workdir.region}_*_loo{year}_*.nc"):
        f.unlink()
    print(f"{year}: done", flush=True)


def compare_year(year: int, workdir: Workdir, top: int) -> tuple[pd.DataFrame, dict]:
    """Per-setting matches of the reference top events, and the summary row."""
    def load(area, pct, window):
        path = workdir.catalogue_file(year, tag_for(area, pct, window))
        if not path.exists():
            return None
        run = pd.read_csv(path)
        for col in DATES:
            run[col] = pd.to_datetime(run[col])
        return run

    def fp(area, pct, window):
        return footprint(workdir.labels_file(year, tag_for(area, pct, window)))

    ref_all = load(*REFERENCE)
    if ref_all is None:
        return pd.DataFrame(), {}
    ref = ref_all.head(top)
    ref_fp = fp(*REFERENCE)

    rows, moved, merge_settings, merged_ranks = [], {}, 0, set()
    for area, pct, window in product(AREAS_KM2, PERCENTILES, WINDOWS):
        if (area, pct, window) == REFERENCE:
            continue
        run = load(area, pct, window)
        if run is None:
            continue
        # Match ALL reference events, so a leader that absorbs a lower-ranked
        # event still shows as merged.
        m = match_by_overlap(ref_all, ref_fp, run, fp(area, pct, window))
        top_m = m.iloc[:len(ref)]
        merge_settings += bool(top_m.merged.any())
        merged_ranks.update(int(r) for r in ref["rank"][top_m.merged.to_numpy()])
        param = changed_parameter(area, pct, window)
        for (_, e), j, merged in zip(ref.iterrows(), top_m.run_index,
                                     top_m.merged):
            new_rank = int(run.loc[j, "rank"]) if j >= 0 else np.nan
            did_move = j < 0 or new_rank != e["rank"]
            rows.append({"year": year, "min_area_km2": area, "percentile": pct,
                         "window_days": window, "changed": param or "several",
                         "n_events": len(run), "ref_rank": int(e["rank"]),
                         "rank": new_rank, "moved": did_move,
                         "merged": bool(merged)})
            if did_move:
                moved.setdefault(param or "several", set()).add(int(e["rank"]))

    every = set().union(*moved.values()) if moved else set()
    summary = {
        "year": year,
        "in_baseline": BASELINE[0] <= year <= BASELINE[1],
        "n_events": len(ref_all),
        "n_ref": len(ref),
        "top1_moved": 1 in every,
        "n_moved_any": len(every),
        "n_moved_area": len(moved.get("area", ())),
        "n_moved_percentile": len(moved.get("percentile", ())),
        "n_moved_window": len(moved.get("window", ())),
        "highest_rank_moved": min(every) if every else np.nan,
        "merge_settings": merge_settings,
        "n_top_merged": len(merged_ranks),
        "crowding": crowding(ref),
    }
    return pd.DataFrame(rows), summary


def plot(summary: pd.DataFrame, full: pd.DataFrame, rho: float, p: float,
         top: int, path: Path) -> None:
    """Left: every summer. Right: only summers with `top` events, the ones
    the correlation is computed on."""
    from hwcat.plots import mpl_apply
    mpl_apply()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    x = summary.year.values
    width = 0.27
    for k, (col, label, colour) in enumerate([
            ("n_moved_area", "minimum area", "#1D4E89"),
            ("n_moved_percentile", "percentile", "#B5470F"),
            ("n_moved_window", "window", "#7FB069")]):
        ax1.bar(x + (k - 1) * width, summary[col], width, label=label,
                color=colour)
    ax1.set_xlabel("summer")
    ax1.set_ylabel(f"top-{top} events that change rank")
    ax1.legend(title="changing only the", frameon=False, fontsize=8)

    # Integer on integer: size each point by how many summers share it.
    pts = full.groupby(["crowding", "n_moved_any"]).year.agg(list)
    ax2.scatter([c for c, _ in pts.index], [m for _, m in pts.index],
                s=[25 * len(v) for v in pts], color="#1B1F2A", alpha=0.8)
    for (c, m), years in pts.items():
        named = [y for y in years if y in (2003, 2022, 2026)]
        if named:
            ax2.annotate(", ".join(map(str, named)), (c, m), xytext=(5, 3),
                         textcoords="offset points", fontsize=8)
    ax2.set_xlabel(f"crowded pairs among the top {top}")
    ax2.set_ylabel(f"top-{top} events that change rank (any setting)")
    ax2.set_title(rf"Spearman $\rho$ = {rho:.2f}, $p$ = {p:.2g}"
                  if np.isfinite(rho) else "Spearman undefined", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"-> {path}")


def sweep(years: list[int], workdir: Workdir, top: int = 6, jobs: int = 3,
          skip_runs: bool = False) -> pd.DataFrame:
    """Every setting for every summer, the per-summer stability summary, and
    its figure, all under workdir.robustness."""

    workdir.robustness.mkdir(parents=True, exist_ok=True)
    if not skip_runs:
        # Summers outside the baseline share one threshold per setting. Build
        # those with a single summer first, so parallel runs never race to
        # write the same cache file.
        # ponytail: assumes every shared summer has the same month span (part
        # of the cache key), true for the April-October downloads. Group the
        # shared summers by span if a download ever differs.
        todo = list(years)
        shared = [y for y in todo if not BASELINE[0] <= y <= BASELINE[1]]
        if shared:
            run_year(shared[0], workdir)
            todo.remove(shared[0])
        with ThreadPoolExecutor(jobs) as pool:
            list(pool.map(lambda y: run_year(y, workdir), todo))

    tables, rows = [], []
    for year in years:
        table, summary = compare_year(year, workdir, top)
        if not summary:
            print(f"{year}: no reference catalogue, skipped")
            continue
        table.to_csv(per_year_table(workdir, year),
                     index=False)
        tables.append(table)
        rows.append(summary)

    summary = pd.DataFrame(rows)
    out = workdir.robustness / "catalogue_stability_allyears.csv"
    summary.to_csv(out, index=False)
    print(f"\n-> {out}\n")
    print(summary.to_string(index=False))

    if summary.empty:
        raise RuntimeError("no summers had a reference catalogue")
    # Summers with fewer than `top` events cap both quantities, which would
    # correlate them through the event count; test only full summers.
    full = summary[summary.n_ref == top]
    rho = p = np.nan
    if len(full) > 2 and full.crowding.nunique() > 1 and \
            full.n_moved_any.nunique() > 1:
        from scipy.stats import spearmanr
        rho, p = spearmanr(full.crowding, full.n_moved_any)
    print(f"\nSpearman over {len(full)} summers with {top} events, "
          f"crowding vs events moved: rho={rho:.2f} p={p:.3g}")
    print(f"summers where number 1 moved: {int(summary.top1_moved.sum())} "
          f"of {len(summary)}")
    plot(summary, full, rho, p, top,
         workdir.robustness / "catalogue_stability_allyears.pdf")
    return summary


