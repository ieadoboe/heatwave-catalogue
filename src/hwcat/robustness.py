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

With --one-at-a-time it instead sweeps each setting across its full range
(ONE_AT_A_TIME: area to 625k km2, window 7 to 31 days, minimum days 3 to 7)
with the others at the reference.

Outputs, under <workdir>/robustness/:
  stability_<region>_<year>.csv               per setting, per top event
  catalogue_stability_allyears.csv            one row per summer
  (merge_settings counts settings in which two reference events became one)
  catalogue_stability_allyears.pdf
  stability_one_at_a_time.csv / .pdf          --one-at-a-time
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from hwcat.match import footprint, match_by_overlap
from hwcat.workdir import Workdir

AREAS_KM2 = (25_000, 50_000, 100_000)
PERCENTILES = (0.90, 0.95)
WINDOWS = (7, 15, 31)
REFERENCE = (50_000, 0.90, 15)
BASELINE = (1991, 2020)
DATES = ["start", "peak", "end"]
GRID = [(a, p, w, 3) for a, p, w in product(AREAS_KM2, PERCENTILES, WINDOWS)]

# One setting at a time from the reference, each across its full range:
# which value first moves the leader, not only whether some value does.
ONE_AT_A_TIME = {"min_area_km2": (25_000, 50_000, 100_000, 625_000),
                 "window_days": (7, 11, 15, 21, 31),
                 "min_days": (3, 4, 5, 6, 7)}


def one_at_a_time_settings() -> list[tuple[str, float, tuple]]:
    """(parameter, value, setting) with setting = (area, pct, window, days)."""
    ref = dict(zip(("min_area_km2", "percentile", "window_days", "min_days"),
                   (*REFERENCE, 3)))
    out = []
    for param, values in ONE_AT_A_TIME.items():
        for v in values:
            s = {**ref, param: v}
            out.append((param, v, (s["min_area_km2"], s["percentile"],
                                   s["window_days"], s["min_days"])))
    return out


def tag_for(area_km2: float, pct: float, window: int, min_days: int = 3) -> str:
    """The default three days adds nothing, so the grid's files keep their names."""
    tag = f"_a{int(area_km2 / 1000)}k_p{int(round(pct * 100))}_w{window}"
    return tag if min_days == 3 else f"{tag}_d{min_days}"






def per_year_table(workdir: Workdir, year: int) -> Path:
    """Per setting, per top event, for one summer. Named apart from
    paper_table's catalogue_sensitivity_<region>_<year>.csv, which shares the
    folder."""
    return workdir.robustness / f"stability_{workdir.region}_{year}.csv"


def paper_table(year: int, workdir: Workdir, top: int = 6,
                window: int = 15) -> pd.DataFrame:
    """The table behind paper Figure 4: the reference top events under each
    minimum area and percentile at one window, matched by shared cell-days,
    leave-one-out like every run here. Two reference events matched to the
    same new event share a new rank, which is how the figure marks a merge.

    Written to <workdir>/robustness/catalogue_sensitivity_<region>_<year>.csv."""
    def load(area, pct):
        tag = tag_for(area, pct, window)
        run = pd.read_csv(workdir.catalogue_file(year, tag))
        for col in DATES:
            run[col] = pd.to_datetime(run[col])
        return run, footprint(workdir.labels_file(year, tag))

    ref_all, ref_fp = load(*REFERENCE[:2])
    ref = ref_all.head(top)
    rows = []
    for pct in PERCENTILES:
        for area in AREAS_KM2:
            run, run_fp = load(area, pct)
            m = match_by_overlap(ref_all, ref_fp, run, run_fp).iloc[:len(ref)]
            for (_, e), j in zip(ref.iterrows(), m.run_index):
                hit = run.iloc[j] if j >= 0 else None
                rows.append({
                    "min_area_km2": area, "percentile": pct, "n_events": len(run),
                    "ref_rank": int(e["rank"]), "survives": j >= 0,
                    "rank": int(hit["rank"]) if hit is not None else np.nan,
                    "start": hit.start.date() if hit is not None else pd.NaT,
                    "end": hit.end.date() if hit is not None else pd.NaT,
                    "duration_days": hit.duration_days if hit is not None else np.nan,
                    "peak_area_km2": hit.peak_area_km2 if hit is not None else np.nan,
                    "severity_C_km2_day": (hit.severity_C_km2_day
                                           if hit is not None else np.nan),
                })
    table = pd.DataFrame(rows)
    workdir.robustness.mkdir(parents=True, exist_ok=True)
    table.to_csv(workdir.robustness /
                 f"catalogue_sensitivity_{workdir.region}_{year}.csv", index=False)
    return table


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


def run_year(year: int, workdir: Workdir, skip_runs: bool = False,
             settings: list[tuple] = GRID) -> None:
    """Every setting (area, pct, window, days) for one summer, then drop its
    leave-one-out thresholds."""
    for area, pct, window, days in settings:
        tag = tag_for(area, pct, window, days)
        done = (workdir.catalogue_file(year, tag).exists()
                and workdir.labels_file(year, tag).exists())
        if skip_runs or done:
            continue
        subprocess.run(
            [sys.executable, "-m", "hwcat", "detect", "--year", str(year),
             "--min-area", str(area), "--percentile", str(pct),
             "--window", str(window), "--min-days", str(days),
             "--leave-one-out", "--no-plot",
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


def run_all(years: list[int], workdir: Workdir, jobs: int,
            settings: list[tuple]) -> None:
    # Summers outside the baseline share one threshold per setting. Build
    # those with a single summer first, so parallel runs never race to
    # write the same cache file.
    # ponytail: assumes every shared summer has the same month span (part
    # of the cache key), true for the April-October downloads. Group the
    # shared summers by span if a download ever differs.
    todo = list(years)
    shared = [y for y in todo if not BASELINE[0] <= y <= BASELINE[1]]
    if shared:
        run_year(shared[0], workdir, settings=settings)
        todo.remove(shared[0])
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(lambda y: run_year(y, workdir, settings=settings), todo))


def compare_one_at_a_time(year: int, workdir: Workdir, top: int) -> pd.DataFrame:
    """One row per (parameter, value): what that single change does to the
    reference top events."""
    def load(setting):
        tag = tag_for(*setting)
        if not workdir.catalogue_file(year, tag).exists():
            return None
        return (pd.read_csv(workdir.catalogue_file(year, tag)),
                footprint(workdir.labels_file(year, tag)))

    loaded = load((*REFERENCE, 3))
    if loaded is None or loaded[0].empty:
        return pd.DataFrame()
    ref_all, ref_fp = loaded
    ref = ref_all.head(top)
    rows = []
    for param, value, setting in one_at_a_time_settings():
        got = load(setting)
        if got is None:
            continue
        run, run_fp = got
        m = match_by_overlap(ref_all, ref_fp, run, run_fp).iloc[:len(ref)]
        new = np.array([run.loc[j, "rank"] if j >= 0 else np.inf
                        for j in m.run_index])
        old = ref["rank"].to_numpy()
        rows.append({"year": year, "parameter": param, "value": value,
                     "is_reference": setting == (*REFERENCE, 3),
                     "n_events": len(run),
                     "leader_moved": bool(new[0] != 1),
                     "n_top_moved": int((new != old).sum()),
                     "top3_left_top": bool((new[:3] > top).any()),
                     "top_merged": bool(m.merged.any())})
    return pd.DataFrame(rows)


def plot_one_at_a_time(table: pd.DataFrame, top: int, path: Path) -> None:
    from hwcat.plots import mpl_apply
    mpl_apply()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    n = table.year.nunique()
    by = table.groupby(["parameter", "value"])[
        ["leader_moved", "top3_left_top", "top_merged"]].sum()
    labels = {"min_area_km2": r"minimum area (1000 km$^2$)",
              "window_days": "window (days)", "min_days": "minimum days"}
    ref = {"min_area_km2": REFERENCE[0], "window_days": REFERENCE[2],
           "min_days": 3}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, param in zip(axes, ONE_AT_A_TIME):
        d = by.loc[param]
        x = np.arange(len(d))
        ax.bar(x - 0.27, d.leader_moved, 0.27, color="#B5470F",
               label="leader moved")
        ax.bar(x, d.top3_left_top, 0.27, color="#1D4E89",
               label=f"a top-3 event left the top {top}")
        ax.bar(x + 0.27, d.top_merged, 0.27, color="#7FB069",
               label=f"two top-{top} events merged")
        ticks = [f"{v / 1000:g}" if param == "min_area_km2" else f"{v:g}"
                 for v in d.index]
        # Under LaTeX text, set_fontweight is ignored: bold has to be markup.
        bold = (lambda s: rf"\textbf{{{s}}}") if plt.rcParams["text.usetex"] \
            else (lambda s: s)
        i_ref = list(d.index).index(ref[param])
        ticks[i_ref] = bold(ticks[i_ref])
        ax.set_xticks(x, ticks)
        if not plt.rcParams["text.usetex"]:
            ax.get_xticklabels()[i_ref].set_fontweight("bold")
        ax.set_xlabel(labels[param])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel(f"summers (of {n})")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("One setting changed at a time; bold value is the reference",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"-> {path}")


def one_at_a_time(years: list[int], workdir: Workdir, top: int = 6,
                  jobs: int = 3, skip_runs: bool = False) -> pd.DataFrame:
    """Each setting across its full range, the others at the reference."""
    workdir.robustness.mkdir(parents=True, exist_ok=True)
    if not skip_runs:
        run_all(years, workdir, jobs, [s for _, _, s in one_at_a_time_settings()])
    table = pd.concat([compare_one_at_a_time(y, workdir, top) for y in years],
                      ignore_index=True)
    out = workdir.robustness / "stability_one_at_a_time.csv"
    table.to_csv(out, index=False)
    print(f"-> {out}\n")
    print(table.groupby(["parameter", "value"])[
        ["leader_moved", "top3_left_top", "top_merged"]].sum().to_string())
    plot_one_at_a_time(table, top,
                       workdir.robustness / "stability_one_at_a_time.pdf")
    return table


def sweep(years: list[int], workdir: Workdir, top: int = 6, jobs: int = 3,
          skip_runs: bool = False) -> pd.DataFrame:
    """Every setting for every summer, the per-summer stability summary, and
    its figure, all under workdir.robustness."""

    workdir.robustness.mkdir(parents=True, exist_ok=True)
    if not skip_runs:
        run_all(years, workdir, jobs, GRID)

    tables, rows = [], []
    for year in years:
        table, summary = compare_year(year, workdir, top)
        if not summary:
            print(f"{year}: no reference catalogue, skipped")
            continue
        paper_table(year, workdir, top)
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


