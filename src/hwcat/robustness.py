"""How stable is the top of the catalogue, summer by summer?

    hwcat robustness --years 1991-2026 --workdir hwcat-data

The reference setting is 50,000 km2, 90th percentile, a 15-day window with
the seasonal cycle removed (Brunner and Voigt 2024), 3 days, 10-neighbour
labelling, the region's own countries, and a leave-one-out baseline: a
summer inside 1991-2020 is judged against the other 29 years.

Every summer is rerun with one setting changed at a time across the range
used in the literature (SWEEPS), the windows again with raw pooling
(window_raw, to show the bias the reference avoids), and a percentile x
minimum-duration grid, the two settings that move the leader most. Every
reference event is matched into every run by the cell-days it shares
(hwcat.match.match_by_overlap, MIN_OVERLAP), and the leader's fate is one of
held, absorbed, replaced, split or vanished (hwcat.match.leader_outcome).

Each run is judged under three rankings: severity above its own bar (the
catalogue's), severity above the fixed reference bar (so a stricter
percentile changes which cells count, not how they are scored), and
severity normalized by the local spread of daily maxima.

Outputs, under <workdir>/robustness/:
  stability_<region>_<year>.csv          per run and ranking
  catalogue_stability_allyears.csv       one row per summer
  catalogue_stability_allyears.pdf
  catalogue_sensitivity_<region>_<year>.csv   paper Figure 4, PAPER_YEARS
"""

import subprocess
import sys
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from hwcat.match import MIN_OVERLAP, footprint, leader_outcome, match_by_overlap
from hwcat.workdir import Workdir

Setting = namedtuple("Setting", "area pct window days cycle conn")
REFERENCE = Setting(50_000, 0.90, 15, 3, True, 10)
BASELINE = (1991, 2020)
DATES = ["start", "peak", "end"]

# One setting at a time from the reference. Every value lies within the
# range published for that setting (the published-range standard, decided
# 28 October 2026: interior points allowed, nothing outside):
#   percentile   90-95: 90th in Perkins and Alexander 2013 and Zhang et
#                al. 2025; 90th and 95th in Mandonnet et al. 2026.
#   window       5-31 days: 5 in the ETCCDI indices (TX90p, WSDI); 15 in
#                Zhang et al. 2025; 31 in Brunner and Voigt 2024's example.
#   min. days    3-6: 3 in CTX90pct; 3 to 6 in Mandonnet et al. 2026.
#   min. area    0 to 1,000,000 km2: both ends in Zhang et al. 2025;
#                625,000 in Lhotka and Kysely 2022.
#   neighbours   10, 18, 26: 10 and 26 in Zhang et al. 2025; 18 and 26 in
#                Mandonnet et al. 2026.
SWEEPS = {"min_area_km2": ("area", (25_000, 50_000, 100_000, 250_000, 625_000)),
          "percentile": ("pct", (0.90, 0.925, 0.95)),
          "window_days": ("window", (7, 11, 15, 21, 31)),
          "min_days": ("days", (3, 4, 5, 6)),
          "connectivity": ("conn", (10, 18, 26))}
RAW_WINDOWS = (7, 15, 31)
GRID = list(product(SWEEPS["percentile"][1], SWEEPS["min_days"][1]))
# Paper Figure 4: area x percentile at the reference window, three summers.
PAPER_YEARS = (2003, 2022, 2026)
PAPER_AREAS = (25_000, 50_000, 100_000)
PAPER_PCTS = (0.90, 0.95)
# The viewer's selectors.
AREAS_KM2, PERCENTILES, WINDOWS = PAPER_AREAS, PAPER_PCTS, SWEEPS["window_days"][1]

METRICS = {"severity": "severity_C_km2_day",
           "severity_ref": "severity_ref_C_km2_day",
           "severity_norm": "severity_norm_km2_day"}
SINGLE = tuple(SWEEPS)            # families that change one setting


def settings(year: int | None = None) -> list[tuple[str, float, Setting]]:
    """(family, value, setting) for every run of one summer, reference
    included once per sweep. The paper's area x percentile combinations are
    added for PAPER_YEARS."""
    out = []
    for family, (field, values) in SWEEPS.items():
        out += [(family, v, REFERENCE._replace(**{field: v})) for v in values]
    out += [("window_raw", w, REFERENCE._replace(window=w, cycle=False))
            for w in RAW_WINDOWS]
    out += [("grid", f"p{pct:g}_d{d}", REFERENCE._replace(pct=pct, days=d))
            for pct, d in GRID]
    if year in PAPER_YEARS:
        out += [("paper", f"a{a // 1000}k_p{p:g}", REFERENCE._replace(area=a, pct=p))
                for a, p in product(PAPER_AREAS, PAPER_PCTS)]
    return out


def tag_for(area: float, pct: float, window: int, min_days: int = 3,
            remove_cycle: bool = False, connectivity: int = 10) -> str:
    """File tag of one setting. Raw pooling, 3 days and 10 neighbours add
    nothing, so the earlier grid's files keep their names."""
    tag = f"_a{area / 1000:g}k_p{round(pct * 100, 1):g}_w{window}"
    if min_days != 3:
        tag += f"_d{min_days}"
    if remove_cycle:
        tag += "_sc"
    if connectivity != 10:
        tag += f"_c{connectivity}"
    return tag


def tag_of(s: Setting) -> str:
    return tag_for(s.area, s.pct, s.window, s.days, s.cycle, s.conn)


def is_current(csv: Path) -> bool:
    """A catalogue written by this version: it carries the rescored
    severities and the edge flag, so an older run on disk is redone."""
    if not csv.exists():
        return False
    header = csv.open().readline()
    return "touches_edge" in header or header.strip() == "rank,start,peak,end"


def per_year_table(workdir: Workdir, year: int) -> Path:
    return workdir.robustness / f"stability_{workdir.region}_{year}.csv"


def run_year(year: int, workdir: Workdir, skip_runs: bool = False,
             todo: list[Setting] | None = None) -> None:
    """Every setting (default: settings(year)) for one summer, then drop its
    leave-one-out thresholds."""
    todo = set(todo) if todo is not None else {s for _, _, s in settings(year)}
    for s in sorted(todo, key=tag_of):
        tag = tag_of(s)
        if skip_runs or (is_current(workdir.catalogue_file(year, tag))
                         and workdir.labels_file(year, tag).exists()):
            continue
        subprocess.run(
            [sys.executable, "-m", "hwcat", "detect", "--year", str(year),
             "--min-area", str(s.area), "--percentile", str(s.pct),
             "--window", str(s.window), "--min-days", str(s.days),
             "--connectivity", str(s.conn), "--leave-one-out", "--no-plot",
             "--rescore", *([] if s.cycle else ["--raw-window"]),
             "--save-labels", "--workdir", str(workdir.root),
             "--region-name", workdir.region, "--tag", tag],
            check=True, stdout=subprocess.DEVNULL)
    for f in workdir.thresholds.glob(
            f"heatwave_threshold_{workdir.region}_*_loo{year}_*.nc"):
        f.unlink()
    print(f"{year}: done", flush=True)


def run_all(years: list[int], workdir: Workdir, jobs: int,
            todo: list[Setting] | None = None) -> None:
    # Summers outside the baseline share one threshold per setting. Build
    # those with a single summer first, so parallel runs never race to
    # write the same cache file.
    # ponytail: assumes every shared summer has the same month span (part
    # of the cache key), true for the April-October downloads.
    setups, todo = todo, list(years)
    shared = [y for y in todo if not BASELINE[0] <= y <= BASELINE[1]]
    if shared:
        run_year(shared[0], workdir, todo=setups)
        todo.remove(shared[0])
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(lambda y: run_year(y, workdir, todo=setups), todo))


def metric_rank(cat: pd.DataFrame, col: str) -> np.ndarray:
    """1 = largest. Ties go to the more severe event."""
    order = cat.sort_values([col, "severity_C_km2_day"], ascending=False,
                            kind="stable").index
    return pd.Series(np.arange(1, len(cat) + 1), index=order) \
        .reindex(cat.index).to_numpy()


def crowding(top: pd.DataFrame, max_gap_days: int = 3,
             max_dist_deg: float = 10.0) -> int:
    """Pairs of leading events that sit close in both time and place: they
    overlap or one starts within max_gap_days of the other ending, and
    their centroids lie within max_dist_deg (longitude scaled by latitude).

    ponytail: dates and centroids only, not footprint overlap."""
    n = 0
    for (_, a), (_, b) in combinations(top.iterrows(), 2):
        gap = (max(a.start, b.start) - min(a.end, b.end)).days
        dlat = a.centroid_lat - b.centroid_lat
        dlon = (a.centroid_lon - b.centroid_lon) * np.cos(
            np.radians(0.5 * (a.centroid_lat + b.centroid_lat)))
        if gap <= max_gap_days and np.hypot(dlat, dlon) <= max_dist_deg:
            n += 1
    return n


def compare_year(year: int, workdir: Workdir, top: int = 6
                 ) -> tuple[pd.DataFrame, dict]:
    """Per run and ranking: what happened to the reference top events.
    And the summer's summary row."""
    def load(s):
        tag = tag_of(s)
        path = workdir.catalogue_file(year, tag)
        if not path.exists():
            return None
        run = pd.read_csv(path)
        for col in DATES:
            if col in run:
                run[col] = pd.to_datetime(run[col])
        return run, footprint(workdir.labels_file(year, tag))

    got = load(REFERENCE)
    if got is None or got[0].empty:
        return pd.DataFrame(), {}
    ref, ref_fp = got
    metrics = {k: c for k, c in METRICS.items() if c in ref}

    rows = []
    for family, value, s in settings(year):
        if s == REFERENCE or family == "paper":
            continue
        got = load(s)
        if got is None:
            continue
        run, run_fp = got
        m = match_by_overlap(ref, ref_fp, run, run_fp, MIN_OVERLAP)
        for metric, col in metrics.items():
            ref_rank = metric_rank(ref, col)
            order = np.argsort(ref_rank, kind="stable")
            mm = m.iloc[order]
            run_rank = (pd.DataFrame({"rank": metric_rank(run, col)})
                        if len(run) else pd.DataFrame({"rank": []}))
            new = np.array([run_rank["rank"].iloc[j] if j >= 0 else np.inf
                            for j in mm.run_index[:top]])
            old = np.arange(1, len(new) + 1)
            top_hits = mm.run_index[:top][mm.run_index[:top] >= 0]
            # Which reference event the run's number 1 is (0: none of them),
            # and the log severity gap between reference 1 and 2, before and
            # in this run, for the null model in hwcat.review.
            winner = 0
            if len(run):
                r1 = int(np.argmin(run_rank["rank"].to_numpy()))
                on_top = np.flatnonzero(mm.run_index.to_numpy() == r1)
                winner = int(on_top[0]) + 1 if on_top.size else 0
            ref_sev = ref[col].to_numpy()[order]
            gap_ref = (np.log(ref_sev[0] / ref_sev[1])
                       if len(ref_sev) > 1 and ref_sev[1] > 0 else np.nan)
            j1, j2 = (mm.run_index.iloc[:2].tolist() + [-1, -1])[:2]
            gap_run = (np.log(run[col].iloc[j1] / run[col].iloc[j2])
                       if min(j1, j2) >= 0 and j1 != j2
                       and run[col].iloc[j2] > 0 else np.nan)
            rows.append({
                "year": year, "family": family, "value": value,
                **s._asdict(), "metric": metric, "n_events": len(run),
                "leader": leader_outcome(mm, run_rank, "rank", top),
                "leader_new_rank": new[0],
                "n_top_moved": int((new != old).sum()),
                "top3_left_top": bool((new[:3] > top).any()),
                "top_pair_merged": bool(top_hits.duplicated().any()),
                "winner_ref_rank": winner, "gap_ref": gap_ref,
                "gap_run": gap_run})
    table = pd.DataFrame(rows)

    sev = np.sort(ref.severity_C_km2_day.to_numpy())[::-1]
    lead = 1 - sev[1] / sev[0] if len(sev) > 1 else np.nan
    leader = ref.iloc[0]
    summary = {"year": year, "in_baseline": BASELINE[0] <= year <= BASELINE[1],
               "n_events": len(ref), "n_ref": min(top, len(ref)), "lead": lead,
               "leader_start": leader.start.date(), "leader_peak": leader.peak.date(),
               "leader_end": leader.end.date(),
               "leader_peak_month": int(leader.peak.month),
               "leader_touches_edge": bool(leader.get("touches_edge", False)),
               "crowding": crowding(ref.head(top))}
    if not table.empty:
        moved = table.leader.isin(["replaced", "vanished"])
        for metric in metrics:
            t = table.metric == metric
            single = t & table.family.isin(SINGLE)
            sfx = "" if metric == "severity" else f"_{metric}"
            summary[f"leader_moved{sfx}"] = bool((single & moved).any())
            summary[f"n_settings_moving_leader{sfx}"] = int((single & moved).sum())
        for metric in metrics:
            sfx = "" if metric == "severity" else f"_{metric}"
            t = table.metric == metric
            for family in (*SINGLE, "window_raw", "grid"):
                summary[f"moved_{family}{sfx}"] = bool(
                    (t & (table.family == family) & moved).any())
        t = (table.metric == "severity")
        single = t & table.family.isin(SINGLE)
        summary["leader_absorbed"] = bool((single & (table.leader == "absorbed")).any())
        summary["leader_split"] = bool((single & (table.leader == "split")).any())
        summary["top3_left_top"] = bool((single & table.top3_left_top).any())
        summary["top_pair_merged"] = bool((single & table.top_pair_merged).any())
    return table, summary


def paper_table(year: int, workdir: Workdir, top: int = 6) -> pd.DataFrame:
    """The table behind paper Figure 4: the reference top events under each
    minimum area and percentile at the reference window, matched by shared
    cell-days. Two reference events matched to the same new event share a
    new rank, which is how the figure marks a merge."""
    def load(area, pct):
        s = REFERENCE._replace(area=area, pct=pct)
        run = pd.read_csv(workdir.catalogue_file(year, tag_of(s)))
        for col in DATES:
            run[col] = pd.to_datetime(run[col])
        return run, footprint(workdir.labels_file(year, tag_of(s)))

    ref_all, ref_fp = load(REFERENCE.area, REFERENCE.pct)
    ref = ref_all.head(top)
    rows = []
    for pct in PAPER_PCTS:
        for area in PAPER_AREAS:
            run, run_fp = load(area, pct)
            m = match_by_overlap(ref_all, ref_fp, run, run_fp,
                                 MIN_OVERLAP).iloc[:len(ref)]
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


def plot(summary: pd.DataFrame, path: Path) -> None:
    """Summers in which each change of setting moves the leader, by ranking."""
    from hwcat.plots import mpl_apply
    mpl_apply()
    import matplotlib.pyplot as plt

    families = [f for f in (*SINGLE, "window_raw", "grid")
                if f"moved_{f}" in summary]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(families)), [summary[f"moved_{f}"].sum() for f in families],
           color="#B5470F")
    ax.set_xticks(range(len(families)),
                  [f.replace("_", " ") for f in families], rotation=20)
    ax.set_ylabel(f"summers (of {len(summary)}) where the leader moved")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"-> {path}")


def sweep(years: list[int], workdir: Workdir, top: int = 6, jobs: int = 3,
          skip_runs: bool = False) -> pd.DataFrame:
    """Every setting for every summer, the per-summer summary and its figure."""
    workdir.robustness.mkdir(parents=True, exist_ok=True)
    if not skip_runs:
        run_all(years, workdir, jobs)

    rows = []
    for year in years:
        table, summary = compare_year(year, workdir, top)
        if not summary:
            print(f"{year}: no reference catalogue, skipped")
            continue
        if year in PAPER_YEARS:
            paper_table(year, workdir, top)
        table.to_csv(per_year_table(workdir, year), index=False)
        rows.append(summary)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("no summers had a reference catalogue")
    out = workdir.robustness / "catalogue_stability_allyears.csv"
    summary.to_csv(out, index=False)
    print(f"\n-> {out}\n")
    n = len(summary)
    for col in [c for c in summary if c.startswith(("leader_moved", "moved_"))]:
        print(f"{col:32s} {int(summary[col].sum()):3d} of {n}")
    plot(summary, workdir.robustness / "catalogue_stability_allyears.pdf")
    return summary
