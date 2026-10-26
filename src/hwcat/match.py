"""Pair the events of two catalogue runs by the cell-days they share.

Matching by dates and centroids was dropped on 2026-10-14: it reported a
small event that vanished as having merged into a big neighbour."""

from pathlib import Path

import numpy as np
import pandas as pd

MIN_OVERLAP = 0.5   # share of a reference event's cell-days its match must hold


def footprint(path: Path) -> pd.DataFrame:
    """One row per event cell-day: a flat (time, lat, lon) index and the
    event_id it belongs to, from detect_heatwaves_europe.py --save-labels."""
    z = np.load(path)
    cell = np.ravel_multi_index((z["t"], z["i"], z["j"]), tuple(z["shape"]))
    return pd.DataFrame({"cell": cell, "event_id": z["event_id"]})


def match_by_overlap(ref: pd.DataFrame, ref_fp: pd.DataFrame,
                     run: pd.DataFrame, run_fp: pd.DataFrame,
                     min_overlap: float = 0.0) -> pd.DataFrame:
    """Match every reference event to the run event sharing most cell-days.

    Returns one row per row of `ref`: run_index (row of `run`, or -1 when
    unmatched), shared (cell-days in common with the best run event), share
    (shared over the reference event's own cell-days), n_pieces (run events
    it overlaps, so >1 means it split) and merged (another reference event
    has the same match, so the run joined them).

    The best run event counts as the match only if it holds at least
    `min_overlap` of the reference event's cell-days. Below that the event
    has either vanished or broken up with no piece dominant, and run_index
    is -1; n_pieces tells the two apart. Ties go to the lower run event id.

    Shared cells, unlike dates and centroids, cannot mistake a vanished
    event for a merged one: a reference event counts as merged only if its
    own cells lie inside the joint run event. Under a stricter setting (higher
    percentile, larger area) cells are only removed, so a true merge is
    impossible there and none is reported."""
    if not len(run) or "event_id" not in run:   # the setting found nothing
        return pd.DataFrame({"run_index": -1, "shared": 0, "share": 0.0,
                             "n_pieces": 0, "merged": False}, index=ref.index)
    both = ref_fp.merge(run_fp, on="cell", suffixes=("_ref", "_run"))
    counts = (both.groupby(["event_id_ref", "event_id_run"]).size()
              .rename("shared").reset_index())
    best = (counts.sort_values(["shared", "event_id_run"],
                               ascending=[False, True], kind="stable")
            .drop_duplicates("event_id_ref").set_index("event_id_ref"))
    pieces = counts.groupby("event_id_ref").size()
    size = ref_fp.groupby("event_id").size()
    row_of = pd.Series(np.arange(len(run)), index=run.event_id.to_numpy())

    out = pd.DataFrame(index=ref.index)
    ids = ref.event_id.to_numpy()
    shared = best.shared.reindex(ids).fillna(0).astype(int).to_numpy()
    share = shared / size.reindex(ids).fillna(1).clip(lower=1).to_numpy()
    run_id = best.event_id_run.reindex(ids).to_numpy()
    ok = pd.notna(run_id) & (share >= min_overlap) & (shared > 0)
    out["run_index"] = [int(row_of[r]) if k else -1 for r, k in zip(run_id, ok)]
    out["shared"] = shared
    out["share"] = share.round(4)
    out["n_pieces"] = pieces.reindex(ids).fillna(0).astype(int).to_numpy()
    hit = out.run_index[out.run_index >= 0]
    out["merged"] = out.run_index.isin(hit[hit.duplicated(keep=False)])
    return out


def leader_outcome(m: pd.DataFrame, run: pd.DataFrame, rank_col: str = "rank",
                   top: int = 6) -> str:
    """What happened to reference event 1 (first row of `m`) in a run.

    held      still ranked 1, and no other top-`top` reference event joined it
    absorbed  still ranked 1, but another top-`top` reference event merged in,
              so the case is a different object (2026 at 25,000 km2)
    replaced  matched, but another event now ranks above it
    split     broke up with no piece holding MIN_OVERLAP of its cell-days
    vanished  no cell-day survives"""
    j = int(m.run_index.iloc[0])
    if j < 0:
        return "split" if m.n_pieces.iloc[0] > 1 else "vanished"
    if int(run[rank_col].iloc[j]) != 1:
        return "replaced"
    others = m.run_index.iloc[1:top].to_numpy()
    return "absorbed" if (others == j).any() else "held"
