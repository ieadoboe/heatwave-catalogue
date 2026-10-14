"""Pair the events of two catalogue runs: by dates and centroids (fast, used
by the 3-summer sweep) or by shared cell-days (exact, used everywhere else)."""

from pathlib import Path

import numpy as np
import pandas as pd


def match_events(ref: pd.DataFrame, run: pd.DataFrame,
                 max_dist_deg: float = 10.0) -> pd.Series:
    """For each reference event, the index in `run` of its counterpart, or -1.

    A counterpart must overlap the reference in time. Among those, the one
    with the most overlapping days wins, nearest centroid as tie-break, and
    it must lie within max_dist_deg of the reference centroid. Frames must
    carry start, end, centroid_lat and centroid_lon; dates as datetimes.

    ponytail: date-plus-centroid matching, not cell-day overlap, so a run
    that splits one reference event into two reports only the larger half.
    Swap in overlap of the label grids if that ever decides a result."""
    out = []
    for _, e in ref.iterrows():
        overlap = ((np.minimum(run.end, e.end) - np.maximum(run.start, e.start))
                   .dt.days + 1).clip(lower=0)
        dlat = run.centroid_lat - e.centroid_lat
        dlon = (run.centroid_lon - e.centroid_lon) * np.cos(np.radians(e.centroid_lat))
        dist = np.hypot(dlat, dlon)
        ok = (overlap > 0) & (dist <= max_dist_deg)
        if not ok.any():
            out.append(-1)
            continue
        score = pd.DataFrame({"overlap": overlap[ok], "dist": dist[ok]})
        best = score.sort_values(["overlap", "dist"],
                                 ascending=[False, True]).index[0]
        out.append(int(best))
    return pd.Series(out, index=ref.index)


def footprint(path: Path) -> pd.DataFrame:
    """One row per event cell-day: a flat (time, lat, lon) index and the
    event_id it belongs to, from detect_heatwaves_europe.py --save-labels."""
    z = np.load(path)
    cell = np.ravel_multi_index((z["t"], z["i"], z["j"]), tuple(z["shape"]))
    return pd.DataFrame({"cell": cell, "event_id": z["event_id"]})


def match_by_overlap(ref: pd.DataFrame, ref_fp: pd.DataFrame,
                     run: pd.DataFrame, run_fp: pd.DataFrame) -> pd.DataFrame:
    """Match every reference event to the run event sharing most cell-days.

    Returns one row per row of `ref`: run_index (row of `run`, or -1 when no
    cell-day is shared), shared (cell-days in common), n_pieces (run events
    it overlaps, so >1 means it split) and merged (another reference event
    has the same best match, so the run joined them).

    Shared cells, unlike dates and centroids, cannot mistake a vanished
    event for a merged one: a reference event counts as merged only if its
    own cells lie inside the joint run event. Under a stricter setting (higher
    percentile, larger area) cells are only removed, so a true merge is
    impossible there and none is reported."""
    if not len(run) or "event_id" not in run:   # the setting found nothing
        return pd.DataFrame({"run_index": -1, "shared": 0, "n_pieces": 0,
                             "merged": False}, index=ref.index)
    both = ref_fp.merge(run_fp, on="cell", suffixes=("_ref", "_run"))
    counts = (both.groupby(["event_id_ref", "event_id_run"]).size()
              .rename("shared").reset_index())
    best = (counts.sort_values("shared", ascending=False, kind="stable")
            .drop_duplicates("event_id_ref").set_index("event_id_ref"))
    pieces = counts.groupby("event_id_ref").size()
    row_of = pd.Series(np.arange(len(run)), index=run.event_id.to_numpy())

    out = pd.DataFrame(index=ref.index)
    ids = ref.event_id.to_numpy()
    run_id = best.event_id_run.reindex(ids)
    out["run_index"] = [int(row_of[r]) if pd.notna(r) else -1 for r in run_id]
    out["shared"] = best.shared.reindex(ids).fillna(0).astype(int).to_numpy()
    out["n_pieces"] = pieces.reindex(ids).fillna(0).astype(int).to_numpy()
    hit = out.run_index[out.run_index >= 0]
    out["merged"] = out.run_index.isin(hit[hit.duplicated(keep=False)])
    return out
