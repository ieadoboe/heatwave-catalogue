"""Pair the events of two catalogue runs by the cell-days they share.

Matching by dates and centroids was dropped on 2026-10-14: it reported a
small event that vanished as having merged into a big neighbour."""

from pathlib import Path

import numpy as np
import pandas as pd


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
