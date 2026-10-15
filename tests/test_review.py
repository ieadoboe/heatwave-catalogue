"""The reviewer checks on a hand-made summer with one alternative setting.

    python tests/test_review.py
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from hwcat import review as R, robustness as AY
from hwcat.workdir import Workdir

SHAPE = np.array([10, 5, 5])


def save(path, cells_by_event):
    ids = [e for e, cells in cells_by_event.items() for _ in cells]
    flat = [c for cells in cells_by_event.values() for c in cells]
    t, i, j = np.unravel_index(np.asarray(flat, dtype=int), tuple(SHAPE))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=t, i=i, j=j,
                        event_id=np.asarray(ids, dtype=np.int32), shape=SHAPE)


def cell(t, i, j):
    return int(np.ravel_multi_index((t, i, j), tuple(SHAPE)))


def catalogue(ids, severity, peak, days):
    cat = pd.DataFrame({"event_id": ids, "severity_C_km2_day": severity,
                        "max_anom_C": peak, "duration_days": days})
    cat = cat.sort_values("severity_C_km2_day", ascending=False)
    cat.insert(0, "rank", range(1, len(cat) + 1))
    return cat


def test_metric_rank_breaks_ties_by_severity():
    cat = catalogue([1, 2, 3], [50, 90, 10], [1, 1, 1], [5, 5, 3])
    # Durations 5, 5, 3: the more severe of the two five-day events leads.
    ranks = R.metric_rank(cat, "duration_days")
    assert list(cat.event_id[ranks == 1]) == [2], ranks


def test_review_year():
    """A, B and C; widening the area makes B the most severe. B was already
    the longest and hottest, so only the severity leader moves."""
    fp = {1: [cell(t, 0, 0) for t in range(5)],     # A
          2: [cell(t, 2, 2) for t in range(10)],    # B
          3: [cell(t, 4, 4) for t in range(3)]}     # C
    with tempfile.TemporaryDirectory() as d:
        w = Workdir(Path(d))
        w.catalogues.mkdir(parents=True)
        ref_tag = AY.tag_for(*AY.REFERENCE)
        catalogue([1, 2, 3], [100, 90, 10], [3, 8, 1], [5, 10, 3]).to_csv(
            w.catalogue_file(2003, ref_tag), index=False)
        save(w.labels_file(2003, ref_tag), fp)
        alt = AY.tag_for(25_000, 0.90, 15)
        catalogue([7, 8, 9], [80, 95, 10], [3, 8, 1], [5, 10, 3]).to_csv(
            w.catalogue_file(2003, alt), index=False)
        save(w.labels_file(2003, alt), {7: fp[1], 8: fp[2], 9: fp[3]})

        rows = {r["metric"]: r for r in R.review_year(2003, w)}
    assert rows["severity"]["leader_moved"], rows["severity"]
    assert rows["severity"]["leader_moved_by"] == "area"
    assert rows["severity"]["settings_moving_leader"] == 1
    assert abs(rows["severity"]["leader_margin"] - 0.10) < 1e-9
    assert not rows["duration"]["leader_moved"], rows["duration"]
    assert not rows["peak exceedance"]["leader_moved"]
    assert not any(r["top3_left_top6"] for r in rows.values())
    assert rows["severity"]["in_baseline"] and rows["severity"]["n_settings"] == 1


if __name__ == "__main__":
    test_metric_rank_breaks_ties_by_severity()
    test_review_year()
    print("ok")
