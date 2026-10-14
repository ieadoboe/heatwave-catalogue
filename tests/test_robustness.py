"""Check the leave-one-out baseline, the threshold cache names, and the
all-summers summary helpers.

    python tests/test_catalogue_stability.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


from hwcat import catalogue as DH, robustness as AY
from hwcat.workdir import Workdir


def test_leave_one_out():
    inside = DH.baseline_years_for(2003, 1991, 2020, leave_one_out=True)
    assert 2003 not in inside and len(inside) == 29, inside
    outside = DH.baseline_years_for(2026, 1991, 2020, leave_one_out=True)
    assert len(outside) == 30, outside
    shared = DH.baseline_years_for(2003, 1991, 2020, leave_one_out=False)
    assert 2003 in shared and len(shared) == 30


def test_threshold_name():
    full = list(range(1991, 2021))
    # The default keeps its original name, so old caches still match.
    assert DH.threshold_name("europe", 1991, 2020, 0.90, "m04-10", 15, full) == \
        "heatwave_threshold_europe_1991_2020_p90_m04-10.nc"
    loo = DH.baseline_years_for(2003, 1991, 2020, True)
    name = DH.threshold_name("europe", 1991, 2020, 0.95, "m04-10", 31, loo)
    assert name == "heatwave_threshold_europe_1991_2020_p95_w31_loo2003_m04-10.nc"
    # Two summers must never share a leave-one-out cache.
    other = DH.baseline_years_for(2004, 1991, 2020, True)
    assert name != DH.threshold_name("europe", 1991, 2020, 0.95, "m04-10", 31, other)


def test_names_never_collide():
    full = list(range(1991, 2021))
    # Truncating 0.57 * 100 gave 56, colliding with 0.56.
    a = DH.threshold_name("europe", 1991, 2020, 0.57, "m04-10", 15, full)
    b = DH.threshold_name("europe", 1991, 2020, 0.56, "m04-10", 15, full)
    assert a != b and "_p57_" in a, (a, b)
    assert "_p97.5_" in DH.threshold_name("europe", 1991, 2020, 0.975, "m04-10", 15, full)
    # A standalone run must not overwrite the reference catalogue.
    loo = DH.baseline_years_for(2003, 1991, 2020, True)
    assert DH.default_tag(15, 1991, 2020, full) == ""
    assert DH.default_tag(31, 1991, 2020, loo) == "_w31_loo"


def test_changed_parameter():
    assert AY.changed_parameter(25_000, 0.90, 15) == "area"
    assert AY.changed_parameter(50_000, 0.95, 15) == "percentile"
    assert AY.changed_parameter(50_000, 0.90, 7) == "window"
    assert AY.changed_parameter(25_000, 0.95, 15) is None


def test_crowding():
    top = pd.DataFrame({
        "start": pd.to_datetime(["2026-06-12", "2026-06-30", "2026-08-19"]),
        "end": pd.to_datetime(["2026-07-03", "2026-07-24", "2026-09-12"]),
        "centroid_lat": [48.9, 46.8, 41.3],
        "centroid_lon": [13.6, 4.8, 17.1]})
    # June and July overlap by four days, about 6 degrees apart: one pair.
    # August starts 26 days after July ends: not close in time.
    assert AY.crowding(top) == 1
    assert AY.crowding(top, max_dist_deg=5.0) == 0


def test_empty_setting():
    """compare_year end to end, with one setting that found nothing."""
    import tempfile
    import numpy as np
    ref = pd.DataFrame({
        "rank": [1, 2], "event_id": [1, 2],
        "start": ["2003-07-29", "2003-04-29"], "peak": ["2003-08-10", "2003-05-08"],
        "end": ["2003-08-15", "2003-05-16"],
        "centroid_lat": [46.0, 45.0], "centroid_lon": [4.0, 19.0]})
    shape = np.array([10, 5, 5])

    def save(path, ids, cells):
        t, i, j = np.unravel_index(np.asarray(cells, dtype=int), tuple(shape))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=t, i=i, j=j,
                            event_id=np.asarray(ids, dtype=np.int32), shape=shape)

    with tempfile.TemporaryDirectory() as d:
        w = Workdir(Path(d))
        w.labels.mkdir(parents=True)
        for area, pct, window in AY.product(AY.AREAS_KM2, AY.PERCENTILES,
                                            AY.WINDOWS):
            tag = AY.tag_for(area, pct, window)
            csv = w.catalogue_file(2003, tag)
            if (area, pct, window) == (100_000, 0.95, 7):
                pd.DataFrame(columns=["rank", "start", "peak", "end"]).to_csv(
                    csv, index=False)
                save(w.labels_file(2003, tag), [], [])
            else:
                ref.to_csv(csv, index=False)
                save(w.labels_file(2003, tag), [1] * 5 + [2] * 3, list(range(8)))
        table, summary = AY.compare_year(2003, w, top=6)
    empty = table[(table.min_area_km2 == 100_000) & (table.percentile == 0.95)
                  & (table.window_days == 7)]
    assert empty.moved.all() and empty["rank"].isna().all(), empty
    assert summary["n_ref"] == 2 and summary["n_moved_any"] == 2, summary
    assert summary["n_moved_area"] == 0, summary   # only the "several" bucket
    assert summary["n_top_merged"] == 0, summary


def test_restart_skips_finished_settings():
    """A second run_year must not relaunch settings whose outputs exist."""
    import tempfile
    calls = []
    real = AY.subprocess.run
    AY.subprocess.run = lambda cmd, **kw: calls.append(cmd)
    try:
        with tempfile.TemporaryDirectory() as d:
            w = Workdir(Path(d))
            w.labels.mkdir(parents=True)
            for area, pct, window in AY.product(AY.AREAS_KM2, AY.PERCENTILES, AY.WINDOWS):
                tag = AY.tag_for(area, pct, window)
                w.catalogue_file(2003, tag).write_text("rank,start,peak,end\n")
                w.labels_file(2003, tag).write_bytes(b"")
            AY.run_year(2003, w)
    finally:
        AY.subprocess.run = real
    assert calls == [], calls


def test_paper_table():
    """The 3-summer table behind paper Figure 4: same columns as before,
    15-day window only, shared-cell matching, merges as a shared new rank."""
    import tempfile
    import numpy as np
    shape = np.array([10, 5, 5])
    cols = ["rank", "event_id", "start", "peak", "end", "duration_days",
            "peak_area_km2", "severity_C_km2_day", "centroid_lat", "centroid_lon"]
    ref = pd.DataFrame([[1, 1, "2003-07-29", "2003-08-10", "2003-08-15", 18, 9e5, 5e7, 46.0, 4.0],
                        [2, 2, "2003-07-20", "2003-07-25", "2003-07-28", 9, 4e5, 2e7, 47.0, 6.0]],
                       columns=cols)
    joint = pd.DataFrame([[1, 7, "2003-07-20", "2003-08-10", "2003-08-15", 27, 1.3e6, 7e7, 46.5, 5.0]],
                         columns=cols)

    def save(path, ids, cells):
        t, i, j = np.unravel_index(np.asarray(cells, dtype=int), tuple(shape))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=t, i=i, j=j,
                            event_id=np.asarray(ids, dtype=np.int32), shape=shape)

    with tempfile.TemporaryDirectory() as d:
        w = Workdir(Path(d))
        w.catalogues.mkdir(parents=True)
        for area, pct, window in AY.product(AY.AREAS_KM2, AY.PERCENTILES, AY.WINDOWS):
            tag = AY.tag_for(area, pct, window)
            if (area, pct, window) == (25_000, 0.90, 15):      # joins 1 and 2
                joint.to_csv(w.catalogue_file(2003, tag), index=False)
                save(w.labels_file(2003, tag), [7] * 8, list(range(8)))
            else:
                ref.to_csv(w.catalogue_file(2003, tag), index=False)
                save(w.labels_file(2003, tag), [1] * 5 + [2] * 3, list(range(8)))
        table = AY.paper_table(2003, w)
        written = w.robustness / "catalogue_sensitivity_europe_2003.csv"
        assert written.exists(), written
    assert list(table.columns) == ["min_area_km2", "percentile", "n_events",
                                   "ref_rank", "survives", "rank", "start", "end",
                                   "duration_days", "peak_area_km2",
                                   "severity_C_km2_day"], list(table.columns)
    assert len(table) == 2 * 6, len(table)       # 2 events x 6 settings, window 15 only
    merged = table[(table.min_area_km2 == 25_000) & (table.percentile == 0.90)]
    assert list(merged["rank"]) == [1, 1], merged   # two references, one new event
    assert merged.survives.all()


if __name__ == "__main__":
    test_leave_one_out()
    test_threshold_name()
    test_names_never_collide()
    test_empty_setting()
    test_restart_skips_finished_settings()
    test_paper_table()
    test_changed_parameter()
    test_crowding()
    print("ok")
