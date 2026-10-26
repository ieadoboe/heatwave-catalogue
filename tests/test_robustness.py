"""Check the leave-one-out baseline, the threshold cache names, the settings
list, and the all-summers comparison on hand-made catalogues.

    python tests/test_robustness.py
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from hwcat import catalogue as DH, robustness as AY
from hwcat.workdir import Workdir

SHAPE = np.array([10, 5, 5])


def save(path, ids, cells):
    t, i, j = np.unravel_index(np.asarray(cells, dtype=int), tuple(SHAPE))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=t, i=i, j=j,
                        event_id=np.asarray(ids, dtype=np.int32), shape=SHAPE)


def test_leave_one_out():
    inside = DH.baseline_years_for(2003, 1991, 2020, leave_one_out=True)
    assert 2003 not in inside and len(inside) == 29, inside
    outside = DH.baseline_years_for(2026, 1991, 2020, leave_one_out=True)
    assert len(outside) == 30, outside
    shared = DH.baseline_years_for(2003, 1991, 2020, leave_one_out=False)
    assert 2003 in shared and len(shared) == 30


def test_threshold_name():
    full = list(range(1991, 2021))
    # Raw pooling keeps the original cache names, so old caches still match.
    assert DH.threshold_name("europe", 1991, 2020, 0.90, "m04-10", 15, full) == \
        "heatwave_threshold_europe_1991_2020_p90_m04-10.nc"
    loo = DH.baseline_years_for(2003, 1991, 2020, True)
    name = DH.threshold_name("europe", 1991, 2020, 0.95, "m04-10", 31, loo)
    assert name == "heatwave_threshold_europe_1991_2020_p95_w31_loo2003_m04-10.nc"
    other = DH.baseline_years_for(2004, 1991, 2020, True)
    assert name != DH.threshold_name("europe", 1991, 2020, 0.95, "m04-10", 31, other)
    sc = DH.threshold_name("europe", 1991, 2020, 0.90, "m04-10", 21, loo, True)
    assert sc == "heatwave_threshold_europe_1991_2020_p90_w21_sc_loo2003_m04-10.nc"
    assert Path(sc).match("heatwave_threshold_europe_*_loo2003_*.nc")
    # The interquartile thresholds behind --rescore get their own names.
    q = DH.threshold_name("europe", 1991, 2020, 0.25, "m04-10", 15, loo, True)
    assert "_p25_sc_loo2003" in q, q


def test_default_tag():
    full = list(range(1991, 2021))
    loo = DH.baseline_years_for(2003, 1991, 2020, True)
    # The seasonal cycle is removed by default; raw pooling is marked.
    assert DH.default_tag(15, 1991, 2020, full) == ""
    assert DH.default_tag(15, 1991, 2020, full, remove_cycle=False) == "_raw"
    assert DH.default_tag(31, 1991, 2020, loo) == "_w31_loo"
    assert DH.default_tag(21, 1991, 2020, loo, False) == "_w21_raw_loo"


def test_names_never_collide():
    full = list(range(1991, 2021))
    a = DH.threshold_name("europe", 1991, 2020, 0.57, "m04-10", 15, full)
    b = DH.threshold_name("europe", 1991, 2020, 0.56, "m04-10", 15, full)
    assert a != b and "_p57_" in a, (a, b)
    assert "_p97.5_" in DH.threshold_name("europe", 1991, 2020, 0.975, "m04-10", 15, full)


def test_settings_and_tags():
    assert AY.tag_of(AY.REFERENCE) == "_a50k_p90_w15_sc"
    assert AY.tag_for(50_000, 0.90, 15) == "_a50k_p90_w15"   # the old grid's name
    assert AY.tag_for(250_000, 0.925, 15, 4, True, 26) == "_a250k_p92.5_w15_d4_sc_c26"
    s = AY.settings(2010)
    tags = {AY.tag_of(x) for _, _, x in s}
    assert len(tags) == len({x for _, _, x in s}), "two settings share a tag"
    fams = {f for f, _, _ in s}
    assert fams == {*AY.SWEEPS, "window_raw", "grid"}, fams
    assert ("window_raw", 31, AY.REFERENCE._replace(window=31, cycle=False)) in s
    # The paper's area x percentile combinations only for its three summers.
    assert "paper" not in fams and "paper" in {f for f, _, _ in AY.settings(2003)}
    # Every single-change family includes the reference value once.
    for fam in AY.SWEEPS:
        assert sum(x == AY.REFERENCE for f, _, x in s if f == fam) == 1, fam


def test_crowding():
    top = pd.DataFrame({
        "start": pd.to_datetime(["2026-06-12", "2026-06-30", "2026-08-19"]),
        "end": pd.to_datetime(["2026-07-03", "2026-07-24", "2026-09-12"]),
        "centroid_lat": [48.9, 46.8, 41.3],
        "centroid_lon": [13.6, 4.8, 17.1]})
    assert AY.crowding(top) == 1
    assert AY.crowding(top, max_dist_deg=5.0) == 0


def catalogue(ids, sev, sev_ref=None, sev_norm=None, start="2003-07-01",
              peak="2003-07-05", end="2003-07-10"):
    n = len(ids)
    return pd.DataFrame({
        "rank": range(1, n + 1), "event_id": ids,
        "start": [start] * n, "peak": [peak] * n, "end": [end] * n,
        "centroid_lat": [46.0 + 5 * k for k in range(n)],
        "centroid_lon": [4.0 + 10 * k for k in range(n)],
        "severity_C_km2_day": sev,
        "severity_ref_C_km2_day": sev_ref if sev_ref is not None else sev,
        "severity_norm_km2_day": sev_norm if sev_norm is not None else sev,
        "touches_edge": [False] * n})


def test_compare_year():
    """Event 1 (cells 0-4) leads event 2 (cells 5-7). At the 95th percentile
    event 2 overtakes by its own bar but not by the reference bar; at
    25,000 km2 the two join; at 250,000 km2 nothing is found."""
    with tempfile.TemporaryDirectory() as d:
        w = Workdir(Path(d))
        w.labels.mkdir(parents=True)
        ref = catalogue([1, 2], [9e7, 5e7])
        for fam, value, s in AY.settings(2003):
            tag = AY.tag_of(s)
            csv = w.catalogue_file(2003, tag)
            if s == AY.REFERENCE._replace(pct=0.95):
                catalogue([2, 1], [6e7, 4e7], sev_ref=[5e7, 9e7]).to_csv(csv, index=False)
                save(w.labels_file(2003, tag), [1] * 5 + [2] * 3, range(8))
            elif s == AY.REFERENCE._replace(area=25_000):
                catalogue([7], [1.4e8]).to_csv(csv, index=False)
                save(w.labels_file(2003, tag), [7] * 8, range(8))
            elif s == AY.REFERENCE._replace(area=250_000):
                pd.DataFrame(columns=["rank", "start", "peak", "end"]).to_csv(
                    csv, index=False)
                save(w.labels_file(2003, tag), [], [])
            else:
                ref.to_csv(csv, index=False)
                save(w.labels_file(2003, tag), [1] * 5 + [2] * 3, range(8))
        table, summary = AY.compare_year(2003, w, top=6)

    def leader(metric, **kw):
        t = table[table.metric == metric]
        for k, v in kw.items():
            t = t[t[k] == v]
        return t.leader.iloc[0]

    assert leader("severity", family="percentile", pct=0.95) == "replaced"
    assert leader("severity_ref", family="percentile", pct=0.95) == "held"
    assert leader("severity", family="min_area_km2", area=25_000) == "absorbed"
    assert leader("severity", family="min_area_km2", area=250_000) == "vanished"
    assert leader("severity", family="min_days", days=4) == "held"
    assert summary["leader_moved"] and summary["moved_percentile"]
    assert summary["moved_min_area_km2"]          # the vanishing counts
    # By the reference bar the 95th no longer moves it; only the vanishing does.
    assert summary["n_settings_moving_leader"] == 2, summary
    assert summary["n_settings_moving_leader_severity_ref"] == 1, summary
    assert summary["leader_absorbed"] and not summary["leader_split"]
    assert summary["top_pair_merged"]
    assert abs(summary["lead"] - (1 - 5 / 9)) < 1e-9
    assert summary["leader_peak_month"] == 7 and summary["in_baseline"]


def test_restart_skips_finished_settings():
    """A second run_year must not relaunch settings whose current outputs
    exist, and must redo ones an older version wrote."""
    calls = []
    real = AY.subprocess.run
    AY.subprocess.run = lambda cmd, **kw: calls.append(cmd)
    try:
        with tempfile.TemporaryDirectory() as d:
            w = Workdir(Path(d))
            w.labels.mkdir(parents=True)
            stale = AY.REFERENCE._replace(area=25_000)
            for _, _, s in AY.settings(2010):
                tag = AY.tag_of(s)
                header = "rank,severity_C_km2_day\n" if s == stale else \
                    "rank,severity_C_km2_day,touches_edge\n"
                w.catalogue_file(2010, tag).write_text(header)
                w.labels_file(2010, tag).write_bytes(b"")
            AY.run_year(2010, w)
    finally:
        AY.subprocess.run = real
    assert len(calls) == 1 and AY.tag_of(stale) in calls[0], calls


def test_paper_table():
    cols = ["rank", "event_id", "start", "peak", "end", "duration_days",
            "peak_area_km2", "severity_C_km2_day", "centroid_lat", "centroid_lon"]
    ref = pd.DataFrame([[1, 1, "2003-07-29", "2003-08-10", "2003-08-15", 18, 9e5, 5e7, 46.0, 4.0],
                        [2, 2, "2003-07-20", "2003-07-25", "2003-07-28", 9, 4e5, 2e7, 47.0, 6.0]],
                       columns=cols)
    joint = pd.DataFrame([[1, 7, "2003-07-20", "2003-08-10", "2003-08-15", 27, 1.3e6, 7e7, 46.5, 5.0]],
                         columns=cols)
    with tempfile.TemporaryDirectory() as d:
        w = Workdir(Path(d))
        w.catalogues.mkdir(parents=True)
        for area in AY.PAPER_AREAS:
            for pct in AY.PAPER_PCTS:
                tag = AY.tag_of(AY.REFERENCE._replace(area=area, pct=pct))
                if (area, pct) == (25_000, 0.90):
                    joint.to_csv(w.catalogue_file(2003, tag), index=False)
                    save(w.labels_file(2003, tag), [7] * 8, range(8))
                else:
                    ref.to_csv(w.catalogue_file(2003, tag), index=False)
                    save(w.labels_file(2003, tag), [1] * 5 + [2] * 3, range(8))
        table = AY.paper_table(2003, w)
        assert (w.robustness / "catalogue_sensitivity_europe_2003.csv").exists()
    assert len(table) == 2 * 6, len(table)
    merged = table[(table.min_area_km2 == 25_000) & (table.percentile == 0.90)]
    assert list(merged["rank"]) == [1, 1], merged
    assert merged.survives.all()


if __name__ == "__main__":
    test_leave_one_out()
    test_threshold_name()
    test_default_tag()
    test_names_never_collide()
    test_settings_and_tags()
    test_crowding()
    test_compare_year()
    test_restart_skips_finished_settings()
    test_paper_table()
    print("ok")
