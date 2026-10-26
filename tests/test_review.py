"""The reviewer checks on a hand-made summary.

    python tests/test_review.py
"""
import numpy as np
import pandas as pd

from hwcat import review as R


def test_wilson():
    lo, hi = R.wilson(15, 36)
    assert abs(lo - 0.271) < 1e-3 and abs(hi - 0.578) < 1e-3, (lo, hi)
    assert R.wilson(0, 10)[0] == 0.0


def test_checks():
    summary = pd.DataFrame({
        "year": range(2000, 2008),
        "in_baseline": [True] * 6 + [False] * 2,
        "lead": [0.05, 0.10, 0.60, 0.50, 0.02, 0.40, 0.70, 0.03],
        "crowding": [0, 1, 0, 2, 1, 0, 0, 1],
        "leader_peak_month": [7, 7, 5, 8, 9, 6, 7, 4],
        "leader_moved": [True, True, False, False, True, False, False, True],
        "leader_moved_severity_norm": [True, False, False, False, True,
                                       False, False, False],
        "n_settings_moving_leader": [3, 2, 0, 0, 4, 0, 0, 2]})
    t = R.checks(summary).set_index("check")
    assert t.loc["leader_moved", "k"] == 4 and t.loc["leader_moved", "n"] == 8
    assert t.loc["leader_moved_severity_norm", "k"] == 2
    assert t.loc["spearman_lead", "rho"] < -0.8          # small lead, more moves
    assert t.loc["season_jja_vs_rest", "n"] == 5
    assert t.loc["in_baseline_vs_rest", "n_rest"] == 2
    assert 0 <= t.loc["in_baseline_vs_rest", "p"] <= 1


if __name__ == "__main__":
    test_wilson()
    test_checks()
    print("ok")
