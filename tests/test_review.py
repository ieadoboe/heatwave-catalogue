"""The reviewer checks on hand-made summaries and stability tables.

    python tests/test_review.py
"""
import numpy as np
import pandas as pd

from hwcat import review as R


def summary():
    return pd.DataFrame({
        "year": range(2000, 2008),
        "in_baseline": [True] * 6 + [False] * 2,
        "lead": [0.05, 0.10, 0.60, 0.50, 0.02, 0.40, 0.70, 0.03],
        "crowding": [0, 1, 0, 2, 1, 0, 0, 1],
        "leader_peak_month": [7, 7, 5, 8, 9, 6, 7, 4],
        "leader_touches_edge": [False, True, False, False, True, False, False, False],
        "leader_moved": [True, True, False, False, True, False, False, True],
        "leader_moved_severity_norm": [True, False, False, False, True,
                                       False, False, False],
        "n_settings_moving_leader": [3, 2, 0, 0, 4, 0, 0, 2]})


def test_wilson():
    lo, hi = R.wilson(15, 36)
    assert abs(lo - 0.271) < 1e-3 and abs(hi - 0.578) < 1e-3, (lo, hi)
    assert R.wilson(0, 10)[0] == 0.0


def test_checks():
    t = R.checks(summary()).set_index("check")
    assert t.loc["leader_moved", "k"] == 4 and t.loc["leader_moved", "n"] == 8
    assert t.loc["leader_moved_severity_norm", "k"] == 2
    assert t.loc["spearman_lead", "rho"] < -0.8          # small lead, more moves
    assert t.loc["season_jja_vs_rest", "n"] == 5
    assert t.loc["in_baseline_vs_rest", "n_rest"] == 2
    assert t.loc["edge_vs_rest", "k"] == 2


def stability_table():
    """Two settings over 40 summers. 'noise' shifts the log gap by +-0.05
    and moves only leaders whose gap is under that; 'bias' cuts every gap by
    0.5, moving most leaders."""
    rng = np.random.default_rng(3)
    gap = rng.uniform(0.0, 0.8, 40)
    rows = []
    for fam, shift in (("noise", rng.choice([-0.05, 0.05], 40)),
                       ("bias", np.full(40, -0.5))):
        new = gap + shift
        for y, (g, n) in enumerate(zip(gap, new)):
            rows.append({"year": 2000 + y, "family": fam, "value": 1,
                         "metric": "severity", "gap_ref": g, "gap_run": n,
                         "leader": "replaced" if n < 0 else "held",
                         "winner_ref_rank": 2 if n < 0 else 1})
    return pd.DataFrame(rows)


def test_null_model():
    t = R.null_model(stability_table(), n_sim=4000).set_index("family")
    # Noise of the observed size explains the noise setting's few moves...
    assert t.loc["noise", "expected_lo"] <= t.loc["noise", "moved"] \
        <= t.loc["noise", "expected_hi"] + 2, t.loc["noise"]
    # ...but a constant shift has no spread, so symmetric noise explains
    # none of the bias setting's many moves.
    assert t.loc["bias", "moved"] > 10 and t.loc["bias", "p_at_least"] < 0.01


def test_agreement():
    t = stability_table().replace({"family": {"noise": "min_days",
                                              "bias": "percentile"}})
    a = R.agreement(t)
    assert len(a) == 40 and a.leader_share.between(0, 1).all()
    moved_twice = a[a.leader_share == 0]
    assert (moved_twice.usual_rival == 2).all()
    # A grid run repeating a single change counts once.
    t2 = pd.concat([t.assign(pct=0.90, days=3),
                    t[t.family == "percentile"].assign(family="grid", pct=0.90,
                                                       days=3)])
    t2.loc[t2.family == "min_days", "days"] = 4
    a2 = R.agreement(t2)
    assert (a2.n_settings == 2).all(), a2.n_settings.unique()


def test_lead_rule():
    rng = np.random.default_rng(5)
    lead = np.exp(rng.uniform(np.log(0.01), np.log(0.9), 200))
    p = 1 / (1 + np.exp(-(-2.0 - 1.5 * np.log(lead))))
    s = pd.DataFrame({"lead": lead, "leader_moved": rng.random(200) < p})
    r = R.lead_rule(s, s.iloc[:100]).iloc[0]
    assert -2.0 < r.slope < -1.0, r                 # recovers about -1.5
    assert 0.1 < r.lead_at_even_odds < 0.4, r       # exp(-2/1.5) = 0.26
    assert r.auc_validate > 0.6 and r.brier_validate < r.brier_climatology


if __name__ == "__main__":
    test_wilson()
    test_checks()
    test_null_model()
    test_agreement()
    test_lead_rule()
    print("ok")
