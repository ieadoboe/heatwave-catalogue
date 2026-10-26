"""paper/write_numbers.py on hand-made result tables.

    python tests/test_numbers.py
"""
import importlib.util
import tempfile
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "write_numbers", Path(__file__).resolve().parents[1] / "paper" / "write_numbers.py")
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)


def test_macros():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pd.DataFrame({"year": [2003, 2026], "n_events": [40, 46],
                      "leader_moved": [True, False],
                      "moved_percentile": [True, False],
                      "moved_percentile_severity_norm": [False, False],
                      "leader_peak_month": [8, 5]}).to_csv(
            d / "catalogue_stability_allyears.csv", index=False)
        pd.DataFrame({"check": ["leader_moved", "spearman_lead"],
                      "k": [1, None], "n": [2, 2], "lo": [0.09, None],
                      "hi": [0.91, None], "rho": [None, -0.42],
                      "p": [None, 0.0105]}).to_csv(d / "review_checks.csv", index=False)
        pd.DataFrame({"severity_C_km2_day": [50, 30, 10, 5, 3, 2],
                      "duration_days": [22, 24, 14, 5, 4, 3]}).to_csv(
            d / "heatwave_events_europe_2026.csv", index=False)
        v = N.macros(d)
        N.write(v, d / "numbers.tex")
        tex = (d / "numbers.tex").read_text()
    assert v["NSummers"] == "2" and v["NLeaderMoved"] == "1"
    assert v["NMovedPercentile"] == "1" and v["NLeaderJJA"] == "1"
    assert v["NMovedPercentileNorm"] == "0"
    assert v["LeaderMovedLo"] == "9" and v["RhoLead"] == "$-$0.42"
    assert v["NEventsTwentySix"] == "6" and v["TopOneShareTwentySix"] == "50"
    assert v["NForEightyTwentySix"] == "2" and v["LeadTwentySix"] == "40"
    assert r"\newcommand{\NSummers}{2}" in tex


if __name__ == "__main__":
    test_macros()
    print("ok")
