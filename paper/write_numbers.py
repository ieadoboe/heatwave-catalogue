"""Write paper/numbers.tex: every number the manuscript quotes, as a macro.

    python paper/write_numbers.py      # reads paper/results/

The text uses \\NLeaderMoved{} and friends instead of typed numbers, so a
rerun of reproduce.sh can never leave the prose behind the tables. A number
whose table is missing prints as a bold ?? in the PDF.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
YEAR_WORDS = {2003: "OhThree", 2022: "TwentyTwo", 2026: "TwentySix"}


def macros(results: Path) -> dict[str, str]:
    out: dict[str, str] = {}

    def read(name):
        path = results / name
        return pd.read_csv(path) if path.exists() else None

    def pct(x):
        return f"{100 * x:.0f}"

    s = read("catalogue_stability_allyears.csv")
    if s is not None:
        out["NSummers"] = str(len(s))
        out["NEventsMedian"] = f"{s.n_events.median():g}"
        for col, name in (("leader_moved", "NLeaderMoved"),
                          ("leader_moved_severity_ref", "NLeaderMovedRefBar"),
                          ("leader_moved_severity_norm", "NLeaderMovedNorm"),
                          ("leader_absorbed", "NLeaderAbsorbed"),
                          ("leader_split", "NLeaderSplit"),
                          ("top3_left_top", "NTopThreeLeft"),
                          ("top_pair_merged", "NTopPairMerged"),
                          ("moved_percentile", "NMovedPercentile"),
                          ("moved_min_area_km2", "NMovedArea"),
                          ("moved_window_days", "NMovedWindow"),
                          ("moved_min_days", "NMovedDays"),
                          ("moved_connectivity", "NMovedConnectivity"),
                          ("moved_window_raw", "NMovedRawWindow"),
                          ("moved_grid", "NMovedGrid")):
            for sfx, tail in (("", ""), ("_severity_ref", "RefBar"),
                              ("_severity_norm", "Norm")):
                if col + sfx in s and not name.endswith(("RefBar", "Norm")):
                    out[name + tail] = str(int(s[col + sfx].sum()))
        if "leader_peak_month" in s:
            jja = s.leader_peak_month.isin([6, 7, 8])
            out["NLeaderJJA"] = str(int(jja.sum()))

    c = read("review_checks.csv")
    if c is not None:
        c = c.set_index("check")
        if "leader_moved" in c.index:
            out["LeaderMovedLo"] = pct(c.loc["leader_moved", "lo"])
            out["LeaderMovedHi"] = pct(c.loc["leader_moved", "hi"])
        for check, stem in (("spearman_lead", "Lead"), ("spearman_crowding", "Crowding")):
            if check in c.index:
                out[f"Rho{stem}"] = f"{c.loc[check, 'rho']:.2f}".replace("-", "$-$")
                out[f"P{stem}"] = f"{c.loc[check, 'p']:.2g}"
        for check, stem in (("season_jja_vs_rest", "Season"),
                            ("in_baseline_vs_rest", "Baseline"),
                            ("edge_vs_rest", "Edge")):
            if check in c.index:
                r = c.loc[check]
                out[f"{stem}K"], out[f"{stem}N"] = str(int(r.k)), str(int(r.n))
                out[f"{stem}KRest"] = str(int(r.k_rest))
                out[f"{stem}NRest"] = str(int(r.n_rest))
                out[f"P{stem}"] = f"{r.p:.2g}"

    r = read("review_lead_rule.csv")
    if r is not None:
        r = r.iloc[0]
        # "More likely than not below X" only reads right if a smaller lead
        # means more moves, i.e. a negative slope.
        if r.slope < 0:
            out["LeadEvenOdds"] = pct(r.lead_at_even_odds)
        out["AUCFit"] = f"{r.auc_fit:.2f}"
        if "auc_validate" in r and pd.notna(r.auc_validate):
            out["AUCValidate"] = f"{r.auc_validate:.2f}"
            out["BrierValidate"] = f"{r.brier_validate:.3f}"
            out["BrierClimatology"] = f"{r.brier_climatology:.3f}"

    a = read("review_agreement.csv")
    if a is not None:
        sev = a[a.metric == "severity"]
        out["NLeaderAgreeAll"] = str(int((sev.leader_share == 1).sum()))
        out["LeaderShareMedian"] = pct(sev.leader_share.median())

    for year, word in YEAR_WORDS.items():
        e = read(f"heatwave_events_europe_{year}.csv")
        if e is None or e.empty:
            continue
        sev = np.sort(e.severity_C_km2_day.to_numpy())[::-1]
        out[f"NEvents{word}"] = str(len(e))
        out[f"TopOneShare{word}"] = pct(sev[0] / sev.sum())
        out[f"TopThreeShare{word}"] = pct(sev[:3].sum() / sev.sum())
        out[f"TopSixShare{word}"] = pct(sev[:6].sum() / sev.sum())
        out[f"NForEighty{word}"] = str(int(np.searchsorted(
            np.cumsum(sev) / sev.sum(), 0.8) + 1))
        out[f"MedianDuration{word}"] = f"{e.duration_days.median():g}"
        if len(sev) > 1:
            out[f"Lead{word}"] = pct(1 - sev[1] / sev[0])
    return out


def write(values: dict[str, str], path: Path) -> None:
    lines = ["% Written by paper/write_numbers.py from paper/results/. Do not edit.",
             r"\providecommand{\missingnumber}{\textbf{??}}"]
    for name in sorted(values):
        assert name.isalpha(), name
        lines.append(rf"\newcommand{{\{name}}}{{{values[name]}}}")
    path.write_text("\n".join(lines) + "\n")


def used_in(tex: Path) -> set[str]:
    """Number macros the manuscript calls, written \\Name{}."""
    import re
    return set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", tex.read_text()))


def main(argv: list[str]) -> None:
    results = Path(argv[0]) if argv else HERE / "results"
    out = HERE / "numbers.tex"
    values = macros(results)
    missing = sorted(used_in(HERE / "catalogue_paper.tex") - set(values))
    for name in missing:
        values[name] = r"\missingnumber"
    write(values, out)
    print(f"{len(values) - len(missing)} numbers -> {out}")
    if missing:
        print(f"no table for {len(missing)}: {', '.join(missing)}")


if __name__ == "__main__":
    main(sys.argv[1:])
