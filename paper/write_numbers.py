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

    def pfull(x):
        """$p = 0.03$, or $p < 0.001$: the relation travels with the value."""
        return r"$p < 0.001$" if x < 0.001 else f"$p = {x:.2g}$"

    def moved_counts(s, tail):
        # Summers moved under one ranking but not the other, and an exact
        # McNemar test of the two rankings' move rates.
        from scipy.stats import binomtest
        a = s.leader_moved.astype(bool)
        b = s.leader_moved_severity_norm.astype(bool)
        n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
        out[f"NNormOnly{tail}"], out[f"NSevOnly{tail}"] = str(n01), str(n10)
        if n01 + n10:
            out[f"PNormMcNemar{tail}"] = pfull(binomtest(n01, n01 + n10).pvalue)

    if s is not None and {"lead", "leader_moved_severity_norm"} <= set(s):
        moved = s.leader_moved.astype(bool)
        big = s.lead >= 0.25
        out["NMovedBigLead"] = str(int((moved & big).sum()))
        out["NStableSmallLead"] = str(int((~moved & ~big).sum()))
        out["MaxMovedLead"] = pct(s.lead[moved].max())
        moved_counts(s, "")

    if c is not None and "held_all_mild" in c.index:
        out["NHeldAllMild"] = str(int(c.loc["held_all_mild", "k"]))

    v = read("review_settings.csv")
    if v is not None:
        row = v.loc[v.p_at_least.idxmin()]
        out["NMovedNullMin"] = str(int(row.moved))
        out["ExpectedNullMin"] = f"{row.expected:.0f}"
        out["PNullMinRaw"] = pfull(row.p_at_least)
        if "p_holm" in v:
            out["NSettingsAboveNull"] = str(int((v.p_holm < 0.05).sum()))
            out["PNullMinHolm"] = pfull(v.p_holm.min())
        out["NSettingValues"] = str(len(v))
        # The strict end of each range against the largest count among the
        # milder values (single changes only).
        def moved(fam, val):
            r = v[(v.family == fam)
                  & (pd.to_numeric(v.value, errors="coerce") == val)]
            return int(r.moved.iloc[0]) if len(r) else None
        for name, fam, val in (("NMovedStrictPct", "percentile", 0.95),
                               ("NMovedStrictArea", "min_area_km2", 625_000),
                               ("NMovedStrictDays", "min_days", 6),
                               ("NMovedDaysFour", "min_days", 4),
                               ("NMovedDaysFive", "min_days", 5),
                               ("NMovedMildPct", "percentile", 0.925),
                               ("NMovedMildArea", "min_area_km2", 250_000),
                               ("NMovedWindowLong", "window_days", 31)):
            if moved(fam, val) is not None:
                out[name] = str(moved(fam, val))
        mild = v[((v.family == "percentile") & (v.value.astype(str) == "0.925"))
                 | ((v.family == "min_days") & (v.value.astype(str) == "4"))
                 | ((v.family == "min_area_km2")
                    & (v.value.astype(str) != "625000"))
                 | v.family.isin(["window_days", "connectivity"])]
        if len(mild):
            out["NMovedMildMax"] = str(int(mild.moved.max()))

    # Canada: the same tables, copied to paper/results/ with a canada_ prefix.
    k = read("canada_catalogue_stability_allyears.csv")
    if k is not None:
        for col, name in (("leader_moved", "NLeaderMovedCanada"),
                          ("leader_moved_severity_ref", "NLeaderMovedRefBarCanada"),
                          ("leader_moved_severity_norm", "NLeaderMovedNormCanada"),
                          ("moved_percentile", "NMovedPercentileCanada"),
                          ("moved_min_area_km2", "NMovedAreaCanada"),
                          ("moved_window_days", "NMovedWindowCanada"),
                          ("moved_min_days", "NMovedDaysCanada"),
                          ("moved_connectivity", "NMovedConnectivityCanada"),
                          ("moved_window_raw", "NMovedRawWindowCanada"),
                          ("moved_grid", "NMovedGridCanada")):
            if col in k:
                out[name] = str(int(k[col].sum()))
        if "leader_moved_severity_norm" in k:
            moved_counts(k, "Canada")
    kc = read("canada_review_checks.csv")
    if kc is not None:
        kc = kc.set_index("check")
        if "held_all_mild" in kc.index:
            out["NHeldAllMildCanada"] = str(int(kc.loc["held_all_mild", "k"]))

    # Share of each leader's area-days on cells where the domain cuts events.
    for region, tail in (("europe", ""), ("canada", "Canada")):
        e = read(f"edge_share_{region}.csv")
        if e is not None:
            out[f"EdgeShareMedian{tail}"] = f"{e.share_pct.median():.1f}"
            out[f"EdgeShareMax{tail}"] = f"{e.share_pct.max():.1f}"

    cv = read("consensus_validation.csv")
    ce = read("consensus_europe_setups.csv")
    cc = read("consensus_canada_setups.csv")
    if cv is not None:
        out["RhoConsensus"] = f"{cv.spearman_rho.iloc[0]:.2f}"
        out["PConsensus"] = pfull(cv.spearman_p.iloc[0])
        out["NCandidates"] = str(int(cv.n_candidates.iloc[0]))
    if ce is not None:
        ref = ce[ce.is_reference].iloc[0]
        top = ce.sort_values("score", ascending=False).iloc[0]
        out["RefAgree"], out["RefAgreeOf"] = str(int(ref.n_agree)), str(int(ref.n_summers))
        out["RefRank"] = str(int(ref["rank"]))
        out["TopAgree"], out["TopAgreeOf"] = str(int(top.n_agree)), str(int(top.n_summers))
    if ce is not None and cc is not None:
        # The regions' agreement without the strictest values (625,000 km2,
        # six days), to show how much of it those setups carry.
        from scipy.stats import spearmanr
        key = ["area", "pct", "window", "days", "conn"]
        both = ce.merge(cc, on=key, suffixes=("_e", "_c"))
        mild = both[(both.area < 625_000) & (both.days < 6)]
        rho, pv = spearmanr(mild.score_e, mild.score_c)
        out["RhoConsensusMild"] = f"{rho:.2f}"
        out["PConsensusMild"] = pfull(pv)
        out["NCandidatesMild"] = str(len(mild))
        out["RefRankCanada"] = str(int(cc[cc.is_reference].iloc[0]["rank"]))
    vs = read("consensus_europe_versus.csv")
    if vs is not None:
        out["NTopOnly"], out["NRefOnly"] = str(int(vs.top_only.iloc[0])), str(int(vs.ref_only.iloc[0]))
        out["PTopVsRef"] = pfull(vs.p.iloc[0])
    cs = read("consensus_europe_summers.csv")
    if cs is not None:
        out["NNoConsensus"] = str(int((~cs.consensus.astype(bool)).sum()))

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
