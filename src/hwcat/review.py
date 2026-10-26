"""Reviewer checks on what `hwcat robustness` wrote.

    python -m hwcat.review --workdir hwcat-data
    python -m hwcat.review --workdir hwcat-data --validate hwcat-data/canada \\
        --validate-region canada

Reads <workdir>/robustness/catalogue_stability_allyears.csv and the
per-summer stability_<region>_<year>.csv tables, and writes, under
<workdir>/robustness/:

  review_checks.csv     headline counts with 95% Wilson intervals; Spearman
                        of lead and of crowding against the same outcome;
                        Fisher tests for June-August leaders, summers inside
                        the baseline and leaders touching the domain edge
  review_settings.csv   per setting: summers in which the leader moved, its
                        interval, and the null model's expected count
  review_agreement.csv  per summer: the share of plausible settings in which
                        the reference leader is still number 1
  review_lead_rule.csv  logistic fit of "leader moved" on log lead, the lead
                        at which a move becomes even odds, and its skill on
                        another region (--validate)

The null model asks how often each setting would move the leader if it only
added symmetric noise to the log severity gap between numbers 1 and 2, with
the noise as large as the gap changes that setting actually produced. A
setting moving the leader far more often than that favours a different kind
of event; one moving it about as often is reshuffling near-ties.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import fisher_exact, norm, spearmanr

from hwcat.workdir import Workdir

MOVED = ("replaced", "vanished")
# Settings a careful author might pick: one change from the reference, or
# the percentile x days grid. Raw pooling is the bias, not a choice.
PLAUSIBLE = ("min_area_km2", "percentile", "window_days", "min_days",
             "connectivity", "grid")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k of n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def fisher_row(name: str, moved: pd.Series, group: pd.Series) -> dict:
    a, b = moved[group], moved[~group]
    _, p = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
    return {"check": name, "k": int(a.sum()), "n": len(a),
            "k_rest": int(b.sum()), "n_rest": len(b), "p": p}


def checks(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(summary)
    for col in [c for c in summary if c.startswith(("leader_moved", "moved_",
                                                     "leader_absorbed",
                                                     "leader_split"))]:
        k = int(summary[col].sum())
        lo, hi = wilson(k, n)
        rows.append({"check": col, "k": k, "n": n, "lo": lo, "hi": hi})
    out = "n_settings_moving_leader"
    for predictor in ("lead", "crowding"):
        if summary[predictor].nunique() > 1 and summary[out].nunique() > 1:
            rho, p = spearmanr(summary[predictor], summary[out])
            rows.append({"check": f"spearman_{predictor}", "rho": rho, "p": p,
                         "n": n})
    moved = summary.leader_moved.astype(bool)
    rows.append(fisher_row("season_jja_vs_rest", moved,
                           summary.leader_peak_month.isin([6, 7, 8])))
    rows.append(fisher_row("in_baseline_vs_rest", moved,
                           summary.in_baseline.astype(bool)))
    if "leader_touches_edge" in summary:
        rows.append(fisher_row("edge_vs_rest", moved,
                               summary.leader_touches_edge.astype(bool)))
    return pd.DataFrame(rows)


def null_model(table: pd.DataFrame, n_sim: int = 20_000,
               seed: int = 0) -> pd.DataFrame:
    """Per setting: observed leader moves against symmetric noise of the
    same size as that setting's changes to the log 1-2 gap.

    expected sums, over summers, the chance that N(0, sigma) noise exceeds
    the summer's own gap; lo and hi bound 95% of simulated counts."""
    rng = np.random.default_rng(seed)
    t = table[table.metric == "severity"]
    rows = []
    for (family, value), g in t.groupby(["family", "value"], sort=False):
        moved = g.leader.isin(MOVED).to_numpy()
        gap = g.gap_ref.to_numpy()
        d = (g.gap_run - g.gap_ref).dropna().to_numpy()
        sigma = float(np.std(d)) if d.size > 1 else np.nan
        ok = np.isfinite(gap)
        p = norm.sf(gap[ok] / sigma) if sigma > 0 else np.zeros(ok.sum())
        sims = (rng.random((n_sim, p.size)) < p).sum(axis=1)
        k, n = int(moved.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"family": family, "value": value, "n": n, "moved": k,
                     "lo": lo, "hi": hi, "sigma_log_gap": sigma,
                     "expected": float(p.sum()),
                     "expected_lo": float(np.quantile(sims, 0.025)),
                     "expected_hi": float(np.quantile(sims, 0.975)),
                     "p_at_least": float((sims >= k).mean())})
    return pd.DataFrame(rows)


def agreement(table: pd.DataFrame) -> pd.DataFrame:
    """Per summer and ranking: in what share of plausible settings is the
    reference leader still number 1 (held or absorbed), and which reference
    rank most often takes its place."""
    t = table[table.family.isin(PLAUSIBLE)]
    # The grid repeats some single changes (90th with 4-6 days, 3 days with
    # higher percentiles): count each distinct setting once.
    setting = [c for c in ("area", "pct", "window", "days", "cycle", "conn")
               if c in t]
    if setting:
        t = t.drop_duplicates(["year", "metric", *setting])
    rows = []
    for (year, metric), g in t.groupby(["year", "metric"]):
        held = g.leader.isin(["held", "absorbed"])
        others = g.winner_ref_rank[~held]
        rows.append({"year": year, "metric": metric, "n_settings": len(g),
                     "leader_share": float(held.mean()),
                     "usual_rival": int(others.mode().iloc[0]) if len(others) else 0})
    return pd.DataFrame(rows)


def fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Maximum-likelihood intercept and slope of P(y) = 1/(1+exp(-(a+bx)))."""
    def nll(beta):
        z = beta[0] + beta[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)
    return minimize(nll, np.zeros(2), method="BFGS").x


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Probability a moved summer scores above a stable one (ties half)."""
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return np.nan
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def lead_rule(summary: pd.DataFrame, validate: pd.DataFrame | None = None
              ) -> pd.DataFrame:
    """P(leader moved) against log lead, fitted on `summary`, and its skill
    on `validate` (another region) when given."""
    def xy(s):
        s = s[s.lead > 0]
        return np.log(s.lead.to_numpy()), s.leader_moved.astype(int).to_numpy()
    x, y = xy(summary)
    a, b = fit_logistic(x, y)
    row = {"intercept": a, "slope": b, "n_fit": len(x),
           "lead_at_even_odds": float(np.exp(-a / b)) if b else np.nan,
           "auc_fit": auc(-x, y)}
    if validate is not None:
        xv, yv = xy(validate)
        pv = 1 / (1 + np.exp(-(a + b * xv)))
        row.update(n_validate=len(xv), auc_validate=auc(-xv, yv),
                   brier_validate=float(np.mean((pv - yv) ** 2)),
                   brier_climatology=float(np.mean((yv.mean() - yv) ** 2)))
    return pd.DataFrame([row])


def load(workdir: Workdir) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(workdir.robustness / "catalogue_stability_allyears.csv")
    tables = [pd.read_csv(workdir.robustness /
                          f"stability_{workdir.region}_{y}.csv")
              for y in summary.year]
    return summary, pd.concat(tables, ignore_index=True)


def review(workdir: Workdir, validate: Workdir | None = None) -> dict:
    summary, table = load(workdir)
    other = load(validate)[0] if validate is not None else None
    out = {"checks": checks(summary), "settings": null_model(table),
           "agreement": agreement(table), "lead_rule": lead_rule(summary, other)}
    for name, df in out.items():
        path = workdir.robustness / f"review_{name}.csv"
        df.to_csv(path, index=False)
        print(f"\n{name}  -> {path}")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workdir", type=Path, default=Path("hwcat-data"))
    ap.add_argument("--region-name", default="europe")
    ap.add_argument("--validate", type=Path, default=None,
                    help="Another region's working folder, to test the lead rule on")
    ap.add_argument("--validate-region", default="canada")
    a = ap.parse_args(argv)
    review(Workdir(a.workdir, a.region_name),
           Workdir(a.validate, a.validate_region) if a.validate else None)


if __name__ == "__main__":
    main()
