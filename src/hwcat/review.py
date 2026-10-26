"""Reviewer checks on the all-summers summary that `hwcat robustness` wrote.

    python -m hwcat.review --workdir hwcat-data

Reads <workdir>/robustness/catalogue_stability_allyears.csv and reports, with
95% Wilson intervals:
  headline  summers in which one change of setting moves the leader, per
            ranking (own bar, reference bar, normalized)
  margin    does a small lead predict it, and does crowding, tested against
            the same outcome
  season    leaders peaking June-August against the shoulder months
  baseline  summers inside 1991-2020 against later ones

Outputs <workdir>/robustness/review_checks.csv, one row per check.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

from hwcat.workdir import Workdir


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k of n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def checks(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(summary)
    for col in [c for c in summary if c.startswith("leader_moved")]:
        k = int(summary[col].sum())
        lo, hi = wilson(k, n)
        rows.append({"check": col, "k": k, "n": n, "lo": lo, "hi": hi})
    out = "n_settings_moving_leader"
    for predictor in ("lead", "crowding"):
        if summary[predictor].nunique() > 1 and summary[out].nunique() > 1:
            rho, p = spearmanr(summary[predictor], summary[out])
            rows.append({"check": f"spearman_{predictor}", "rho": rho, "p": p,
                         "n": n})
    for name, group in (("season_jja", summary.leader_peak_month.isin([6, 7, 8])),
                        ("in_baseline", summary.in_baseline.astype(bool))):
        a = summary.leader_moved[group]
        b = summary.leader_moved[~group]
        _, p = fisher_exact([[a.sum(), len(a) - a.sum()],
                             [b.sum(), len(b) - b.sum()]])
        rows.append({"check": f"{name}_vs_rest", "k": int(a.sum()), "n": len(a),
                     "k_rest": int(b.sum()), "n_rest": len(b), "p": p})
    return pd.DataFrame(rows)


def review(workdir: Workdir) -> pd.DataFrame:
    summary = pd.read_csv(workdir.robustness / "catalogue_stability_allyears.csv")
    table = checks(summary)
    out = workdir.robustness / "review_checks.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"-> {out}")
    return table


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workdir", type=Path, default=Path("hwcat-data"))
    ap.add_argument("--region-name", default="europe")
    a = ap.parse_args(argv)
    review(Workdir(a.workdir, a.region_name))


if __name__ == "__main__":
    main()
