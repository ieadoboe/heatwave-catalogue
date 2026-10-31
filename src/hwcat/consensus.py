"""Which detector setup agrees most often on a summer's leading heat wave?

    hwcat consensus --years 1991-2026 --workdir hwcat-data \\
        --validate hwcat-data/canada --validate-region canada

The robustness sweep changes one setting at a time around a reference, so
any "consensus" drawn from it is anchored to that reference. Here every
candidate setup competes on equal terms:

  * The candidates are a Latin-hypercube sample of N_SAMPLE setups across
    the published ranges (percentile, minimum duration, minimum area,
    window with the seasonal cycle removed, connectivity), fixed by SEED so
    every summer and region uses the same set, plus the reference as one
    more candidate.
  * Each run's winner is its rank-1 event. Winners are matched to each other,
    not to any reference catalogue: two are the same event when they share
    at least SAME_EVENT of the smaller one's cell-days.
  * A candidate's score is the share of summers in which its winner is the
    consensus winner of the OTHER candidates (it never votes on itself).
    The consensus needs a strict majority of those votes; a summer without
    one is a genuine near-tie and is left out of that candidate's score.

Europe picks the best setup; Canada, untouched, tests it.

Outputs, under <workdir>/robustness/:
  consensus_<region>_setups.csv    one row per candidate
  consensus_<region>_summers.csv   one row per summer
  consensus_<region>_levels.csv    mean score by value of each setting
  consensus_<region>_versus.csv    the best setup against the reference
  consensus_validation.csv         the European pick, scored on Canada
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hwcat.match import footprint
from hwcat.robustness import REFERENCE, SWEEPS, Setting, run_all, tag_of
from hwcat.workdir import Workdir

N_SAMPLE = 30
SEED = 2026
SAME_EVENT = 0.5
LEVELS = {"pct": SWEEPS["percentile"][1], "days": SWEEPS["min_days"][1],
          "area": SWEEPS["min_area_km2"][1], "window": SWEEPS["window_days"][1],
          "conn": SWEEPS["connectivity"][1]}


def balanced_sample(n: int = N_SAMPLE, seed: int = SEED) -> list[Setting]:
    """n distinct setups, each setting's values used as evenly as n allows
    (a Latin hypercube over discrete levels), never the reference itself."""
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        cols = {k: rng.permutation(np.resize(np.asarray(v), n))
                for k, v in LEVELS.items()}
        out = [REFERENCE._replace(**{k: type(getattr(REFERENCE, k))(cols[k][i].item())
                                     for k in LEVELS})
               for i in range(n)]
        if len(set(out)) == n and REFERENCE not in out:
            return out
    raise RuntimeError("no distinct sample found")


def candidates() -> list[Setting]:
    return [REFERENCE, *balanced_sample()]


def winner_cells(workdir: Workdir, year: int, s: Setting) -> np.ndarray | None:
    """Flat cell-day indices of the setup's rank-1 event, or None if it found
    nothing."""
    tag = tag_of(s)
    cat = pd.read_csv(workdir.catalogue_file(year, tag))
    if cat.empty:
        return None
    fp = footprint(workdir.labels_file(year, tag))
    eid = int(cat.loc[cat["rank"] == 1, "event_id"].iloc[0])
    return np.sort(fp.cell[fp.event_id == eid].to_numpy())


def cluster(winners: list[np.ndarray | None]) -> list[int]:
    """Group winners that are the same event (share SAME_EVENT of the smaller
    one's cell-days), by union-find. -1 for a setup that found nothing."""
    n = len(winners)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = winners[i], winners[j]
            if a is None or b is None:
                continue
            shared = np.intersect1d(a, b, assume_unique=True).size
            if shared >= SAME_EVENT * min(a.size, b.size):
                parent[find(i)] = find(j)
    return [find(i) if winners[i] is not None else -1 for i in range(n)]


def majority(labels: list[int]) -> int | None:
    """The label held by more than half of the votes, else None."""
    votes = [l for l in labels if l >= 0]
    if not votes:
        return None
    vals, counts = np.unique(votes, return_counts=True)
    k = counts.argmax()
    return int(vals[k]) if counts[k] * 2 > len(votes) else None


def score_year(labels: list[int]) -> list[bool | None]:
    """Per candidate: does its winner match the others' majority? None when
    the others have no majority."""
    out = []
    for i, own in enumerate(labels):
        cons = majority(labels[:i] + labels[i + 1:])
        out.append(None if cons is None else own == cons)
    return out


def evaluate(workdir: Workdir, years: list[int],
             cands: list[Setting]) -> dict[str, pd.DataFrame]:
    agree = np.full((len(years), len(cands)), np.nan)
    summers = []
    for y_i, year in enumerate(years):
        labels = cluster([winner_cells(workdir, year, s) for s in cands])
        agree[y_i] = [np.nan if v is None else float(v) for v in score_year(labels)]
        votes = [l for l in labels if l >= 0]
        top = np.unique(votes, return_counts=True)[1].max() if votes else 0
        summers.append({"year": year, "n_votes": len(votes),
                        "top_share": top / max(len(votes), 1),
                        "consensus": majority(labels) is not None,
                        "reference_agrees": agree[y_i, 0] == 1.0})
    setups = pd.DataFrame([s._asdict() for s in cands])
    setups["is_reference"] = [s == REFERENCE for s in cands]
    setups["n_summers"] = np.isfinite(agree).sum(axis=0)
    setups["n_agree"] = np.nansum(agree, axis=0).astype(int)
    setups["score"] = setups.n_agree / setups.n_summers.clip(lower=1)
    setups["rank"] = setups.score.rank(ascending=False, method="min").astype(int)
    levels = pd.concat([setups.groupby(k).score.agg(["mean", "count"])
                        .reset_index().rename(columns={k: "value"})
                        .assign(setting=k) for k in LEVELS], ignore_index=True)
    # The best setup against the reference, summer by summer: an exact
    # McNemar test on the summers where only one of the two agrees.
    from scipy.stats import binomtest
    best = int(setups.score.idxmax())
    ref, top = agree[:, 0], agree[:, best]
    both = np.isfinite(ref) & np.isfinite(top)
    top_only = int(((top == 1) & (ref == 0) & both).sum())
    ref_only = int(((ref == 1) & (top == 0) & both).sum())
    versus = pd.DataFrame([{**cands[best]._asdict(), "top_only": top_only,
                            "ref_only": ref_only,
                            "p": (binomtest(top_only, top_only + ref_only).pvalue
                                  if top_only + ref_only else 1.0)}])
    return {"setups": setups.sort_values("rank"), "summers": pd.DataFrame(summers),
            "levels": levels[["setting", "value", "mean", "count"]],
            "versus": versus}


def validate(europe: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Score the European pick (best score; ties all listed) on the other
    region, and compare the two regions' scores across all candidates."""
    key = list(LEVELS)
    both = europe.merge(other, on=key, suffixes=("_fit", "_test"))
    from scipy.stats import spearmanr
    rho, p = spearmanr(both.score_fit, both.score_test)
    best = both[both.score_fit == both.score_fit.max()]
    return best[key + ["score_fit", "score_test", "rank_test"]].assign(
        n_candidates=len(both), spearman_rho=rho, spearman_p=p)


def write(workdir: Workdir, tables: dict[str, pd.DataFrame]) -> None:
    workdir.robustness.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = workdir.robustness / f"consensus_{workdir.region}_{name}.csv"
        df.to_csv(path, index=False)
        print(f"-> {path}")


def consensus(years: list[int], workdir: Workdir, jobs: int = 3,
              skip_runs: bool = False, other: Workdir | None = None) -> None:
    cands = candidates()
    for wd in [workdir] + ([other] if other else []):
        if not skip_runs:
            run_all(years, wd, jobs, todo=cands)
        tables = evaluate(wd, years, cands)
        write(wd, tables)
        print(tables["setups"].head(10).to_string(index=False))
        if wd is workdir:
            fit = tables["setups"]
        else:
            v = validate(fit, tables["setups"])
            path = workdir.robustness / "consensus_validation.csv"
            v.to_csv(path, index=False)
            print(f"-> {path}\n{v.to_string(index=False)}")
