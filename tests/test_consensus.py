"""The balanced sample and the leave-one-out consensus on hand-made runs.

    python tests/test_consensus.py
"""
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from hwcat import consensus as C, robustness as AY
from hwcat.workdir import Workdir

SHAPE = np.array([10, 10, 10])


def test_balanced_sample():
    s = C.balanced_sample()
    assert len(s) == len(set(s)) == C.N_SAMPLE and AY.REFERENCE not in s
    assert s == C.balanced_sample(), "the sample must be fixed by its seed"
    for k, values in C.LEVELS.items():
        counts = Counter(getattr(x, k) for x in s)
        assert set(counts) == set(values), (k, counts)
        assert max(counts.values()) - min(counts.values()) <= 1, (k, counts)
    assert all(x.cycle for x in s), "raw pooling is a bias, not a candidate"
    assert C.candidates()[0] == AY.REFERENCE and len(C.candidates()) == C.N_SAMPLE + 1


def test_cluster_and_majority():
    a = np.arange(100)
    shrunk = np.arange(20, 60)            # wholly inside a: the same event
    other = np.arange(500, 600)
    touching = np.arange(90, 300)         # shares 10 of 100: a different event
    labels = C.cluster([a, shrunk, other, None, touching])
    assert labels[0] == labels[1] != labels[2] and labels[3] == -1
    assert labels[4] != labels[0]
    assert C.majority([0, 0, 1]) == 0
    assert C.majority([0, 1]) is None                  # a tie: no consensus
    assert C.majority([-1, -1]) is None


def test_score_year_leaves_itself_out():
    # Two votes for A, two for B: each candidate sees a 2-1 majority for the
    # other side when it leaves itself out, so all four disagree.
    assert C.score_year([0, 0, 1, 1]) == [False, False, False, False]
    # Three for A, one for B.
    assert C.score_year([0, 0, 0, 1]) == [True, True, True, False]
    # With only one other voter, that voter is the majority.
    assert C.score_year([0, 1]) == [False, False]


def save(path, ids, cells):
    t, i, j = np.unravel_index(np.asarray(cells, dtype=int), tuple(SHAPE))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=t, i=i, j=j,
                        event_id=np.asarray(ids, dtype=np.int32), shape=SHAPE)


def test_evaluate_and_validate():
    """Events A (cells 0-99), B (200-299) and C (400-499). In 2000 every
    setup ranks A first except the 97.5th-percentile ones, which rank B; in
    2001 the setups split three ways, so there is no majority."""
    cands = C.candidates()
    years = [2000, 2001]
    with tempfile.TemporaryDirectory() as d:
        w = Workdir(Path(d))
        w.labels.mkdir(parents=True)
        for year in years:
            for k, s in enumerate(cands):
                if year == 2000:
                    first = 2 if s.pct == 0.975 else 1
                else:
                    first = k % 3 + 1
                order = [first] + [e for e in (1, 2, 3) if e != first]
                pd.DataFrame({"rank": [1, 2, 3], "event_id": order}).to_csv(
                    w.catalogue_file(year, AY.tag_of(s)), index=False)
                save(w.labels_file(year, AY.tag_of(s)),
                     [1] * 100 + [2] * 100 + [3] * 100,
                     list(range(100)) + list(range(200, 300)) + list(range(400, 500)))
        t = C.evaluate(w, years, cands)
    setups, summers = t["setups"], t["summers"]
    assert list(summers.consensus) == [True, False]
    strict = setups[setups.pct == 0.975]
    assert (strict.score == 0).all() and (setups[setups.pct != 0.975].score == 1).all()
    assert (setups.n_summers <= 2).all()
    lv = t["levels"].set_index(["setting", "value"])["mean"]
    assert lv[("pct", 0.975)] == 0 and lv[("pct", 0.9)] == 1
    v = C.validate(setups, setups)
    assert (v.score_fit == 1).all() and (v.rank_test == 1).all()
    assert abs(v.spearman_rho.iloc[0] - 1) < 1e-9


if __name__ == "__main__":
    test_balanced_sample()
    test_cluster_and_majority()
    test_score_year_leaves_itself_out()
    test_evaluate_and_validate()
    print("ok")
