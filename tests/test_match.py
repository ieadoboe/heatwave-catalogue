"""The shared-cell event matcher on hand-made tables.

    python tests/test_match.py
"""
import pandas as pd

from hwcat import match as S


def fp(cells_by_event):
    """Footprint table from {event_id: [flat cell indices]}."""
    rows = [(c, e) for e, cells in cells_by_event.items() for c in cells]
    return pd.DataFrame(rows, columns=["cell", "event_id"])


def events(ids):
    return pd.DataFrame({"event_id": ids, "rank": range(1, len(ids) + 1)})


def test_match_by_overlap():
    # Reference: big event 1 on cells 0-99, small neighbour 2 on 100-109.
    ref, ref_fp = events([1, 2]), fp({1: range(100), 2: range(100, 110)})

    # Stricter setting: event 1 shrinks, event 2 vanishes. Dates and centres
    # would have matched 2 to 1's counterpart and called it a merge.
    m = S.match_by_overlap(ref, ref_fp, events([7]), fp({7: range(20, 60)}))
    assert list(m.run_index) == [0, -1] and not m.merged.any(), m
    assert m.share[0] == 0.4, m

    # Looser setting: one run event covers both, a real merge.
    m = S.match_by_overlap(ref, ref_fp, events([3]), fp({3: range(0, 110)}))
    assert list(m.run_index) == [0, 0] and m.merged.all(), m

    # Event 1 splits in two; the larger piece is its match.
    m = S.match_by_overlap(ref, ref_fp, events([4, 5]),
                           fp({4: range(0, 30), 5: range(30, 100)}))
    assert m.run_index[0] == 1 and m.n_pieces[0] == 2, m

    # A setting that found nothing.
    empty = pd.DataFrame(columns=["rank", "start", "peak", "end"])
    m = S.match_by_overlap(ref, ref_fp, empty, fp({}))
    assert (m.run_index == -1).all() and (m.share == 0).all(), m


def test_min_overlap():
    ref, ref_fp = events([1, 2]), fp({1: range(100), 2: range(100, 110)})
    # Only 40 of event 1's 100 cell-days survive, but the run event lies
    # wholly inside it: it shrank, as under a stricter percentile, so it
    # still matches.
    run, run_fp = events([7]), fp({7: range(20, 60)})
    m = S.match_by_overlap(ref, ref_fp, run, run_fp, min_overlap=0.5)
    assert m.run_index[0] == 0 and m.n_pieces[0] == 1, m
    assert S.leader_outcome(m, run) == "held"

    # A run event that holds 40 of event 1's cell-days but is mostly
    # elsewhere (60 more cells outside): event 1 vanished into a neighbour.
    run, run_fp = events([7]), fp({7: list(range(20, 60)) + list(range(200, 260))})
    m = S.match_by_overlap(ref, ref_fp, run, run_fp, min_overlap=0.5)
    assert m.run_index[0] == -1 and m.n_pieces[0] == 1, m
    assert S.leader_outcome(m, run) == "vanished"

    # Three equal pieces wholly inside event 1: the largest is its match,
    # and n_pieces records the split.
    run = events([4, 5, 6])
    run_fp = fp({4: range(0, 33), 5: range(33, 66), 6: range(66, 100)})
    m = S.match_by_overlap(ref, ref_fp, run, run_fp, min_overlap=0.5)
    assert m.run_index[0] >= 0 and m.n_pieces[0] == 3, m

    # Three pieces each mostly outside event 1: a split with no piece its own.
    run_fp = fp({4: list(range(0, 33)) + list(range(300, 340)),
                 5: list(range(33, 66)) + list(range(400, 440)),
                 6: list(range(66, 100)) + list(range(500, 540))})
    m = S.match_by_overlap(ref, ref_fp, run, run_fp, min_overlap=0.5)
    assert m.run_index[0] == -1 and m.n_pieces[0] == 3, m
    assert S.leader_outcome(m, run) == "split"

    # A tie between two run events goes to the lower id, whatever their order.
    run = events([9, 8])
    run_fp = fp({9: range(0, 50), 8: range(50, 100)})
    m = S.match_by_overlap(ref, ref_fp, run, run_fp)
    assert run.event_id[m.run_index[0]] == 8, m


def test_leader_outcome():
    ref = events([1, 2, 3])
    ref_fp = fp({1: range(100), 2: range(100, 150), 3: range(150, 170)})
    # Unchanged.
    m = S.match_by_overlap(ref, ref_fp, ref, ref_fp)
    assert S.leader_outcome(m, ref) == "held"
    # Event 2 grows past event 1.
    run = pd.DataFrame({"event_id": [1, 2, 3], "rank": [2, 1, 3]})
    m = S.match_by_overlap(ref, ref_fp, run, ref_fp)
    assert S.leader_outcome(m, run) == "replaced"
    # Events 1 and 2 join: rank 1 holds, but the case changed.
    run, run_fp = events([5, 6]), fp({5: range(150), 6: range(150, 170)})
    m = S.match_by_overlap(ref, ref_fp, run, run_fp)
    assert S.leader_outcome(m, run) == "absorbed"
    # Joining an event outside the top `top` does not count.
    assert S.leader_outcome(m, run, top=1) == "held"


if __name__ == "__main__":
    test_match_by_overlap()
    test_min_overlap()
    test_leader_outcome()
    print("ok")
