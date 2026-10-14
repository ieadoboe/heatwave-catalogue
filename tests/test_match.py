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
    assert (m.run_index == -1).all(), m


if __name__ == "__main__":
    test_match_by_overlap()
    print("ok")
