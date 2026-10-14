"""Both event matchers on hand-made tables.

    python tests/test_match.py
"""
import pandas as pd

from hwcat import match as S, sensitivity


def frame(rows):
    df = pd.DataFrame(rows, columns=["start", "end", "centroid_lat",
                                     "centroid_lon"])
    df[["start", "end"]] = df[["start", "end"]].apply(pd.to_datetime)
    return df


def test_match_events():
    ref = frame([("2026-06-12", "2026-07-03", 48.9, 13.6),   # central Europe
                 ("2026-05-14", "2026-05-22", 57.0, 37.6),   # western Russia
                 ("2026-08-19", "2026-09-05", 41.3, 17.1)])  # Italy
    run = frame([("2026-05-15", "2026-05-20", 56.5, 36.0),   # Russia, shorter
                 ("2026-06-10", "2026-06-20", 49.5, 12.0),   # first half of #1
                 ("2026-06-21", "2026-07-05", 48.0, 15.0),   # second half of #1
                 ("2026-08-20", "2026-09-01", 34.0, 5.0)])   # same dates, Algeria
    hit = S.match_events(ref, run).tolist()
    assert hit[0] == 2, hit        # longer overlap wins over nearer centroid
    assert hit[1] == 0, hit
    assert hit[2] == -1, hit       # right dates but 1,200 km away: no match
    assert sensitivity.tag_for(25_000, 0.95) == "_a25k_p95"
    print("ok")


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
    test_match_events()
    test_match_by_overlap()
