"""The island table's meeting words (km::team_meet): after the children, their number
(the most ranks an island of the load) and then a word a rank, each 0 at the load. A
rank writes a meeting's marker into its word of the leader's, so a collective given up
before its leader met leaves nothing a later meeting counts."""

from types import SimpleNamespace

import numpy as np

from komorebi.islands.parts import link_plan


def test_the_island_table_holds_a_meeting_word_a_rank_of_its_largest_island():
    graph = SimpleNamespace(
        islands=[SimpleNamespace(cores=range(2))] * 2,
        table=lambda island: np.zeros(4, np.uint32).tobytes(),
        children=lambda island: np.zeros(2, np.uint32),
        count=lambda island: (3, 5)[island],
    )
    session = SimpleNamespace(grid=(2, 1), tensor=lambda words, *a, **k: words)
    self = SimpleNamespace(
        graph=graph,
        split=1,
        sessions=[session, session],
        _far_tiles=lambda island: [np.zeros(2, np.uint32), np.zeros(2, np.uint32)],
    )
    words = link_plan.LinkPlanMixin._table_of(self, session)
    # after 6 x 2 words their number, 5, then the 5 words: 0 (no marker is 0)
    assert words[:, 12].tolist() == [5, 5]
    assert not words[:, 13:].any()
    assert words.shape == (2, 32)  # 18 words, 16 at a time
