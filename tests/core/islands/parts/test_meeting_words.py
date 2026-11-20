"""The island table's meeting words (km::team_meet): after the children, their number
and then a word a rank, each 0 at the load; a rank writes a meeting's marker into its
word of the leader's. Two words follow them, the collectives' generation pages' DRAM
address and page bytes, then at the next 32 B the core's generation row."""

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

    class Words(np.ndarray):
        address, aligned_page_bytes = 0x5000, 64

    session = SimpleNamespace(
        grid=(2, 1), tensor=lambda words, *a, **k: words.view(Words)
    )
    self = SimpleNamespace(
        graph=graph,
        split=1,
        sessions=[session, session],
        _coll_gen={},
        _far_tiles=lambda island: [np.zeros(2, np.uint32), np.zeros(2, np.uint32)],
    )
    words = link_plan.LinkPlanMixin._table_of(self, session)
    # after 6 x 2 words their number, 5, then the 5 words: 0 (no marker is 0)
    assert words[:, 12].tolist() == [5, 5]
    assert not words[:, 13:18].any()
    # the generation pages' address and page bytes, the row (0 until a leader reads)
    assert words[:, 18:20].tolist() == [[0x5000, 64]] * 2
    assert not words[:, 20:].any()
    # in DRAM, a page a core: generation 1, no mark, led under none
    assert self._coll_gen[0].tolist() == [[1] + [0] * 15] * 2
    # 20 words, up to 8 to align the row (its address at 32 B), its 8: 16 at a time
    assert words.shape == (2, 48)
