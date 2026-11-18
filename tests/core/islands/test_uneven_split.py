"""A split over a group spanning islands whose islands' steps are not in proportion to
their ranks (no card): named at the tensor's making, by a session log line and a
warning, with the steps a command splitting the group by island would read across."""

from types import SimpleNamespace

import pytest

from komorebi.islands import graph as islands
from komorebi.islands.graph import Island, Link
from komorebi.islands.parts.entries import EntriesMixin

TWO = islands.Graph([Island(tuple(range(120)), 12)] * 2, [Link(0, 1)])


def test_a_split_of_unequal_shares_names_each_island_and_the_steps_across():
    ranks = TWO.ranks  # 168 tile columns over every rank: the first 168 take one each
    placement = islands.place(TWO, (32, 168 * 32), "tile", 1, (0, ranks))
    shares, across = islands.uneven_split(TWO, placement)
    half = TWO.count(0)
    assert shares == [(half, half), (168 - half, TWO.count(1))]
    assert across == half - 84  # by island 84 steps each: island 0's extra ones


def test_shares_within_a_step_of_the_ranks_are_not_named():
    edge = TWO.first(1)
    each = [
        islands.place(TWO, (32, 168 * 32), "tile", 1, (edge - 84, 168)),  # centred
        islands.place(TWO, (TWO.ranks * 32, 32), "tile", 0, (0, TWO.ranks)),
        islands.place(TWO, ((TWO.ranks + 1) * 32, 32), "tile", 0, (0, TWO.ranks)),
        islands.place(TWO, (168 * 32, 32), "tile", 0, (0, TWO.count(0))),  # one island
        islands.place(TWO, (168 * 32, 32), "tile", copy=True),
    ]
    assert [islands.uneven_split(TWO, placement) for placement in each] == [None] * 5


class Entries(EntriesMixin):
    """An island session's tensor entries over TWO, no card: parts are stand-ins."""

    def __init__(self):
        self.graph, self.card_dram, self.lines = TWO, False, []
        self.sessions, self._entries = [None, None], [None] * 4
        self._tables = []  # no load's table yet
        self.log = self.lines.append

    def _part(self, session, island, placement, shape, dtype, layout):
        return SimpleNamespace(
            address=1 << 21, page_bytes=2048, aligned_page_bytes=2048, free=lambda: None
        )

    def _table_row(self, word):
        pass


def test_empty_logs_and_warns_once_for_unequal_shares_and_not_for_equal_ones():
    entries = Entries()
    with pytest.warns(UserWarning, match=r"split=1 over group=\(0, 2\d\d\)"):
        entries.empty((32, 168 * 32), "bfloat16", "tile", split=1)
    (line,) = entries.lines
    assert "not in proportion to their ranks" in line and "steps (" in line
    entries.empty((32, 168 * 32), "bfloat16", "tile", 1, (TWO.first(1) - 84, 168))
    assert len(entries.lines) == 1
