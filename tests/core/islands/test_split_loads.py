"""A card split into islands (no card): its load remembers the kernel buffer as a
session over cards does, and a class's rect= is each island's own."""

from types import SimpleNamespace

import pytest

from komorebi import layout
from komorebi.device import mock
from komorebi.dsl.program import island_rectangles
from komorebi.errors import KernelSizeError
from komorebi.islands.session import IslandSession
from komorebi.program.description import KernelSizes, Program
from komorebi.session.run import Run


class Kernel(Program):
    """A kernel-form Program with nothing to build (the split load's checks)."""

    def __init__(self):
        self.name = "swiglu"
        self.defines = {}

    def form(self):
        return "kernel"


class Memory:
    def __init__(self):
        self.said = []

    def wanted(self, total, smaller=False):
        self.said.append(("wanted", total, smaller))

    def loaded(self, program, total):
        self.said.append(("loaded", program.name, total))


def split_session(load_islands):
    made = IslandSession.__new__(IslandSession)
    card = SimpleNamespace(_last_total=0)
    card.load_islands = lambda program, graph, **options: load_islands(card)
    made.sessions, made.split, made.links, made.card_dram = (card, card), 2, 0, True
    made.run, made.graph, made._memory = None, None, Memory()
    made.kernel_bytes, made.said = 163840, []
    made.log = made.said.append
    return made


def test_a_split_card_s_refused_load_is_remembered_for_the_next_run():
    def refused(card):
        card._last_total = 207808
        sizes = KernelSizes({"reader": 90000}, 207808, 163840)
        raise KernelSizeError("swiglu", sizes, 1 << 20)

    made = split_session(refused)
    with pytest.raises(KernelSizeError):
        made.load(Kernel())
    assert made._memory.said == [("wanted", 207808, False)]
    assert made.said[-1].startswith(
        "kernel buffer: the kernels take 207,808 B, this session's buffer is "
        "163,840 B; remembered, the script's next run opens at"
    )


def test_a_split_card_s_load_remembers_what_it_took(monkeypatch):
    def loaded(card):
        card._last_total = 120000
        return [object(), object()]

    monkeypatch.setattr(mock, "active", lambda: True)
    made = split_session(loaded)
    made.load(Kernel())
    assert made._memory.said == [("loaded", "swiglu", 120000)]


def nocs_of(columns=12, rows=10):
    """A grid's NoCs as layout.nocs gives them, enough for the place arguments."""
    noc_xy = {(x, y): (x + 1, y + 2) for y in range(rows) for x in range(columns)}
    return SimpleNamespace(grid=(columns, rows), noc_xy=noc_xy, panels=[], noc0=True)


PLAIN = SimpleNamespace(broadcasts=[])  # a class with self.x / self.y, no broadcast


def test_a_split_card_s_rect_is_each_island_s_own_rows():
    islands = [(0, 5, 59, 0), (5, 5, 59, 59)]
    args, parts = island_rectangles(PLAIN, nocs_of(), islands, (2, 1, 4, 3))
    rect = (2, 1, 4, 3)  # island-local, as on each card of a session over cards
    assert parts == {0: ((14, 28), rect), 60: ((74, 28), rect)}
    assert args(14)[2] == [0, 0, 1, 4, 3]
    assert args(74 + 12 * 2 + 3)[2] == [3, 2, 1, 4, 3]


def test_a_split_card_s_rect_past_an_island_s_ranks_is_refused_by_name():
    islands = [(0, 4, 47, 0), (4, 3, 34, 47), (7, 3, 35, 81)]
    with pytest.raises(ValueError) as refused:
        island_rectangles(PLAIN, nocs_of(), islands, (0, 0, 12, 3))
    assert str(refused.value) == (
        "rect=(0, 0, 12, 3) is past island 1's 34 rank cores (3 rows of 12, its link "
        "cores last): give a rect= that every island's ranks hold"
    )


def split_run(first, base, count):
    program = SimpleNamespace(
        broadcast_lines={"down": "line 101: km.Broadcast(self, along='column')"},
        rectangle={
            "group": (0, 48), "rect": (0, 0, 12, 4),
            "islands": {0: ((0, 48), (0, 0, 12, 4)), 60: ((60, 48), (0, 0, 12, 4))},
        },
    )  # fmt: skip
    return SimpleNamespace(
        programs=[program], island_first=first, island_cores=(base, count),
        ranks=118, island_ranks=count,
    )  # fmt: skip


def test_a_split_card_s_broadcast_is_its_island_s_own_rectangle():
    down = SimpleNamespace(name="down")
    Run._check_rectangle(split_run(0, 0, 59), down, 0, (0, 48))
    Run._check_rectangle(split_run(59, 60, 59), down, 0, (59, 48))
    with pytest.raises(ValueError, match=r"is the group \(59, 48\), fixed at load"):
        Run._check_rectangle(split_run(59, 60, 59), down, 0, (0, 48))
    run = split_run(59, 60, 59)
    run.programs[0].name = "Attention"
    Run._check_load_rectangle(run, run.programs[0], 0, 118)


def test_a_split_card_hands_each_island_s_rows_to_a_class_s_rectangle():
    nocs = nocs_of()
    graph = SimpleNamespace(
        islands=[0, 1],
        rank_cores=[tuple(range(59)), tuple(range(60, 119))],
        link_cores=[(59,), (119,)],
        count=lambda island: 59,
        first=lambda island: 59 * island,
    )
    made = IslandSession.__new__(IslandSession)
    made.split, made.graph = 2, graph
    made.sessions = (SimpleNamespace(grid=(12, 10), nocs=lambda: nocs),) * 2
    program = Kernel()
    program.rectangle = {"rect": None, "nocs": None, "args": None, "group": None}
    made._hand_rectangles(program)
    assert program.rectangle["nocs"] is nocs
    assert program.rectangle["split"] == ((0, 5, 59, 0), (5, 5, 59, 59))


def test_a_split_card_s_broadcast_carries_the_command_s_flat_group():
    """The cores compare a broadcast's group with the command's, which is in flat
    ranks: island 1's (59, 48), not its cores (60, 48)."""
    buffer = SimpleNamespace(index_a=10, block=4)
    down = SimpleNamespace(broadcasts=[("column", buffer, 2, "A", 4, 0)])
    order = layout.core_order((12, 10))
    words = [(x + 1) | (y + 2) << 16 for x, y in order]
    nocs = layout.nocs((12, 10), words, words)
    islands = [(0, 5, 59, 0), (5, 5, 59, 59)]
    args, parts = island_rectangles(down, nocs, islands, (0, 0, 12, 4))
    assert parts == {0: ((0, 48), (0, 0, 12, 4)), 60: ((60, 48), (0, 0, 12, 4))}
    assert args(0)[0][5 + 1] == 0 | 48 << 16  # common words, then the group
    assert args(60)[0][5 + 1] == 59 | 48 << 16
    assert args(60)[0][5 + 4] == 60  # the sender: a core of the card


def test_a_split_card_s_run_names_its_session_so_its_chains_see_card_dram(monkeypatch):
    def loaded(card):
        card._last_total = 120000
        part = SimpleNamespace(program=None, backend=SimpleNamespace(trace_cap=0))
        return [part, SimpleNamespace(**vars(part))]

    monkeypatch.setattr(mock, "active", lambda: False)
    made = split_session(loaded)
    made._regions = {}
    made.graph = SimpleNamespace(ranks=236, count=lambda island: 118)
    run = made.load(Kernel())
    assert run.session is made and run.session.card_dram  # a gather: one DRAM, plain
