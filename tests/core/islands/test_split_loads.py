"""A card split into islands, and two cards (no card): the load remembers the kernel
buffer, opens the cards again when the kernels need another (as a card's own session
does) unless what it holds would be lost, and a class's rect= is each island's own."""

import inspect
from types import SimpleNamespace

import pytest

from komorebi import layout
from komorebi.device import kernel_buffer, mock, native
from komorebi.dsl.program import island_rectangles
from komorebi.errors import KernelSizeError
from komorebi.islands.session import IslandSession
from komorebi.program.description import KernelSizes, Program
from komorebi.session.parts.load import LoadMixin
from komorebi.session.run import Run
from komorebi.session.session import Session


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
    made.arch = "blackhole"
    card = SimpleNamespace(_last_total=0, kernel_bytes=163840, _worker_l1=0)
    card._kept = SimpleNamespace(stop=lambda: made.said.append("kept stopped"))
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
    made.arch = "blackhole"
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


def test_a_split_cards_load_logs_its_kernels_as_a_cards_load():
    lines = []
    program = SimpleNamespace(
        sizes=KernelSizes({"BRISC": 50_000, "NCRISC": 60_000}, 164_992, 163_840)
    )
    LoadMixin._log_kernel_buffer(SimpleNamespace(log=lines.append), program)
    assert lines == [
        "kernel buffer: 164992 of 163840 B (text: BRISC 50000, NCRISC 60000)"
    ]
    for load in (LoadMixin._load, LoadMixin.load_islands):  # a card's, a split card's
        assert "self._log_kernel_buffer(program)" in inspect.getsource(load)


def test_a_split_card_whose_kernels_need_more_opens_its_card_again(monkeypatch):
    loads = []

    def load_islands(card):
        loads.append(card.kernel_bytes)
        if len(loads) == 1:
            card._last_total = 164976
            raise kernel_buffer.Resize(176576, "the kernels take 164,976 B")
        return [object(), object()]

    def reopen(session, size, worker_l1, why):
        session._worker_l1 = worker_l1
        made.said.append(("reopen", size, why))

    monkeypatch.setattr(mock, "active", lambda: True)
    monkeypatch.setattr(kernel_buffer, "reopen", reopen)
    made = split_session(load_islands)
    made.load(Kernel())
    assert loads == [163840, 176576]  # loaded again at the size the kernels named
    assert ("reopen", 176576, "the kernels take 164,976 B") in made.said
    assert made.kernel_bytes == made.sessions[0].kernel_bytes == 176576
    assert made.said[-1] == "kept stopped"  # a run loaded: no reopen after it
    assert made._memory.said == [("loaded", "swiglu", 164976)]


def resizing(followed):
    """What LoadMixin._resize reads of a card's session opened at 163,840 B."""
    return SimpleNamespace(
        kernel_bytes=163840,
        kernel_followed=followed,
        _own_device=True,
        island=None,
        stuck=False,
        _kept=SimpleNamespace(refused=lambda: None),
        _memory=None,
        _most=lambda: 1531712,  # a P150's largest kernel buffer
        l1=lambda: (163840, 0),
    )


def test_a_followed_kernel_bytes_resizes_and_a_given_one_holds():
    with pytest.raises(kernel_buffer.Resize) as resize:
        LoadMixin._resize(resizing(True), 164976)
    assert resize.value.size == kernel_buffer.need(164976)
    refused = LoadMixin._resize(resizing(False), 164976)
    assert refused.startswith("the session was given kernel_bytes; ")


class Module:
    """What _not_reopened reads of the native module: each device's kernel buffer
    (bytes, L1 left, L1 size) and its lowest L1 buffer (0: none)."""

    def __init__(self, lowest):
        self.lowest = lowest

    def kernel_buffer(self, device):
        return 163840, 1310720 - 0x2B000, 1310720  # the buffers' base at 0x2b000

    def lowest_l1(self, device):
        return self.lowest


def cards_session(lowest=0, refused=None, followed=True):
    made = IslandSession.__new__(IslandSession)
    made.arch = "blackhole"
    island = SimpleNamespace(
        device=object(), _kept=SimpleNamespace(refused=lambda: refused)
    )
    made.sessions, made.cards, made.tree = (island, island), (0, 1), None
    made._kernel_followed, made.said = followed, []
    made.log = made.said.append
    return made, Module(lowest)


def over(total):
    return KernelSizeError(
        "two", KernelSizes({"reader": 90000}, total, 163840), 1 << 20
    )


def test_two_cards_whose_kernels_need_more_open_the_cards_again(monkeypatch):
    made, module = cards_session(lowest=0x17FF00)  # an L1 tensor at the top
    monkeypatch.setattr(native, "load_module", lambda tree=None: module)
    with pytest.raises(kernel_buffer.Resize) as resize:
        made._follow(over(204784))
    assert resize.value.size == kernel_buffer.need(204784)
    made._kernel_followed = False  # kernel_bytes= given: a pin
    assert made._follow(over(204784)) is None and made.said == []


@pytest.mark.parametrize(
    "lowest, refused, total, why",
    [
        # 0x2b000 + 205,824 - 163,840: the base at the larger buffer is 0x35400
        (0x33000, None, 204784, "island 0: an L1 buffer at 0x33000 is under"),
        (0, "a run was loaded on it", 204784, "island 0: a run was loaded on it"),
        (0, None, 2_000_000, "is over the most"),
    ],
)
def test_two_cards_are_not_opened_again_under_what_would_be_lost(
    monkeypatch, lowest, refused, total, why
):
    made, module = cards_session(lowest, refused)
    monkeypatch.setattr(native, "load_module", lambda tree=None: module)
    assert made._follow(over(total)) is None  # the refusal stands: remembered
    assert made.said[0].startswith(
        "kernel buffer: the cards are not opened again in the run: "
    )
    assert why in made.said[0]


class Device:
    def close(self):
        pass


def test_two_cards_opened_again_map_their_channels_again(monkeypatch):
    monkeypatch.setattr(mock, "active", lambda: True)
    islands = [Session(card=card, device=Device()) for card in (0, 1)]
    for island in islands:
        island._map_channel()  # the first transfer: mapped (the mock's, for it)
        assert island._mock_mapped()
    mesh = SimpleNamespace(close=lambda: None, device=lambda island: Device())
    module = SimpleNamespace(Islands=lambda count, worker_l1, packet: mesh)
    monkeypatch.setattr(native, "load_module", lambda tree=None: module)
    made = IslandSession.__new__(IslandSession)
    made.arch = "blackhole"
    made.sessions, made.cards, made.tree, made.links = tuple(islands), (0, 1), None, 0
    made.mesh, made.packet, made.kernel_bytes, made.said = mesh, 8192, 163840, []
    made.log = made.said.append
    made._cards_reopen(kernel_buffer.Resize(205824, "the kernels take 204,784 B"))
    for island in islands:
        assert not island._mock_mapped()  # a new device: not mapped yet
        island._map_channel()  # the rewrite's (or the load's) first transfer
        assert island._mock_mapped()  # mapped again, as the card's must be
