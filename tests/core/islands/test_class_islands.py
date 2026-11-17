"""A class on an island session (no card): the islands' common grid, a broadcast's
rectangle taken as one island's flat ranks, or one over the islands (taller than
island 0, or the default: one island's in the flat ranks)."""

from types import SimpleNamespace

import pytest

from komorebi import abi, layout
from komorebi.dsl.program import (
    flat_groups,
    flat_rectangle,
    session_nocs,
    spans,
    stacked_rectangle,
)
from komorebi.errors import KomorebiError, ProgramError
from komorebi.islands.session import IslandSession
from komorebi.session.run import Run


def islands_of(*grids):
    made = IslandSession.__new__(IslandSession)
    made.sessions = tuple(SimpleNamespace(grid=grid) for grid in grids)
    return made


def test_an_island_session_s_grid_is_each_island_s():
    assert islands_of((12, 10), (12, 10)).grid == (12, 10)
    with pytest.raises(KomorebiError, match=r"the islands' grids differ"):
        islands_of((12, 10), (11, 10)).grid  # noqa: B018


def island_run(first):
    program = SimpleNamespace(
        broadcast_lines={"grid": "line 79: km.Broadcast(self, along='row')"},
        rectangle={"group": (0, 96), "rect": (0, 0, 12, 8)},
    )
    return SimpleNamespace(
        programs=[program], island_first=first, island_cores=(0, 120), ranks=238,
        island_ranks=119,
    )  # fmt: skip


def test_a_broadcast_s_rectangle_is_its_island_s_flat_ranks():
    grid = SimpleNamespace(name="grid")
    Run._check_rectangle(island_run(119), grid, 0, (119, 96))
    Run._check_rectangle(island_run(0), grid, 0, (0, 96))
    with pytest.raises(ValueError) as refused:
        Run._check_rectangle(island_run(119), grid, 0, (0, 96))
    assert str(refused.value) == (
        "SimpleNamespace.grid, line 79: km.Broadcast(self, along='row'): the "
        "broadcast's rectangle (0, 0, 12, 8) is the group (119, 96), fixed at load; "
        "submit it to that group (not (0, 96)), or make the program with rect="
    )


def test_a_rectangle_taller_than_island_0_is_one_over_the_islands():
    """A make's rect= taller than island 0 is one rectangle from its first row (over
    cards island 1's rows 10 on): its group the flat ranks of its cells, each core's
    place its place in the whole, its broadcasts layout.island_broadcast_args'."""
    nocs = layout.nocs((12, 10), [(x + 1) | (y + 1) << 16 for x, y in
                       layout.core_order((12, 10))], [(x + 1) | (y + 1) << 16 for x, y
                       in layout.core_order((12, 10))])  # fmt: skip
    stack = {"islands": [(nocs, 0, 118), (nocs, 118, 118)], "bands": None,
             "tops": [0, 10], "rows": [10, 10]}  # fmt: skip
    buffer = SimpleNamespace(index_a=3, block=2)
    compiled = SimpleNamespace(broadcasts=[("column", buffer, 2, "B", None, 0)])
    rect = (2, 8, 4, 4)  # island 0's rows 8 and 9, island 1's rows 0 and 1
    assert spans(stack, rect) and not spans(stack, (2, 6, 4, 4))
    flat = {"now": 118}
    args, group = stacked_rectangle(compiled, stack, rect, lambda: flat["now"])
    assert group == (98, 118 + 12 + 5 + 1 - 98)
    want = layout.island_broadcast_args(
        stack["islands"], layout.Group(*group), rect, "column", 1, (0, 3, 2, 2)
    )
    fields = abi.struct_fields(abi.headers().device[-1], "BroadcastArgs")
    reader, writer, compute = args(15)  # island 1's core 15: row 1, column 3
    assert compute == [1, 3, 1, 4, 4] and reader == compute
    assert writer[5:] == [want[1][15][name] for name in fields]
    assert writer[5 + fields.index("group")] == 98 | 38 << 16  # flat, as the command's
    flat["now"] = 0
    assert args(98)[1][:5] == [0, 0, 1, 4, 4]
    assert args(97)[1] == [0, 0, 0, 4, 4]  # outside the group: no broadcast
    rank = 5 + fields.index("rank")
    assert args(117)[1][rank] == layout.BROADCAST_NONE  # in it, not in the rectangle
    with pytest.raises(ValueError, match=r"a cell of it is past the islands' rank"):
        stacked_rectangle(compiled, stack, (8, 8, 4, 4), lambda: 0)


def test_a_rectangle_over_three_islands_is_one_rectangle_fed_part_to_part():
    """A make's rect= over three islands (a card split in 3, rows 4/3/3): one rectangle,
    island 1's part both fed (from island 2's, the sender's NoC 1 last row) and
    feeding island 0's, as layout.island_broadcast_args writes it."""
    nocs = layout.nocs((12, 10), [(x + 1) | (y + 1) << 16 for x, y in
                       layout.core_order((12, 10))], [(x + 1) | (y + 1) << 16 for x, y
                       in layout.core_order((12, 10))])  # fmt: skip
    stack = {"islands": [(nocs, 0, 47), (nocs, 47, 34), (nocs, 81, 35)],
             "bands": [(0, 4), (4, 3), (7, 3)], "tops": [0, 4, 7],
             "rows": [4, 3, 3]}  # fmt: skip
    buffer = SimpleNamespace(index_a=3, block=2)
    compiled = SimpleNamespace(broadcasts=[("column", buffer, 2, "B", None, 0)])
    rect = (2, 1, 4, 8)
    args, group = stacked_rectangle(compiled, stack, rect, lambda: 0)
    assert group == (14, 85)
    want = layout.island_broadcast_args(
        stack["islands"], layout.Group(*group), rect, "column", 1, (0, 3, 2, 2),
        "below", stack["bands"],
    )  # fmt: skip
    fields = abi.struct_fields(abi.headers().device[-1], "BroadcastArgs")
    middle = 4 * 12 + 2 * 12 + 2  # the card's core of island 1's row 2, column 2
    writer = args(middle)[1]
    assert writer[5:] == [want[1][2 * 12 + 2][name] for name in fields]
    assert writer[5 + fields.index("fed")] != 0  # fed from island 2's part
    assert writer[5 + fields.index("next")] == 3 * 12 + 2  # feeds island 0's


def spanning_run(first):
    program = SimpleNamespace(
        name="columns",
        broadcast_lines={"grid": "line 79: km.Broadcast(self, along='column')"},
        rectangle={"group": (98, 38), "rect": (2, 8, 4, 4), "span": True},
    )
    return SimpleNamespace(
        programs=[program], island_first=first, island_cores=(0, 120), ranks=236,
        island_ranks=118,
    )  # fmt: skip


def test_a_broadcast_over_the_islands_takes_its_one_group_on_every_island():
    grid = SimpleNamespace(name="grid")
    Run._check_rectangle(spanning_run(0), grid, 0, (98, 38))
    Run._check_rectangle(spanning_run(118), grid, 0, (98, 38))
    with pytest.raises(ValueError) as refused:
        Run._check_rectangle(spanning_run(118), grid, 0, (216, 38))
    assert str(refused.value) == (
        "SimpleNamespace.grid, line 79: km.Broadcast(self, along='column'): the "
        "broadcast's rectangle (2, 8, 4, 4) is the group (98, 38), one rectangle over "
        "the islands from island 0's first row (it is taller than island 0) in flat "
        "ranks, fixed at load; submit it to that group (not (216, 38)), or make the "
        "program with rect="
    )
    run = spanning_run(118)
    Run._check_load_rectangle(run, run.programs[0], 0, 236)
    with pytest.raises(ProgramError, match=r"the flat ranks \(98, 38\) over the"):
        Run._check_load_rectangle(run, run.programs[0], 118, 118)


def loading(first, group=(0, 118), rect=None):
    run = SimpleNamespace(
        island_first=first, island_cores=(0, 120), island_ranks=118, ranks=236
    )
    return run, SimpleNamespace(
        name="columns", rectangle={"group": group, "rect": rect}
    )


def test_a_load_s_rectangle_is_checked_against_this_island_s_part_of_the_group():
    run, program = loading(118)
    Run._check_load_rectangle(run, program, 0, 236)  # island 1's part: 118 to 235
    Run._check_load_rectangle(run, program, 118, 118)
    with pytest.raises(ProgramError) as refused:
        Run._check_load_rectangle(run, program, 0, 166)
    assert str(refused.value) == (
        "run.load: columns's rectangle None is the ranks (118, 118) of this island, "
        "not inside the group (0, 166); this island's part of it (118, 48)"
    )
    run, program = loading(0, group=(0, 120))  # the whole grid: the link cores too
    with pytest.raises(ProgramError, match=r"\(0, 120\) of this island, its link"):
        Run._check_load_rectangle(run, program, 0, 236)
    one = SimpleNamespace(island_first=0, island_cores=(0, 120), island_ranks=120,
                          ranks=120)  # fmt: skip
    Run._check_load_rectangle(one, program, 0, 120)
    with pytest.raises(ProgramError, match=r"is the cores \(0, 120\), not inside"):
        Run._check_load_rectangle(one, program, 0, 60)


def grid_nocs(panels=None):
    """A 12 x 10 grid's NoCs, one panel (or the NoC0 x of `panels`' columns)."""
    order = layout.core_order((12, 10))
    words = [(x + 1) | (y + 1) << 16 for x, y in order]
    noc0 = words if panels is None else [panels[x] | (y + 1) << 16 for x, y in order]
    return layout.nocs((12, 10), words, noc0)


TWO = {"bands": None, "tops": [0, 10], "rows": [10, 10]}  # two cards
LINE3 = {"bands": [(0, 4), (4, 3), (7, 3)], "tops": [0, 4, 7], "rows": [4, 3, 3]}


def stack_of(kind, ranks, nocs):
    firsts = [sum(ranks[:i]) for i in range(len(ranks))]
    return {**kind, "islands": [(nocs, f, n) for f, n in zip(firsts, ranks)]}


def test_a_class_s_default_rectangle_on_islands_is_one_island_s_in_flat_ranks():
    """No rect=: one island's rectangle, rank r at (r % 12, r // 12) wherever it is:
    two cards' (0, 120), its ranks 118 and 119 island 1's first; a line of 3's 116."""
    stack, flat = stack_of(TWO, [118, 118], grid_nocs()), {"now": 0}
    plain = SimpleNamespace(broadcasts=[])
    args, group, rect = flat_rectangle(plain, stack, lambda: flat["now"])
    assert (group, rect) == ((0, 120), None)
    assert args(17)[2] == [5, 1, 1, 12, 10]  # x, y, inside, columns, rows
    assert args(118) == ([0, 0, 0, 12, 10],) * 3  # a link core: never inside
    flat["now"] = 118
    assert args(1)[2] == [11, 9, 1, 12, 10]  # rank 119
    assert args(2)[2] == [0, 0, 0, 12, 10]  # rank 120: past one island's grid
    line3 = stack_of(LINE3, [47, 34, 35], grid_nocs())
    args, group, rect = flat_rectangle(plain, line3, lambda: 0)
    assert (group, rect) == ((0, 116), None)
    assert args(48)[2] == [11, 3, 1, 12, 10]  # island 1's first rank: 47
    assert args(47) == ([0, 0, 0, 12, 10],) * 3  # island 0's link core
    buffer = SimpleNamespace(index_a=3, block=2)
    rows = SimpleNamespace(broadcasts=[("row", buffer, 2, "A", None, 0)])
    _, group, rect = flat_rectangle(rows, line3, lambda: 0)
    assert (group, rect) == ((0, 108), (0, 0, 12, 9))  # its whole rows of ranks


def chains(out, islands):
    """Each broadcast's flat ranks, part by part from its sender (next), and every
    part's first core fed by the one before it (fed, from)."""
    words, parts = {}, {}  # flat rank: words; (island, its sender core): flat ranks
    for island, ((_, first, _), one) in enumerate(zip(islands, out)):
        for i, w in one.items():
            words[first + i] = w
            if w["rank"] != layout.BROADCAST_NONE:
                parts.setdefault((island, w["sender"]), []).append(first + i)
    found = []
    for rank, w in sorted(words.items()):
        if w["rank"] != 0 or w["fed"] != 0:
            continue
        cast, head = [], rank
        while True:
            (part,) = [p for p in parts.values() if head in p]
            cast.append(sorted(part))
            after = words[head]["next"]
            if after == layout.BROADCAST_NO_NEXT:
                break
            assert words[after]["fed"] == words[head]["receivers"] + 1
            assert words[after]["from"] == head and words[after]["rank"] == 0
            head = after
        found.append((rank, cast))
    return found


@pytest.mark.parametrize("along, noc", [("row", 0), ("column", 1), ("row", 1)])
@pytest.mark.parametrize("kind", ["two", "line3"])
def test_a_default_rectangle_s_broadcasts_are_one_island_s_parted(along, noc, kind):
    """Its broadcasts are one island's (the same senders, the same cores), each cut
    into parts at an island's edge, a physical row's end and a panel, a part fed by
    the one before it (layout.island_broadcast_args flat)."""
    panels = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14]  # panels (0, 6), (7, 11)
    nocs = grid_nocs(panels)
    ranks, kind = ([118, 118], TWO) if kind == "two" else ([47, 34, 35], LINE3)
    stack = stack_of(kind, ranks, nocs)
    buffer = SimpleNamespace(index_a=3, block=2)
    compiled = SimpleNamespace(broadcasts=[(along, buffer, 2, "A", None, 0)])
    _, group, rect = flat_rectangle(compiled, stack, lambda: 0)
    rect = rect or (0, 0, 12, 10)
    one = layout.broadcast_args(
        (12, 10), nocs.panels, layout.Group(*group), rect, along, noc, nocs.noc0,
        nocs.noc_xy, (0, 3, 2, 2),
    )  # fmt: skip
    out = layout.island_broadcast_args(
        stack["islands"], layout.Group(*group), rect, along, noc, (0, 3, 2, 2),
        "below", stack["bands"], flat=True,
    )  # fmt: skip
    found = chains(out, stack["islands"])
    want = {}
    for core, w in one.items():
        if w["rank"] != layout.BROADCAST_NONE:
            want.setdefault(w["sender"], []).append(core)
    assert {s: sorted(c for part in cast for c in part) for s, cast in found} == want
    assert max(len(cast) for _, cast in found) > 1  # some are parted


def test_a_default_rectangle_s_row_is_parted_at_a_physical_row_s_end():
    """A line of 3: flat row 4 is ranks 48..59, island 1's 1..12: its card row 4's
    columns 1..11, then row 5's column 0, a part fed by the first."""
    stack = stack_of(LINE3, [47, 34, 35], grid_nocs())
    group = layout.Group(0, 108)
    out = layout.island_broadcast_args(
        stack["islands"], group, (0, 0, 12, 9), "row", 0, (0, 3, 2, 2), "below",
        stack["bands"], flat=True,
    )  # fmt: skip
    head, tail = out[1][1], out[1][12]  # island 1's ranks 48 and 59
    assert (head["rank"], head["receivers"], head["next"]) == (0, 10, 59)
    assert head["sender"] == 4 * 12 + 1  # a core of the card
    assert (tail["rank"], tail["fed"], tail["from"]) == (0, 11, 48)
    assert out[0][36]["next"] == 47 and out[1][0]["from"] == 36  # rows 3: an edge


def test_a_broadcast_on_the_second_card_carries_the_command_s_flat_group():
    """Over cards a card's cores take flat ranks from its first: island 1's broadcast
    group (0, 48) in its cores is (118, 48) in the command's flat ranks."""
    size = 14  # BroadcastArgs' words
    mover = [0, 0, 1, 12, 4] + [0, 0 | 48 << 16, 0, 10, 0] + [0] * (size - 5)
    reader, writer, compute = flat_groups((mover, [0, 0, 1, 12, 4], [7]), 118)
    assert reader[6] == 118 | 48 << 16 and reader[:6] == mover[:6]
    assert writer == [0, 0, 1, 12, 4] and compute == [7]  # no broadcast: as it was
    graph = SimpleNamespace(first=lambda island: 118 * island)
    program = SimpleNamespace(rectangle={"nocs": None})
    session_nocs(program, "nocs", (graph, 1))
    assert program.rectangle == {"nocs": "nocs", "flat": 118}
    session_nocs(program, "other", None)  # one card: no flat ranks
    assert program.rectangle == {"nocs": "nocs", "flat": None}


def test_a_default_rectangle_s_broadcast_takes_its_flat_group():
    run = spanning_run(118)
    run.programs[0].rectangle = {"group": (0, 120), "rect": None, "span": "flat"}
    with pytest.raises(ValueError) as refused:
        Run._check_rectangle(run, SimpleNamespace(name="grid"), 0, (118, 120))
    assert str(refused.value) == (
        "SimpleNamespace.grid, line 79: km.Broadcast(self, along='column'): the "
        "broadcast's rectangle the whole grid is the group (0, 120), one island's "
        "rectangle in the islands' flat ranks (the default), fixed at load; submit it "
        "to that group (not (118, 120)), or make the program with rect="
    )
    Run._check_rectangle(run, SimpleNamespace(name="grid"), 0, (0, 120))
