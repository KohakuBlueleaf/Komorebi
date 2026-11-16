"""A class on an island session (no card): the islands' common grid, a broadcast's
rectangle taken as one island's flat ranks, or one over the islands (taller than
island 0)."""

from types import SimpleNamespace

import pytest

from komorebi import abi, layout
from komorebi.dsl.program import (
    flat_groups,
    rank_rectangle,
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


def test_a_rectangle_over_more_than_two_islands_is_refused_for_now():
    """A make's rect= over three islands (a card split in 3, rows 4/3/3) is refused by
    name, a limit being lifted; one over two of them is taken."""
    nocs = layout.nocs((12, 10), [(x + 1) | (y + 1) << 16 for x, y in
                       layout.core_order((12, 10))], [(x + 1) | (y + 1) << 16 for x, y
                       in layout.core_order((12, 10))])  # fmt: skip
    stack = {"islands": [(nocs, 0, 47), (nocs, 47, 34), (nocs, 81, 35)],
             "bands": [(0, 4), (4, 3), (7, 3)], "tops": [0, 4, 7],
             "rows": [4, 3, 3]}  # fmt: skip
    buffer = SimpleNamespace(index_a=3, block=2)
    compiled = SimpleNamespace(broadcasts=[("column", buffer, 2, "B", None, 0)])
    stacked_rectangle(compiled, stack, (2, 1, 4, 5), lambda: 0)  # islands 0 and 1
    with pytest.raises(ValueError) as refused:
        stacked_rectangle(compiled, stack, (2, 1, 4, 8), lambda: 0)
    assert str(refused.value) == (
        "rect=(2, 1, 4, 8) is taller than island 0: one rectangle from its first row "
        "on, over islands 0 to 2; a class's rectangle over more than two islands is "
        "refused for now, a limit being lifted: give one over at most two islands"
    )


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


def test_a_class_s_default_rectangle_on_an_island_is_its_rank_cores():
    nocs = SimpleNamespace(grid=(12, 10))
    args, group, rect = rank_rectangle(SimpleNamespace(broadcasts=[]), nocs, 118)
    assert (group, rect) == ((0, 118), None)
    assert args(17)[2] == [5, 1, 1, 12, 10]  # x, y, inside, columns, rows
    assert args(118) == ([0, 0, 0, 12, 10],) * 3  # a link core: never inside
    _, group, rect = rank_rectangle(SimpleNamespace(broadcasts=[]), nocs, 120)
    assert (group, rect) == ((0, 120), None)  # one card: the whole grid


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
