"""A class on an island session (no card): the islands' common grid, a broadcast's
rectangle taken as one island's flat ranks, a reduce group that spans two islands
refused by name."""

from types import SimpleNamespace

import pytest

from komorebi.dsl.program import flat_groups, rank_rectangle, session_nocs
from komorebi.errors import KomorebiError, ProgramError
from komorebi.islands.run import _check_reduce_islands
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
        "submit it to that group (not (0, 96)), or make the program with rect=; a "
        "broadcast stays on one island"
    )


def test_self_x_and_y_on_a_group_across_two_islands_is_refused_by_name():
    run = island_run(119)
    run.programs[0].rect_lines = {"place": "line 40"}
    place = SimpleNamespace(name="place")
    Run._check_places(run, place, 0, (119, 60))  # on island 1 alone
    Run._check_places(run, SimpleNamespace(name="grid"), 0, (60, 120))  # no self.x
    with pytest.raises(ValueError) as refused:
        Run._check_places(run, place, 0, (60, 120))
    assert str(refused.value) == (
        "SimpleNamespace.place, line 40: self.x and self.y are a core's "
        "place on its own island, and the group 60..179 spans more than this "
        "island's ranks 119..237, whose places the next island's ranks repeat: "
        "submit it on one island's ranks"
    )


class Linear:
    reduce_groups = {"split": (14, "line 70: km.reduce(self.sums, group=14)")}


def test_a_reduce_group_across_two_islands_is_refused_by_name():
    graph = SimpleNamespace(islands=[0, 1], first=lambda island: 119 * island)
    split = SimpleNamespace(method=SimpleNamespace(name="split"), program=Linear())
    _check_reduce_islands(graph, split, (63, 112))  # island 1 begins a group
    _check_reduce_islands(graph, split, (0, 112))  # on island 0 alone
    with pytest.raises(ValueError) as refused:
        _check_reduce_islands(graph, split, (60, 112))
    assert str(refused.value) == (
        "Linear.split, line 70: km.reduce(self.sums, group=14): its reduce group of "
        "14 ranks at flat ranks 116..129 spans islands 0 and 1; start the command so "
        "that island 1's first rank 119 begins a group"
    )


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
