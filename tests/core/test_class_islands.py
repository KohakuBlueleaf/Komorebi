"""A class on an island session (no card): the islands' common grid, a broadcast's
rectangle taken as one island's flat ranks, a reduce group that spans two islands
refused by name."""

from types import SimpleNamespace

import pytest

from komorebi.errors import KomorebiError
from komorebi.island_session import IslandSession, _check_reduce_islands
from komorebi.run import Run


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


def test_a_tensor_made_while_an_island_run_lives_is_refused_by_name():
    made = islands_of((12, 10), (12, 10))
    made._tables, made.run = [object()], SimpleNamespace(stopped=False)
    with pytest.raises(KomorebiError) as refused:
        made._refuse_live("a tensor")
    assert str(refused.value) == (
        "a tensor made after the load: on islands a tensor's table row is written at "
        "the load; make it before session.load"
    )
    made.run.stopped = True
    made._refuse_live("a tensor")  # the run stopped: the next load writes it
