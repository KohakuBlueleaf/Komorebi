"""A local=True reduce's check on an island run that holds two programs of one class
(no card): a handler named through its program is checked against that program; the
class's bare handler is still refused as naming none."""

from types import SimpleNamespace

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.islands import local as island_local
from komorebi.session.run import Run

EDGE = 70  # island 1's first rank


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def grid(pages):
    return Tensor(Pages(4096, pages), (pages * 32, 32), "float32", "tile")


class Sevens(km.Program):
    """A tile a rank summed over groups of 7, each group's sum at page rank // 7."""

    BASE = km.constant(0)

    @km.handler
    def sevens(self, s: km.Tensor, t: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.BASE + self.rank, 1)):
            total += tile
        summed = km.reduce(total, form="tree", group=7, local=True)
        if self.rank == self.leader:
            t.pages((self.BASE + self.rank) // 7, 1).write(summed)


class IslandPart:
    """What check_local reads of an island's Run: its programs, Run._handler."""

    def __init__(self, programs):
        self.programs = programs

    _handler = Run._handler
    _class_form_refused = Run._class_form_refused
    _program_told = Run._program_told
    program_field = Run.program_field


def test_a_local_reduce_named_through_one_of_two_programs_is_checked():
    s, t = grid(160), grid(160)
    first, second = (
        Sevens(l1=1 << 20, log=lambda line: None, s=s, t=t, BASE=base)
        for base in (0, 7)
    )
    part = IslandPart([first, second])
    runs, graph = [part, part], SimpleNamespace(first=lambda island: EDGE)
    for program in (first, second):  # on one island: accepted
        island_local.check_local(runs, graph, program.sevens, (0, 21))
        island_local.check_local(runs, graph, program.sevens, (EDGE, 21))
    with pytest.raises(ValueError, match=r"Sevens.sevens, .*local=True, and a group "
                       r"of 7 ranks of the command \(60, 21\) holds ranks of islands "
                       r"0 and 1"):  # fmt: skip
        island_local.check_local(runs, graph, second.sevens, (60, 21))
    with pytest.raises(ValueError, match=r"the run holds 2 programs of the class"):
        island_local.check_local(runs, graph, Sevens.sevens, (0, 21))
