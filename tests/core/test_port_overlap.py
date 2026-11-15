"""A group's port commands do not overlap: run.submit refuses a port command on cores
a port command without its event still holds unless it is exclusive (no card)."""

import numpy as np
import pytest

import komorebi as km
from komorebi.dsl.compiler import compile_class
from komorebi.run import Run
from komorebi.tensor import Tensor


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


class Summed(km.Program):
    @km.handler
    def total(self, source: km.Tensor, output: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(source.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:
            output.pages(0, 1).write(whole)

    @km.handler
    def copy(self, source: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


class Held:
    """What Run's port check reads of a program."""

    handlers = {"reduce": 3, "map": 4}
    port_handlers = {"reduce"}


def run_of(program):
    """A Run with no card: only the columns the port check reads."""
    run = Run.__new__(Run)
    run.programs = [program]
    run._ports, run._base, run._handed, run._next = {}, 1, 1, 1
    run._has = np.zeros(0, bool)
    return run


def test_a_class_handler_that_reduces_is_a_port_handler():
    tensors = {name: Tensor(Pages(4096), (2048, 32), "float32", "tile") for name in (
        "source", "output")}  # fmt: skip
    compiled = compile_class(Summed, {}, tensors)
    assert compiled.port_handlers == {"total"}


def test_an_overlapping_port_command_is_refused_unless_exclusive():
    run = run_of(Held())
    assert run._check_ports(0, 4, (0, 8), False) is False  # not a port handler
    assert run._check_ports(0, 3, (0, 4), False) is True
    run._ports[1] = (0, 4)  # command 1 holds cores 0 to 3
    run._next = 2
    with pytest.raises(ValueError, match=r"command 2 .* command 1 \(cores 0 to 3\)"):
        run._check_ports(0, 3, (2, 4), False)
    assert run._check_ports(0, 3, (2, 4), True) is True  # exclusive: ordered
    assert run._check_ports(0, 3, (4, 4), False) is True  # other cores


def test_a_port_command_whose_event_came_frees_its_cores():
    run = run_of(Held())
    run._ports[1] = (0, 4)
    run._handed, run._next, run._has = 2, 2, np.array([True])
    assert run._check_ports(0, 3, (0, 4), False) is True
    assert 1 not in run._ports
