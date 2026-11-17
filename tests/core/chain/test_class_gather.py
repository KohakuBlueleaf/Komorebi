"""km.all_gather in a class's km.chain method: its kernels get the gather's case and
KM_CHAIN_GATHER (a class without it gets neither), its record on one island and over
islands, and what is refused, on the fake backend (no card)."""

from types import SimpleNamespace

import numpy as np
import pytest
from class_chain_fixture import BlockBackend, Pages
from fake_port import FakePort

import komorebi as km
from komorebi.chain import calls as kc
from komorebi.chain import marks as chain_marks
from komorebi.device.tensor import Tensor
from komorebi.session.run import Run

ONE = int(np.float32(1).view(np.uint32))
CORES, WIDTH = 8, 2
TILES = CORES * WIDTH


def tensor(address, copy=False):
    made = Tensor(Pages(TILES, address), (TILES, 1024), "float32", "row")
    if copy:  # an island tensor's hint (komorebi.islands.graph.place)
        made.placement = SimpleNamespace(copy=True)
    return made


SCALE = """
#include "api/compute/eltwise_unary/binop_with_scalar.h"
copy_init(tile_buffer);
copy_tile(tile_buffer, tile_index, 0);
binop_with_scalar_tile_init();
mul_unary_tile(0, factor);
"""


class Scales(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, factor: int, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            scaled = km.cxx(SCALE, takes=[tile, factor], gives=km.Tile)
            results.push(scaled)

    @km.chain
    def plain(self, x, y, z, width: int):
        self.scale(x, y, ONE, width)
        self.scale(y, z, ONE, width)


class Gathers(Scales):
    @km.chain
    def gathered(self, x, y, z, width: int):
        self.scale(x, y, ONE, width)
        km.all_gather(y, 0, TILES)
        self.scale(y, z, ONE, width)


def made(cls):
    x = tensor(0x10000)
    return cls(l1=1 << 20, log=lambda line: None, a=x, out=tensor(0x20000))


def sources(program):
    folder = program.built.folder
    return {name: (folder / name).read_text() for name in program.built.files}


def given(copy=False):
    return tensor(0x10000), tensor(0x20000, copy), tensor(0x30000)


def test_a_class_that_gathers_has_the_case_and_its_define():
    program = made(Gathers)
    movement = sources(program)["movement.cpp"]
    assert program.chain_gather
    assert "case kHandlerKmChain: {  // km.all_gather's record" in movement
    assert "km::chain_gather(km::ChainShape{}, movement, command);" in movement
    assert program.make_given["defines"]["KM_CHAIN_GATHER"] == "1"


def test_a_class_that_does_not_gather_has_neither():
    program = made(Scales)
    text = sources(program)
    assert not program.chain_gather
    assert "KM_CHAIN_GATHER" not in program.make_given["defines"]
    assert not any("chain_gather" in body for body in text.values())


def test_one_island_a_plain_record_of_the_chains_first_rank():
    program = made(Gathers)
    run = Run(BlockBackend(FakePort()), (12, 10), program=program)
    x, y, z = given()
    chain = run.chain(program.gathered(x, y, z, WIDTH), cores=(0, CORES))
    gather = chain.calls[1]
    assert gather.group == (0, 1) and gather.width == TILES and gather.gather == (0,)
    records = np.frombuffer(run.backend.blocks[-1].data, np.uint32).reshape(-1, 16)
    assert records[1][11] >> 24 == program.chain_handler
    assert list(records[1][:3]) == [y.address, 0, TILES]
    marks = chain_marks.marks(chain.calls)
    assert marks[1] == {0: (CORES, list(range(CORES)))}
    assert marks[2] == {core: (1, [0]) for core in range(CORES)}


def test_two_islands_a_record_over_the_edge():
    program = made(Gathers)
    run = Run(BlockBackend(FakePort()), (12, 10), program=program)
    x, y, z = given(copy=True)
    calls = run._class_calls(
        program.gathered(x, y, z, WIDTH), None, True, (0, CORES), (0, 4)
    )
    kc.build(calls)
    assert calls[1].group == (3, 2) and calls[1].gather == (0, 4)
    marks = chain_marks.marks(calls)
    assert marks[1] == {3: (4, [0, 1, 2, 3]), 4: (4, [4, 5, 6, 7])}
    assert marks[2] == {core: (1, [3 if core < 4 else 4]) for core in range(CORES)}


def test_over_islands_a_tensor_not_a_copy_is_refused():
    program = made(Gathers)
    run = Run(BlockBackend(FakePort()), (12, 10), program=program)
    with pytest.raises(kc.ChainRefused, match=r"a copy \(copy=True\)"):
        run._class_calls(
            program.gathered(*given(), WIDTH), None, True, (0, CORES), (0, 4)
        )


def test_a_listed_gather_in_a_class_without_one_is_refused():
    program = made(Scales)
    run = Run(BlockBackend(FakePort()), (12, 10), program=program)
    x, y, z = given()
    listed = [Scales.scale(x, y, ONE, WIDTH), km.all_gather(y, 0, TILES)]
    with pytest.raises(kc.ChainRefused, match="no km.all_gather in a km.chain method"):
        run.chain([*listed, Scales.scale(y, z, ONE, WIDTH)], cores=(0, CORES))
