"""DST after `self.m.write(acc)` (no card): the write packs acc's sums and releases
DST, so a tile function or a body's tile after it takes DST 0; sums not yet written
(or a live tile) still hold DST and refuse the next tile by name; a scan's state does
not hold it while its loop carries it."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render

BODY = "result = tile;"


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensors():
    """Three float32 tensors of tiles: s, t, y."""
    page = formats.tile_bytes("float32")
    return {n: Tensor(Pages(page), (2048, 32), "float32", "tile") for n in "sty"}


@km.tile_function
def doubled(x):
    return x + x


class Function(km.Program):
    m: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, s: km.Tensor, t: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            acc += tile
        self.m.write(acc)
        results = km.Stream(y.pages(self.rank, 1), write=True)
        for tile in km.Stream(t.pages(self.rank, 1)):
            results.push(doubled(tile))


class Body(km.Program):
    m: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, s: km.Tensor, t: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            acc += tile
        self.m.write(acc)
        results = km.Stream(y.pages(self.rank, 1), write=True)
        for tile in km.Stream(t.pages(self.rank, 1)):
            copied = km.cxx(BODY, takes=[tile], gives=km.Tile)
            results.push(copied)


class Unwritten(km.Program):
    m: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, s: km.Tensor, t: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            acc += tile
        results = km.Stream(y.pages(self.rank, 1), write=True)
        for tile in km.Stream(t.pages(self.rank, 1)):
            results.push(doubled(tile))
        self.m.write(acc)


@km.tile_function
def gated(a, h, u):
    return a * h + u


class ScanThenSums(km.Program):
    m: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, s: km.Tensor, t: km.Tensor, y: km.Tensor):
        state = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, 8), write=True)
        for g, x in zip(km.Stream(s.pages(0, 8)), km.Stream(t.pages(0, 8))):
            state = gated(g, state, x)
            results.push(state)
        y.pages(8, 1).write(state)
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            acc += tile
        self.m.write(acc)
        doubles = km.Stream(y.pages(9, 1), write=True)
        for tile in km.Stream(t.pages(self.rank, 1)):
            doubles.push(doubled(tile))


class ScanUnwritten(km.Program):
    @km.handler
    def go(self, s: km.Tensor, t: km.Tensor, y: km.Tensor):
        state = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, 8), write=True)
        for g, x in zip(km.Stream(s.pages(0, 8)), km.Stream(t.pages(0, 8))):
            state = gated(g, state, x)
            results.push(state)
        doubles = km.Stream(y.pages(9, 1), write=True)
        for tile in km.Stream(t.pages(self.rank, 1)):
            doubles.push(doubled(tile))


def compute(cls):
    compiled = compile_class(cls, {}, tensors())
    return "\n".join(render(prune(compiled.compute["go"]), 0))


@pytest.mark.parametrize("cls", [Function, Body])
def test_a_tile_after_the_write_of_acc_takes_dst_0(cls):
    text = compute(cls)
    written = text.index("kFieldM")
    released = text.index("tile_regs_release();", written)
    assert released < text.index("tile_regs_acquire();", released)


def test_sums_not_yet_written_still_hold_dst():
    with pytest.raises(Refused) as refusal:
        compute(Unwritten)
    message = str(refusal.value)
    assert "`results.push(doubled(tile))`: a tile function's result is DST 0, " in (
        message
    )
    assert "which another tile holds; write or push that tile first" in message


def test_a_scan_then_sums_written_to_a_field_then_a_tile_build():
    text = compute(ScanThenSums)  # the scan's own call is not refused
    assert text.index("kFieldM") < text.rindex("tile_regs_acquire();")


def test_a_scans_state_back_in_dst_after_its_loop_holds_it():
    with pytest.raises(Refused) as refusal:
        compute(ScanUnwritten)
    assert "`doubles.push(doubled(tile))`: a tile function's result is DST 0, " in (
        str(refusal.value)
    )
