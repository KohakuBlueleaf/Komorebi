"""A block a broadcast cast once and taken in a loop the cast does not stand in (a
weight held across turns, no card): compute waits for it once at the cast, every take
reads it in place, and it is freed after the loop; two products a turn take it."""

import pytest
from broadcast_product_fixture import project, tensors

import komorebi as km
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source, movement_source

K, N = 8, 4


class HeldWeight(km.Program):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1, mover="A")
        if column.sender():
            ws = column.cast(w.tiles[0:K, 0:N].read())
        else:
            ws = column.arrive()
        for i in range(2):
            acc = km.tile.zeros(km.fp32, N)
            for xb in km.Stream(x.tiles[i : i + 1, 0:K], block=2, pattern="B"):
                acc += project(xb, ws)
            y.pages((self.rank * 2 + i) * N, N).write(acc)


class TwoProducts(km.Program):
    """A weight read by two products a turn: the second take line of the block."""

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1, mover="A")
        if column.sender():
            ws = column.cast(w.tiles[0:K, 0:N].read())
        else:
            ws = column.arrive()
        for i in range(2):
            acc = km.tile.zeros(km.fp32, N)
            for xb in km.Stream(x.tiles[i : i + 1, 0:K], block=2, pattern="B"):
                acc += project(xb, ws)
            y.pages((self.rank * 4 + i) * N, N).write(acc)
            again = km.tile.zeros(km.fp32, N)
            for xb in km.Stream(x.tiles[i + 2 : i + 3, 0:K], block=2, pattern="B"):
                again += project(xb, ws)
            y.pages((self.rank * 4 + 2 + i) * N, N).write(again)


class HeldTiles(km.Program):
    """A block read tile by tile each turn, and by a body each turn."""

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1, mover="A")
        if column.sender():
            ws = column.cast(w.tiles[0:1, 0:2].read())
        else:
            ws = column.arrive()
        for i in range(2):
            total = km.tile.zeros(km.fp32)
            for tile in ws:
                total += tile
            y.pages(self.rank * 2 + i, 1).write(total)
            km.cxx("(void)ws_tile_buffer;", takes=[ws])


@pytest.mark.parametrize("cls", [HeldWeight, TwoProducts, HeldTiles])
def test_a_block_cast_once_is_waited_once_read_each_turn_and_freed_after(cls):
    compiled = compile_class(cls, {}, tensors())
    compute = compute_source(compiled)
    case = compute[compute.index("kernel_main") :]
    # a held block: cast once, waited and freed once around the loop that reads it
    assert case.count("km::broadcast_receive(compute, kBroadcast") == 1, cls.__name__
    assert case.count("km::broadcast_done(kBroadcast") == 1, cls.__name__
    assert movement_source(compiled).count("column.cast(") == 1
