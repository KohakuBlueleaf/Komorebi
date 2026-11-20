"""An out buffer is shared by name across handlers only while all their items are of one
width: an item is one contiguous region and the ring's positions last across commands,
so a 5-tile sum after one 1-tile sum in a ring of 10 pages passed its end (bad_item);
a handler with items of another width takes a buffer of its own (no card)."""

from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class


class TwoWidths(km.Program):
    @km.handler
    def turns(self, x: km.Tensor, out: km.Tensor):
        for pos in range(3):
            acc = km.tile.zeros(km.fp32, 4)
            for xb in km.Stream(x.tiles[0:1, 0:4], block=4):
                acc += xb
            whole = km.reduce(acc, form="tree", group=2)
            if self.rank == self.leader:
                out.tiles[0:1, pos * 4 : pos * 4 + 4].write(whole)

    @km.handler
    def once(self, x: km.Tensor, out: km.Tensor):
        acc = km.tile.zeros(km.fp32, 3)
        for xb in km.Stream(x.tiles[0:1, 0:3], block=3):
            acc += xb
        whole = km.reduce(acc, form="tree", group=2)
        if self.rank == self.leader:
            out.tiles[0:1, 0:3].write(whole)

    @km.handler
    def one(self, x: km.Tensor, out: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for xb in km.Stream(x.tiles[0:1, 0:1], block=1):
            acc += xb
        whole = km.reduce(acc, form="tree", group=2)
        if self.rank == self.leader:
            out.tiles[0:1, 0:1].write(whole)


class OneAndFive(km.Program):
    @km.handler
    def narrow(self, x: km.Tensor, out: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for xb in km.Stream(x.tiles[0:1, 0:4], block=2):
            acc += xb
        whole = km.reduce(acc, form="tree", group=5)
        if self.rank == self.leader:
            out.tiles[0:1, 0:1].write(whole)

    @km.handler
    def wide(self, x: km.Tensor, out: km.Tensor):
        acc = km.tile.zeros(km.fp32, 5)
        for xb in km.Stream(x.tiles[0:1, 0:5], block=5):
            acc += xb
        whole = km.reduce(acc, form="tree", group=2)
        if self.rank == self.leader:
            out.tiles[0:1, 0:5].write(whole)


def out_buffers(cls):
    tensors = {
        "x": Tensor(Pages(2048), (32, 8 * 32), "bfloat16", "tile"),
        "out": Tensor(Pages(2048), (32, 12 * 32), "bfloat16", "tile"),
    }
    compiled = compile_class(cls, {}, tensors)
    return [b for b in compiled.buffers if b.role == "out"]


def test_each_out_buffer_holds_items_of_one_width():
    for buffer in out_buffers(TwoWidths) + out_buffers(OneAndFive):
        assert len(buffer.items) == 1, buffer
        assert buffer.pages[0] % buffer.items[0] == 0, buffer


def test_a_sum_of_another_width_takes_a_buffer_of_its_own():
    by = {(b.dtype, tuple(b.items)): b for b in out_buffers(TwoWidths)}
    assert by[("float32", (4,))].pages == (8,)
    assert by[("float32", (3,))].pages == (6,)
    assert by[("float32", (1,))].pages == (2,)
    assert by[("bfloat16", (4,))].pages == (8,)
    assert len({b.index_a for b in by.values()}) == 6


def test_a_one_tile_sum_beside_five_tile_sums_has_its_own_ring():
    by = {(b.dtype, tuple(b.items)): b for b in out_buffers(OneAndFive)}
    assert by[("float32", (1,))].pages == (2,)
    assert by[("float32", (5,))].pages == (10,)
    assert by[("float32", (1,))].index_a != by[("float32", (5,))].index_a
