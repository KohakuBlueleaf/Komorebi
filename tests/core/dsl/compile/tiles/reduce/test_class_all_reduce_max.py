"""km.all_reduce(self.part, op="max"): the arrivals folded into the field by the
SFPU's elementwise max, not added, and the result at every rank; op= takes "sum"
or "max", and a tile's all-reduce keeps the sum (no card)."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.pages, self.address = page_size, 64, 0x9000


def tensors():
    """A float32 tensor of tiles: s."""
    page = formats.tile_bytes("float32")
    return {"s": Tensor(Pages(page), (2048, 32), "float32", "tile")}


class RowMax(km.Program):
    part: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, s: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            acc += tile
        self.part.write(acc)
        km.all_reduce(self.part, form="tree", group=4, op="max")


class RowSum(km.Program):
    part: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, s: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            acc += tile
        self.part.write(acc)
        km.all_reduce(self.part, form="tree", group=4)


class OtherOp(km.Program):
    part: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, s: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            acc += tile
        self.part.write(acc)
        km.all_reduce(self.part, form="tree", group=4, op="min")


def compute(cls):
    compiled = compile_class(cls, {}, tensors())
    return compiled, "\n".join(render(prune(compiled.compute["go"]), 0))


def test_max_folds_the_arrivals_by_the_sfpu_max():
    compiled, text = compute(RowMax)
    folded = text.split("km::reduce(compute", 1)[1].split("});", 1)[0]
    assert "binary_max_tile(0, 1, 0);" in folded
    assert "add_binary_tile" not in folded
    assert "max" in compiled.includes


def test_the_default_still_adds():
    compiled, text = compute(RowSum)
    folded = text.split("km::reduce(compute", 1)[1].split("});", 1)[0]
    assert "add_binary_tile(0, 1, 0);" in folded
    assert "binary_max_tile" not in text
    assert "max" not in compiled.includes


def test_another_op_is_refused_by_name():
    with pytest.raises(Refused) as refusal:
        compute(OtherOp)
    assert 'km.all_reduce\'s op= is "sum" or "max"' in str(refusal.value)
