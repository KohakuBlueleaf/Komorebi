"""A scan's state carried by a tile function in a stream loop, `h = f(x, h)` (no
card): h made before the loop (zeros, or a field's tile) packed into an fp32 carry,
read back as f's operand a tile and packed again after it, pushed or not, back in
DST after the loop; the refusals."""

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class


@km.tile_function
def gated(a, h, u):
    return a * h + u


@km.tile_function
def both(a, h):
    return a * h, h


def built(cls, dtype="float32"):
    """The class on tiles of `dtype` (a and u; the rest float32)."""
    tensors = {n: Tensor(Pages(4096), (256, 32), "float32", "tile") for n in "yh"}
    tensors |= {
        n: Tensor(Pages(4096 if dtype == "float32" else 2048), (256, 32), dtype, "tile")
        for n in "au"
    }
    tensors["hs"] = tensors["h0"] = tensors["y"]
    tensors["b0"] = Tensor(Pages(2048), (256, 32), "bfloat16", "tile")
    return compile_class(cls, {}, tensors)


class Unset(km.Program):
    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor):
        results = km.Stream(y.pages(0, 8), write=True)
        for g, x in zip(km.Stream(a.pages(0, 8)), km.Stream(u.pages(0, 8))):
            h = gated(g, h, x)  # noqa: F821
            results.push(h)


class TwoTiles(km.Program):
    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor):
        h = km.tile.zeros(km.fp32, 2)
        results = km.Stream(y.pages(0, 8), write=True)
        for g, x in zip(km.Stream(a.pages(0, 8)), km.Stream(u.pages(0, 8))):
            h = gated(g, h, x)
            results.push(h)


class Twice(km.Program):
    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor):
        h = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, 8), write=True)
        for g in km.Stream(a.pages(0, 8)):
            h = gated(g, h, h)
            results.push(h)


class Field(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor, h0: km.Tensor):
        self.state.fill(km.Stream(h0.pages(0, 1)))
        h = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, 8), write=True)
        for g in km.Stream(a.pages(0, 8)):
            h = gated(g, h, self.state)
            results.push(h)


class Results(km.Program):
    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor):
        h = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, 8), write=True)
        for g in km.Stream(a.pages(0, 8)):
            h = both(g, h)
            results.push(h)


class Bf16Field(km.Program):
    half: km.L1[km.bf16, 1, 32, 32]

    @km.handler
    def run(self, a: km.Tensor, u: km.Tensor, y: km.Tensor, b0: km.Tensor):
        self.half.fill(km.Stream(b0.pages(0, 1)))
        h = self.half
        results = km.Stream(y.pages(0, 8), write=True)
        for g, x in zip(km.Stream(a.pages(0, 8)), km.Stream(u.pages(0, 8))):
            h = gated(g, h, x)
            results.push(h)


REFUSALS = {
    "unset": Unset,
    "two_tiles": TwoTiles,
    "twice": Twice,
    "field": Field,
    "results": Results,
    "bf16_field": Bf16Field,
}


@pytest.mark.parametrize(
    "handler,why",
    [
        ("unset", "h is read before it is set"),
        ("two_tiles", "one DST tile, passed once, and it is 2 tiles"),
        ("twice", "one DST tile, passed once"),
        ("field", "no field"),
        ("results", "several results"),
        ("bf16_field", "a float32 field of one tile"),
    ],
)
def test_the_refusals(handler, why):
    with pytest.raises(Refused, match=why):
        built(REFUSALS[handler])
