"""A slice's length the build knows when its two ends are one part plus ints (no
card): `1 : J + 1` in a stream's rectangle, `j - 1 : j + 1` in a range loop and
beside a product's other stream, and `block=2 * self.WIDTH` (ints folded). Two ends
of different parts (`j : 2 * j`) stay refused, by the form."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render

J = 2


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 4096, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor():
    """A bfloat16 tensor of 64 x 64 tiles."""
    page = formats.tile_bytes("bfloat16")
    return Tensor(Pages(page), (2048, 2048), "bfloat16", "tile")


class Twice(km.Program):
    @km.handler
    def h(self, x: km.Tensor, out: km.Tensor, j: int):
        results = km.Stream(out.tiles[0:2, j : 2 * j], write=True)
        for t in km.Stream(x.tiles[0:2, j : 2 * j]):
            results.push(t)


def movement(cls, **constants):
    given = {"x": tensor(), "out": tensor(), "s": tensor()}
    compiled = compile_class(cls, {}, given, constants=constants or None)
    return "\n".join(render(prune(compiled.movement["h"]), 0))


def test_ends_of_two_parts_are_refused_by_the_form():
    with pytest.raises(Refused) as refusal:
        movement(Twice)
    message = str(refusal.value)
    assert "a stream's blocks are whole rows of its rectangle, and its columns " in (
        message
    )
    assert "j : 2 * j have a length the build does not know ((2 * j) - (j) is not " in (
        message
    )
