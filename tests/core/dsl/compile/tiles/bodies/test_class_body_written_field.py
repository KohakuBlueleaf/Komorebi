"""A body takes a field compute wrote above, or one km.all_reduce left (no card): its
pages waited for, read in place and freed by the next write or the handler's end; on
pack handed the address, a release handed back. A mover's body, or a field neither
filled nor written, is refused by name."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render

BODY = "result = m;"


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensors():
    """Two float32 tensors of tiles: s and y."""
    page = formats.tile_bytes("float32")
    return {n: Tensor(Pages(page), (2048, 32), "float32", "tile") for n in "sy"}


class OnMover(km.Program):
    m: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, s: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            acc += tile
        self.m.write(acc)
        km.cxx(BODY, takes=[self.m], on="mover_b")


class Unfilled(km.Program):
    m: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, s: km.Tensor, y: km.Tensor):
        km.cxx(BODY, takes=[self.m])


def compute(cls):
    compiled = compile_class(cls, {}, tensors())
    return "\n".join(render(prune(compiled.compute["go"]), 0))


def test_a_body_on_a_mover_is_refused():
    with pytest.raises(Refused) as refusal:
        compute(OnMover)
    message = str(refusal.value)
    assert "self.m is written by compute above: its pages are compute's; " in message
    assert 'take it in a body on compute (no on=, or on="pack")' in message


def test_a_field_neither_filled_nor_written_is_still_refused():
    with pytest.raises(Refused) as refusal:
        compute(Unfilled)
    assert "self.m is not filled above in this handler (or a body took it)" in str(
        refusal.value
    )
