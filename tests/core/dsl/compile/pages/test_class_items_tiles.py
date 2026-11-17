"""An item of a stream of items (`t.items(first, n, size=B)`) is not a tile of compute:
each place where compute would unpack or pack one as a tile refuses it by name (no
card). A body still takes it."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor():
    """A bfloat16 tensor of tiles."""
    return Tensor(Pages(formats.tile_bytes("bfloat16")), (2048, 32), "bfloat16", "tile")


@km.tile_function
def doubled(x):
    return x + x


@km.tile_function
def product(a, b):
    return a @ b


class IntoItems(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor):
        results = km.Stream(out.items(0, 1, size=32), write=True)
        for tile in km.Stream(t.pages(0, 1)):
            results.push(tile)


class Called(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 1), write=True)
        for item in km.Stream(t.items(0, 1, size=32)):
            results.push(doubled(item))


class Expression(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 1), write=True)
        for item in km.Stream(t.items(0, 1, size=32)):
            results.push(item + item)


class Added(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for item in km.Stream(t.items(0, 1, size=32)):
            total += item
        out.pages(0, 1).write(total)


class Multiplied(km.Program):
    @km.handler
    def go(self, t: km.Tensor, w: km.Tensor, out: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for item, tile in zip(
            km.Stream(t.items(0, 1, size=32)), km.Stream(w.pages(0, 1))
        ):
            acc += product(item, tile)
        out.pages(0, 1).write(acc)


@pytest.mark.parametrize(
    ("cls", "line", "name", "what"),
    [
        (Called, "results.push(doubled(item))", "item", "a tile function"),
        (Expression, "results.push(item + item)", "item", "a tile function"),
        (Added, "total += item", "item", "an add"),
        (Multiplied, "acc += product(item, tile)", "item", "a product"),
    ],
)
def test_an_item_is_refused_where_compute_takes_a_tile(cls, line, name, what):
    message = refused(cls)
    assert f"`{line}`: {name} is an item of 32 B of a stream of items" in message
    assert f"not a tile of compute: {what} unpacks or packs a whole tile" in message


def test_a_write_stream_of_items_takes_no_tile():
    message = refused(IntoItems)
    assert "`results.push(tile)`: results writes items of 32 B, not tiles of " in (
        message
    )
    assert "compute: a push packs a whole tile into each, past the item's page" in (
        message
    )


def refused(cls):
    """The refusal of cls's build, checked for what it says to write instead."""
    tensors = {"t": tensor(), "w": tensor(), "out": tensor()}
    with pytest.raises(Refused) as refusal:
        compile_class(cls, {}, tensors)
    message = str(refusal.value)
    assert "stream the tensor's tiles (`t.tiles[r : r + 1, c0:c1]`)" in message
    assert 'or move the items on a mover in a body (`km.cxx(..., on="mover_b")`)' in (
        message
    )
    return message


class Body(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor):
        results = km.Stream(out.items(0, 1, size=32), write=True)
        for item in km.Stream(t.items(0, 1, size=32)):
            km.cxx("copy(item, results);", takes=[item, results], on="pack")


def test_a_body_still_takes_an_item_and_a_write_stream_of_items():
    compile_class(Body, {}, {"t": tensor(), "out": tensor()})
