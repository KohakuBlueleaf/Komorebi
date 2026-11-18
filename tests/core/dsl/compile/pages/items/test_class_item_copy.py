"""`out.push(item)`, an item of a stream of items into a write stream of items of its
size (no card): the movers copy it, mover A reading into the loop's half and mover B
writing from it, no compute line and no buffer of the write stream's own. Different
sizes, other lines in the loop, two streams or a pattern are refused by name."""

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


def tensors():
    """Two float32 tensors of 4,096 B rows, and one of tiles."""
    rows = {n: Tensor(Pages(4096), (64, 1024), "float32", "row") for n in ("t", "out")}
    page = formats.tile_bytes("float32")
    return rows | {"w": Tensor(Pages(page), (2048, 32), "float32", "tile")}


class Copied(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor, w: km.Tensor, first: int):
        results = km.Stream(out.items(0, 8, size=64), write=True)
        for item in km.Stream(t.items(first, 8, size=64)):
            results.push(item)


class Sized(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor, w: km.Tensor, first: int):
        results = km.Stream(out.items(0, 8, size=128), write=True)
        for item in km.Stream(t.items(first, 8, size=64)):
            results.push(item)


class Other(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor, w: km.Tensor, first: int):
        results = km.Stream(out.items(0, 8, size=64), write=True)
        for item in km.Stream(t.items(first, 8, size=64)):
            results.push(item)
            km.cxx("look(item);", takes=[item], on="pack")


class Patterned(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor, w: km.Tensor, first: int):
        results = km.Stream(out.items(0, 8, size=64), write=True, pattern="AB")
        for item in km.Stream(t.items(first, 8, size=64)):
            results.push(item)


class Zipped(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor, w: km.Tensor, first: int):
        results = km.Stream(out.items(0, 8, size=64), write=True)
        for item, _other in zip(
            km.Stream(t.items(first, 8, size=64)), km.Stream(t.items(0, 8, size=64))
        ):
            results.push(item)


class Shared(km.Program):
    """A tile write stream in one handler, an item copy in the other: the copy's
    write stream has no pair, the other keeps its own."""

    @km.handler
    def tiles(self, t: km.Tensor, out: km.Tensor, w: km.Tensor, first: int):
        results = km.Stream(w.pages(1, 1), write=True)
        for tile in km.Stream(w.pages(0, 1)):
            results.push(tile)

    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor, w: km.Tensor, first: int):
        results = km.Stream(out.items(0, 8, size=64), write=True)
        for item in km.Stream(t.items(first, 8, size=64)):
            results.push(item)


def compiled(cls):
    return compile_class(cls, {}, tensors())


def test_mover_a_reads_and_mover_b_writes_from_its_half():
    program = compiled(Copied)
    (pair,) = [b for b in program.buffers if b.name == "kRead0"]
    assert pair.patterns["kRead0Pattern1"] == "A"
    assert pair.patterns["kRead0Pattern2"] == "B"
    assert [b.name for b in program.buffers] == ["kRead0"]  # no write pair


def test_a_write_pair_of_another_handler_stays():
    names = [b.name for b in compiled(Shared).buffers]
    assert "kWrite0" in names and len(names) == 3  # kRead0, kWrite0, kRead0b


def refused(cls):
    with pytest.raises(Refused) as refusal:
        compiled(cls)
    return str(refusal.value)


def test_items_of_two_sizes_are_refused():
    message = refused(Sized)
    assert "`results.push(item)`: item is an item of 64 B and results " in message
    assert "writes items of 128 B: a mover copies an item whole, into an item of " in (
        message
    )
    assert "its size; give both streams size=64" in message


def test_another_line_in_the_loop_is_refused():
    message = refused(Other)
    assert "an item copy's loop holds the push alone: compute takes no item" in message


def test_a_pattern_is_refused():
    message = refused(Patterned)
    assert "an item copy is mover A's read and mover B's write: its streams take " in (
        message
    )


def test_a_zip_is_refused():
    message = refused(Zipped)
    assert "item is copied by the movers of the loop that reads it, alone" in message
