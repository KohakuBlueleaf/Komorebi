"""Which streams of items the movers move (no card): a DRAM tensor's items are a
multiple of 64 B (Blackhole's DRAM read alignment), read or written, in both forms of
`t.items`; an L1 tensor's need not be. On islands a stream of items is refused by
name: it reads one island's tensor."""

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensors(shard_grid=None):
    """Two uint32 tensors of 4,096 B rows: t and out (in L1 with a shard grid)."""
    return {
        n: Tensor(Pages(4096), (64, 1024), "uint32", "row", shard_grid=shard_grid)
        for n in ("t", "out")
    }


class Read(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor):
        km.Stream(t.items(0, 8, size=32)).drain()


class Written(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor):
        results = km.Stream(out.items(0, 8, size=32), write=True)
        for item in km.Stream(t.items(0, 8, size=64)):
            results.push(item)


class Stepped(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor):
        km.Stream(t.items(first=0, count=8, size=32, step=4)).drain()


class Copied(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor):
        results = km.Stream(out.items(0, 8, size=64), write=True)
        for item in km.Stream(t.items(0, 8, size=64)):
            results.push(item)


def refused(cls, **given):
    with pytest.raises(Refused) as refusal:
        compile_class(cls, {}, tensors(), **given)
    return str(refusal.value)


@pytest.mark.parametrize("cls, tensor", [(Read, "t"), (Written, "out"), (Stepped, "t")])
def test_a_dram_tensors_items_of_32_bytes_are_refused(cls, tensor):
    message = refused(cls)
    assert f"items of 32 B of {tensor}, a DRAM tensor: Blackhole reads DRAM at " in (
        message
    )
    assert "give a size that is a multiple of 64" in message


@pytest.mark.parametrize(
    "cls, given, options",
    [(Read, {"shard_grid": (1, 1)}, {}), (Copied, {}, {"islands": (2, 64)})],
)
def test_items_an_l1_tensor_or_islands_take_build(cls, given, options):
    compile_class(cls, {}, tensors(**given), **options)  # not refused


def test_items_on_islands_not_a_multiple_of_64_bytes_are_refused():
    message = refused(Read, islands=(2, 64))
    assert "items of 32 B of t, and the class is built for 2 islands: an item " in (
        message
    )
    assert "give a size that is a multiple of 64" in message
