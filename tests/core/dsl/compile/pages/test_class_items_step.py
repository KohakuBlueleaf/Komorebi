"""Items with a step: `t.items(first=, count=, size=, step=)`, one item of each of
several pages as a walk of one item a row; and the refusals (no card)."""

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def words(pages):
    """A uint32 tensor of 8192 B pages."""
    return Tensor(Pages(8192, pages), (pages, 2048), "uint32", "row")


def movement_of(cls, handler, **tensors):
    compiled = compile_class(cls, {}, tensors)
    return "\n".join(render(prune(compiled.movement[handler]), 0))


class Slices(km.Program):
    @km.handler
    def gather(self, counters: km.Tensor, out: km.Tensor, cores: int, slice: int):
        seen = counters.items(first=slice, count=cores, size=64, step=128)
        for item in km.Stream(seen, block=4):
            km.cxx("add(item);", takes=[item])
        km.Stream(out.items(first=slice, count=1, size=64)).drain()


def test_items_with_a_step_are_a_walk_of_one_item_a_row():
    movement = movement_of(Slices, "gather", counters=words(8), out=words(1))
    # the items of 64 B of the counters' pages, 128 a page; a row an item, a block 4
    assert "km::items(Counters(counters), 64, 128), km::Walk{slice, " in movement
    walk = "((cores) + 3) / 4, 4 * 128, 4, 128, 1, 1, 1, 4, (cores) - (((cores) + 3)"
    assert walk in movement
    # no step: the run of items as before
    assert "km::items(Out(out), 64, 128), slice, 1, kRead1Block" in movement


class Product(km.Program):
    @km.handler
    def gather(self, counters: km.Tensor, n: int):
        km.Stream(counters.items(first=0, count=n * 2, size=64, step=128)).drain()


class Zero(km.Program):
    @km.handler
    def gather(self, counters: km.Tensor, n: int):
        km.Stream(counters.items(first=0, count=n, size=64, step=0)).drain()


class NoCount(km.Program):
    @km.handler
    def gather(self, counters: km.Tensor, n: int):
        km.Stream(counters.items(first=0, size=64, step=2)).drain()


class Wide(km.Program):
    @km.handler
    def gather(self, counters: km.Tensor, n: int):
        km.Stream(counters.items(first=0, count=n, size=100)).drain()


@pytest.mark.parametrize(
    "cls, why",
    [
        (Product, "count= n * 2: the build bounds an int or a word"),
        (Zero, "step= 0: 1 or more"),
        (NoCount, "give first=, count= and size="),
        (Wide, "a page holds a whole number of items"),
    ],
)
def test_a_count_or_step_the_build_cannot_bound_is_refused(cls, why):
    with pytest.raises(Refused, match=why.replace("*", r"\*")):
        compile_class(cls, {}, {"counters": words(8)})
