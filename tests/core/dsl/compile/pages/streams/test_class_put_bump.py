"""An item put at a word index (`t.items(size=B).put(source, at=i)`) and a field's
word bumped (`slot = self.f.bump(i)`): what each lowers to and refuses (no card)."""

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def lists():
    """A uint32 tensor of 2048 B pages, 16 items of 128 B a page."""
    return Tensor(Pages(2048, 4), (4, 512), "uint32", "row")


class OtherMover(km.Program):
    record: km.L1[km.u32, 1, 32, 32] = km.own(push=None)
    next_slots: km.L1[km.u32, 1, 32, 32] = km.own(push="mover_b")

    @km.handler
    def place(self, lists: km.Tensor, first: int):
        slot = self.next_slots.bump(first)
        lists.items(size=128).put(self.record, at=slot, on="mover_a")


class NotOwn(km.Program):
    @km.handler
    def place(self, lists: km.Tensor, first: int):
        lists.items(size=128).put(lists, at=first)


class NoOwner(km.Program):
    next_slots: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def place(self, lists: km.Tensor, first: int):
        slot = self.next_slots.bump(first)
        km.cxx("use(slot);", takes=[slot], on="mover_b")


class Compute(km.Program):
    record: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def place(self, lists: km.Tensor, first: int):
        lists.items(size=128).put(self.record, at=first, on="pack")


class Odd(km.Program):
    record: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def place(self, lists: km.Tensor, first: int):
        lists.items(size=100).put(self.record, at=first)


@pytest.mark.parametrize(
    "cls, why",
    [
        (OtherMover, "slot is a word of mover B, and this line runs on mover A"),
        (NotOwn, "a put's source is an own field"),
        (NoOwner, "self.next_slots's words are no one CPU's"),
        (Compute, "a put is a mover's NoC write"),
        (Odd, "a put is whole words, a page a whole number of items"),
    ],
)
def test_each_refusal(cls, why):
    with pytest.raises(Refused, match=why):
        compile_class(cls, {}, {"lists": lists()})
