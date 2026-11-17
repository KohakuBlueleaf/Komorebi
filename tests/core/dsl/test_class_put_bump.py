"""An item put at a word index (`t.items(size=B).put(source, at=i)`) and a field's
word bumped (`slot = self.f.bump(i)`): what each lowers to and refuses (no card)."""

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.nodes import prune, render


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def lists():
    """A uint32 tensor of 2048 B pages, 16 items of 128 B a page."""
    return Tensor(Pages(2048, 4), (4, 512), "uint32", "row")


def texts(cls, handler):
    compiled = compile_class(cls, {}, {"lists": lists()})
    return (
        "\n".join(render(prune(compiled.movement[handler]), 0)),
        "\n".join(render(prune(compiled.compute[handler]), 0)),
    )


class Places(km.Program):
    record: km.L1[km.u32, 1, 32, 32] = km.own(push=None)
    next_slots: km.L1[km.u32, 1, 32, 32] = km.own(push="mover_b")

    @km.handler
    def place(self, lists: km.Tensor, first: int):
        km.cxx("build(record);", takes=[self.record], on="mover_b")
        slot = self.next_slots.bump(first)
        lists.items(size=128).put(self.record, at=slot)
        lists.items(size=128).put(self.record[1], at=first + 2, on="mover_a")


def test_a_put_is_a_movers_unwaited_write_and_a_bump_its_word():
    movement, compute = texts(Places, "place")
    # the bump: the word, then one more, on the mover the field's push= names
    assert "km_bumped = km::l1(get_write_ptr(kFieldNextSlots));" in movement
    assert "slot = km_bumped[first]++;" in movement
    # the put: item `slot` of 16 a page, from the field's start; item 1 of it
    assert (
        "km::put_bytes(movement, command, Lists(lists), (slot) / 16, ((slot) % 16) "
        "* 128, 128, get_write_ptr(kFieldRecord))"
    ) in movement
    assert "get_write_ptr(kFieldRecord) + (1) * 128)" in movement
    # each mover that put waits for its puts at the handler's end, once
    assert movement.count("km::put_wait(movement, command)") == 2
    assert "put_bytes" not in compute


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
