"""A stream compute fills and one mover drains (`out = km.Stream(self.f, mover="B")`),
ended by `out.end()` or by the close of its with block, drained by `for page in
out:` with the bumps and puts a mover's loop holds; and the refusals (no card)."""

import re

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


def lists():
    """A uint32 tensor of 2048 B pages, 16 items of 128 B a page."""
    return Tensor(Pages(2048, 4), (4, 512), "uint32", "row")


class Ended(km.Program):
    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")
    next_slots: km.L1[km.u32, 1, 32, 32] = km.own(push="mover_b")

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        for tri in range(count):
            km.cxx("bin(tri, out);", takes=[out, tri], on="pack")
        out.end()
        for page in out:
            slot = self.next_slots.bump(page[1])
            lists.items(size=128).put(page, at=slot)


class Within(km.Program):
    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")
    next_slots: km.L1[km.u32, 1, 32, 32] = km.own(push="mover_b")

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        with km.Stream(self.out, mover="B") as out:
            for tri in range(count):
                km.cxx("bin(tri, out);", takes=[out, tri], on="pack")
        for page in out:
            slot = self.next_slots.bump(page[1])
            lists.items(size=128).put(page, at=slot)


def texts(cls):
    compiled = compile_class(cls, {}, {"lists": lists()})
    return [
        "\n".join(render(prune(nodes["setup"]), 0))
        for nodes in (compiled.movement, compiled.compute)
    ]


def test_compute_fills_and_ends_the_mover_drains_to_the_end():
    movement, compute = texts(Ended)
    # compute: a page reserved before each body, pushed and counted after it
    assert "compute.reserve_back(kFieldOut, 1);" in compute
    assert "cb_push_back(kFieldOut, 1);\n        ++km_pushed_out;" in compute
    # the end: the count, pack's, in the end's page
    end = compute.index("if (compute.reserve_back(kEndOut, 1)) {")
    assert end > compute.index("++km_pushed_out;")
    assert "fifo_wr_ptr << 4)[0] = km_pushed_out));" in compute
    # mover B: a page or the end's count, the page's words its ints, freed after
    assert "km_total_out = km::l1(get_read_ptr(kEndOut))[0];" in movement
    assert "return km_taken_out == km_total_out || cb_pages_available_at_front(" in (
        movement
    )
    assert "slot = km_bumped[page_words[1]]++;" in movement
    assert "128, 128, page)) {" in movement
    pop = movement.index("cb_pop_front(kFieldOut, 1);")
    assert movement.index("noc_async_writes_flushed();") < pop
    assert movement.index("++km_taken_out;") > pop


def test_a_with_blocks_close_is_its_end():
    def lines(text):  # the C++ alone: no comments, no #line, no line numbers
        kept = [
            x for x in text.split("\n") if not x.strip().startswith(("//", "#line"))
        ]
        return re.sub(r"km_room_out\d+", "km_room_out", "\n".join(kept))

    assert [lines(text) for text in texts(Within)] == [
        lines(text) for text in texts(Ended)
    ]


class Unended(km.Program):
    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        km.cxx("bin(out);", takes=[out], on="pack")
        for page in out:
            lists.items(size=128).put(page, at=count)


class Twice(km.Program):
    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        out.end()
        out.end()


class Inside(km.Program):
    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        with km.Stream(self.out, mover="B") as out:
            out.end()


class After(km.Program):
    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        out.end()
        km.cxx("bin(out);", takes=[out], on="pack")


class NoMover(km.Program):
    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out)
        out.end()


class Unfilled(km.Program):
    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        out.end()
        for page in out:
            lists.items(size=128).put(page, at=count)


class Mixed(km.Program):
    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        km.cxx("bin(out);", takes=[out], on="pack")
        out.end()
        for _page in out:
            km.cxx("count();", on="pack")


@pytest.mark.parametrize(
    "cls, why",
    [
        (Unended, "no line ends out above"),
        (Twice, r"out ended at line \d+ already"),
        (Inside, "out ends where its with block closes"),
        (After, r"out ended at line \d+: no page follows"),
        (NoMover, "mover= names it"),
        (Unfilled, "nothing pushes out's pages"),
        (Mixed, "inside a stream mover B drains, mover B's lines alone"),
    ],
)
def test_each_refusal(cls, why):
    with pytest.raises(Refused, match=why):
        compile_class(cls, {}, {"lists": lists()})
