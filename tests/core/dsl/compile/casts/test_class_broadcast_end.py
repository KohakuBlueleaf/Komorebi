"""Where compute's padding pop of a broadcast (`km::broadcast_end`) stands in the
generated text (no card): after the block's last use, in the kernel form and in the
runtime's, for a cast outside every loop and for one in a loop; a broadcast casts
outside a loop once."""

import pytest
from class_lines_fixture import tensor

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import (
    compute_source,
    movement_source,
    runtime_compute_source,
    runtime_movement_source,
)


class Rowcast(km.Program):
    """One block outside a loop, window 2: a padding of one page."""

    @km.handler
    def cast(self, x: km.Tensor, out: km.Tensor):
        row = km.Broadcast(self, along="row", window=2)
        if row.sender():  # noqa: SIM108 (the compiler's form)
            tiles = row.cast(x.pages(self.y, 1).read())
        else:
            tiles = row.arrive()
        total = km.tile.zeros(km.fp32)
        for tile in tiles:
            total += tile
        out.pages(self.rank, 1).write(total)


class Narrow(km.Program):
    """The same with a window of 1: no padding."""

    @km.handler
    def cast(self, x: km.Tensor, out: km.Tensor):
        row = km.Broadcast(self, along="row", window=1)
        if row.sender():  # noqa: SIM108 (the compiler's form)
            tiles = row.cast(x.pages(self.y, 1).read())
        else:
            tiles = row.arrive()
        total = km.tile.zeros(km.fp32)
        for tile in tiles:
            total += tile
        out.pages(self.rank, 1).write(total)


class Two(km.Program):
    """The cast in a loop of 2 blocks of a window of 2: no padding."""

    @km.handler
    def cast(self, x: km.Tensor, out: km.Tensor):
        row = km.Broadcast(self, along="row", window=2)
        total = km.tile.zeros(km.fp32)
        for block in range(2):
            if row.sender():
                tiles = row.cast(x.pages(self.y * 2 + block, 1).read())
            else:
                tiles = row.arrive()
            for tile in tiles:
                total += tile
        out.pages(self.rank, 1).write(total)


class Three(km.Program):
    """The same in a loop of 3: a padding of one page."""

    @km.handler
    def cast(self, x: km.Tensor, out: km.Tensor):
        row = km.Broadcast(self, along="row", window=2)
        total = km.tile.zeros(km.fp32)
        for block in range(3):
            if row.sender():
                tiles = row.cast(x.pages(self.y * 3 + block, 1).read())
            else:
                tiles = row.arrive()
            for tile in tiles:
                total += tile
        out.pages(self.rank, 1).write(total)


def twice():
    class Twice(km.Program):
        @km.handler
        def cast(self, x: km.Tensor, out: km.Tensor):
            row = km.Broadcast(self, along="row", window=2)
            if row.sender():
                first = row.cast(x.pages(self.y, 1).read())
            else:
                first = row.arrive()
            for _tile in first:
                pass
            if row.sender():
                second = row.cast(x.pages(self.y + 1, 1).read())
            else:
                second = row.arrive()
            for _tile in second:
                pass

    return Twice


def sources(cls):
    """{form: (movement, compute)} of the class in the kernel and the runtime's form."""
    both = {"x": tensor("bfloat16"), "out": tensor("bfloat16")}
    compiled = compile_class(cls, {}, both)
    return {
        "kernel": (movement_source(compiled), compute_source(compiled)),
        "runtime": (
            runtime_movement_source(compiled),
            runtime_compute_source(compiled),
        ),
    }


def at(text, needle):
    """The offset of the only line holding `needle` (the call, not an include)."""
    found = [n for n, line in enumerate(text.split("\n")) if needle in line]
    assert len(found) == 1, (needle, found)
    return found[0]


@pytest.mark.parametrize("form", ["kernel", "runtime"])
def test_a_cast_outside_a_loop_pops_its_padding_after_the_block_is_used(form):
    movement, compute = sources(Rowcast)[form]
    assert "kBroadcast0Window, 1)" in compute  # one block, window 2: a page of padding
    receive = at(compute, "km::broadcast_receive(")
    done = at(compute, "km::broadcast_done(")
    end = at(compute, "km::broadcast_end(")
    assert receive < done < end
    # the mover's finish stays at the cast, before any later line of the handler
    assert at(movement, ".finish()") < at(movement, "write_pages")


@pytest.mark.parametrize("form", ["kernel", "runtime"])
def test_the_end_stands_after_the_loop_of_a_cast_in_a_loop(form):
    for blocks, cls in ((2, Two), (3, Three)):
        _, compute = sources(cls)[form]
        done = at(compute, "km::broadcast_done(")
        end = at(compute, "km::broadcast_end(")
        assert done < end
        assert f"kBroadcast0Window, {blocks})" in compute
        before = compute.split("km::broadcast_end(")[0]
        assert before.count("{") > before.count("}")  # in the handler's case


def test_a_window_of_one_has_no_padding_and_the_end_follows_the_use_too():
    for _, compute in sources(Narrow).values():
        assert at(compute, "km::broadcast_done(") < at(compute, "km::broadcast_end(")


def test_a_broadcast_casts_outside_a_loop_once():
    with pytest.raises(Refused, match=r"casts outside a loop already"):
        compile_class(twice(), {}, {"x": tensor("bfloat16"), "out": tensor("bfloat16")})
