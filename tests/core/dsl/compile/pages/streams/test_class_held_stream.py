"""A read stream that bodies take (a held stream): one block, moved at its line,
waited for there, taken by bodies in the loops inside its block and freed when that
block's pass ends; and each refusal (no card)."""

from pathlib import Path

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


def grid(rows, columns):
    """A bfloat16 tensor of tiles, rows x columns of them."""
    return Tensor(Pages(2048, rows * columns), (rows * 32, columns * 32), "bfloat16",
                  "tile")  # fmt: skip


def line_of(text):
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


def texts(cls, handler, **tensors):
    """(movement, compute) C++ of one handler of `cls`."""
    compiled = compile_class(cls, {}, tensors)
    return (
        "\n".join(render(prune(compiled.movement[handler]), 0)),
        "\n".join(render(prune(compiled.compute[handler]), 0)),
    )


@km.tile_function
def twice(a):
    return a + a


class Held(km.Program):
    @km.handler
    def copy(self, q: km.Tensor, k: km.Tensor, v: km.Tensor, rows: int, jobs: int):
        down = km.Broadcast(self, along="column")
        for job in range(jobs):
            r0 = job * 4
            qs = km.Stream(q.tiles[r0 : min(r0 + 4, r0 + rows), 0:2], block=8)
            for chunk in range(2):
                if down.sender():
                    kv = down.cast(k.tiles[chunk : chunk + 1, 0:2])
                else:
                    kv = down.arrive()
                km.cxx("score(qs_buffer, qs, kv_buffer, kv);", takes=[qs, kv])
            km.cxx("finish(qs_buffer);", takes=[qs])


def test_a_read_stream_bodies_take_is_held_across_the_loops_inside():
    tensors = {"q": grid(16, 2), "k": grid(8, 2),
               "v": grid(8, 2)}  # fmt: skip
    _, compute = texts(Held, "copy", **tensors)
    # a held operand: its block freed once, after the loops that read it
    assert compute.count("cb_pop_front(km_held_qs, kRead0Block);") == 1
    # one block a half: the block is held, so no second is read ahead into it
    compiled = compile_class(Held, {}, tensors)
    (pair,) = [b for b in compiled.buffers if b.name == "kRead0"]
    assert pair.pages == (8, 8)


class HeldLarge(km.Program):
    @km.handler
    def copy(self, q: km.Tensor, k: km.Tensor, v: km.Tensor, rows: int, jobs: int):
        qs = km.Stream(q.tiles[0:rows, 0:2], block=8)  # unbounded
        km.cxx("use(qs);", takes=[qs])


class HeldBlocks(km.Program):
    @km.handler
    def copy(self, q: km.Tensor, k: km.Tensor, v: km.Tensor, rows: int, jobs: int):
        qs = km.Stream(q.tiles[0:8, 0:2], block=8)  # two blocks
        km.cxx("use(qs);", takes=[qs])


class HeldMover(km.Program):
    @km.handler
    def copy(self, q: km.Tensor, k: km.Tensor, v: km.Tensor, rows: int, jobs: int):
        qs = km.Stream(q.tiles[0:4, 0:2], block=8)
        km.cxx("use(qs);", takes=[qs], on="mover_a")  # on a mover


class HeldB(km.Program):
    @km.handler
    def copy(self, q: km.Tensor, k: km.Tensor, v: km.Tensor, rows: int, jobs: int):
        qs = km.Stream(q.tiles[0:4, 0:2], block=8, pattern="B")  # mover B's
        km.cxx("use(qs);", takes=[qs])


class HeldIterated(km.Program):
    @km.handler
    def copy(self, q: km.Tensor, k: km.Tensor, v: km.Tensor, rows: int, jobs: int):
        qs = km.Stream(q.tiles[0:4, 0:2], block=8)
        for _tile in qs:  # iterated
            pass


class HeldTraced(km.Program):
    @km.handler
    def copy(self, q: km.Tensor, k: km.Tensor, v: km.Tensor, rows: int, jobs: int):
        qs = km.Stream(q.tiles[0:4, 0:2], block=8)
        t = twice(qs)  # traced
        k.tiles[0:1, 0:1].write(t)


class HeldAfter(km.Program):
    @km.handler
    def copy(self, q: km.Tensor, k: km.Tensor, v: km.Tensor, rows: int, jobs: int):
        for job in range(jobs):
            qs = km.Stream(q.tiles[job : job + 1, 0:2], block=2)
            km.cxx("use(qs);", takes=[qs])
        km.cxx("use(qs);", takes=[qs])  # after its lines


@pytest.mark.parametrize(
    "cls, text, why",
    [
        (HeldLarge, "# unbounded", "its pages are at most unknown to the build"),
        (HeldBlocks, "# two blocks", "its pages are at most 16, its block 8"),
        (HeldMover, "# on a mover", "a block of compute's three CPUs"),
        (HeldB, "# mover B's", "a stream bodies take is mover A's block"),
        (
            HeldAfter,
            "# after its lines",
            "qs is first assigned inside the lines around",
        ),
    ],
)
def test_held_stream_refusals(cls, text, why):
    line = line_of(text)
    tensors = {k: grid(16, 2) for k in "qkv"}
    with pytest.raises(Refused, match=rf"^{cls.__name__}.copy, line {line}, .*{why}"):
        compile_class(cls, {}, tensors)


class HeldTop(km.Program):
    @km.handler
    def copy(self, q: km.Tensor, k: km.Tensor, v: km.Tensor, rows: int, jobs: int):
        qs = km.Stream(q.tiles[0:4, 0:2], block=8)
        for i in range(jobs):
            km.cxx("use(qs_buffer, i);", takes=[qs, i])
        km.cxx("last(qs_buffer);", takes=[qs])


def test_a_held_stream_at_the_handlers_top_is_freed_at_its_end():
    _, compute = texts(HeldTop, "copy", **{k: grid(16, 2) for k in "qkv"})
    # a held operand: its block freed once, at the handler's end
    assert compute.count("cb_pop_front(km_held_qs, kRead0Block);") == 1
