"""A stream compute fills and one mover drains (`out = km.Stream(self.f, mover="B")`),
ended by `out.end()` or by the close of its with block, drained by `for page in
out:` with the bumps and puts a mover's loop holds; and the refusals (no card)."""

import pytest
from class_drained_stream_fixture import lists

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class


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
