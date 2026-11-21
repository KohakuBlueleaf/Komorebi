"""A field filled on both movers (no card): `self.f.fill(km.Stream(t.pages(first,
n), block=b, pattern="AB"))` reads mover A's blocks on mover A and mover B's on mover
B (km::fill_field on each), the plain fill's kernels unchanged; refused: one option
alone, a rectangle, another option."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError

K, N = 4, 2  # a field of K x N tiles


def tensors():
    def made(shape, address, dtype="bfp8_b"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "w": made((64 * 32, K * N * 32), 0x1000000),
        "x": made((32, K * 32), 0x2000000, "bfloat16"),
        "y": made((8 * 32, N * 32), 0x3000000, "bfloat16"),
    }


@km.tile_function
def project(a, b):
    return a @ b


class Letters(km.Program):
    wf: km.L1[km.bfp8_b, K, N, 32, 32]

    @km.handler
    def load(self, w: km.Tensor):
        own = self.rank * K * N
        self.wf.fill(km.Stream(w.pages(own, K * N), block=2, pattern="AB"))

    @km.handler
    def apply(self, x: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[0:1, 0:K], block=K):
            acc += project(xb, self.wf)
        y.tiles[self.rank : self.rank + 1, 0:N].write(acc)


class Plain(km.Program):
    wf: km.L1[km.bfp8_b, K, N, 32, 32]

    @km.handler
    def run(self, w: km.Tensor, x: km.Tensor, y: km.Tensor, pattern: km.Pattern):
        own = self.rank * K * N
        self.wf.fill(km.Stream(w.pages(own, K * N)))
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[0:1, 0:K], block=K):
            acc += project(xb, self.wf)
        y.tiles[self.rank : self.rank + 1, 0:N].write(acc)


class Word(km.Program):
    wf: km.L1[km.bfp8_b, K, N, 32, 32]

    @km.handler
    def run(self, w: km.Tensor, x: km.Tensor, y: km.Tensor, pattern: km.Pattern):
        own = self.rank * K * N
        self.wf.fill(km.Stream(w.pages(own, K * N), block=3, pattern=pattern))
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[0:1, 0:K], block=K):
            acc += project(xb, self.wf)
        y.tiles[self.rank : self.rank + 1, 0:N].write(acc)


class BlockAlone(km.Program):
    wf: km.L1[km.bfp8_b, K, N, 32, 32]

    @km.handler
    def run(self, w: km.Tensor, x: km.Tensor, y: km.Tensor, pattern: km.Pattern):
        own = self.rank * K * N
        self.wf.fill(km.Stream(w.pages(own, K * N), block=2))
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[0:1, 0:K], block=K):
            acc += project(xb, self.wf)
        y.tiles[self.rank : self.rank + 1, 0:N].write(acc)


class Rectangle(km.Program):
    wf: km.L1[km.bfp8_b, K, N, 32, 32]

    @km.handler
    def run(self, w: km.Tensor, x: km.Tensor, y: km.Tensor, pattern: km.Pattern):
        self.wf.fill(km.Stream(w.tiles[0:K, 0:N], block=2, pattern="AB"))
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[0:1, 0:K], block=K):
            acc += project(xb, self.wf)
        y.tiles[self.rank : self.rank + 1, 0:N].write(acc)


class BFirst(km.Program):
    wf: km.L1[km.bfp8_b, K, N, 32, 32]

    @km.handler
    def run(self, w: km.Tensor, x: km.Tensor, y: km.Tensor, pattern: km.Pattern):
        own = self.rank * K * N
        self.wf.fill(km.Stream(w.pages(own, K * N), block=2, pattern="BA"))
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[0:1, 0:K], block=K):
            acc += project(xb, self.wf)
        y.tiles[self.rank : self.rank + 1, 0:N].write(acc)


class Repeated(km.Program):
    wf: km.L1[km.bfp8_b, K, N, 32, 32]

    @km.handler
    def run(self, w: km.Tensor, x: km.Tensor, y: km.Tensor, pattern: km.Pattern):
        own = self.rank * K * N
        self.wf.fill(km.Stream(w.pages(own, K * N), block=2, repeat=2))
        acc = km.tile.zeros(km.fp32, N)
        for xb in km.Stream(x.tiles[0:1, 0:K], block=K):
            acc += project(xb, self.wf)
        y.tiles[self.rank : self.rank + 1, 0:N].write(acc)


def mover_text(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return (Path(program.built.folder) / "movement.cpp").read_text()


@pytest.mark.parametrize(
    "cls, call",
    [
        (Letters, 'kFieldWfPages, 2, km::pattern("AB"), kFieldWf)'),
        (Word, "kFieldWfPages, 3, pattern, kFieldWf)"),
        (BFirst, 'kFieldWfPages, 2, km::pattern("BA"), kFieldWf)'),
    ],
)
def test_both_movers_fill_their_blocks(cls, call):
    text = mover_text(cls)
    assert text.count("km::fill_field(movement, command,") == 2, text
    assert call in text
    assert "km::read_pages(movement, command, W(w)" not in text


def test_the_plain_fill_is_mover_a_alone():
    text = mover_text(Plain)
    assert "km::fill_field" not in text
    assert text.count("km::read_pages(movement, command, W(w)") == 1


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (BlockAlone, r"a fill on both movers names its block= and its pattern="),
        (Rectangle, r"a fill of a rectangle of tiles is mover A's"),
        (Repeated, r"a field is filled from a stream of a page range"),
    ],
)
def test_refusals_name_the_line(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        mover_text(cls)
