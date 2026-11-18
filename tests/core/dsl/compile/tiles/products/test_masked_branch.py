"""A field of masked sums over an if / else (no card): a field written by a step of
sums started from a mask row is masked (zero past the keys), and a product of it
skips its run-time window check; written so on one branch only, it is not masked
after the if, whichever branch masks it, and the product after keeps the check."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor

K, KH, N = 4, 2, 2
H_BLOCK = 2 * KH
CHECK = "the window inside the cast's rectangle"


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((128, K * 32), 0x100000),
        "w1": made((K * 32, KH * 32), 0x1000000),
        "w2": made((4 * KH * 32, N * 32), 0x2000000),
        "init": made((32, 32 * (1 + KH)), 0x3000000, "float32"),
        "y": made((128, N * 32), 0x4000000, "float32"),
    }


@km.tile_function
def project(x, w):
    return x @ w


@km.tile_function
def start(row, state):
    return row, (state[0],)


@km.tile_function
def step(row, state):
    return row, (state[0],)


class MaskedFirst(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    mask: km.L1[km.fp32, 1, KH, 32, 32]
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(
        self,
        x: km.Tensor,
        w1: km.Tensor,
        w2: km.Tensor,
        init: km.Tensor,
        y: km.Tensor,
        rows: int,
    ):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.mask.fill(km.Stream(init.pages(1, KH)))
        column = km.Broadcast(self, along="column", window=1, mover="A")
        if column.sender():
            ws = column.cast(w2.tiles[0 : min(0 + KH, rows), 0:N].read())
        else:
            ws = column.arrive()
        for r in range(2):
            if r < 1:
                s = start(self.mask[0], self.state[0])
                for xb, wb in zip(
                    km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2),
                    km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
                ):
                    s += project(xb, wb)
                self.h.write(step(s, self.state[0]))
            else:
                s = km.tile.zeros(km.fp32, KH)
                for xb, wb in zip(
                    km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2),
                    km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
                ):
                    s += project(xb, wb)
                self.h.write(step(s, self.state[0]))
            out = km.tile.zeros(km.fp32, N)
            out += project(self.h, ws[0:KH, :])
            y.pages(self.y * N, N).write(out)


class MaskedSecond(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    mask: km.L1[km.fp32, 1, KH, 32, 32]
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(
        self,
        x: km.Tensor,
        w1: km.Tensor,
        w2: km.Tensor,
        init: km.Tensor,
        y: km.Tensor,
        rows: int,
    ):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.mask.fill(km.Stream(init.pages(1, KH)))
        column = km.Broadcast(self, along="column", window=1, mover="A")
        if column.sender():
            ws = column.cast(w2.tiles[0 : min(0 + KH, rows), 0:N].read())
        else:
            ws = column.arrive()
        for r in range(2):
            if r < 1:
                s = km.tile.zeros(km.fp32, KH)
                for xb, wb in zip(
                    km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2),
                    km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
                ):
                    s += project(xb, wb)
                self.h.write(step(s, self.state[0]))
            else:
                s = start(self.mask[0], self.state[0])
                for xb, wb in zip(
                    km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2),
                    km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
                ):
                    s += project(xb, wb)
                self.h.write(step(s, self.state[0]))
            out = km.tile.zeros(km.fp32, N)
            out += project(self.h, ws[0:KH, :])
            y.pages(self.y * N, N).write(out)


class MaskedBoth(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    mask: km.L1[km.fp32, 1, KH, 32, 32]
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(
        self,
        x: km.Tensor,
        w1: km.Tensor,
        w2: km.Tensor,
        init: km.Tensor,
        y: km.Tensor,
        rows: int,
    ):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.mask.fill(km.Stream(init.pages(1, KH)))
        column = km.Broadcast(self, along="column", window=1, mover="A")
        if column.sender():
            ws = column.cast(w2.tiles[0 : min(0 + KH, rows), 0:N].read())
        else:
            ws = column.arrive()
        for r in range(2):
            if r < 1:
                s = start(self.mask[0], self.state[0])
                for xb, wb in zip(
                    km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2),
                    km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
                ):
                    s += project(xb, wb)
                self.h.write(step(s, self.state[0]))
            else:
                s = start(self.mask[0], self.state[0])
                for xb, wb in zip(
                    km.Stream(x.tiles[self.y : self.y + 1, 0:K], block=2),
                    km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
                ):
                    s += project(xb, wb)
                self.h.write(step(s, self.state[0]))
            out = km.tile.zeros(km.fp32, N)
            out += project(self.h, ws[0:KH, :])
            y.pages(self.y * N, N).write(out)


def text_of(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return (Path(program.built.folder) / "compute.cpp").read_text()


@pytest.mark.parametrize(
    "cls, checked",
    [(MaskedFirst, True), (MaskedSecond, True), (MaskedBoth, False)],
)
def test_the_product_after_the_if_checks_its_window_unless_masked_on_each_branch(
    cls, checked
):
    text = text_of(cls)
    assert (CHECK in text[text.index("out += project(self.h") :]) == checked
