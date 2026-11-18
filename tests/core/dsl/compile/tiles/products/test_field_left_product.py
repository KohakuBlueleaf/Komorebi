"""A product whose left side is a field (no card): `self.h.write(acc)` (or f(acc), a
step's row) packs DST sums into the field, freed by its next write; `out += f(self.h,
wb)` takes stream row k by field tile k; refused: a field neither written nor filled,
a filled one written, a width not the field's, a stream of other rows."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError

K, KH, N, R = 8, 4, 4, 2
H_BLOCK = 2 * KH  # W1's tiles a block: 2 rows of KH
O_BLOCK = 2 * N  # W2's tiles a block: 2 rows of N


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((R * 128, K * 32), 0x100000),
        "w1": made((K * 32, KH * 32), 0x1000000),
        "w2": made((KH * 32, N * 32), 0x2000000),
        "y": made((R * 128, N * 32), 0x4000000, "float32"),
    }


@km.tile_function
def project(x, w):
    return x @ w


class Unwritten(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        out = km.tile.zeros(km.fp32, N)
        for wb in km.Stream(w2.tiles[0:KH, 0:N], block=O_BLOCK):
            out += project(self.h, wb)
        y.pages(self.rank * N, N).write(out)


class Filled(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        self.h.fill(km.Stream(w2.pages(0, KH)))
        acc = km.tile.zeros(km.fp32, KH)
        for xb, wb in zip(
            km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
            km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
        ):
            acc += project(xb, wb)
        self.h.write(acc)


class Narrow(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for xb, wb in zip(
            km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
            km.Stream(w1.tiles[0:K, 0:2], block=4),
        ):
            acc += project(xb, wb)
        self.h.write(acc)


class Short(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, KH)
        for xb, wb in zip(
            km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
            km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
        ):
            acc += project(xb, wb)
        self.h.write(acc)
        out = km.tile.zeros(km.fp32, N)
        for wb in km.Stream(w2.tiles[0:2, 0:N], block=O_BLOCK):
            out += project(self.h, wb)
        y.pages(self.rank * N, N).write(out)


def folder_of(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return Path(program.built.folder)


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (Unwritten, r"self\.h is neither written above in this handler nor filled"),
        (Filled, r"self\.h is filled, summed into or a body's"),
        (Narrow, r"self\.h holds 4 tile\(s\) and acc is 2"),
        (Short, r"self\.h is a row of 4 tile\(s\) and acc 4: the stream is 4 rows"),
    ],
)
def test_refusals_name_the_line(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        folder_of(cls)
