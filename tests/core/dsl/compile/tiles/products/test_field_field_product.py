"""A product of two fields (no card): `out += f(self.h, self.w2)`, self.h a row of K
tiles compute wrote, self.w2 a K x n field another handler fills (kept): tile k of
self.h by row k of self.w2 into out's DST; refused: shapes that disagree (both
named), a left field not written above, a pair of formats of two families."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError

K, KH, N, R = 2, 2, 2, 2  # input tiles, hidden tiles, output tiles, rows a core


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((8 * 32, K * 32), 0x100000),
        "w1": made((K * 32, KH * 32), 0x1000000),
        "w2": made((KH * 32, N * 32), 0x2000000),
        "y": made((8 * 32, N * 32), 0x4000000),
        "v": made((KH * 32, N * 32), 0x3000000, "bfp8"),
    }


@km.tile_function
def project(a, b):
    return a @ b


class Wide(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]
    w2: km.L1[km.bf16, KH + 1, N, 32, 32]

    @km.handler
    def mlp(self, x: km.Tensor, w2: km.Tensor, y: km.Tensor):
        self.w2.fill(km.Stream(w2.pages(0, (KH + 1) * N)))
        acc = km.tile.zeros(km.fp32, KH)
        for xb, wb in zip(
            km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=K),
            km.Stream(w2.tiles[0:K, 0:KH], block=4),
        ):
            acc += project(xb, wb)
        self.h.write(acc)
        out = km.tile.zeros(km.fp32, N)
        out += project(self.h, self.w2)
        y.pages(self.rank * N, N).write(out)


class Unwritten(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]
    w2: km.L1[km.bf16, KH, N, 32, 32]

    @km.handler
    def mlp(self, w2: km.Tensor, y: km.Tensor):
        self.w2.fill(km.Stream(w2.tiles[0:KH, 0:N]))
        out = km.tile.zeros(km.fp32, N)
        out += project(self.h, self.w2)
        y.pages(self.rank * N, N).write(out)


class Families(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]
    w2: km.L1[km.bfp8, KH, N, 32, 32]

    @km.handler
    def mlp(self, x: km.Tensor, w2: km.Tensor, v: km.Tensor, y: km.Tensor):
        self.w2.fill(km.Stream(v.pages(0, KH * N)))
        acc = km.tile.zeros(km.fp32, KH)
        for xb, wb in zip(
            km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=K),
            km.Stream(w2.tiles[0:K, 0:KH], block=4),
        ):
            acc += project(xb, wb)
        self.h.write(acc)
        out = km.tile.zeros(km.fp32, N)
        out += project(self.h, self.w2)
        y.pages(self.rank * N, N).write(out)


def compute_of(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return (Path(program.built.folder) / "compute.cpp").read_text()


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (Wide, r"self\.h is a row of 2 tile\(s\) and self\.w2 3 x 2 tiles"),
        (Unwritten, r"self\.h is not written above in this handler"),
        (Families, r"a product of bfloat16 and bfp8: its operands are of one"),
    ],
)
def test_refusals_name_the_line(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        compute_of(cls)
