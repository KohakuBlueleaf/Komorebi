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


@km.tile_function
def act(x):
    return km.relu(x)


class Both(km.Program):
    w1: km.L1[km.bf16, K, KH, 32, 32]
    w2: km.L1[km.bf16, KH, N, 32, 32]
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def load(self, w1: km.Tensor, w2: km.Tensor):
        self.w1.fill(km.Stream(w1.tiles[0:K, 0:KH]))
        self.w2.fill(km.Stream(w2.tiles[0:KH, 0:N]))

    @km.handler
    def mlp(self, x: km.Tensor, y: km.Tensor):
        for r in range(R):
            row = self.rank * R + r
            acc = km.tile.zeros(km.fp32, KH)
            for xb in km.Stream(x.tiles[row : row + 1, 0:K], block=K):
                acc += project(xb, self.w1)
            self.h.write(act(acc))
            out = km.tile.zeros(km.fp32, N)
            out += project(self.h, self.w2)
            y.pages(row * N, N).write(out)


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


def test_tile_k_of_the_written_field_by_row_k_of_the_filled_one():
    text = compute_of(Both)
    mlp = text[text.index("case kHandlerMlp:") :]
    packed = mlp.index("cb_push_back(kFieldH, 2);")
    weights = mlp.index("km_ok = km_ok && compute.wait_front(kFieldW2, kFieldW2Pages);")
    hidden = mlp.index("km_ok = km_ok && compute.wait_front(kFieldH, kFieldHPages);")
    init = mlp.index("matmul_block_init(kFieldH, kFieldW2, 0, 2, 1, 2);")
    first = mlp.index("matmul_block(kFieldH, kFieldW2, 0, 0, 2, 0, 2, 1, 2);")
    second = mlp.index("matmul_block(kFieldH, kFieldW2, 1, 2, 2, 0, 2, 1, 2);")
    assert packed < weights < hidden < init < first < second
    assert "cb_pop_front(kFieldW2, kFieldW2Pages);" not in mlp  # kept: load fills it


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
