"""Products of a field and a broadcast's block (no card): `acc += f(self.h, w[r0:r0 +
n, :])` takes tile k of the field by row r0 + k of the block, any number a block,
the block waited for at its cast and freed at the end of the cast's lines; refused:
a field not written, rows not the field's tiles, a stream on the cast's mover."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor

K, KH, N, R = 4, 2, 2, 2
W2_ROWS = 4  # R x KH: W2's rows, a row's KH each
H_BLOCK = 2 * KH


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((R * 128, K * 32), 0x100000),
        "w1": made((K * 32, KH * 32), 0x1000000),
        "w2": made((2 * KH * 32, N * 32), 0x2000000),
        "y": made((R * 128, N * 32), 0x4000000, "float32"),
    }


@km.tile_function
def project(x, w):
    return x @ w


class Rows(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w2.tiles[0:W2_ROWS, 0:N].read())
        else:
            ws = column.arrive()
        for r in range(R):
            acc = km.tile.zeros(km.fp32, KH)
            for xb, wb in zip(
                km.Stream(
                    x.tiles[r * 4 + self.y : r * 4 + self.y + 1, 0:K],
                    block=2,
                    pattern="A",
                ),
                km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK, pattern="A"),
            ):
                acc += project(xb, wb)
            self.h.write(acc)
            out = km.tile.zeros(km.fp32, N)
            out += project(self.h, ws[r * KH : r * KH + COUNT, :])
            y.pages((r * 4 + self.y) * N, N).write(out)


class OnB(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w2.tiles[0:W2_ROWS, 0:N].read())
        else:
            ws = column.arrive()
        for r in range(R):
            acc = km.tile.zeros(km.fp32, KH)
            for xb, wb in zip(
                km.Stream(
                    x.tiles[r * 4 + self.y : r * 4 + self.y + 1, 0:K],
                    block=2,
                    pattern="B",
                ),
                km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK, pattern="B"),
            ):
                acc += project(xb, wb)
            self.h.write(acc)
            out = km.tile.zeros(km.fp32, N)
            out += project(self.h, ws[r * KH : r * KH + COUNT, :])
            y.pages((r * 4 + self.y) * N, N).write(out)


class Unwritten(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        if column.sender():
            ws = column.cast(w2.tiles[0:KH, 0:N].read())
        else:
            ws = column.arrive()
        out = km.tile.zeros(km.fp32, N)
        out += project(self.h, ws[0:KH, :])
        y.pages(self.y * N, N).write(out)


COUNT = KH


def folder_of(cls, pattern="A", count=KH):
    global PATTERN, COUNT
    PATTERN, COUNT = pattern, count
    try:
        program = cls(l1=1 << 20, log=quiet, rect=(0, 0, 2, 4), **tensors())
    finally:
        COUNT = KH
    return Path(program.built.folder)


def test_the_block_is_waited_at_its_cast_and_freed_after_every_product():
    text = (folder_of(Rows) / "compute.cpp").read_text()
    received = text.index(
        "km::broadcast_receive(compute, kBroadcast0, kBroadcast0Pages)"
    )
    rows = text.index("for (uint32_t r = 0;", received)
    init = text.index("matmul_block_init(kFieldH, kBroadcast0, 0, 2, 1, 2);", rows)
    first = text.index(
        "matmul_block(kFieldH, kBroadcast0, 0, 0 + ((r * 2) + 0) * 2, 2, 0, 2, 1, 2);",
        init,
    )
    second = text.index(
        "matmul_block(kFieldH, kBroadcast0, 1, 0 + ((r * 2) + 1) * 2, 2, 0, 2, 1, 2);",
        first,
    )
    done = text.index("km::broadcast_done(kBroadcast0, kBroadcast0Pages);", second)
    assert received < rows < init < first < second < done
    assert text.count("km::broadcast_receive(") == 1
    assert text.count("km::broadcast_done(") == 1


@pytest.mark.parametrize(
    "cls, options, refusal",
    [
        (Unwritten, {}, r"self\.h is not written above in this handler"),
        (Rows, {"count": 1}, r"self\.h is 2 tile\(s\) and the block's rows 1"),
        (OnB, {}, r"ws was cast outside a loop and products take it"),
    ],
)
def test_refusals_name_the_line(cls, options, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        folder_of(cls, **options)
