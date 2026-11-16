"""Products of a held stream's row and a broadcast's block (no card): `acc +=
f(xs[r], w[i][:, c0:c0 + n])` takes tile j of row r of the held block by row j of
the block's part, transposed where the cast turned it; refused: a part not of acc's
columns, a row past the held block's, a product inside a stream loop."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor

K, N, R = 2, 2, 2
X_BLOCK = 4  # R x K: the held block


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((R * 4 * 32, K * 32), 0x100000),
        "w": made((K * 32, 4 * N * 32), 0x1000000),
        "wt": made((4 * N * 32, K * 32), 0x2000000),
        "y": made((R * 4 * 32, N * 32), 0x4000000, "float32"),
    }


@km.tile_function
def project(x, w):
    return x @ w


class Held(km.Program):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        xs = km.Stream(x.tiles[self.y * R : self.y * R + R, 0:K], block=X_BLOCK)
        for b in range(2):
            if column.sender():
                kv = column.cast(wt.tiles[b * N : b * N + N, 0:K].T)
            else:
                kv = column.arrive()
            for r in range(R):
                acc = km.tile.zeros(km.fp32, N)
                acc += project(xs[r], kv[:, 0 : 0 + N])
                y.pages((self.y * R + r) * N, N).write(acc)


class Looped(km.Program):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        xs = km.Stream(x.tiles[self.y * R : self.y * R + R, 0:K], block=X_BLOCK)
        for b in range(2):
            if column.sender():
                kv = column.cast(w.tiles[0:K, b * N : b * N + N])
            else:
                kv = column.arrive()
            acc = km.tile.zeros(km.fp32, N)
            for _xb in km.Stream(x.tiles[0:1, 0:K], block=2, pattern="B"):
                acc += project(xs[0], kv)
            y.pages(self.y * N, N).write(acc)


class Past(Held):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        xs = km.Stream(x.tiles[self.y * R : self.y * R + R, 0:K], block=X_BLOCK)
        if column.sender():  # noqa: SIM108 (the compiler's form)
            kv = column.cast(wt.tiles[0:N, 0:K].T)
        else:
            kv = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        acc += project(xs[2], kv)
        y.pages(self.y * N, N).write(acc)


class Narrow(Held):
    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, wt: km.Tensor, y: km.Tensor):
        column = km.Broadcast(self, along="column", window=1)
        xs = km.Stream(x.tiles[self.y * R : self.y * R + R, 0:K], block=X_BLOCK)
        if column.sender():  # noqa: SIM108 (the compiler's form)
            kv = column.cast(wt.tiles[0:N, 0:K].T)
        else:
            kv = column.arrive()
        acc = km.tile.zeros(km.fp32, N)
        acc += project(xs[0], kv[:, 0 : 0 + 1])
        y.pages(self.y * N, N).write(acc)


def folder_of(cls):
    program = cls(l1=1 << 20, log=quiet, rect=(0, 0, 2, 4), **tensors())
    return Path(program.built.folder)


def test_a_held_row_by_a_turned_block_in_the_blocks_lines():
    text = (folder_of(Held) / "compute.cpp").read_text()
    held = text.index("km_ok = km_ok && compute.wait_front(km_held_xs, kRead0Block);")
    blocks = text.index("for (uint32_t b = 0;", held)
    received = text.index("km::broadcast_receive(compute, kBroadcast0, ", blocks)
    rows = text.index("for (uint32_t r = 0;", received)
    init = text.index(
        "matmul_block_init(km_held_xs, kBroadcast0, true, 2, 1, 2);", rows
    )
    second = text.index(
        "matmul_block(km_held_xs, kBroadcast0, (r) * 2 + 1, 0 + 1 * 2 + 0, 0, true, "
        "2, 1, 2);",
        init,
    )
    done = text.index("km::broadcast_done(kBroadcast0, kBroadcast0Pages);", second)
    freed = text.index("cb_pop_front(km_held_xs, kRead0Block);", done)
    assert held < blocks < received < rows < init < second < done < freed


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (Past, r"the held block has 2 rows, 0 \.\. 1"),
        (Narrow, r"the block's part is 1 columns and acc holds 2"),
        (Looped, r"a product of a held stream's row runs on its own"),
    ],
)
def test_refusals_name_the_line(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        folder_of(cls)
