"""A field's rows divided by a per-row value at their write (no card): `last =
norm(self.o, self.s[r])`, norm `row / state[1].col()`, then `pages.write(last)`: the
field and state r through DST, the step form's division by slot 1 a row, the row
packed to the pages and the state back."""

from pathlib import Path

from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.tensor import Tensor

K, N = 4, 2
W_BLOCK = 2 * N


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((128, K * 32), 0x100000),
        "w": made((K * 32, N * 32), 0x1000000),
        "init": made((32, 32), 0x2000000, "float32"),
        "y": made((128, N * 32), 0x3000000, "float32"),
    }


@km.tile_function
def project(x, w):
    return x @ w


@km.tile_function
def norm(row, state):
    return row / state[1].col(), (state[0], state[1])


class Divided(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        acc = km.tile.zeros(km.fp32, N)
        for xb, wb in zip(
            km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
            km.Stream(w.tiles[0:K, 0:N], block=W_BLOCK),
        ):
            acc += project(xb, wb)
        self.o.write(acc)
        last = NORM(self.o, self.state[0])
        y.pages(self.rank * N, N).write(last)


NORM = norm


def folder_of(function):
    global NORM
    NORM = function
    try:
        program = Divided(l1=1 << 20, log=quiet, **tensors())
    finally:
        NORM = norm
    return Path(program.built.folder)


def test_the_rows_are_divided_by_slot_1_and_written_then_the_state_goes_back():
    folder = folder_of(norm)
    text = (folder / "compute.cpp").read_text()
    body = text.index("MATH((km_tile::norm(0)));")
    row = text.index("pack_tile(1, kOut0);", body)
    back = text.index("pack_tile<true>(2, kFieldState, 0);", row)
    assert body < row < back
    unit = (folder / "tile_norm.txt").read_text()
    assert "SFPLOAD" in unit
