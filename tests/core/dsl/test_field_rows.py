"""Rows of a field (no card): `acc = f(self.o[i], self.s[r])` copies row i in place
after its token; `self.o[j].write(acc)` packs it into row j, pushes the token, then
the state; a page write of acc gives the token back. Refused: a row written from
other sums, a field not of rows, an index past its rows."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError

K, N, R = 4, 2, 2
O_PAGES = 4  # R x N
W_BLOCK = 2 * N


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((R * 128, K * 32), 0x100000),
        "w": made((K * 32, N * 32), 0x1000000),
        "init": made((32, 32 * (1 + R * N)), 0x2000000, "float32"),
        "y": made((32, 32 * R * N * 12), 0x3000000, "float32"),
    }


@km.tile_function
def project(x, w):
    return x @ w


@km.tile_function
def decay(row, state):
    return row * state[0].col(), (state[0],)


class Rows(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, R, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.o.fill(km.Stream(init.pages(1, O_PAGES)))
        for r in range(R):
            acc = decay(self.o[r], self.state[0])
            for xb, wb in zip(
                km.Stream(
                    x.tiles[r * 4 + self.rank : r * 4 + self.rank + 1, 0:K], block=2
                ),
                km.Stream(w.tiles[0:K, 0:N], block=W_BLOCK),
            ):
                acc += project(xb, wb)
            self.o[r].write(acc)
        y.pages(self.rank * O_PAGES, O_PAGES).write(self.o)


class Whole(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, R * N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.o.fill(km.Stream(init.pages(1, O_PAGES)))
        acc = decay(self.o[0], self.state[0])
        self.o[0].write(acc)


class Read(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, R, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.o.fill(km.Stream(init.pages(1, O_PAGES)))
        for r in range(R):
            last = decay(self.o[r], self.state[0])
            y.pages(r * N, N).write(last)


class Other(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, R, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.o.fill(km.Stream(init.pages(1, O_PAGES)))
        acc = km.tile.zeros(km.fp32, N)
        self.o[0].write(acc)


class Past(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, R, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.o.fill(km.Stream(init.pages(1, O_PAGES)))
        acc = decay(self.o[2], self.state[0])
        self.o[0].write(acc)


def folder_of(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return Path(program.built.folder)


def test_a_row_is_read_and_written_in_place_between_its_tokens():
    text = (folder_of(Rows) / "compute.cpp").read_text()
    primed = text.index("if (compute.reserve_back(kStepTokenO, 1)) {")
    rows = text.index("for (uint32_t r = 0;", primed)
    taken = text.index("cb_pop_front(kStepTokenO, 1);", rows)
    copied = text.index("copy_tile(kFieldO, (r) * 2 + 1, 1);", taken)
    body = text.index("MATH((km_tile::decay(0)));", copied)
    product = text.index("matmul_block(km_half0, km_half1, ", body)
    packed = text.index("pack_tile<true>(1, kFieldO, (r) * 2 + 1);", product)
    pushed = text.index("cb_push_back(kStepTokenO, 1);", packed)
    state = text.index("pack_tile<true>(2, kFieldState, 0);", pushed)
    written = text.index("copy_tile(kFieldO, km_state", state)
    assert primed < rows < taken < copied < body < product < packed < pushed < state
    assert state < written


def test_a_row_written_elsewhere_gives_its_token_back():
    text = (folder_of(Read) / "compute.cpp").read_text()
    rows = text.index("for (uint32_t r = 0;")
    taken = text.index("cb_pop_front(kStepTokenO, 1);", rows)
    packed = text.index("pack_tile(1, kOut0", taken)
    given = text.index("cb_push_back(kStepTokenO, 1);", packed)
    released = text.index("tile_regs_release();", given)
    drained = text.index("cb_pop_front(kStepTokenO, 1);", released)
    assert rows < taken < packed < given < released < drained


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (Other, r"a row of self\.o takes sums started from one of its rows"),
        (Whole, r"self\.o's rows are not tiles; a step's row is 1 to 3 fp32"),
        (Past, r"self\.o holds 2 rows, 0 \.\. 1"),
    ],
)
def test_refusals_name_the_line(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        folder_of(cls)
