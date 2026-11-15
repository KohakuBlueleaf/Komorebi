"""A step function on DST sums (no card): state r through DST between the token and
its page, the states written after; a value reading a slot stored a level before its
reader kept in a free slot; refused: a row past 3 tiles, a field not filled, an index
past its pages, a field not fp32, a function not returning the row and its state."""

import math
from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor

K, N, C = 4, 2, 2
LOG2E = 1.0 / math.log(2.0)
WIDTH = N  # the row's tiles a class below sums
W_BLOCK = 2 * N
WIDE_BLOCK = 8  # W's tiles a block of a row of 4


def tensors():
    def made(shape, address, dtype):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((128, K * 32), 0x100000, "bfloat16"),
        "w": made((K * 32, 8 * 32), 0x1000000, "bfloat16"),
        "init": made((32, 64), 0x2000000, "float32"),
        "p": made((32, 64 * 32), 0x3000000, "float32"),
        "st": made((32, 64 * 32), 0x4000000, "float32"),
    }


@km.tile_function
def project(x, w):
    return x @ w


@km.tile_function
def softmax_step(row, state):
    x = row * LOG2E
    m_old, l_old = state[0], state[1]
    m_new = km.lagging_max(m_old, km.reduce_max(x, exact=True), 0.0)
    alpha = km.exp2(m_old - m_new, error=2e-3, over=(-math.inf, 0.0))
    p = km.exp2(x - m_new.col(), error=2e-3, over=(-math.inf, 0.0))
    return p, (m_new, alpha * l_old + km.reduce_sum(p, exact=True))


@km.tile_function
def only_row(row, state):
    return row * state[0].col()


class Rows(km.Program):
    state: km.L1[km.fp32, 2, 32, 32]

    @km.handler
    def run(
        self, x: km.Tensor, w: km.Tensor, init: km.Tensor, p: km.Tensor, st: km.Tensor
    ):
        self.state.fill(km.Stream(init.pages(0, 2)))
        for c in range(C):
            acc = km.tile.zeros(km.fp32, WIDTH)
            for xb, wb in zip(
                km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
                km.Stream(w.tiles[0:K, c * N : c * N + WIDTH], block=W_BLOCK),
            ):
                acc += project(xb, wb)
            p.pages(self.rank * 8 + c * N, WIDTH).write(STEP(acc, self.state[INDEX]))
        st.pages(self.rank * 2, 2).write(self.state)


class Indexed(km.Program):
    state: km.L1[km.fp32, 2, 32, 32]

    @km.handler
    def run(
        self, x: km.Tensor, w: km.Tensor, init: km.Tensor, p: km.Tensor, st: km.Tensor
    ):
        self.state.fill(km.Stream(init.pages(0, 2)))
        for c in range(C):
            acc = km.tile.zeros(km.fp32, N)
            for xb, wb in zip(
                km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
                km.Stream(w.tiles[0:K, c * N : c * N + N], block=W_BLOCK),
            ):
                acc += project(xb, wb)
            p.pages(self.rank * 8 + c * N, N).write(softmax_step(acc, self.state[c]))
        st.pages(self.rank * 2, 2).write(self.state)


class Wide(km.Program):
    state: km.L1[km.fp32, 2, 32, 32]

    @km.handler
    def run(
        self, x: km.Tensor, w: km.Tensor, init: km.Tensor, p: km.Tensor, st: km.Tensor
    ):
        self.state.fill(km.Stream(init.pages(0, 2)))
        acc = km.tile.zeros(km.fp32, 4)
        for xb, wb in zip(
            km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
            km.Stream(w.tiles[0:K, 0:4], block=WIDE_BLOCK),
        ):
            acc += project(xb, wb)
        p.pages(self.rank * 8, 4).write(softmax_step(acc, self.state[0]))


class Unfilled(km.Program):
    state: km.L1[km.fp32, 2, 32, 32]

    @km.handler
    def run(
        self, x: km.Tensor, w: km.Tensor, init: km.Tensor, p: km.Tensor, st: km.Tensor
    ):
        acc = km.tile.zeros(km.fp32, N)
        for xb, wb in zip(
            km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
            km.Stream(w.tiles[0:K, 0:N], block=W_BLOCK),
        ):
            acc += project(xb, wb)
        p.pages(self.rank * 8, N).write(softmax_step(acc, self.state[0]))


class Halves(km.Program):
    state: km.L1[km.bf16, 2, 32, 32]

    @km.handler
    def run(
        self, x: km.Tensor, w: km.Tensor, init: km.Tensor, p: km.Tensor, st: km.Tensor
    ):
        acc = km.tile.zeros(km.fp32, N)
        for xb, wb in zip(
            km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
            km.Stream(w.tiles[0:K, 0:N], block=W_BLOCK),
        ):
            acc += project(xb, wb)
        p.pages(self.rank * 8, N).write(softmax_step(acc, self.state[0]))


STEP, INDEX = softmax_step, 0


def folder_of(cls, step=softmax_step, index=0, width=N):
    global STEP, INDEX, WIDTH
    STEP, INDEX, WIDTH = step, index, width
    try:
        program = cls(l1=1 << 20, log=quiet, **tensors())
    finally:
        STEP, INDEX, WIDTH = softmax_step, 0, N
    return Path(program.built.folder)


def test_the_state_goes_through_dst_between_the_token_and_its_page():
    folder = folder_of(Rows)
    text = (folder / "compute.cpp").read_text()
    primed = text.index("if (compute.reserve_back(kStepTokenState, 1)) {")
    taken = text.index("compute.wait_front(kStepTokenState, 1);\n")
    copied = text.index("copy_tile(kFieldState, 0, 2);")
    called = text.index("MATH((km_tile::softmax_step(0)));")
    row = text.index("pack_tile(1, kOut0);")
    state = text.index("pack_tile<true>(2, kFieldState, 0);")
    pushed = text.index("cb_push_back(kStepTokenState, 1);\n", state)
    written = text.index("// self.state's states, after the last step's")
    freed = text.rindex("cb_pop_front(kFieldState, kFieldStatePages);")
    assert primed < taken < copied < called < row < state < pushed < written < freed
    assert "km_state" in text and "copy_tile(kFieldState, km_state" in text


def test_alpha_read_by_the_sums_level_is_kept_in_a_free_slot():
    unit = next(folder_of(Rows).glob("tile_softmax_step.txt")).read_text()
    store_m = unit.index("dest_reg_addr=128  ; slot 0 (0 0)", unit.index("SFPSTORE"))
    kept = unit.index(
        "SFPSTORE lreg_ind=1 instr_mod0=0 sfpu_addr_mode=7 dest_reg_addr=160"
    )
    assert store_m < kept  # alpha l_old from the old m, stored before the sums' level


def test_a_word_index_is_checked_at_run_time():
    text = (folder_of(Indexed) / "compute.cpp").read_text()
    assert "if (!((c) < 2)) {  // self.state's states" in text
    assert "copy_tile(kFieldState, c, 2);" in text
    assert "pack_tile<true>(2, kFieldState, c);" in text


@pytest.mark.parametrize(
    "cls, options, refusal",
    [
        (Wide, {}, r"acc is 4 tile\(s\) of DST from tile 0; a step takes a row"),
        (Unfilled, {}, r"self\.state is not filled above in this handler"),
        (Rows, {"index": 2}, r"self\.state holds 2 states, 0 \.\. 1"),
        (Halves, {}, r"self\.state holds a step's states: fp32 tiles"),
        (Rows, {"step": only_row}, r"returns the row and its state"),
    ],
)
def test_refusals_name_the_line(cls, options, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        folder_of(cls, **options)
