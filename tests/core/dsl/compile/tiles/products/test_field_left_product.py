"""A product whose left side is a field (no card): `self.h.write(acc)` (or f(acc), a
step's row) packs DST sums into the field, freed by its next write; `out += f(self.h,
wb)` takes stream row k by field tile k; refused: a field neither written nor filled,
a filled one written, a width not the field's, a stream of other rows."""

import math
from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError

K, KH, N, R = 8, 4, 4, 2
LOG2E = 1.0 / math.log(2.0)
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


@km.tile_function
def act(t):
    return km.relu(t)


class Hidden(km.Program):
    h: km.L1[km.bf16, KH, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        for r in range(R):
            acc = km.tile.zeros(km.fp32, KH)
            for xb, wb in zip(
                km.Stream(
                    x.tiles[r * 4 + self.rank : r * 4 + self.rank + 1, 0:K], block=2
                ),
                km.Stream(w1.tiles[0:K, 0:KH], block=H_BLOCK),
            ):
                acc += project(xb, wb)
            self.h.write(act(acc))
            out = km.tile.zeros(km.fp32, N)
            for wb in km.Stream(w2.tiles[0:KH, 0:N], block=O_BLOCK):
                out += project(self.h, wb)
            y.pages((r * 4 + self.rank) * N, N).write(out)


@km.tile_function
def softmax_step(row, state):
    x = row * LOG2E
    m_old, l_old = state[0], state[1]
    m_new = km.lagging_max(m_old, km.reduce_max(x, exact=True), 0.0)
    alpha = km.exp2(m_old - m_new, error=2e-3, over=(-math.inf, 0.0))
    p = km.exp2(x - m_new.col(), error=2e-3, over=(-math.inf, 0.0))
    return p, (m_new, alpha * l_old + km.reduce_sum(p, exact=True))


class Stepped(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    p: km.L1[km.bf16, 2, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w1: km.Tensor, w2: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(y.pages(0, 1)))
        acc = km.tile.zeros(km.fp32, 2)
        for xb, wb in zip(
            km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
            km.Stream(w1.tiles[0:K, 0:2], block=4),
        ):
            acc += project(xb, wb)
        self.p.write(softmax_step(acc, self.state[0]))
        out = km.tile.zeros(km.fp32, N)
        for vb in km.Stream(w2.tiles[0:2, 0:N], block=O_BLOCK):
            out += project(self.p, vb)
        y.pages(self.rank * N, N).write(out)


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


def test_the_field_is_written_then_taken_on_the_left_row_by_row():
    text = (folder_of(Hidden) / "compute.cpp").read_text()
    flag = text.index("bool km_packed_h = false;")
    rows = text.index("for (uint32_t r = 0;")
    freed = text.index("if (km_packed_h) {  // the last write's pages")
    reserved = text.index("compute.reserve_back(kFieldH, 4)", freed)
    pushed = text.index("cb_push_back(kFieldH, 4);", reserved)
    set_ = text.index("km_packed_h = km_ok;", pushed)
    waited = text.index("km_ok = km_ok && compute.wait_front(kFieldH, kFieldHPages);")
    init = text.index("matmul_block_init(kFieldH, kRead2.mover_a, 0, 4, 1, 4);")
    step = text.index("km_tile2 += 4) {")
    call = text.index(
        "matmul_block(kFieldH, km_half2, (km_block2 * kRead2Block + km_tile2) / 4, "
        "km_tile2, 4, 0, 4, 1, 4);"
    )
    last = text.rindex("cb_pop_front(kFieldH, kFieldHPages);")
    assert flag < rows < freed < reserved < pushed < set_ < waited < init < step
    assert step < call < last
    assert "KM_UNPACK_FORMATS(kRead2.mover_a, kFieldH);" in text
    assert text.index("MATH((km_tile::act(3)));") < freed  # relu before the pack


def test_the_field_is_packed_in_its_own_format():
    header = (folder_of(Hidden) / "hidden.h").read_text()
    assert "#define KM_PACK_FORMAT(cb) pack_reconfig_data_format(cb)" in header


def test_a_steps_row_goes_into_the_field_and_its_state_after():
    text = (folder_of(Stepped) / "compute.cpp").read_text()
    row = text.index("pack_tile(1, kFieldP);")
    state = text.index("pack_tile<true>(2, kFieldState, 0);")
    taken = text.index("matmul_block(kFieldP, km_half2, ")
    assert row < state < taken


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
