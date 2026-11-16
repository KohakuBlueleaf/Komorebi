"""Step lines at two loop depths (no card), refused as the class lines are: a step's
state field or a field of rows filled in a loop (freed once, at the handler's end),
and a row written in a loop from sums started outside it."""

import math

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor

K, N, C = 4, 2, 2
W_BLOCK = 2 * N
O_PAGES = 4
LOG2E = 1.0 / math.log(2.0)


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((128, K * 32), 0x100000),
        "w": made((K * 32, N * 32 * C), 0x1000000),
        "init": made((32, 32 * 8), 0x2000000, "float32"),
        "y": made((32, 32 * 64), 0x3000000, "float32"),
    }


@km.tile_function
def project(x, w):
    return x @ w


@km.tile_function
def softmax_step(row, state):
    x = row * LOG2E
    m_new = km.lagging_max(state[0], km.reduce_max(x, exact=True), 0.0)
    p = km.exp2(x - m_new.col(), error=2e-3, over=(-math.inf, 0.0))
    return p, (m_new, km.reduce_sum(p, exact=True))


@km.tile_function
def decay(row, state):
    return row * state[0].col(), (state[0],)


class StateInLoop(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        for c in range(C):
            self.state.fill(km.Stream(init.pages(0, 1)))
            acc = km.tile.zeros(km.fp32, N)
            for xb, wb in zip(
                km.Stream(x.tiles[self.rank : self.rank + 1, 0:K], block=2),
                km.Stream(w.tiles[0:K, c * N : c * N + N], block=W_BLOCK),
            ):
                acc += project(xb, wb)
            y.pages(c * N, N).write(softmax_step(acc, self.state[0]))


class RowsInLoop(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, 2, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        for c in range(C):
            self.o.fill(km.Stream(init.pages(1, O_PAGES)))
            acc = decay(self.o[c], self.state[0])
            self.o[c].write(acc)


class RowOutside(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, 2, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.o.fill(km.Stream(init.pages(1, O_PAGES)))
        acc = decay(self.o[0], self.state[0])
        for c in range(C):
            self.o[c].write(acc)


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (StateInLoop, r"self\.state is filled in a loop \(line \d+\) and its states"),
        (RowsInLoop, r"self\.o is filled in a loop \(line \d+\) and its rows"),
        (RowOutside, r"acc was made at line \d+, outside the loop this write stands"),
    ],
)
def test_refusals_name_the_line(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        cls(l1=1 << 20, log=quiet, **tensors())
