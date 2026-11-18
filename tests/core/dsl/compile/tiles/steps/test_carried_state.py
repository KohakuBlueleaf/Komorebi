"""A step on sums that carry their state (no card): `s = g(self.m[i], self.s[r])`
copies state r into DST with the row, so `pages.write(f(s, self.s[r]))` runs f on
it there, with no second take of its token, and s's write packs it once; refused:
a step on another state than the one s started with."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError

N, M = 2, 3
FIRST_MASK, MASK_PAGES = 2, 6  # init: 2 states, then M rows of N mask tiles


def tensors():
    def made(shape, address, dtype="float32"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "init": made((32, 32 * 8), 0x2000000),
        "y": made((32, 32 * 16), 0x3000000),
    }


@km.tile_function
def start(row, state):
    return row, (state[0],)


@km.tile_function
def shift(row, state):
    return row - state[0].col(), (km.reduce_max(row, exact=True),)


class Carried(km.Program):
    state: km.L1[km.fp32, 2, 32, 32]
    mask: km.L1[km.fp32, M, N, 32, 32]

    @km.handler
    def run(self, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 2)))
        self.mask.fill(km.Stream(init.pages(FIRST_MASK, MASK_PAGES)))
        for i in range(M):
            s = start(self.mask[i], self.state[1])
            y.pages(i * N, N).write(shift(s, self.state[1]))


class Other(km.Program):
    state: km.L1[km.fp32, 2, 32, 32]
    mask: km.L1[km.fp32, M, N, 32, 32]

    @km.handler
    def run(self, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 2)))
        self.mask.fill(km.Stream(init.pages(FIRST_MASK, MASK_PAGES)))
        s = start(self.mask[0], self.state[1])
        y.pages(0, N).write(shift(s, self.state[0]))


def folder_of(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return Path(program.built.folder)


def test_the_step_runs_on_the_carried_state_and_packs_it_once():
    text = (folder_of(Carried) / "compute.cpp").read_text()
    rows = text.index("for (uint32_t i = 0;")
    taken = text.index("cb_pop_front(kStepTokenState, 1);", rows)
    given = text.index("cb_push_back(kStepTokenMask, 1);", taken)
    end = text.index("tile_regs_release();", given)
    body = text[taken:end]
    # a held state: taken and given back once a turn, whatever the values show
    assert body.count("cb_pop_front(kStepTokenState, 1);") == 1
    assert body.count("cb_push_back(kStepTokenState, 1);") == 1


def test_a_step_on_another_state_is_refused():
    with pytest.raises(
        ProgramError,
        match=r"line \d+, `.*s holds a state of its start in DST, and this step takes "
        r"self\.state\[0\]; step on the state s started with",
    ):
        folder_of(Other)
