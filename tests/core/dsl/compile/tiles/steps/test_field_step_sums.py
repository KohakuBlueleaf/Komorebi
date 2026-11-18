"""Sums started from a field (no card): `acc = f(self.o, self.s[r])` copies the
field's tiles into DST 0 .. and state r after them, runs f's step form on both;
products add into acc, and its write packs the row and then the state with the
token; refused: a field not written above, a field not 1 to 3 fp32 tiles, DST held."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError

K, N, C = 4, 2, 2


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((128, C * K * 32), 0x100000),
        "w": made((C * K * 32, N * 32), 0x1000000),
        "init": made((32, 32), 0x2000000, "float32"),
        "y": made((128, N * 32), 0x3000000, "float32"),
    }


@km.tile_function
def decay(row, state):
    return row * state[0].col(), (state[0],)


class Unwritten(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        last = decay(self.o, self.state[0])
        y.pages(self.rank * N, N).write(last)


class Halves(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.bf16, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        zero = km.tile.zeros(km.fp32, N)
        self.o.write(zero)
        last = decay(self.o, self.state[0])
        y.pages(self.rank * N, N).write(last)


class Held(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, N, 32, 32]

    @km.handler
    def run(self, x: km.Tensor, w: km.Tensor, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        zero = km.tile.zeros(km.fp32, N)
        self.o.write(zero)
        other = km.tile.zeros(km.fp32, N)
        last = decay(self.o, self.state[0])
        y.pages(self.rank * N, N).write(last)
        y.pages(self.rank * N, N).write(other)


def folder_of(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return Path(program.built.folder)


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (Unwritten, r"self\.o is not written above in this handler"),
        (Halves, r"self\.o is 2 bfloat16 tile\(s\); a step's row is 1 to 3 fp32"),
        (Held, r"other still holds DST; a step's row starts at DST 0"),
    ],
)
def test_refusals_name_the_line(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        folder_of(cls)
