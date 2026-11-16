"""A field zeroed on the core (no card): `self.f.zero()` packs a zero tile into each
of the field's pages on compute and pushes them, as a fill does, with no DRAM read;
step states and rows of a field then read it; refused: arguments, a field filled
above, sums in DST, an own field."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.errors import ProgramError

N, R = 2, 3


def tensors():
    def made(shape, address, dtype="float32"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {"init": made((32, 32 * 8), 0x2000000), "y": made((32, 32 * 8), 0x3000000)}


@km.tile_function
def decay(row, state):
    return row * state[0].col(), (state[0],)


class Zeroed(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, R, N, 32, 32]

    @km.handler
    def run(self, init: km.Tensor, y: km.Tensor):
        self.state.zero()
        self.o.zero()
        for r in range(R):
            acc = decay(self.o[r], self.state[0])
            self.o[r].write(acc)
        y.pages(0, 1).write(self.state)


class Twice(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(self, init: km.Tensor, y: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.state.zero()


class Argued(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(self, init: km.Tensor, y: km.Tensor):
        self.state.zero(0)


class Held(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run(self, init: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 1)
        self.state.zero()
        y.pages(0, 1).write(acc)


class Own(km.Program):
    state: km.L1[km.fp32, 1, 32, 32] = km.own(push="pack")

    @km.handler
    def run(self, init: km.Tensor, y: km.Tensor):
        self.state.zero()


def folder_of(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return Path(program.built.folder)


def test_zero_tiles_are_packed_into_the_pages_before_the_rows_read_them():
    folder = folder_of(Zeroed)
    text = (folder / "compute.cpp").read_text()
    state = text.index("// self.state zeroed on the core")
    filled = text.index("fill_tile(0, 0.0f);", state)
    packed = text.index("pack_tile<true>(0, kFieldState, km_page);", filled)
    pushed = text.index("cb_push_back(kFieldState, kFieldStatePages);", packed)
    rows = text.index("// self.o zeroed on the core", pushed)
    pushed_o = text.index("cb_push_back(kFieldO, kFieldOPages);", rows)
    read = text.index("compute.wait_front(kFieldO, kFieldOPages)", pushed_o)
    assert state < filled < packed < pushed < rows < pushed_o < read
    assert "read_pages" not in (folder / "movement.cpp").read_text()


@pytest.mark.parametrize(
    "cls, refusal",
    [
        (Twice, r"self\.state is filled above and no body took it"),
        (Argued, r"a zero takes no arguments"),
        (Held, r"acc still holds DST, and a zero packs DST 0"),
        (Own, r"self\.state is an own field: only bodies use it, no line zeroes it"),
    ],
)
def test_refusals_name_the_line(cls, refusal):
    with pytest.raises(ProgramError, match=rf"line \d+, `.*{refusal}"):
        folder_of(cls)
