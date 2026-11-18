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
