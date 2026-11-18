"""Stamps of step calls (no card): in a handler with `stamps=`, each step's setup and
body run between two tensix_sync() and their spans go to the stamps page, two a step,
and each compute CPU's span and waits go to the page's last 9 words; a handler
without stamps emits the step as before."""

from pathlib import Path

import pytest
from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.pages.stamp_pages import timed_text
from komorebi.errors import ProgramError

N, R = 2, 3


def tensors():
    def made(shape, address, dtype="float32", layout="tile"):
        if layout == "tile":
            pages, page = shape[0] * shape[1] // 1024, formats.tile_bytes(dtype)
        else:
            pages, page = shape[0], shape[1] * 4
        return Tensor(Pages(page, pages, address), shape, dtype, layout)

    return {
        "init": made((32, 32 * 8), 0x2000000),
        "y": made((32, 32 * 8), 0x3000000),
        "stamps": made((120, 64), 0x4000000, "uint32", "row"),
    }


@km.tile_function
def decay(row, state):
    return row * state[0].col(), (state[0],)


class Stamped(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, R, N, 32, 32]

    @km.handler(stamps="stamps")
    def run(self, init: km.Tensor, y: km.Tensor, stamps: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.o.zero()
        for r in range(R):
            acc = decay(self.o[r], self.state[0])
            self.o[r].write(acc)


def compute_of(cls, **changed):
    program = cls(l1=1 << 20, log=quiet, **{**tensors(), **changed})
    return (Path(program.built.folder) / "compute.cpp").read_text()


def test_a_page_without_room_for_the_cpus_words_is_refused():
    small = Tensor(Pages(80, 120, 0x5000000), (120, 20), "uint32", "row")
    with pytest.raises(
        ProgramError, match=r"a stamps page of a handler of steps holds 28"
    ):
        compute_of(Stamped, stamps=small)


def test_a_wait_with_its_first_argument_in_the_start_is_wrapped_whole():
    text = "km_ok = km_ok && km::broadcast_receive(compute, kB0, f(1, 2));"
    assert timed_text(text) == (
        "km_ok = km_ok && KM_WAITED(km_waited_in, "
        "km::broadcast_receive(compute, kB0, f(1, 2)));"
    )
