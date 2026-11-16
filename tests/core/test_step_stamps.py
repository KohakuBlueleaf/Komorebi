"""Stamps of step calls (no card): in a handler with `stamps=`, each step's setup and
body run between two tensix_sync() and their spans go to the stamps page, two a step;
a handler without stamps emits the step as before."""

from pathlib import Path

from chain_fixture import Pages, quiet

import komorebi as km
from komorebi import formats
from komorebi.tensor import Tensor

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


class Plain(km.Program):
    state: km.L1[km.fp32, 1, 32, 32]
    o: km.L1[km.fp32, R, N, 32, 32]

    @km.handler
    def run(self, init: km.Tensor, y: km.Tensor, stamps: km.Tensor):
        self.state.fill(km.Stream(init.pages(0, 1)))
        self.o.zero()
        for r in range(R):
            acc = decay(self.o[r], self.state[0])
            self.o[r].write(acc)


def compute_of(cls):
    program = cls(l1=1 << 20, log=quiet, **tensors())
    return (Path(program.built.folder) / "compute.cpp").read_text()


def test_a_step_s_setup_and_body_are_each_timed():
    text = compute_of(Stamped)
    step = text.index("// decay's step")
    synced = text.index("MATH(tensix_sync());", step)
    setup = text.index("MATH((km_tile::decay_init()));", synced)
    spans = text.index("MATH(km_spans[1] += 1);", setup)
    body = text.index("MATH((km_tile::decay(0)));", spans)
    second = text.index("MATH(km_spans[1] += 1);", body)
    assert step < synced < setup < spans < body < second


def test_without_stamps_the_step_is_as_before():
    text = compute_of(Plain)
    assert "tensix_sync" not in text
    assert "    MATH((km_tile::decay_init()));\n" in text
