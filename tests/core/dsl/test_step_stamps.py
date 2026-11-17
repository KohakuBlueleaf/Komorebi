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


def compute_of(cls, **changed):
    program = cls(l1=1 << 20, log=quiet, **{**tensors(), **changed})
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


def test_each_compute_cpu_s_span_and_waits_end_the_page():
    text = compute_of(Stamped)
    opened = text.index("const uint32_t km_opened = km::wall();")
    acquired = text.index("KM_WAITED_DO(km_waited_out, tile_regs_acquire());", opened)
    waited = text.index("KM_WAITED(km_waited_in, compute.wait_front(kStepTokenO, 1))")
    written = text.index("PACK(km::l1(km_spans_page)[61 + 2] = km_waited_out);")
    assert opened < acquired < waited < written
    for cpu, first in (("UNPACK", 55), ("MATH", 58), ("PACK", 61)):
        assert (
            f"{cpu}(km::l1(km_spans_page)[{first} + 0] = km::wall() - km_opened);"
            in (text)
        )
    assert "KM_WAITED" not in compute_of(Plain)


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
