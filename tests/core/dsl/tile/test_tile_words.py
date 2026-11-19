"""The tile compiler's words against the chip of the load (chips.Chip.tile_refusal):
Blackhole takes every one; Wormhole refuses by name the instructions, modifiers and
next-cycle reads it runs otherwise; a class's words come from its make (no card)."""

from types import SimpleNamespace

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device import chips
from komorebi.device.tensor import Tensor
from komorebi.dsl import program as dsl_program
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.tile import emit, trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam
from komorebi.errors import ProgramError
from komorebi.session import kernels

WORMHOLE = chips.chip("wormhole_b0").tile_refusal


def word(op, kind="simple", reads=(), writes=(), **args):
    return (op, args, "", kind, list(reads), list(writes))


def silu_header():
    """SiLU's unit program and its emitted header."""
    x, y = TensorParam("x", False, F32), TensorParam("y", True, F32)
    ir = KernelIR("silu", [x, y], [], {})
    with trace.Tracer(ir):
        xs, ys = trace.TensorProxy(x), trace.TensorProxy(y)
        for t in trace.grid(trace.num_tiles(xs)):
            trace.store(ys.tiles[t], trace.silu(trace.load(xs.tiles[t])))
    unit = unit_program(ir)
    return unit, emit(unit)


def test_the_stall_on_the_matrix_unit_is_written_by_its_names():
    """p_stall::MATH is 0x10 on Blackhole (its word unchanged) and 0x80 on Wormhole."""
    _, text = silu_header()
    stall = "TTI_STALLWAIT(ckernel::p_stall::STALL_SFPU, ckernel::p_stall::MATH);"
    assert stall in text and "TTI_STALLWAIT(256" not in text


def test_wormhole_refuses_the_instructions_and_modifiers_it_does_not_have():
    assert "Wormhole does not have" in WORMHOLE(None, word("SFPGT", instr_mod1=1))
    assert "NEGATE_VA" in WORMHOLE(None, word("SFPMAD", "mad", instr_mod1=1))
    assert WORMHOLE(None, word("SFPMAD", "mad", instr_mod1=0)) is None
    assert "ARG_IMM_USE_VC" in WORMHOLE(None, word("SFPSHFT", instr_mod1=5))
    assert "integer cast" in WORMHOLE(None, word("SFPCAST", instr_mod1=2))
    assert "address mode 7" in WORMHOLE(None, word("SFPLOAD", sfpu_addr_mode=7))
    assert WORMHOLE(None, word("SFPLOAD", sfpu_addr_mode=3)) is None
    assert WORMHOLE(None, word("STALLWAIT", wait_res=0x10)) is None  # by its name
    assert "p_stall::MATH by its name" in WORMHOLE(
        None, word("STALLWAIT", wait_res=0x20)
    )
    assert WORMHOLE(None, word("STALLWAIT", wait_res=0x4)) is None  # C2: the same


def test_wormhole_refuses_a_result_read_the_cycle_after_it_is_not_held():
    mad = word("SFPMAD", "mad", writes=[3], instr_mod1=0)
    reader = word("SFPSTORE", "store", reads=[3], sfpu_addr_mode=3)
    assert "LReg 3, SFPMAD's result, the next cycle" in WORMHOLE(mad, reader)
    assert WORMHOLE(word("SFPNOP"), reader) is None
    lanes = word("SFPSHFT2", writes=[2], instr_mod1=3)
    assert "lane mode" in WORMHOLE(lanes, word("SFPMOV", reads=[2], instr_mod1=0))
    swap = word("SFPSWAP", writes=[0, 1], instr_mod1=1)
    assert "collide" in WORMHOLE(swap, word("SFPLOADI", instr_mod0=8))
    template = word("SFPSWAP", instr_mod1=1, lreg_dest=13)
    assert "runs it twice" in WORMHOLE(template, word("SFPIADD", instr_mod1=1))
    assert WORMHOLE(template, word("SFPNOP")) is None
    assert "replays 1 to 32" in WORMHOLE(None, word("REPLAY", len=0))
    assert WORMHOLE(None, word("REPLAY", len=32)) is None


class Gated(km.Program):
    @km.tile_function
    def gate(x, y):
        return km.silu(x) * y

    @km.handler
    def run(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 8), write=True, block=8)
        for x, y in zip(
            km.Stream(a.pages(0, 8), block=8), km.Stream(b.pages(0, 8), block=8)
        ):
            results.push(self.gate(x, y))


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def test_a_class_s_words_load_on_blackhole_and_are_refused_by_name_on_wormhole():
    page = formats.tile_bytes("bfloat16")
    given = {n: Tensor(Pages(page), (2048, 32), "bfloat16", "tile") for n in "ab"}
    given["out"] = Tensor(Pages(page), (2048, 32), "bfloat16", "tile")
    words = dsl_program.tile_words(compile_class(Gated, {}, given))
    ((function, runs),) = words
    assert function.startswith("gate (in run, line ") and sum(map(len, runs)) > 20
    program = SimpleNamespace(name="gated", tile_words=words)
    kernels.tile_words_checked([program], "blackhole")
    with pytest.raises(ProgramError, match=r"gated: tile function gate \(in run, line"):
        kernels.tile_words_checked([program], "wormhole_b0")
