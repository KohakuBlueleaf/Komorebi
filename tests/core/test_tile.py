"""The tile layer: instruction words, SiLU's unit program (passes, checks, cost) and
the emitted header."""

import random

import pytest

from komorebi.dsl.tile import UnitError, emit, trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam
from komorebi.dsl.tile.unit import OPS, Call, Insn, Loop, decode, encode


def silu():
    x, y = TensorParam("x", False, F32), TensorParam("y", True, F32)
    ir = KernelIR("silu", [x, y], [], {})
    with trace.Tracer(ir):
        xs, ys = trace.TensorProxy(x), trace.TensorProxy(y)
        for t in trace.grid(trace.num_tiles(xs)):
            trace.store(ys.tiles[t], trace.silu(trace.load(xs.tiles[t])))
    return unit_program(ir)


@pytest.mark.parametrize("name", sorted(OPS))
def test_words_round_trip(name):
    rng = random.Random(name)
    for _ in range(20):
        args = {arg: rng.randrange(1 << width) for arg, width in OPS[name].args}
        assert decode(encode(name, args)) == (name, args)


def test_word_refuses_a_wide_field():
    with pytest.raises(UnitError, match="does not fit"):
        encode("SFPLOADI", {"lreg_ind": 0, "instr_mod0": 0, "imm16": 1 << 16})


def test_silu_unit_program():
    unit = silu()
    assert {p.kind for p in unit.passes} == {"algebra", "schedule", "layout"}
    assert all(not c.verdict.startswith("fail") for c in unit.checks)
    faces = next(x for x in unit.tile["math"] if isinstance(x, Loop))
    record, rest = faces.body[0], faces.body[-3]
    assert (record.op, record.args["len"], record.args["load_mode"]) == (
        "REPLAY",
        32,
        1,
    )
    assert isinstance(rest, Loop) and rest.count == 7
    assert rest.body[0].op == "REPLAY" and rest.body[0].args["load_mode"] == 0
    assert unit.cost("math")[1] == 1738
    assert unit.constants and unit.not_checked


def test_silu_header():
    unit = silu()
    text = emit(unit)
    assert "TTI_REPLAY(0, 32, 1, 1);" in text and "TTI_SFPNOP;" in text
    assert "inline void silu_init()" in text
    assert "inline void silu(const uint32_t dst)" in text
    assert any(c.rule == "no-float" for c in unit.checks)


def test_header_refuses_a_float():
    unit = silu()
    unit.init["math"].append(Call("ckernel::scale(1.5f)", "a float"))
    with pytest.raises(UnitError, match="float"):
        emit(unit)


def test_header_refuses_an_immediate_too_wide():
    unit = silu()
    args = {"lreg_ind": 0, "instr_mod0": 0, "imm16": 1 << 16}
    unit.init["math"].append(Insn("SFPLOADI", args, "too wide"))
    with pytest.raises(UnitError, match="does not fit"):
        emit(unit)
