"""The tile layer: instruction words, SiLU's unit program (passes, checks, cost), the
emitted header, the load-macro rule and the row form's lane checks."""

import random

import pytest

from komorebi.dsl.tile import UnitError, emit, trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam
from komorebi.dsl.tile.rows import row_program
from komorebi.dsl.tile.unit import OPS, Call, Insn, Loop, decode, encode


def stream(name, inputs, fn, **options):
    params = [TensorParam(n, False, F32) for n in "ab"[:inputs]]
    y = TensorParam("y", True, F32)
    ir = KernelIR(name, [*params, y], [], {})
    with trace.Tracer(ir):
        proxies, ys = [trace.TensorProxy(p) for p in params], trace.TensorProxy(y)
        for t in trace.grid(trace.num_tiles(proxies[0])):
            trace.store(ys.tiles[t], fn(*[trace.load(p.tiles[t]) for p in proxies]))
    return unit_program(ir, **options)


def silu(**options):
    return stream("silu", 1, trace.silu, **options)


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
    assert unit.cost("math")[1] == 1533.6
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


def test_load_macro_fits_add_and_refuses_silu():
    unit = stream("add", 2, lambda a, b: a + b, load_macro=True)
    fit = next(p for p in unit.passes if p.name == "load_macro")
    assert "SFPLOADMACRO" in fit.done[0] and unit.cost("math")[1] == 91.6
    refused = next(p for p in silu(load_macro=True).passes if p.name == "load_macro")
    assert refused.done[0].startswith("refused: the row computes with 39")


def test_row_norm_lanes_and_broadcast():
    x, g = TensorParam("x", False, F32), TensorParam("g", False, F32)
    y = TensorParam("y", True, F32)
    ir = KernelIR("norm", [x, g, y], [], {})
    with trace.Tracer(ir):
        xs, gs, ys = (trace.TensorProxy(t) for t in (x, g, y))
        weight = trace.resident(gs.tiles[0, :])
        for t in trace.grid(trace.num_tile_rows(xs)):
            row = trace.load(xs.tiles[t, :])
            r = trace.rsqrt(trace.reduce_sum(row * row, exact=True) * (1 / 64) + 1e-6)
            trace.store(ys.tiles[t, :], row * r.col() * weight.row())
    checks = {c.rule: c.verdict for c in row_program(ir, 2, 2, 4).checks}
    assert checks["lanes"].startswith("ok") and checks["broadcast"].startswith("ok")
    # each lane rotation of a sum waits a cycle: the unit does not stall it
    assert checks["mad-reader"] == "repaired: 32 SFPNOP"
