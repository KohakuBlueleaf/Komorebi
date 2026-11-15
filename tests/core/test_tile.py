"""The tile layer: instruction words, SiLU's unit program (passes, checks, cost), the
emitted header, the load-macro rule and the row form's lane checks."""

import random

import pytest

from komorebi.dsl.tile import UnitError, emit, trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam
from komorebi.dsl.tile.rows import row_program
from komorebi.dsl.tile.steps import step_program
from komorebi.dsl.tile.unit import OPS, Call, Insn, Loop, decode, encode, insns


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


def test_exp_step_per_row_operand():
    s, m = TensorParam("s", False, F32), TensorParam("m", False, F32)
    p = TensorParam("p", True, F32)
    ir = KernelIR("expstep", [s, m, p], [], {})
    with trace.Tracer(ir):
        ss, ms, ps = (trace.TensorProxy(t) for t in (s, m, p))
        for t in trace.grid(trace.num_tiles(ss)):
            x, state = trace.load(ss.tiles[t]), trace.load(ms.tiles[t])
            trace.store(ps.tiles[t], trace.exp(x * 0.125 - state.col()))
    unit = unit_program(ir)
    code = [x for x in insns(unit.tile["math"]) if x.op == "SFPLOAD"]
    # per 4-row block: the state once (tile 1), the two column groups of s
    assert [x.args["dest_reg_addr"] for x in code[:3]] == [0, 64, 2]
    fused = [x for x in insns(unit.tile["math"]) if x.op == "SFPMAD"][0]
    assert fused.args["lreg_src_c"] == code[1].args["lreg_ind"]
    assert fused.args["instr_mod1"] == 2  # s 0.125 - m in one multiply-add


def test_softmax_step_state_lanes_and_word():
    s, st = TensorParam("s", False, F32), TensorParam("st", False, F32)
    p, out = TensorParam("p", True, F32), TensorParam("out", True, F32)
    ir = KernelIR("softstep", [s, st, p, out], [], {})
    with trace.Tracer(ir):
        ss, sts, ps, outs = (trace.TensorProxy(t) for t in (s, st, p, out))
        for t in trace.grid(trace.num_tile_rows(ss)):
            x = trace.load(ss.tiles[t, :]) * 0.18
            state = trace.load_state(sts.tiles[t])
            m_new = trace.maximum(state[0], trace.reduce_max(x, exact=True))
            prob = trace.exp2(x - m_new.col())
            l_new = state[1] + trace.reduce_sum(prob, exact=True)
            trace.store(ps.tiles[t, :], prob)
            moved = trace.word_max(m_new - state[0])
            trace.store_state(outs.tiles[t], [m_new, l_new], word=moved)
    unit = step_program(ir, 2, 3)
    checks = {c.rule: c.verdict for c in unit.checks}
    assert checks["lanes"].startswith("ok") and checks["broadcast"].startswith("ok")
    scaled = next(p for p in unit.passes if p.name == "max_scale")
    assert scaled.done and "out of a maximum" in scaled.done[0]
    code = list(insns(unit.tile["math"]))
    assert [x.op for x in code].count("SFPTRANSP") == 1  # the word's lanes
    assert code[-1].op == "SFPSTORE" and code[-1].args["dest_reg_addr"] == 3 * 64 + 34
