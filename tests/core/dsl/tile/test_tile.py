"""The tile layer: instruction words, SiLU's unit program (passes, checks, cost), the
emitted header and its refusals, the load-macro and max-macro rules, the row form's
lane checks and operand layout, and the matrix unit's words and prices."""

import math
import random
import re

import pytest

from komorebi.dsl.tile import UnitError, emit, row_operand, trace, unit_program
from komorebi.dsl.tile.forms.rows import row_program
from komorebi.dsl.tile.forms.steps import step_program
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam
from komorebi.dsl.tile.lower import allocate
from komorebi.dsl.tile.macro import macro_setup, max_macro
from komorebi.dsl.tile.matrix import block_cost, product_program, tile_products
from komorebi.dsl.tile.unit import (
    OPS,
    Call,
    Insn,
    Loop,
    Sync,
    UnitProgram,
    decode,
    encode,
    insns,
    matrix_cost,
    rotate_blocks,
)


def stream(name, inputs, fn, **options):
    params = [TensorParam(n, False, F32) for n in "abc"[:inputs]]
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
    assert unit.cost("math")[1] == 1475.6
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
    assert refused.done[0].startswith("refused: the row computes with 37")


def norm_ir():
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
    return ir


def test_row_norm_lanes_and_broadcast():
    checks = {c.rule: c.verdict for c in row_program(norm_ir(), 2, 2, 4).checks}
    assert checks["lanes"].startswith("ok") and checks["broadcast"].startswith("ok")
    # a lane rotation reading a sum is not held: the interleave puts other work between
    assert checks["mad-reader"] == "ok"


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


def test_a_per_row_operand_at_a_mapped_dst_tile():
    s, m = TensorParam("s", False, F32), TensorParam("m", False, F32)
    p = TensorParam("p", True, F32)
    ir = KernelIR("expapart", [s, m, p], [], {})
    with trace.Tracer(ir):
        ss, ms, ps = (trace.TensorProxy(t) for t in (s, m, p))
        for t in trace.grid(trace.num_tiles(ss)):
            x, state = trace.load(ss.tiles[t]), trace.load(ms.tiles[t])
            trace.store(ps.tiles[t], trace.exp(x * 0.125 - state.col()))
    unit = unit_program(ir, tiles={"s": 0, "m": 3})  # m held 3 tiles after s
    code = [x for x in insns(unit.tile["math"]) if x.op == "SFPLOAD"]
    assert [x.args["dest_reg_addr"] for x in code[:3]] == [0, 192, 2]
    assert unit.col_operands == [1]


def _exp_ir(**exp):
    s, m = TensorParam("s", False, F32), TensorParam("m", False, F32)
    p = TensorParam("p", True, F32)
    ir = KernelIR("expflash", [s, m, p], [], {})
    with trace.Tracer(ir):
        ss, ms, ps = (trace.TensorProxy(t) for t in (s, m, p))
        for t in trace.grid(trace.num_tiles(ss)):
            x, state = trace.load(ss.tiles[t]), trace.load(ms.tiles[t])
            trace.store(ps.tiles[t], trace.exp(x * 0.125 - state.col(), **exp))
    return ir


def test_exp_at_a_stated_error_takes_the_flash_template():
    unit = unit_program(_exp_ir(error=2e-3, over=(-math.inf, 0.0)))
    done = {p.name: p.done for p in unit.passes}
    assert done["templates"][0].split(": ", 1)[1].startswith("flash, relative")
    # the bias folded into m's term: 639 - m log2 e once a block, one MAD a vector
    assert done["common_values"] and not done["interleave"][0].startswith("0 ")
    mads = [x for x in insns(unit.tile["math"]) if x.op == "SFPMAD"]
    assert len(mads) == 7  # per block: the state's term, then 3 for each vector
    with pytest.raises(PlanError, match="no template meets"):
        unit_program(_exp_ir(error=1e-9))


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
            prob = trace.exp2(x - m_new.col(), error=2e-3, over=(-math.inf, 0.0))
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


def test_matrix_unit_rotation_prices_and_rule():
    # probe 11: a block added into again after K words costs max(1, 6 / K) a word
    n = 24 * 64
    for k, want in ((1, 6.0), (2, 3.0), (3, 2.0), (4, 1.5), (5, 1.2), (6, 1.0)):
        words = [("MVMUL", i % k) for i in range(n)]
        assert matrix_cost(words, synced=False) / n == pytest.approx(want, abs=0.01)
    assert matrix_cost([("MVMUL", 0)]) == 8.0
    assert matrix_cost([("MOVA2D", 0)] * 4, synced=False) == 4.0  # moves wait for none
    inner_fastest = [("MVMUL", b) for b in range(8) for _ in range(2)]
    rotated = rotate_blocks(inner_fastest)
    assert [b for _, b in rotated] == list(range(8)) * 2
    assert matrix_cost(inner_fastest) == 63.0 and matrix_cost(rotated) == 23.0


def test_header_refuses_a_buffer_with_two_producers():
    unit = stream("add", 2, lambda a, b: a + b)
    emit(unit)
    assert any(c.rule == "one-producer" for c in unit.checks)
    # a state waited from the reader and packed back into the same buffer
    unit.tile["unpack"].insert(0, Sync("unpack", "reader", "cb_wait_front(st, 1)", ""))
    unit.tile["pack"].insert(1, Call("pack_tile(dst + 3, st)", "the new state"))
    with pytest.raises(UnitError, match="buffer st is pushed by pack and reader"):
        emit(unit)


def _load(reg, at):
    args = {"lreg_ind": reg, "instr_mod0": 0, "sfpu_addr_mode": 7, "dest_reg_addr": at}
    return Insn("SFPLOAD", args, "")


def _swap(c, d):
    args = {"imm12_math": 0, "lreg_src_c": c, "lreg_dest": d, "instr_mod1": 1}
    return Insn("SFPSWAP", args, "")


def _max_code():
    # max into LReg 0 over 3 loads, then a clamp against constant register 12
    code = [_load(0, 0), _load(1, 2), _swap(0, 1), _load(1, 16), _swap(0, 1)]
    return code + [_swap(0, 12)]


def test_max_macro_takes_a_template_no_word_writes():
    code = _max_code()
    templates, done = max_macro(code)
    assert templates == {1: 0} and done[0].startswith("2 load and max pairs")
    macros = [x for x in code if x.op == "SFPLOADMACRO"]
    assert [x.args["lreg_ind"] >> 2 for x in macros] == [1, 1]


def _macro_unit(templates, code):
    unit = UnitProgram("maxes", "fp32")
    unit.init["math"] = macro_setup(templates)
    unit.tile["math"] = code
    return unit


def test_header_refuses_a_template_the_tile_overwrites_or_never_stores():
    code = _max_code()
    templates, _ = max_macro(code)
    emit(_macro_unit(templates, code[:-1]))
    # the back door open, a clamp against LReg 12 stores template 0 and runs nothing
    with pytest.raises(UnitError, match="has VD 12: with the back door open"):
        emit(_macro_unit(templates, code))
    clobbered = [*code[:-1], _swap(0, 13)]  # the clamp now writes template 1
    with pytest.raises(UnitError, match="writes VD 13 in the tile's code"):
        emit(_macro_unit(templates, clobbered))
    unstored = _macro_unit(templates, code)
    unstored.init["math"] = [x for x in unstored.init["math"] if x.op != "SFPSWAP"]
    with pytest.raises(UnitError, match="which the init does not store"):
        emit(unstored)


def test_row_operand_places_a_plain_row():
    out = row_operand([float(i) for i in range(64)], 2)
    # face c, rows 8 t .. 8 t + 3 hold columns 32 t + 16 c ..; the rest 0
    assert out[256 * 1 + 16 * 8 + 3] == 32 + 16 + 3 and out[16 * 4] == 0.0
    assert sum(1 for v in out if v) == 63 * 4  # each value in 4 rows, 0 once
    with pytest.raises(PlanError, match="has 64 values"):
        row_operand([0.0] * 32, 2)


def test_matrix_block_words_and_price():
    pairs = tile_products("HiFi2", 1, 1, 0)
    assert [x.op for _, x in pairs].count("MVMUL") == 32 and len(pairs) == 34
    assert block_cost("HiFi2", 1, 1) == (41.0, [w for w, _ in pairs])
    assert block_cost("HiFi4", 2, 2, 4)[0] == 1055.0


def test_band_takes_the_live_rows_at_run_time():
    whole = stream("add", 2, lambda a, b: a + b, load_macro=True)
    unit = stream("add", 2, lambda a, b: a + b, load_macro=True, rows=True)
    assert "inline void add(const uint32_t dst) {" in emit(whole)
    text = emit(unit)
    assert "inline void add(const uint32_t dst, const uint32_t rows) {" in text
    assert "face < (rows - 1u >= 16u ? 2u : 0u)" in text
    costs = [unit.cost("math", rows)[1] for rows in (1, 4, 5, 16, 17, 32)]
    assert costs[0] == costs[1] < costs[2] < costs[3] < costs[4] < costs[5]
    # at 32 rows: the whole tile's words; the bands' loop is priced at nothing
    assert costs[5] == whole.cost("math")[1]
    loops = [x for x in unit.tile["math"] if isinstance(x, Loop)]
    assert [x.count for x in loops][0] == 2 and len(loops) == 2


def test_band_in_the_row_form_guards_each_band():
    whole = row_program(norm_ir(), 2, 2, 4)
    unit = row_program(norm_ir(), 2, 2, 4, rows=True)
    assert "const uint32_t rows" in emit(unit)
    guards = [x for x in unit.tile["math"] if isinstance(x, Loop)]
    assert guards and all(x.count.guard for x in guards)
    words = sum(1 for _ in insns(whole.tile["math"]))
    assert sum(1 for _ in insns(unit.tile["math"])) == words
    assert unit.cost("math", 1)[1] < unit.cost("math", 9)[1] < unit.cost("math", 32)[1]


def code(text, name):
    """The C++ lines of `name` in an emitted header, without comments."""
    body = text.split(f"inline void {name}(", 1)[1].split("\n    }", 1)[0]
    return [x for x in body.splitlines()[1:] if not x.lstrip().startswith("//")]


def test_product_in_a_trace_is_one_body():
    alone = stream("mm", 2, lambda a, b: a @ b, fidelity="HiFi2")
    assert code(emit(alone), "mm") == code(emit(product_program("mm", "HiFi2")), "mm")
    gate = stream("g", 3, lambda a, b, c: trace.silu(a @ b) * (a @ c))
    assert gate.operands == [None, None] and [p.dst for p in gate.products] == [0, 1]
    text = emit(gate)
    assert text.count("TTI_MVMUL") == 2 * 64 and "(dst + 1);" in text
    vector = stream("v", 2, lambda a, b: trace.silu(a) * b)
    # the words of two HiFi4 products (66 a product; the last 8 cycles to the stall)
    assert gate.cost("math")[1] == vector.cost("math")[1] + 66 + 73
    band = stream("gb", 3, lambda a, b, c: trace.silu(a @ b) * (a @ c), rows=True)
    assert band.cost("math", 1)[1] < band.cost("math", 32)[1] == gate.cost("math")[1]


@pytest.mark.parametrize(
    "fn, refusal",
    [
        (lambda a, b, c: trace.silu(a) @ b, "a product after vector code"),
        (lambda a, b, c: (a @ b) @ c, "a product, its result in Dst"),
        (
            lambda a, b, c: (a @ b) * (b @ c) + (a @ c) * (c @ a) + b @ a,
            "reads at most 4 operand tiles",  # 5 products: Dst holds 4 operands
        ),
    ],
)
def test_product_refusals(fn, refusal):
    with pytest.raises(PlanError, match=re.escape(refusal)):
        stream("refused", 3, fn)


def test_flash_clamp_runs_with_the_back_door_open():
    unit = stream("e2", 1, lambda a: trace.exp2(a, error=2e-3, over=(-math.inf, 0.0)))
    words = [x for x in insns(unit.tile["math"]) if x.op.startswith("SFP")]
    held = [x.args.get("lreg_dest", x.args.get("lreg_ind")) for x in words]
    assert max(v for v in held if v is not None) < 12
    clamp = [i for i, x in enumerate(words) if x.note == "max(u, 512)"]
    assert clamp and all(words[i - 1].args["lreg_c"] == 12 for i in clamp)
    emit(unit)
    checks = {c.rule: c.verdict for c in unit.checks}
    assert checks["back-door"].startswith("ok")


def _col(fn):
    a, s, y = (TensorParam(n, n == "y", F32) for n in "asy")
    ir = KernelIR("col", [a, s, y], [], {})
    with trace.Tracer(ir):
        pa, ps, py = (trace.TensorProxy(t) for t in (a, s, y))
        for t in trace.grid(trace.num_tiles(pa)):
            x, rows = trace.load(pa.tiles[t]), trace.load(ps.tiles[t])
            trace.store(py.tiles[t], fn(x, rows.col()))
    return unit_program(ir)


@pytest.mark.parametrize(
    "fn",
    [lambda a, s: a * trace.rsqrt(s), lambda a, s: trace.sigmoid(a + s)],
)
def test_per_row_path_keeps_each_constant_register_load(fn):
    assert {0, "L0"} <= set(Insn("SFPCONFIG", _config(12, 0)).reads())
    assert not Insn("SFPCONFIG", _config(15, 1)).reads()  # Imm16 is the value
    unit = _col(fn)
    loaded = 0
    for x in insns(unit.init["math"]):
        if x.op == "SFPLOADI" and x.args["lreg_ind"] == 0:
            loaded += 1
        elif x.op == "SFPCONFIG" and 11 <= x.args["config_dest"] <= 14:
            assert loaded, f"LReg {x.args['config_dest']} takes a stale LReg 0"
            loaded = 0


def _config(vd, mod):
    return {"imm16_math": 0, "config_dest": vd, "instr_mod1": mod}


def test_too_many_live_values_is_a_unit_error():
    code = [
        Insn("SFPLOADI", {"lreg_ind": f"v{k}", "instr_mod0": 0, "imm16": k})
        for k in range(9)
    ]
    code += [_store(f"v{k}") for k in range(9)]
    with pytest.raises(UnitError, match="komorebi.dsl.tile: more live values"):
        allocate(code, {})


def _store(value):
    args = {"lreg_ind": value, "instr_mod0": 0, "sfpu_addr_mode": 7, "dest_reg_addr": 0}
    return Insn("SFPSTORE", args, "")
