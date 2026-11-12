"""Micro-op streams: header constants, encoding, check and the simulator's semantics."""

import json
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest
import stream_reference as ref

from komorebi.engines import stream as st
from komorebi.engines.stream import Reg, Stream, StreamFormatError, check, simulate

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "src" / "komorebi_native" / "include"
ABI = INCLUDE / "komorebi" / "abi"
DEFINE = re.compile(r"^#define\s+(KM_\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?\b", re.M)


def header_constants():
    found = {}
    for name in ("km_abi.h", "km_stream.h"):
        found.update(
            {k: int(v, 0) for k, v in DEFINE.findall((ABI / name).read_text())}
        )
    return found


H = header_constants()


def halt_value(build, **options):
    """The 32-bit status a stream halts with: build(s) returns the register."""
    s = Stream(max_steps=1000)
    s.halt(build(s))
    done = s.simulate(**options)
    assert done.status == "ended", done
    return done.halt


def pcs(s, op):
    """Program counters of the micro-ops with opcode `op`, prologue included."""
    return [pc for pc, o in enumerate(s.uops()) if o.op == op]


def run(build, **options):
    s = Stream(max_steps=100_000)
    build(s)
    return s.simulate(**options)


def xorshift_by_hand(x):
    x = (x ^ (x << 13)) & 0xFFFFFFFF
    x = x ^ (x >> 17)
    x = (x ^ (x << 5)) & 0xFFFFFFFF
    return x


# header constants


PAIRS = {
    "MAGIC": "KM_STREAM_MAGIC",
    "VERSION": "KM_STREAM_VERSION",
    "REGS": "KM_STREAM_REGS",
    "SEED_REG": "KM_STREAM_SEED_REG",
    "MAX_OPS": "KM_STREAM_MAX_OPS",
    "MAX_ACTIVE": "KM_STREAM_MAX_ACTIVE",
    "LINE": "KM_STREAM_LINE",
    "TPL_NONE": "KM_TPL_NONE",
    "PATCHES": "KM_TPL_PATCHES",
    "ARGS": "KM_TPL_ARGS",
    "STEP_OPS": "KM_STEP_OPS",
    "QUEUE_JOBS": "KM_QUEUE_JOBS",
    "QUEUE_LOOK": "KM_QUEUE_LOOK",
    "QUEUE_PASSES": "KM_QUEUE_PASSES",
    "RING_SLOTS": "KM_RING_SLOTS",
    "SEG_USER_MAX": "KM_SEG_USER_MAX",
    "F_EXCL": "KM_F_EXCL",
    "F_RESULT": "KM_F_RESULT",
    "F_QUEUE": "KM_F_QUEUE",
    "EV_STREAM": "KM_EV_STREAM",
    "EV_STREAM_END": "KM_EV_STREAM_END",
    "EV_STREAM_ERROR": "KM_EV_STREAM_ERROR",
}


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_python_constant_matches_the_header(name):
    assert getattr(st, name) == H[PAIRS[name]]


def test_opcodes_and_error_reasons_match_the_header():
    for code, (name, _, _) in st.OPS.items():
        assert H[f"KM_OP_{name}"] == code
    assert H["KM_OP_INVALID"] == 0 and len(st.OPS) == 21
    ops = {k for k in H if k.startswith("KM_OP_")}
    assert len(ops) == 22
    reasons = [
        "HEADER",
        "OPCODE",
        "JUMP",
        "REGISTER",
        "TEMPLATE",
        "STEPS",
        "MOD_ZERO",
        "DEADLINE",
    ]
    reasons += ["NEVER_FITS", "CANCELLED", "TEAM"]
    for number, name in enumerate(reasons, 1):
        assert H[f"KM_SERR_{name}"] == number == getattr(st, f"ERR_{name}")
    assert H["KM_SEG_RUN"] == 251 and H["KM_SEG_CANCEL"] == 250


def test_layout_matches_the_c_structs(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("needs a C compiler")
    src = tmp_path / "layout.c"
    src.write_text(
        '#include <stdio.h>\n#include "komorebi/abi/km_stream.h"\n'
        'int main(void) { printf("%zu %zu %zu %zu %zu %zu %zu\\n", sizeof(km_stream_t),'
        " sizeof(km_uop_t), sizeof(km_template_t), offsetof(km_stream_t, seed),"
        " offsetof(km_uop_t, imm), offsetof(km_template_t, arg),"
        " offsetof(km_template_t, first_reg)); return 0; }\n"
    )
    exe = tmp_path / "layout"
    cmd = [cc, "-std=c11", "-Wall", "-Werror", "-I", str(INCLUDE), str(src), "-o", exe]
    subprocess.run(cmd, check=True)
    out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    sizes = tuple(map(int, out.split()))
    assert sizes == (64, 16, 80, 28, 4, 64, 72)
    assert (st.HEADER.size, st.UOP.size, st.TAIL.size) == (64, 16, 16)
    assert 64 + st.TAIL.size == st.TEMPLATE_BYTES


# builder


def test_registers_run_out_with_the_count_in_the_message():
    s = Stream()
    s.regs(14)
    with pytest.raises(ValueError, match=r"2 requested, 1 free of 15"):
        s.regs(2)
    assert s.regs(1) == [Reg(14)]
    with pytest.raises(ValueError, match="out of registers: 1 requested, 0 free"):
        s.const(5)
    with pytest.raises(ValueError):
        Reg(16)


def test_equal_constants_share_one_register():
    s = Stream()
    assert s.const(7) == s.const(7) != s.const(8)
    assert s.const(-1) == s.const(0xFFFFFFFF)
    assert [o.imm for o in s.uops()] == [7, 8, 0xFFFFFFFF]
    assert all(o.op == st.OP_SET for o in s.uops())


def test_register_operands_must_be_registers():
    s = Stream()
    a = s.regs(1)[0]
    with pytest.raises(TypeError):
        s.add(a, a, 3)
    with pytest.raises(TypeError):
        s.jmp(5)
    with pytest.raises(ValueError):
        s.set(a, 1 << 32)


def test_labels_resolve_after_the_prologue():
    s = Stream()
    a = s.regs(1)[0]
    top, end = s.label(), s.forward()
    s.jlt(a, s.const(5), end)
    s.jmp(top)
    s.bind(end)
    s.halt(a)
    ops = s.uops()
    assert ops[0].op == st.OP_SET and ops[1].imm == 3 and ops[2].imm == 1
    with pytest.raises(ValueError, match="already bound"):
        s.bind(end)


def test_unbound_and_trailing_labels_do_not_encode():
    s = Stream()
    s.jmp(s.forward())
    with pytest.raises(ValueError, match="never bound"):
        s.encode()
    t = Stream()
    end = t.forward()
    t.jmp(end)
    t.bind(end)
    with pytest.raises(ValueError, match="past the last"):
        t.encode()


def test_template_arguments_and_limits():
    s = Stream()
    a, b = s.regs(2)
    s.issue(3, team=(a, 4), args=[1, a, 2, b], exclusive=True, result=True)
    t = s.templates()[0]
    assert (t.seg, t.flags) == (3, st.F_EXCL | st.F_RESULT)
    assert t.rec[2] == 4 << 16 and t.first_reg == 0 and t.count_reg == st.TPL_NONE
    assert t.n_patch == 2 and t.arg[:2] == (1, 3) and t.reg[:2] == (0, 1)
    assert t.rec[st.ARG_WORD[0]] == 1 and t.rec[st.ARG_WORD[2]] == 2
    with pytest.raises(ValueError, match="at most 4"):
        s.issue(1, team=(0, 1), args=[a] * 5)
    with pytest.raises(ValueError, match="11 arguments"):
        s.issue(1, team=(0, 1), args=[0] * 11)
    for bad in (0, 240, 250):
        with pytest.raises(ValueError, match="program segment"):
            s.issue(bad, team=(0, 1))
    with pytest.raises(ValueError, match="16 bits"):
        s.issue(1, team=(0, 70000))


def test_submit_takes_a_count_register_or_makes_one():
    s = Stream()
    c = s.regs(1)[0]
    s.submit(1, c)
    s.submit(1, 9)
    ops = s.uops()
    assert [o.op for o in ops] == [st.OP_SET, st.OP_SUBMIT, st.OP_SUBMIT]
    assert ops[1].a == 0 and ops[2].a == ops[0].a == 1 and ops[0].imm == 9


# encoding


def test_encoding_of_a_small_stream_by_hand():
    s = Stream(deadline_us=7, max_steps=9, seed=5, req=3)
    a = s.regs(1)[0]
    s.set(a, 0x01020304)
    s.issue(2, team=(1, 2), args=[0xAA, a])
    s.halt(a)
    data = s.encode()
    assert len(data) == 64 + 64 + 128
    assert data[:32] == struct.pack("<8I", 0x4B4D5354, 1, 3, 1, 9, 7, 3, 5)
    assert data[32:64] == bytes(32)
    assert data[64:80] == bytes([1, 0, 0, 0, 4, 3, 2, 1]) + bytes(8)
    assert data[80:96] == bytes([32, 0, 0, 0, 0, 0, 0, 0]) + bytes(8)
    assert data[96:112] == bytes([20, 0, 0, 0]) + bytes(12)
    assert data[112:128] == bytes(16)
    rec = struct.unpack_from("<16I", data, 128)
    assert rec[1] == 2 and rec[2] == 1 | 2 << 16 and rec[3] == 0xAA
    assert rec[0] == rec[4] == rec[8] == rec[12] == 0
    assert data[192:203] == bytes([1, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 1])
    assert data[203:] == bytes(256 - 203)


def test_round_trip_of_every_micro_op():
    s = Stream(deadline_us=5, max_steps=77, seed=0xABCDEF, req=42)
    a, b, c = s.regs(3)
    top, end = s.label(), s.forward()
    s.set(a, 3)
    s.mov(b, a)
    s.add(c, a, b)
    s.addi(c, c, 0xFFFFFFFF)
    s.sub(c, c, a)
    s.mul(c, c, b)
    s.and_(c, c, s.const(0xFF))
    s.shr(c, c, 3)
    s.mod(c, c, s.const(9))
    s.rand(a)
    s.jlt(a, b, top)
    s.jeq(a, b, end)
    s.jnz(a, end)
    s.time(b)
    s.event(b)
    s.issue(4, team=(1, c), args=[a, 5, b], exclusive=True)
    s.submit(5, count=c, args=[b, c, a, 9, 1, 2, 3, 4, 5, 6], result=True)
    s.wait(2)
    s.result(c)
    s.jmp(top)
    s.bind(end)
    s.halt(c)
    data = s.encode()
    d = Stream.decode(data)
    assert d.uops() == s.uops() and d.templates() == s.templates()
    assert (d.deadline_us, d.max_steps, d.seed, d.req) == (5, 77, 0xABCDEF, 42)
    assert d.encode() == data
    assert {o.op for o in d.uops()} == set(st.OPS)


def test_decode_refuses_a_bad_upload():
    data = bytearray(ref.build("arith_wrap")[0])
    data[0] ^= 1
    with pytest.raises(StreamFormatError) as err:
        Stream.decode(bytes(data))
    assert err.value.error == (1, 0)


# micro-op semantics


@pytest.mark.parametrize(
    ("op", "x", "y", "want"),
    [
        ("add", 0xFFFFFFFF, 2, 1),
        ("add", 5, 6, 11),
        ("sub", 0, 1, 0xFFFFFFFF),
        ("sub", 9, 4, 5),
        ("mul", 0x10001, 0x10001, 0x20001),
        ("mul", 0xFFFFFFFF, 0xFFFFFFFF, 1),
        ("and_", 0xF0F0, 0x0FF0, 0x00F0),
        ("mod", 1000, 7, 6),
        ("mod", 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF),
        ("mod", 3, 5, 3),
    ],
)
def test_three_register_ops(op, x, y, want):
    def build(s):
        a, b, c = s.regs(3)
        s.set(a, x)
        s.set(b, y)
        getattr(s, op)(c, a, b)
        return c

    assert halt_value(build) == want


@pytest.mark.parametrize(
    ("op", "x", "imm", "want"),
    [
        ("addi", 0xFFFFFFFF, 1, 0),
        ("addi", 5, 0xFFFFFFFF, 4),
        ("addi", 5, -2, 3),
        ("shr", 0x80000000, 31, 1),
        ("shr", 0xFFFFFFFF, 0, 0xFFFFFFFF),
        ("shr", 0xFFFFFFFF, 32, 0xFFFFFFFF),
        ("shr", 0xFFFFFFFF, 36, 0x0FFFFFFF),
    ],
)
def test_immediate_ops(op, x, imm, want):
    def build(s):
        a, b = s.regs(2)
        s.set(a, x)
        getattr(s, op)(b, a, imm)
        return b

    assert halt_value(build) == want


def test_mov_set_and_register_zero_at_start():
    def build(s):
        a, b = s.regs(2)
        s.addi(b, a, 7)
        s.mov(a, b)
        return a

    assert halt_value(build) == 7


def test_rand_follows_the_xorshift_written_out():
    s = Stream(seed=1)
    a = s.regs(1)[0]
    for _ in range(6):
        s.rand(a)
        s.event(a)
    s.halt(a)
    done = s.simulate()
    x, want = 1, []
    for _ in range(6):
        x = xorshift_by_hand(x)
        want.append(x & 0xFFFFFF)
    assert [e.aux for e in done.events[:6]] == want
    assert done.halt == x
    assert want[0] == 270369  # 1 ^ 1 << 13, then the two shifts


def test_rand_runs_on_register_15_whatever_the_target():
    s = Stream(seed=0x1234)
    a = s.regs(1)[0]
    s.rand(a)
    s.rand(Reg(15))
    s.rand(a)
    s.halt(Reg(15))
    x = 0x1234
    for _ in range(3):
        x = xorshift_by_hand(x)
    assert s.simulate().halt == x


@pytest.mark.parametrize(
    ("op", "x", "y", "taken"),
    [
        ("jlt", 1, 2, True),
        ("jlt", 2, 2, False),
        ("jlt", 0xFFFFFFFF, 1, False),
        ("jlt", 1, 0xFFFFFFFF, True),
        ("jeq", 4, 4, True),
        ("jeq", 4, 5, False),
    ],
)
def test_two_register_jumps(op, x, y, taken):
    def build(s):
        a, b, out = s.regs(3)
        skip = s.forward()
        s.set(a, x)
        s.set(b, y)
        getattr(s, op)(a, b, skip)
        s.set(out, 1)
        s.bind(skip)
        return out

    assert halt_value(build) == (0 if taken else 1)


@pytest.mark.parametrize(("x", "taken"), [(0, False), (1, True), (0xFFFFFFFF, True)])
def test_jnz(x, taken):
    def build(s):
        a, out = s.regs(2)
        skip = s.forward()
        s.set(a, x)
        s.jnz(a, skip)
        s.set(out, 1)
        s.bind(skip)
        return out

    assert halt_value(build) == (0 if taken else 1)


def test_halt_status_and_event_keep_the_low_24_bits():
    s = Stream()
    a = s.regs(1)[0]
    s.set(a, 0xAB123456)
    s.event(a)
    s.halt(a)
    done = s.simulate()
    assert done.halt == 0xAB123456
    assert [(e.kind, e.aux) for e in done.events] == [(4, 0x123456), (5, 0x123456)]
    assert all(e.req == 1 for e in done.events)


def test_time_reads_the_clock_of_the_pass():
    s = Stream()
    a, b = s.regs(2)
    for _ in range(64):
        s.addi(b, b, 1)
    s.time(a)
    s.halt(a)
    done = s.simulate(ticks_per_pass=10)
    assert done.halt == 10 and done.passes == 2


# steps, passes and errors


def test_a_pass_runs_at_most_64_micro_ops():
    for n, passes in ((63, 1), (64, 2), (127, 2), (128, 3)):
        s = Stream(max_steps=1000)
        a = s.regs(1)[0]
        for _ in range(n):
            s.addi(a, a, 1)
        s.halt(a)
        done = s.simulate()
        assert (done.halt, done.passes) == (n, passes), n


def test_step_limit_counts_executed_micro_ops():
    def ops(limit):
        s = Stream(max_steps=limit)
        a = s.regs(1)[0]
        s.addi(a, a, 1)
        s.addi(a, a, 1)
        s.halt(a)
        return s.simulate()

    assert ops(3).status == "ended"
    done = ops(2)
    assert done.error == (st.ERR_STEPS, 2) and done.halt is None
    assert ops(1).error == (st.ERR_STEPS, 1)


def test_a_stalled_micro_op_does_not_count_as_a_step():
    s = Stream(max_steps=5)
    s.issue(1, team=(0, 1))
    s.wait(0)
    s.halt(s.const(0))
    assert s.simulate(delay=50).status == "ended"


@pytest.mark.parametrize("count", [0, 120, 1 << 31])
def test_a_job_that_can_never_fit_is_an_error(count):
    s = Stream()
    s.set(s.regs(1)[0], 0)
    s.submit(1, count)
    s.halt(s.const(0))
    done = s.simulate()
    assert done.error == (st.ERR_NEVER_FITS, pcs(s, st.OP_SUBMIT)[0])
    assert done.records == []
    assert done.events[-1].kind == st.EV_STREAM_ERROR
    assert done.events[-1].aux == 9 | done.error.pc << 8


def test_a_job_of_the_whole_grid_fits():
    s = Stream()
    s.submit(1, 119)
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(n_cores=119)
    assert [(r.first, r.count) for r in done.records] == [(0, 119)]
    assert s.simulate(n_cores=118).error.reason == st.ERR_NEVER_FITS


def test_mod_by_zero_is_an_error_at_its_pc():
    s = Stream()
    a, b = s.regs(2)
    s.set(a, 5)
    s.mod(b, a, Reg(5))
    s.halt(a)
    done = s.simulate()
    assert done.error == (st.ERR_MOD_ZERO, 1) and done.passes == 1
    assert [e.kind for e in done.events] == [st.EV_STREAM_ERROR]


@pytest.mark.parametrize(
    ("first", "count"), [(100, 20), (119, 1), (0, 120), (0, 0), (5, 0), (1 << 20, 1)]
)
def test_a_team_outside_the_grid_is_an_error(first, count):
    s = Stream()
    f, c = s.regs(2)
    s.set(f, first)
    s.set(c, count)
    s.issue(1, team=(f, c))
    s.halt(f)
    done = s.simulate()
    assert done.error == (st.ERR_TEAM, 2) and done.records == []


def test_the_last_core_is_a_valid_team():
    s = Stream()
    s.issue(1, team=(118, 1))
    s.wait(0)
    s.halt(s.const(0))
    assert [(r.first, r.count) for r in s.simulate().records] == [(118, 1)]


def test_running_off_the_end_is_a_jump_error_at_n_ops():
    s = Stream()
    a = s.regs(1)[0]
    s.set(a, 1)
    done = s.simulate()
    assert done.error == (st.ERR_JUMP, 1)


def test_a_record_deadline_ends_the_stream_and_stops_the_grid():
    s = Stream()
    s.issue(1, team=(0, 2))
    s.issue(1, team=(2, 2))
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=lambda r: None if r.seq == 2 else 0, deadline_passes=3)
    assert done.error == (st.ERR_DEADLINE, pcs(s, st.OP_ISSUE)[1]) and done.stopped
    assert done.passes == 4 and len(done.records) == 2
    ok = s.simulate(delay=2, deadline_passes=3)
    assert ok.status == "ended" and not ok.stopped
    late = s.simulate(delay=3, deadline_passes=3)
    assert late.error == (st.ERR_DEADLINE, pcs(s, st.OP_ISSUE)[0])


def test_a_stream_that_never_finishes_reports_stuck():
    s = Stream()
    s.issue(1, team=(0, 1))
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=None, max_passes=50)
    assert (done.status, done.passes, done.error) == ("stuck", 50, None)


@pytest.mark.parametrize("cancel", [0, 1, 4])
def test_cancel_ends_the_stream_at_the_pc_it_has_reached(cancel):
    s = Stream(max_steps=10_000)
    top = s.label()
    s.issue(1, team=(0, 1))
    s.wait(0)
    s.jmp(top)
    done = s.simulate(delay=1, cancel_pass=cancel)
    pcs = {0: 0, 1: 1, 4: 1}[cancel]
    assert done.error == (st.ERR_CANCELLED, pcs)
    assert done.events[-1].aux == 10 | pcs << 8 and done.passes == cancel + 1


def test_waits_and_the_completion_order():
    s = Stream()
    for i in range(4):
        s.issue(1, team=(i, 1), args=[i])
    s.wait(2)
    s.event(s.const(1))
    s.wait(0)
    s.event(s.const(2))
    s.halt(s.const(0))
    done = s.simulate(delay=lambda r: 3 - r.args[0])
    e = {x.aux: x.pass_index for x in done.events if x.kind == st.EV_STREAM}
    assert e == {1: 2, 2: 4}
    done = s.simulate(delay=0)
    assert {x.aux: x.pass_index for x in done.events if x.kind == st.EV_STREAM} == {
        1: 1,
        2: 1,
    }


def test_halt_waits_for_every_record():
    s = Stream()
    s.issue(1, team=(0, 1))
    s.halt(s.const(9))
    done = s.simulate(delay=6)
    assert done.passes == 8 and done.halt == 9


def test_result_is_the_latest_completed_result_record():
    s = Stream()
    r1, r2, r3 = s.regs(3)
    s.result(r1)
    s.issue(1, team=(0, 2), args=[5], result=True)
    s.issue(1, team=(2, 2), args=[7])
    s.wait(0)
    s.result(r2)
    s.issue(1, team=(0, 3), args=[1], result=True)
    s.issue(1, team=(3, 3), args=[2], result=True)
    s.wait(0)
    s.result(r3)
    s.event(r1)
    s.event(r2)
    s.event(r3)
    s.halt(r3)

    def fn(r):
        return r.args[0] * 100 + r.count

    done = s.simulate(result=fn, delay=0)
    # records 3 and 4 are swept together; the higher slot is swept last
    assert [e.aux for e in done.events[:3]] == [0, 502, 203]


def test_result_order_follows_completion_not_issue():
    s = Stream()
    r = s.regs(1)[0]
    s.issue(1, team=(0, 1), args=[1], result=True)
    s.issue(1, team=(1, 1), args=[2], result=True)
    s.wait(0)
    s.result(r)
    s.halt(r)

    def late_first(rec):
        return 5 if rec.args[0] == 1 else 1

    assert s.simulate(delay=late_first, result=lambda x: x.args[0]).halt == 1
    assert s.simulate(delay=lambda x: x.args[0], result=lambda x: x.args[0]).halt == 2


def test_issued_records_carry_the_template_and_the_patches():
    s = Stream()
    a, b, f = s.regs(3)
    s.set(a, 11)
    s.set(b, 22)
    s.set(f, 3)
    s.issue(7, team=(f, 2), args=[1, a, 2, b, 5, 6, 7, 8, 9, 10], exclusive=True)
    s.wait(0)
    s.halt(a)
    (r,) = s.simulate().records
    assert (r.seg, r.first, r.count, r.flags) == (7, 3, 2, st.F_EXCL)
    assert r.args == (1, 11, 2, 22, 5, 6, 7, 8, 9, 10)
    assert (r.seq, r.slot, r.pc, r.queued, r.pass_index) == (1, 0, 3, False, 0)


def test_a_later_patch_of_the_same_argument_wins():
    s = Stream()
    x, y = s.regs(2)
    s.set(x, 1)
    s.set(y, 2)
    s.issue(1, team=(0, 1), args=[x, y])
    s.halt(x)
    data = bytearray(s.encode())
    base = st.HEADER_BYTES + 64 + 64  # one line of micro-ops, then the record
    poke(data, base + 1, "<B", 0)  # both patches name argument 0
    assert simulate(bytes(data)).records[0].args[0] == 2


# slots, allocator, queue


def test_slots_are_the_lowest_free_one():
    s = Stream()
    for i in range(3):
        s.issue(1, team=(i, 1), args=[i])
    s.wait(0)
    s.issue(1, team=(0, 1), args=[9])
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=lambda r: 4 if r.args[0] == 0 else 0)
    assert [r.slot for r in done.records] == [0, 1, 2, 0]
    s2 = Stream()
    for i in range(3):
        s2.issue(1, team=(i, 1), args=[i])
    s2.wait(2)
    s2.issue(1, team=(5, 1), args=[7])
    s2.wait(0)
    s2.halt(s2.const(0))
    done = s2.simulate(delay=lambda r: 4 if r.args[0] == 0 else 0)
    assert [r.slot for r in done.records] == [0, 1, 2, 1]


def test_issue_waits_while_all_16_slots_are_busy():
    s = Stream()
    top = s.label()
    s.issue(1, team=(0, 1))
    s.addi(s.regs(1)[0], Reg(0), 1)
    s.jmp(top)
    done = s.simulate(delay=None, max_passes=5)
    assert len(done.records) == 16 and done.status == "stuck"
    assert [r.slot for r in done.records] == list(range(16))


@pytest.mark.parametrize(
    ("busy", "count", "n", "want"),
    [
        ([], 5, 119, 0),
        ([], 119, 119, 0),
        ([], 120, 119, None),
        ([(0, 3)], 4, 119, 3),
        ([(2, 3)], 2, 119, 0),
        ([(2, 3)], 3, 119, 5),
        ([(0, 2), (4, 2)], 2, 119, 2),
        ([(0, 2), (4, 2)], 3, 119, 6),
        ([(0, 100)], 20, 119, None),
        ([(0, 100)], 19, 119, 100),
        ([(118, 1)], 1, 119, 0),
        ([(0, 118)], 1, 119, 118),
        ([(0, 119)], 1, 119, None),
        ([(10, 5), (0, 10)], 1, 119, 15),
    ],
)
def test_alloc_is_the_lowest_first_that_overlaps_nothing(busy, count, n, want):
    assert st.alloc(busy, count, n) == want


def test_the_queue_picks_teams_with_the_allocator():
    s = Stream()
    for c in (10, 20, 5):
        s.submit(1, c, args=[c])
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=3)
    assert [(r.first, r.count, r.slot) for r in done.records] == [
        (0, 10, 0),
        (10, 20, 1),
        (30, 5, 2),
    ]
    assert all(r.queued and r.pass_index == 0 for r in done.records)


def test_a_job_that_does_not_fit_does_not_block_those_behind_it():
    s = Stream()
    s.submit(1, 100, args=[1])
    s.submit(1, 100, args=[2])
    s.submit(1, 10, args=[3])
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=2)
    assert [(r.args[0], r.first, r.pass_index) for r in done.records] == [
        (1, 0, 0),
        (3, 100, 0),
        (2, 0, 3),
    ]


def test_the_queue_looks_at_the_first_16_jobs_only():
    def first_pass_of_small(blockers):
        s = Stream()
        for _ in range(blockers):
            s.submit(1, 100, args=[0])
        s.submit(1, 19, args=[1])
        s.wait(0)
        s.halt(s.const(0))
        done = s.simulate(delay=2)
        return next(r for r in done.records if r.args[0] == 1).pass_index

    assert first_pass_of_small(15) == 0
    assert first_pass_of_small(16) > 0


def blocked_big_job(smalls):
    """A long job, a 119-core job, then `smalls` one-core jobs."""
    s = Stream(max_steps=100_000)
    i = s.regs(1)[0]
    s.submit(1, 1, args=[0])
    s.submit(2, 119, args=[1])
    top = s.label()
    s.submit(1, 1, args=[2])
    s.addi(i, i, 1)
    s.jlt(i, s.const(smalls), top)
    s.wait(0)
    s.halt(s.const(0))
    return s


def long_first_job(r):
    return 3000 if r.seq == 1 else 1


def issued_before_the_big_job(s):
    done = s.simulate(delay=long_first_job, max_passes=10_000)
    assert done.status == "ended"
    (big,) = [r for r in done.records if r.seg == 2]
    ahead = [r for r in done.records if r.seg == 1 and r.seq > 1]
    return big, [r for r in ahead if r.pass_index < big.pass_index], done


@pytest.mark.parametrize("limit", [5, 1024])
def test_a_job_is_passed_over_at_most_the_limit(monkeypatch, limit):
    monkeypatch.setattr(st, "QUEUE_PASSES", limit)
    big, ahead, done = issued_before_the_big_job(blocked_big_job(1500))
    assert len(ahead) == limit
    assert big.pass_index == 3001  # when the long job is acknowledged
    assert len(done.records) == 1502


def test_without_the_limit_the_large_job_waits_for_every_small_one(monkeypatch):
    monkeypatch.setattr(st, "QUEUE_PASSES", 10**9)
    big, ahead, _ = issued_before_the_big_job(blocked_big_job(1500))
    assert len(ahead) == 1500 and big.pass_index == 3001


def test_a_full_queue_makes_submit_wait():
    s = Stream()
    for i in range(70):
        s.submit(1, 119, args=[i])
    s.event(s.const(1))
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=4)
    (progress,) = [e for e in done.events if e.kind == st.EV_STREAM]
    # 64 jobs fit in pass 0; each of the six more waits for one job to leave the queue
    assert progress.pass_index == 26
    assert [r.args[0] for r in done.records] == list(range(70))
    unlimited = s.simulate(delay=4, n_cores=119)
    assert unlimited.status == "ended"


def test_the_queue_holds_exactly_64_jobs():
    s = Stream()
    for _ in range(65):
        s.submit(1, 119)
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=1)
    assert done.status == "ended" and len(done.records) == 65


def test_wait_counts_queued_jobs_as_pending():
    s = Stream()
    s.submit(1, 119)
    s.submit(1, 119)
    s.wait(1)
    s.event(s.const(1))
    s.halt(s.const(0))
    done = s.simulate(delay=2)
    ev = next(e for e in done.events if e.kind == st.EV_STREAM)
    assert ev.pass_index == 3


def test_a_cancelled_stream_drops_its_queue():
    s = Stream()
    for _ in range(10):
        s.submit(1, 119)
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=5, cancel_pass=2)
    assert done.error.reason == st.ERR_CANCELLED and len(done.records) == 1


def test_excl_waits_for_an_overlapping_team_and_others_do_not():
    s = Stream()
    s.issue(1, team=(0, 10), args=[0])
    s.issue(1, team=(5, 10), args=[1], exclusive=True)
    s.issue(1, team=(20, 5), args=[2], exclusive=True)
    s.issue(1, team=(8, 4), args=[3])
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=2)
    got = [(r.args[0], r.pass_index, r.flags) for r in done.records]
    assert got == [(0, 0, 0), (1, 3, 2), (2, 3, 2), (3, 3, 0)]


def test_the_section_five_program_gives_1000_records():
    s = Stream(deadline_us=200_000)
    i, count = s.regs(2)
    s.set(i, 0)
    top = s.label()
    s.rand(count)
    s.mod(count, count, s.const(119))
    s.addi(count, count, 1)
    s.submit(seg=3, count=count, args=[i, 77])
    s.addi(i, i, 1)
    s.jlt(i, s.const(1000), top)
    s.wait(0)
    s.halt(s.const(0))
    data = s.encode()
    assert check(data, 1 << 20) is None
    done = s.simulate(delay=lambda r: 1 + r.seq % 3)
    assert done.status == "ended" and len(done.records) == 1000
    counts, x = [], s.seed
    for _ in range(1000):
        x = xorshift_by_hand(x)
        counts.append(x % 119 + 1)
    by_job = sorted(done.records, key=lambda r: r.args[0])
    assert [r.count for r in by_job] == counts
    assert {r.count for r in done.records} <= set(range(1, 120))
    assert min(counts) == 1 and max(counts) == 119
    assert all(r.first + r.count <= 119 and r.args[1] == 77 for r in done.records)
    assert len({r.slot for r in done.records}) > 1  # several teams at once
    assert [e.kind for e in done.events] == [st.EV_STREAM_END]


def test_jobs_in_flight_never_overlap_in_the_simulation():
    s = Stream()
    i, c = s.regs(2)
    top = s.label()
    s.rand(c)
    s.mod(c, c, s.const(40))
    s.addi(c, c, 1)
    s.submit(1, c, args=[i])
    s.addi(i, i, 1)
    s.jlt(i, s.const(300), top)
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=lambda r: 1 + r.seq % 4)
    for r in done.records:
        for q in done.records:
            live = q.seq < r.seq and q.pass_index + 1 + (1 + q.seq % 4) > r.pass_index
            if live:
                assert q.first + q.count <= r.first or r.first + r.count <= q.first


# check: every reason, with its program counter


def tiny(**changes):
    s = Stream(**changes)
    a = s.regs(1)[0]
    s.set(a, 1)
    s.issue(1, team=(0, 1))
    s.halt(a)
    return bytearray(s.encode())


def poke(data, offset, fmt, *values):
    struct.pack_into(fmt, data, offset, *values)
    return data


def op_at(index):
    return st.HEADER_BYTES + index * st.UOP_BYTES


TPL = st.HEADER_BYTES + 64  # three micro-ops pad to one line


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: poke(d, 0, "<I", 0),
        lambda d: poke(d, 4, "<I", 2),
        lambda d: poke(d, 8, "<I", 0),
        lambda d: poke(d, 8, "<I", 65536),
        lambda d: poke(d, 8, "<I", 5),
        lambda d: poke(d, 12, "<I", 65536),
        lambda d: poke(d, 12, "<I", 2),
        lambda d: poke(d, 16, "<I", 0),
        lambda d: poke(d, 28, "<I", 0),
        lambda d: d + bytes(64),
        lambda d: d[:-64],
        lambda d: d[:-1],
        lambda d: d[:63],
        lambda d: bytearray(),
    ],
)
def test_bad_headers_are_reason_1_at_pc_0(mutate):
    assert check(bytes(mutate(tiny())), 1 << 20) == (st.ERR_HEADER, 0)


def test_the_upload_must_fit_the_region():
    data = bytes(tiny())
    assert check(data, len(data)) is None
    assert check(data, len(data) - 64) == (st.ERR_HEADER, 0)


def test_a_clean_upload_checks_clean():
    assert check(bytes(tiny()), 1 << 20) is None


def test_unknown_opcodes_are_reason_2():
    for code in (0, 11, 15, 21, 31, 34, 39, 42, 47, 50, 255):
        d = poke(tiny(), op_at(1), "<B", code)
        assert check(bytes(d), 1 << 20) == (st.ERR_OPCODE, 1), code


@pytest.mark.parametrize("field", [1, 2, 3])
def test_registers_above_15_are_reason_4_for_fields_in_use(field):
    s = Stream()
    a = s.regs(1)[0]
    s.add(a, a, a)
    s.halt(a)
    data = bytearray(s.encode())
    poke(data, op_at(0) + field, "<B", 16)
    assert check(bytes(data), 1 << 20) == (st.ERR_REGISTER, 0)
    poke(data, op_at(0) + field, "<B", 255)
    assert check(bytes(data), 1 << 20) == (st.ERR_REGISTER, 0)


def test_unused_operand_fields_are_ignored():
    d = tiny()
    poke(d, op_at(0) + 2, "<BB", 200, 201)  # SET uses only register a
    poke(d, op_at(1) + 1, "<BBB", 200, 201, 202)  # ISSUE uses none
    poke(d, op_at(1) + 8, "<I", 0xDEADBEEF)
    assert check(bytes(d), 1 << 20) is None


@pytest.mark.parametrize("op", ["jmp", "jlt", "jeq", "jnz"])
def test_jump_targets_at_or_above_n_ops_are_reason_3(op):
    s = Stream()
    a, b = s.regs(2)
    top = s.label()
    if op == "jmp":
        s.jmp(top)
    elif op == "jnz":
        s.jnz(a, top)
    else:
        getattr(s, op)(a, b, top)
    s.halt(a)
    data = bytearray(s.encode())
    assert check(bytes(data), 1 << 20) is None
    poke(data, op_at(0) + 4, "<I", 2)
    assert check(bytes(data), 1 << 20) == (st.ERR_JUMP, 0)
    poke(data, op_at(0) + 4, "<I", 1)
    assert check(bytes(data), 1 << 20) is None
    poke(data, op_at(0) + 4, "<I", 0xFFFFFFFF)
    assert check(bytes(data), 1 << 20) == (st.ERR_JUMP, 0)


def test_template_index_out_of_range_is_reason_5_at_the_micro_op():
    for index in (1, 7, 0xFFFFFFFF):
        d = poke(tiny(), op_at(1) + 4, "<I", index)
        assert check(bytes(d), 1 << 20) == (st.ERR_TEMPLATE, 1)


@pytest.mark.parametrize(
    ("offset", "fmt", "value"),
    [
        (4, "<I", 0),  # segment 0
        (4, "<I", 240),
        (4, "<I", 250),
        (4, "<I", 255),
        (4, "<I", 1 | 1 << 8),  # reserved flag bit
        (4, "<I", 1 | 8 << 8),  # KM_F_QUEUE is not for templates
        (4, "<I", 1 | 16 << 8),
        (4, "<I", 1 | 128 << 8),
        (64 + 10, "<B", 5),  # n_patch
        (64 + 8, "<B", 16),  # first_reg
        (64 + 9, "<B", 200),  # count_reg
    ],
)
def test_bad_templates_are_reason_5_at_their_first_user(offset, fmt, value):
    d = poke(tiny(), TPL + offset, fmt, value)
    assert check(bytes(d), 1 << 20) == (st.ERR_TEMPLATE, 1)


def test_patch_entries_are_checked_only_up_to_n_patch():
    d = tiny()
    poke(d, TPL + 64 + 10, "<B", 1)
    poke(d, TPL + 64, "<B", 10)  # argument number 10
    assert check(bytes(d), 1 << 20) == (st.ERR_TEMPLATE, 1)
    poke(d, TPL + 64, "<B", 9)
    poke(d, TPL + 64 + 4, "<B", 16)  # register 16
    assert check(bytes(d), 1 << 20) == (st.ERR_TEMPLATE, 1)
    poke(d, TPL + 64 + 4, "<B", 15)
    assert check(bytes(d), 1 << 20) is None
    poke(d, TPL + 64 + 10, "<B", 0)
    poke(d, TPL + 64 + 1, "<B", 99)  # beyond n_patch: ignored
    assert check(bytes(d), 1 << 20) is None


def test_an_unused_bad_template_is_reported_at_pc_0():
    s = Stream()
    s.issue(1, team=(0, 1))
    s.issue(2, team=(0, 1))
    s.halt(s.const(0))
    data = bytearray(s.encode())
    base = st.HEADER_BYTES + 64 + st.TEMPLATE_BYTES
    poke(data, base + 4, "<I", 250)
    assert check(bytes(data), 1 << 20) == (st.ERR_TEMPLATE, 2)
    poke(data, op_at(2) + 4, "<I", 0)  # the second issue now uses template 0
    assert check(bytes(data), 1 << 20) == (st.ERR_TEMPLATE, 0)


def test_the_first_fault_in_micro_op_order_wins():
    d = tiny()
    poke(d, op_at(2), "<B", 99)
    poke(d, op_at(1) + 4, "<I", 9)
    poke(d, TPL + 4, "<I", 250)
    assert check(bytes(d), 1 << 20) == (st.ERR_TEMPLATE, 1)
    poke(d, op_at(0), "<B", 0)
    assert check(bytes(d), 1 << 20) == (st.ERR_OPCODE, 0)


def test_a_bad_upload_posts_one_error_event_with_the_header_request_id():
    d = poke(tiny(req=77), op_at(1), "<B", 99)
    done = simulate(bytes(d))
    assert done.status == "error" and done.records == [] and done.passes == 1
    assert done.events == [st.Event(st.EV_STREAM_ERROR, 77, 2 | 1 << 8, 0)]
    done = simulate(bytes(d[:10]))
    assert done.events == [st.Event(st.EV_STREAM_ERROR, 0, 1, 0)]


def test_error_event_status_word_packs_reason_then_pc():
    done = simulate(bytes(poke(tiny(), op_at(2), "<B", 99)))
    (e,) = done.events
    assert e.kind | e.aux << 8 == 6 | (2 | 2 << 8) << 8


# reference streams


NAMES = sorted(ref.REFERENCES)


def test_there_are_at_least_20_reference_streams():
    assert len(NAMES) >= 20
    assert len({p.stem for p in ref.DATA.glob("*.bin")}) == len(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_reference_files_are_what_the_generator_writes(name):
    upload, doc = ref.document(name)
    assert (ref.DATA / f"{name}.bin").read_bytes() == upload
    assert json.loads((ref.DATA / f"{name}.json").read_text()) == json.loads(
        json.dumps(doc)
    )


@pytest.mark.parametrize("name", NAMES)
def test_reference_answers_replay_from_the_files_alone(name):
    upload = (ref.DATA / f"{name}.bin").read_bytes()
    doc = json.loads((ref.DATA / f"{name}.json").read_text())
    want = doc["expect"]
    done = ref.run(upload, doc["params"])
    assert [
        [r.seg, r.first, r.count, r.flags, list(r.args)] for r in done.records
    ] == want["records"]
    assert [list(e) for e in done.events] == want["events"]
    assert (done.status, done.halt, done.passes) == (
        want["status"],
        want["halt"],
        want["passes"],
    )
    assert (list(done.error) if done.error else None) == want["error"]
    assert list(check(upload, ref.REGION) or []) == (want["check"] or [])


def test_reference_streams_reach_every_error_reason():
    reasons = set()
    for name in NAMES:
        doc = json.loads((ref.DATA / f"{name}.json").read_text())
        if doc["expect"]["error"]:
            reasons.add(doc["expect"]["error"][0])
    assert reasons == set(range(1, 12))


def test_reference_streams_use_every_micro_op():
    seen = set()
    for name in NAMES:
        upload = (ref.DATA / f"{name}.bin").read_bytes()
        if check(upload, ref.REGION) is None:
            seen |= {o.op for o in Stream.decode(upload).uops()}
    assert seen == set(st.OPS)
