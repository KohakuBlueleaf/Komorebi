"""Micro-op streams: header constants, encoding, check and the simulator's semantics."""

import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from komorebi.engines import stream as st
from komorebi.engines.stream import Reg, Stream, check, simulate

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


def binary(op, x, y, imm=False):
    """The value left by `op` on x and y (an immediate if imm) in a stream."""
    s = Stream()
    a, b, c = s.regs(3)
    s.set(a, x)
    if imm:
        getattr(s, op)(c, a, y)
    else:
        s.set(b, y)
        getattr(s, op)(c, a, b)
    s.halt(c)
    return s.simulate().halt


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
    "EV_TIMEOUT": "KM_EV_TIMEOUT",
    "EV_STREAM": "KM_EV_STREAM",
    "EV_STREAM_END": "KM_EV_STREAM_END",
    "EV_STREAM_ERROR": "KM_EV_STREAM_ERROR",
}


def test_constants_opcodes_and_error_reasons_match_the_header():
    for name, header_name in PAIRS.items():
        assert getattr(st, name) == H[header_name], name
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


def test_the_builder_refuses_what_it_cannot_encode_in_registers():
    s = Stream()
    s.regs(14)
    with pytest.raises(ValueError, match=r"2 requested, 1 free of 15"):
        s.regs(2)
    assert s.regs(1) == [Reg(14)]
    with pytest.raises(ValueError, match="out of registers: 1 requested, 0 free"):
        s.const(5)
    with pytest.raises(ValueError):
        Reg(16)
    b = Stream().regs(1)[0]
    with pytest.raises(TypeError):
        Stream().add(b, b, 3)
    with pytest.raises(TypeError):
        Stream().jmp(5)
    with pytest.raises(ValueError):
        Stream().set(b, 1 << 32)


def test_equal_constants_share_one_register():
    s = Stream()
    assert s.const(7) == s.const(7) != s.const(8)
    assert s.const(-1) == s.const(0xFFFFFFFF)
    assert [o.imm for o in s.uops()] == [7, 8, 0xFFFFFFFF]
    assert all(o.op == st.OP_SET for o in s.uops())


def test_labels_resolve_after_the_prologue_and_must_be_bound():
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


# micro-op semantics


def test_arithmetic_wraps_at_32_bits_and_shr_takes_the_count_mod_32():
    for op, x, y, want in [
        ("add", 0xFFFFFFFF, 2, 1),
        ("sub", 0, 1, 0xFFFFFFFF),
        ("mul", 0xFFFFFFFF, 0xFFFFFFFF, 1),
        ("and_", 0xF0F0, 0x0FF0, 0x00F0),
        ("mod", 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF),
    ]:
        assert binary(op, x, y) == want, op
    for op, x, imm, want in [
        ("addi", 0xFFFFFFFF, 1, 0),
        ("addi", 5, -2, 3),
        ("shr", 0x80000000, 31, 1),
        ("shr", 0xFFFFFFFF, 36, 0x0FFFFFFF),
    ]:
        assert binary(op, x, imm, imm=True) == want, op


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


def test_jumps_compare_unsigned():
    for op, x, y, taken in [
        ("jlt", 2, 2, False),
        ("jlt", 1, 0xFFFFFFFF, True),
        ("jeq", 4, 4, True),
        ("jeq", 4, 5, False),
        ("jnz", 0, None, False),
        ("jnz", 0xFFFFFFFF, None, True),
    ]:
        s = Stream()
        a, b, out = s.regs(3)
        skip = s.forward()
        s.set(a, x)
        s.set(b, y or 0)
        if op == "jnz":
            s.jnz(a, skip)
        else:
            getattr(s, op)(a, b, skip)
        s.set(out, 1)
        s.bind(skip)
        s.halt(out)
        assert s.simulate().halt == (0 if taken else 1), (op, x, y)


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


def test_a_job_that_can_never_fit_is_an_error():
    for count in [0, 120]:
        s = Stream()
        s.set(s.regs(1)[0], 0)
        s.submit(1, count)
        s.halt(s.const(0))
        done = s.simulate()
        assert done.error == (st.ERR_NEVER_FITS, pcs(s, st.OP_SUBMIT)[0])
        assert done.records == []
        assert done.events[-1].kind == st.EV_STREAM_ERROR
        assert done.events[-1].aux == 9 | done.error.pc << 8


def test_mod_by_zero_is_an_error_at_its_pc():
    s = Stream()
    a, b = s.regs(2)
    s.set(a, 5)
    s.mod(b, a, Reg(5))
    s.halt(a)
    done = s.simulate()
    assert done.error == (st.ERR_MOD_ZERO, 1) and done.passes == 1
    assert [e.kind for e in done.events] == [st.EV_STREAM_ERROR]


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


def test_cancel_ends_the_stream_at_the_pc_it_has_reached():
    for cancel in [0, 4]:
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


def test_alloc_is_the_lowest_first_that_overlaps_nothing():
    for busy, count, n, want in [
        ([], 5, 119, 0),
        ([], 120, 119, None),
        ([(2, 3)], 2, 119, 0),
        ([(0, 2), (4, 2)], 3, 119, 6),
        ([(0, 100)], 19, 119, 100),
        ([(10, 5), (0, 10)], 1, 119, 15),
    ]:
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


def test_a_job_is_passed_over_at_most_the_limit(monkeypatch):
    limit = 5
    monkeypatch.setattr(st, "QUEUE_PASSES", limit)
    big, ahead, done = issued_before_the_big_job(blocked_big_job(100))
    assert len(ahead) == limit
    assert big.pass_index == 3001  # when the long job is acknowledged


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
    assert progress.pass_index == 25
    assert [r.args[0] for r in done.records] == list(range(70))
    unlimited = s.simulate(delay=4, n_cores=119)
    assert unlimited.status == "ended"


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


def test_excl_waits_for_every_job_queued_before_it():
    s = Stream()
    for i in range(3):
        s.submit(1, 119, args=[i])
    s.issue(2, team=(100, 4), args=[3], exclusive=True)
    s.wait(0)
    s.halt(s.const(0))
    for delay in (1, 2, 3):
        done = s.simulate(delay=delay)
        got = [(r.args[0], r.queued) for r in done.records]
        assert got == [(0, True), (1, True), (2, True), (3, False)], (delay, got)


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


def test_bad_headers_are_reason_1_at_pc_0():
    for mutate in [
        lambda d: poke(d, 0, "<I", 0),
        lambda d: poke(d, 4, "<I", 2),
        lambda d: poke(d, 8, "<I", 0),
        lambda d: poke(d, 16, "<I", 0),
        lambda d: d + bytes(64),
        lambda d: d[:-1],
        lambda d: bytearray(),
    ]:
        assert check(bytes(mutate(tiny())), 1 << 20) == (st.ERR_HEADER, 0)


def test_the_upload_must_fit_the_region():
    data = bytes(tiny())
    assert check(data, len(data)) is None
    assert check(data, len(data) - 64) == (st.ERR_HEADER, 0)


def test_unknown_opcodes_are_reason_2():
    for code in (0, 21, 255):
        d = poke(tiny(), op_at(1), "<B", code)
        assert check(bytes(d), 1 << 20) == (st.ERR_OPCODE, 1), code


def test_registers_above_15_are_reason_4_for_fields_in_use():
    s = Stream()
    a = s.regs(1)[0]
    s.add(a, a, a)
    s.halt(a)
    for field in (1, 2, 3):
        data = bytearray(s.encode())
        poke(data, op_at(0) + field, "<B", 16)
        assert check(bytes(data), 1 << 20) == (st.ERR_REGISTER, 0)


def test_unused_operand_fields_are_ignored():
    d = tiny()
    poke(d, op_at(0) + 2, "<BB", 200, 201)  # SET uses only register a
    poke(d, op_at(1) + 1, "<BBB", 200, 201, 202)  # ISSUE uses none
    poke(d, op_at(1) + 8, "<I", 0xDEADBEEF)
    assert check(bytes(d), 1 << 20) is None


def test_jump_targets_at_or_above_n_ops_are_reason_3():
    for op in ["jmp", "jlt"]:
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


def test_bad_templates_are_reason_5_at_their_first_user():
    for offset, fmt, value in [
        (4, "<I", 0),  # segment 0
        (4, "<I", 250),
        (4, "<I", 1 | 8 << 8),  # KM_F_QUEUE is not for templates
        (64 + 10, "<B", 5),  # n_patch
        (64 + 8, "<B", 16),  # first_reg
    ]:
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


def test_a_bad_upload_posts_one_error_event_with_the_header_request_id():
    d = poke(tiny(req=77), op_at(1), "<B", 99)
    done = simulate(bytes(d))
    assert done.status == "error" and done.records == [] and done.passes == 1
    assert done.events == [st.Event(st.EV_STREAM_ERROR, 77, 2 | 1 << 8, 0)]
    done = simulate(bytes(d[:10]))
    assert done.events == [st.Event(st.EV_STREAM_ERROR, 0, 1, 0)]


# the team rule, RUN, builder validation, orphans


def test_a_team_outside_the_grid_is_found_without_a_sum():
    for first, count, outside in [
        (0, 119, False),
        (118, 2, True),
        (0, 0, True),
        (0xFFFFFFFF, 2, True),
        (1, 0xFFFFFFFF, True),
    ]:
        assert st.team_outside(first, count, 119) == outside


def test_run_answers_what_the_kernel_must_say():
    data = bytes(tiny(req=9))
    n = len(data)
    assert st.check_run(data, 0, 1 << 20) == (None, 9)
    assert st.check_run(data, 64, 1 << 20) == (None, 9)
    refused = st.StreamError(st.ERR_HEADER, 0)
    assert st.check_run(data, 8, 1 << 20) == (refused, 9)  # not a multiple of 64
    assert st.check_run(data, 0, n - 64) == (refused, 9)  # longer than the region
    assert st.check_run(data, 64, n + 32) == (refused, 9)  # ends outside
    assert st.check_run(data, 64, n + 64) == (None, 9)  # ends exactly at the end
    assert st.check_run(data, 1 << 32, 1 << 20) == (refused, 0)  # no sum is formed
    assert st.check_run(data, 0xFFFFFFC0, 1 << 20) == (refused, 0)
    assert st.check_run(data, 1 << 20, 1 << 20) == (refused, 0)  # header outside


def test_run_is_refused_over_an_active_stream_or_request():
    data = bytes(tiny(req=9))
    n = len(data)
    refused = st.StreamError(st.ERR_HEADER, 0)
    big = 1 << 20
    assert st.check_run(data, 0, big, active=[(0, n)]) == (refused, 9)
    assert st.check_run(data, n, big, active=[(0, n)]) == (None, 9)  # touching
    assert st.check_run(data, 64, big, active=[(0, 128)]) == (refused, 9)
    assert st.check_run(data, 0, big, active=[(n, n)]) == (None, 9)
    assert st.check_run(data, 0, big, active_reqs=[9]) == (refused, 9)
    assert st.check_run(data, 0, big, active_reqs=[8]) == (None, 9)


def test_the_builder_names_the_field_it_refuses():
    for field, value in (
        ("max_steps", 0),
        ("seed", 0),
        ("req", 1 << 32),
    ):
        with pytest.raises(ValueError, match=field):
            Stream(**{field: value})
    s = Stream()
    s.max_steps = 1 << 32
    s.halt(s.const(0))
    with pytest.raises(ValueError, match="max_steps"):
        s.encode()
    with pytest.raises(ValueError, match="pending"):
        s.wait(-1)
    with pytest.raises(ValueError, match="pending"):
        s.wait(1 << 32)


def test_the_builder_refuses_too_many_micro_ops():
    s = Stream()
    a = s.regs(1)[0]
    for _ in range(65535):
        s.mov(a, a)
    s.halt(a)
    with pytest.raises(ValueError, match="65536 micro-ops, at most 65535"):
        s.encode()


def test_the_default_deadline_applies_when_the_header_has_none():
    s = Stream(deadline_us=0)
    s.issue(1, team=(0, 1))
    s.halt(s.const(0))
    done = s.simulate(delay=None, default_deadline_passes=3)
    assert done.error == (st.ERR_DEADLINE, pcs(s, st.OP_ISSUE)[0]) and done.passes == 4
    t = Stream(deadline_us=1)
    t.issue(1, team=(0, 1))
    t.halt(t.const(0))
    assert t.simulate(delay=None, default_deadline_passes=3, max_passes=20).status == (
        "stuck"
    )


def test_a_late_orphan_posts_a_timeout_for_its_slot_with_the_ended_requests_id():
    s = Stream(req=33)
    a = s.regs(1)[0]
    s.issue(1, team=(0, 1), args=[1])
    s.issue(1, team=(1, 1), args=[2])
    s.issue(1, team=(2, 1), args=[3])
    s.mod(a, a, a)
    done = s.simulate(delay=lambda r: None if r.args[0] == 2 else 0, deadline_passes=5)
    assert done.error.reason == st.ERR_MOD_ZERO
    assert done.events[0].kind == st.EV_STREAM_ERROR
    assert done.events[1] == st.Event(st.EV_TIMEOUT, 33, 1, 5)
    assert done.stopped and done.passes == 6
    quiet = s.simulate(delay=1, deadline_passes=5)
    assert len(quiet.events) == 1 and not quiet.stopped


def test_the_two_issuers_take_turns_on_free_slots():
    s = Stream()
    s.set(s.regs(1)[0], 0)
    for _ in range(16):
        s.issue(1, team=(0, 1))
    s.submit(2, 1)
    s.wait(0)
    s.halt(s.const(0))
    done = s.simulate(delay=0)
    job = next(r for r in done.records if r.queued)
    assert job.pass_index % 2 == 1  # odd passes give the queue the first call
    assert all(r.pass_index == job.pass_index or not r.queued for r in done.records)


def test_the_example_in_the_docs_runs():
    text = (ROOT / "docs" / "streams.md").read_text()
    code = re.search(r"```python\n(.*?)```", text, re.S)[1]
    scope = {}
    exec(code, scope)
    done = scope["s"].simulate()
    assert done.status == "ended" and len(done.records) == 1000
    assert all(1 <= r.count <= 119 for r in done.records)
