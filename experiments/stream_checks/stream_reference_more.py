"""Reference streams that pin behaviour the first set leaves open.

Each builder gives a `Stream` or upload bytes; `MORE` maps name to (builder,
parameters, what it shows). `stream_reference` merges `MORE` into its table.
"""

import struct

from komorebi.engines import stream as st
from komorebi.engines.stream import Reg, Stream

SEG_A, SEG_B = 1, 7


def _halt_with(s, value):
    s.halt(s.const(value))
    return s


def _poke(data, offset, fmt, *values):
    out = bytearray(data)
    struct.pack_into(fmt, out, offset, *values)
    return bytes(out)


def _op_at(index):
    return st.HEADER_BYTES + index * st.UOP_BYTES


def _templates_at(data):
    n_ops = struct.unpack_from("<I", data, 8)[0]
    return st.HEADER_BYTES + (n_ops * st.UOP_BYTES + 63 & -64)


def _team(first, count):
    s = Stream()
    f, c = s.regs(2)
    s.set(f, first)
    s.set(c, count)
    s.issue(SEG_A, team=(f, c))
    s.wait(0)
    return _halt_with(s, 5)


def team_wrap_first():
    return _team(0xFFFFFFFF, 2)


def team_edge_118_2():
    return _team(118, 2)


def team_edge_119_1():
    return _team(119, 1)


def team_whole_grid():
    return _team(0, 119)


def team_last_core():
    return _team(118, 1)


def team_count0_issue():
    return _team(3, 0)


def submit_count0():
    s = Stream()
    c = s.regs(1)[0]
    s.submit(SEG_A, c)
    return _halt_with(s, 0)


def submit_whole_grid():
    s = Stream()
    s.submit(SEG_A, 119, args=[4])
    s.wait(0)
    return _halt_with(s, 0)


def deadline_boundary_ok():
    s = Stream()
    s.issue(SEG_A, team=(0, 2))
    s.wait(0)
    return _halt_with(s, 0)


def deadline_tie_lowest_slot():
    s = Stream()
    s.issue(SEG_A, team=(0, 2), args=[1])
    s.issue(SEG_A, team=(2, 2), args=[2])
    s.wait(0)
    return _halt_with(s, 0)


def deadline_before_cancel():
    s = Stream()
    s.issue(SEG_A, team=(0, 2))
    s.wait(0)
    return _halt_with(s, 0)


def orphan_late():
    s = Stream()
    a = s.regs(1)[0]
    s.issue(SEG_A, team=(0, 2))
    s.issue(SEG_A, team=(2, 2))
    s.mod(a, a, a)
    return _halt_with(s, 0)


def default_deadline():
    s = Stream(deadline_us=0)
    s.issue(SEG_A, team=(0, 2))
    s.wait(0)
    return _halt_with(s, 0)


def issue_team_before_slot_wait():
    s = Stream()
    i, c = s.regs(2)
    top = s.label()
    s.issue(SEG_A, team=(i, 1))
    s.addi(i, i, 1)
    s.jlt(i, s.const(16), top)
    s.issue(SEG_A, team=(0, c))
    return _halt_with(s, 0)


def submit_count_before_queue_full():
    s = Stream()
    i, c = s.regs(2)
    top = s.label()
    s.submit(SEG_A, 119, args=[i])
    s.addi(i, i, 1)
    s.jlt(i, s.const(65), top)
    s.submit(SEG_A, c)
    return _halt_with(s, 0)


def stall_not_a_step():
    s = Stream(max_steps=4)
    s.issue(SEG_A, team=(0, 1))
    s.wait(0)
    return _halt_with(s, 0)


def step_limit_at_halt():
    s = Stream(max_steps=3)
    s.issue(SEG_A, team=(0, 1))
    s.wait(0)
    return _halt_with(s, 0)


def err_off_end_at_step_limit():
    s = Stream(max_steps=2)
    a = s.regs(1)[0]
    s.set(a, 1)
    s.addi(a, a, 1)
    return s


def result_tie_highest_slot():
    s = Stream()
    r = s.regs(1)[0]
    s.issue(SEG_A, team=(0, 2), args=[10], result=True)
    s.issue(SEG_A, team=(2, 3), args=[20], result=True)
    s.wait(0)
    s.result(r)
    s.halt(r)
    return s


def result_ignores_plain_records():
    s = Stream()
    r = s.regs(1)[0]
    s.issue(SEG_A, team=(0, 2), args=[10], result=True)
    s.issue(SEG_A, team=(2, 3), args=[20])
    s.wait(0)
    s.result(r)
    s.halt(r)
    return s


def header_deadline_has_no_default():
    s = Stream(deadline_us=5000)
    s.issue(SEG_A, team=(0, 2))
    s.wait(0)
    return _halt_with(s, 0)


def result_before_any():
    s = Stream()
    r = s.regs(1)[0]
    s.addi(r, r, 1)
    s.result(r)
    s.halt(r)
    return s


def jlt_unsigned():
    s = Stream()
    a, b, out = s.regs(3)
    skip = s.forward()
    s.set(a, 0x80000000)
    s.set(b, 1)
    s.jlt(b, a, skip)
    s.set(out, 100)
    s.bind(skip)
    s.event(out)
    s.halt(out)
    return s


def add_wraps():
    s = Stream()
    a, b = s.regs(2)
    s.set(a, 0xFFFFFFFF)
    s.set(b, 2)
    s.add(a, a, b)
    s.halt(a)
    return s


def sub_wraps():
    s = Stream()
    a, b = s.regs(2)
    s.set(a, 1)
    s.set(b, 2)
    s.sub(a, a, b)
    s.halt(a)
    return s


def mul_overflow():
    s = Stream()
    a, b = s.regs(2)
    s.set(a, 0xFFFFFFFF)
    s.set(b, 0xFFFFFFFF)
    s.mul(a, a, b)
    s.halt(a)
    return s


def halt_waits_for_queue():
    s = Stream()
    for i in range(3):
        s.submit(SEG_A, 119, args=[i])
    return _halt_with(s, 8)


def excl_adjacent():
    s = Stream()
    s.issue(SEG_A, team=(0, 10), args=[1], exclusive=True)
    s.issue(SEG_A, team=(10, 10), args=[2], exclusive=True)
    s.issue(SEG_A, team=(9, 2), args=[3])
    s.wait(0)
    return _halt_with(s, 0)


def patch_later_wins():
    s = Stream()
    x, y = s.regs(2)
    s.set(x, 1)
    s.set(y, 2)
    s.issue(SEG_A, team=(0, 1), args=[x, y])
    s.wait(0)
    s.halt(x)
    data = s.encode()
    return _poke(data, _templates_at(data) + 65, "<B", 0)


def submit_among_issues():
    s = Stream()
    i = s.regs(1)[0]
    s.submit(SEG_B, 1, args=[999])
    top = s.label()
    s.issue(SEG_A, team=(10, 1), args=[i])
    s.addi(i, i, 1)
    s.jlt(i, s.const(60), top)
    s.wait(0)
    return _halt_with(s, 0)


def submit_overtake_limit():
    s = Stream(max_steps=100_000)
    i = s.regs(1)[0]
    s.submit(SEG_A, 1, args=[0])
    s.submit(SEG_B, 119, args=[1])
    top = s.label()
    s.submit(SEG_A, 1, args=[2])
    s.addi(i, i, 1)
    s.jlt(i, s.const(1098), top)
    s.wait(0)
    return _halt_with(s, 0)


def ok_ignored_bytes():
    s = Stream()
    a, b = s.regs(2)
    s.set(a, 3)
    s.issue(SEG_A, team=(0, 2), args=[a, 5])
    s.addi(b, b, 0)
    s.wait(0)
    s.halt(a)
    data = bytearray(s.encode())
    n_ops = len(s.uops())
    data[32:64] = bytes(range(1, 33))  # header padding
    for i in range(n_ops):
        data[_op_at(i) + 8 : _op_at(i) + 16] = b"\xaa" * 8  # imm2 and pad
    data[_op_at(0) + 2 : _op_at(0) + 4] = b"\xc8\xc9"  # SET: b and c unused
    data[_op_at(1) + 1 : _op_at(1) + 4] = b"\xc8\xc9\xca"  # ISSUE: no register
    data[_op_at(3) + 1 : _op_at(3) + 4] = b"\xc8\xc9\xca"  # WAIT: no register
    end = _op_at(n_ops)
    data[end : end + (-n_ops * 16 % 64)] = b"\x77" * (-n_ops * 16 % 64)
    t = _templates_at(bytes(data))
    for offset in (0, 16, 32, 48):  # sequence and landing words
        data[t + offset : t + offset + 4] = b"\x5a\x5a\x5a\x5a"
    data[t + 6 : t + 8] = b"\xff\xff"  # slot bits of hdr
    data[t + 65 : t + 68] = b"\x63" * 3  # patch entries past n_patch
    data[t + 69 : t + 72] = b"\xc8" * 3
    data[t + 75 : t + 80] = b"\x42" * 5  # template padding
    data[t + 80 :] = b"\x33" * (len(data) - t - 80)  # line padding
    return bytes(data)


def _tiny():
    s = Stream()
    a = s.regs(1)[0]
    s.set(a, 1)
    s.issue(SEG_A, team=(0, 1))
    s.halt(a)
    return s.encode()


def err_no_micro_ops():
    return struct.pack("<8I32x", st.MAGIC, st.VERSION, 0, 0, 10, 0, 1, 1)


def ok_fills_region():
    return _tiny()


def err_patch_count():
    data = _tiny()
    return _poke(data, _templates_at(data) + 74, "<B", 5)


def err_patch_register_bad():
    data = _tiny()
    data = _poke(data, _templates_at(data) + 74, "<B", 1)
    return _poke(data, _templates_at(data) + 68, "<B", 16)


def err_segment_zero():
    data = _tiny()
    return _poke(data, _templates_at(data) + 4, "<I", 0)


def err_template_first_user():
    s = Stream()
    s.issue(SEG_A, team=(0, 1))
    s.issue(SEG_A, team=(1, 1))
    s.halt(s.const(0))
    data = _poke(s.encode(), _op_at(2) + 4, "<I", 0)
    return _poke(data, _templates_at(data) + 4, "<I", 0)


def err_bad_version():
    return _poke(_tiny(), 4, "<I", 2)


def err_n_ops_zero():
    return _poke(_tiny(), 8, "<I", 0)


def err_n_ops_too_many():
    return _poke(_tiny(), 8, "<I", 65536)


def err_n_records_too_many():
    return _poke(_tiny(), 12, "<I", 65536)


def err_max_steps_zero():
    return _poke(_tiny(), 16, "<I", 0)


def err_longer_upload():
    return _tiny() + bytes(64)


def err_above_region():
    return _tiny()


def err_submit_bad_register():
    s = Stream()
    c = s.regs(1)[0]
    s.set(c, 3)
    s.submit(SEG_A, c)
    s.halt(c)
    data = s.encode()
    return _poke(data, _op_at(1) + 1, "<B", 16)


def err_two_faults_first_wins():
    data = _tiny()
    data = _poke(data, _op_at(0) + 1, "<B", 200)
    return _poke(data, _op_at(1), "<B", 99)


def err_micro_op_before_template():
    data = _tiny()
    data = _poke(data, _templates_at(data) + 4, "<I", 250)
    return _poke(data, _op_at(2) + 1, "<B", 200)


def err_first_reg_bad():
    data = _tiny()
    return _poke(data, _templates_at(data) + 72, "<B", 16)


def err_count_reg_bad():
    data = _tiny()
    return _poke(data, _templates_at(data) + 73, "<B", 40)


def err_patch_argument_bad():
    data = _tiny()
    data = _poke(data, _templates_at(data) + 74, "<B", 1)
    return _poke(data, _templates_at(data) + 64, "<B", 10)


def err_template_flags():
    data = _tiny()
    return _poke(data, _templates_at(data) + 4, "<I", SEG_A | st.F_QUEUE << 8)


def err_template_index():
    data = _tiny()
    return _poke(data, _op_at(1) + 4, "<I", 1)


def err_bad_register_in_use():
    s = Stream()
    a = s.regs(1)[0]
    s.add(a, a, a)
    s.halt(a)
    return _poke(s.encode(), _op_at(0) + 3, "<B", 16)


MORE = {
    "team_wrap_first": (team_wrap_first, {}, "reason 11: first 0xFFFFFFFF, count 2"),
    "team_edge_118_2": (team_edge_118_2, {}, "reason 11: cores 118 and 119"),
    "team_edge_119_1": (team_edge_119_1, {}, "reason 11: core 119 does not exist"),
    "team_whole_grid": (team_whole_grid, {}, "cores 0 to 118 are a valid team"),
    "team_last_core": (team_last_core, {}, "core 118 alone is a valid team"),
    "team_count0_issue": (team_count0_issue, {}, "reason 11: an empty team"),
    "submit_count0": (submit_count0, {}, "reason 9: a job of 0 cores"),
    "submit_whole_grid": (submit_whole_grid, {}, "a job of 119 cores is placed at 0"),
    "deadline_boundary_ok": (
        deadline_boundary_ok,
        {"delay": 2, "deadline_passes": 3},
        "acknowledged by the sweep of the deadline pass: not late",
    ),
    "deadline_boundary_late": (
        deadline_boundary_ok,
        {"delay": 3, "deadline_passes": 3},
        "acknowledged one pass after the deadline: late",
    ),
    "deadline_tie_lowest_slot": (
        deadline_tie_lowest_slot,
        {"delay": None, "deadline_passes": 4},
        "two late records: the lowest slot's pc is named",
    ),
    "deadline_before_cancel": (
        deadline_before_cancel,
        {"delay": None, "deadline_passes": 3, "cancel_pass": 3},
        "a deadline and a cancel in one pass: the deadline wins",
    ),
    "orphan_late": (
        orphan_late,
        {"delay": {"by_seq": {"2": None}, "rest": 1}, "deadline_passes": 4},
        "after an error a late record posts a timeout event for its slot",
    ),
    "orphan_acked": (
        orphan_late,
        {"delay": 2, "deadline_passes": 4},
        "after an error, records that are acknowledged cause nothing",
    ),
    "default_deadline": (
        default_deadline,
        {"delay": None, "default_deadline_passes": 6},
        "deadline_us 0 uses the default deadline",
    ),
    "issue_team_before_slot_wait": (
        issue_team_before_slot_wait,
        {"delay": None, "max_passes": 2000},
        "a bad team is an error even when all 16 slots are busy",
    ),
    "submit_count_before_queue_full": (
        submit_count_before_queue_full,
        {"delay": None, "max_passes": 2000},
        "a bad count is an error even when the queue is full",
    ),
    "stall_not_a_step": (stall_not_a_step, {"delay": 30}, "waiting costs no steps"),
    "step_limit_at_halt": (step_limit_at_halt, {}, "reason 6 at the HALT"),
    "err_off_end_at_step_limit": (
        err_off_end_at_step_limit,
        {},
        "reason 3 wins over reason 6 at the end of the program",
    ),
    "result_tie_highest_slot": (
        result_tie_highest_slot,
        {"delay": 1},
        "two results swept together: the higher slot is the latest",
    ),
    "result_ignores_plain_records": (
        result_ignores_plain_records,
        {"delay": {"by_seq": {"2": 3}, "rest": 0}},
        "a record without result=True never changes RESULT",
    ),
    "header_deadline_has_no_default": (
        header_deadline_has_no_default,
        {"delay": None, "default_deadline_passes": 6, "max_passes": 40},
        "a header deadline is not replaced by the default (stuck)",
    ),
    "result_before_any": (result_before_any, {}, "RESULT before any result gives 0"),
    "jlt_unsigned": (jlt_unsigned, {}, "JLT compares unsigned"),
    "add_wraps": (add_wraps, {}, "ADD wraps at 32 bits"),
    "sub_wraps": (sub_wraps, {}, "SUB wraps at 32 bits"),
    "mul_overflow": (mul_overflow, {}, "MUL keeps the low 32 bits"),
    "halt_waits_for_queue": (
        halt_waits_for_queue,
        {"delay": 3},
        "HALT waits for jobs still queued",
    ),
    "excl_adjacent": (
        excl_adjacent,
        {"delay": 2},
        "exclusive records on touching teams do not wait",
    ),
    "patch_later_wins": (
        patch_later_wins,
        {},
        "two patches of argument 0: the later one",
    ),
    "submit_among_issues": (
        submit_among_issues,
        {"delay": 1},
        "the queue gets its turn first on alternate passes",
    ),
    "submit_overtake_limit": (
        submit_overtake_limit,
        {"delay": {"by_seq": {"1": 3000}, "rest": 0}},
        "1,024 jobs overtake a 119-core job, then it is waited for",
    ),
    "ok_ignored_bytes": (
        ok_ignored_bytes,
        {},
        "bytes the format does not name are ignored",
    ),
    "err_no_micro_ops": (err_no_micro_ops, {}, "reason 1: a header and nothing else"),
    "ok_fills_region": (
        ok_fills_region,
        {"region_size": 256},
        "an upload as big as the region",
    ),
    "err_patch_count": (err_patch_count, {}, "reason 5: five patches"),
    "err_patch_register_bad": (err_patch_register_bad, {}, "reason 5: register 16"),
    "err_segment_zero": (err_segment_zero, {}, "reason 5: segment 0"),
    "err_template_first_user": (
        err_template_first_user,
        {},
        "reason 5 at the first of two micro-ops using one bad template",
    ),
    "err_bad_version": (err_bad_version, {}, "reason 1: version 2"),
    "err_n_ops_zero": (err_n_ops_zero, {}, "reason 1: no micro-ops"),
    "err_n_ops_too_many": (err_n_ops_too_many, {}, "reason 1: 65,536 micro-ops"),
    "err_n_records_too_many": (
        err_n_records_too_many,
        {},
        "reason 1: 65,536 templates",
    ),
    "err_max_steps_zero": (err_max_steps_zero, {}, "reason 1: max_steps 0"),
    "err_longer_upload": (err_longer_upload, {}, "reason 1: a line too long"),
    "err_above_region": (
        err_above_region,
        {"region_size": 128},
        "reason 1: region of 128 bytes",
    ),
    "err_submit_bad_register": (
        err_submit_bad_register,
        {},
        "reason 4 in SUBMIT's register",
    ),
    "err_two_faults_first_wins": (
        err_two_faults_first_wins,
        {},
        "reason 4 at pc 0 although pc 1 has a bad opcode",
    ),
    "err_micro_op_before_template": (
        err_micro_op_before_template,
        {},
        "a micro-op fault is reported before a template fault",
    ),
    "err_first_reg_bad": (err_first_reg_bad, {}, "reason 5: first_reg 16"),
    "err_count_reg_bad": (err_count_reg_bad, {}, "reason 5: count_reg 40"),
    "err_patch_argument_bad": (err_patch_argument_bad, {}, "reason 5: argument 10"),
    "err_template_flags": (err_template_flags, {}, "reason 5: the queue flag"),
    "err_template_index": (err_template_index, {}, "reason 5 at the ISSUE"),
    "err_bad_register_in_use": (err_bad_register_in_use, {}, "reason 4 in field c"),
}

__all__ = ["MORE", "Reg"]
