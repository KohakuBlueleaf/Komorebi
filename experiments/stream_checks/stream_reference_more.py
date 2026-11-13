"""Reference streams that pin behaviour the first set leaves open.

Each builder gives a `Stream` or upload bytes; `MORE` maps name to (builder,
parameters, what it shows). `stream_reference` merges `MORE` into its table.
"""

import struct

from komorebi.engines import stream as st
from komorebi.engines.stream import Register, Stream

HANDLER_A, HANDLER_B = 1, 7


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
    op_count = struct.unpack_from("<I", data, 8)[0]
    return st.HEADER_BYTES + (op_count * st.UOP_BYTES + 63 & -64)


def _group(first, count):
    s = Stream()
    f, c = s.registers(2)
    s.set(f, first)
    s.set(c, count)
    s.submit(HANDLER_A, group=(f, c))
    s.wait(0)
    return _halt_with(s, 5)


def group_wrap_first():
    return _group(0xFFFFFFFF, 2)


def group_edge_118_2():
    return _group(118, 2)


def group_edge_119_1():
    return _group(119, 1)


def group_whole_grid():
    return _group(0, 119)


def group_last_core():
    return _group(118, 1)


def group_count0_submit():
    return _group(3, 0)


def submit_count0():
    s = Stream()
    c = s.registers(1)[0]
    s.enqueue(HANDLER_A, c)
    return _halt_with(s, 0)


def submit_whole_grid():
    s = Stream()
    s.enqueue(HANDLER_A, 119, args=[4])
    s.wait(0)
    return _halt_with(s, 0)


def deadline_boundary_ok():
    s = Stream()
    s.submit(HANDLER_A, group=(0, 2))
    s.wait(0)
    return _halt_with(s, 0)


def deadline_tie_lowest_slot():
    s = Stream()
    s.submit(HANDLER_A, group=(0, 2), args=[1])
    s.submit(HANDLER_A, group=(2, 2), args=[2])
    s.wait(0)
    return _halt_with(s, 0)


def deadline_before_cancel():
    s = Stream()
    s.submit(HANDLER_A, group=(0, 2))
    s.wait(0)
    return _halt_with(s, 0)


def orphan_late():
    s = Stream()
    a = s.registers(1)[0]
    s.submit(HANDLER_A, group=(0, 2))
    s.submit(HANDLER_A, group=(2, 2))
    s.mod(a, a, a)
    return _halt_with(s, 0)


def default_deadline():
    s = Stream(deadline_microseconds=0)
    s.submit(HANDLER_A, group=(0, 2))
    s.wait(0)
    return _halt_with(s, 0)


def submit_group_before_slot_wait():
    s = Stream()
    i, c = s.registers(2)
    top = s.label()
    s.submit(HANDLER_A, group=(i, 1))
    s.addi(i, i, 1)
    s.jlt(i, s.const(16), top)
    s.submit(HANDLER_A, group=(0, c))
    return _halt_with(s, 0)


def submit_count_before_queue_full():
    s = Stream()
    i, c = s.registers(2)
    top = s.label()
    s.enqueue(HANDLER_A, 119, args=[i])
    s.addi(i, i, 1)
    s.jlt(i, s.const(65), top)
    s.enqueue(HANDLER_A, c)
    return _halt_with(s, 0)


def stall_not_a_step():
    s = Stream(max_steps=4)
    s.submit(HANDLER_A, group=(0, 1))
    s.wait(0)
    return _halt_with(s, 0)


def step_limit_at_halt():
    s = Stream(max_steps=3)
    s.submit(HANDLER_A, group=(0, 1))
    s.wait(0)
    return _halt_with(s, 0)


def error_off_end_at_step_limit():
    s = Stream(max_steps=2)
    a = s.registers(1)[0]
    s.set(a, 1)
    s.addi(a, a, 1)
    return s


def result_tie_highest_slot():
    s = Stream()
    r = s.registers(1)[0]
    s.submit(HANDLER_A, group=(0, 2), args=[10], result=True)
    s.submit(HANDLER_A, group=(2, 3), args=[20], result=True)
    s.wait(0)
    s.result(r)
    s.halt(r)
    return s


def result_ignores_plain_commands():
    s = Stream()
    r = s.registers(1)[0]
    s.submit(HANDLER_A, group=(0, 2), args=[10], result=True)
    s.submit(HANDLER_A, group=(2, 3), args=[20])
    s.wait(0)
    s.result(r)
    s.halt(r)
    return s


def header_deadline_has_no_default():
    s = Stream(deadline_microseconds=5000)
    s.submit(HANDLER_A, group=(0, 2))
    s.wait(0)
    return _halt_with(s, 0)


def result_before_any():
    s = Stream()
    r = s.registers(1)[0]
    s.addi(r, r, 1)
    s.result(r)
    s.halt(r)
    return s


def jlt_unsigned():
    s = Stream()
    a, b, out = s.registers(3)
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
    a, b = s.registers(2)
    s.set(a, 0xFFFFFFFF)
    s.set(b, 2)
    s.add(a, a, b)
    s.halt(a)
    return s


def sub_wraps():
    s = Stream()
    a, b = s.registers(2)
    s.set(a, 1)
    s.set(b, 2)
    s.sub(a, a, b)
    s.halt(a)
    return s


def mul_overflow():
    s = Stream()
    a, b = s.registers(2)
    s.set(a, 0xFFFFFFFF)
    s.set(b, 0xFFFFFFFF)
    s.mul(a, a, b)
    s.halt(a)
    return s


def halt_waits_for_queue():
    s = Stream()
    for i in range(3):
        s.enqueue(HANDLER_A, 119, args=[i])
    return _halt_with(s, 8)


def excl_adjacent():
    s = Stream()
    s.submit(HANDLER_A, group=(0, 10), args=[1], exclusive=True)
    s.submit(HANDLER_A, group=(10, 10), args=[2], exclusive=True)
    s.submit(HANDLER_A, group=(9, 2), args=[3])
    s.wait(0)
    return _halt_with(s, 0)


def patch_later_wins():
    s = Stream()
    x, y = s.registers(2)
    s.set(x, 1)
    s.set(y, 2)
    s.submit(HANDLER_A, group=(0, 1), args=[x, y])
    s.wait(0)
    s.halt(x)
    data = s.encode()
    return _poke(data, _templates_at(data) + 65, "<B", 0)


def enqueue_among_submits():
    s = Stream()
    i = s.registers(1)[0]
    s.enqueue(HANDLER_B, 1, args=[999])
    top = s.label()
    s.submit(HANDLER_A, group=(10, 1), args=[i])
    s.addi(i, i, 1)
    s.jlt(i, s.const(60), top)
    s.wait(0)
    return _halt_with(s, 0)


def enqueue_overtake_limit():
    s = Stream(max_steps=100_000)
    i = s.registers(1)[0]
    s.enqueue(HANDLER_A, 1, args=[0])
    s.enqueue(HANDLER_B, 119, args=[1])
    top = s.label()
    s.enqueue(HANDLER_A, 1, args=[2])
    s.addi(i, i, 1)
    s.jlt(i, s.const(1098), top)
    s.wait(0)
    return _halt_with(s, 0)


def ok_ignored_bytes():
    s = Stream()
    a, b = s.registers(2)
    s.set(a, 3)
    s.submit(HANDLER_A, group=(0, 2), args=[a, 5])
    s.addi(b, b, 0)
    s.wait(0)
    s.halt(a)
    data = bytearray(s.encode())
    op_count = len(s.uops())
    data[32:64] = bytes(range(1, 33))  # header padding
    for i in range(op_count):
        data[_op_at(i) + 8 : _op_at(i) + 16] = b"\xaa" * 8  # imm2 and pad
    data[_op_at(0) + 2 : _op_at(0) + 4] = b"\xc8\xc9"  # SET: b and c unused
    data[_op_at(1) + 1 : _op_at(1) + 4] = b"\xc8\xc9\xca"  # SUBMIT: no register
    data[_op_at(3) + 1 : _op_at(3) + 4] = b"\xc8\xc9\xca"  # WAIT: no register
    end = _op_at(op_count)
    data[end : end + (-op_count * 16 % 64)] = b"\x77" * (-op_count * 16 % 64)
    t = _templates_at(bytes(data))
    for offset in (0, 16, 32, 48):  # sequence and landing words
        data[t + offset : t + offset + 4] = b"\x5a\x5a\x5a\x5a"
    data[t + 6 : t + 8] = b"\xff\xff"  # slot bits of hdr
    data[t + 65 : t + 68] = b"\x63" * 3  # patch entries past patch_count
    data[t + 69 : t + 72] = b"\xc8" * 3
    data[t + 75 : t + 80] = b"\x42" * 5  # template padding
    data[t + 80 :] = b"\x33" * (len(data) - t - 80)  # line padding
    return bytes(data)


def _tiny():
    s = Stream()
    a = s.registers(1)[0]
    s.set(a, 1)
    s.submit(HANDLER_A, group=(0, 1))
    s.halt(a)
    return s.encode()


def error_no_micro_ops():
    return struct.pack("<8I32x", st.MAGIC, st.VERSION, 0, 0, 10, 0, 1, 1)


def ok_fills_region():
    return _tiny()


def error_patch_count():
    data = _tiny()
    return _poke(data, _templates_at(data) + 74, "<B", 5)


def error_patch_register_bad():
    data = _tiny()
    data = _poke(data, _templates_at(data) + 74, "<B", 1)
    return _poke(data, _templates_at(data) + 68, "<B", 16)


def error_handler_zero():
    data = _tiny()
    return _poke(data, _templates_at(data) + 4, "<I", 0)


def error_template_first_user():
    s = Stream()
    s.submit(HANDLER_A, group=(0, 1))
    s.submit(HANDLER_A, group=(1, 1))
    s.halt(s.const(0))
    data = _poke(s.encode(), _op_at(2) + 4, "<I", 0)
    return _poke(data, _templates_at(data) + 4, "<I", 0)


def error_bad_version():
    return _poke(_tiny(), 4, "<I", 2)


def error_op_count_zero():
    return _poke(_tiny(), 8, "<I", 0)


def error_op_count_too_many():
    return _poke(_tiny(), 8, "<I", 65536)


def error_command_count_too_many():
    return _poke(_tiny(), 12, "<I", 65536)


def error_max_steps_zero():
    return _poke(_tiny(), 16, "<I", 0)


def error_longer_upload():
    return _tiny() + bytes(64)


def error_above_region():
    return _tiny()


def error_enqueue_bad_register():
    s = Stream()
    c = s.registers(1)[0]
    s.set(c, 3)
    s.enqueue(HANDLER_A, c)
    s.halt(c)
    data = s.encode()
    return _poke(data, _op_at(1) + 1, "<B", 16)


def error_two_faults_first_wins():
    data = _tiny()
    data = _poke(data, _op_at(0) + 1, "<B", 200)
    return _poke(data, _op_at(1), "<B", 99)


def error_micro_op_before_template():
    data = _tiny()
    data = _poke(data, _templates_at(data) + 4, "<I", 250)
    return _poke(data, _op_at(2) + 1, "<B", 200)


def error_first_register_bad():
    data = _tiny()
    return _poke(data, _templates_at(data) + 72, "<B", 16)


def error_count_register_bad():
    data = _tiny()
    return _poke(data, _templates_at(data) + 73, "<B", 40)


def error_patch_argument_bad():
    data = _tiny()
    data = _poke(data, _templates_at(data) + 74, "<B", 1)
    return _poke(data, _templates_at(data) + 64, "<B", 10)


def error_template_flags():
    data = _tiny()
    return _poke(data, _templates_at(data) + 4, "<I", HANDLER_A | st.FLAG_QUEUE << 8)


def error_template_index():
    data = _tiny()
    return _poke(data, _op_at(1) + 4, "<I", 1)


def error_bad_register_in_use():
    s = Stream()
    a = s.registers(1)[0]
    s.add(a, a, a)
    s.halt(a)
    return _poke(s.encode(), _op_at(0) + 3, "<B", 16)


MORE = {
    "group_wrap_first": (group_wrap_first, {}, "reason 11: first 0xFFFFFFFF, count 2"),
    "group_edge_118_2": (group_edge_118_2, {}, "reason 11: cores 118 and 119"),
    "group_edge_119_1": (group_edge_119_1, {}, "reason 11: core 119 does not exist"),
    "group_whole_grid": (group_whole_grid, {}, "cores 0 to 118 are a valid group"),
    "group_last_core": (group_last_core, {}, "core 118 alone is a valid group"),
    "group_count0_submit": (group_count0_submit, {}, "reason 11: an empty group"),
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
        "two late commands: the lowest slot's pc is named",
    ),
    "deadline_before_cancel": (
        deadline_before_cancel,
        {"delay": None, "deadline_passes": 3, "cancel_pass": 3},
        "a deadline and a cancel in one pass: the deadline wins",
    ),
    "orphan_late": (
        orphan_late,
        {"delay": {"by_sequence": {"2": None}, "rest": 1}, "deadline_passes": 4},
        "after an error a late command posts a timeout event for its slot",
    ),
    "orphan_acked": (
        orphan_late,
        {"delay": 2, "deadline_passes": 4},
        "after an error, commands that are acknowledged cause nothing",
    ),
    "default_deadline": (
        default_deadline,
        {"delay": None, "default_deadline_passes": 6},
        "deadline_microseconds 0 uses the default deadline",
    ),
    "submit_group_before_slot_wait": (
        submit_group_before_slot_wait,
        {"delay": None, "max_passes": 2000},
        "a bad group is an error even when all 16 slots are busy",
    ),
    "submit_count_before_queue_full": (
        submit_count_before_queue_full,
        {"delay": None, "max_passes": 2000},
        "a bad count is an error even when the queue is full",
    ),
    "stall_not_a_step": (stall_not_a_step, {"delay": 30}, "waiting costs no steps"),
    "step_limit_at_halt": (step_limit_at_halt, {}, "reason 6 at the HALT"),
    "error_off_end_at_step_limit": (
        error_off_end_at_step_limit,
        {},
        "reason 3 wins over reason 6 at the end of the program",
    ),
    "result_tie_highest_slot": (
        result_tie_highest_slot,
        {"delay": 1},
        "two results swept together: the higher slot is the latest",
    ),
    "result_ignores_plain_commands": (
        result_ignores_plain_commands,
        {"delay": {"by_sequence": {"2": 3}, "rest": 0}},
        "a command without result=True never changes RESULT",
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
        "exclusive commands on touching groups do not wait",
    ),
    "patch_later_wins": (
        patch_later_wins,
        {},
        "two patches of argument 0: the later one",
    ),
    "enqueue_among_submits": (
        enqueue_among_submits,
        {"delay": 1},
        "the queue gets its turn first on alternate passes",
    ),
    "enqueue_overtake_limit": (
        enqueue_overtake_limit,
        {"delay": {"by_sequence": {"1": 3000}, "rest": 0}},
        "1,024 jobs overtake a 119-core job, then it is waited for",
    ),
    "ok_ignored_bytes": (
        ok_ignored_bytes,
        {},
        "bytes the format does not name are ignored",
    ),
    "error_no_micro_ops": (
        error_no_micro_ops,
        {},
        "reason 1: a header and nothing else",
    ),
    "ok_fills_region": (
        ok_fills_region,
        {"region_size": 256},
        "an upload as big as the region",
    ),
    "error_patch_count": (error_patch_count, {}, "reason 5: five patches"),
    "error_patch_register_bad": (error_patch_register_bad, {}, "reason 5: register 16"),
    "error_handler_zero": (error_handler_zero, {}, "reason 5: handler 0"),
    "error_template_first_user": (
        error_template_first_user,
        {},
        "reason 5 at the first of two micro-ops using one bad template",
    ),
    "error_bad_version": (error_bad_version, {}, "reason 1: version 2"),
    "error_op_count_zero": (error_op_count_zero, {}, "reason 1: no micro-ops"),
    "error_op_count_too_many": (
        error_op_count_too_many,
        {},
        "reason 1: 65,536 micro-ops",
    ),
    "error_command_count_too_many": (
        error_command_count_too_many,
        {},
        "reason 1: 65,536 templates",
    ),
    "error_max_steps_zero": (error_max_steps_zero, {}, "reason 1: max_steps 0"),
    "error_longer_upload": (error_longer_upload, {}, "reason 1: a line too long"),
    "error_above_region": (
        error_above_region,
        {"region_size": 128},
        "reason 1: region of 128 bytes",
    ),
    "error_enqueue_bad_register": (
        error_enqueue_bad_register,
        {},
        "reason 4 in ENQUEUE's register",
    ),
    "error_two_faults_first_wins": (
        error_two_faults_first_wins,
        {},
        "reason 4 at pc 0 although pc 1 has a bad opcode",
    ),
    "error_micro_op_before_template": (
        error_micro_op_before_template,
        {},
        "a micro-op fault is reported before a template fault",
    ),
    "error_first_register_bad": (
        error_first_register_bad,
        {},
        "reason 5: first_register 16",
    ),
    "error_count_register_bad": (
        error_count_register_bad,
        {},
        "reason 5: count_register 40",
    ),
    "error_patch_argument_bad": (error_patch_argument_bad, {}, "reason 5: argument 10"),
    "error_template_flags": (error_template_flags, {}, "reason 5: the queue flag"),
    "error_template_index": (error_template_index, {}, "reason 5 at the SUBMIT"),
    "error_bad_register_in_use": (error_bad_register_in_use, {}, "reason 4 in field c"),
}

__all__ = ["MORE", "Register"]
