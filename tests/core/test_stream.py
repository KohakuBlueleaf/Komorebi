"""Micro-op streams: header constants, encoding, check and the simulator's semantics."""

import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from komorebi.engines import stream as stream_module
from komorebi.engines.stream import Register, Stream, check, simulate

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "src" / "komorebi_native" / "include"
ABI = INCLUDE / "komorebi" / "abi"
DEFINE = re.compile(r"^#define\s+(KM_\w+)\s+(0x[0-9A-Fa-f]+|\d+)u?\b", re.M)


def header_constants():
    found = {}
    for name in ("km_abi.h", "km_stream.h"):
        found.update(
            {
                constant: int(text, 0)
                for constant, text in DEFINE.findall((ABI / name).read_text())
            }
        )
    return found


HEADER_CONSTANTS = header_constants()


def binary(operation, x, y, immediate=False):
    """The value `operation` leaves on x and y (y an immediate if immediate)."""
    stream = Stream()
    register_a, register_b, register_c = stream.registers(3)
    stream.set(register_a, x)
    if immediate:
        getattr(stream, operation)(register_c, register_a, y)
    else:
        stream.set(register_b, y)
        getattr(stream, operation)(register_c, register_a, register_b)
    stream.halt(register_c)
    return stream.simulate().halt


def pcs(stream, opcode):
    """Program counters of the micro-ops with opcode `opcode`, prologue included."""
    return [pc for pc, micro_op in enumerate(stream.uops()) if micro_op.op == opcode]


def run(build, **options):
    stream = Stream(max_steps=100_000)
    build(stream)
    return stream.simulate(**options)


def xorshift_by_hand(x):
    x = (x ^ (x << 13)) & 0xFFFFFFFF
    x = x ^ (x >> 17)
    x = (x ^ (x << 5)) & 0xFFFFFFFF
    return x


# header constants


PAIRS = {
    "MAGIC": "KM_STREAM_MAGIC",
    "VERSION": "KM_STREAM_VERSION",
    "REGISTERS": "KM_STREAM_REGISTERS",
    "SEED_REGISTER": "KM_STREAM_SEED_REGISTER",
    "MAX_OPS": "KM_STREAM_MAX_OPS",
    "MAX_ACTIVE": "KM_STREAM_MAX_ACTIVE",
    "LINE": "KM_STREAM_LINE",
    "TEMPLATE_NONE": "KM_TEMPLATE_NONE",
    "PATCHES": "KM_TEMPLATE_PATCHES",
    "ARGS": "KM_TEMPLATE_ARGS",
    "STEP_OPS": "KM_STEP_OPS",
    "QUEUE_JOBS": "KM_QUEUE_JOBS",
    "QUEUE_LOOK": "KM_QUEUE_LOOK",
    "QUEUE_PASSES": "KM_QUEUE_PASSES",
    "RING_SLOTS": "KM_RING_SLOTS",
    "HANDLER_USER_MAX": "KM_HANDLER_USER_MAX",
    "FLAG_EXCLUSIVE": "KM_FLAG_EXCLUSIVE",
    "FLAG_RESULT": "KM_FLAG_RESULT",
    "FLAG_QUEUE": "KM_FLAG_QUEUE",
    "EVENT_TIMEOUT": "KM_EVENT_TIMEOUT",
    "EVENT_STREAM": "KM_EVENT_STREAM",
    "EVENT_STREAM_END": "KM_EVENT_STREAM_END",
    "EVENT_STREAM_ERROR": "KM_EVENT_STREAM_ERROR",
}


def test_constants_opcodes_and_error_reasons_match_the_header():
    for name, header_name in PAIRS.items():
        assert getattr(stream_module, name) == HEADER_CONSTANTS[header_name], name
    for code, (name, _, _) in stream_module.OPS.items():
        assert HEADER_CONSTANTS[f"KM_OP_{name}"] == code
    assert HEADER_CONSTANTS["KM_OP_INVALID"] == 0 and len(stream_module.OPS) == 21
    ops = {constant for constant in HEADER_CONSTANTS if constant.startswith("KM_OP_")}
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
    reasons += ["NEVER_FITS", "CANCELLED", "GROUP"]
    for number, name in enumerate(reasons, 1):
        assert (
            HEADER_CONSTANTS[f"KM_STREAM_ERROR_{name}"]
            == number
            == getattr(stream_module, f"ERROR_{name}")
        )
    assert (
        HEADER_CONSTANTS["KM_HANDLER_RUN"] == 251
        and HEADER_CONSTANTS["KM_HANDLER_CANCEL"] == 250
    )


def test_layout_matches_the_c_structs(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("needs a C compiler")
    source_file = tmp_path / "layout.c"
    source_file.write_text(
        '#include <stdio.h>\n#include "komorebi/abi/km_stream.h"\n'
        'int main(void) { printf("%zu %zu %zu %zu %zu %zu %zu\\n", sizeof(km_stream_t),'
        " sizeof(km_uop_t), sizeof(km_template_t), offsetof(km_stream_t, seed),"
        " offsetof(km_uop_t, immediate), offsetof(km_template_t, arg),"
        " offsetof(km_template_t, first_register)); return 0; }\n"
    )
    exe = tmp_path / "layout"
    command_line = [
        compiler,
        "-std=c11",
        "-Wall",
        "-Werror",
        "-I",
        str(INCLUDE),
        str(source_file),
        "-o",
        exe,
    ]
    subprocess.run(command_line, check=True)
    out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    sizes = tuple(map(int, out.split()))
    assert sizes == (64, 16, 80, 28, 4, 64, 72)
    assert (
        stream_module.HEADER.size,
        stream_module.UOP.size,
        stream_module.TAIL.size,
    ) == (64, 16, 16)
    assert 64 + stream_module.TAIL.size == stream_module.TEMPLATE_BYTES


# builder


def test_the_builder_rejects_what_it_cannot_encode_in_registers():
    stream = Stream()
    stream.registers(14)
    with pytest.raises(ValueError, match=r"2 requested, 1 free of 15"):
        stream.registers(2)
    assert stream.registers(1) == [Register(14)]
    with pytest.raises(ValueError, match="out of registers: 1 requested, 0 free"):
        stream.const(5)
    with pytest.raises(ValueError):
        Register(16)
    register = Stream().registers(1)[0]
    with pytest.raises(TypeError):
        Stream().add(register, register, 3)
    with pytest.raises(TypeError):
        Stream().jmp(5)
    with pytest.raises(ValueError):
        Stream().set(register, 1 << 32)


def test_equal_constants_use_one_register():
    stream = Stream()
    assert stream.const(7) == stream.const(7) != stream.const(8)
    assert stream.const(-1) == stream.const(0xFFFFFFFF)
    assert [micro_op.immediate for micro_op in stream.uops()] == [7, 8, 0xFFFFFFFF]
    assert all(micro_op.op == stream_module.OP_SET for micro_op in stream.uops())


def test_labels_resolve_after_the_prologue_and_must_be_bound():
    stream = Stream()
    register = stream.registers(1)[0]
    top, end = stream.label(), stream.forward()
    stream.jlt(register, stream.const(5), end)
    stream.jmp(top)
    stream.bind(end)
    stream.halt(register)
    ops = stream.uops()
    assert (
        ops[0].op == stream_module.OP_SET
        and ops[1].immediate == 3
        and ops[2].immediate == 1
    )
    with pytest.raises(ValueError, match="already bound"):
        stream.bind(end)
    stream = Stream()
    stream.jmp(stream.forward())
    with pytest.raises(ValueError, match="never bound"):
        stream.encode()
    bound_stream = Stream()
    end = bound_stream.forward()
    bound_stream.jmp(end)
    bound_stream.bind(end)
    with pytest.raises(ValueError, match="past the last"):
        bound_stream.encode()


def test_template_arguments_and_limits():
    stream = Stream()
    register_a, register_b = stream.registers(2)
    stream.submit(
        3,
        group=(register_a, 4),
        args=[1, register_a, 2, register_b],
        exclusive=True,
        result=True,
    )
    template = stream.templates()[0]
    assert (template.handler, template.flags) == (
        3,
        stream_module.FLAG_EXCLUSIVE | stream_module.FLAG_RESULT,
    )
    assert (
        template.rec[2] == 4 << 16
        and template.first_register == 0
        and template.count_register == stream_module.TEMPLATE_NONE
    )
    assert (
        template.patch_count == 2
        and template.arg[:2] == (1, 3)
        and template.register[:2] == (0, 1)
    )
    assert (
        template.rec[stream_module.ARG_WORD[0]] == 1
        and template.rec[stream_module.ARG_WORD[2]] == 2
    )
    with pytest.raises(ValueError, match="at most 4"):
        stream.submit(1, group=(0, 1), args=[register_a] * 5)
    with pytest.raises(ValueError, match="11 arguments"):
        stream.submit(1, group=(0, 1), args=[0] * 11)
    for bad in (0, 240, 250):
        with pytest.raises(ValueError, match="program handler"):
            stream.submit(bad, group=(0, 1))
    with pytest.raises(ValueError, match="16 bits"):
        stream.submit(1, group=(0, 70000))


# encoding


def test_encoding_of_a_small_stream_by_hand():
    stream = Stream(deadline_microseconds=7, max_steps=9, seed=5, req=3)
    register = stream.registers(1)[0]
    stream.set(register, 0x01020304)
    stream.submit(2, group=(1, 2), args=[0xAA, register])
    stream.halt(register)
    data = stream.encode()
    assert len(data) == 64 + 64 + 128
    assert data[:32] == struct.pack("<8I", 0x4B4D5354, 1, 3, 1, 9, 7, 3, 5)
    assert data[32:64] == bytes(32)
    assert data[64:80] == bytes([1, 0, 0, 0, 4, 3, 2, 1]) + bytes(8)
    assert data[80:96] == bytes([32, 0, 0, 0, 0, 0, 0, 0]) + bytes(8)
    assert data[96:112] == bytes([20, 0, 0, 0]) + bytes(12)
    assert data[112:128] == bytes(16)
    command = struct.unpack_from("<16I", data, 128)
    assert command[1] == 2 and command[2] == 1 | 2 << 16 and command[3] == 0xAA
    assert command[0] == command[4] == command[8] == command[12] == 0
    assert data[192:203] == bytes([1, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 1])
    assert data[203:] == bytes(256 - 203)


def test_round_trip_of_every_micro_op():
    stream = Stream(deadline_microseconds=5, max_steps=77, seed=0xABCDEF, req=42)
    register_a, register_b, register_c = stream.registers(3)
    top, end = stream.label(), stream.forward()
    stream.set(register_a, 3)
    stream.mov(register_b, register_a)
    stream.add(register_c, register_a, register_b)
    stream.addi(register_c, register_c, 0xFFFFFFFF)
    stream.sub(register_c, register_c, register_a)
    stream.mul(register_c, register_c, register_b)
    stream.and_(register_c, register_c, stream.const(0xFF))
    stream.shr(register_c, register_c, 3)
    stream.mod(register_c, register_c, stream.const(9))
    stream.rand(register_a)
    stream.jlt(register_a, register_b, top)
    stream.jeq(register_a, register_b, end)
    stream.jnz(register_a, end)
    stream.time(register_b)
    stream.event(register_b)
    stream.submit(
        4, group=(1, register_c), args=[register_a, 5, register_b], exclusive=True
    )
    stream.enqueue(
        5,
        count=register_c,
        args=[register_b, register_c, register_a, 9, 1, 2, 3, 4, 5, 6],
        result=True,
    )
    stream.wait(2)
    stream.result(register_c)
    stream.jmp(top)
    stream.bind(end)
    stream.halt(register_c)
    data = stream.encode()
    decoded = Stream.decode(data)
    assert decoded.uops() == stream.uops() and decoded.templates() == stream.templates()
    assert (
        decoded.deadline_microseconds,
        decoded.max_steps,
        decoded.seed,
        decoded.req,
    ) == (
        5,
        77,
        0xABCDEF,
        42,
    )
    assert decoded.encode() == data
    assert {micro_op.op for micro_op in decoded.uops()} == set(stream_module.OPS)


# micro-op semantics


def test_arithmetic_wraps_at_32_bits_and_shr_takes_the_count_mod_32():
    for operation, x, y, want in [
        ("add", 0xFFFFFFFF, 2, 1),
        ("sub", 0, 1, 0xFFFFFFFF),
        ("mul", 0xFFFFFFFF, 0xFFFFFFFF, 1),
        ("and_", 0xF0F0, 0x0FF0, 0x00F0),
        ("mod", 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF),
    ]:
        assert binary(operation, x, y) == want, operation
    for operation, x, immediate, want in [
        ("addi", 0xFFFFFFFF, 1, 0),
        ("addi", 5, -2, 3),
        ("shr", 0x80000000, 31, 1),
        ("shr", 0xFFFFFFFF, 36, 0x0FFFFFFF),
    ]:
        assert binary(operation, x, immediate, immediate=True) == want, operation


def test_rand_follows_the_xorshift_written_out():
    stream = Stream(seed=1)
    register = stream.registers(1)[0]
    for _ in range(6):
        stream.rand(register)
        stream.event(register)
    stream.halt(register)
    done = stream.simulate()
    x, want = 1, []
    for _ in range(6):
        x = xorshift_by_hand(x)
        want.append(x & 0xFFFFFF)
    assert [event.auxiliary for event in done.events[:6]] == want
    assert done.halt == x
    assert want[0] == 270369  # 1 ^ 1 << 13, then the two shifts


def test_jumps_compare_unsigned():
    for operation, x, y, taken in [
        ("jlt", 2, 2, False),
        ("jlt", 1, 0xFFFFFFFF, True),
        ("jeq", 4, 4, True),
        ("jeq", 4, 5, False),
        ("jnz", 0, None, False),
        ("jnz", 0xFFFFFFFF, None, True),
    ]:
        stream = Stream()
        register_a, register_b, out = stream.registers(3)
        skip = stream.forward()
        stream.set(register_a, x)
        stream.set(register_b, y or 0)
        if operation == "jnz":
            stream.jnz(register_a, skip)
        else:
            getattr(stream, operation)(register_a, register_b, skip)
        stream.set(out, 1)
        stream.bind(skip)
        stream.halt(out)
        assert stream.simulate().halt == (0 if taken else 1), (operation, x, y)


def test_halt_status_and_event_keep_the_low_24_bits():
    stream = Stream()
    register = stream.registers(1)[0]
    stream.set(register, 0xAB123456)
    stream.event(register)
    stream.halt(register)
    done = stream.simulate()
    assert done.halt == 0xAB123456
    assert [(event.kind, event.auxiliary) for event in done.events] == [
        (4, 0x123456),
        (5, 0x123456),
    ]
    assert all(event.req == 1 for event in done.events)


# steps, passes and errors


def test_a_pass_runs_at_most_64_micro_ops():
    for micro_op_count, passes in ((63, 1), (64, 2), (127, 2), (128, 3)):
        stream = Stream(max_steps=1000)
        register = stream.registers(1)[0]
        for _ in range(micro_op_count):
            stream.addi(register, register, 1)
        stream.halt(register)
        done = stream.simulate()
        assert (done.halt, done.passes) == (micro_op_count, passes), micro_op_count


def test_step_limit_counts_executed_micro_ops():
    def ops(limit):
        stream = Stream(max_steps=limit)
        register = stream.registers(1)[0]
        stream.addi(register, register, 1)
        stream.addi(register, register, 1)
        stream.halt(register)
        return stream.simulate()

    assert ops(3).status == "ended"
    done = ops(2)
    assert done.error == (stream_module.ERROR_STEPS, 2) and done.halt is None
    assert ops(1).error == (stream_module.ERROR_STEPS, 1)


def test_a_stalled_micro_op_does_not_count_as_a_step():
    stream = Stream(max_steps=5)
    stream.submit(1, group=(0, 1))
    stream.wait(0)
    stream.halt(stream.const(0))
    assert stream.simulate(delay=50).status == "ended"


def test_a_job_that_can_never_fit_is_an_error():
    for count in [0, 120]:
        stream = Stream()
        stream.set(stream.registers(1)[0], 0)
        stream.enqueue(1, count)
        stream.halt(stream.const(0))
        done = stream.simulate()
        assert done.error == (
            stream_module.ERROR_NEVER_FITS,
            pcs(stream, stream_module.OP_ENQUEUE)[0],
        )
        assert done.commands == []
        assert done.events[-1].kind == stream_module.EVENT_STREAM_ERROR
        assert done.events[-1].auxiliary == 9 | done.error.pc << 8


def test_mod_by_zero_is_an_error_at_its_pc():
    stream = Stream()
    register_a, register_b = stream.registers(2)
    stream.set(register_a, 5)
    stream.mod(register_b, register_a, Register(5))
    stream.halt(register_a)
    done = stream.simulate()
    assert done.error == (stream_module.ERROR_MOD_ZERO, 1) and done.passes == 1
    assert [event.kind for event in done.events] == [stream_module.EVENT_STREAM_ERROR]


def test_running_off_the_end_is_a_jump_error_at_op_count():
    stream = Stream()
    register = stream.registers(1)[0]
    stream.set(register, 1)
    done = stream.simulate()
    assert done.error == (stream_module.ERROR_JUMP, 1)


def test_a_command_deadline_ends_the_stream_and_stops_the_grid():
    stream = Stream()
    stream.submit(1, group=(0, 2))
    stream.submit(1, group=(2, 2))
    stream.wait(0)
    stream.halt(stream.const(0))
    done = stream.simulate(
        delay=lambda command: None if command.sequence == 2 else 0, deadline_passes=3
    )
    assert (
        done.error
        == (stream_module.ERROR_DEADLINE, pcs(stream, stream_module.OP_SUBMIT)[1])
        and done.stopped
    )
    assert done.passes == 4 and len(done.commands) == 2
    on_time = stream.simulate(delay=2, deadline_passes=3)
    assert on_time.status == "ended" and not on_time.stopped
    late = stream.simulate(delay=3, deadline_passes=3)
    assert late.error == (
        stream_module.ERROR_DEADLINE,
        pcs(stream, stream_module.OP_SUBMIT)[0],
    )


def test_a_stream_that_never_finishes_reports_stuck():
    stream = Stream()
    stream.submit(1, group=(0, 1))
    stream.wait(0)
    stream.halt(stream.const(0))
    done = stream.simulate(delay=None, max_passes=50)
    assert (done.status, done.passes, done.error) == ("stuck", 50, None)


def test_cancel_ends_the_stream_at_the_pc_it_has_reached():
    for cancel in [0, 4]:
        stream = Stream(max_steps=10_000)
        top = stream.label()
        stream.submit(1, group=(0, 1))
        stream.wait(0)
        stream.jmp(top)
        done = stream.simulate(delay=1, cancel_pass=cancel)
        pcs = {0: 0, 1: 1, 4: 1}[cancel]
        assert done.error == (stream_module.ERROR_CANCELLED, pcs)
        assert done.events[-1].auxiliary == 10 | pcs << 8 and done.passes == cancel + 1


def test_waits_and_the_completion_order():
    stream = Stream()
    for i in range(4):
        stream.submit(1, group=(i, 1), args=[i])
    stream.wait(2)
    stream.event(stream.const(1))
    stream.wait(0)
    stream.event(stream.const(2))
    stream.halt(stream.const(0))
    done = stream.simulate(delay=lambda command: 3 - command.args[0])
    passes = {
        x.auxiliary: x.pass_index
        for x in done.events
        if x.kind == stream_module.EVENT_STREAM
    }
    assert passes == {1: 2, 2: 4}
    done = stream.simulate(delay=0)
    assert {
        x.auxiliary: x.pass_index
        for x in done.events
        if x.kind == stream_module.EVENT_STREAM
    } == {
        1: 1,
        2: 1,
    }


def test_halt_waits_for_every_command():
    stream = Stream()
    stream.submit(1, group=(0, 1))
    stream.halt(stream.const(9))
    done = stream.simulate(delay=6)
    assert done.passes == 8 and done.halt == 9


def test_result_is_the_latest_completed_result_command():
    stream = Stream()
    first_result, second_result, third_result = stream.registers(3)
    stream.result(first_result)
    stream.submit(1, group=(0, 2), args=[5], result=True)
    stream.submit(1, group=(2, 2), args=[7])
    stream.wait(0)
    stream.result(second_result)
    stream.submit(1, group=(0, 3), args=[1], result=True)
    stream.submit(1, group=(3, 3), args=[2], result=True)
    stream.wait(0)
    stream.result(third_result)
    stream.event(first_result)
    stream.event(second_result)
    stream.event(third_result)
    stream.halt(third_result)

    def result_function(command):
        return command.args[0] * 100 + command.count

    done = stream.simulate(result=result_function, delay=0)
    # commands 3 and 4 are swept together; the higher slot is swept last
    assert [event.auxiliary for event in done.events[:3]] == [0, 502, 203]


def test_result_order_follows_completion_not_submit():
    stream = Stream()
    register = stream.registers(1)[0]
    stream.submit(1, group=(0, 1), args=[1], result=True)
    stream.submit(1, group=(1, 1), args=[2], result=True)
    stream.wait(0)
    stream.result(register)
    stream.halt(register)

    def late_first(command):
        return 5 if command.args[0] == 1 else 1

    assert stream.simulate(delay=late_first, result=lambda x: x.args[0]).halt == 1
    assert (
        stream.simulate(delay=lambda x: x.args[0], result=lambda x: x.args[0]).halt == 2
    )


def test_submitted_commands_carry_the_template_and_the_patches():
    stream = Stream()
    register_a, register_b, first_register = stream.registers(3)
    stream.set(register_a, 11)
    stream.set(register_b, 22)
    stream.set(first_register, 3)
    stream.submit(
        7,
        group=(first_register, 2),
        args=[1, register_a, 2, register_b, 5, 6, 7, 8, 9, 10],
        exclusive=True,
    )
    stream.wait(0)
    stream.halt(register_a)
    (command,) = stream.simulate().commands
    assert (command.handler, command.first, command.count, command.flags) == (
        7,
        3,
        2,
        stream_module.FLAG_EXCLUSIVE,
    )
    assert command.args == (1, 11, 2, 22, 5, 6, 7, 8, 9, 10)
    assert (
        command.sequence,
        command.slot,
        command.pc,
        command.queued,
        command.pass_index,
    ) == (1, 0, 3, False, 0)


def test_a_later_patch_of_the_same_argument_wins():
    stream = Stream()
    x, y = stream.registers(2)
    stream.set(x, 1)
    stream.set(y, 2)
    stream.submit(1, group=(0, 1), args=[x, y])
    stream.halt(x)
    data = bytearray(stream.encode())
    base = (
        stream_module.HEADER_BYTES + 64 + 64
    )  # one line of micro-ops, then the command
    poke(data, base + 1, "<B", 0)  # both patches name argument 0
    assert simulate(bytes(data)).commands[0].args[0] == 2


# slots, allocator, queue


def test_slots_are_the_lowest_free_one():
    stream = Stream()
    for i in range(3):
        stream.submit(1, group=(i, 1), args=[i])
    stream.wait(0)
    stream.submit(1, group=(0, 1), args=[9])
    stream.wait(0)
    stream.halt(stream.const(0))
    done = stream.simulate(delay=lambda command: 4 if command.args[0] == 0 else 0)
    assert [command.slot for command in done.commands] == [0, 1, 2, 0]
    second_stream = Stream()
    for i in range(3):
        second_stream.submit(1, group=(i, 1), args=[i])
    second_stream.wait(2)
    second_stream.submit(1, group=(5, 1), args=[7])
    second_stream.wait(0)
    second_stream.halt(second_stream.const(0))
    done = second_stream.simulate(
        delay=lambda command: 4 if command.args[0] == 0 else 0
    )
    assert [command.slot for command in done.commands] == [0, 1, 2, 1]


def test_submit_waits_while_all_16_slots_are_busy():
    stream = Stream()
    top = stream.label()
    stream.submit(1, group=(0, 1))
    stream.addi(stream.registers(1)[0], Register(0), 1)
    stream.jmp(top)
    done = stream.simulate(delay=None, max_passes=5)
    assert len(done.commands) == 16 and done.status == "stuck"
    assert [command.slot for command in done.commands] == list(range(16))


def test_alloc_is_the_lowest_first_that_overlaps_nothing():
    for busy, count, core_count, want in [
        ([], 5, 119, 0),
        ([], 120, 119, None),
        ([(2, 3)], 2, 119, 0),
        ([(0, 2), (4, 2)], 3, 119, 6),
        ([(0, 100)], 19, 119, 100),
        ([(10, 5), (0, 10)], 1, 119, 15),
    ]:
        assert stream_module.alloc(busy, count, core_count) == want


def test_the_queue_picks_groups_with_the_allocator():
    stream = Stream()
    for count in (10, 20, 5):
        stream.enqueue(1, count, args=[count])
    stream.wait(0)
    stream.halt(stream.const(0))
    done = stream.simulate(delay=3)
    assert [
        (command.first, command.count, command.slot) for command in done.commands
    ] == [
        (0, 10, 0),
        (10, 20, 1),
        (30, 5, 2),
    ]
    assert all(command.queued and command.pass_index == 0 for command in done.commands)


def test_a_job_that_does_not_fit_does_not_block_those_behind_it():
    stream = Stream()
    stream.enqueue(1, 100, args=[1])
    stream.enqueue(1, 100, args=[2])
    stream.enqueue(1, 10, args=[3])
    stream.wait(0)
    stream.halt(stream.const(0))
    done = stream.simulate(delay=2)
    assert [
        (command.args[0], command.first, command.pass_index)
        for command in done.commands
    ] == [
        (1, 0, 0),
        (3, 100, 0),
        (2, 0, 3),
    ]


def test_the_queue_looks_at_the_first_16_jobs_only():
    def first_pass_of_small(blockers):
        stream = Stream()
        for _ in range(blockers):
            stream.enqueue(1, 100, args=[0])
        stream.enqueue(1, 19, args=[1])
        stream.wait(0)
        stream.halt(stream.const(0))
        done = stream.simulate(delay=2)
        return next(
            command for command in done.commands if command.args[0] == 1
        ).pass_index

    assert first_pass_of_small(15) == 0
    assert first_pass_of_small(16) > 0


def blocked_big_job(smalls):
    """A long job, a 119-core job, then `smalls` one-core jobs."""
    stream = Stream(max_steps=100_000)
    i = stream.registers(1)[0]
    stream.enqueue(1, 1, args=[0])
    stream.enqueue(2, 119, args=[1])
    top = stream.label()
    stream.enqueue(1, 1, args=[2])
    stream.addi(i, i, 1)
    stream.jlt(i, stream.const(smalls), top)
    stream.wait(0)
    stream.halt(stream.const(0))
    return stream


def long_first_job(command):
    return 3000 if command.sequence == 1 else 1


def submitted_before_the_big_job(stream):
    done = stream.simulate(delay=long_first_job, max_passes=10_000)
    assert done.status == "ended"
    (big,) = [command for command in done.commands if command.handler == 2]
    ahead = [
        command
        for command in done.commands
        if command.handler == 1 and command.sequence > 1
    ]
    return (
        big,
        [command for command in ahead if command.pass_index < big.pass_index],
        done,
    )


def test_a_job_is_passed_over_at_most_the_limit(monkeypatch):
    limit = 5
    monkeypatch.setattr(stream_module, "QUEUE_PASSES", limit)
    big, ahead, done = submitted_before_the_big_job(blocked_big_job(100))
    assert len(ahead) == limit
    assert big.pass_index == 3001  # when the long job is acknowledged


def test_a_full_queue_makes_submit_wait():
    stream = Stream()
    for i in range(70):
        stream.enqueue(1, 119, args=[i])
    stream.event(stream.const(1))
    stream.wait(0)
    stream.halt(stream.const(0))
    done = stream.simulate(delay=4)
    (progress,) = [
        event for event in done.events if event.kind == stream_module.EVENT_STREAM
    ]
    # 64 jobs fit in pass 0; each of the six more waits for one job to leave the queue
    assert progress.pass_index == 25
    assert [command.args[0] for command in done.commands] == list(range(70))
    unlimited = stream.simulate(delay=4, core_count=119)
    assert unlimited.status == "ended"


def test_a_cancelled_stream_drops_its_queue():
    stream = Stream()
    for _ in range(10):
        stream.enqueue(1, 119)
    stream.wait(0)
    stream.halt(stream.const(0))
    done = stream.simulate(delay=5, cancel_pass=2)
    assert (
        done.error.reason == stream_module.ERROR_CANCELLED and len(done.commands) == 1
    )


def test_excl_waits_for_an_overlapping_group_and_others_do_not():
    stream = Stream()
    stream.submit(1, group=(0, 10), args=[0])
    stream.submit(1, group=(5, 10), args=[1], exclusive=True)
    stream.submit(1, group=(20, 5), args=[2], exclusive=True)
    stream.submit(1, group=(8, 4), args=[3])
    stream.wait(0)
    stream.halt(stream.const(0))
    done = stream.simulate(delay=2)
    got = [
        (command.args[0], command.pass_index, command.flags)
        for command in done.commands
    ]
    assert got == [(0, 0, 0), (1, 3, 2), (2, 3, 2), (3, 3, 0)]


def test_excl_waits_for_every_job_queued_before_it():
    stream = Stream()
    for i in range(3):
        stream.enqueue(1, 119, args=[i])
    stream.submit(2, group=(100, 4), args=[3], exclusive=True)
    stream.wait(0)
    stream.halt(stream.const(0))
    for delay in (1, 2, 3):
        done = stream.simulate(delay=delay)
        got = [(command.args[0], command.queued) for command in done.commands]
        assert got == [(0, True), (1, True), (2, True), (3, False)], (delay, got)


# check: every reason, with its program counter


def tiny(**changes):
    stream = Stream(**changes)
    register = stream.registers(1)[0]
    stream.set(register, 1)
    stream.submit(1, group=(0, 1))
    stream.halt(register)
    return bytearray(stream.encode())


def poke(data, offset, fmt, *values):
    struct.pack_into(fmt, data, offset, *values)
    return data


def op_at(index):
    return stream_module.HEADER_BYTES + index * stream_module.UOP_BYTES


TEMPLATE = stream_module.HEADER_BYTES + 64  # three micro-ops pad to one line


def test_bad_headers_are_reason_1_at_pc_0():
    for mutate in [
        lambda data: poke(data, 0, "<I", 0),
        lambda data: poke(data, 4, "<I", 2),
        lambda data: poke(data, 8, "<I", 0),
        lambda data: poke(data, 16, "<I", 0),
        lambda data: data + bytes(64),
        lambda data: data[:-1],
        lambda data: bytearray(),
    ]:
        assert check(bytes(mutate(tiny())), 1 << 20) == (stream_module.ERROR_HEADER, 0)


def test_the_upload_must_fit_the_region():
    data = bytes(tiny())
    assert check(data, len(data)) is None
    assert check(data, len(data) - 64) == (stream_module.ERROR_HEADER, 0)


def test_unknown_opcodes_are_reason_2():
    for code in (0, 21, 255):
        data = poke(tiny(), op_at(1), "<B", code)
        assert check(bytes(data), 1 << 20) == (stream_module.ERROR_OPCODE, 1), code


def test_registers_above_15_are_reason_4_for_fields_in_use():
    stream = Stream()
    register = stream.registers(1)[0]
    stream.add(register, register, register)
    stream.halt(register)
    for field in (1, 2, 3):
        data = bytearray(stream.encode())
        poke(data, op_at(0) + field, "<B", 16)
        assert check(bytes(data), 1 << 20) == (stream_module.ERROR_REGISTER, 0)


def test_unused_operand_fields_are_ignored():
    data = tiny()
    poke(data, op_at(0) + 2, "<BB", 200, 201)  # SET uses only register a
    poke(data, op_at(1) + 1, "<BBB", 200, 201, 202)  # SUBMIT uses none
    poke(data, op_at(1) + 8, "<I", 0xDEADBEEF)
    assert check(bytes(data), 1 << 20) is None


def test_jump_targets_at_or_above_op_count_are_reason_3():
    for operation in ["jmp", "jlt"]:
        stream = Stream()
        register_a, register_b = stream.registers(2)
        top = stream.label()
        if operation == "jmp":
            stream.jmp(top)
        elif operation == "jnz":
            stream.jnz(register_a, top)
        else:
            getattr(stream, operation)(register_a, register_b, top)
        stream.halt(register_a)
        data = bytearray(stream.encode())
        assert check(bytes(data), 1 << 20) is None
        poke(data, op_at(0) + 4, "<I", 2)
        assert check(bytes(data), 1 << 20) == (stream_module.ERROR_JUMP, 0)
        poke(data, op_at(0) + 4, "<I", 1)
        assert check(bytes(data), 1 << 20) is None
        poke(data, op_at(0) + 4, "<I", 0xFFFFFFFF)
        assert check(bytes(data), 1 << 20) == (stream_module.ERROR_JUMP, 0)


def test_bad_templates_are_reason_5_at_their_first_user():
    for offset, fmt, value in [
        (4, "<I", 0),  # handler 0
        (4, "<I", 250),
        (4, "<I", 1 | 8 << 8),  # KM_FLAG_QUEUE is not for templates
        (64 + 10, "<B", 5),  # patch_count
        (64 + 8, "<B", 16),  # first_register
    ]:
        data = poke(tiny(), TEMPLATE + offset, fmt, value)
        assert check(bytes(data), 1 << 20) == (stream_module.ERROR_TEMPLATE, 1)


def test_patch_entries_are_checked_only_up_to_patch_count():
    data = tiny()
    poke(data, TEMPLATE + 64 + 10, "<B", 1)
    poke(data, TEMPLATE + 64, "<B", 10)  # argument number 10
    assert check(bytes(data), 1 << 20) == (stream_module.ERROR_TEMPLATE, 1)
    poke(data, TEMPLATE + 64, "<B", 9)
    poke(data, TEMPLATE + 64 + 4, "<B", 16)  # register 16
    assert check(bytes(data), 1 << 20) == (stream_module.ERROR_TEMPLATE, 1)
    poke(data, TEMPLATE + 64 + 4, "<B", 15)
    assert check(bytes(data), 1 << 20) is None
    poke(data, TEMPLATE + 64 + 10, "<B", 0)
    poke(data, TEMPLATE + 64 + 1, "<B", 99)  # beyond patch_count: ignored
    assert check(bytes(data), 1 << 20) is None


def test_an_unused_bad_template_is_reported_at_pc_0():
    stream = Stream()
    stream.submit(1, group=(0, 1))
    stream.submit(2, group=(0, 1))
    stream.halt(stream.const(0))
    data = bytearray(stream.encode())
    base = stream_module.HEADER_BYTES + 64 + stream_module.TEMPLATE_BYTES
    poke(data, base + 4, "<I", 250)
    assert check(bytes(data), 1 << 20) == (stream_module.ERROR_TEMPLATE, 2)
    poke(data, op_at(2) + 4, "<I", 0)  # the second submit now uses template 0
    assert check(bytes(data), 1 << 20) == (stream_module.ERROR_TEMPLATE, 0)


def test_a_bad_upload_posts_one_error_event_with_the_header_request_id():
    data = poke(tiny(req=77), op_at(1), "<B", 99)
    done = simulate(bytes(data))
    assert done.status == "error" and done.commands == [] and done.passes == 1
    assert done.events == [
        stream_module.Event(stream_module.EVENT_STREAM_ERROR, 77, 2 | 1 << 8, 0)
    ]
    done = simulate(bytes(data[:10]))
    assert done.events == [
        stream_module.Event(stream_module.EVENT_STREAM_ERROR, 0, 1, 0)
    ]


# the group rule, RUN, builder validation, orphans


def test_a_group_outside_the_grid_is_found_without_a_sum():
    for first, count, outside in [
        (0, 119, False),
        (118, 2, True),
        (0, 0, True),
        (0xFFFFFFFF, 2, True),
        (1, 0xFFFFFFFF, True),
    ]:
        assert stream_module.group_outside(first, count, 119) == outside


def test_run_answers_what_the_kernel_must_say():
    data = bytes(tiny(req=9))
    length = len(data)
    assert stream_module.check_run(data, 0, 1 << 20) == (None, 9)
    assert stream_module.check_run(data, 64, 1 << 20) == (None, 9)
    rejected = stream_module.StreamError(stream_module.ERROR_HEADER, 0)
    assert stream_module.check_run(data, 8, 1 << 20) == (
        rejected,
        9,
    )  # not a multiple of 64
    assert stream_module.check_run(data, 0, length - 64) == (
        rejected,
        9,
    )  # longer than the region
    assert stream_module.check_run(data, 64, length + 32) == (
        rejected,
        9,
    )  # ends outside
    assert stream_module.check_run(data, 64, length + 64) == (
        None,
        9,
    )  # ends exactly at the end
    assert stream_module.check_run(data, 1 << 32, 1 << 20) == (
        rejected,
        0,
    )  # no sum is formed
    assert stream_module.check_run(data, 0xFFFFFFC0, 1 << 20) == (rejected, 0)
    assert stream_module.check_run(data, 1 << 20, 1 << 20) == (
        rejected,
        0,
    )  # header outside


def test_run_is_rejected_over_an_active_stream_or_request():
    data = bytes(tiny(req=9))
    length = len(data)
    rejected = stream_module.StreamError(stream_module.ERROR_HEADER, 0)
    big = 1 << 20
    assert stream_module.check_run(data, 0, big, active=[(0, length)]) == (rejected, 9)
    assert stream_module.check_run(data, length, big, active=[(0, length)]) == (
        None,
        9,
    )  # touching
    assert stream_module.check_run(data, 64, big, active=[(0, 128)]) == (rejected, 9)
    assert stream_module.check_run(data, 0, big, active=[(length, length)]) == (None, 9)
    assert stream_module.check_run(data, 0, big, active_reqs=[9]) == (rejected, 9)
    assert stream_module.check_run(data, 0, big, active_reqs=[8]) == (None, 9)


def test_the_builder_names_the_field_it_rejects():
    for field, value in (
        ("max_steps", 0),
        ("seed", 0),
        ("req", 1 << 32),
    ):
        with pytest.raises(ValueError, match=field):
            Stream(**{field: value})
    stream = Stream()
    stream.max_steps = 1 << 32
    stream.halt(stream.const(0))
    with pytest.raises(ValueError, match="max_steps"):
        stream.encode()
    with pytest.raises(ValueError, match="pending"):
        stream.wait(-1)
    with pytest.raises(ValueError, match="pending"):
        stream.wait(1 << 32)


def test_the_builder_rejects_too_many_micro_ops():
    stream = Stream()
    register = stream.registers(1)[0]
    for _ in range(65535):
        stream.mov(register, register)
    stream.halt(register)
    with pytest.raises(ValueError, match="65536 micro-ops, at most 65535"):
        stream.encode()


def test_the_default_deadline_applies_when_the_header_has_none():
    stream = Stream(deadline_microseconds=0)
    stream.submit(1, group=(0, 1))
    stream.halt(stream.const(0))
    done = stream.simulate(delay=None, default_deadline_passes=3)
    assert (
        done.error
        == (stream_module.ERROR_DEADLINE, pcs(stream, stream_module.OP_SUBMIT)[0])
        and done.passes == 4
    )
    second_stream = Stream(deadline_microseconds=1)
    second_stream.submit(1, group=(0, 1))
    second_stream.halt(second_stream.const(0))
    assert second_stream.simulate(
        delay=None, default_deadline_passes=3, max_passes=20
    ).status == ("stuck")


def test_a_late_orphan_posts_a_timeout_for_its_slot_with_the_ended_requests_id():
    stream = Stream(req=33)
    register = stream.registers(1)[0]
    stream.submit(1, group=(0, 1), args=[1])
    stream.submit(1, group=(1, 1), args=[2])
    stream.submit(1, group=(2, 1), args=[3])
    stream.mod(register, register, register)
    done = stream.simulate(
        delay=lambda command: None if command.args[0] == 2 else 0, deadline_passes=5
    )
    assert done.error.reason == stream_module.ERROR_MOD_ZERO
    assert done.events[0].kind == stream_module.EVENT_STREAM_ERROR
    assert done.events[1] == stream_module.Event(stream_module.EVENT_TIMEOUT, 33, 1, 5)
    assert done.stopped and done.passes == 6
    quiet = stream.simulate(delay=1, deadline_passes=5)
    assert len(quiet.events) == 1 and not quiet.stopped


def test_the_two_submitters_take_turns_on_free_slots():
    stream = Stream()
    stream.set(stream.registers(1)[0], 0)
    for _ in range(16):
        stream.submit(1, group=(0, 1))
    stream.enqueue(2, 1)
    stream.wait(0)
    stream.halt(stream.const(0))
    done = stream.simulate(delay=0)
    job = next(command for command in done.commands if command.queued)
    assert job.pass_index % 2 == 1  # odd passes give the queue the first call
    assert all(
        command.pass_index == job.pass_index or not command.queued
        for command in done.commands
    )


def test_the_example_in_the_docs_runs():
    text = (ROOT / "docs" / "streams.md").read_text()
    code = re.search(r"```python\n(.*?)```", text, re.S)[1]
    scope = {}
    exec(code, scope)
    done = scope["stream"].simulate()
    assert done.status == "ended" and len(done.commands) == 1000
    assert all(1 <= command.count <= 119 for command in done.commands)
