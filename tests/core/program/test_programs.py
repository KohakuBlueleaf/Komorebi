"""Programs in the runtime's form: the program field wherever a command is encoded,
the two forms, the plan of several programs in one run and the host's refusals."""

from types import SimpleNamespace

import numpy as np
import pytest
from fake_port import FakeBackend, FakePort
from programs_fixture import BUFFERS, GRID, KERNEL, PROGRAM, made, sources

import komorebi as km
from komorebi import abi
from komorebi.engines import graph as graph_module
from komorebi.engines import stream as stream_module
from komorebi.errors import L1LayoutError, ProgramError
from komorebi.program import overlay
from komorebi.program import several as programs
from komorebi.program.description import Handler, source_form
from komorebi.session import kernels as session_kernels
from komorebi.session.backend import backend as card_backend
from komorebi.session.run import Run

CONSTANTS = abi.headers().constants


def test_a_command_of_0_1_0_is_what_it_was_and_the_field_is_bits_24_to_31():
    old = abi.command(7, 3, 9, [1, 2], flags=CONSTANTS["KM_FLAG_RESULT"])
    assert old[1] == 7 | CONSTANTS["KM_FLAG_RESULT"] << 8
    assert abi.command(7, 3, 9, [1, 2], program=5)[1] == 7 | 5 << 24


def test_the_forms_are_told_apart_by_the_sources(tmp_path):
    (tmp_path / "a.cpp").write_text("// km_program( in a comment\n" + KERNEL)
    (tmp_path / "b.cpp").write_text(PROGRAM.format(kind="Reader"))
    (tmp_path / "c.cpp").write_text(KERNEL + PROGRAM.format(kind="Reader"))
    assert source_form(tmp_path / "a.cpp") == "kernel"
    assert source_form(tmp_path / "b.cpp") == "program"
    with pytest.raises(ProgramError, match="not both"):
        source_form(tmp_path / "c.cpp")
    program = made(tmp_path, "mixed")
    (tmp_path / "mixed" / "writer.cpp").write_text(KERNEL)
    with pytest.raises(ProgramError, match="mix the two forms"):
        program.form()
    assert made(tmp_path, "new").form() == "program"


def test_a_handler_names_its_program_and_its_number(tmp_path):
    first, second = made(tmp_path, "first"), made(tmp_path, "second")
    plan = programs.plan([first, second], 120)
    one = second.handler("one")
    assert isinstance(one, Handler) and one == 1 and one.program is second
    assert one.field == 1 and first.handler(1).field == 0
    assert [program.program_field for program in plan.programs] == [0, 1]
    with pytest.raises(ValueError, match="not a handler of"):
        first.handler(one)


def test_streams_and_graphs_carry_the_field(tmp_path):
    first, second = made(tmp_path, "first"), made(tmp_path, "second")
    programs.plan([first, second], 120)
    stream = stream_module.Stream()
    stream.submit(second.handler("two"), (0, 4))
    stream.submit(2, (0, 4))
    words = [template.rec[1] for template in stream.templates()]
    assert words == [2 | 1 << 24, 2]
    assert [template.program for template in stream.templates()] == [1, 0]
    assert [c.program for c in stream.simulate().commands] == [1, 0]
    graph = graph_module.Graph()
    graph.command(second.handler("one"), (0, 4))
    assert graph.templates()[0].rec[1] >> 24 == 1


def test_the_plan_of_disjoint_groups_and_of_combined_programs(tmp_path):
    first, second = made(tmp_path, "first"), made(tmp_path, "second")
    plan = programs.plan({first: (0, 88), second: (88, 20)}, 120)
    assert [(g.first, g.count, g.programs) for g in plan.groups] == [
        (0, 88, (0,)),
        (88, 20, (1,)),
        (108, 12, ()),
    ]
    holds = plan.holds()
    assert holds[0][:88].all() and not holds[0][88:].any() and holds[1][100]
    with pytest.raises(ValueError, match="overlaps"):
        programs.plan({first: (0, 90), second: (88, 20)}, 120)
    with pytest.raises(ValueError, match="not inside"):
        programs.plan({first: (110, 20)}, 120)
    combined = programs.plan([first, second], 120)
    assert [(g.first, g.count, g.programs) for g in combined.groups] == [
        (0, 120, (0, 1))
    ]
    kernel_form = km.Program(
        "kernel_form", kernels=sources(tmp_path / "kernel_form", "kernel")
    )
    with pytest.raises(ProgramError, match="runtime's form"):
        programs.plan([first, kernel_form], 120)


def test_programs_that_share_cores_share_one_l1_layout(tmp_path):
    first = made(tmp_path, "first")
    same = made(tmp_path, "same")
    other = made(
        tmp_path, "other", buffers=[BUFFERS[0], km.Buffer(17, "uint32", 64, 1)]
    )
    assert programs.joint_buffers([first, same], 1 << 20) == BUFFERS
    assert [b.index for b in programs.joint_buffers([first, other], 1 << 20)] == [
        0,
        16,
        17,
    ]
    clash = made(tmp_path, "clash", buffers=[km.Buffer(16, "float32", 4096, 2)])
    with pytest.raises(L1LayoutError, match="buffer 16 is declared") as error:
        programs.joint_buffers([first, clash], 1 << 20)
    assert "clash: buffer 16" in error.value.table and "total" in error.value.table
    options = made(tmp_path, "options", compute={"fp32_dest_acc_en": True})
    with pytest.raises(ProgramError, match="compute options"):
        programs.plan([first, options], 120)
    programs.plan({first: (0, 60), options: (60, 60)}, 120)  # apart: allowed


def test_a_program_is_an_overlay_numbered_from_63_down(tmp_path):
    first, second = made(tmp_path, "first"), made(tmp_path, "second")
    second.compile_args = {"reader": [3, 4]}
    programs.plan([first, second], 120)
    made_overlay = programs.program_overlay(second)
    assert made_overlay.number == overlay.constants()["kOverlayService"] - 2
    assert made_overlay.program and made_overlay.compile_args["reader"] == [3, 4]
    assert programs.start_buffers(BUFFERS) == (0, 16)
    assert programs.start_buffers([]) == (0, 16)
    assert programs.rectangles(88, 32, 10) == [(8, 8, 9, 8), (0, 9, 9, 11)]


def test_the_runs_commands_carry_the_field_and_refuse_cores_without_the_program(
    tmp_path,
):
    first, second = made(tmp_path, "first"), made(tmp_path, "second")
    plan = programs.plan({first: (0, 88), second: (88, 32)}, 120)
    port = FakePort()
    run = Run(
        FakeBackend(port), GRID, first, programs=plan.programs, holds=plan.holds()
    )
    run.submit(second.handler("one"), (90, 4))
    run.submit("two", (0, 4))  # a name: the first program's
    run.submit_many(second.handler(2), np.array([88, 100]), 2)
    run.wait(run.submit(0, (0, 120)))  # NOP: every core
    words = [port.ring[req][1] for req in sorted(port.ring)]
    assert words[:4] == [1 | 1 << 24, 2, 2 | 1 << 24, 2 | 1 << 24]
    with pytest.raises(ValueError, match="does not hold second"):
        run.submit(second.handler("one"), (80, 10))
    with pytest.raises(ValueError, match="does not hold first"):
        run.submit_many(first.handler(1), np.array([0, 87]), 2)
    with pytest.raises(ValueError, match="every core does not hold"):
        run.submit(first.handler(1), 4, queued=True)
    stranger = made(tmp_path, "stranger")
    with pytest.raises(ValueError, match="not a program of this run"):
        run.submit(stranger.handler(1), (0, 1))
    stream = stream_module.Stream()
    stream.submit(second.handler("one"), (0, 4))
    with pytest.raises(ValueError, match="does not hold second"):
        run.check_templates(stream.templates())
    assert run.field_handler(2, 1).program is second and run.field_handler(2, 0) == 2
    assert run._rejections[CONSTANTS["KMD_REJECT_PROGRAM"]] == "program"


def test_each_group_gets_its_own_kernels_whatever_order_the_runtime_lists_them(
    tmp_path,
):
    rows = []
    for number in (2, 0, 1):  # the runtime's order is not the groups'
        compute = tmp_path / f"compute{number}" / "trisc0"
        compute.mkdir(parents=True)
        (compute.parent / "defines_generated.h").write_text(
            f"#pragma once\n#define KM_PROGRAM_GROUP {number}\n"
        )
        reader = ["brisc", str(tmp_path / f"r{number}.elf"), "riscv-g++", "O2"]
        reader += ["", "", "", "O2", f"-DKM_PROGRAM_GROUP={number}"]
        trisc = ["trisc0", str(compute / "trisc0.elf"), "riscv-g++", "O3"]
        trisc += ["", "", "", "O3"]
        rows.append(("program/reader.cpp", [reader]))
        rows.append(("program/compute.cpp", [trisc]))
    groups = session_kernels._group_recipes(rows, 3)
    for number, recipes in enumerate(groups):
        assert recipes["brisc"].elf.name == f"r{number}.elf"
        assert recipes["trisc0"].elf.parent.parent.name == f"compute{number}"
    with pytest.raises(ProgramError, match="no group number"):
        session_kernels._group_recipes(rows, 2)


def switching_run(tmp_path, refused=None):
    first, second = made(tmp_path, "first"), made(tmp_path, "second")
    plan = programs.plan({first: (0, 88), second: (88, 32)}, 120)
    port = FakePort()
    port.programs = [0] * 88 + [1] * 32
    switch = {"records": {0: 0x1000, 1: 0x1240}, "refused": refused}
    run = Run(
        FakeBackend(port),
        GRID,
        first,
        programs=plan.programs,
        holds=plan.holds(),
        switch=switch,
    )
    return run, port, first, second


def test_a_load_is_a_command_naming_the_program_and_its_record(tmp_path):
    run, port, first, second = switching_run(tmp_path)
    assert programs.plan({first: (0, 88), second: (88, 32)}, 120).switch
    assert not programs.plan([first, second], 120).switch
    run.wait(run.load(second, (60, 28)))
    words = port.ring[1]
    assert words[1] == CONSTANTS["KM_HANDLER_LOAD"] | 1 << 24
    assert words[2] == 60 | 28 << 16 and words[3] == 0x1240
    assert port.programs[59:61] == [0, 1] and port.programs[87] == 1
    run.wait(run.submit(second.handler("one"), (60, 40)))  # the host's own check
    with pytest.raises(ValueError, match="does not hold first"):
        run.submit(first.handler("one"), (50, 20))
    together = Run(FakeBackend(FakePort()), GRID, first)
    with pytest.raises(ProgramError, match="can switch program"):
        together.load(first, (0, 4))
    stuck, *_ = switching_run(tmp_path / "again", refused="kernels differ")
    with pytest.raises(ProgramError, match="kernels differ"):
        stuck.load(second, (0, 4))


def test_the_dispatchers_table_refuses_with_reason_13(tmp_path):
    run, port, first, second = switching_run(tmp_path)
    run._holds = None  # past the host's check: the dispatcher's own
    handles = [
        run.submit(second.handler("one"), (80, 10)),
        run.submit(0, (80, 10)),  # NOP: any core
        run.submit(first.handler("one"), (0, 88)),
        run.submit(first.handler("one"), 4, queued=True),
        run.load(second, (0, 60)),
        run.submit(second.handler("one"), (0, 60)),
        run.submit(first.handler("one"), (59, 2)),
    ]
    rejected = []
    for handle in handles:
        try:
            run.wait(handle)
        except km.CommandRejected as error:
            rejected.append((handle, error.reason_name))
    assert rejected == [(1, "program"), (4, "program"), (7, "program")]
    port.programs = None
    with pytest.raises(km.CommandRejected, match="program"):
        run.wait(run.load(second, (0, 4)))


def test_each_program_has_room_for_its_own_layout_after_the_runtimes(tmp_path):
    runtime = [(29, "uint32", 64, 1), (30, "uint32", 64, 1)]
    _, start = programs.buffer_layout(runtime, 0x10010)
    assert start == 0x10040 + 128
    cbs = [(0, "bfloat16", 2048, 2), (16, "uint32", 48, 1), (17, "float32", 16, 3)]
    layout, end = programs.buffer_layout(cbs, start)
    assert layout == {
        0: (start, 4096, 2, 2048),
        16: (start + 4096, 48, 1, 48),
        17: (start + 4096 + 64, 48, 3, 16),
    }
    assert end == start + 4096 + 64 + 48
    words = programs.record_words(3, layout, [(0x2000, 768)] + [(0, 0)] * 4)
    c = overlay.constants()
    assert words[c["kRecordMask"]] == 1 | 1 << 16 | 1 << 17
    assert words[c["kRecordField"]] == 3 and list(words[2:4]) == [0x2000, 768]
    at = c["kRecordBuffers"] + 4 * 17
    assert list(words[at : at + 4]) == [start + 4160, 48, 3, 16]
    wide = made(tmp_path, "wide", buffers=[km.Buffer(40, "uint32", 64, 1)])
    with pytest.raises(ProgramError, match="indices 0 to 31, and it sits at 32 or above"):
        programs.plan({wide: (0, 10)}, 120)


def test_buffers_declared_out_of_index_order_are_laid_out_in_index_order(tmp_path):
    declared = [
        km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=2),
        km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        km.Buffer(index=2, format="bfloat16", page_bytes=2048, pages=1),
    ]
    program = made(tmp_path, "unordered", buffers=declared)
    plan = programs.plan({program: (0, 10)}, 120)
    module = SimpleNamespace(kernel_buffer=lambda device: (0, 0x1000, 0x20000))
    backend = card_backend._Backend.__new__(card_backend._Backend)
    backend.card, backend.device, backend.trace_cap = (
        SimpleNamespace(module=module),
        None,
        0,
    )
    backend._switch_layouts(plan)
    layout = backend.switch_layouts[program.program_field]
    addresses = [layout[index][0] for index in (0, 2, 16)]
    assert addresses == sorted(addresses)
    sizes = [layout[index][1] for index in (0, 2)]
    assert layout[2][0] == layout[0][0] + sizes[0]
    assert layout[16][0] == layout[2][0] + sizes[1]
