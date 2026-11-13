"""Programs in the runtime's form: the program field wherever a command is encoded,
the two forms, the plan of several programs in one run and the host's refusals."""

import numpy as np
import pytest
from fake_port import FakeBackend, FakePort

import komorebi as km
from komorebi import abi, overlay, programs
from komorebi.engines import graph as graph_module
from komorebi.engines import stream as stream_module
from komorebi.errors import L1LayoutError, ProgramError
from komorebi.program import Handler, source_form
from komorebi.run import Run

CONSTANTS = abi.headers().constants
GRID = (12, 10)
BUFFERS = [
    km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
    km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
]
PROGRAM = "void km_program(km::{kind}& kernel, const km_command_t& command) {{}}\n"
KERNEL = "void kernel_main() {}\n"


def sources(folder, form="program"):
    folder.mkdir(parents=True, exist_ok=True)
    for kernel, kind in (("reader", "Reader"), ("writer", "Writer")):
        text = PROGRAM.format(kind=kind) if form == "program" else KERNEL
        (folder / f"{kernel}.cpp").write_text(text)
    text = PROGRAM.format(kind="Compute") if form == "program" else KERNEL
    (folder / "compute.cpp").write_text(text)
    return folder


def made(tmp_path, name, buffers=BUFFERS, **options):
    return km.Program(
        name,
        kernels=sources(tmp_path / name),
        buffers=list(buffers),
        handlers={"one": 1, "two": 2},
        **options,
    )


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
    today = km.Program("today", kernels=sources(tmp_path / "today", "kernel"))
    with pytest.raises(ProgramError, match="runtime's form"):
        programs.plan([first, today], 120)


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
    assert made_overlay.number == overlay.constants()["kOverlaysMax"] - 2
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
