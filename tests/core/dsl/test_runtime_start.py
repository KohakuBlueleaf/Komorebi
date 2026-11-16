"""A start-up of a class's own in the runtime's form: a body at a kernel's start and
startup= as km_program_start, run at the load of the program; a class alone (or alone
in a list) runs as a dict over every core, which loads it first; a list of several,
km.merge of such a class, and a body at a kernel's end refused (no card)."""

import pytest
from test_class_runtime_form import Copy, Started, StartedUp, made

import komorebi as km
from komorebi.errors import ProgramError
from komorebi.program import several as programs


def start_of(program, kernel):
    text = program.sources()[kernel].read_text()
    return (
        text[text.index("#define KM_PROGRAM_START") :]
        if "KM_PROGRAM_START" in text
        else ""
    )


def test_the_start_body_and_the_startup_run_at_the_load():
    started, up = made(Started, form="runtime"), made(StartedUp, form="runtime")
    assert started.load_start and up.load_start
    assert not made(Copy, form="runtime").load_start
    compute = start_of(started, "compute")
    assert "static void km_program_start(km::Compute& compute) {" in compute
    default = "compute_kernel_hw_startup(KM_PROGRAM_SETUP_IN, KM_PROGRAM_SETUP_OUT);"
    assert compute.index(default) < compute.index("prepare();")
    assert start_of(started, "reader") == ""  # no body on the movers: nothing there
    text = up.sources()["compute"].read_text()
    assert "compute_kernel_hw_startup(kFieldOnes, kFieldOnes, kFieldOnes);" in start_of(
        up, "compute"
    )
    assert "reconfig_data_format(" not in text  # its start-up's formats hold
    assert made(Started).form() == "kernel"  # the kernel form as it was


def test_a_class_alone_runs_as_a_dict_that_loads_it_first():
    up = made(StartedUp, form="runtime")
    for spec in (up, [up]):
        placed = programs.plan(spec, 120)
        assert placed.switch and [
            (g.first, g.count, g.programs) for g in placed.groups
        ] == [(0, 120, (0,))]
    assert not programs.plan(made(Copy, form="runtime"), 120).switch
    copy = made(Copy, form="runtime")
    assert programs.plan({copy: (0, 60), up: (60, 60)}, 120).switch


def test_the_refusals():
    up, copy = made(StartedUp, form="runtime"), made(Copy, form="runtime")
    with pytest.raises(
        ProgramError, match=r"startedup: a start-up of its own .* in a list"
    ):
        programs.plan([copy, up], 120)
    with pytest.raises(ProgramError, match=r"a merged program has one load"):
        km.merge(copy, up, l1=1 << 20, log=lambda line: None)
