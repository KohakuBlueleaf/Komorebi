"""The path for code added to a live run (KM_ADDED_HANDLERS) is defined only for a
program that asks: each way of asking, what the runtime's kernels get, and the refusal
of run.addon on a run that did not ask, before any command (no card)."""

import pytest
from test_added import a_room, fake_run, parts
from test_pieces_meet import Gate, made

import komorebi as km
from komorebi.program import overlay
from komorebi.program.addon.code import AddError
from komorebi.session import kernels as session_kernels
from komorebi.session.run import Run

DEFINE = "KM_ADDED_HANDLERS"


def program(tmp_path):
    return km.Program("p", kernels=parts(tmp_path / "k", overlay.KERNELS))


def test_each_way_of_asking_defines_the_path(tmp_path):
    for bytes in (2048, {"writer": 2048}, {}):
        asked = program(tmp_path).slots(count=2, bytes=bytes)
        assert asked.defines[DEFINE] == "1"
    assert DEFINE not in program(tmp_path).slots(count=2).defines
    assert DEFINE not in program(tmp_path).defines
    assert (
        DEFINE in made(Gate, handlers={"gate": km.ANY}).defines
    )  # a class's first use
    assert DEFINE not in made(Gate).defines


def test_the_runtime_s_kernels_get_the_path_only_when_every_program_asks(tmp_path):
    asked = program(tmp_path).slots(bytes={})
    plain = program(tmp_path)
    assert session_kernels.added_path([asked, asked]) == [(DEFINE, "1")]
    assert session_kernels.added_path([asked, plain]) == []
    assert session_kernels.added_path([plain]) == []


def test_an_add_on_a_run_that_did_not_ask_is_refused_before_any_command(tmp_path):
    run, writes, queued = fake_run(a_room(entered=False))
    folder = parts(tmp_path / "mix")
    for kind in ({}, {"handler": True}, {"message": True}):
        with pytest.raises(AddError, match=r"Program\.slots\(bytes=.*km\.ANY"):
            Run.addon(run, "mix", folder, **kind)
    assert not writes and not queued
