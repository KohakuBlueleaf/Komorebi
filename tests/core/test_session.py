"""Session.load and its failure paths over a stub ttnn and the fake kernel."""

import pytest
from fake_session import load_session, program

from komorebi.errors import ProgramError


@pytest.fixture
def card(monkeypatch, tmp_path):
    return load_session(monkeypatch, tmp_path)


def test_a_load_launches_the_grid_then_starts_the_kernel(card, tmp_path):
    session, log, _, _ = card
    run = session.Session(device=object()).load(program(tmp_path))
    assert log == ["prepare", "describe", "generic_op", "start", "wait_ready"]
    assert not run.stopped


def test_an_interrupt_during_the_launch_still_reads_the_grid_back(card, tmp_path):
    session, log, ttnn, _ = card

    def launched_then_interrupted(io, descriptor):
        log.append("generic_op")
        raise KeyboardInterrupt  # delivered as the launch call returns

    ttnn.generic_op = launched_then_interrupted
    with pytest.raises(KeyboardInterrupt):
        session.Session(device=object()).load(program(tmp_path))
    assert log[-2:] == ["close", "read_back"]


def test_a_compile_error_raises_program_error_and_reads_back(card, tmp_path):
    session, log, ttnn, _ = card

    def compile_error(io, descriptor):
        raise RuntimeError("reader.cpp: error: expected ';'")

    ttnn.generic_op = compile_error
    with pytest.raises(ProgramError, match="expected"):
        session.Session(device=object()).load(program(tmp_path))
    assert "start" not in log and log[-2:] == ["close", "read_back"]
