"""Session.load and its failure paths over a stub ttnn and the fake kernel."""

import numpy as np
import pytest
from fake_session import FakeDispatch, K, load_session, program

import komorebi as km
from komorebi.errors import GridTimeout, ProgramError, StartError


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
    assert log[-2:] == ["close", "read_back, grid stopped"]


def test_a_compile_error_raises_program_error_and_reads_back(card, tmp_path):
    session, log, ttnn, _ = card

    def compile_error(io, descriptor):
        raise RuntimeError("reader.cpp: error: expected ';'")

    ttnn.generic_op = compile_error
    with pytest.raises(ProgramError, match="expected"):
        session.Session(device=object()).load(program(tmp_path))
    assert "start" not in log and log[-2:] == ["close", "read_back, grid stopped"]


def test_a_start_error_carries_the_kernels_result(card, tmp_path, monkeypatch):
    session, log, _, _ = card
    monkeypatch.setattr(FakeDispatch, "boot_error", K["KMD_ERR_START"])
    with pytest.raises(StartError, match="error start") as caught:
        session.Session(device=object()).load(program(tmp_path))
    assert caught.value.result["error_name"] == "start"
    assert caught.value.result["state_name"] == "failed"
    assert log[-3:] == ["finish", "close", "read_back, grid stopped"]


def test_a_kernel_that_never_started_leaves_the_host_to_stop_the_grid(
    card, tmp_path, monkeypatch
):
    session, log, _, _ = card
    _, _, _, module = card
    original = module.Dispatch

    def failing(device, pll_mhz):
        d = original(device, pll_mhz)
        d.fail["start"] = RuntimeError("X280 enqueue failed")
        return d

    module.Dispatch = failing
    with pytest.raises(RuntimeError, match="enqueue"):
        session.Session(device=object()).load(program(tmp_path))
    assert log[-2:] == ["close", "read_back, grid stopped"]


def test_the_image_and_the_x280_are_checked_before_the_launch(
    card, tmp_path, monkeypatch
):
    session, log, _, module = card
    missing = tmp_path / "none.bin"
    with pytest.raises(StartError, match="missing"):
        session.Session(device=object()).load(program(tmp_path), image=missing)
    assert log == []
    original = module.Dispatch

    def not_idle(device, pll_mhz):
        d = original(device, pll_mhz)
        d.idle = False
        return d

    module.Dispatch = not_idle
    with pytest.raises(StartError, match="not idle"):
        session.Session(device=object()).load(program(tmp_path))
    assert log == ["close"]


def backend_over(session, log, tmp_path, monkeypatch):
    """The real card backend over a closed fake dispatch and a stub readback."""
    sess = session.Session(device="dev", grid_timeout=0.5)
    sess._module = session.native.load_module()
    port = sess._module.Dispatch("dev", 1750)
    port.close()
    backend = sess._module.card_backend(sess, port, 1350, 0)
    backend.ctl, backend.ctl_addr = "ctl", 0x20000
    words = (K["KM_CTL_BYTES"] + 64) // 4
    rows = np.zeros((120, words), np.int64)
    monkeypatch.setattr(session, "_read_uint32", lambda t: rows)
    log.clear()
    return sess, backend


def test_the_host_writes_the_stop_word_into_every_core_then_waits_bounded(
    card, tmp_path, monkeypatch
):
    session, log, _, _ = card
    sess, backend = backend_over(session, log, tmp_path, monkeypatch)
    state, trace = backend.read_back(stop_grid=True)
    writes = [e for e in log if e[0] == "write_l1"]
    stop = 0x20000 + K["KM_CTL_STOP"]
    assert sorted((x, y) for _, _, x, y, _, _ in writes) == sorted(
        (x, y) for x in range(12) for y in range(10)
    )
    assert {(addr, tuple(w)) for _, _, _, _, addr, w in writes} == {(stop, (1,))}
    assert log[len(writes) :] == ["wait_grid", "synchronize"]
    assert state.shape == (120, K["kStateWords"]) and trace is None


def test_a_grid_that_does_not_return_raises_and_leaves_the_device_open(
    card, tmp_path, monkeypatch
):
    session, log, _, module = card
    sess, backend = backend_over(session, log, tmp_path, monkeypatch)
    module.grid_returns = False
    with pytest.raises(GridTimeout, match="reset"):
        backend.read_back()
    assert log == ["wait_grid"] and sess.stuck
    assert km.GridTimeout is GridTimeout
    sess.close()
    assert "close_device" not in log


def test_a_base_off_the_16_byte_grid_is_refused_before_the_launch(card, tmp_path):
    session, log, _, module = card
    FakeDispatch.lim_base = 0x10008
    try:
        with pytest.raises(StartError, match=r"multiple of 16.*lim_base 0x10008"):
            session.Session(device=object()).load(program(tmp_path))
    finally:
        FakeDispatch.lim_base = 0x10000
    assert log == ["prepare", "close"]
    bases = session.lim_bases(0x10000)
    assert bases["KMD_LIM_MAILBOX"] == 0x10000 + K["KMD_LIM_MAILBOX"]
    assert {"KMD_LIM_UPLOAD", "KMD_LIM_ACKS", "KM_ACK_STRIDE"} <= set(bases)
    session.check_bases(bases)
    with pytest.raises(StartError, match="control block 0x20004"):
        session.check_bases({"control block": 0x20004})
