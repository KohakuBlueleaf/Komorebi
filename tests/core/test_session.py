"""Session.load and its failure paths over a fake native module and the fake kernel."""

import gc
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fake_session import (
    CONSTANTS,
    FakeDevice,
    FakeDispatch,
    built,
    load_session,
    program,
)

import komorebi as km
from komorebi import insn_check
from komorebi.errors import (
    GridTimeout,
    KernelInstructionError,
    ProgramError,
    StartError,
)


@pytest.fixture
def card(monkeypatch, tmp_path):
    return load_session(monkeypatch, tmp_path)


def test_a_load_launches_the_grid_then_starts_the_kernel(card, tmp_path):
    session, log, _ = card
    run = session.Session(device=object()).load(program(tmp_path))
    assert log == ["prepare", "describe", "launch", "start", "wait_ready"]
    assert not run.stopped


def test_an_interrupt_during_the_launch_still_reads_the_grid_back(card, tmp_path):
    session, log, module = card

    def launched_then_interrupted(launch):
        log.append("launch")
        raise KeyboardInterrupt  # delivered as the launch call returns

    module.run = launched_then_interrupted
    with pytest.raises(KeyboardInterrupt):
        session.Session(device=object()).load(program(tmp_path))
    assert log[-2:] == ["close", "read_back, grid stopped"]


def test_a_compile_error_raises_program_error_and_reads_back(card, tmp_path):
    session, log, module = card

    def compile_error(launch):
        raise RuntimeError("reader.cpp: error: expected ';'")

    module.run = compile_error
    with pytest.raises(ProgramError, match="expected"):
        session.Session(device=object()).load(program(tmp_path))
    assert "start" not in log and log[-2:] == ["close", "read_back, grid stopped"]


def test_a_compile_error_in_the_checked_build_raises_program_error(card, tmp_path):
    session, log, module = card

    def compile_error(launch):
        raise RuntimeError("compute.cpp: error: deliberate")

    module.kernel_build = compile_error
    with pytest.raises(ProgramError, match="build failed.*deliberate"):
        session.Session(device=object()).load(program(tmp_path))
    assert "launch" not in log and log[-1] == "close"


def test_the_exit_timeout_is_set_only_when_the_environment_has_none(card):
    session, _, _ = card
    name, log = session.OP_TIMEOUT, []
    unset = {}
    assert session.exit_timeout(unset, log.append) == 30.0
    assert (
        unset[name] == "30"
        and len(log) == 1
        and session.exit_wait(unset) == "up to 30 s"
    )
    for given, timeout, phrase in (
        ("12.5", 12.5, "up to 12.5 s"),
        ("0", 0.0, "without a timeout"),
    ):
        env = {name: given}
        assert session.exit_timeout(env, log.append) == timeout
        assert env[name] == given and session.exit_wait(env) == phrase
    assert len(log) == 1  # a user's value, 0 included, is kept without a line


def test_a_kernel_instruction_finding_stops_the_load_before_the_launch(
    card, tmp_path, monkeypatch
):
    session, log, module = card
    elf = tmp_path / "trisc2.elf"
    elf.write_bytes(b"")
    module.kernel_build = lambda launch: built([("compute.cpp", [str(elf)])])
    found = insn_check.Finding("TRISC2", "compute.cpp", "fdiv.s", "f", "ratio", "", 0)
    monkeypatch.setattr(insn_check, "check_elf", lambda *args: [found])
    with pytest.raises(KernelInstructionError, match="fdiv.s"):
        session.Session(device=object()).load(program(tmp_path))
    assert "launch" not in log and "start" not in log and log[-1] == "close"


def test_a_program_over_the_kernel_buffer_is_rejected_before_the_launch(card, tmp_path):
    session, log, module = card
    said = "Program size (70688) too large for kernel config buffer (70656) on TENSIX"
    module.kernel_build = lambda launch: built([], total=0, error=said)
    prog = program(tmp_path)
    with pytest.raises(km.KernelSizeError) as caught:
        session.Session(device=object()).load(prog)
    message = str(caught.value)
    for part in ("70688 B of the 70656 B", "BRISC, 1024 B", "kernel_bytes", "'Os'"):
        assert part in message
    assert prog.sizes.total == 70688 and prog.sizes.text["TRISC2"] == 768
    assert "launch" not in log and log[-1] == "close"


def test_the_kernel_buffer_is_passed_to_the_open_only_when_set(card, monkeypatch):
    session, _, module = card
    monkeypatch.setattr(session, "_hook_exit", lambda: None)  # not this process's
    session.Session().open()
    session.Session(kernel_bytes=128 << 10).open()
    assert module.opened == [None, 1572864 - 41088 - (128 << 10)]
    assert session.worker_l1_size(69 * 1024) == 1461120  # the runtime's default
    for bad in (100, 0, 1531776, "big"):
        with pytest.raises(ValueError, match="multiple of 64 from 64 to 1531712"):
            session.Session(kernel_bytes=bad)
    with pytest.raises(ValueError, match="opens the device"):
        session.Session(device=object(), kernel_bytes=128 << 10)


def test_a_start_error_carries_the_kernels_result(card, tmp_path, monkeypatch):
    session, log, _ = card
    monkeypatch.setattr(FakeDispatch, "boot_error", CONSTANTS["KMD_ERROR_START"])
    with pytest.raises(StartError, match="error start") as caught:
        session.Session(device=object()).load(program(tmp_path))
    assert caught.value.result["error_name"] == "start"
    assert caught.value.result["state_name"] == "failed"
    assert log[-3:] == ["finish", "close", "read_back, grid stopped"]


def offering(module, clocks):
    """The native Dispatch of a runtime whose clock table is `clocks`."""
    original, asked = module.Dispatch, []

    def dispatch(device, pll_mhz):
        asked.append(pll_mhz)
        if pll_mhz not in clocks:
            raise RuntimeError(
                f"could not establish PLL4 on chip 0: no PLL solution for {pll_mhz} "
                f"MHz. Supported: {' '.join(map(str, clocks))}"
            )
        return original(device, pll_mhz)

    module.Dispatch = dispatch
    return asked


def test_the_default_clock_falls_back_to_1000_once_on_an_older_runtime(card, tmp_path):
    session, _, module = card
    asked = offering(module, [15, 200, 800, 1000, 1750])
    messages = []
    sess = session.Session(device=object(), log=messages.append)
    sess.load(program(tmp_path)).stop()
    sess.load(program(tmp_path)).stop()
    assert asked == [1350, 1000, 1000] and sess.pll_mhz == 1000
    assert len([m for m in messages if "not offered" in m]) == 1


def test_a_clock_asked_for_that_the_runtime_lacks_is_an_error(card, tmp_path):
    session, _, module = card
    asked = offering(module, [15, 200, 800, 1000, 1750])
    with pytest.raises(RuntimeError, match="1350 MHz. Supported: 15 200 800 1000 1750"):
        session.Session(device=object(), pll_mhz=1350).load(program(tmp_path))
    assert asked == [1350]


def test_a_kernel_that_never_started_leaves_the_host_to_stop_the_grid(
    card, tmp_path, monkeypatch
):
    session, log, _ = card
    _, _, module = card
    original = module.Dispatch

    def failing(device, pll_mhz):
        dispatch = original(device, pll_mhz)
        dispatch.fail["start"] = RuntimeError("X280 enqueue failed")
        return dispatch

    module.Dispatch = failing
    with pytest.raises(RuntimeError, match="enqueue"):
        session.Session(device=object()).load(program(tmp_path))
    assert log[-2:] == ["close", "read_back, grid stopped"]


def test_the_image_and_the_x280_are_checked_before_the_launch(
    card, tmp_path, monkeypatch
):
    session, log, module = card
    missing = tmp_path / "none.bin"
    with pytest.raises(StartError, match="missing"):
        session.Session(device=object()).load(program(tmp_path), image=missing)
    assert log == []
    original = module.Dispatch

    def not_idle(device, pll_mhz):
        dispatch = original(device, pll_mhz)
        dispatch.idle = False
        return dispatch

    module.Dispatch = not_idle
    with pytest.raises(StartError, match="not idle"):
        session.Session(device=object()).load(program(tmp_path))
    assert log == ["close"]


def backend_over(session, log, tmp_path, monkeypatch):
    """The real card backend over a closed fake dispatch and a stub readback."""
    sess = session.Session(device=FakeDevice(log), grid_timeout=0.5)
    sess._module = session.native.load_module()
    port = sess._module.Dispatch("dev", 1750)
    port.close()
    backend = sess._module.card_backend(sess, port, 1350, 0)
    backend.control, backend.control_addr = "control", 0x20000
    words = (CONSTANTS["KM_CONTROL_BYTES"] + 64) // 4
    rows = np.zeros((120, words), np.int64)
    monkeypatch.setattr(session, "_read_uint32", lambda tensor: rows)
    log.clear()
    return sess, backend


def test_the_host_writes_the_stop_flag_into_every_core_then_waits_with_a_timeout(
    card, tmp_path, monkeypatch
):
    session, log, _ = card
    sess, backend = backend_over(session, log, tmp_path, monkeypatch)
    state, trace = backend.read_back(stop_grid=True)
    writes = [entry for entry in log if entry[0] == "write_l1"]
    stop = 0x20000 + CONSTANTS["KM_CONTROL_STOP"]
    assert sorted((x, y) for _, _, x, y, _, _ in writes) == sorted(
        (x, y) for x in range(12) for y in range(10)
    )
    assert {(address, tuple(values)) for _, _, _, _, address, values in writes} == {
        (stop, (1,))
    }
    assert log[len(writes) :] == ["wait_grid", "synchronize"]
    assert state.shape == (120, CONSTANTS["KM_CONTROL_BYTES"] // 4) and trace is None


def test_a_grid_that_does_not_return_raises_and_leaves_the_device_open(
    card, tmp_path, monkeypatch
):
    session, log, module = card
    sess, backend = backend_over(session, log, tmp_path, monkeypatch)
    module.grid_returns = False
    with pytest.raises(GridTimeout, match="reset"):
        backend.read_back()
    assert log == ["wait_grid"] and sess.stuck
    assert km.GridTimeout is GridTimeout
    sess.close()
    assert "close_device" not in log


def test_the_backend_reports_a_stopped_core_0_and_the_runtimes_error(
    card, tmp_path, monkeypatch
):
    session, log, _ = card
    sess, backend = backend_over(session, log, tmp_path, monkeypatch)
    report = backend.kernel_report()
    assert report.startswith("the dispatcher's core 0 is not running its loop")
    assert "state ready, in loop" in report and "mcause=0x7" in report
    backend.port.markers = (7, 9)
    assert "core 0 is running its loop" in backend.kernel_report()
    backend.port.cleaned = False
    log.clear()
    assert not backend.abort() and "stage=4" in backend.close_failed
    assert [entry[0] for entry in log[:120]] == ["write_l1"] * 120  # the grid first
    assert (
        log[120:] == ["wait_grid", "close"] and backend.aborted["x280_cores"][13] == 2
    )


def test_after_a_failure_the_backend_logs_the_kernels_state_lines(
    card, tmp_path, monkeypatch
):
    session, log, module = card
    sess, backend = backend_over(session, log, tmp_path, monkeypatch)
    lines, asked = [], []
    sess.log = lines.append
    module.dram_banks = lambda device: 8
    words = np.zeros((8, CONSTANTS["KMD_STATE_BYTES"] // 8), "<u8")
    words[5, :16] = np.arange(16)
    words[5, 2:4] = backend.port.lim_base  # args and lim: the kernel's bank

    def read(device, module, tensors, first=0, pages=None):  # the device's read
        asked.append((*tensors, first, pages))
        return [words.view(np.uint8)]

    monkeypatch.setattr(session, "read_tensors", read)
    backend.state_tensor = "state"
    backend.log_state()
    assert asked == [("state", 0, 8)]
    assert len(lines) == session.STATE_DUMP_LINES
    assert lines[1] == "kernel state +0x040: " + " ".join(
        hex(word) for word in range(8, 16)
    )
    words[5, 2] = 0  # no bank holds the kernel's lines
    backend.log_state()
    assert lines[-1] == "kernel state not found in DRAM (lines not flushed)"
    monkeypatch.setattr(session, "read_tensors", None)  # logged, not raised
    backend.log_state()
    assert lines[-1].startswith("kernel state not read (TypeError")


def test_a_base_off_the_16_byte_grid_is_rejected_before_the_launch(card, tmp_path):
    session, log, module = card
    FakeDispatch.lim_base = 0x10008
    try:
        with pytest.raises(StartError, match=r"multiple of 16.*lim_base 0x10008"):
            session.Session(device=object()).load(program(tmp_path))
    finally:
        FakeDispatch.lim_base = 0x10000
    assert log == ["prepare", "close"]
    bases = session.lim_bases(0x10000)
    assert bases["KMD_LIM_MAILBOX"] == 0x10000 + CONSTANTS["KMD_LIM_MAILBOX"]
    assert {"KMD_LIM_UPLOAD", "KMD_LIM_ACKS", "KM_ACK_STRIDE"} <= set(bases)
    session.check_bases(bases)
    with pytest.raises(StartError, match="control block 0x20004"):
        session.check_bases({"control block": 0x20004})


def test_compute_options_name_the_fidelity_and_the_unpack_mode_per_buffer(card):
    session, _, _ = card
    config = session.compute_config(
        {
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "enable_trisc2_rvv": True,  # not a constructor keyword
            "unpack_to_dest_mode": {3: "UnpackToDestFp32"},
        }
    )
    assert config["math_fidelity"] == "HiFi4"
    assert config["flags"] == [("fp32_dest_acc_en", True), ("enable_trisc2_rvv", True)]
    modes = config["unpack_to_dest_mode"]
    assert len(modes) == 64 and modes[3] == "UnpackToDestFp32"
    assert set(modes[:3] + modes[4:]) == {"Default"}
    every = session.compute_config({"unpack_to_dest_mode": "UnpackToDestFp32"})
    assert set(every["unpack_to_dest_mode"]) == {"UnpackToDestFp32"}
    assert "unpack_to_dest_mode" not in session.compute_config({})
    with pytest.raises(ProgramError, match="'fp32_dest_acc'"):  # a typo is rejected
        session.compute_config({"fp32_dest_acc": True})


def test_an_exit_without_a_stop_closes_the_runtime_before_the_device(
    card, tmp_path, monkeypatch
):
    session, log, module = card
    monkeypatch.setattr(session, "_hook_exit", lambda: None)  # not this process's
    said = []
    host_session = session.Session(log=said.append)
    host_session.open()
    host_session.load(program(tmp_path))
    module.dispatch.fail["close"] = RuntimeError("cleanup timed out")
    with pytest.raises(SystemExit) as caught:  # SIGTERM; its clean-up raises
        session._on_sigterm(signal.SIGTERM, None)
    assert (
        caught.value.code == 143
        and not host_session.run.stopped
        and host_session in session._LIVE
    )
    assert said[-1] == "SIGTERM: clean-up raised RuntimeError: cleanup timed out"
    del module.dispatch.fail["close"]
    log.clear()
    session._release_all("exit")  # the exit hook tries again, then closes the device
    assert log == ["abort", "close", "read_back, grid stopped", "close_device"]
    assert (
        host_session.run.stopped
        and host_session.device is None
        and host_session not in session._LIVE
    )


def test_the_chip_lock_holder_is_a_live_process_that_maps_the_lock(card, tmp_path):
    session = card[0]

    def lock(pid, size=56, name="lock"):
        path = tmp_path / f"{name}-{pid}"
        data = bytearray(size)
        if size >= 56:
            data[52:56] = pid.to_bytes(4, "little")
        path.write_bytes(bytes(data))
        return path

    def mapper(path, stay):
        """A process that maps `path` (as UMD maps its lock), then waits or ends."""
        code = (
            "import mmap, sys; f = open(sys.argv[1], 'r+b')"
            "; m = mmap.mmap(f.fileno(), 0); print(1, flush=True)"
            "; sys.stdin.read() if sys.argv[2] == '1' else None"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(path), str(int(stay))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        process.stdout.readline()
        return process

    path = lock(0, name="umd")
    holder = mapper(path, stay=True)
    path.write_bytes(lock(holder.pid).read_bytes())
    assert session.chip_holder(0, path) == holder.pid
    holder.stdin.close()
    holder.wait()
    zombie = mapper(path, stay=False)  # ended, not reaped yet: no mappings left
    stat, deadline = Path(f"/proc/{zombie.pid}/stat"), time.monotonic() + 10.0
    while stat.read_text().rsplit(")", 1)[1].split()[0] != "Z":
        assert time.monotonic() < deadline, "the mapping process did not end in 10 s"
        time.sleep(0.01)
    path.write_bytes(lock(zombie.pid).read_bytes())
    assert session.chip_holder(0, path) is None
    zombie.wait()
    assert session.chip_holder(0, path) is None  # reaped: the PID is gone
    assert session.chip_holder(0, lock(os.getppid())) is None  # alive, not mapping it
    assert session.chip_holder(0, lock(os.getpid())) is None  # this process
    assert session.chip_holder(0, lock(0)) is None  # no owner
    assert session.chip_holder(0, lock(os.getppid(), size=40)) is None  # not UMD's
    assert session.chip_holder(0, tmp_path / "missing") is None


def test_the_exit_hook_closes_the_device_when_the_stop_raises(
    card, tmp_path, monkeypatch
):
    session, log, module = card
    monkeypatch.setattr(session, "_hook_exit", lambda: None)  # not this process's
    said = []
    host_session = session.Session(log=said.append)
    host_session.open()
    host_session.load(program(tmp_path))
    closed = host_session.run.backend.abort
    host_session.run.backend.abort = (
        lambda: closed() and False
    )  # closed, but cluster 0 not restored
    log.clear()
    session._release_all("exit")  # the stop raises KernelFailed once the run is stopped
    assert said[-1].startswith("exit: clean-up raised KernelFailed")
    assert log == ["abort", "close", "read_back, grid stopped", "close_device"]
    assert (
        host_session.run.stopped
        and host_session.device is None
        and host_session not in session._LIVE
    )


def test_the_live_sessions_are_held_weakly(card):
    session = card[0]
    host_session = session.Session(log=lambda line: None)
    session._LIVE.add(host_session)
    assert host_session in session._LIVE
    del host_session
    gc.collect()
    assert len(session._LIVE) == 0
