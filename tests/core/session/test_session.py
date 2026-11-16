"""The session's exit timeout, kernel buffer, state log, compute options, chip lock
holder and live sessions, over a fake native module."""

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fake_session import CONSTANTS, FakeDevice, load_session

from komorebi.device import native
from komorebi.errors import ProgramError
from komorebi.program import several
from komorebi.session import process, sizes, tensor_bytes
from komorebi.session.backend import backend as card_backend


@pytest.fixture
def card(monkeypatch, tmp_path):
    return load_session(monkeypatch, tmp_path)


def test_the_exit_timeout_is_set_only_when_the_environment_has_none(card):
    session, _, _ = card
    name, log = process.OP_TIMEOUT, []
    unset = {}
    assert process.exit_timeout(unset, log.append) == 30.0
    assert (
        unset[name] == "30"
        and len(log) == 1
        and process.exit_wait(unset) == "up to 30 s"
    )
    for given, timeout, phrase in (
        ("12.5", 12.5, "up to 12.5 s"),
        ("0", 0.0, "without a timeout"),
    ):
        env = {name: given}
        assert process.exit_timeout(env, log.append) == timeout
        assert env[name] == given and process.exit_wait(env) == phrase
    assert len(log) == 1  # a user's value, 0 included, is kept without a line


def test_the_kernel_buffer_is_passed_to_the_open_only_when_set(card, monkeypatch):
    session, _, module = card
    monkeypatch.setattr(process, "_hook_exit", lambda: None)  # not this process's
    session.Session().open()
    session.Session(kernel_bytes=128 << 10).open()
    assert module.opened == [None, 1572864 - 41088 - (128 << 10)]
    assert sizes.worker_l1_size(69 * 1024) == 1461120  # the runtime's default
    for bad in (100, 0, 1531776, "big"):
        with pytest.raises(ValueError, match="multiple of 64 from 64 to 1531712"):
            session.Session(kernel_bytes=bad)
    with pytest.raises(ValueError, match="opens the device"):
        session.Session(device=object(), kernel_bytes=128 << 10)


def backend_over(session, log, tmp_path, monkeypatch):
    """The real card backend over a closed fake dispatch and a stub readback."""
    sess = session.Session(device=FakeDevice(log), grid_timeout=0.5)
    sess._module = native.load_module()
    port = sess._module.Dispatch("dev", 1750)
    port.close()
    backend = sess._module.card_backend(sess._card(), port, 1350, 0)
    backend.control, backend.control_addr = "control", 0x20000
    words = (CONSTANTS["KM_CONTROL_BYTES"] + 64) // 4
    rows = np.zeros((120, words), np.int64)
    monkeypatch.setattr(tensor_bytes, "_read_uint32", lambda tensor: rows)
    log.clear()
    return sess, backend


def test_after_a_failure_the_backend_logs_the_kernels_state_lines(
    card, tmp_path, monkeypatch
):
    session, log, module = card
    sess, backend = backend_over(session, log, tmp_path, monkeypatch)
    lines, asked = [], []
    backend.card = backend.card._replace(log=lines.append)  # the session's log
    module.dram_banks = lambda device: 8
    words = np.zeros((8, CONSTANTS["KMD_STATE_BYTES"] // 8), "<u8")
    words[5, :16] = np.arange(16)
    words[5, 2:4] = backend.port.lim_base  # args and lim: the kernel's bank

    def read(device, module, tensors, first=0, pages=None):  # the device's read
        asked.append((*tensors, first, pages))
        return [words.view(np.uint8)]

    monkeypatch.setattr(tensor_bytes, "read_tensors", read)
    backend.state_tensor = "state"
    backend.log_state()
    assert asked == [("state", 0, 8)]
    assert len(lines) == card_backend.STATE_DUMP_LINES
    assert lines[1] == "kernel state +0x040: " + " ".join(
        hex(word) for word in range(8, 16)
    )
    words[5, 2] = 0  # no bank holds the kernel's lines
    backend.log_state()
    assert lines[-1] == "kernel state not found in DRAM (lines not flushed)"
    monkeypatch.setattr(tensor_bytes, "read_tensors", None)  # logged, not raised
    backend.log_state()
    assert lines[-1].startswith("kernel state not read (TypeError")


def test_compute_options_name_the_fidelity_and_the_unpack_mode_per_buffer(card):
    session, _, _ = card
    config = several.compute_config(
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
    every = several.compute_config({"unpack_to_dest_mode": "UnpackToDestFp32"})
    assert set(every["unpack_to_dest_mode"]) == {"UnpackToDestFp32"}
    assert "unpack_to_dest_mode" not in several.compute_config({})
    with pytest.raises(ProgramError, match="'fp32_dest_acc'"):  # a typo is rejected
        several.compute_config({"fp32_dest_acc": True})


def test_the_chip_lock_holder_is_a_live_process_that_maps_the_lock(
    card, tmp_path, monkeypatch
):
    monkeypatch.delenv(process.MOCK, raising=False)

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
    assert process.chip_holder(0, path) == holder.pid
    monkeypatch.setenv(process.MOCK, str(tmp_path / "cluster.yaml"))
    assert process.chip_holder(0, path) is None  # a mock device holds no chip
    monkeypatch.delenv(process.MOCK)
    holder.stdin.close()
    holder.wait()
    zombie = mapper(path, stay=False)  # ended, not reaped yet: no mappings left
    stat, deadline = Path(f"/proc/{zombie.pid}/stat"), time.monotonic() + 10.0
    while stat.read_text().rsplit(")", 1)[1].split()[0] != "Z":
        assert time.monotonic() < deadline, "the mapping process did not end in 10 s"
        time.sleep(0.01)
    path.write_bytes(lock(zombie.pid).read_bytes())
    assert process.chip_holder(0, path) is None
    zombie.wait()
    assert process.chip_holder(0, path) is None  # reaped: the PID is gone
    assert process.chip_holder(0, lock(os.getppid())) is None  # alive, not mapping it
    assert process.chip_holder(0, lock(os.getpid())) is None  # this process
    assert process.chip_holder(0, lock(0)) is None  # no owner
    assert process.chip_holder(0, lock(os.getppid(), size=40)) is None  # not UMD's
    assert process.chip_holder(0, tmp_path / "missing") is None


def test_the_live_sessions_are_held_weakly(card):
    session = card[0]
    host_session = session.Session(log=lambda line: None)
    process._LIVE.add(host_session)
    assert host_session in process._LIVE
    del host_session
    gc.collect()
    assert len(process._LIVE) == 0
