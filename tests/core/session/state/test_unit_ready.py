"""A Wormhole core whose compute unit is not ready, as the Tensix dispatcher reports it
(at its start in the result block, after its reset of a core in the give-up's block):
the session's error names the core and says the card needs a reset (no card)."""

from types import SimpleNamespace

import pytest

import komorebi as km
from komorebi import abi
from komorebi.session.parts.run_stop import unit_not_ready
from komorebi.session.run import Run

SAID = (
    "core 5's compute unit is not ready (status 0x11: the instruction FIFOs of "
    "threads 1, 2 are full); on Wormhole no reset by software clears this; the card "
    "needs a reset"
)


def test_the_text_names_the_core_its_status_and_the_full_fifos():
    assert unit_not_ready(5, 0x11) == SAID
    assert "(status 0x55: the instruction FIFO of thread 1 is full)" in (
        unit_not_ready(5, 0x55)
    )
    assert "(status 0x70: the instruction FIFOs of threads 0, 1, 2 are full)" in (
        unit_not_ready(5, 0x70)  # every FIFO full
    )


def test_the_start_raises_start_error_naming_the_core():
    h = abi.headers()
    result = {
        "error": h["KMD_ERROR_START"],
        "where": h["KMD_WHERE_UNIT"],
        "seen": 5 | 0x11 << 16,
        "result": 0,
    }

    def raised(error, stop_failed):
        raise error

    run = SimpleNamespace(
        h=h,
        _finish_kept=lambda timeout, stop_grid: (result, None),
        _finish_problems=lambda result: "error start in unit",
        _raise_after_stop=raised,
    )
    run._unit_start = lambda result: Run._unit_start(run, result)
    with pytest.raises(km.StartError) as said:
        Run._fail_start(run, "failed")
    assert str(said.value) == (
        f"{SAID}; the dispatcher did not report READY (state failed, core result 0): "
        "error start in unit; core 5's kernels cannot end: the session leaves the "
        "device open and the process's exit waits for them"
    )
    assert said.value.result is result


def test_the_finish_of_a_unit_start_does_not_wait_for_the_grid():
    h = abi.headers()
    words = [0] * h["KMD_R_WORDS"]
    words[h["KMD_R_STATE"]] = h["KMD_STATE_FAILED"]
    words[h["KMD_R_ERROR"]] = h["KMD_ERROR_START"]
    words[h["KMD_R_WHERE"]] = h["KMD_WHERE_UNIT"]
    words[h["KMD_R_SEEN"]] = 75 | 0x55 << 16
    done = []
    backend = SimpleNamespace(
        finish=lambda timeout: {"words": words},
        stop_grid=lambda: done.append("stop flags"),
        card=SimpleNamespace(mark_stuck=lambda: done.append("stuck")),
        release_l1=lambda: done.append("released"),
    )
    run = SimpleNamespace(h=h, backend=backend)
    run._unit_start = lambda result: Run._unit_start(run, result)
    run._read_back = lambda stop_grid: done.append("grid waited")
    Run._finish(run, 1.0, stop_grid=True)
    assert done == ["stop flags", "stuck"]  # no wait, no read, its L1 kept
    assert run._result["seen"] == 75 | 0x55 << 16
    words[h["KMD_R_WHERE"]] = h["KMD_WHERE_READY"]  # any other start: as before
    done.clear()
    Run._finish(run, 1.0, stop_grid=True)
    assert done == ["grid waited", "released"]


def test_a_reset_that_left_the_unit_not_ready_is_named_in_the_give_up():
    h = abi.headers()
    status = h["KMD_RESET_UNIT"] | 0x11 << 8
    run = SimpleNamespace(h=h, _timeout_cores=[], _reset_failed=(5, status))
    assert Run._timeout_seen(run) == f"after its reset {SAID}"
    other = SimpleNamespace(h=h, _timeout_cores=[], _reset_failed=(5, 3))
    assert Run._timeout_seen(other) == (
        "the dispatcher's reset of core 5 failed (NO_FIRMWARE)"
    )
