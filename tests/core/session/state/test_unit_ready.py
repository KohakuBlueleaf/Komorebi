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
    assert "(status 0x55: the instruction FIFOs of threads 1 are" in (
        unit_not_ready(5, 0x55)
    )
    assert "(status 0x70: the instruction FIFOs of threads 0, 1, 2 are" in (
        unit_not_ready(5, 0x70)  # every FIFO full
    )


def test_the_start_raises_start_error_naming_the_core():
    h = abi.headers()
    result = {"where": h["KMD_WHERE_UNIT"], "seen": 5 | 0x11 << 16, "result": 0}

    def raised(error, stop_failed):
        raise error

    run = SimpleNamespace(
        h=h,
        _finish_kept=lambda timeout, stop_grid: (result, None),
        _finish_problems=lambda result: "error start in unit",
        _raise_after_stop=raised,
    )
    with pytest.raises(km.StartError) as said:
        Run._fail_start(run, "failed")
    assert str(said.value) == (
        f"{SAID}; the dispatcher did not report READY (state failed, core result 0): "
        "error start in unit"
    )
    assert said.value.result is result


def test_a_reset_that_left_the_unit_not_ready_is_named_in_the_give_up():
    h = abi.headers()
    status = h["KMD_RESET_UNIT"] | 0x11 << 8
    run = SimpleNamespace(h=h, _timeout_cores=[], _reset_failed=(5, status))
    assert Run._timeout_seen(run) == f"after its reset {SAID}"
    other = SimpleNamespace(h=h, _timeout_cores=[], _reset_failed=(5, 3))
    assert Run._timeout_seen(other) == (
        "the dispatcher's reset of core 5 failed (NO_FIRMWARE)"
    )
