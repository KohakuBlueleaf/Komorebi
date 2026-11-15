"""Inspect and give-up: the protocol words, the block's decoding, run.inspect and a
command the dispatcher gave up while the run goes on, against a fake dispatcher."""

import numpy as np
import pytest
from fake_port import FakeBackend, FakePort

from komorebi import abi
from komorebi import state as device_state
from komorebi.errors import CommandTimeout, KomorebiError
from komorebi.run import Run

CONSTANTS = abi.headers().constants
GRID = (12, 10)


def make_run(port=None):
    port = port or FakePort()
    return Run(FakeBackend(port), GRID), port


def test_the_inspect_block_fits_the_dispatchers_lim_and_reads_inside_a_record():
    head, record = CONSTANTS["KMD_INSPECT_HEAD"], CONSTANTS["KMD_INSPECT_RECORD"]
    zeroed = CONSTANTS["KMD_LIM_INSPECT"] + head
    assert CONSTANTS["KMD_LIM_ZEROED"] == zeroed
    assert CONSTANTS["KMD_LIM_BYTES"] == zeroed + 128 * record
    state = CONSTANTS["KMD_INSPECT_STATE"] + 4 * CONSTANTS["KM_STATE_WORDS"]
    assert state <= CONSTANTS["KMD_INSPECT_LIVE"]
    live = CONSTANTS["KMD_INSPECT_LIVE"] + CONSTANTS["KM_LIVE_READ"]
    assert live <= CONSTANTS["KMD_INSPECT_TRACE"]
    trace = CONSTANTS["KMD_INSPECT_ENTRIES"] * CONSTANTS["KM_TRACE_ENTRY"]
    assert CONSTANTS["KMD_INSPECT_TRACE"] + trace <= record
    assert CONSTANTS["KM_TRACE_ENTRY"] == 4 * CONSTANTS["kTraceWords"]
    assert CONSTANTS["KMD_HANDLER_INSPECT"] > CONSTANTS["KM_HANDLER_USER_MAX"]


def test_the_live_words_and_the_abort_words_lie_in_the_control_block():
    slots = CONSTANTS["KM_RING_SLOTS"]
    abort, live = CONSTANTS["KM_CONTROL_ABORT"], CONSTANTS["KM_CONTROL_LIVE"]
    assert live == abort + 4 * slots and abort % 64 == 0
    assert live + CONSTANTS["KM_LIVE_BYTES"] == CONSTANTS["KM_CONTROL_ISLANDS"]
    assert CONSTANTS["kLvTrace"] == CONSTANTS["KM_LIVE_TRACE"]
    assert CONSTANTS["kLvTraceCap"] == CONSTANTS["KM_LIVE_TRACE_CAP"]
    assert 4 * (CONSTANTS["kLvTraceCap"] + 1) <= CONSTANTS["KM_LIVE_READ"]


def test_live_words_name_each_kernels_wait_and_command():
    words = np.zeros(CONSTANTS["KM_LIVE_READ"] // 4, np.uint32)
    compute = device_state.ROLES.index("unpack")
    words[CONSTANTS["kLvWhere"] + compute] = CONSTANTS["kWhereCInput"]
    words[CONSTANTS["kLvSequence"] + compute] = 9
    words[CONSTANTS["kLvGivenUp"]] = 2
    live = device_state.live(words)
    assert live["unpack"]["where"] == "cinput" and live["unpack"]["sequence"] == 9
    assert live["reader"]["where"] is None and live["given_up"] == 2
    names = device_state.where_names()
    for where in ("NocFlush", "CMailbox", "CConfirm", "Clear"):
        assert CONSTANTS[f"kWhere{where}"] in names


def test_a_block_without_its_mark_is_incomplete_and_an_unread_core_is_named():
    port = FakePort()
    port._inspect(CONSTANTS["KMD_INSPECT_HOST"], 5, 3, 2)
    block = port.inspect_block.copy()
    block[CONSTANTS["KMD_I_MARK"]] = 0
    record = CONSTANTS["KMD_INSPECT_HEAD"] // 4 + CONSTANTS["KMD_IR_FLAGS"]
    block[record + CONSTANTS["KMD_INSPECT_RECORD"] // 4] = 0
    decoded = device_state.inspect(block)
    assert not decoded["complete"] and decoded["kind"] == "host"
    assert decoded["first"] == 3 and decoded["count"] == 2
    assert decoded["cores"][0]["read"] and decoded["cores"][0]["magic_ok"]
    assert decoded["cores"][1] == {"core": 4, "read": False, "reset": False}


def test_inspect_names_the_wait_of_the_cores_that_wait_and_none_elsewhere():
    run, port = make_run()
    port.wheres = {5: (device_state.ROLES.index("reader"), CONSTANTS["kWherePort"])}
    cores = run.inspect((4, 3))
    assert [core["core"] for core in cores] == [4, 5, 6]
    assert cores[1]["live"]["reader"]["where"] == "port"
    assert all(core["live"]["reader"]["where"] is None for core in (cores[0], cores[2]))
    assert run.inspect_ticks == 1350 and port.submitted == []


def test_an_inspect_block_another_inspect_overwrote_is_refused():
    class Overwritten(FakePort):
        def _inspect(self, kind, req, first, count, slot=0, sequence=0):
            super()._inspect(CONSTANTS["KMD_INSPECT_TIMEOUT"], req + 1, first, count)

    run, _ = make_run(Overwritten())
    with pytest.raises(KomorebiError, match="inspect again"):
        run.inspect((0, 2))


def test_a_given_up_command_raises_and_the_next_one_on_its_cores_is_done():
    run, port = make_run()
    port.wheres = {1: (device_state.ROLES.index("writer"), CONSTANTS["kWhereData"])}
    stuck = run.submit(1, (0, 2))
    port.give_up_req = stuck
    with pytest.raises(CommandTimeout) as caught:
        run.wait(stuck)
    error = caught.value
    assert error.given_up and error.handle == stuck and "given up" in str(error)
    assert "core 1 writer waits in data" in str(error) and "unpack" not in str(error)
    assert [core["core"] for core in error.cores] == [0, 1]
    assert error.cores[1]["live"]["writer"]["where"] == "data"
    after = run.submit(1, (0, 2))
    assert run.wait(after).kind == "done"
    assert port.state == CONSTANTS["KMD_STATE_READY"]


def test_a_timeout_without_a_give_up_still_ends_the_run():
    run, port = make_run()
    stalled = run.submit(1, (0, 2))
    port.stall_req = stalled
    with pytest.raises(CommandTimeout) as caught:
        run.wait(stalled)
    assert not caught.value.given_up
    assert run.result()["error_name"] == "deadline"
