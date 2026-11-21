"""A grid wait the runtime refuses on a core's run mailbox (its control words
overwritten) is a grid that did not stop: the cores are marked stuck, the report names
them, the device is not closed in the normal way; any other failure still raises."""

import pytest
from fake_session import load_session

from komorebi import abi
from komorebi.errors import GridTimeout, KernelFailed
from komorebi.session import process
from komorebi.session import run as session_run
from komorebi.session.backend import backend as card_backend

# the runtime's refusal as the wait for the grid raised it on a Wormhole card
REFUSED = (
    "the wait for the grid's kernels failed: TT_FATAL @ llrt.cpp:347: run == "
    "run_state || run == tt_metal::dev_msgs::RUN_MSG_DONE\ninfo:\nRead unexpected "
    "run_mailbox value from core 24-25\nbacktrace:\n --- wait_until_cores_done"
)


class StandInRun:
    """A loaded run reduced to its own stop: Run.abort and Run._read_back over the
    session's real card backend."""

    abort = session_run.Run.abort
    _read_back = session_run.Run._read_back

    def __init__(self, backend):
        self.backend, self._stopped, self.h = backend, False, abi.headers()

    @property
    def stopped(self):
        return self._stopped

    def _thread_runs(self):
        return {}


def opened(card, monkeypatch, failure):
    """A session that opened its (stand-in) device, its run over the real card
    backend, and a wait for the grid that raises `failure`."""
    session, log, module = card
    monkeypatch.setattr(process, "_hook_exit", lambda log=None: None)
    lines, waits = [], []

    def wait_grid(device, grid_x, grid_y, timeout_s, first_row=0):
        waits.append(timeout_s)
        raise failure

    module.wait_grid = wait_grid
    sess = session.Session(log=lines.append, grid_timeout=0.5)
    sess.open()
    port = module.Dispatch(sess.device, 1750)
    backend = module.card_backend(sess._card(), port, 1350, 0)
    backend.control_addr = 0x20000
    sess.run = StandInRun(backend)
    log.clear()
    return sess, log, lines, waits


@pytest.fixture
def card(monkeypatch, tmp_path):
    return load_session(monkeypatch, tmp_path)


def test_a_refused_mailbox_takes_the_stuck_close(card, monkeypatch):
    sess, log, lines, waits = opened(card, monkeypatch, RuntimeError(REFUSED))
    given_up = KernelFailed("command 7 was given up by the dispatcher")
    with pytest.raises(GridTimeout) as raised, sess:
        raise given_up
    report = (
        "the grid did not stop: the runtime refused the run mailbox of at least core "
        "24-25 (raw value not in the runtime's message (it prints it to stderr only); "
        "neither GO nor DONE: its control words were overwritten); its wait ends at "
        "the first such core, so none of the 120 cores of rows 0 to 9 is confirmed "
        "stopped"
    )
    assert str(raised.value).startswith(report + ". The session does not close")
    assert raised.value.__context__ is given_up  # the give-up's text is kept
    assert not raised.value.__suppress_context__
    assert sess.stuck and sess.run.stopped and waits == [2.0, 0.5]
    assert lines.count(report) == 2  # the abort's wait and the read back's
    assert "grid still running" in lines
    assert lines[-1].startswith("device not closed by the session")
    assert "close_device" not in log and sess.device is not None


def test_another_failure_of_the_wait_still_raises(card, monkeypatch):
    other = RuntimeError("the wait for the grid's kernels failed: device lost")
    sess, log, lines, waits = opened(card, monkeypatch, other)
    with pytest.raises(RuntimeError, match="device lost") as raised, sess:
        raise KernelFailed("command 7 was given up by the dispatcher")
    assert raised.value is other and not sess.stuck
    assert "stopping the grid raised RuntimeError: " + str(other) in lines
    assert "close_device" in log  # not a refusal: the normal close


def test_the_report_names_each_core_and_a_value_the_message_holds():
    text = (
        "Read unexpected run_mailbox value: 0x21 (expected 0x80 or 0x0)\n"
        "Read unexpected run_mailbox value from core 24-25\n"
        "Read unexpected run_mailbox value: 0x20 (expected 0x80 or 0x0)\n"
        "Read unexpected run_mailbox value from core 25-21"
    )
    group = "the 64 cores of rows 0 to 7"
    assert card_backend.mailbox_refused(text, group) == (
        "the grid did not stop: the runtime refused the run mailbox of at least core "
        "24-25, 25-21 (raw value 0x21, 0x20; neither GO nor DONE: its control words "
        "were overwritten); its wait ends at the first such core, so none of the 64 "
        "cores of rows 0 to 7 is confirmed stopped"
    )
    ran_out = "waiting for physical cores to finish"
    assert card_backend.mailbox_refused(ran_out, group) is None
