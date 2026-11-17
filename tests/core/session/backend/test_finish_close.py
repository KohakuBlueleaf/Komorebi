"""A finish's close of the X280 runtime keeps what it could not restore, as an abort's
does: logged, and named by the stop (and every failed finish) as a reason the card may
need a reset."""

from types import SimpleNamespace

from komorebi.session import run as session_run
from komorebi.session.backend import backend as card_backend


class Port:
    def __init__(self, cleaned):
        self.cleaned = cleaned
        self.close_failed = "cluster 1: still_running_active_fw"

    def finish(self, timeout):
        return {"words": []}

    def close(self):
        return self.cleaned


def finished(cleaned):
    logged = []
    made = SimpleNamespace(port=Port(cleaned), card=SimpleNamespace(log=logged.append))
    card_backend._Backend.finish(made, 1.0)
    return made, logged


def test_a_finish_keeps_what_its_close_could_not_restore():
    made, logged = finished(False)
    assert made.close_failed == "cluster 1: still_running_active_fw"
    assert logged == [
        "x280 runtime closed (cleaned False): cluster 1: still_running_active_fw"
    ]
    made, logged = finished(True)
    assert made.close_failed == "" and logged == ["x280 runtime closed (cleaned True)"]


def test_the_stop_names_a_close_that_did_not_restore():
    h = {"KMD_ERROR_ENDED": 9, "KMD_STATE_DONE": 3}
    run = SimpleNamespace(h=h, backend=SimpleNamespace(close_failed="cluster 1: x"))
    run._close_failed = lambda: session_run.Run._close_failed(run)
    result = {"error": 0, "state": 3}
    wrong = session_run.Run._finish_problems(run, result)
    assert wrong == (
        "the X280 runtime could not restore its clusters at the close; the card may "
        "need a reset: cluster 1: x"
    )
    run.backend.close_failed = ""
    assert session_run.Run._finish_problems(run, result) == ""
