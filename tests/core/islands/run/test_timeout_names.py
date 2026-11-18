"""A given-up command of an island run names the cores of every island that held a
part: the other islands' parts as their own dispatchers' inspects read them; a
timeout takes another island's named give-up (MessagesGaveUp)."""

import time
from types import SimpleNamespace

from komorebi import abi
from komorebi.errors import CommandTimeout, KomorebiError, MessagesGaveUp, describe_core
from komorebi.islands import run as island_run
from komorebi.session import messages_gave
from komorebi.session import state as device_state

CONSTANTS = abi.headers().constants


def fake_run(local, block):
    backend = SimpleNamespace(inspect=lambda count: {"block": block, "given_up": 0})
    return SimpleNamespace(_local=lambda group: local, backend=backend, h=None)


def test_a_far_part_s_cores_are_named_with_their_island(monkeypatch):
    word = CONSTANTS["kGaveRefused"] | CONSTANTS["kWhereAsk"] << 8 | 0 << 24
    refused = device_state.gave_up(word, 2)
    blocks = {
        "island 0": {
            "complete": True,
            "kind": "timeout",
            "first": 118,
            "count": 1,
            "cores": [{"core": 118, "writer": refused}],
        }
    }
    monkeypatch.setattr(device_state, "inspect", lambda block, h: blocks[block])
    graph = SimpleNamespace(
        split=lambda group: [(0, 118, 1), (1, 119, 1)],
        rank_cores=[list(range(119)), list(range(119))],
        link_cores=[[119], [119]],
        first=lambda island: (0, 119)[island],
    )
    run = SimpleNamespace(
        graph=graph,
        runs=(fake_run((118, 1), "island 0"), fake_run((0, 1), "island 1")),
    )
    run._ranked = lambda island, cores: island_run.IslandRun._ranked(run, island, cores)
    far = island_run.IslandRun._far_cores(run, 1, (118, 2))
    assert [(core["rank"], core["island"], core["far"]) for core in far] == [
        (118, 0, True)
    ]
    assert describe_core(far[0]).startswith(
        "rank 118 (island 0, core 118) writer gave up (refused) in ask: a page line "
        "toward its own island 0"
    )
    assert island_run.IslandRun._ranked(run, 1, far) == far  # kept as it is
    blocks["island 0"]["count"] = 2  # another inspect since: not this part's
    assert island_run.IslandRun._far_cores(run, 1, (118, 2)) == []


def test_a_timeout_takes_another_island_s_named_give_up():
    named = MessagesGaveUp(3, 0, 5, (110, 16), [], gave={"core": 2, "text": "quota"})

    def drains(raised):
        def drain(timeout):
            raise raised

        return SimpleNamespace(drain=drain)

    plain = CommandTimeout(1, 0, 5, (110, 16), [])
    run = SimpleNamespace(runs=(drains(plain), drains(named)))
    gave = island_run.IslandRun._gave_elsewhere
    assert gave(run, plain, 0, time.monotonic() + 1) == (named, 1)
    run.runs = (drains(plain), drains(plain))
    assert gave(run, plain, 0, time.monotonic() + 1) == (plain, 0)
    stopped = KomorebiError("stopped")
    assert gave(run, stopped, 0, time.monotonic() + 1) == (stopped, 0)


def test_a_raised_timeout_s_ticket_is_kept_for_the_stop():
    plain = CommandTimeout(4, 0, 5, (110, 16), [])
    run = SimpleNamespace(runs=(), _timed_out=set())
    run._error = lambda error, island: error
    run._gave_elsewhere = lambda error, island, deadline: (error, island)
    raised = island_run.IslandRun._raised(run, plain, 0, time.monotonic() + 1)
    assert raised is plain and run._timed_out == {4}
    island_run.IslandRun._raised(run, KomorebiError("x"), 0, time.monotonic() + 1)
    assert run._timed_out == {4}


def test_an_x280_rank_s_give_up_is_named_by_its_index():
    h = abi.headers()
    words = [0] * 28
    words[h["KMD_H_MESSAGES_GAVE"]] = 1 + h["KM_CORES_MAX"] - h["KM_X280_RANKS_MAX"]
    gave = messages_gave.named({"x280_cores": words}, None, h)
    assert gave["x280"] == 0 and gave["core"] == 124
    error = MessagesGaveUp(3, 0, 5, (106, 26), [], gave=gave | {"island": 0})
    assert "given up by X280 rank 0 (island 0)" in str(error)
