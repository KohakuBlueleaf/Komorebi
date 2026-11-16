"""A given-up command of an island run names the cores of every island that held a
part: the other islands' parts as their own dispatchers' inspects read them."""

from types import SimpleNamespace

from komorebi import abi, island_session
from komorebi import state as device_state
from komorebi.errors import describe_core

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
    run._ranked = lambda island, cores: island_session.IslandRun._ranked(
        run, island, cores
    )
    far = island_session.IslandRun._far_cores(run, 1, (118, 2))
    assert [(core["rank"], core["island"], core["far"]) for core in far] == [
        (118, 0, True)
    ]
    assert describe_core(far[0]).startswith(
        "rank 118 (island 0, core 118) writer gave up (refused) in ask: a page line "
        "toward its own island 0"
    )
    assert island_session.IslandRun._ranked(run, 1, far) == far  # kept as it is
    blocks["island 0"]["count"] = 2  # another inspect since: not this part's
    assert island_session.IslandRun._far_cores(run, 1, (118, 2)) == []
