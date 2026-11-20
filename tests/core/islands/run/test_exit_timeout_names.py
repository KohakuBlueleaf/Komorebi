"""A time-out of the stop's EXIT on an island run (no card): its group is every core of
the island in the run's own numbering, not flat ranks, so every core that did not
report is named, the ranks and link cores of island 1 by their own words."""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi import abi
from komorebi.errors import CommandTimeout
from komorebi.islands import run as island_run
from komorebi.session import state as device_state
from komorebi.session.parts.run_events import EventsMixin
from komorebi.session.parts.run_stop import StopMixin
from komorebi.session.run import Run

H = abi.headers()
RANKS, CORES = 78, 80  # each island: 78 ranks, core 78 its link core, 79 not a worker
EXIT, USER = 6, 5  # handles: the stop's EXIT, a command over both islands


class IslandOne:
    """Island 1's run as _fail_timeout reads it: flat ranks 78 on, its cores 0 on."""

    _fail_timeout = StopMixin._fail_timeout
    _raise_after_stop = StopMixin._raise_after_stop
    _remember = StopMixin._remember
    _close_failed = StopMixin._close_failed
    _log = StopMixin._log
    _own_part = EventsMixin._own_part
    _local = EventsMixin._local
    _far_given_up = EventsMixin._far_given_up
    _part = Run._part

    def __init__(self):
        self.h, self.backend, self.programs = H, SimpleNamespace(), []
        self.island_first, self.island_ranks = RANKS, RANKS
        self.island_cores, self.island_span = (0, RANKS), CORES
        self.cores, self._workers = CORES, CORES - 1  # a Tensix dispatcher's core
        self._link_base, self._cluster_base = H["KM_GROUP_LINK_BASE"], 1 << 15
        self._exit_handle, self._blocks, self._failure = EXIT, None, None
        blocks = np.zeros((CORES, H["KM_CONTROL_BYTES"] // 4), np.uint32)
        self._state = device_state.state_words(blocks, 0, H)  # none reported
        self.far_cores = lambda group: [{"core": 0, "far": True}]

    def group(self, handle):
        return (0, CORES) if handle == EXIT else (0, 2 * RANKS)

    def sequence(self, handle):
        return handle

    def _timeout_seen(self):
        return ""

    def _finish_kept(self, timeout):
        return None, None

    def _name_lines(self, error):
        pass


def timeout_of(handle):
    run = IslandOne()
    with pytest.raises(CommandTimeout) as raised:
        run._fail_timeout(SimpleNamespace(handle=handle, auxiliary=0))
    return run, raised.value


def test_the_stop_s_exit_names_every_core_of_the_island():
    run, error = timeout_of(EXIT)
    assert [core["core"] for core in error.cores] == list(range(CORES - 1))
    assert error.group == (0, CORES - 1) and error.own
    assert "group first 0 count 79 of its own cores (the stop's EXIT" in str(error)
    assert run._far_given_up(EXIT) == []  # no other island's part to read
    graph = SimpleNamespace(
        rank_cores=[list(range(RANKS))] * 2, link_cores=[[RANKS]] * 2
    )
    graph.first = lambda island: (0, RANKS)[island]
    islands = SimpleNamespace(graph=graph, _tickets=lambda island, handles: handles)
    islands._ranked = lambda island, cores: island_run.IslandRun._ranked(
        islands, island, cores
    )
    error = island_run.IslandRun._error(islands, error, 1)  # as island 1's run raised
    text = str(error)
    assert error.own and "of its own cores (the stop's EXIT" in text
    assert (
        "rank 78 (island 1, core 0)" in text and "rank 155 (island 1, core 77)" in text
    )
    assert "link core 0 (island 1, core 78)" in text


def test_a_command_s_flat_group_is_still_cut_to_the_island_s_part():
    run, error = timeout_of(USER)
    assert [core["core"] for core in error.cores] == list(range(RANKS))
    assert error.group == (0, 2 * RANKS) and not error.own
    assert "of its own cores" not in str(error)
    assert run._far_given_up(USER) == [{"core": 0, "far": True}]
