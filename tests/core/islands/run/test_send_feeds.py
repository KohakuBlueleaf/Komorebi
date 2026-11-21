"""A wait on joined islands feeds every island that has work for the card (no card):
the island serving a crossing command's pages, with its own commands queued, is
stored though the waited ticket has no part there; an idle island is not called;
wait_together and a transfer on one island's run store every island as well."""

from types import SimpleNamespace

import numpy as np

from komorebi.islands.parts.calls import CallsMixin
from komorebi.islands.run import IslandRun
from komorebi.session.run import Run


class Island:
    """An island's Run as send() and _store() see it: queued commands (_counts),
    the port's stored-not-sent count, and what was handed, stored and stepped."""

    def __init__(self, queued=0, counts=True):
        self.port = SimpleNamespace(queued=0)
        if counts:
            self._counts = [1] * queued
        self.calls = []
        self.done = set()

    def _hand(self):
        self.calls.append("hand")
        if getattr(self, "_counts", None):
            self.port.queued += len(self._counts)
            self._counts = []

    def _store_handed(self):
        self.calls.append("store")
        self.port.queued = 0
        return ()

    def _arrived(self, handles):
        return np.isin(handles, list(self.done))

    def _check_alive(self):
        pass

    def _step(self, handles, upto, left):
        self.calls.append("step")
        self.done.update(handles.tolist())


def islands(*runs):
    run = IslandRun.__new__(IslandRun)
    run.runs = runs
    run._parts = np.array([[0, 1], [0, 0], [0, 0]])  # ticket 1: island 0 only
    return run


def test_the_island_serving_a_crossing_command_s_pages_is_stored_by_the_wait():
    reader, server, idle = Island(queued=1), Island(queued=1), Island()
    run = islands(reader, server, idle)
    run.send()  # island 0 reads island 1's pages; island 1's serving command queued
    assert reader.calls == ["hand", "store"] and server.calls == ["hand", "store"]
    assert idle.calls == [] and server.port.queued == 0


def test_an_island_whose_ring_still_holds_handed_commands_is_stored():
    reader, server, idle = Island(queued=1), Island(), Island()
    server.port.queued = 2  # handed earlier, the ring was full: not yet stored
    islands(reader, server, idle).send()
    assert server.calls == ["hand", "store"] and server.port.queued == 0


def test_a_stand_in_without_counts_is_fed_as_before():
    reader, other = Island(queued=1), Island(counts=False)
    run = islands(reader, other, Island())
    run.send()
    assert other.calls == ["hand", "store"]


def test_the_store_steps_every_island_holding_a_part_until_it_arrived():
    reader, server, idle = Island(), Island(), Island()
    run = islands(reader, server, idle)
    run._store(np.array([1]), deadline=float("inf"))
    assert reader.calls == ["step"] and reader.done == {1}
    assert server.calls == [] and idle.calls == []


class Waited(Island):
    """An island whose own commands are waited for: its events after the steps."""

    def wait(self, handles, timeout):
        self.calls.append("wait")
        return [f"event {handle}" for handle in handles]


def test_a_wait_on_some_islands_stores_every_island_s_queue_first():
    """A crossing command queued on islands 0 and 2, then a wait on island 0 alone:
    island 2's part is stored too (before, only island 0 was handed and stepped)."""
    waited, idle, crossing = Waited(queued=2), Island(), Island(queued=1)
    session = SimpleNamespace(run=islands(waited, idle, crossing))
    events = CallsMixin.wait_together(session, [(0, 7)])
    assert events == ["event 7"]
    assert waited.calls == ["hand", "store", "step", "wait"]
    assert crossing.calls == ["hand", "store"] and crossing.port.queued == 0
    assert idle.calls == []


def test_a_transfer_on_one_island_s_run_stores_every_island_before_its_wait():
    calls = []
    run = Run.__new__(Run)
    run._group = lambda group: group
    run._queue_command = lambda *a, **k: calls.append("queue") or 5
    run.wait = lambda handle, timeout: calls.append(f"wait {handle}")
    run.send_islands = lambda: calls.append("send")
    run.transfer(1, (0, 2, 0), (), 1.0)
    assert calls == ["send", "queue", "wait 5"]
    run.send_islands, calls[:] = None, []  # one card: no send
    run.transfer(1, (0, 2, 0), (), 1.0)
    assert calls == ["queue", "wait 5"]
