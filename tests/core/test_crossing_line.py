"""A (handler, group)'s first command on an island run: the links' counts read at the
submit, which does not wait, and again when its event is seen done (run.wait,
run.drain); what crossed said as one line, raised from that wait under "error", or
not read under "off" (no card)."""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi.errors import KomorebiError
from komorebi.island_session import Crossings, IslandRun, _handler_label
from komorebi.program import Handler

FIELDS = dict.fromkeys(Crossings.FIELDS + Crossings.CYCLES, 0)


class Part:
    """An island's Run: commands queued, none filed until `finish`."""

    def __init__(self):
        self.handles, self._base, self._has = 0, 1, np.zeros(64, bool)

    def submit(self, handler, group, args, **options):
        self.handles += 1
        return self.handles

    def finish(self):
        self._has[: self.handles] = True

    port = None

    def _hand(self):
        pass

    def _store_handed(self):
        return []

    def drain(self, timeout):
        self.finish()
        return self.handles


def island_run(moved):
    """An IslandRun of two islands whose links' counts grow by `moved` a `step`."""
    graph = SimpleNamespace(
        ranks=236, split=lambda group: [(0, group[0], group[1])], islands=[0, 1]
    )
    run = IslandRun.__new__(IslandRun)
    IslandRun.__init__(run, graph, [SimpleNamespace(program=None)] * 2)
    run.runs = (Part(), Part())
    steps, reads, lines = [0], [], []
    run.session = SimpleNamespace(log=lines.append)

    def counts():
        reads.append(steps[0])
        return Crossings({(0, 0, 1): {**FIELDS, **{k: v * steps[0] for k, v in moved}}})

    run._counts = counts
    run._dynamic = lambda handler: False
    run.step = lambda: steps.__setitem__(0, steps[0] + 1)
    return run, reads, lines


@pytest.fixture(autouse=True)
def no_reduce_check(monkeypatch):
    monkeypatch.setattr("komorebi.island_session._check_reduce_islands", lambda *a: 0)


def test_the_submit_does_not_wait_and_the_line_comes_at_the_wait():
    run, reads, lines = island_run([("pages", 40), ("puts", 80)])
    handler = SimpleNamespace(name="cross")
    ticket = run.submit(handler, (0, 40))
    assert reads == [0] and lines == []  # read before; the command not waited
    run.submit(handler, (0, 40))
    assert reads == [0]  # the same (handler, group): not read
    run.step()
    run._crossed([ticket], "events")
    assert lines == [
        "cross on ranks 0..39 moved 40 pages across the link (asks 0, puts 80) (with 1 "
        "other command in flight); if that is not meant, see its tensors' group="
    ]
    run._crossed([ticket])
    assert len(lines) == 1 and reads == [0, 1]  # said once


def test_drain_says_it_and_an_earlier_command_not_filed_is_in_flight():
    run, reads, lines = island_run([("asks", 2), ("pages", 8)])
    run.submit(SimpleNamespace(name="a"), (0, 8))
    run.submit(SimpleNamespace(name="b"), (8, 8))
    run.step()
    run.drain()
    assert reads == [0, 0, 1]  # both read before, one read after for both
    assert [line.split(" (asks")[0] for line in lines] == [
        "a on ranks 0..7 moved 8 pages across the link",
        "b on ranks 8..15 moved 8 pages across the link",
    ]
    assert all("(with 1 other command in flight)" in line for line in lines)
    run.submit(SimpleNamespace(name="c"), (16, 8))  # a and b filed: nothing out
    run.step()
    run.drain()
    assert "in flight" not in lines[-1]


def test_nothing_crossed_is_not_said_and_off_or_messages_read_nothing():
    run, reads, lines = island_run([])
    ticket = run.submit(SimpleNamespace(name="local"), (0, 8))
    run._crossed([ticket])
    assert reads == [0, 0] and lines == []
    for setting in ("off", "linked"):
        run, reads, lines = island_run([("pages", 8)])
        run.crossing_line = "off" if setting == "off" else "log"
        run.linked = setting == "linked"
        run._crossed([run.submit(SimpleNamespace(name="cross"), (0, 8))])
        assert reads == [] and lines == []


def test_error_is_raised_by_the_wait_with_the_events():
    run, _, lines = island_run([("asks", 3), ("pages", 24)])
    run.crossing_line = "error"
    ticket = run.submit(SimpleNamespace(name="gather"), (0, 8))
    run.step()
    with pytest.raises(
        KomorebiError, match=r"moved 24 pages .*\(ticket 1 has run\)"
    ) as e:
        run._crossed([ticket], ["event"])
    assert e.value.events == ["event"] and lines == []
    run.crossing_line = "loud"
    with pytest.raises(ValueError, match="'log', 'error' or 'off'"):
        run.submit(SimpleNamespace(name="gather"), (0, 16))


def test_the_line_names_a_handler_by_program_and_method():
    named = SimpleNamespace(name="Cross")
    method = SimpleNamespace(method=SimpleNamespace(name="cross"), program=named)
    assert _handler_label(method) == "Cross.cross"
    assert _handler_label(Handler(3, named)) == "Cross handler 3"
    assert _handler_label(7) == "handler 7"
