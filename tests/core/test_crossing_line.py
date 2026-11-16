"""A (handler, group)'s first command on an island run: the links' counts read before
it and after its wait, and what crossed said as one line, raised after the command
under crossing_line "error", or not read under "off" (no card)."""

from types import SimpleNamespace

import pytest

from komorebi.errors import KomorebiError
from komorebi.island_session import Crossings, IslandRun

FIELDS = dict.fromkeys(Crossings.FIELDS + Crossings.CYCLES, 0)


class Part:
    def __init__(self):
        self.handles = 0

    def submit(self, handler, group, args, **options):
        self.handles += 1
        return self.handles


def island_run(moved):
    """An IslandRun of two islands whose links' counts grow by `moved` at each wait."""
    graph = SimpleNamespace(
        ranks=236, split=lambda group: [(0, group[0], group[1])], islands=[0, 1]
    )
    run = IslandRun.__new__(IslandRun)
    IslandRun.__init__(run, graph, [SimpleNamespace(program=None)] * 2)
    run.runs = (Part(), Part())
    done, reads, lines = [0], [], []
    run.session = SimpleNamespace(log=lines.append)

    def counts():
        reads.append(done[0])
        return Crossings({(0, 0, 1): {**FIELDS, **{k: v * done[0] for k, v in moved}}})

    def wait(tickets, timeout=10.0):
        done[0] += 1
        return ("done", tickets)

    run._counts, run.wait = counts, wait
    run._dynamic = lambda handler: False
    return run, reads, lines


@pytest.fixture(autouse=True)
def no_reduce_check(monkeypatch):
    monkeypatch.setattr("komorebi.island_session._check_reduce_islands", lambda *a: 0)


def test_a_first_use_says_what_crossed_as_a_fact_once():
    run, reads, lines = island_run([("pages", 40), ("puts", 40)])
    handler = SimpleNamespace(name="cross")
    ticket = run.submit(handler, (0, 40))
    assert reads == [0, 1]  # one read before, one after its wait
    assert lines == [
        "cross on ranks 0..39 moved 40 pages across the link (asks 0, puts 40); "
        "if that is not meant, see its tensors' group="
    ]
    run.submit(handler, (0, 40))
    assert reads == [0, 1] and len(lines) == 1  # the same (handler, group): not read
    del run.wait  # IslandRun.wait: the first command's events, waited at its use
    assert run.wait(ticket) == ("done", ticket) and not run._early


def test_nothing_crossed_is_not_said_and_off_reads_nothing():
    run, reads, lines = island_run([])
    run.submit(SimpleNamespace(name="local"), (0, 8))
    assert reads == [0, 1] and lines == []
    run, reads, lines = island_run([("pages", 8)])
    run.crossing_line = "off"
    run.submit(SimpleNamespace(name="cross"), (0, 8))
    assert reads == [] and lines == []


def test_error_is_raised_after_the_command_ran_and_its_events_kept():
    run, _, lines = island_run([("asks", 3), ("pages", 24)])
    run.crossing_line = "error"
    with pytest.raises(KomorebiError, match=r"moved 24 pages .* ticket 1 has run"):
        run.submit(SimpleNamespace(name="gather"), (0, 8))
    assert run._early[1] == ("done", 1) and lines == []
    run.crossing_line = "loud"
    with pytest.raises(ValueError, match="'log', 'error' or 'off'"):
        run.submit(SimpleNamespace(name="gather"), (0, 16))
