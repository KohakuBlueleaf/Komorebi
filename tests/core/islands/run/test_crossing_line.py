"""A (handler, group)'s first command on an island run: the links' counts read at the
submit, which does not wait, and again when its event is seen done (run.wait,
run.drain) under "measure"; what crossed said as one line, raised from that wait under
"error"; a tensor placed off the group's islands said at submit (no card)."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from komorebi.errors import CrossingWarning, KomorebiError
from komorebi.islands.graph import Placement
from komorebi.islands.records import Crossings
from komorebi.islands.run import IslandRun, _handler_label
from komorebi.program.description import Handler

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
    programs = (SimpleNamespace(),)  # a program with no dynamic handlers

    def _handler(self, handler):
        return 0, 0

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
    IslandRun.__init__(run, graph, [SimpleNamespace(program=None, backend=None)] * 2)
    run.runs = (Part(), Part())
    run.crossing_line = "measure"
    steps, reads, lines = [0], [], []
    run.session = SimpleNamespace(log=lines.append)

    def counts(islands=None):  # the islands of the command's group: island 0
        assert islands == (0,)
        reads.append(steps[0])
        return Crossings({(0, 0, 1): {**FIELDS, **{k: v * steps[0] for k, v in moved}}})

    run._counts = counts
    run._dynamic = lambda handler: False
    run.step = lambda: steps.__setitem__(0, steps[0] + 1)
    return run, reads, lines


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
        "other command in flight); if that is not meant, pass its tensors split=0, "
        "group=(0, 40) for cross's group, or copy=True if every rank reads all of it"
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


def test_check_the_default_reads_no_counts_and_log_still_measures():
    assert linked_run().crossing_line == "check"
    run, reads, lines = island_run([("pages", 8)])
    run.crossing_line = "check"
    run._crossed([run.submit(SimpleNamespace(name="cross"), (0, 8))])
    assert reads == [] and lines == []
    run.crossing_line = "log"  # the measured line, as before "measure" was named
    ticket = run.submit(SimpleNamespace(name="cross"), (0, 8))
    run.step()
    run._crossed([ticket])
    assert reads == [0, 1] and len(lines) == 1


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
    with pytest.raises(ValueError, match="'check', 'measure' \\('log'\\), 'error'"):
        run.submit(SimpleNamespace(name="gather"), (0, 16))


def test_the_line_names_a_handler_by_program_and_method():
    named = SimpleNamespace(name="Cross")
    method = SimpleNamespace(method=SimpleNamespace(name="cross"), program=named)
    assert _handler_label(method) == "Cross.cross"
    assert _handler_label(Handler(3, named)) == "Cross handler 3"
    assert _handler_label(7) == "handler 7"


class Joined(Part):
    """An island's Run whose part of a command over two islands checks nothing."""

    def _submit_words(self, *words):
        return None


def linked_run(log=None):
    """A linked IslandRun of two islands of 100 ranks (no counts): its notices."""
    graph = SimpleNamespace(
        ranks=200,
        islands=[0, 1],
        split=lambda g: [
            (i, max(g[0], low), min(g[0] + g[1], low + 100) - max(g[0], low))
            for i, low in ((0, 0), (1, 100))
            if g[0] < low + 100 and low < g[0] + g[1]
        ],
    )
    run = IslandRun.__new__(IslandRun)
    IslandRun.__init__(
        run, graph, [SimpleNamespace(program=None, backend=None)] * 2, linked=True
    )
    run.runs = (Joined(), Joined())
    run._dynamic = lambda handler: False
    if log is not None:
        run.session = SimpleNamespace(log=log.append, log_given=True)
    return run


def tensor(*parts, copy=False):
    """An island tensor's stand-in of 4 pages a step, island i holding `parts[i]`."""
    steps = sum(n for _, n in parts) or 1
    return SimpleNamespace(placement=Placement(tuple(parts), copy, steps, 4))


def attend(ports=False):
    """A class's handler attend(w, out) that reads w and writes out."""
    program = SimpleNamespace(
        name="Attn",
        moved_tensors={"attend": (frozenset({"w"}), frozenset({"out"}))},
        port_handlers={"attend"} if ports else set(),
        broadcast_lines={},
    )
    method = SimpleNamespace(name="attend", places=lambda: ["w", "out"])
    return SimpleNamespace(method=method, program=program)


def said(run, handler, group, args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run.submit(handler, group, args)
    return [str(w.message) for w in caught if w.category is CrossingWarning]


def test_a_linked_command_reading_another_islands_tensor_warns_once():
    run = linked_run()
    far, out = tensor((0, 0), (0, 2)), tensor((0, 1), (1, 0))
    assert said(run, attend(), (0, 40), [far, out]) == [
        "Attn.attend on ranks 0..39 (island 0) reads w, whose pages lie 8 on island 1: "
        "those a call touches on island 1 cross the link, at every call; if that is "
        "not meant, place w on the command's islands (split=0, group=(0, 40)), or "
        "copy=True if every rank reads all of it (run.crossing_line = 'off' silences "
        "this)"
    ]
    assert said(run, attend(), (0, 40), [far, out]) == []  # the repeated submit
    near = tensor((0, 2), (2, 0))
    assert said(run, attend(), (0, 40), [near, out]) == []  # its own island's


def test_a_group_island_with_no_page_is_said_and_a_log_given_takes_the_line():
    lines = []
    run = linked_run(lines)
    out = tensor((0, 1), (1, 1))
    assert said(run, attend(), (50, 100), [tensor((0, 2), (2, 0)), out]) == []
    assert lines == [
        "Attn.attend on ranks 50..149 (islands 0 and 1) reads w, whose pages lie 8 on "
        "island 0; its ranks on island 1 hold none of them and read them across the "
        "link, at every call; if that is not meant, place w on the command's islands "
        "(split=0, group=(50, 100)), or copy=True if every rank reads all of it "
        "(run.crossing_line = 'off' silences this)"
    ]


def test_off_a_copy_a_collective_and_an_unmoved_tensor_say_nothing():
    run = linked_run()
    run.crossing_line = "off"
    assert said(run, attend(), (0, 40), [tensor((0, 0), (0, 2)), None]) == []
    run = linked_run()
    copy = tensor((0, 1), (0, 1), copy=True)
    assert said(run, attend(), (0, 40), [copy, tensor((0, 1), (1, 0))]) == []
    reduce = attend(ports=True)  # its reduce brings island 0's pages to island 1
    assert said(run, reduce, (50, 100), [tensor((0, 2), (2, 0)), None]) == []
    unmoved = attend()
    unmoved.program.moved_tensors = {"attend": (frozenset(), frozenset({"out"}))}
    assert said(run, unmoved, (0, 40), [tensor((0, 0), (0, 2)), None]) == []
    lines = said(run, reduce, (40, 20), [tensor((0, 0), (0, 2)), None])
    assert len(lines) == 1 and "touches on island 1 cross" in lines[0]
