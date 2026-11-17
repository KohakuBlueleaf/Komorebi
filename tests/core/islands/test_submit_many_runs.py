"""IslandRun.submit_many with a dynamic RUN over a link among its rows (no card): the
RUN goes as submit, its links served first, the rows around it as submit_many; the
tickets one range in the rows' order, each island's commands queued in that order."""

from types import SimpleNamespace

import numpy as np

from komorebi.islands.run import IslandRun

RUN = 7  # the dynamic handler's number


class Part:
    """An island's Run: what was queued, in order."""

    def __init__(self, island, said):
        self.island, self.said, self.handles = island, said, 0

    def _words_built_for(self, handler, rows, how):
        return handler

    def submit(self, handler, group, args, **options):
        self.said.append((self.island, "submit", int(handler), tuple(group)))
        self.handles += 1
        return self.handles

    def submit_many(self, handlers, first, count, args, exclusive, part=False):
        handlers = np.broadcast_to(np.asarray(handlers), first.shape)
        for handler, low, n in zip(handlers, first, count):
            self.said.append((self.island, "many", int(handler), (int(low), int(n))))
        self.handles += len(first)
        return list(range(self.handles - len(first) + 1, self.handles + 1))


def island_run():
    def split(group):
        first, count = group
        out = []
        for island in (0, 1):
            low, high = max(first, 118 * island), min(first + count, 118 * island + 118)
            if high > low:
                out.append((island, low - 118 * island, high - low))
        return out

    graph = SimpleNamespace(
        ranks=236, islands=[0, 1], split=split, first=lambda i: 118 * i,
        count=lambda i: 118,
    )  # fmt: skip
    run = IslandRun.__new__(IslandRun)
    IslandRun.__init__(run, graph, [SimpleNamespace(program=None, backend=None)] * 2)
    said = []
    run.runs = (Part(0, said), Part(1, said))
    run.linked, run.crossing_line = True, "off"
    run._dynamic = lambda handler: int(handler) == RUN
    run._serve_links = lambda group: said.append(("links", tuple(group)))
    return run, said


def test_a_dynamic_run_among_the_rows_goes_as_submit_in_its_place():
    run, said = island_run()
    handlers = [3, RUN, 4, 5]
    firsts, counts = [0, 0, 110, 0], [10, 236, 20, 1]
    tickets = run.submit_many(handlers, firsts, counts)
    assert tickets == range(1, 5)
    assert said == [
        (0, "many", 3, (0, 10)),
        ("links", (0, 236)),
        (0, "submit", RUN, (0, 236)),
        (1, "submit", RUN, (0, 236)),
        (0, "many", 4, (110, 20)),
        (0, "many", 5, (0, 1)),
        (1, "many", 4, (110, 20)),
    ]
    assert [int(run._parts[1][t]) for t in tickets] == [0, 1, 2, 0]


def test_one_dynamic_run_alone_is_a_submit():
    run, said = island_run()
    assert run.submit_many(RUN, [0], [236]) == range(1, 2)
    assert said[0] == ("links", (0, 236)) and len(said) == 3
