"""IslandRun.submit(handler, cluster=n) (no card): thread n, flat over islands
(KMX_THREAD(island, c) = 3 x island + c), takes the command on its island's run alone,
as that island's cluster c; one ticket of one part. A number no island serves is
refused, naming each island's numbers."""

from types import SimpleNamespace

import pytest

from komorebi.islands.graph import Graph, Island
from komorebi.islands.run import IslandRun
from komorebi.session.run import Run


class Part:
    """An island's Run: what was queued."""

    def __init__(self, island, said):
        self.island, self.said, self.handles = island, said, 0

    def submit(self, handler, group, args, exclusive, result, cluster=None, **options):
        self.said.append((self.island, handler, group, tuple(args), result, cluster))
        self.handles += 1
        return self.handles


def island_run():
    graph = Graph(
        [Island(tuple(range(8)), 4, 3), Island(tuple(range(8)), 4, 3)], [(0, 1)]
    )
    run = IslandRun.__new__(IslandRun)
    IslandRun.__init__(run, graph, [SimpleNamespace(program=None, backend=None)] * 2)
    said = []
    run.runs = (Part(0, said), Part(1, said))
    return run, said


def test_a_thread_number_goes_to_its_island_as_its_cluster():
    run, said = island_run()
    first = run.submit("fold", cluster=1, args=[5], result=True)
    second = run.submit("fold", cluster=5, args=[6], result=True)  # island 1's 2
    assert (first, second) == (1, 2)
    assert said == [
        (0, "fold", None, (5,), True, 1),
        (1, "fold", None, (6,), True, 2),
    ]
    assert [int(run._parts[i][first]) for i in (0, 1)] == [1, 0]
    assert [int(run._parts[i][second]) for i in (0, 1)] == [0, 1]


def test_a_number_no_island_serves_is_refused():
    run, said = island_run()
    refused = r"cluster 7: no thread .*island 1: \[4, 5, 6\]"
    with pytest.raises(ValueError, match=refused):
        run.submit("fold", cluster=7)
    with pytest.raises(ValueError, match="a thread's flat number"):
        run.submit("fold", cluster="1")
    assert said == []


def test_an_island_run_names_its_own_threads():
    """Island 1's run (thread numbers from 4): its cluster 1 is flat thread 4."""
    run = Run.__new__(Run)
    run._cluster_base, run._work_clusters, run.thread_program = 0xFF00, 3, None
    run.thread_first = 4
    assert run._to_thread(9, 1, None, False, False) == (9, (0xFF00 + 4, 1))
    run.thread_first = 1
    assert run._to_thread(9, 3, None, False, False) == (9, (0xFF00 + 3, 1))
