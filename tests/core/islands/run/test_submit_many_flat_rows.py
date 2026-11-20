"""IslandRun.submit_many with rows whose group spans islands (no card): each island
gets the rows that touch it, the flat group word unchanged and marked a part, as
submit sends a flat group; its own Run checks only its part of each row."""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi import abi
from komorebi.islands.run import IslandRun
from komorebi.session.run import Run

H = abi.headers()
CORES = 80  # a Wormhole grid, 8 by 10: ranks, then link cores, then the dispatcher
HANDLER = 3
SHAPES = {
    "2 islands": [78, 77],
    "8 islands": [77, 77, 76, 77, 76, 77, 76, 76],
    "31 islands": [75] * 15 + [76] * 16,
}


def island(first, ranks, total, sent):
    """An island's Run with no card: what _many_words and submit_many read; its
    holds over its own ranks; each block it sends recorded."""
    run = Run.__new__(Run)
    run.h = H
    run._link_base = H["KM_GROUP_LINK_BASE"]
    run._cluster_base = H.constants["KM_GROUP_CLUSTER_BASE"]
    run.cores, run.ranks = CORES, total
    run.island_first, run.island_ranks = first, ranks
    run.island_cores, run.island_span = (0, CORES), CORES
    run._holds, run.check_programs = [np.ones(ranks, bool)], True
    run.programs = [SimpleNamespace(name="Counts", handlers={}, port_handlers=())]
    run._nop, run._user_max = H["KM_HANDLER_NOP"], H["KM_HANDLER_USER_MAX"]
    run._flag_part = H.constants["KM_FLAG_PART"]
    run._flag_exclusive = H.constants["KM_FLAG_EXCLUSIVE"]
    run._hdr, run._group_at, run._arg_at = H.positions
    run._ports, run._next, run.half_added = {}, 1, {}
    run._writers, run._order_methods = {}, {}
    run._failure, run._stopped, run.submit_s = None, False, 0.0
    run._hand = lambda: None
    run._send_block = lambda words, *rest: sent.append((first, words.copy()))
    return run


def island_run(ranks):
    firsts = np.concatenate([[0], np.cumsum(ranks)[:-1]]).tolist()
    total = int(sum(ranks))

    def split(group):
        first, count = group
        parts = []
        for i, (low, n) in enumerate(zip(firsts, ranks)):
            span = min(first + count, low + n) - max(first, low)
            if span > 0:
                parts.append((i, max(first, low) - low, span))
        return parts

    graph = SimpleNamespace(
        ranks=total, islands=list(range(len(ranks))), split=split,
        first=lambda i: firsts[i], count=lambda i: ranks[i],
    )  # fmt: skip
    run = IslandRun.__new__(IslandRun)
    IslandRun.__init__(run, graph, [SimpleNamespace(program=None, backend=None)] * 2)
    sent = []
    run.runs = tuple(island(a, n, total, sent) for a, n in zip(firsts, ranks))
    run._parts = [np.zeros(1024, np.int64) for _ in run.runs]
    run.linked = False
    return run, sent, firsts, total


@pytest.mark.parametrize("ranks", SHAPES.values(), ids=SHAPES.keys())
def test_rows_over_islands_go_to_every_island_they_touch(ranks):
    run, sent, firsts, total = island_run(ranks)
    rows = 50
    first = np.zeros(rows, np.int64)
    count = np.full(rows, total)
    first[1], count[1] = firsts[1] + 2, 3  # one row inside island 1 alone
    words = np.arange(rows * 2, dtype=np.int64).reshape(rows, 2)
    tickets = run.submit_many(HANDLER, first, count, args=words)
    assert tickets == range(1, rows + 1)
    assert [island for island, _ in sent] == firsts  # one block an island
    hdr, group_at, arg_at = H.positions
    part = H.constants["KM_FLAG_PART"] << 8
    for island, block in sent:
        mine = np.arange(rows) != 1 if island != firsts[1] else np.ones(rows, bool)
        assert block.shape[0] == int(mine.sum())
        assert (block[:, hdr] & 0xFF == HANDLER).all() and (block[:, hdr] & part).all()
        assert (block[:, group_at] == first[mine] | count[mine] << 16).all()
        assert (block[:, arg_at[:2]] == words[mine]).all()


def test_a_row_whose_part_on_an_island_does_not_hold_the_program_is_refused():
    run, sent, firsts, total = island_run(SHAPES["8 islands"])
    run.runs[5]._holds[0][10] = False
    with pytest.raises(ValueError, match=r"group \(0, 612\) does not hold Counts"):
        run.submit_many(HANDLER, [0, 0], [total, total])
    assert sent == []  # refused before any island's rows were sent
