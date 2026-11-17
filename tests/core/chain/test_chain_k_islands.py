"""A share of K whose base steps by island (kc.IslandK): found from each group's
reads, its record's range words (word 1 the step), each core's tiles and marks from
its own island's base, and what is refused (no card)."""

from types import SimpleNamespace

import pytest

from komorebi.chain import calls as kc
from komorebi.chain import marks as chain_marks
from komorebi.dsl.chain import chain_calls

STARTS = (0, 8)  # two islands of 8 ranks


def spans(bases, ranks=16, size=4, k=8):
    """Each rank's (first, count): rank r of a group km::partition's share of K
    tiles from its island's base (`bases` by island)."""
    out = []
    for core in range(ranks):
        start, length = kc.partition(k, core % size, size)
        out.append((bases[kc.island_of(STARTS, core)] + start, length))
    return out


def test_the_share_steps_by_island_or_is_one_or_is_none():
    share = chain_calls.k_share(spans([0, 8]), 4, 0, STARTS)
    assert isinstance(share, kc.IslandK) and (share, share.step) == ((0, 8), 8)
    assert [share.at(core) for core in (0, 7, 8, 15)] == [(0, 8)] * 2 + [(8, 8)] * 2
    assert chain_calls.k_share(spans([4, 4]), 4, 0, STARTS) == (4, 8)  # as before
    assert chain_calls.k_share(spans([8, 0]), 4, 0, STARTS) is None  # steps back
    assert chain_calls.k_share(spans([0, 8]), 4, 2, STARTS) is None  # across an edge
    assert chain_calls.k_share(spans([0, 8]), 4) is None  # one island: refused


def steps(k):
    """A: rank g writes tile g of c; B: groups of 4 read c by the share `k`."""
    return [
        kc.Call(1, (0,), (0, 16), 1, 1, "c", name="A"),
        kc.Call(2, (1,), (0, 16), 4, 2, "d", "c", k=k, sends=True, name="B"),
    ]


def test_each_core_waits_for_its_islands_tiles_and_their_roots_mark_it():
    calls = steps(kc.IslandK(0, 8, 8, STARTS))
    kc.build(calls)
    for core, (wanted, marked) in chain_marks.marks(calls)[1].items():
        first = 8 * (core >= 8) + 2 * (core % 4)
        assert (wanted, marked) == (2, [first, first + 1]), core
    record = kc.records(calls, 0x1000, [1, 2])[1]
    w = kc.words()
    assert list(record[w["kChainRanges"] : w["kChainRanges"] + 2]) == [
        kc.Range(0, 0, 8).word(),
        8,
    ]
    plain = kc.records(steps((0, 8)), 0x1000, [1, 2])[1]
    assert plain[w["kChainRanges"] + 1] == 0  # a share as before: word 1 unused


def test_a_direct_input_by_island_and_a_class_made_without_it_are_refused():
    calls = steps(kc.IslandK(0, 8, 8, STARTS))
    calls[1] = calls[1]._replace(direct=kc.Direct(8, 4096))
    with pytest.raises(kc.ChainRefused, match="share of K by island"):
        kc.check_direct(calls)
    assert not chain_calls.k_islands(SimpleNamespace(defines={}))
    assert chain_calls.k_islands(SimpleNamespace(defines={"KM_CHAIN_K_ISLANDS": 1}))
