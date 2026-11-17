"""A gather in a chain (km.all_gather's record) on the host: on one island a plain
record of one rank, the same marks and counts; over islands each island's first rank
of it waits for its own island's share and marks its own island's cores only; its
record and what is refused (no card)."""

import pytest

from komorebi import abi
from komorebi.chain import calls as kc
from komorebi.chain import marks as chain_marks
from komorebi.session import state

GATHER = 9  # the walk's handler number (any here)


def steps(starts, ranks, gather_cores, width=2, gathered=True):
    """A: rank g writes tiles g x width on of c; the gather of c's tiles (or none);
    B: every rank reads all of c's tiles, its own tile of d."""
    tiles = ranks * width
    calls = [kc.Call(1, (0,), (0, ranks), 1, width, "c", name="A")]
    if gathered:
        calls.append(
            kc.Call(
                GATHER,
                (0, 0, tiles),
                gather_cores,
                gather_cores[1],
                tiles,
                "c",
                "c",
                ranges=(kc.Range(0, 0, tiles),),
                gather=starts,
                name="gather",
            )
        )
    calls.append(
        kc.Call(2, (1,), (0, ranks), 1, 1, "d", "c", ranges=(kc.Range(0, 0, tiles),))
    )
    return calls


def test_one_island_marks_and_counts_as_a_plain_record():
    gathered = steps((0,), 16, (0, 1))
    plain = [*gathered[:1], gathered[1]._replace(gather=None), *gathered[2:]]
    kc.build(gathered)
    marks = chain_marks.marks(gathered)
    assert marks == chain_marks.marks(plain)
    assert marks[1] == {0: (16, list(range(16)))}  # every block of A, its one rank
    assert marks[2] == {core: (1, [0]) for core in range(16)}
    info = kc.records(gathered, 0x1000, [1, GATHER, 2])[1][11]
    assert info & kc.words()["kChainGather"] == 0  # no gather bit on one island


def test_two_islands_each_leader_its_share_and_its_own_cores():
    calls = steps((0, 8), 16, (7, 2))
    kc.build(calls)
    marks = chain_marks.marks(calls)
    assert marks[1] == {7: (8, list(range(8))), 8: (8, list(range(8, 16)))}
    assert marks[2] == {core: (1, [7 if core < 8 else 8]) for core in range(16)}
    record = kc.records(calls, 0x1000, [1, GATHER, 2])[1]
    w = kc.words()
    assert record[10] == 7 | 2 << 16  # island 0's last rank and island 1's first
    assert record[11] & w["kChainGather"]
    assert (record[11] >> w["kChainSizeShift"]) & w["kChainSizeMask"] == 2
    assert (record[11] >> w["kChainWidthShift"]) & w["kChainWidthMask"] == 0
    assert record[12] == 32 << 20  # range 0: the gathered tiles 0 to 31 (its width)


def test_a_gather_over_islands_takes_pages_past_255_in_its_range_word():
    calls = steps((0, 8), 16, (7, 2), width=42)  # 672 tiles: 336 an island
    kc.build(calls)
    marks = chain_marks.marks(calls)
    assert marks[1] == {7: (8, list(range(8))), 8: (8, list(range(8, 16)))}
    assert marks[2] == {core: (1, [7 if core < 8 else 8]) for core in range(16)}
    record = kc.records(calls, 0x1000, [1, GATHER, 2])[1]
    w = kc.words()
    assert (record[11] >> w["kChainWidthShift"]) & w["kChainWidthMask"] == 0
    assert record[12] == 672 << 20


def test_a_line_of_three_each_island_its_third():
    calls = steps((0, 8, 16), 24, (7, 10))
    kc.build(calls)
    marks = chain_marks.marks(calls)
    leaders = {7: range(0, 8), 8: range(8, 16), 16: range(16, 24)}
    assert marks[1] == {
        core: (8, list(leaders[core])) if core in leaders else (0, [])
        for core in range(7, 17)
    }
    for core in range(24):
        island = 0 if core < 8 else 1 if core < 16 else 2
        assert marks[2][core] == (1, [(7, 8, 16)[island]])


def test_a_read_further_on_is_proven_after_the_gather():
    calls = steps((0, 8), 16, (7, 2))
    calls.append(kc.Call(3, (2,), (0, 16), 1, 1, "e", "d", ranges=(kc.Range(0, 1, 1),)))
    calls[-1] = calls[-1]._replace(reads=(kc.Read("c", ranges=(kc.Range(0, 0, 32),)),))
    kc.build(calls)  # c known complete on every island through its own leader


def test_a_root_writing_another_islands_share_is_refused():
    calls = steps((0, 4), 16, (3, 2))  # roots 4 to 7 (island 1) write share 0's tiles
    with pytest.raises(kc.ChainRefused, match=r"root 4 \(island 1\) writes tiles 8"):
        kc.build(calls)


def test_a_read_past_the_gathered_pages_is_refused():
    calls = steps((0, 8), 16, (7, 2))
    calls[1] = calls[1]._replace(ranges=(kc.Range(0, 0, 16),), width=16)
    with pytest.raises(kc.ChainRefused, match="read the gathered pages only"):
        kc.build(calls)


def test_two_gathers_of_one_copy_are_refused():
    calls = steps((0, 8), 16, (7, 2))
    calls += [calls[0], calls[1], calls[2]]
    with pytest.raises(kc.ChainRefused, match="gather into another copy"):
        kc.build(calls)


def test_a_gather_needs_a_rank_of_every_island():
    calls = steps((0, 8, 16), 24, (7, 2))  # no rank of island 2
    with pytest.raises(kc.ChainRefused, match="hold no rank of island 2"):
        kc.build(calls)


def test_a_gathers_give_up_names_its_leader_and_the_share():
    h = abi.headers()
    word = h["kGaveDeadline"] | h["kWherePort"] << 8 | state.GATHER_SLOT << 16
    gave = state.gave_up(word, 3 | 1 << 8 | 0 << 16 | 1 << 24)
    assert gave["gather"] == {
        "record": 3,
        "island": 1,
        "waited for": "share 0's pages from its parent",
    }
