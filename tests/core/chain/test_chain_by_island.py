"""A write by island: each island's groups write their own copy from block 0 (an
add of two cards' partials into each card's copy). The groups found from each
group's write, the record's flag, each core's marks from its own island's roots,
the proof of a later read, and what is refused (no card)."""

import pytest

from komorebi.chain import calls as kc
from komorebi.chain import marks as chain_marks
from komorebi.dsl.chain import chain_calls

STARTS = (0, 8)  # two islands of 8 ranks


def test_each_group_is_counted_on_its_island_when_every_island_holds_the_same():
    assert chain_calls.island_groups((4, 8), 1, STARTS) == [0, 1, 2, 3] * 2
    assert chain_calls.island_groups((0, 16), 2, STARTS) == [0, 1, 2, 3] * 2
    assert chain_calls.island_groups((2, 8), 1, STARTS) == list(range(8))  # 6 and 2
    assert chain_calls.island_groups((6, 4), 4, STARTS) == [0]  # across the edge
    assert chain_calls.island_groups((4, 8), 1, (0,)) == list(range(8))
    assert chain_calls.first_on_island((2, 8), 1, STARTS, 7) == 6


def steps(by_island=STARTS, after_k=(0, 8)):
    """A: 8 ranks (4 an island) each write 2 tiles of s from block 0 on its island;
    B: groups of 2 read s by the share `after_k`; C reads s again (earlier)."""
    return [
        kc.Call(1, (0,), (4, 8), 1, 2, "s", name="A", by_island=by_island),
        kc.Call(2, (1,), (4, 8), 2, 1, "o", "s", k=after_k, name="B"),
        kc.Call(3, (2,), (4, 8), 2, 1, "p", "o", k=(0, 4), name="C",
                reads=(kc.Read("s", (0, 8)),)),  # fmt: skip
    ]


def test_each_core_waits_for_its_own_islands_roots():
    calls = steps()
    marks = chain_marks.marks(calls)[1]
    for core, (wanted, marked) in marks.items():
        island = kc.island_of(STARTS, core)
        assert wanted == len(marked) == 2  # 4 tiles of the share: 2 blocks there
        assert {kc.island_of(STARTS, root) for root in marked} == {island}
    assert kc.prove(calls)
    plain = chain_marks.marks(steps(None))[1]  # by the call's groups: blocks 0 to 7
    assert plain[4] == (2, [4, 5]) and plain[8][1] == [4, 5]


def test_the_record_carries_the_flag_and_the_handler_keeps_seven_bits():
    w = kc.words()
    calls = steps()
    info = kc.records(calls, 0x1000, handlers=[5, 6, 7])[:, w["kChainInfo"]]
    assert [int(word) >> w["kChainByIslandShift"] for word in info] == [1, 0, 0]
    assert int(info[0]) >> w["kChainHandlerShift"] & 0x7F == 5
    with pytest.raises(kc.ChainRefused, match="handler 128, 1 to 127"):
        kc.records(calls, 0x1000, handlers=[128, 6, 7])


def test_a_write_by_island_is_refused_unless_every_island_holds_the_same_groups():
    uneven = steps()
    uneven[0] = uneven[0]._replace(group=(2, 8))  # 6 groups on island 0, 2 on 1
    with pytest.raises(kc.ChainRefused, match=r"by island \[6, 2\]: every island"):
        kc.check_calls(uneven)
    across = steps()
    across[0] = across[0]._replace(group=(4, 8), size=8, width=16)
    with pytest.raises(kc.ChainRefused, match="group 0 across an edge"):
        kc.check_calls(across)
    direct = steps()
    direct[1] = direct[1]._replace(direct=kc.Direct(8, 2048))
    with pytest.raises(kc.ChainRefused, match="written by island, read direct"):
        kc.check_calls(direct)


def test_a_read_of_a_block_no_mark_told_its_island_of_is_refused():
    calls = steps()
    calls[2] = calls[2]._replace(reads=(kc.Read("s", (8, 8)),))
    assert kc.prove(calls)  # tiles 8 to 15: no block of A on either island
    wide = steps(after_k=(0, 2))  # B waits for block 0 only, C reads all 8 tiles
    with pytest.raises(kc.ChainRefused, match=r"tile 2, written by call 0 \(A\)"):
        kc.prove(wide)
