"""Direct messages: `direct=` as offsets, pairs or a mapping, what the load refuses,
and each core's tables pointing at its partner's slot and credit line."""

import pytest

from komorebi.program.description import Dynamic


def test_offsets_pairs_and_a_mapping_declare_the_same_pairs():
    ring = Dynamic(handler=1, max_live=64, queue_messages=64, direct=(1,))
    pairs = Dynamic(handler=1, direct=[(0, 1), (1, 2), (2, 0)])
    mapping = Dynamic(handler=1, direct={0: [1], 1: [2], 2: [0]})
    assert ring.direct_pairs(3) == [(0, 1), (1, 2), (2, 0)]
    assert pairs.direct_pairs(3) == mapping.direct_pairs(3) == ring.direct_pairs(3)


def test_the_load_refuses_a_pair_twice_a_rank_outside_and_x280_ranks():
    with pytest.raises(ValueError, match="twice"):
        Dynamic(handler=1, direct=[(0, 1), (0, 1)]).direct_pairs(4)
    with pytest.raises(ValueError, match="message ranks"):
        Dynamic(handler=1, direct=[(0, 4)]).direct_pairs(4)
    with pytest.raises(ValueError, match="X280"):
        Dynamic(handler=1, x280_ranks=1, direct=(1,))


def test_a_receiver_entry_names_the_slot_and_a_sender_entry_the_credit_line():
    rows = Dynamic(handler=1, direct=[(0, 2), (1, 2)]).direct_rows(5)  # 2 counters
    receivers, senders, out, incoming, credit = (int(w) for w in rows[2][:5])
    assert (receivers, senders) == (0, 2)
    slots = int(rows[2][8])
    for sender in (0, 1):
        entry = int(rows[sender][int(rows[sender][2]) // 4])
        assert entry & 0xFFFF == 2 and (entry >> 16) * 16 == slots + 64 * sender
        back = int(rows[2][incoming // 4 + sender])
        assert back & 0xFFFF == sender and (back >> 16) * 16 == int(rows[sender][4])
