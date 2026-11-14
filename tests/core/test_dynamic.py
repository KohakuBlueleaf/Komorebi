"""The launcher's side of a dynamic handler: options, rejections, buffers, words."""

import pytest

from komorebi.layout import counter_tiles, queue_pages
from komorebi.program import Dynamic, messages_header


def test_queue_pages_are_consecutive_per_core():
    assert queue_pages(0, 64) == range(0, 2)
    assert queue_pages(3, 64) == range(6, 8)
    with pytest.raises(ValueError):
        queue_pages(0, 48)


@pytest.mark.parametrize(
    "options",
    [
        {"message_bytes": 48},
        {"message_bytes": 512},
        {"queue_messages": 1000},
        {"queue_messages": 16},
        {"bins": 0},
        {"max_live": 0},
        {"counters": 0},
        {"counters": 5},
        {"body_bytes": 100},
        {"body_bytes": 8192},
    ],
)
def test_bad_options_are_rejected(options):
    with pytest.raises(ValueError):
        Dynamic(handler=1, **options)


def test_max_live_above_the_queue_capacity_is_rejected():
    with pytest.raises(ValueError, match="max_live"):
        Dynamic(handler=1, max_live=1 << 17, queue_messages=1 << 16).check(
            119, 4 << 30, 4
        )
    Dynamic(handler=1, max_live=1 << 16, queue_messages=1 << 16).check(119, 4 << 30, 4)


def test_queues_beyond_the_dram_given_and_a_shallow_ring_are_rejected():
    dynamic = Dynamic(handler=1, queue_messages=1 << 16)
    with pytest.raises(ValueError, match="queues"):
        dynamic.check(119, dynamic.queue_bytes(119) - 1, 4)
    with pytest.raises(ValueError, match="depth"):
        dynamic.check(119, 4 << 30, 1)
    with pytest.raises(ValueError, match="quota"):
        Dynamic(handler=1, max_live=100).check(119, 4 << 30, 4)


def test_packet_pages_keep_items_64_byte_aligned():
    header_constants, _ = messages_header()
    for buffer in Dynamic(handler=1, message_bytes=128).buffers():
        assert buffer.page_bytes % 64 == 0, buffer
    assert header_constants["kMessagesHeader"] % 64 == 0


def test_words_follow_the_device_struct():
    _, fields = messages_header()
    dynamic = Dynamic(handler=7, bins=4, idle_cycles=99)
    words = dict(zip(fields, dynamic.args(0x1000, 0x2000)))
    assert words == {
        "handler": 7,
        "queue": 0x1000,
        "cap": 1 << 16,
        "message_bytes": 64,
        "lines": 0x2000,
        "idle": 99,
        "bins": 4,
        "counters": 2,
        "body": 0,
        "body_bytes": 0,
        "x280": 0,
        "x280_queue": 0,
        "x280_bank": 0,
        "x280_tile": 0,
        "x280_credits": 0,
        "inbox": 0,
    }
    assert dynamic.compute_args(119) == [64, (1 << 16) // 117, 2, 0]
    with pytest.raises(ValueError, match="no core"):
        dynamic.check(2, 4 << 30, 4)


def test_counter_tiles_are_the_last_k_cores_and_core_c_reserves_on_c_mod_k():
    tiles = counter_tiles(10, 7, 2)
    assert tiles.workers == range(10, 15) and tiles.counters == range(15, 17)
    assert tiles.counter_of == {10: 15, 11: 16, 12: 15, 13: 16, 14: 15}
    assert tiles.serves == {15: [10, 12, 14], 16: [11, 13]}
    assert counter_tiles(0, 2, 1).serves == {1: [0]}
    for first, count, counter_count in ((0, 2, 2), (0, 9, 0), (0, 9, 5)):
        with pytest.raises(ValueError):
            counter_tiles(first, count, counter_count)
