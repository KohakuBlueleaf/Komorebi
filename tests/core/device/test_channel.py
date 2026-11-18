"""The host channel's region, a transfer's bounds and its slices (no card)."""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi import ChannelError, Session
from komorebi.device import channel as ch
from komorebi.session import tensor_bytes

REGION = ch.check_region(4 << 20)
L1 = 1536 * 1024
PAGE = 4096


def transfer(pages, **changes):
    fields = dict(
        address=0x1000, page=PAGE, first=0, pages=pages, offset=0,
        words=REGION.words, region=REGION.bytes, nonce=7, buffer=0x40000,
        chunk=4 * PAGE,
    )  # fmt: skip
    return ch.Transfer(**(fields | changes))


def test_the_region_is_a_multiple_of_64_kib_inside_the_channel():
    assert ch.check_region(ch.REGION_DEFAULT).words == (256 << 20) - (64 << 10)
    for wrong in (100, 64 << 10, (1 << 30) + (64 << 10), 1.5):
        with pytest.raises(ValueError, match="multiple of 65536"):
            ch.check_region(wrong)


def test_the_cards_channel_is_the_drivers_mapping_under_either_name():
    line = "7f00{}0000000-7f00{}0000000 rw-s 00000000 00:2f 7 /dev/hugepages-1G/{}"
    line += "tenstorrent"
    for name in ("", "device_1_"):  # card 0's file, card 1's file
        maps = "\n".join([line.format(0, 4, name), line.format(4, 8, "x")])
        assert ch.find(maps, 0x7F0000000000) == (
            0x7F0000000000,
            f"/dev/hugepages-1G/{name}tenstorrent",
        )
        assert ch.find(maps, 0)[0] is None  # the driver mapped none
        assert ch.find(maps, 0x7F0000100000)[0] is None
        assert ch.find(maps.replace("rw-s", "r--s"), 0x7F0000000000)[0] is None
    assert ch.find(line.format(0, 2, ""), 0x7F0000000000)[0] is None  # 512 MiB
    assert ch.warning_in(
        "NOC address of a hugepage does not match the expected address"
    )
    assert ch.warning_in("Opened device 1") is None


def test_ranks_take_every_page_once_and_their_words_say_how_many_bytes():
    pages, count = 37, 5
    shares = [ch.partition(pages, rank, count) for rank in range(count)]
    assert shares[0] == (0, 8) and shares[-1] == (30, 7)
    assert sum(length for _, length in shares) == pages
    assert all(a + n == b for (a, n), (b, _) in zip(shares, shares[1:], strict=False))
    words = ch.sequence_words(7, pages, PAGE, count)
    assert words[:, 1].tolist() == [n * PAGE for _, n in shares]
    assert (words[:, 3] == 7 ^ 0x50555348).all()


def test_a_large_read_goes_in_rounds_each_fitting_the_regions_data():
    rounds = ch.rounds(2000, PAGE, REGION)
    assert rounds == [(0, 1008), (1008, 992)]
    assert ch.ranks(3, PAGE, 4 * PAGE) == 1 and ch.ranks(10**6, PAGE, 4 * PAGE) == 16
    assert (
        ch.chunk_bytes(3000, 16384) == 15000 and ch.chunk_bytes(20000, 16384) == 20000
    )


def test_a_transfer_is_refused_unless_its_data_and_words_fit_the_region():
    assert ch.fits(transfer(1008), 16, L1)
    assert transfer(1008).args() == [0x1000, PAGE, 0, 1008, 0, REGION.words] + [
        REGION.bytes,
        7,
        0x40000,
        4 * PAGE,
    ]  # KM_TRANSFER_* order
    for wrong in (
        transfer(1009),  # past the data, into the words
        transfer(4, offset=8),  # not a multiple of 16
        transfer(4, words=0),  # the words over the data
        transfer(4, region=(1 << 30) + 16),  # past the channel
        transfer(4, chunk=PAGE + 16),  # not whole pages
        transfer(4, buffer=L1 - 2 * 4 * PAGE),  # no room for the word slot
    ):
        assert not ch.fits(wrong, 16, L1)
        with pytest.raises(ChannelError, match="does not keep its bound: "):
            ch.check(wrong, 16, L1)


def test_a_transfer_group_holds_one_program():
    holds = [
        np.array([1, 1, 0, 0, 1, 1, 1], bool),
        np.array([0, 0, 1, 1, 0, 0, 0], bool),
    ]
    assert ch.transfer_group(None, 120, 16) == (0, 16, 0)
    assert ch.transfer_group(holds, 7, 3) == (4, 3, 0)
    assert ch.transfer_group(holds, 7, 4) == (4, 3, 0)  # the largest window
    assert ch.transfer_group([holds[1]], 7, 4) == (2, 2, 0)


def test_an_l1_tensors_pages_go_to_the_ranks_that_hold_them():
    assert ch.l1_ranks(0, 12, 12, 12) == [(0, 12, 0, 0)]  # the grid's width: one run
    assert ch.l1_ranks(2, 4, 3, 12) == [(2, 1, 2, 0), (3, 3, 12, 0)]  # a row each
    holds = [np.arange(24) < 13, np.arange(24) >= 13]
    assert ch.l1_ranks(0, 6, 3, 12, holds) == [
        (0, 3, 0, 0),
        (3, 1, 12, 0),
        (4, 2, 13, 1),
    ]
    assert ch.l1_ranks(0, 6, 3, 12, [holds[0]]) is None  # core 13 holds no program
    assert ch.l1_cores(2, 4, 3) == (
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(2, 0, 2, 0), (0, 1, 2, 1)],
    )
    l1 = transfer(4, address=0x80000, chunk=0)  # one page a rank, apart from the slot
    assert ch.fits(l1, 4, L1) and not ch.fits(l1, 3, L1)
    assert not ch.fits(l1._replace(address=0x40000 - PAGE + 16), 4, L1)


def routed(pages, page=PAGE, run=None, channel=True):
    """The way a read of `pages` pages takes in a session that is not opened."""
    session = Session(channel=REGION.bytes)
    session.channel = np.zeros(REGION.bytes, np.uint8) if channel else None
    session.l1 = lambda: (0, 1 << 20)  # what a core has left for buffers
    session.run = run
    tensor = SimpleNamespace(in_dram=True, aligned_page_bytes=page, pages=1000)
    return session._transfer_way([tensor], 0, pages)


def test_a_read_goes_through_the_run_the_kernel_or_the_device():
    live = SimpleNamespace(
        stopped=False, alive=lambda: True, backend=SimpleNamespace(transfer=None)
    )
    assert routed(1) is None and routed(20) == "launch"  # 80 KiB on: a launch
    assert routed(20, channel=False) is None
    assert routed(20, page=32 << 10) == "launch"  # a page over a chunk: its own chunk
    assert routed(20, page=600 << 10) is None  # two of them do not fit the L1
    assert routed(20, run=live) is None  # the run has no transfer buffer
    live.backend.transfer = SimpleNamespace(chunk=16 << 10)
    assert routed(1, run=live) == "run"
    assert routed(1, page=32 << 10, run=live) is None  # larger than the run's chunk
    live.backend.transfer = SimpleNamespace(chunk=32 << 10)
    assert routed(1, page=32 << 10, run=live) == "run"
    session = Session(channel=0)
    session._module = object()
    tensor = SimpleNamespace(in_dram=True, aligned_page_bytes=16, page_bytes=16)
    tensor.kept = None
    tensor.pages = 4
    with pytest.raises(ChannelError, match="host channel"):
        session.write(tensor, np.zeros((2, 16), np.uint8))


def test_a_chunk_is_a_page_when_pages_are_larger_and_two_fit():
    assert ch.chunk_for(PAGE, 1 << 20) == 16 << 10
    assert ch.chunk_for(32 << 10, 1 << 20) == 32 << 10
    assert ch.chunk_for(32 << 10, 64 << 10) is None


def test_a_session_given_a_device_maps_the_channel_at_its_first_transfer():
    messages = []
    device = SimpleNamespace(host_channel=lambda: 0)  # the driver mapped none
    session = Session(device=device, log=messages.append)
    assert session.channel is None and not messages
    tensor = SimpleNamespace(in_dram=True, aligned_page_bytes=PAGE, pages=1)
    assert session._transfer_way([tensor], 0, None) is None
    assert "host channel not used" in messages[0]
    session._transfer_way([tensor], 0, None)
    assert len(messages) == 1


def test_a_page_wider_than_the_live_chunk_is_written_in_pieces_of_the_same_bytes():
    assert tensor_bytes.write_pieces(13_760, 16_384) == 1
    assert tensor_bytes.write_pieces(41_280, 16_384) == 3  # 13,760 B, 215 x 64 B
    assert tensor_bytes.write_pieces(41_280 + 16, 16_384) is None
    banks, aligned, pieces = 8, 41_280, 3
    rng = np.random.default_rng(5)
    for first, count in ((16, 24), (3, 13), (0, 1), (7, 9)):
        rows = rng.integers(0, 256, (count, aligned), np.uint8)
        whole = np.zeros((banks, 8 * aligned), np.uint8)  # a bank's bytes
        for p, row in enumerate(rows, start=first):
            at = (p // banks) * aligned
            whole[p % banks, at : at + aligned] = row
        pieced = np.zeros_like(whole)
        width = aligned // pieces
        runs = tensor_bytes.split_pages(first, rows, pieces, banks)
        for start, part in runs:
            for q, row in enumerate(part, start=start):
                at = (q // banks) * width
                pieced[q % banks, at : at + width] = row
        assert np.array_equal(whole, pieced)
        if first % banks == 0 and count % banks == 0:  # whole rows of banks: one run
            assert len(runs) == 1
