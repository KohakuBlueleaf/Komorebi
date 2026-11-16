"""Interleaved DRAM pages to banks and back."""

import numpy as np
import pytest

from komorebi import KomorebiError
from komorebi.device import dram

PAGES, BANKS, PAGE_BYTES, ALIGNED, BASE = 11, 4, 6, 8, 0x1000
DATA = np.arange(PAGES * PAGE_BYTES, dtype=np.uint8).reshape(PAGES, PAGE_BYTES)
# the banks holding DATA: page p in bank p % BANKS at BASE + p // BANKS * ALIGNED
MEMORY = np.zeros((BANKS, BASE + PAGES * ALIGNED), np.uint8)
for p in range(PAGES):
    MEMORY[p % BANKS, BASE + p // BANKS * ALIGNED :][:PAGE_BYTES] = DATA[p]


def read(first, pages):
    """Pages read at bank_spans from MEMORY, put in order by pages_in_order."""
    spans = dram.bank_spans(BASE, ALIGNED, pages, BANKS, first)
    chunks = [MEMORY[bank, offset : offset + size] for bank, offset, size in spans]
    return dram.pages_in_order(chunks, PAGE_BYTES, ALIGNED, pages, BANKS)


def test_pages_read_bank_by_bank_come_back_in_page_order():
    spans = dram.bank_spans(BASE, ALIGNED, PAGES, BANKS)
    assert spans == [(0, BASE, 24), (1, BASE, 24), (2, BASE, 24), (3, BASE, 16)]
    assert (read(0, PAGES) == DATA).all()


def test_a_page_range_starting_in_any_bank_reads_those_pages_only():
    for first in range(BANKS + 1):
        assert (read(first, 6) == DATA[first : first + 6]).all()
    assert (read(5, 2) == DATA[5:7]).all()  # fewer pages than banks


def test_a_range_outside_the_tensor_names_its_page_count():
    assert dram.page_range(3, None, PAGES) == (3, 8)
    for first, pages in [(-1, 2), (10, 2), (11, None), (0, 0)]:
        with pytest.raises(KomorebiError, match="has 11 pages"):
            dram.page_range(first, pages, PAGES)


def test_a_region_reads_its_own_pages_of_one_bank_only():
    assert dram.bank_chunks(BASE, ALIGNED, 1, 0, 3, 2) == [
        (1, BASE, 16),
        (1, BASE + 16, 8),
    ]
    # bank 1 holds pages 1, 5, 9; a region of n bytes takes -(-n // ALIGNED) of them
    for first, size in [(0, 3), (1, 8), (1, 13), (0, 24)]:  # start, middle, end
        pages = -(-size // ALIGNED)
        spans = dram.bank_chunks(BASE, ALIGNED, 1, first, pages, 2)
        got = np.concatenate(
            [MEMORY[bank, offset : offset + length] for bank, offset, length in spans]
        )
        want = DATA[1 + BANKS * first :: BANKS][:pages]
        assert (got.reshape(pages, ALIGNED)[:, :PAGE_BYTES] == want).all()
