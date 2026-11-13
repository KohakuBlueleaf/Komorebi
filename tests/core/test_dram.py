"""Interleaved DRAM pages to banks and back."""

import numpy as np
import pytest

from komorebi import KomorebiError, dram

PAGES, BANKS, PAGE_BYTES, ALIGNED, BASE = 11, 4, 6, 8, 0x1000
DATA = np.arange(PAGES * PAGE_BYTES, dtype=np.uint8).reshape(PAGES, PAGE_BYTES)


def read(first, pages):
    """Pages read at bank_spans from banks that hold DATA (page p in bank p % BANKS
    at BASE + p // BANKS * ALIGNED), put in order by pages_in_order."""
    memory = np.zeros((BANKS, BASE + PAGES * ALIGNED), np.uint8)
    for p in range(PAGES):
        at = BASE + p // BANKS * ALIGNED
        memory[p % BANKS, at : at + PAGE_BYTES] = DATA[p]
    spans = dram.bank_spans(BASE, ALIGNED, pages, BANKS, first)
    chunks = [memory[bank, at : at + size] for bank, at, size in spans]
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
