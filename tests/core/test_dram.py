"""Interleaved DRAM pages to banks and back."""

import numpy as np

from komorebi import dram


def test_pages_read_bank_by_bank_come_back_in_page_order():
    pages, banks, page_bytes, aligned, base = 11, 4, 6, 8, 0x1000
    data = np.arange(pages * page_bytes, dtype=np.uint8).reshape(pages, page_bytes)
    spans = dram.bank_spans(base, aligned, pages, banks)
    assert spans == [(0, base, 24), (1, base, 24), (2, base, 24), (3, base, 16)]
    chunks = []
    for bank, _, size in spans:  # page p at bank p % banks, slot p // banks
        chunk = np.zeros((size // aligned, aligned), np.uint8)
        chunk[:, :page_bytes] = data[bank::banks]
        chunks.append(chunk.reshape(-1))
    got = dram.pages_in_order(chunks, page_bytes, aligned, pages, banks)
    assert (got == data).all()
