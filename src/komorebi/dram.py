"""Interleaved DRAM buffers read bank by bank: page p lies in bank p % banks at the
buffer's base + (p // banks) * the aligned page size, so one read per bank fetches it.
"""

import numpy as np

TILE = 32
FACE = 16


def bank_spans(base, aligned, pages, banks):
    """(bank, address, bytes) of the banks holding a buffer's pages, one per bank."""
    return [
        (b, base, -(-(pages - b) // banks) * aligned) for b in range(min(banks, pages))
    ]


def pages_in_order(chunks, page_bytes, aligned, pages, banks):
    """The pages as [pages, page_bytes] uint8, from the chunks read at bank_spans."""
    out = np.empty((pages, page_bytes), np.uint8)
    for b, chunk in enumerate(chunks):
        out[b::banks] = np.asarray(chunk, np.uint8).reshape(-1, aligned)[:, :page_bytes]
    return out


def untilize_bf16(pages, rows, cols):
    """bfloat16 tiles (row-major tile order, four 16 x 16 faces each) as a float32
    [rows, cols] array; rows and cols are multiples of 32."""
    bits = np.asarray(pages).reshape(-1).view(np.uint16).astype(np.uint32) << 16
    t = bits.view(np.float32).reshape(rows // TILE, cols // TILE, 2, 2, FACE, FACE)
    return t.transpose(0, 2, 4, 1, 3, 5).reshape(rows, cols)
