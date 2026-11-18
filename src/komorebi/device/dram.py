"""Interleaved DRAM buffers read bank by bank: page p lies in bank p % banks at the
buffer's base + (p // banks) * the aligned page size, so one read per bank fetches it.
"""

import numpy as np

from komorebi.errors import KomorebiError

TILE = 32
FACE = 16
DRAM_BANKS = 8  # interleaved page i lies in bank i % DRAM_BANKS


def bank_spans(base, aligned, pages, banks, first=0):
    """(bank, address, bytes) of the banks holding pages first .. first + pages - 1
    of a buffer, one per bank: span j starts at page first + j."""
    return [
        (
            (first + j) % banks,
            base + (first + j) // banks * aligned,
            -(-(pages - j) // banks) * aligned,
        )
        for j in range(min(banks, pages))
    ]


def bank_chunks(base, aligned, bank, first, pages, chunk_pages):
    """(bank, address, bytes) reads of one bank's own pages first .. first + pages - 1
    (its page k at base + k * aligned), at most `chunk_pages` pages each."""
    return [
        (bank, base + page * aligned, min(chunk_pages, first + pages - page) * aligned)
        for page in range(first, first + pages, chunk_pages)
    ]


def page_range(first, pages, total, name="the tensor"):
    """(first, pages) of a read of pages first .. first + pages - 1 (pages=None: to
    the end); KomorebiError, naming the page count, when it is outside the buffer."""
    count = total - first if pages is None else pages
    if first < 0 or count < 1 or first + count > total:
        raise KomorebiError(
            f"pages {first} .. {first + count - 1}: {name} has {total} pages"
        )
    return first, count


def pages_in_order(chunks, page_bytes, aligned, pages, banks):
    """The pages as [pages, page_bytes] uint8, from the chunks read at bank_spans
    (with any `first`)."""
    out = np.empty((pages, page_bytes), np.uint8)
    for bank, chunk in enumerate(chunks):
        out[bank::banks] = np.asarray(chunk, np.uint8).reshape(-1, aligned)[
            :, :page_bytes
        ]
    return out


def untilize_bf16(pages, rows, cols):
    """bfloat16 tiles (row-major tile order, four 16 x 16 faces each) as a float32
    [rows, cols] array; rows and cols are multiples of 32."""
    bits = np.asarray(pages).reshape(-1).view(np.uint16).astype(np.uint32) << 16
    values = bits.view(np.float32).reshape(rows // TILE, cols // TILE, 2, 2, FACE, FACE)
    return values.transpose(0, 2, 4, 1, 3, 5).reshape(rows, cols)


def untilize_bf16_tensor(pages, shape, turned=False):
    """untilize_bf16 of a rank-2 tensor's pages: at its tiles' rows and cols, each
    tile turned back when the tensor holds it transposed (`turned`, transposed_tiles),
    then cut to `shape`."""
    rows, cols = shape
    high, wide = -(-rows // TILE) * TILE, -(-cols // TILE) * TILE
    values = untilize_bf16(pages, high, wide)
    if turned:
        values = (
            values.reshape(high // TILE, TILE, wide // TILE, TILE)
            .swapaxes(1, 3)
            .reshape(high, wide)
        )
    return values[:rows, :cols]


def page_rows(tensor, pages):
    """`pages` as write() takes them: [n, page_bytes] uint8 rows, refused by name
    otherwise (another dtype is not cast value by value)."""
    pages = np.asarray(pages)
    if pages.dtype != np.uint8:
        raise ValueError(
            f"pages are [n, {tensor.page_bytes}] uint8 rows, not {pages.dtype}: "
            f"x.view(np.uint8).reshape(n, {tensor.page_bytes}) gives a page's bytes"
        )
    if pages.ndim != 2 or pages.shape[1] != tensor.page_bytes:
        raise ValueError(f"pages are [n, {tensor.page_bytes}] uint8 rows")
    return np.ascontiguousarray(pages)
