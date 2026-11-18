"""The island session's host reads and writes across islands."""

import concurrent.futures
from typing import Any

import numpy as np

from komorebi.device import dram
from komorebi.device import tensor as tensor_module
from komorebi.errors import (
    KomorebiError,
)


def _rows(indices):
    """Ascending row indices as a slice when they are one run (a view of the rows, no
    copy), else as they are."""
    if indices[-1] - indices[0] == len(indices) - 1:
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


class IoMixin:
    """The island session's host reads and writes across islands."""

    # its state the constructor sets (komorebi.islands.session)
    host_reaches: tuple | None
    # the state of the session's other parts this one reads or sets, owned there
    card_dram: bool  # MeshMixin
    links: int  # MeshMixin
    run: Any  # MeshMixin
    sessions: tuple  # MeshMixin
    split: int  # MeshMixin

    def _each_island(self, work, spans):
        """work(island, at, low, high) for every span: with a run loaded on every
        island, one thread an island at once; else in turn."""
        live = self.run is not None and not self.run.stopped
        if not live or len(spans) < 2 or self.split > 1:  # a split card: one channel
            for span in spans:
                work(*span)
            return
        with concurrent.futures.ThreadPoolExecutor(len(spans)) as pool:
            for done in [pool.submit(work, *span) for span in spans]:
                done.result()

    def _reached(self, spans):
        """The spans whose island the host reaches through its own channel, and the
        others (host_reaches)."""
        if self.host_reaches is None:
            return spans, []
        direct = [span for span in spans if span[0] in self.host_reaches]
        return direct, [span for span in spans if span[0] not in self.host_reaches]

    def _read_across(self, part, island, first, pages, timeout=30.0):
        """Pages first .. of island `island`'s part, sent by its link cores into a
        buffer of the island the host reaches, then read through that one's channel."""
        via, staging = self._staging(part, island)
        try:
            sent = self._send(
                island, via, part.address, part.aligned_page_bytes, first, pages,
                staging.address,
            )  # fmt: skip
            landed = self.wait_pages(via, self._across(island, via), pages)
            self.wait_together([sent, (via, landed)], timeout)
            (got,) = self.sessions[via].read(staging, first=first, pages=pages)
        finally:
            staging.free()
        return got

    def _spans(self, tensor, start, count, reading=False):
        """[(island, its part's pages in order, first index, end index)] of the part
        indices whose pages lie in start .. start + count - 1 (a copy read: island 0
        alone)."""
        spans = []
        for island, part in enumerate(tensor.parts):
            if part is None or (reading and tensor.placement.copy and island):
                continue
            at = tensor.placement.part(island)
            low, high = np.searchsorted(at, [start, start + count])
            if high > low:
                spans.append((island, at, int(low), int(high)))
        return spans

    def _staging(self, part, island):
        """A buffer like `part` on the island the host reaches, for a part of island
        `island` crossing the page path; KomorebiError without a run with links."""
        if self.run is None or self.run.stopped or not self.links:
            raise KomorebiError(
                f"host_reaches={list(self.host_reaches)}: island {island}'s part "
                "crosses by the page path, which runs in a loaded run with links: "
                "write or read it after the load"
            )
        via = self.host_reaches[0]
        return via, self.sessions[via].empty(part.shape, part.dtype, part.layout)

    def _write_across(self, part, island, first, rows, timeout=30.0):
        """`rows` into pages first .. of island `island`'s part: written through the
        reached island's channel into a buffer there, sent by its link cores."""
        via, staging = self._staging(part, island)
        try:
            self.sessions[via].write(staging, rows, first)
            sent = self._send(
                via, island, staging.address, part.aligned_page_bytes, first, len(rows),
                part.address,
            )  # fmt: skip
            landed = self.wait_pages(island, self._across(via, island), len(rows))
            self.wait_together([sent, (island, landed)], timeout)
        finally:
            staging.free()

    def read(self, *tensors, first=0, pages=None, out=None):
        """The raw pages of island tensors as Session.read gives them: each island's
        pages through its own host channel (with a run loaded, the islands at once),
        gathered into one array (`out`, one tensor: into those rows)."""
        if (first or pages is not None or out is not None) and len(tensors) != 1:
            raise ValueError("a page range, or rows to fill, is read from one tensor")
        tensor_module.in_dram(*tensors)  # not one a chain sent direct (as one card)
        if self.card_dram:  # one tensor in the card's DRAM
            parts = [tensor.parts[0] for tensor in tensors]
            return self.sessions[0].read(*parts, first=first, pages=pages, out=out)
        rows_out = out
        out = []
        for tensor in tensors:
            start, count = dram.page_range(first, pages, tensor.pages)
            if rows_out is not None and rows_out.shape != (count, tensor.page_bytes):
                raise ValueError(f"out is [{count}, {tensor.page_bytes}] uint8 rows")
            rows = (
                np.empty((count, tensor.page_bytes), np.uint8)
                if rows_out is None
                else rows_out
            )

            def gather(island, at, low, high, tensor=tensor, rows=rows, start=start):
                part = tensor.parts[island]
                mine = _rows(at[low:high] - start)
                if isinstance(mine, slice):  # straight into the rows: no second copy
                    self.sessions[island].read(
                        part, first=low, pages=high - low, out=rows[mine]
                    )
                    return
                (got,) = self.sessions[island].read(part, first=low, pages=high - low)
                rows[mine] = got

            spans = self._spans(tensor, start, count, reading=True)
            direct, far = self._reached(spans)
            self._each_island(gather, direct)
            for island, at, low, high in far:
                rows[_rows(at[low:high] - start)] = self._read_across(
                    tensor.parts[island], island, low, high - low
                )
            out.append(rows)
        return out

    def read_chunks(self, tensor, chunk_bytes=2 << 20):
        """Yields (first, pages) over the whole tensor, each read() of at most
        `chunk_bytes` (aligned pages), as Session.read_chunks does."""
        total, aligned = tensor.pages, tensor.aligned_page_bytes
        per = chunk_bytes // aligned
        if per < 1:
            raise ValueError(f"chunk_bytes {chunk_bytes}: a page takes {aligned}")
        for first in range(0, total, per):
            (pages,) = self.read(tensor, first=first, pages=min(per, total - first))
            yield first, pages

    def read_bf16(self, *tensors):
        """bfloat16 tile tensors of rank 2 as float32 arrays, read as read() does."""
        out = []
        for tensor, rows in zip(tensors, self.read(*tensors)):
            if (
                tensor.layout != "tile"
                or tensor.dtype != "bfloat16"
                or len(tensor.shape) != 2
            ):
                raise ValueError("read_bf16 takes bfloat16 tile tensors of rank 2")
            turned = getattr(tensor, "transposed_tiles", False)
            out.append(dram.untilize_bf16_tensor(rows, tensor.shape, turned))
        return out

    def read_tiles_bf16(self, tensor, first, pages):
        """Tiles first .. first + pages - 1 of a bfloat16 tile tensor as float32
        [pages, 32, 32], as Session.read_tiles_bf16, read as read() does."""
        if tensor.layout != "tile" or tensor.dtype != "bfloat16":
            raise ValueError("read_tiles_bf16 takes a bfloat16 tile tensor")
        (raw,) = self.read(tensor, first=first, pages=pages)
        tile = dram.TILE
        tiles = dram.untilize_bf16(raw, len(raw) * tile, tile).reshape(-1, tile, tile)
        if getattr(tensor, "transposed_tiles", False):  # each tile turned back
            tiles = tiles.swapaxes(1, 2).copy()
        return tiles

    def write(self, tensor, pages, first=0):
        """Writes `pages` ([n, page_bytes] uint8) into pages first .. first + n - 1:
        each island's part through its own host channel (with a run loaded, the
        islands at once; a copy: every island)."""
        pages = dram.page_rows(tensor, pages)
        tensor.written = True  # its transposed_tiles is fixed from here
        tensor.sent = None  # written: DRAM holds it again (in_dram)
        if self.card_dram:  # one tensor in the card's DRAM
            self.sessions[0].write(tensor.parts[0], pages, first)
            return
        start, count = dram.page_range(first, len(pages), tensor.pages)

        def scatter(island, at, low, high):  # a run of rows: a view, not a copy
            mine = pages[_rows(at[low:high] - start)]
            self.sessions[island].write(tensor.parts[island], mine, low)

        direct, far = self._reached(self._spans(tensor, start, count))
        self._each_island(scatter, direct)
        for island, at, low, high in far:
            rows = pages[_rows(at[low:high] - start)]
            self._write_across(tensor.parts[island], island, low, rows)
