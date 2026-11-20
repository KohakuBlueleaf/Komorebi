"""The session's host transfer of tensor bytes: reads and writes, by the channel or
the device API."""

import secrets
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from komorebi import abi, formats
from komorebi.device import channel as channel_module
from komorebi.device import dram, mock, native
from komorebi.device import tensor as tensor_module
from komorebi.errors import (
    ChannelError,
)
from komorebi.program import overlay
from komorebi.session import kernels as kernel_module
from komorebi.session import tensor_bytes
from komorebi.session.channel_transfer import ChannelTransfer

# A transfer without a run launches its kernel from this many bytes on: a launch costs
# about 30 ms, as the device API's read of some 76 KiB does (measured on a p150a).
LAUNCH_MIN_BYTES = 80 << 10
TRANSFER_WAIT_S = 1.0  # beyond the bytes at 1 GB/s: a transfer's wait for its words
WORDS_SPIN_S = 300e-6  # a transfer's words looked at without a pause this long


class TransferMixin:
    """The session's host transfer of tensor bytes: reads and writes, by the channel or
    the device API."""

    # the state of the session's other parts this one reads or sets, owned there
    _region: int | None  # Session
    device: Any  # Session
    log: Callable  # Session
    tree: Any  # Session
    _mock_mapped: Callable  # OpenMixin
    _module: Any  # OpenMixin
    channel: Any  # OpenMixin
    _starting: Any  # LoadMixin
    island: Any  # LoadMixin
    run: Any  # LoadMixin

    def _init_transfer(self):
        """No transfer made yet."""
        self._transfer_built = False  # a prebuild: the transfer kernel built
        self._built_before = []  # (found, compiled) built before the next load
        self.transfer_cores = channel_module.CORES_DEFAULT  # cores of a transfer, most
        self.transfer_cycles = None  # the last transfer's span on the cores
        self._channel_transfers = []  # the live ChannelTransfers, their region slots

    def channel_transfer(self, tensor, push):
        """A ChannelTransfer of the whole DRAM `tensor` for a list the loaded run's
        program issues: a PULL into the tensor, or (`push`) a PUSH out of it."""
        return ChannelTransfer(self, tensor, push)

    def read(self, *tensors, first=0, pages=None, out=None):
        """The raw pages of tensors in DRAM or L1 ([pages, page_bytes] uint8 each);
        with `first` or `pages`, only those pages of one tensor (None: to the end); with
        `out` (one tensor), into those rows. Through the host channel when it is there
        (host.md, "Reads"); while a run runs, only what no kernel writes any more."""
        if (first or pages is not None or out is not None) and len(tensors) != 1:
            raise ValueError("a page range, or rows to fill, is read from one tensor")
        tensor_module.in_dram(*tensors)
        self._module = self._module or native.load_module(self.tree)
        way = self._transfer_way(tensors, first, pages)
        if way is None:
            got = self._read_device(tensors, first, pages)
            if out is None:
                return got
            out[:] = got[0]
            return [out]
        rows = []
        for tensor in tensors:
            start, count = dram.page_range(first, pages, tensor.pages)
            if out is not None and out.shape != (count, tensor.page_bytes):
                raise ValueError(f"out is [{count}, {tensor.page_bytes}] uint8 rows")
            pages_out = (
                np.empty((count, tensor.page_bytes), np.uint8) if out is None else out
            )
            page = tensor.aligned_page_bytes
            for done, step in channel_module.rounds(count, page, self._region):
                raw = self._push(tensor, start + done, step, way)
                pages_out[done : done + step] = raw.reshape(step, page)[
                    :, : tensor.page_bytes
                ]
            rows.append(pages_out)
        return rows

    def _read_device(self, tensors, first, pages):
        """read() through the device API: DRAM tensors by banks, an L1 tensor whole
        (which waits for launched cores: not while a run is loaded)."""
        if mock.active():  # a mock run computed nothing: zeros
            if self.device is None:  # as the card's read once the device closed
                raise RuntimeError("the device is closed")
            return [
                np.zeros(
                    (dram.page_range(first, pages, t.pages)[1], t.page_bytes), "u1"
                )
                for t in tensors
            ]
        in_l1 = [tensor for tensor in tensors if not tensor.in_dram]
        if in_l1 and self.run is not None and not self.run.stopped:
            raise ChannelError(
                "an L1 tensor is read while a run is loaded through the host channel, "
                "which this session cannot use for it"
            )
        in_dram = [tensor for tensor in tensors if tensor.in_dram]
        banked = iter(
            tensor_bytes.read_tensors(self.device, self._module, in_dram, first, pages)
        )
        out = []
        for tensor in tensors:
            if tensor.in_dram:
                out.append(next(banked))
                continue
            start, count = dram.page_range(first, pages, tensor.pages)
            rows = tensor.read_bytes().reshape(tensor.pages, tensor.page_bytes)
            out.append(rows[start : start + count].copy())
        return out

    def read_chunks(self, tensor, chunk_bytes=2 << 20):
        """Yields (first, pages) over the whole tensor, each read() of at most
        `chunk_bytes` (aligned pages), so that the caller can log every chunk. With the
        host channel, the cores push a region's worth at a time."""
        total, aligned = tensor.pages, tensor.aligned_page_bytes
        per = chunk_bytes // aligned
        if per < 1:
            raise ValueError(f"chunk_bytes {chunk_bytes}: a page takes {aligned}")
        tensor_module.in_dram(tensor)  # as read(): not one a chain sent direct
        self._module = self._module or native.load_module(self.tree)
        way = self._transfer_way([tensor], 0, None)
        if way is None:
            for first in range(0, total, per):
                (pages,) = self.read(tensor, first=first, pages=min(per, total - first))
                yield first, pages
            return
        for done, step in channel_module.rounds(total, aligned, self._region):
            raw = self._push(tensor, done, step, way).reshape(step, aligned)
            for at in range(0, step, per):
                yield done + at, raw[at : at + per, : tensor.page_bytes].copy()

    def write(self, tensor, pages, first=0):
        """Writes `pages` ([n, page_bytes] uint8, as read() returns them) into pages
        first .. first + n - 1 of a tensor in DRAM or L1, through the host channel
        (also while a run is loaded); without it only a whole tensor, before a load.
        Block-float pages with a negative zero code are refused (formats.bfp_check)."""
        given = pages
        pages = dram.page_rows(tensor, pages)
        if tensor.dtype in formats.BLOCK:  # a negative zero code reads -inf on the card
            formats.bfp_check(
                pages, formats.BLOCK[tensor.dtype], f"pages written to {tensor}"
            )
        start, count = dram.page_range(first, len(pages), tensor.pages)
        if tensor.kept is not None:  # the array given, for a reopened card
            tensor.kept.append(("pages", given, first))
        tensor.written = True  # its transposed_tiles is fixed from here
        tensor.sent = None  # written: DRAM holds it again (in_dram)
        self._module = self._module or native.load_module(self.tree)
        pieces = self._live_pieces(tensor)
        if pieces and pieces > 1:  # a page wider than the run's chunk: in pieces
            self._write_pieces(tensor, pages, start, pieces)
            return
        way = self._transfer_way([tensor], start, count)
        if way is None:
            if self._mock_mapped() and (count != tensor.pages or self.run is not None):
                return  # the card's channel would take it; a mock run reads nothing
            if count != tensor.pages or (self.run is not None and not self.run.stopped):
                raise ChannelError(self._write_refused(tensor))
            tensor.buffer.write(pages.reshape(-1))
            return
        aligned = tensor.aligned_page_bytes
        for done, step in channel_module.rounds(count, aligned, self._region):
            rows = self.channel[: step * aligned].reshape(step, aligned)
            rows[:, : tensor.page_bytes] = pages[done : done + step]
            self._transfer(tensor, start + done, step, way, False)

    def _live_pieces(self, tensor):
        """write_pieces of a DRAM tensor's page against the live run's chunk (None: no
        live run with a transfer buffer and a channel, or an L1 tensor)."""
        run = self.run
        transfer = getattr(getattr(run, "backend", None), "transfer", None)
        if self.channel is None or transfer is None or not tensor.in_dram:
            return None
        if run.stopped or not run.alive():
            return None
        return tensor_bytes.write_pieces(tensor.aligned_page_bytes, transfer.chunk)

    def _write_pieces(self, tensor, pages, start, pieces):
        """Pages start .. of a DRAM tensor written while a run lives, each page as
        `pieces` pieces at byte offsets of the same bank bytes (split_pages): a view of
        the tensor with pages of aligned / pieces bytes, each run of them a transfer."""
        aligned = tensor.aligned_page_bytes
        rows = np.zeros((len(pages), aligned), np.uint8)
        rows[:, : tensor.page_bytes] = pages
        view = SimpleNamespace(
            address=tensor.address, page_bytes=aligned // pieces,
            aligned_page_bytes=aligned // pieces, pages=tensor.pages * pieces,
            in_dram=True, shard_grid=None,
        )  # fmt: skip
        for at, part in tensor_bytes.split_pages(start, rows, pieces, self.dram_banks):
            for done, step in channel_module.rounds(
                len(part), view.aligned_page_bytes, self._region
            ):
                way = self._transfer_way([view], at + done, step)
                if way != "run":
                    raise ChannelError(self._write_refused(tensor))
                channel = self.channel[: step * view.aligned_page_bytes]
                channel.reshape(step, -1)[:] = part[done : done + step]
                self._transfer(view, at + done, step, way, False)

    def _write_refused(self, tensor):
        """Why a page range, or a write while a run is loaded, cannot go: as it is."""
        if self.channel is None:
            return (
                "a page range, or a write while a run is loaded, goes through the host "
                "channel, which this session does not have"
            )
        run = self.run
        transfer = getattr(getattr(run, "backend", None), "transfer", None)
        if run is not None and not run.stopped and not run.alive():
            return "a write while a run is loaded needs the run alive: it has failed"
        if run is not None and not run.stopped and transfer is not None:
            return (
                f"a write while a run is loaded moves at most {transfer.chunk:,} B a "
                f"page through the run's transfer buffer; this tensor's page is "
                f"{tensor.aligned_page_bytes:,} B and splits into no equal pieces of "
                f"whole {tensor_bytes.DRAM_PIECE_ALIGN} B within it"
            )
        return (
            "a page range, or a write while a run is loaded, goes through the run's "
            "transfer buffer, which this run does not have"
        )

    def _transfer_way(self, tensors, first, pages):
        """How a read of these pages goes: "run" (the live run's cores push them),
        "launch" (the transfer kernel, from LAUNCH_MIN_BYTES on), None (the device).
        An L1 tensor's pages are moved by the cores that hold them."""
        self._map_channel()
        if self.channel is None:
            if self._mock_mapped():  # the card would launch the transfer kernel
                self._built_before += self._transfer_kernel_built()
            return None
        if max(tensor.aligned_page_bytes for tensor in tensors) > self._region.data:
            return None
        largest = max(
            (tensor.aligned_page_bytes for tensor in tensors if tensor.in_dram),
            default=0,
        )
        run = self.run
        if run is not None and not run.stopped:
            transfer = getattr(run.backend, "transfer", None)
            fits = transfer is not None and largest <= transfer.chunk
            fits = fits and all(
                self._l1_ranks(tensor, 0, tensor.pages) is not None
                for tensor in tensors
                if not tensor.in_dram
            )
            return "run" if fits and run.alive() else None
        if self._starting is not None:
            return None
        if largest > channel_module.CHUNK_DEFAULT and (
            channel_module.chunk_for(largest, self.l1()[1]) is None
        ):
            return None  # two pages do not fit a core's L1
        total = sum(
            dram.page_range(first, pages, tensor.pages)[1] * tensor.aligned_page_bytes
            for tensor in tensors
        )
        return "launch" if total >= LAUNCH_MIN_BYTES else None

    def _push(self, tensor, first, pages, way):
        """Pages first .. first + pages - 1 (one round: they fit the region's data)
        pushed into the region; a view of their bytes, valid until the next transfer.
        """
        self._transfer(tensor, first, pages, way, True)
        return self.channel[: pages * tensor.aligned_page_bytes]

    def _l1_ranks(self, tensor, first, pages):
        """channel.l1_ranks of an L1 tensor's pages in the live run."""
        return channel_module.l1_ranks(
            first, pages, tensor.shard_grid[0], self.grid[0], self.run.holds()
        )

    def _transfer(self, tensor, first, pages, way, push):
        """One round of a PUSH or PULL of pages first .. first + pages - 1, through
        the live run ("run") or the transfer kernel ("launch"), checked first."""
        if not tensor.in_dram:
            self._transfer_l1(tensor, first, pages, way, push)
            return
        page, region = tensor.aligned_page_bytes, self._region
        nonce = secrets.randbits(32)
        timeout = TRANSFER_WAIT_S + pages * page / 1e9
        most = self.transfer_cores
        if way == "run":
            run = self.run
            chunk = channel_module.chunk_bytes(page, run.backend.transfer.chunk)
            wanted = channel_module.ranks(pages, page, chunk, most)
            group = channel_module.transfer_group(run.holds(), run.cores, wanted)
            if self.island is not None:  # the run takes flat ranks
                graph, here = self.island
                group = (group[0] + graph.first(here), *group[1:])
            buffer, count = run.backend.transfer.address, group[1]
        else:
            chunk = channel_module.chunk_bytes(page, channel_module.CHUNK_DEFAULT)
            most = min(most, self.grid[0])
            buffer, count = 0, channel_module.ranks(pages, page, chunk, most)
        transfer = channel_module.Transfer(
            tensor.address, page, first, pages, 0, region.words, region.bytes, nonce,
            buffer, chunk,
        )  # fmt: skip
        channel_module.check(
            transfer, count, self.l1_bytes, self.dram_banks, tensor.pages
        )
        words = channel_module.words_view(self.channel, region, count)
        words[:] = 0
        want = channel_module.sequence_words(nonce, pages, page, count)
        if way == "run":
            handler = abi.headers()["KM_HANDLER_PUSH" if push else "KM_HANDLER_PULL"]
            run.transfer(handler, group, transfer.args(), timeout)
            if not self._words_arrived(words, want, timeout):
                raise ChannelError(self._words_missing(words, want, timeout))
        else:
            self._launch_transfer(transfer, count, push, words, want, timeout)
        self.transfer_cycles = channel_module.cycles(words)

    def _transfer_l1(self, tensor, first, pages, way, push):
        """One round of an L1 tensor: each core holding one of the pages moves it
        between its L1 and the region, as one rank of a command a run of consecutive
        ranks (one program held) or of one launch over those cores."""
        page, region = tensor.aligned_page_bytes, self._region
        timeout = TRANSFER_WAIT_S + pages * page / 1e9
        h = abi.headers()
        if way != "run":
            transfer = channel_module.Transfer(
                tensor.address, page, first, pages, 0, region.words, region.bytes,
                secrets.randbits(32), 0, h["KM_TRANSFER_L1"],
            )  # fmt: skip
            channel_module.check(
                transfer, pages, self.l1_bytes, self.dram_banks, tensor.pages
            )
            words = channel_module.words_view(self.channel, region, pages)
            words[:] = 0
            want = channel_module.sequence_words(transfer.nonce, pages, page, pages)
            self._launch_transfer(transfer, pages, push, words, want, timeout, tensor)
            self.transfer_cycles = channel_module.cycles(words)
            return
        run = self.run
        handler = h["KM_HANDLER_PUSH" if push else "KM_HANDLER_PULL"]
        here = self.island[0].first(self.island[1]) if self.island else 0
        held = self._l1_ranks(tensor, first, pages)
        self._check_link_cores(tensor, held, push)
        self.transfer_cycles = 0
        for at, count, rank, field in held:
            transfer = channel_module.Transfer(
                tensor.address, page, at, count, (at - first) * page, region.words,
                region.bytes, secrets.randbits(32), run.backend.transfer.address,
                h["KM_TRANSFER_L1"],
            )  # fmt: skip
            channel_module.check(
                transfer, count, self.l1_bytes, self.dram_banks, tensor.pages
            )
            words = channel_module.words_view(self.channel, region, count)
            words[:] = 0
            want = channel_module.sequence_words(transfer.nonce, count, page, count)
            run.transfer(handler, (rank + here, count, field), transfer.args(), timeout)
            if not self._words_arrived(words, want, timeout):
                raise ChannelError(self._words_missing(words, want, timeout))
            self.transfer_cycles += channel_module.cycles(words)

    def _check_link_cores(self, tensor, held, push):
        """Refuses, before anything is sent, a live transfer of an island's L1 tensor
        whose pages reach past the island's ranks: the link cores run no rank's
        handlers (their dispatcher would reject the command by its handler)."""
        if self.island is None or not held:
            return
        graph, island = self.island
        ranks = graph.count(island)
        past = [(at + k, rank + k) for at, n, rank, _ in held for k in range(n)]
        past = [(page, core) for page, core in past if core >= ranks]
        if past:
            page, core = past[0]
            raise ChannelError(
                f"{tensor}: page {page} is on core {core} of island {island}, a link "
                f"core (the island's ranks are 0 to {ranks - 1}): a live "
                f"{'read' if push else 'write'} of an L1 tensor reaches ranks only; "
                "make the tensor over the island's ranks (group=), or "
                f"{'read' if push else 'write'} it after run.stop()"
            )

    def _channel_region(self):
        """The region's bytes for the dispatcher (a thread's transfers stay inside
        it), 0 without a mapped channel."""
        return self._region.bytes if self.channel is not None else 0

    @staticmethod
    def _words_arrived(words, want, timeout, spin=None):
        """Waits (bounded) until every rank's sequence word is the one expected:
        looks without a pause for the first `spin` seconds (None: WORDS_SPIN_S), then
        every 200 us."""
        started = time.monotonic()
        spin = WORDS_SPIN_S if spin is None else spin
        deadline, spin = started + timeout, started + spin
        while not np.array_equal(words[:, :4], want):
            now = time.monotonic()
            if now > deadline:
                return False
            if now > spin:
                time.sleep(0.0002)
        return True

    @staticmethod
    def _words_missing(words, want, timeout):
        seen = int(np.sum(np.all(words[:, :4] == want, axis=1)))
        return (
            f"{len(want) - seen} of {len(want)} cores' sequence words did not arrive "
            f"in the host channel within {timeout:.1f} s"
        )

    def _launch_transfer(self, transfer, count, push, words, want, timeout, l1=None):
        """The transfer kernel without a run, one rank a core: cores 0 .. count - 1 of
        row 0, or the cores holding the pages of L1 tensor `l1`; waits for their
        words, then for the cores; their reports on a failure."""
        if l1 is None:
            grid, places = (count, 1), [(rank, 0) for rank in range(count)]
            cores = [(0, 0, count - 1, 0)]
        else:
            grid = l1.shard_grid
            places, cores = channel_module.l1_cores(transfer.first, count, grid[0])
        report = self.empty((grid[0] * grid[1], 16), "uint32", shard_grid=grid)
        args = [
            (x, y, [report.address, int(push), rank, count, *transfer.args()])
            for rank, (x, y) in enumerate(places)
        ]
        launch = self._transfer_launch(cores, args, transfer.chunk)
        started = time.time()
        launch.run()
        if kernel_module.built_files(launch.recipes(), started)[1]:
            self.log("built: the host transfer kernel compiled (not found built)")
        try:
            arrived = self._words_arrived(words, want, timeout)
        finally:
            self.device.finish()
        status = None
        if not arrived:
            rows = report.read()[:, 0].astype(np.int64) - 1
            status = rows[[y * grid[0] + x for x, y in places]]
        report.free()
        if status is not None:
            raise ChannelError(
                f"{self._words_missing(words, want, timeout)}; the kernel's status "
                f"{status.tolist()} (0 done, 1 refused, 2 reads and 3 writes not done "
                "in time, 4 PCIe routing left, -1 not finished)"
            )

    def _transfer_launch(self, cores, args, chunk):
        """The transfer kernel's launch on `cores` with `args`, chunks of `chunk`."""
        launch = self._module.Launch(self.device)
        include = abi.headers().abi.parents[2]
        source = include.parent / "kernels" / "host_transfer.cpp"
        launch.kernel(str(source), cores, 0, [], [], [str(include)], args, noc=0)
        # the two chunks and the slot, from the buffer's first 64-aligned address
        launch.cb(cores, 0, "uint32", 16, 2 * chunk + 64 + channel_module.DRAM_LINE)
        return launch

    def _transfer_kernel_built(self):
        """A prebuild: the transfer kernel built as _launch_transfer launches it, once a
        session; [(found, compiled)] for the next load's line."""
        if self._transfer_built:
            return []
        self._transfer_built = True
        self._module = self._module or native.load_module(self.tree)
        launch = self._transfer_launch([(0, 0, 0, 0)], [(0, 0, [0] * 16)], 4096)
        started = time.time()
        launch.build()
        return [kernel_module.built_files(launch.recipes(), started)]

    def _log_built(self, descriptor, started, program):
        """The load's line: its kernels and overlay parts found built and compiled
        (with a transfer kernel built before it), each compiled kernel's defines of
        `program` and the runtime's (KM_*)."""
        found, compiled = kernel_module.built_files(descriptor.recipes(), started)
        for before_found, before_compiled in self._built_before:
            found, compiled = found + before_found, compiled + before_compiled
        self._built_before = []
        parts = overlay.PARTS_BUILT
        found, compiled = found + parts["found"], compiled + parts["compiled"]
        parts.update(found=0, compiled=0)
        self.log(f"built: {found} found, {compiled} compiled (kernels and parts)")
        for source, builds in descriptor.recipes():
            for row in builds:
                elf = Path(row[1])
                if elf.is_file() and elf.stat().st_mtime >= started:
                    defines = " ".join(
                        str(define)[2:]
                        for define in row[8:]
                        if str(define).startswith("-DKM_")
                        or str(define)[2:].split("=")[0] in program.defines
                    )
                    self.log(f"compiled: {Path(source).name} {row[0]} ({defines})")

    def read_bf16(self, *tensors):
        """bfloat16 tile tensors of rank 2 as float32 arrays, read as read() does."""
        out = []
        for tensor, pages in zip(tensors, self.read(*tensors)):
            if (
                tensor.layout != "tile"
                or tensor.dtype != "bfloat16"
                or len(tensor.shape) != 2
            ):
                raise ValueError("read_bf16 takes bfloat16 tile tensors of rank 2")
            turned = getattr(tensor, "transposed_tiles", False)
            out.append(dram.untilize_bf16_tensor(pages, tensor.shape, turned))
        return out

    def read_tiles_bf16(self, tensor, first, pages):
        """Tiles first .. first + pages - 1 of a bfloat16 tile tensor (one tile a page,
        row-major tile order) as float32 [pages, 32, 32], read as read() does."""
        if tensor.layout != "tile" or tensor.dtype != "bfloat16":
            raise ValueError("read_tiles_bf16 takes a bfloat16 tile tensor")
        if getattr(tensor, "page_tiles", 1) != 1:
            raise ValueError(
                f"read_tiles_bf16 reads a page a tile; this tensor is in pages of "
                f"{tensor.page_tiles} tiles: read_bf16 reads it whole"
            )
        (raw,) = self.read(tensor, first=first, pages=pages)
        tile = dram.TILE
        tiles = dram.untilize_bf16(raw, len(raw) * tile, tile).reshape(-1, tile, tile)
        if getattr(tensor, "transposed_tiles", False):  # each tile turned back
            tiles = tiles.swapaxes(1, 2).copy()
        return tiles
