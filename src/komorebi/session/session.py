"""A card session: opens the device, loads a program as persistent kernels on every
Tensix core and the dispatcher on X280 cluster 0, and hands out a `Run`.

Needs a tt-metal tree or the runtime package; `import komorebi` does not import it.
"""

import contextlib
import copy
import ctypes
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from komorebi import abi, formats, layout
from komorebi.device import channel as channel_module
from komorebi.device import dram, extents, kernel_buffer, mock, native, remembered
from komorebi.device import tensor as tensor_module
from komorebi.dsl.program import session_nocs
from komorebi.errors import (
    CardBusy,
    ChannelError,
    KernelSizeError,
    ProgramError,
    StartError,
)
from komorebi.program import overlay
from komorebi.program import several as programs
from komorebi.program import thread as thread_module
from komorebi.program.addon import code as added
from komorebi.program.addon import program as added_program
from komorebi.program.checks import instructions as insn_check
from komorebi.program.description import (
    Program,
    guard_says,
    unguarded,
)
from komorebi.program.several import (  # COMPUTE_FLAGS: read here by the compiler
    COMPUTE_FLAGS,  # noqa: F401
)
from komorebi.session import kernels as kernel_module
from komorebi.session import process, sizes, tensor_bytes
from komorebi.session import x280 as x280_module
from komorebi.session.backend.backend import _Backend
from komorebi.session.backend.card import Card
from komorebi.session.run import Run

# The X280 clock: 1750 MHz loses loads under full Tensix load. 1350 is the default; a
# runtime without it gets 1000 (its error: "no PLL solution") unless one was asked for.
X280_MHZ = 1350
X280_FALLBACK_MHZ = 1000
NO_PLL_SOLUTION = "no PLL solution for"
TAKEN_ROW = "L1 held by the session (its L1 tensors, runs not stopped)"
# A transfer without a run launches its kernel from this many bytes on: a launch costs
# about 30 ms, as the device API's read of some 76 KiB does (measured on a p150a).
LAUNCH_MIN_BYTES = 80 << 10
TRANSFER_WAIT_S = 1.0  # beyond the bytes at 1 GB/s: a transfer's wait for its words
WORDS_SPIN_S = 300e-6  # a transfer's words looked at without a pause this long


def _echo(source, target, seen):
    """Copies a pipe to a file descriptor as the bytes come, keeping them in `seen`."""
    with contextlib.suppress(OSError):
        while data := os.read(source, 1 << 16):
            seen.append(data)
            os.write(target, data)
    os.close(source)


@contextlib.contextmanager
def _output_seen():
    """While the block runs, this process's stdout and stderr (the descriptors, so the
    runtime's own lines too) pass through pipes to where they went; yields the bytes."""
    seen, threads, kept = [], [], []
    flush = ctypes.CDLL(None).fflush
    sys.stdout.flush()
    sys.stderr.flush()
    flush(None)
    for fd in (1, 2):
        source, sink = os.pipe()
        kept.append(os.dup(fd))
        os.dup2(sink, fd)
        os.close(sink)
        thread = threading.Thread(target=_echo, args=(source, kept[-1], seen))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    try:
        yield seen
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        flush(None)
        for fd, saved in zip((1, 2), kept, strict=True):
            os.dup2(saved, fd)
        for thread in threads:
            thread.join(timeout=5.0)
        for saved in kept:
            os.close(saved)


class Session:
    """One card: `card` selects it, `device` uses an open one; `grid_timeout` bounds
    every grid wait. As a context manager it stops a loaded run at the exit (through
    the runtime on an exception) and closes the device it opened; so do the process's
    exit and SIGTERM. km.Session (komorebi.sessions) gives islands an IslandSession."""

    def __init__(
        self,
        card=None,
        device=None,
        pll_mhz=None,
        tree=None,
        log=None,
        grid_timeout=10.0,
        kernel_bytes=None,
        channel=channel_module.REGION_DEFAULT,
        slow_code=False,
        expect=None,
    ):
        self.card = card
        self.device = device
        self._own_device = device is None
        self.pll_mhz = X280_MHZ if pll_mhz is None else pll_mhz
        self._default_clock = pll_mhz is None  # may fall back to X280_FALLBACK_MHZ
        self.tree = tree
        self.log = log or (lambda message: None)
        self.log_given = log is not None  # without one, a reopen is printed to stderr
        self.grid_timeout = grid_timeout
        self.kernel_bytes = kernel_bytes  # None: the runtime's, then the kernels'
        self._kept = kernel_buffer.Kept()  # the tensors made before the first load
        self._dram = extents.Registry()  # its DRAM tensors: KM_GUARD's page bounds
        self.page_bounds = extents.MOST  # the most a load's list holds (0: none)
        self.slow_code = slow_code  # slots may end above L1_FAST_CODE
        self._worker_l1 = sizes.worker_l1_size(kernel_bytes)
        if device is not None and kernel_bytes is not None:
            raise ValueError("kernel_bytes is set when the session opens the device")
        self._memory = None  # the remembered kernel buffer (komorebi.device.remembered)
        self._last_total = 0  # the kernels' total of the last load built
        self._mock_channel = False  # a prebuild: the card would map the channel
        self._transfer_built = False  # a prebuild: the transfer kernel built
        self._built_before = []  # (found, compiled) built before the next load
        if kernel_bytes is None and device is None:
            self._memory = remembered.Memory(self.log, expect)
            if self._memory.size and self._memory.size <= sizes.KERNEL_BUFFER_MAX:
                self._worker_l1 = sizes.worker_l1_size(self._memory.size)
        self.run = None
        self.island = None  # (graph, i) of the loaded run: Session.load's `island`
        self.link_area = 0  # its link area's L1 address (Session.load's `link`), or 0
        self.link_plan = None  # Session.load's `link`
        self.link_plans = None  # per island LinkPlan of a load over a split card
        self.packet = 0  # an island session's link packet bytes (0: none)
        self.transfer_chunk = None  # a larger transfer chunk asked for, where it fits
        self.tensor_table = 0  # its tensor table's L1 address (load's `tensors`)
        # the owner's L1 on every core of this session's card, kept at the top (an
        # island session's link areas, tensor table): () -> [(name, bytes)]
        self.l1_held = lambda: []
        self._taken = None  # l1_taken() as a load found it before its own L1
        self.stuck = False  # the grid did not return: the runtime closes it at exit
        self._module = None
        self._starting = None  # the Dispatch of a load before its Run exists
        # the host channel: the region asked for, its bytes once mapped (else None)
        self._region = channel_module.check_region(channel) if channel else None
        self.channel = None
        self._channel_said = False  # the fallback has been logged
        # a given device's channel is mapped at the first transfer (open() maps its own)
        self._channel_mapped = device is None
        self.largest_page = 0  # bytes of the largest page of a DRAM tensor made so far
        self.transfer_cores = channel_module.CORES_DEFAULT  # cores of a transfer, most
        self.transfer_cycles = None  # the last transfer's span on the cores

    def __enter__(self):
        if self.device is None:
            self.open()
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if self.run is not None and not self.run.stopped:
                if kind is None:
                    self.run.stop()
                else:
                    self.run.abort()
        finally:
            self.close()

    def open(self):
        """Opens the card (slow dispatch) if the session was not given a device."""
        if self.card is not None:
            visible = os.environ.get("TT_VISIBLE_DEVICES")
            if visible is not None and visible.split(",")[0] != str(self.card):
                raise ValueError(f"card {self.card} is not visible ({visible})")
            os.environ["TT_VISIBLE_DEVICES"] = str(self.card)
        os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
        if process.exit_timeout(os.environ, self.log) == 0:
            self.log(
                f"{process.OP_TIMEOUT}=0: an exit after a grid that did not return "
                "hangs"
            )
        card = self.card if self.card is not None else process.visible_card()
        holder = process.chip_holder(card)
        if holder is not None:
            raise CardBusy(
                f"card {card} is held by process {holder} "
                f"({process._command(holder)}): an open would wait until that process "
                "exits",
                pid=holder,
            )
        process.refuse_reopen("alone", [card])
        self._module = self._module or native.load_module(self.tree)
        if self._region is None:
            self.device = self._module.Device(self._worker_l1)  # program cache off
        else:
            with _output_seen() as seen:  # the driver's warning, if it gives one
                self.device = self._module.Device(self._worker_l1)
        process._hook_exit()
        process._LIVE.add(self)
        self._own_device = True
        if self._region is not None:
            self._open_channel(b"".join(seen).decode(errors="replace"))

    def _map_channel(self):
        """A session given its device maps the channel at its first transfer. It did
        not see the open, so the driver's warning is unknown to it: a transfer whose
        sequence words do not arrive raises ChannelError."""
        if self._channel_mapped or self._region is None:
            return
        self._channel_mapped = True
        self._open_channel("")

    def _open_channel(self, said):
        """Maps the region of the card's channel, or logs why reads go the device API's
        way; ChannelError (the device closed) if the open logged the warning."""
        if mock.active():  # the card's would be mapped: its loads are built so
            self._mock_channel = True
            return
        warning = channel_module.warning_in(said)
        if warning is not None:
            self.close()
            raise ChannelError(
                f"the device open logged '{warning}': the cores' writes to the host "
                "channel would not land in this process's mapping; close the other "
                "process holding the card's memory, or open with channel=0"
            )
        with open("/proc/self/maps") as maps:
            address, where = channel_module.find(
                maps.read(), self.device.host_channel()
            )
        if address is None:
            self.log(f"host channel not used ({where}): reads go through the device")
            self._channel_said = True
            return
        self.channel = channel_module.view(address, self._region)
        channel_module.words_view(
            self.channel, self._region, abi.headers()["KM_CORES_MAX"]
        )[:] = 0
        self.log(
            f"host channel {where}: a region of {self._region.bytes >> 20} MiB from 0"
        )

    def close(self):
        """Stops a loaded run through the X280 runtime, then closes the device if the
        session opened it. A session on a given device is closed before that device."""
        if self._memory is not None:
            self._memory.closed()
        try:
            if self.run is not None and not self.run.stopped:
                self.run.abort()
        finally:
            if self.run is None or self.run.stopped:  # else the runtime needs it
                self._close_device()

    def _close_device(self):
        if self.stuck:
            self.log(
                "device not closed by the session: the grid did not return; at the "
                f"process's exit the runtime waits {process.exit_wait()} for the "
                "cores, then puts them in reset (a stuck thread recovers at the next "
                "open, a compute unit left running does not)"
            )
            return
        if self.device is not None and self._own_device:
            self.device.close()
            self.device = None
        process._LIVE.discard(self)

    def _reopen_device(self, worker_l1):
        """Closes the card and opens it again with the runtime's `worker_l1` (a new
        kernel buffer, kernel_buffer.reopen); the channel's mapping goes with it."""
        self.channel = None
        self._close_device()
        self._worker_l1 = worker_l1
        self.open()

    def _card(self):
        """What a backend of this session's load reads of it (backend.card.Card): the
        load's values, fixed before the backend is made, and the session's calls."""
        return Card(
            log=self.log,
            module=self._module,
            tree=self.tree,
            device=self.device,
            island=self.island,
            link_area=self.link_area,
            link_plan=self.link_plan,
            link_plans=self.link_plans,
            packet=self.packet,
            tensor_table=self.tensor_table,
            page_bounds=self.page_bounds,
            slow_code=self.slow_code,
            grid_timeout=self.grid_timeout,
            transfer_chunk=self.transfer_chunk,
            channel=self.channel is not None or self._mock_channel,
            tensor=self.tensor,
            empty=self.empty,
            write=self.write,
            l1=self.l1,
            l1_rows=self.l1_rows,
            l1_held=self.l1_held,
            l1_taken=self.l1_taken,
            nocs=self.nocs,
            page_bounds_list=self.page_bounds_list,
            largest_page=lambda: self.largest_page,
            hand_nocs=self._hand_nocs,
            resize=self._resize,
            mark_stuck=self._mark_stuck,
        )

    def _mark_stuck(self):
        """The grid did not return after a stop: the runtime closes it at exit."""
        self.stuck = True

    def _release(self, why, close_device):
        """The exit hooks' timed clean-up of this session: closes the X280 runtime of
        a load in progress or a loaded run, then (`close_device`) closes the session.
        Errors are logged, not raised: the process is ending. The close runs even when
        the stop raised."""
        try:
            if self._starting is not None:
                self.log(f"{why}: closing the X280 runtime of the load in progress")
                if not self._starting.close():
                    self.log(f"{why}: not restored: {self._starting.close_failed}")
            if self.run is not None and not self.run.stopped:
                self.log(f"{why}: stopping the loaded run through the X280 runtime")
                self.run.abort()
        except Exception as error:  # noqa: BLE001 (the process ends: logged)
            self.log(f"{why}: clean-up raised {type(error).__name__}: {error}")
        finally:
            if close_device:
                try:
                    self.close()
                except Exception as error:  # noqa: BLE001 (as above)
                    self.log(f"{why}: close raised {type(error).__name__}: {error}")

    @property
    def grid(self):
        """The Tensix grid (x, y) of the open device."""
        return tuple(self.device.grid())

    def tensor(
        self, data, dtype, layout="row", shard_grid=None, transposed_tiles=False
    ):
        """A tensor on the card holding `data` (anything np.asarray takes) as `dtype`
        (formats.DTYPES; a block format in tiles) in `layout` ("row" or "tile"), each
        tile transposed if `transposed_tiles`; in DRAM, or a row in each core's L1 of
        `shard_grid`. Written before the grid starts, or (DRAM) while it runs."""
        self._module = self._module or native.load_module(self.tree)
        raw, shape = formats.to_bytes(data, dtype, layout, transposed_tiles)
        tensor = tensor_module.empty(
            self._module, self.device, shape, dtype, layout, shard_grid
        )
        tensor.transposed_tiles = transposed_tiles
        self._note_page(tensor)
        if self._transfer_way([tensor], 0, None) is None:
            tensor.buffer.write(raw)
        else:
            self.write(tensor, raw.reshape(tensor.pages, tensor.page_bytes))
        self._kept.keep(tensor, data=(data, transposed_tiles))  # after: not the raw
        self._dram.add(tensor)
        tensor.loaded = self._run_loaded
        return tensor

    def empty(
        self, shape, dtype, layout="row", shard_grid=None, address=None, own=False
    ):
        """A tensor as tensor() makes it, not written. `own`: a region the runtime
        makes for itself, written by bank address: no transfer chunk follows its page.
        """
        self._module = self._module or native.load_module(self.tree)
        tensor = tensor_module.empty(
            self._module, self.device, shape, dtype, layout, shard_grid, address
        )
        self._kept.keep(tensor, address)
        self._dram.add(tensor)
        if not own:
            self._note_page(tensor)
        tensor.loaded = self._run_loaded
        return tensor

    def _run_loaded(self):
        """Whether a run of this session is loaded and not stopped: Tensor.read()
        refuses then (tensor.refuse_live)."""
        return self.run is not None and not getattr(self.run, "stopped", False)

    def _note_page(self, tensor):
        """Keeps the largest page of a DRAM tensor: a load sizes the run's transfer
        buffer by it."""
        if tensor.in_dram:
            self.largest_page = max(self.largest_page, tensor.aligned_page_bytes)

    def launch(self):
        """An empty native Launch on the open device, for a host-side runner that
        builds and launches a program of its own (a test feeder) instead of load()."""
        self._module = self._module or native.load_module(self.tree)
        return self._module.Launch(self.device)

    def l1(self):
        """(kernel buffer, L1 left for buffers) of each core in bytes, from the
        runtime's layout of the open device: what a larger buffer costs the buffers."""
        self._module = self._module or native.load_module(self.tree)
        kernel, free, _ = self._module.kernel_buffer(self.device)
        return int(kernel), int(free)

    def l1_rows(self, program, trace=0, placed=None, buffers=None):
        """A core's L1 budget rows for `program` loaded with `trace`, the slots aside:
        its buffers (`buffers` bytes if given) with the token and trace pages, the
        control block and overlay area, the host transfer buffer, what the session
        keeps. The load's table is these and the slots; a class's build checks them."""
        abi_headers = abi.headers()
        trace_cap = (trace + 3) // 4 * 4 if trace else 0
        if buffers is None:
            buffers = sum(b.page_bytes * b.pages for b in program.buffers)
        buffers += 2 * abi_headers["kTokenBytes"] + trace_cap * 16
        control = tensor_bytes._control_bytes(kernel_module.slots_most(program, placed))
        rows = [
            ("circular buffers", buffers),
            ("control block and overlay area", control),
        ]
        # A mock device maps no host channel: its budget holds the card's buffer.
        mock = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))
        if self.channel is not None or program.threads or mock:
            held = placed.programs if placed else [program]
            switch = bool(placed and placed.switch)
            chunk = tensor_bytes._transfer_chunk_of(
                self.largest_page, self.l1, self.transfer_chunk, held, switch,
                trace_cap, control,
            )  # fmt: skip
            size = 2 * chunk + abi_headers["KM_TRANSFER_WORD_BYTES"] + sizes.NOC_ALIGN
            rows.append(("host transfer buffer", size))
        messages = getattr(program, "messages", None)
        if messages is not None and messages.direct:
            cores = self.grid[0] * self.grid[1]
            rows.append(
                ("direct messages (tables, slots)", messages.direct_bytes(cores))
            )
        bounds = self.page_bounds_list(placed.programs if placed else [program])
        if bounds is not None:
            rows.append(("page bounds list", bounds[0].nbytes))
        return rows + list(self.l1_held()) + self.l1_taken()

    def page_bounds_list(self, programs):
        """KM_GUARD's list of this session's DRAM tensors for a load of `programs`
        (komorebi.device.extents.listed); None: none (every program unguarded, or the
        load's tensor table bounds them, or page_bounds 0)."""
        if not self.page_bounds or not any(getattr(p, "guard", True) for p in programs):
            return None
        if any(name.startswith("session: tensor table") for name, _ in self.l1_held()):
            return None
        return extents.listed(self._dram.live(), self.page_bounds)

    def _hand_nocs(self, program):
        """A program written as a class whose rectangle needs the grid's NoCs gets
        this session's (komorebi.dsl.program.session_nocs)."""
        if isinstance(program, programs.Plan):
            loaded = program.programs
        else:
            loaded = list(program) if isinstance(program, dict | list) else [program]
        loaded = [  # and a merged program's sources made apart (komorebi.merge)
            each
            for one in loaded
            for each in [one, *getattr(one, "merged_sources", ())]
            if isinstance(each, Program)
        ]
        needing = [one for one in loaded if getattr(one, "rectangle", None)]
        if needing:
            nocs = self.nocs()
            for one in needing:
                session_nocs(one, nocs, self.island)

    def nocs(self):
        """The grid's NoC coordinates by core (a layout.Nocs), from the open device:
        for program arguments that name other cores, before load()."""
        self._module = self._module or native.load_module(self.tree)
        grid_x, grid_y, table, noc0 = self._module.core_nocs(self.device)
        return layout.nocs((grid_x, grid_y), table, noc0)

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
        (also while a run is loaded); without it only a whole tensor, before a load."""
        if tensor.kept is not None:  # the array given, for a reopened card
            tensor.kept.append(("pages", pages, first))
        pages = np.ascontiguousarray(pages, np.uint8)
        if pages.ndim != 2 or pages.shape[1] != tensor.page_bytes:
            raise ValueError(f"pages are [n, {tensor.page_bytes}] uint8 rows")
        start, count = dram.page_range(first, len(pages), tensor.pages)
        self._module = self._module or native.load_module(self.tree)
        pieces = self._live_pieces(tensor)
        if pieces and pieces > 1:  # a page wider than the run's chunk: in pieces
            self._write_pieces(tensor, pages, start, pieces)
            return
        way = self._transfer_way([tensor], start, count)
        if way is None:
            if self._mock_channel and (count != tensor.pages or self.run is not None):
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
        for at, part in tensor_bytes.split_pages(start, rows, pieces):
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
            if self._mock_channel:  # the card would launch the transfer kernel
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
        channel_module.check(transfer, count, sizes.L1_BYTES)
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
            channel_module.check(transfer, pages, sizes.L1_BYTES)
            words = channel_module.words_view(self.channel, region, pages)
            words[:] = 0
            want = channel_module.sequence_words(transfer.nonce, pages, page, pages)
            self._launch_transfer(transfer, pages, push, words, want, timeout, tensor)
            self.transfer_cycles = channel_module.cycles(words)
            return
        run = self.run
        handler = h["KM_HANDLER_PUSH" if push else "KM_HANDLER_PULL"]
        here = self.island[0].first(self.island[1]) if self.island else 0
        self.transfer_cycles = 0
        for at, count, rank, field in self._l1_ranks(tensor, first, pages):
            transfer = channel_module.Transfer(
                tensor.address, page, at, count, (at - first) * page, region.words,
                region.bytes, secrets.randbits(32), run.backend.transfer.address,
                h["KM_TRANSFER_L1"],
            )  # fmt: skip
            channel_module.check(transfer, count, sizes.L1_BYTES)
            words = channel_module.words_view(self.channel, region, count)
            words[:] = 0
            want = channel_module.sequence_words(transfer.nonce, count, page, count)
            run.transfer(handler, (rank + here, count, field), transfer.args(), timeout)
            if not self._words_arrived(words, want, timeout):
                raise ChannelError(self._words_missing(words, want, timeout))
            self.transfer_cycles += channel_module.cycles(words)

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
        launch.cb(cores, 0, "uint32", 16, 2 * chunk + 64)
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
            rows, cols = tensor.shape
            out.append(dram.untilize_bf16(pages, rows, cols))
        return out

    def read_tiles_bf16(self, tensor, first, pages):
        """Tiles first .. first + pages - 1 of a bfloat16 tile tensor (one tile a page,
        row-major tile order) as float32 [pages, 32, 32], read as read() does."""
        if tensor.layout != "tile" or tensor.dtype != "bfloat16":
            raise ValueError("read_tiles_bf16 takes a bfloat16 tile tensor")
        (raw,) = self.read(tensor, first=first, pages=pages)
        tile = dram.TILE
        return dram.untilize_bf16(raw, len(raw) * tile, tile).reshape(-1, tile, tile)

    def _open_dispatch(self, module):
        """The runtime's dispatch at the session's clock. A runtime that does not offer
        the default clock gets 1000, logged once; an asked-for clock is an error."""
        if mock.active():  # a mock has no X280
            return mock.Dispatch(self.device)
        try:
            return module.Dispatch(self.device, self.pll_mhz)
        except RuntimeError as error:
            if not (self._default_clock and NO_PLL_SOLUTION in str(error)):
                raise
            self.log(
                f"X280 clock {self.pll_mhz} MHz is not offered by this runtime "
                f"({error}); using {X280_FALLBACK_MHZ} MHz"
            )
        self.pll_mhz = X280_FALLBACK_MHZ
        return module.Dispatch(self.device, self.pll_mhz)

    def load(self, program, *args, **options):
        """Launches `program` (a list or {program: (first, count)}: the runtime's form)
        on every core, then the dispatcher and the X280 thread; returns a Run
        (docs/reference/host.md). A session not given `kernel_bytes` opens its card
        again first when the kernels need another kernel buffer (kernel_buffer)."""
        loading, self._kept.loading = self._kept.loading, True
        try:
            run = self._try_load(program, *args, **options)
        except kernel_buffer.Resize as resize:
            size, why = resize.size, resize.why
        else:
            self._kept.stop()
            self._remember(program)
            return run
        finally:
            self._kept.loading = loading
        kernel_buffer.reopen(self, size, sizes.worker_l1_size(size), why)
        self._kept.loading = True
        try:
            run = self._try_load(program, *args, **options)
        finally:
            self._kept.loading = loading
        self._kept.stop()
        self._remember(program)
        return run

    def _remember(self, program):
        """The kernel buffer a load needed, kept for the script's next run."""
        if self._memory is not None and self._last_total:
            self._memory.loaded(program, self._last_total)

    def _try_load(self, program, *args, **options):
        """_load with the L1 the session held before it (l1_taken) fixed for its
        table: the load's own L1 is counted by its own rows."""
        self._taken = self.l1_taken()
        try:
            return self._load(program, *args, **options)
        finally:
            self._taken = None

    def l1_taken(self):
        """The L1 budget table's row of the L1 this session already holds on every
        core (its L1 tensors, a run not stopped): the L1 top down to the lowest L1
        buffer, less what the owner keeps (l1_held). [] when none."""
        if self._taken is not None:
            return self._taken
        if self.device is None:
            return []
        self._module = self._module or native.load_module(self.tree)
        lowest = int(self._module.lowest_l1(self.device))
        if not lowest:
            return []
        top = int(self._module.kernel_buffer(self.device)[2])
        held = top - lowest - sum(size for _, size in self.l1_held())
        return [(TAKEN_ROW, held)] if held > 0 else []

    def _resize(self, total, l1_failed=False):
        """Raises kernel_buffer.Resize when this session follows its kernels (no
        `kernel_bytes` given, its own card alone) and kernels of `total` bytes need
        another buffer: a larger one, or a smaller one when the L1 table failed.
        Returns why it does not follow them, or None."""
        size, buffer = kernel_buffer.need(total), self.l1()[0]
        grow = size > buffer
        if not (grow or (l1_failed and size < buffer)):
            return None
        refused = (
            "the session was given kernel_bytes"
            if self.kernel_bytes is not None
            else (
                "it did not open its card"
                if not self._own_device or self.island is not None
                else (
                    f"{size} B is over the most, {sizes.KERNEL_BUFFER_MAX} B"
                    if size > sizes.KERNEL_BUFFER_MAX
                    else self._kept.refused()
                )
            )
        )
        if refused is not None:
            if self._memory is not None and grow:  # the next run opens right
                self._memory.wanted(total)
                refused += " (remembered: the script's next run opens at it)"
            return f"{refused}; Session(kernel_bytes={size}) holds them"
        why = (
            f"the kernels take {total:,} B"
            if grow
            else f"the L1 table does not fit and the kernels take {total:,} B"
        )
        raise kernel_buffer.Resize(size, why)

    def _load(
        self,
        program,
        depth=None,
        trace=0,
        deadline=1.0,
        wait=2.0,
        idle=0.0,
        kernel_idle=0.0,
        start=10.0,
        host_depth=64,
        event_depth=128,
        batch=64,
        settle_microseconds=1.0,
        image=None,
        thread="card",
        island=None,
        link=None,
        tensors=0,
        recover=True,
        reserve=added.RESERVE,
        buffers=None,
    ):
        """Session.load's one try on the card as it is opened."""
        arguments = {k: v for k, v in locals().items() if k != "self"}
        fits = getattr(program, "fits_kernels", None)  # a merged class's: a load again
        if fits is not None and not program.fitting:  # with a handler as an overlay
            return fits(self, arguments)
        if reserve < 0:
            raise ValueError(f"reserve={reserve}: bytes, 0 or more")
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        self.island = kernel_module._island(island, self.grid)
        if link is not None and self.island is None:
            raise ValueError("link= is an island's (island= too)")
        self.link_plan = link
        self.link_area = link.area if link is not None else 0
        self.tensor_table = tensors
        self._hand_nocs(program)
        placed = None
        if isinstance(program, programs.Plan):  # an island's part of a flat plan
            placed = program
            program = placed.programs[0]
        elif not isinstance(program, Program) or program.form() == "program":
            grid = self.grid
            placed = programs.plan(program, grid[0] * grid[1])
            program = placed.programs[0]
        owner = placed.threads_owner() if placed is not None else program
        if thread not in ("card", "host"):
            raise ValueError(f"thread={thread!r}: 'card' or 'host'")
        works = [owner.threads[cluster] for cluster in sorted(owner.threads)]
        owner.check_threads()
        if owner.x280_ports is not None and (not works or thread != "card"):
            raise ValueError(
                f"{owner.name}: X280 ports need its X280 threads on the card"
            )
        for work in works:
            if thread == "host" and work.host_fallback is None:
                raise ValueError(
                    f"{owner.name}: the thread on cluster {work.cluster} has no "
                    "host_fallback"
                )
        several = (
            placed.programs if placed is not None and len(placed.programs) > 1 else ()
        )
        for work in works:  # a class's thread: its header with the load's programs
            rewritten = getattr(work, "programs_header", None)
            if rewritten is not None and thread == "card":
                rewritten(several)
        image = Path(image) if image else native.image_path(self.tree)
        if not image.is_file() or image.stat().st_size == 0:
            raise StartError(f"dispatcher image {image}: missing or empty")
        module = self._module = native.load_module(self.tree)
        on_card = bool(works) and thread == "card"
        images = (
            {
                work.cluster: native.thread_image(work, self.tree, several)
                for work in works
            }
            if on_card
            else {}
        )
        ranks = x280_module._x280_ranks(program, owner, placed, on_card)
        depth = depth or program.depth
        sources = program.sources()  # before anything opens: a missing file raises
        if placed is not None:
            sources = programs.runtime_kernels()
        mhz = mock.MHZ if mock.active() else process.aiclk_mhz(self.card)
        # the regions go to the card before the grid starts: a tensor write waits
        # for every core to be idle, which a persistent grid never is
        regions = x280_module._bank_regions(self, owner, works)
        dispatch = self._open_dispatch(module)
        self.log(f"X280 clock set: {self.pll_mhz} MHz")
        process._hook_exit()
        process._LIVE.add(self)
        self._starting = dispatch
        dispatch.settle_microseconds = settle_microseconds
        backend, launched = None, False
        try:
            if not dispatch.idle:
                raise StartError("X280 cluster 0 is not idle: no image can start")
            state_tensor = tensor_bytes._state_area(self)
            rings = {
                work.cluster: x280_module._work_ring(dispatch, work)
                for work in works
                if on_card
            }
            ends = {cluster: end for cluster, (_, end) in rings.items()}
            tag = dispatch.prepare(
                depth,
                host_depth,
                event_depth,
                start,
                deadline,
                kernel_idle,
                state_tensor.address,
                **thread_module.kernel_work_args(ends),
                messages_handler=(
                    program.messages.dispatcher_word("KM_DELIVER" in program.defines)
                    if program.messages is not None
                    else 0
                ),
                programs=(
                    kernel_module._held_fields(
                        placed,
                        kernel_module._link_kernels(
                            self.island,
                            program,
                            None,
                            dispatch.grid_x * dispatch.grid_y,
                        )[1],
                    )
                    if placed and placed.switch
                    else []
                ),
                island=kernel_module._island_word(self.island),
                link=list(link.line(dispatch)) if link is not None else [],
                channel_region=self._channel_region(),
            )
            kernel_module.check_bases(kernel_module.lim_bases(dispatch.lim_base))
            # an island's ports serve its ranks, not its link cores
            served, _ = kernel_module._link_kernels(
                self.island, program, None, dispatch.grid_x * dispatch.grid_y
            )
            plan, port_lines = x280_module._port_lines(owner, dispatch, rings, served)
            backend = _Backend(self._card(), dispatch, mhz, trace)
            backend.state_tensor = state_tensor  # kept while the kernel may use it
            backend.reserve = reserve
            backend.added_buffers = added_program.buffers_given(buffers)
            if ranks is not None:
                ranks = x280_module._x280_words(
                    self, program, dispatch, rings, regions, plan
                )
            descriptor = backend.describe(
                program, sources, tag, depth, wait, idle, trace, port_lines, placed,
                ranks,
            )  # fmt: skip
            started, built_at = time.perf_counter(), time.time()
            try:
                elves, *build = descriptor.build()
            except Exception as error:
                raise ProgramError(f"{program.name}: build failed: {error}") from error
            program.sizes = kernel_module.kernel_sizes(build)
            self._last_total = program.sizes.total
            for other in placed.programs if placed is not None else ():
                other.sizes = program.sizes
            text = ", ".join(
                f"{name} {size}" for name, size in program.sizes.text.items()
            )
            self.log(
                f"kernel buffer: {program.sizes.total} of {program.sizes.limit} B "
                f"(text: {text})"
            )
            if build[-1]:
                self._log_built(descriptor, built_at, program)
                refused = self._resize(program.sizes.total or sizes.KERNEL_BUFFER_MAX)
                raise KernelSizeError(
                    program.name, program.sizes, sizes.KERNEL_BUFFER_MAX, refused
                )
            if insn_check.enabled():
                insn_check.check_kernels(
                    program.name,
                    elves,
                    native.metal_tree(self.tree),
                    native.cache_directory() / "insn-check",
                )
                self.log(
                    f"kernels built and checked ({time.perf_counter() - started:.2f} s)"
                )
            if placed is not None:
                backend.load_programs(placed, descriptor)
            else:
                backend.load_overlays(program, descriptor)
            self._log_built(descriptor, built_at, program)
            if mock.active():  # built: a mock run stands in for the launch
                self._starting = None
                self.run = mock.Run(counted=self.island is None)
                return self.run
            if backend.apart_links:  # after every other write: a transfer's buffer
                link.clear(self)  # starts at the link cores' buffers' base too
            self.log("launch")
            # From here the grid may run: an interrupt is delivered when the launch
            # call returns, after the kernels started.
            launched = True
            try:
                descriptor.run()
            except Exception as error:
                raise ProgramError(f"{program.name}: launch failed: {error}") from error
            backend.launch = descriptor  # its program: a core's kernels reloaded
            dispatch.start(str(image))
            self.log("dispatcher started")
            state = dispatch.wait_ready(start + 1.0)
        except BaseException:
            try:
                if not dispatch.close():
                    self.log(f"x280 runtime not restored: {dispatch.close_failed}")
            except BaseException:
                if launched:  # the runtime stays open: the exit hooks close it again
                    backend.stop_grid()
                raise
            self._starting = None
            if launched:
                backend.read_back(stop_grid=True)  # the kernel never stopped it
            raise
        run = Run(
            backend,
            (dispatch.grid_x, dispatch.grid_y),
            program,
            batch,
            programs=placed.programs if placed is not None else None,
            holds=placed.holds() if placed is not None else None,
            thread_program=owner,
            switch=getattr(backend, "switch", None),
            ranks=self.island[0].ranks if self.island else None,
            island_first=self.island[0].first(self.island[1]) if self.island else 0,
            island_ranks=self.island[0].count(self.island[1]) if self.island else None,
        )
        self.run = run
        self._starting = None
        if state != abi.headers()["KMD_STATE_READY"]:
            run._fail_start(state)
        if placed is not None and placed.switch:
            kernel_module._start_loads(run, placed, part=self.island is not None)
        if recover and not (placed is not None and placed.switch):
            # an island's link cores keep their held connection: no reset list
            exclude = ()
            if self.island is not None:
                graph, here = self.island
                exclude = set(range(run.cores)) - set(graph.rank_cores[here])
            if run.keep_resets(exclude):
                self.log(
                    f"reset lists kept: {len(run.recovery['lists'])}, "
                    f"{run.recovery['bytes']} B in "
                    f"{run.recovery['seconds'] * 1e3:.1f} ms"
                )
            else:
                self.log(f"no reset list: {run.recovery_refused}")
        elif recover:
            run.recovery_refused = "a run whose groups switch program"
        if unguarded(run.programs):
            self.log(guard_says(unguarded(run.programs), recover))
        self.log("ready")
        if works:
            shared = {
                name: x280_module._host_array(value)
                for name, value in owner.shared.items()
            }
            peers = x280_module._peers(self, dispatch, images) if on_card else {}
            switch = getattr(backend, "switch", None)
            records = thread_module.load_records(switch) if on_card else None
            if records is not None:  # a thread may load: reason 13 is the judge
                run.check_programs = False
            try:
                for work in works:
                    cluster = work.cluster
                    view = regions[work.bank].view(work)
                    thread_run = thread_module.ThreadRun(
                        work, dispatch, view, images.get(cluster), on_card, view.base,
                        view.check, run, shared,
                    )  # fmt: skip
                    thread_run.ring = rings[cluster][0] if on_card else None
                    thread_run.peers = peers.get(cluster)
                    thread_run.records = records
                    if self.island is not None:
                        graph, here = self.island
                        thread_run.island = here | len(graph.islands) << 16
                    if backend.transfer is not None:
                        thread_run.buffer = backend.transfer.address
                    if plan is not None:
                        thread_run.ports = thread_module.Ports(
                            backend.control_addr,
                            len(port_lines.cores),
                            port_lines.cores,
                            plan,
                            owner.x280_ports["noc"],
                        )
                        thread_run.write_ports()
                    if ranks is not None and cluster == program.messages.x280_cluster:
                        thread_run.ranks = x280_module._x280_line(
                            program, dispatch, backend, ranks, wait, mhz
                        )
                        thread_run.write_ranks()
                    run.threads[cluster] = thread_run
                for (
                    thread_run
                ) in run.threads.values():  # every ports area is written first
                    thread_run.start()
            except BaseException:
                run.abort()
                raise
            run.thread = run.threads[works[0].cluster]
            where = "on the card" if on_card else "on the host"
            self.log(f"threads {list(run.threads)} {where}")
        return run

    def load_islands(
        self,
        program,
        graph,
        depth=None,
        trace=0,
        deadline=1.0,
        wait=2.0,
        idle=0.0,
        kernel_idle=0.0,
        start=10.0,
        host_depth=64,
        event_depth=128,
        batch=64,
        settle_microseconds=1.0,
        image=None,
        links=None,
        tensors=0,
    ):
        """This card split into the islands of `graph` (whole rows, ranks then link
        cores): `program` launched once, island i's dispatcher on X280 cluster i,
        `links` each island's LinkPlan or None, `tensors` the tensor table's address.
        A Run an island over flat ranks; Session.load's options, no X280 threads."""
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        if not isinstance(program, Program) or program.form() == "program":
            raise ValueError("a card split into islands loads one Program")
        if (
            program.threads
            or program.x280_ports
            or (program.messages is not None and program.messages.x280_ranks)
        ):
            raise ValueError(
                f"{program.name}: X280 threads, ports and ranks need the work "
                "clusters, which a split card's dispatchers take"
            )
        rows = sizes._split_rows(graph, self.grid)
        self.link_plans, self.link_plan, self.link_area = links, None, 0
        self.tensor_table = tensors
        image = Path(image) if image else native.image_path(self.tree)
        if not image.is_file() or image.stat().st_size == 0:
            raise StartError(f"dispatcher image {image}: missing or empty")
        module = self._module = native.load_module(self.tree)
        depth = depth or program.depth
        sources = program.sources()
        mhz = mock.MHZ if mock.active() else process.aiclk_mhz(self.card)
        dispatches = kernel_module._Dispatches([self._open_dispatch(module)])
        self.log(f"X280 clock set: {self.pll_mhz} MHz")
        process._hook_exit()
        process._LIVE.add(self)
        self._starting = dispatches
        backends, launched = [], False
        try:
            for cluster in range(1, len(rows)):
                dispatches.append(
                    mock.Dispatch(self.device)
                    if mock.active()
                    else module.Dispatch(dispatches[0], cluster)
                )
            states, tags = [], []
            for island, (dispatch, (first_row, count)) in enumerate(
                zip(dispatches, rows)
            ):
                dispatch.settle_microseconds = settle_microseconds
                if not dispatch.idle:
                    raise StartError(f"X280 cluster {island} is not idle")
                states.append(tensor_bytes._state_area(self))
                base = first_row * self.grid[0]
                plan = links[island] if links else None
                line = b""
                if plan:
                    line = plan.line(dispatch, dispatches[plan.far], dispatches)
                tags.append(
                    dispatch.prepare(
                        depth,
                        host_depth,
                        event_depth,
                        start,
                        deadline,
                        kernel_idle,
                        states[-1].address,
                        messages_handler=(
                            program.messages.dispatcher_word(
                                "KM_DELIVER" in program.defines
                            )
                            if program.messages is not None
                            else 0
                        ),
                        island=kernel_module._island_word((graph, island)) | base << 32,
                        link=list(line),
                        first_row=first_row,
                        rows=count,
                        channel_region=self._channel_region(),
                    )  # fmt: skip
                )
                kernel_module.check_bases(kernel_module.lim_bases(dispatch.lim_base))
            if len({dispatch.lim_base for dispatch in dispatches}) != 1:
                raise StartError(  # a core's words name one LIM address on every one
                    "the islands' dispatchers differ in LIM base: "
                    f"{[hex(dispatch.lim_base) for dispatch in dispatches]}"
                )
            if len({dispatch.upload_bytes for dispatch in dispatches}) != 1:
                raise StartError(  # the link area after it: one place on every one
                    "the islands' dispatchers differ in upload region: "
                    f"{[dispatch.upload_bytes for dispatch in dispatches]}"
                )
            first = _Backend(self._card(), dispatches[0], mhz, trace)
            first.split = (graph, list(dispatches), tags, rows, set())
            first.state_tensor = states[0]
            descriptor = first.describe(
                program, sources, tags[0], depth, wait, idle, trace
            )
            built_at = time.time()
            try:
                elves, *build = descriptor.build()
            except Exception as error:
                raise ProgramError(f"{program.name}: build failed: {error}") from error
            program.sizes = kernel_module.kernel_sizes(build)
            self._last_total = program.sizes.total
            if build[-1]:
                raise KernelSizeError(
                    program.name, program.sizes, sizes.KERNEL_BUFFER_MAX
                )
            if insn_check.enabled():
                insn_check.check_kernels(
                    program.name,
                    elves,
                    native.metal_tree(self.tree),
                    native.cache_directory() / "insn-check",
                )
            first.load_overlays(program, descriptor)
            self._log_built(descriptor, built_at, program)
            if mock.active():  # built: a mock run an island for the launch
                self._starting = None
                runs = [mock.Run(counted=False) for _ in rows]
                self.run, self.island = runs[0], (graph, 0)
                return runs
            for island, dispatch in enumerate(dispatches):
                backend = first if island == 0 else copy.copy(first)
                backend.port, backend.state_tensor = dispatch, states[island]
                backend.rows = rows[island]
                backends.append(backend)
            self.log(f"launch: one grid, {len(rows)} islands")
            launched = True
            try:
                descriptor.run()
            except Exception as error:
                raise ProgramError(f"{program.name}: launch failed: {error}") from error
            for island, (backend, dispatch) in enumerate(zip(backends, dispatches)):
                backend.launch = descriptor
                dispatch.start(str(image))
                self.log(
                    f"island {island}: dispatcher started on X280 cluster {island}"
                )
            ready = [dispatch.wait_ready(start + 1.0) for dispatch in dispatches]
        except BaseException:
            if not dispatches.close():
                self.log(f"x280 runtime not restored: {dispatches.close_failed}")
            self._starting = None
            if launched:
                for backend in backends:  # the kernels never stopped the grid
                    backend.read_back(stop_grid=True)
            raise
        runs = []
        for island, backend in enumerate(backends):
            run = Run(
                backend,
                self.grid,
                program,
                batch,
                ranks=graph.ranks,
                island_first=graph.first(island),
                island_ranks=graph.count(island),
                island_cores=(rows[island][0] * self.grid[0], graph.count(island)),
                island_span=rows[island][1] * self.grid[0],
            )
            run.recovery_refused = "a card split into islands"
            runs.append(run)
        self.run, self.island = runs[0], (graph, 0)
        self._starting = None
        for run, state in zip(runs, ready):
            if state != abi.headers()["KMD_STATE_READY"]:
                for other in runs:
                    if other is not run:
                        other.abort()
                run._fail_start(state)
        self.log("ready: every island")
        return runs
