"""A card session: opens the device, loads a program as resident workers on every
Tensix core and the dispatch kernel on X280 cluster 0, and hands out a `Run`.

Needs a tt-metal tree or the runtime package; `import komorebi` does not import it.
"""

import atexit
import contextlib
import os
import re
import signal
import time
import weakref
from pathlib import Path
from typing import NamedTuple

import numpy as np

from komorebi import abi, dram, insn_check, layout, native
from komorebi import state as st
from komorebi import tensor as kmt
from komorebi import thread as kt
from komorebi.errors import (
    CardBusy,
    GridTimeout,
    KernelSizeError,
    ProgramError,
    StartError,
)
from komorebi.program import KernelSizes
from komorebi.run import Run

CTL_ALIGN = 64  # the dispatch kernel stores whole 64-byte lines into the control block
IPI_ACK_DDR = 0x470020  # the X280 runtime's IPI ack word in the cluster's DRAM bank
DYN_DRAM_BYTES = 4 << 30  # the most DRAM a dynamic segment's queues may take
STATE_PAGES = 32  # pages of KMD_STATE_BYTES, interleaved: several in every DRAM bank
NOC_ALIGN = 16  # a NoC write's source and destination share their low 4 address bits
DRAM_BANKS = 8  # interleaved page i lies in bank i % DRAM_BANKS
# The X280 clock: at 1750 MHz, under full Tensix load (lowered core voltage), core 0's
# loads returned wrong values; 1000 MHz is the next step the runtime offers.
X280_MHZ = 1000
STATE_DUMP_LINES = 8  # 64-byte lines of the kernel's state logged after a failure
CHIP_LOCK = "/dev/shm/TT_UMD_LOCK.CHIP_IN_USE_{card}_PCIe"  # UMD's lock per card
CHIP_LOCK_BYTES, CHIP_LOCK_PID = 56, 52  # its size; the owner PID's byte offset
CBS = 64  # circular buffers of a Tensix core: one unpack-to-DST mode each
PROBE_GAP_S = 0.02  # core 0's pass stamp moves every 1,024 passes (about 33 us)
ABORT_GRID_S = 2.0  # an abort's wait for the grid after the stop word, before the close
REGION_WHOLE = 4 << 20  # a region up to this size is read in one read, else in chunks
REGION_CHUNK = 2 << 20  # of this size, with a log line each
# The runtime's bound on its waits for the cores, the one at device close included;
# unset, it is 0, which means no bound. Read when the process first opens a device.
OP_TIMEOUT = "TT_METAL_OPERATION_TIMEOUT_SECONDS"
OP_TIMEOUT_DEFAULT = 30.0  # the runtime's own example value
# A Blackhole Tensix core's L1 and where the runtime's kernel buffer starts in it
# (MEM_L1_SIZE, MEM_MAP_END); the buffer ends where the allocator's L1 base begins.
L1_BYTES = 1536 * 1024
KERNEL_BUFFER_BASE = 41088
KERNEL_BUFFER_DEFAULT = 69 * 1024  # the runtime's own (70,656 B)
KERNEL_BUFFER_ALIGN = 64  # the allocator's base is aligned up to this
KERNEL_BUFFER_MAX = L1_BYTES - KERNEL_BUFFER_BASE - KERNEL_BUFFER_ALIGN
TENSIX_THREADS = ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2")
OPT_DEFINE = "KM_OPT_LEVEL"  # defined to a kernel's level when the program sets one


# The runtime's ComputeConfigDescriptor fields that are flags.
COMPUTE_FLAGS = (
    "fp32_dest_acc_en",
    "dst_full_sync_en",
    "bfp8_pack_precise",
    "math_approx_mode",
    "enable_trisc2_rvv",
)


def compute_config(options):
    """The compute kernel's options for the native Launch from Program.compute: the
    runtime's ComputeConfigDescriptor fields (ProgramError names an unknown one);
    `math_fidelity` a MathFidelity name, `unpack_to_dest_mode` an UnpackToDestMode
    name for every circular buffer, or {cb: name} with the others Default."""
    options = dict(options)
    unpack = options.pop("unpack_to_dest_mode", None)
    fidelity = options.pop("math_fidelity", None)
    for name in options:
        if name not in COMPUTE_FLAGS:
            raise ProgramError(
                f"compute option {name!r}: no such ComputeConfigDescriptor field"
            )
    config = {
        "math_fidelity": fidelity,
        "flags": [(name, bool(value)) for name, value in options.items()],
    }
    if unpack is not None:
        by_cb = (
            unpack if isinstance(unpack, dict) else dict.fromkeys(range(CBS), unpack)
        )
        modes = ["Default"] * CBS
        for cb, name in by_cb.items():
            modes[cb] = name
        config["unpack_to_dest_mode"] = modes
    return config


# sessions holding a device they opened or an X280 runtime; weak, so that the exit
# hooks never keep a session alive
_LIVE = weakref.WeakSet()
_HOOKED = []  # [True] once the exit hooks are installed


def _release_all(why, close_device=True):
    """The exit hooks' clean-up: every live session's run is stopped through the X280
    runtime (bounded), then the device it opened is closed."""
    for session in list(_LIVE):
        session._release(why, close_device)


def _on_sigterm(signum, frame):
    """SIGTERM: the runtimes are closed while the devices are open, then the process
    exits through its `with` blocks and the exit hook, which close the devices."""
    _release_all("SIGTERM", close_device=False)
    raise SystemExit(128 + signum)


def _hook_exit():
    """atexit for every session; SIGTERM too when the process has no handler of its
    own (signals are set from the main thread only)."""
    if _HOOKED:
        return
    _HOOKED.append(True)
    atexit.register(_release_all, "exit")
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        with contextlib.suppress(ValueError):  # not the main thread
            signal.signal(signal.SIGTERM, _on_sigterm)


def _command(pid):
    """A process's command line, shortened, or "?"."""
    try:
        words = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return "?"
    return " ".join(w.decode(errors="replace") for w in words if w)[:160]


def aiclk_mhz(card=None):
    """The Tensix clock of the card this process uses, from the driver (MHz)."""
    card = card if card is not None else visible_card()
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    return int(path.read_text().split()[0])


def bound_exit(environ, log):
    """Sets OP_TIMEOUT to its default in `environ` when unset or empty and logs it; a
    value already there is kept. The bound in seconds (0: none; None: not a number)."""
    value = environ.get(OP_TIMEOUT, "")
    if value.strip():
        try:
            return max(float(value), 0.0)
        except ValueError:
            return None  # the runtime refuses it at the open
    environ[OP_TIMEOUT] = f"{OP_TIMEOUT_DEFAULT:g}"
    log(f"{OP_TIMEOUT} unset: set to {OP_TIMEOUT_DEFAULT:g} s (bounds the exit)")
    return OP_TIMEOUT_DEFAULT


def exit_wait(environ=None):
    """How long the device's close waits for the cores, as a phrase."""
    value = (os.environ if environ is None else environ).get(OP_TIMEOUT, "")
    try:
        seconds = float(value)
    except ValueError:
        seconds = 0.0
    return f"up to {seconds:g} s" if seconds > 0 else "without a bound"


def worker_l1_size(kernel_bytes):
    """The runtime's worker_l1_size for a kernel buffer of `kernel_bytes` (None: the
    default, nothing passed); a size the runtime would not give exactly is refused."""
    if kernel_bytes is None:
        return None
    if (
        not isinstance(kernel_bytes, int)
        or kernel_bytes % KERNEL_BUFFER_ALIGN
        or not KERNEL_BUFFER_ALIGN <= kernel_bytes <= KERNEL_BUFFER_MAX
    ):
        raise ValueError(
            f"kernel_bytes={kernel_bytes!r}: a multiple of {KERNEL_BUFFER_ALIGN} from "
            f"{KERNEL_BUFFER_ALIGN} to {KERNEL_BUFFER_MAX} (default "
            f"{KERNEL_BUFFER_DEFAULT})"
        )
    return L1_BYTES - KERNEL_BUFFER_BASE - kernel_bytes


def kernel_sizes(build):
    """A KernelSizes from the native kernel_build's (config bytes, buffer, text per
    thread, the runtime's message); the total from that message when it does not fit."""
    total, limit, text, error = build
    if error:
        said = re.search(r"Program size \((\d+)\) too large", error)
        total = int(said.group(1)) if said else None
    return KernelSizes(dict(zip(TENSIX_THREADS, text, strict=False)), total, limit)


def kernel_build(program, kernel, defines):
    """A kernel's defines, and its optimisation level when the program sets one. A set
    level also adds a define naming it: the runtime's build cache key leaves the level
    out, and two levels must not share a cached ELF."""
    level = program.opt.get(kernel)
    if level is None:
        return {"defines": defines}
    return {"defines": [*defines, (OPT_DEFINE, level)], "opt_level": level}


def visible_card():
    return int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])


def chip_holder(card, path=None):
    """The PID of another live process that holds `card`'s chip lock, else None. Read
    from UMD's lock file (a pthread mutex, a 64-bit word, the owner's TID and PID)
    without taking it; UMD keeps the lock from a process's first open to its exit."""
    # the file's mtime cannot date the PID: UMD writes the lock through a mapping,
    # which leaves the mtime unchanged
    path = Path(path or CHIP_LOCK.format(card=card))
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) != CHIP_LOCK_BYTES:
        return None
    pid = int.from_bytes(data[CHIP_LOCK_PID : CHIP_LOCK_PID + 4], "little", signed=True)
    if pid <= 0 or pid == os.getpid():
        return None
    try:  # the owner holds it only while it maps the file: a zombie, a reused PID
        # or a process that died holding the lock does not
        with open(f"/proc/{pid}/maps") as maps:
            return pid if any(str(path) in line for line in maps) else None
    except (FileNotFoundError, ProcessLookupError):
        return None  # the process ended
    except PermissionError:
        return pid  # alive, another user's


def _state_area(session):
    """A DRAM buffer whose bank address holds the dispatch kernel's state in every
    bank (KMD_STATE_BYTES at least); refused if it covers the runtime's IPI word."""
    h = abi.headers()
    rows = _words(session, np.zeros((STATE_PAGES, h["KMD_STATE_BYTES"] // 4)))
    at = rows.address
    if at <= IPI_ACK_DDR < at + STATE_PAGES * h["KMD_STATE_BYTES"]:
        raise StartError(f"the kernel state at {at:#x} covers the X280 IPI word")
    return rows


def lim_bases(lim_base, h=None):
    """The X280 local-memory addresses the kernels are given or derive from the
    runtime's base: the base, every KMD_LIM_* offset from it, the ack stride."""
    h = h or abi.headers()
    bases = {"lim_base": lim_base, "KM_ACK_STRIDE": h["KM_ACK_STRIDE"]}
    for name, offset in h.constants.items():
        if name.startswith("KMD_LIM_"):
            bases[name] = lim_base + offset
    return bases


def check_bases(bases):
    """Refuses bases that are not multiples of NOC_ALIGN: the runtime's small writes
    go slot to slot and rely on them."""
    bad = {k: v for k, v in bases.items() if v % NOC_ALIGN}
    if bad:
        named = ", ".join(f"{k} {v:#x}" for k, v in bad.items())
        raise StartError(f"not a multiple of {NOC_ALIGN} bytes: {named}")


def _words(session, rows, shard_grid=None):
    """A row-major uint32 tensor of the rows (one page a row): in DRAM, or one row
    in the L1 of each core of `shard_grid`."""
    data = np.asarray(rows, dtype=np.uint64).astype(np.uint32)
    return session.tensor(data, "uint32", shard_grid=shard_grid)


def read_tensors(device, module, tensors, first=0, pages=None):
    """The raw pages first .. first + pages - 1 (pages=None: to the end) of interleaved
    DRAM tensors as [pages, page_bytes] uint8 each: one DRAM barrier, one read per bank
    and tensor, no wait for idle cores."""
    banks = module.dram_banks(device)
    spans, layout = [], []
    for t in tensors:
        if not t.in_dram:
            raise ValueError("only interleaved DRAM tensors are read by banks")
        page, aligned = t.page_bytes, t.aligned_page_bytes
        at, count = dram.page_range(first, pages, t.pages)
        mine = dram.bank_spans(t.address, aligned, count, banks, at)
        layout.append((len(spans), len(mine), page, aligned, count))
        spans += mine
    chunks = module.read_dram(device, spans)
    return [
        dram.pages_in_order(chunks[at : at + n], page, aligned, pages, banks)
        for at, n, page, aligned, pages in layout
    ]


def _read_uint32(tensor, device=None, module=None):
    """A row-major 32-bit tensor's words as int64: by banks when it is in DRAM and a
    module is given, else as one buffer read (which waits for the launched cores)."""
    if module is not None and tensor.in_dram:
        (pages,) = read_tensors(device, module, [tensor])
        words = pages.reshape(-1).view(np.uint32).reshape(tuple(tensor.shape))
        return words.astype(np.int64)
    return tensor.read_bytes().view(np.uint32).reshape(tensor.shape).astype(np.int64)


class Session:
    """One card: `card` selects it, `device` uses an open one; `grid_timeout` bounds
    every wait for the grid's kernels. As a context manager it stops a loaded run at
    the exit (through the runtime on an exception) and closes the device it opened.
    A session left open is closed the same way at the process's exit and on SIGTERM.
    """

    def __init__(
        self,
        card=None,
        device=None,
        pll_mhz=X280_MHZ,
        tree=None,
        log=None,
        grid_timeout=10.0,
        kernel_bytes=None,
    ):
        self.card = card
        self.device = device
        self._own_device = device is None
        self.pll_mhz = pll_mhz
        self.tree = tree
        self.log = log or (lambda msg: None)
        self.grid_timeout = grid_timeout
        self.kernel_bytes = kernel_bytes  # None: the runtime's default buffer
        self._worker_l1 = worker_l1_size(kernel_bytes)
        if device is not None and kernel_bytes is not None:
            raise ValueError("kernel_bytes is set when the session opens the device")
        self.run = None
        self.stuck = False  # the grid did not return: the runtime closes it at exit
        self._module = None
        self._starting = None  # the Dispatch of a load before its Run exists

    def __enter__(self):
        if self.device is None:
            self.open()
        return self

    def __exit__(self, kind, error, tb):
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
        if bound_exit(os.environ, self.log) == 0:
            self.log(f"{OP_TIMEOUT}=0: an exit after a grid that did not return hangs")
        card = self.card if self.card is not None else visible_card()
        holder = chip_holder(card)
        if holder is not None:
            raise CardBusy(
                f"card {card} is held by process {holder} ({_command(holder)}): "
                "an open would wait until that process exits",
                pid=holder,
            )
        self._module = self._module or native.load_module(self.tree)
        self.device = self._module.Device(self._worker_l1)  # program cache off
        _hook_exit()
        _LIVE.add(self)
        self._own_device = True

    def close(self):
        """Stops a loaded run through the X280 runtime, then closes the device if the
        session opened it. A session on a given device is closed before that device."""
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
                f"process's exit the runtime waits {exit_wait()} for the cores, then "
                "puts them in reset (a stuck thread recovers at the next open, a "
                "compute unit left running does not)"
            )
            return
        if self.device is not None and self._own_device:
            self.device.close()
            self.device = None
        _LIVE.discard(self)

    def _release(self, why, close_device):
        """The exit hooks' bounded clean-up of this session: closes the X280 runtime of
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
        except Exception as err:  # noqa: BLE001 (the process ends; logged, not raised)
            self.log(f"{why}: clean-up raised {type(err).__name__}: {err}")
        finally:
            if close_device:
                try:
                    self.close()
                except Exception as err:  # noqa: BLE001 (as above)
                    self.log(f"{why}: close raised {type(err).__name__}: {err}")

    @property
    def grid(self):
        """The Tensix grid (x, y) of the open device."""
        return tuple(self.device.grid())

    def tensor(self, data, dtype, layout="row", shard_grid=None):
        """A tensor on the card holding `data` (anything np.asarray takes) as `dtype`
        ("bfloat16", "float32", "int32", "uint32", "uint16", "uint8") in `layout`
        ("row": a page a row; "tile": 32 x 32 tiles); in DRAM, or (`shard_grid`
        (x, y)) one row in each core's L1. Written before the grid starts."""
        self._module = self._module or native.load_module(self.tree)
        return kmt.from_array(
            self._module, self.device, data, dtype, layout, shard_grid
        )

    def empty(self, shape, dtype, layout="row", shard_grid=None):
        """A tensor as tensor() makes it, not written."""
        self._module = self._module or native.load_module(self.tree)
        return kmt.empty(self._module, self.device, shape, dtype, layout, shard_grid)

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

    def nocs(self):
        """The grid's NoC coordinates by core (a layout.Nocs), from the open device:
        for program arguments that name other cores, before load()."""
        self._module = self._module or native.load_module(self.tree)
        gx, gy, table, noc0 = self._module.core_nocs(self.device)
        return layout.nocs((gx, gy), table, noc0)

    def read(self, *tensors, first=0, pages=None):
        """The raw pages of interleaved DRAM tensors ([pages, page_bytes] uint8 each),
        under one DRAM barrier; with `first` or `pages`, only those pages of one tensor
        (pages=None: to the end). It does not wait for idle cores, so it may run while
        a program runs: read only what no kernel writes any more (after its event)."""
        if (first or pages is not None) and len(tensors) != 1:
            raise ValueError("a page range is read from one tensor")
        self._module = self._module or native.load_module(self.tree)
        return read_tensors(self.device, self._module, tensors, first, pages)

    def read_chunks(self, tensor, chunk_bytes=2 << 20):
        """Yields (first, pages) over the whole tensor, each read() of at most
        `chunk_bytes` (aligned pages), so that the caller can log every chunk."""
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
        for t, pages in zip(tensors, self.read(*tensors)):
            if t.layout != "tile" or t.dtype != "bfloat16" or len(t.shape) != 2:
                raise ValueError("read_bf16 takes bfloat16 tile tensors of rank 2")
            rows, cols = t.shape
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

    def load(
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
        settle_us=1.0,
        image=None,
        thread="card",
    ):
        """Launches `program` on every core, then the dispatch kernel, then the
        program's X280 thread (`thread="card"`, or "host" for its host fallback);
        returns a Run. The arguments are described in docs/reference/host.md.
        """
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        if thread not in ("card", "host"):
            raise ValueError(f"thread={thread!r}: 'card' or 'host'")
        works = [program.threads[c] for c in sorted(program.threads)]
        program.check_threads()
        if program.x280_ports is not None and (not works or thread != "card"):
            raise ValueError(
                f"{program.name}: X280 ports need its X280 threads on the card"
            )
        for work in works:
            if thread == "host" and work.host_fallback is None:
                raise ValueError(
                    f"{program.name}: the thread on cluster {work.cluster} has no "
                    "host_fallback"
                )
        image = Path(image) if image else native.image_path(self.tree)
        if not image.is_file() or image.stat().st_size == 0:
            raise StartError(f"dispatch kernel image {image}: missing or empty")
        module = self._module = native.load_module(self.tree)
        on_card = bool(works) and thread == "card"
        images = (
            {t.cluster: native.thread_image(t, self.tree) for t in works}
            if on_card
            else {}
        )
        depth = depth or program.depth
        sources = program.sources()  # before anything opens: a missing file raises
        mhz = aiclk_mhz(self.card)
        # the regions go to the card before the grid starts: a tensor write waits
        # for every core to be idle, which a resident grid never is
        regions = _bank_regions(self, program, works)
        dispatch = module.Dispatch(self.device, self.pll_mhz)
        self.log(f"X280 clock set: {self.pll_mhz} MHz")
        _hook_exit()
        _LIVE.add(self)
        self._starting = dispatch
        dispatch.settle_us = settle_us
        backend, launched = None, False
        try:
            if not dispatch.idle:
                raise StartError("X280 cluster 0 is not idle: no image can start")
            state_t = _state_area(self)
            rings = {t.cluster: _work_ring(dispatch, t) for t in works if on_card}
            ends = {c: end for c, (_, end) in rings.items()}
            tag = dispatch.prepare(
                depth,
                host_depth,
                event_depth,
                start,
                deadline,
                kernel_idle,
                state_t.address,
                **kt.kernel_work_args(ends),
                dyn_seg=program.dyn.seg if program.dyn is not None else 0,
            )
            check_bases(lim_bases(dispatch.lim_base))
            plan, port_lines = _port_lines(program, dispatch, rings)
            backend = _Backend(self, dispatch, mhz, trace)
            backend.state_t = state_t  # kept while the kernel may use it
            descriptor = backend.describe(
                program, sources, tag, depth, wait, idle, trace, port_lines
            )
            t0 = time.perf_counter()
            try:
                elves, *build = descriptor.build()
            except Exception as err:
                raise ProgramError(f"{program.name}: build failed: {err}") from err
            program.sizes = kernel_sizes(build)
            text = ", ".join(f"{k} {v}" for k, v in program.sizes.text.items())
            self.log(
                f"kernel buffer: {program.sizes.total} of {program.sizes.limit} B "
                f"(text: {text})"
            )
            if build[-1]:
                raise KernelSizeError(program.name, program.sizes, KERNEL_BUFFER_MAX)
            if insn_check.enabled():
                insn_check.check_kernels(
                    program.name,
                    elves,
                    native.metal_tree(self.tree),
                    native.cache_dir() / "insn-check",
                )
                self.log(
                    f"kernels built and checked ({time.perf_counter() - t0:.2f} s)"
                )
            self.log("launch")
            # From here the grid may run: an interrupt is delivered when the launch
            # call returns, after the kernels started.
            launched = True
            try:
                descriptor.run()
            except Exception as err:
                raise ProgramError(f"{program.name}: launch failed: {err}") from err
            dispatch.start(str(image))
            self.log("dispatch kernel started")
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
        run = Run(backend, (dispatch.grid_x, dispatch.grid_y), program, batch)
        self.run = run
        self._starting = None
        if state != abi.headers()["KMD_STATE_READY"]:
            run._fail_start(state)
        self.log("ready")
        if works:
            shared = {n: _host_array(v) for n, v in program.shared.items()}
            peers = _peers(self, dispatch, images) if on_card else {}
            try:
                for work in works:
                    c = work.cluster
                    view = regions[work.bank].view(work)
                    t = kt.ThreadRun(
                        work, dispatch, view, images.get(c), on_card, view.base,
                        view.check, run, shared,
                    )  # fmt: skip
                    t.ring = rings[c][0] if on_card else None
                    t.peers = peers.get(c)
                    if plan is not None:
                        t.ports = kt.Ports(
                            backend.ctl_addr,
                            len(port_lines.cores),
                            port_lines.cores,
                            plan,
                            program.x280_ports["noc"],
                        )
                        t.write_ports()
                    run.threads[c] = t
                for t in run.threads.values():  # every ports area is written first
                    t.start()
            except BaseException:
                run.abort()
                raise
            run.thread = run.threads[works[0].cluster]
            where = "on the card" if on_card else "on the host"
            self.log(f"threads {list(run.threads)} {where}")
        return run


def _peers(session, dispatch, images):
    """{cluster: komorebi.thread.Peers} for the threads on the card: each with the
    clusters running its image, their tiles and their banks' DRAM tiles."""
    out, nonce = {}, kt.new_check_value() & 0xFFFFFFFF
    for c, group in kt.peer_groups(images).items():
        tiles, banks = {}, {}
        for g in group:
            info = dispatch.work_info(g)
            tiles[g] = info["virtual_x"] | info["virtual_y"] << 16
            x, y = session.device.dram_core(kt.BANK_OF_CLUSTER[g])
            banks[g] = int(x) | int(y) << 16
        out[c] = kt.Peers(nonce, min(group), tiles, banks)
    return out


class _PortLines(NamedTuple):
    """Each core's tile by core index, and {core: (X280 tile, mailbox address)}."""

    cores: list
    lines: dict


def _port_lines(program, dispatch, rings):
    """The program's X280 ports: ({core: (cluster, cpu)}, _PortLines), or (None,
    None) when it declares none."""
    if program.x280_ports is None:
        return None, None
    cores = dispatch.grid_x * dispatch.grid_y
    plan = kt.port_plan(program.x280_ports["serve"], cores, set(rings))
    lines = {}
    for core, (c, _) in plan.items():
        info = dispatch.work_info(c)
        tile = info["virtual_x"] | info["virtual_y"] << 16
        lines[core] = (tile, kt.port_box(info["lim_base"], rings[c][0], core))
    return plan, _PortLines([int(t) for t in dispatch.core_table()[:cores]], lines)


def _region_bytes(value):
    return int(value) if isinstance(value, int) else np.asarray(value).nbytes


def _region_initial(value):
    """The initial bytes as uint8, or None for a zeroed region."""
    if isinstance(value, int):
        return None
    return np.ascontiguousarray(value).view(np.uint8).reshape(-1)


def _host_array(value):
    """A shared region for the host fallbacks: uint32 words of its initial bytes."""
    n = _region_bytes(value)
    data = np.zeros(-(-n // 4) * 4, dtype=np.uint8)
    initial = _region_initial(value)
    if initial is not None:
        data[:n] = initial
    return data.view(np.uint32)


def _bank_regions(session, program, works):
    """{bank: _Regions} for the threads' own regions and the shared ones."""
    out = {}
    for bank in sorted({t.bank for t in works}):
        mine = [t for t in works if t.bank == bank]
        names = {n for t in mine for n in t.shared}
        shared = {n: v for n, v in program.shared.items() if n in names}
        out[bank] = _Regions(session, bank, mine, shared)
    return out


def _work_ring(dispatch, work):
    """The thread's ring (RingPlacement) and the kernel's end of it (tile, area)."""
    info = dispatch.work_info(work.cluster)
    return kt.work_ring(
        dispatch.lim_base,
        (dispatch.l2_x, dispatch.l2_y),
        info["lim_base"],
        (info["virtual_x"], info["virtual_y"]),
        work.cluster,
    )


def _pages(data, pages):
    """`data` (uint8) zero-padded to `pages` region pages, as [pages, words] uint32."""
    out = np.zeros(pages * kt.PAGE, dtype=np.uint8)
    out[: data.size] = data
    return out.view(np.uint32).reshape(pages, -1)


class _Regions:
    """The regions of one DRAM bank in one DRAM tensor whose pages of that bank hold
    them: page 0 the check line, then each thread's own regions (keys (cluster,
    name)), then the shared ones (("shared", name)). Written before the grid starts;
    read(key) only once the grid has stopped, from that bank's pages alone."""

    def __init__(self, session, bank, threads, shared):
        self.session = session
        self.check = kt.new_check_value()
        self.bank = bank
        entries = [
            ((t.cluster, n), t.region_bytes(n), t.initial(n))
            for t in threads
            for n in t.regions
        ]
        entries += [
            (("shared", n), _region_bytes(v), _region_initial(v))
            for n, v in shared.items()
        ]
        self.placed, pages = {}, 1
        for key, n, _ in entries:
            self.placed[key] = kt.Placement(key[1], pages, -(-n // kt.PAGE), n)
            pages += self.placed[key].pages
        self.readable = False  # set by the run once the grid has stopped
        rows = np.zeros((DRAM_BANKS * pages, kt.PAGE // 4), dtype=np.uint32)
        mine = rows[self.bank :: DRAM_BANKS]
        mine[0, :2] = [self.check & abi.M32, self.check >> 32]
        for key, _, data in entries:
            p = self.placed[key]
            if data is not None:
                mine[p.page : p.page + p.pages] = _pages(data, p.pages)
        self.tensor = _words(session, rows)
        self.base = self.tensor.address
        if self.base <= kt.IPI_ACK_DDR < self.base + pages * kt.PAGE:
            raise StartError(
                f"the thread's regions at {self.base:#x} cover the IPI word"
            )

    def view(self, thread):
        """The regions `thread` sees, by name."""
        return _ThreadRegions(self, thread)

    def address(self, key):
        """(bank, bank address) of a region, for Tensix kernels."""
        return self.bank, self.base + self.placed[key].page * kt.PAGE

    def read(self, key):
        """The region's bytes, as uint8."""
        if not self.readable:
            raise RuntimeError("regions are read after run.stop()")
        p, s = self.placed[key], self.session
        name = p.name
        per = p.pages if p.pages * kt.PAGE <= REGION_WHOLE else REGION_CHUNK // kt.PAGE
        spans = dram.bank_chunks(self.base, kt.PAGE, self.bank, p.page, p.pages, per)
        out = np.empty(p.pages * kt.PAGE, np.uint8)
        for at, span in zip(range(0, out.size, per * kt.PAGE), spans):
            (chunk,) = s._module.read_dram(s.device, [span])
            out[at : at + span[2]] = np.asarray(chunk, np.uint8)
            if len(spans) > 1:
                s.log(f"region {name}: {at + span[2]} of {out.size} bytes read")
        return out[: p.bytes].copy()


class _ThreadRegions:
    """One thread's regions in its bank's _Regions: its own, then the shared ones it
    names, in id order (table() gives the argument block's region table)."""

    def __init__(self, regions, thread):
        self.all = regions
        self.keys = {n: (thread.cluster, n) for n in thread.regions}
        self.keys |= {n: ("shared", n) for n in thread.shared}
        self.bank, self.base, self.check = regions.bank, regions.base, regions.check

    @property
    def readable(self):
        return self.all.readable

    @readable.setter
    def readable(self, value):
        self.all.readable = value

    def table(self):
        return [
            (self.address(n)[1], self.all.placed[k].bytes) for n, k in self.keys.items()
        ]

    def address(self, name):
        """(bank, bank address) of a region, for Tensix kernels."""
        return self.all.address(self.keys[name])

    def read(self, name):
        """The region's bytes, as uint8, after run.stop()."""
        return self.all.read(self.keys[name])


class _Backend:
    """The card side of a Run: the native port, the device tensors, the readback."""

    def __init__(self, session, dispatch, mhz, trace):
        self.session = session
        self.device = session.device
        self.port = dispatch
        self.mhz = mhz
        self.trace_cap = (trace + 3) // 4 * 4 if trace else 0  # whole 64-byte pages
        self.io = []
        self.ctl = None
        self.ctl_addr = 0
        self.ctl_offset = 0
        self.trace_t = None
        self.state_t = None
        self.read = False
        self.close_failed = ""  # the runtime's account when an abort did not restore
        self.aborted = None  # {words, harts}: what the kernel left at an abort

    def _control_blocks(self, cores, grid, port_lines=None):
        """Zeroed control blocks, one shard per core, padded so that the block can
        start on a 64-byte line: the same L1 address on every core. `port_lines`: the
        KM_X280_TO line of each served core, written once the address is known."""
        h = abi.headers()
        words = (h["KM_CTL_BYTES"] + CTL_ALIGN) // 4
        self.ctl = _words(self.session, np.zeros((cores, words)), grid)
        base = self.ctl.address
        self.ctl_addr = (base + CTL_ALIGN - 1) // CTL_ALIGN * CTL_ALIGN
        self.ctl_offset = (self.ctl_addr - base) // 4
        check_bases({"control block": self.ctl_addr})
        if port_lines is not None:
            rows = np.zeros((cores, words), dtype=np.uint32)
            to = self.ctl_offset + (h["KM_CTL_X280"] + h["KM_X280_TO"]) // 4
            for core, (tile, box) in port_lines.lines.items():
                rows[core, to + h["KM_X280_TO_TILE"]] = tile
                rows[core, to + h["KM_X280_TO_BOX"]] = box
            self.ctl.write(rows)
        return self.ctl_addr

    def _dynamic(self, program, cores, depth, base):
        """A dynamic segment's queues in DRAM (zero-filled by the cores) and its
        reader / writer and compute words; status lines in the kernel's LIM."""
        dyn = program.dyn
        if dyn is None:
            return [], []
        h = abi.headers()
        dyn.check(cores, DYN_DRAM_BYTES, depth)
        page = 32 * dyn.item_bytes  # kDynPageItems items a page
        queue = self.session.empty((cores * dyn.queue_items // 32, page // 4), "int32")
        self.io.append(queue)
        lines = base + h["KMD_LIM_DYN"]
        return dyn.args(queue.address, lines), dyn.compute_args(cores)

    def describe(
        self, program, sources, tag, depth, wait, idle, trace, port_lines=None
    ):
        """The program descriptor for every core, with the start-up arguments."""
        h = abi.headers()
        d = self.port
        grid = (d.grid_x, d.grid_y)
        cores = grid[0] * grid[1]
        ctl = self._control_blocks(cores, grid, port_lines)
        table = np.zeros((1, h["KM_CORES_MAX"]), dtype=np.uint64)
        table[0] = d.core_table()
        cores_t = _words(self.session, table)
        self.io = [self.ctl, cores_t]
        if self.trace_cap:
            self.trace_t = _words(
                self.session, np.zeros((cores, self.trace_cap * h["kTraceWords"]))
            )
            self.io.append(self.trace_t)
        base = d.lim_base
        groups = abi.column_groups(grid, cores)
        common = {
            "ctl": ctl,
            "cores": cores_t.address,
            "disp_xy": d.l2_x | d.l2_y << 16,
            "ack_base": base + h["KMD_LIM_ACKS"],
            "ready_base": base + h["KMD_LIM_READY"],
            "tag": tag,
            "depth": depth,
            "workers": cores,
            "row": grid[0],
            "wait_limit": abi.limit_units(wait, self.mhz),
            "idle_limit": abi.limit_units(idle, self.mhz) if idle else 0,  # 0: none
            "trace": self.trace_t.address if self.trace_t else 0,
            "trace_cap": self.trace_cap,
        }
        dyn_rw, dyn_c = self._dynamic(program, cores, depth, base)
        reader_args, writer_args, compute_args = [], [], []
        for core in range(cores):
            x, y = core % grid[0], core // grid[0]
            leader, rank, members, slot = groups[core]
            start = abi.start_args(
                common
                | {
                    "me": core,
                    "mailbox": base + h["KMD_LIM_MAILBOX"] if core == 0 else 0,
                    "leader": leader,
                    "rank": rank,
                    "members": members,
                    "ready_slot": slot,
                }
            )
            reader, writer, compute = program.core_args(core)
            reader_args.append((x, y, start + dyn_rw + list(reader)))
            writer_args.append((x, y, start + dyn_rw + list(writer)))
            limits = [common["wait_limit"], common["idle_limit"]]
            compute_args.append((x, y, [ctl, core, *limits, *dyn_c, *compute]))
        all_cores = [(0, 0, grid[0] - 1, grid[1] - 1)]
        include = [str(abi.headers().abi.parents[2])]
        defines = [(k, str(v)) for k, v in program.defines.items()]
        launch = self.session._module.Launch(self.device)
        tok = h["kTokenBytes"]
        cbs = [(b.index, b.format, b.page_bytes, b.pages) for b in program.buffers]
        cbs += [
            (h["kCbTokenCompute"], "uint32", tok, 1),
            (h["kCbTokenWriter"], "uint32", tok, 1),
        ]
        if self.trace_cap:
            cbs.append((h["kCbTrace"], "uint32", self.trace_cap * 16, 1))
        for index, fmt, page, pages in cbs:
            launch.cb(all_cores, index, fmt, page, page * pages)
        kernels = [("reader", reader_args, 0, {}), ("writer", writer_args, 1, {})]
        kernels.append(("compute", compute_args, 2, compute_config(program.compute)))
        for kernel, args, processor, config in kernels:
            launch.kernel(
                str(sources[kernel]),
                all_cores,
                processor,
                list(program.compile_args.get(kernel, [])),
                includes=include,
                args=args,
                **kernel_build(program, kernel, defines),
                **config,
            )
        return launch

    def finish(self, timeout):
        """Waits for the kernel to finish, stops it, then closes the X280 runtime."""
        try:
            return self.port.finish(timeout)
        finally:
            self.port.close()
            self.session.log("x280 runtime closed")

    def abort(self):
        """The failure path: the stop word to every core, a bounded wait for the grid,
        then the runtime's close, so core 0 is not held behind a grid that runs on.
        Returns whether cluster 0 was restored; `aborted` holds what the kernel left."""
        gx, gy = self.port.grid_x, self.port.grid_y
        try:
            self.stop_grid()
            back = self.session._module.wait_grid(self.device, gx, gy, ABORT_GRID_S)
            self.session.log("grid returned" if back else "grid still running")
        except Exception as err:  # noqa: BLE001 (the close below runs whatever happens)
            self.session.log(f"stopping the grid raised {type(err).__name__}: {err}")
        cleaned = self.port.close()
        self.close_failed = "" if cleaned else self.port.close_failed
        self.aborted = self.port.aborted()
        why = f": {self.close_failed}" if self.close_failed else ""
        if self.aborted is not None:
            n = self.aborted["harts"][abi.headers()["KMD_H_ABANDONED"]]
            why += f"; {n} records abandoned"
        self.session.log(f"x280 runtime closed after a stop (cleaned {cleaned}){why}")
        self.log_state()
        return cleaned

    def log_state(self):
        """Logs the first STATE_DUMP_LINES lines of the kernel's state as DRAM holds
        them (the kernel flushes them when it finishes); never raises."""
        try:
            words = self.kernel_state()
            if words is None:
                self.session.log("kernel state not found in DRAM (lines not flushed)")
                return
            for at, line in enumerate(words.reshape(-1, 8)):
                text = " ".join(f"{int(w):#x}" for w in line)
                self.session.log(f"kernel state +{64 * at:#05x}: {text}")
        except Exception as err:  # noqa: BLE001 (a failure path: reported, not raised)
            self.session.log(f"kernel state not read ({type(err).__name__}: {err})")

    def kernel_state(self):
        """The first STATE_DUMP_LINES lines of the kernel's state (uint64 words) from
        the first page of each bank: the one whose args and lim words name the LIM
        base is the bank the cluster's Memory Port reaches. None if no page does."""
        banks = self.session._module.dram_banks(self.device)
        (pages,) = self.session.read(self.state_t, first=0, pages=banks)
        words = np.asarray(pages, np.uint8)[:, : 64 * STATE_DUMP_LINES]
        words = np.ascontiguousarray(words).view("<u8")
        base = self.port.lim_base
        found = [w for w in words if w[2] == base and w[3] == base]  # args, lim
        return found[0] if len(found) == 1 else None

    def kernel_report(self):
        """While the kernel runs: whether core 0 still runs its loop (its pass stamp
        over PROBE_GAP_S), its state and where, and the runtime's view of each hart."""
        p = self.port.probe(PROBE_GAP_S)
        if not p["running"]:
            return "no dispatch kernel is running"
        if p["failed"]:
            return f"the dispatch kernel's {p['failed']}; harts: {p['harts']}"
        r = st.result_block(p["words"])
        before, after = p["stamp_before"], p["stamp_after"]
        doing = "is running its loop" if after != before else "is not running its loop"
        report = (
            f"the dispatch kernel's core 0 {doing} (pass stamp {before} -> {after} "
            f"in {PROBE_GAP_S} s; state {r['state_name']}, in {r['where_name']}); "
            f"harts: {p['harts']}"
        )
        self.session.log(report)
        return report

    def stop_grid(self):
        """Writes the stop word into every core through the device API."""
        h = abi.headers()
        module, gx, gy = self.session._module, self.port.grid_x, self.port.grid_y
        for core in range(gx * gy):
            at = self.ctl_addr + h["KM_CTL_STOP"]
            module.write_l1(self.device, core % gx, core // gx, at, [1])
        self.session.log("stop word written by the host")

    def read_back(self, stop_grid=False):
        """After the X280 runtime is closed: waits (bounded) for the grid's kernels,
        then reads the control blocks ([cores, KM_CTL_BYTES / 4] uint32) and the trace.
        `stop_grid`: the dispatch kernel was not there to stop the grid, so the host
        first writes the stop word into every core through the device API."""
        if self.port.is_open:
            raise RuntimeError("close the X280 runtime before reading back")
        h = abi.headers()
        module, gx, gy = self.session._module, self.port.grid_x, self.port.grid_y
        if stop_grid:
            self.stop_grid()
        t0 = time.monotonic()
        bound = self.session.grid_timeout
        if not module.wait_grid(self.device, gx, gy, bound):
            self.session.stuck = True
            raise GridTimeout(
                f"the grid's kernels did not return within {bound} s after the stop. "
                "The session does not close the device; at the process's exit the "
                f"runtime waits {exit_wait()} for the cores, then puts them in reset. "
                "A thread that is only stuck is recovered by the next open; a kernel "
                "that left a compute unit running is not, and if the next open fails "
                "the card needs a reset"
            )
        self.device.finish()  # returns at once: every core is done
        self.session.log(f"grid returned ({time.monotonic() - t0:.1f} s)")
        rows = _read_uint32(self.ctl)
        start = self.ctl_offset
        blocks = rows[:, start : start + h["KM_CTL_BYTES"] // 4].astype(np.uint32)
        trace = None
        if self.trace_t is not None:
            words = _read_uint32(self.trace_t, self.device, module).astype(np.uint32)
            trace = words.reshape(words.shape[0], self.trace_cap, h["kTraceWords"])
        self.session.log("state and trace read back")
        return blocks, trace
