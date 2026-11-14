"""A card session: opens the device, loads a program as persistent kernels on every
Tensix core and the dispatcher on X280 cluster 0, and hands out a `Run`.

Needs a tt-metal tree or the runtime package; `import komorebi` does not import it.
"""

import atexit
import contextlib
import copy
import ctypes
import os
import re
import secrets
import signal
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import NamedTuple

import numpy as np

from komorebi import (
    abi,
    dram,
    formats,
    insn_check,
    island_session,
    islands,
    layout,
    native,
    overlay,
    programs,
)
from komorebi import channel as channel_module
from komorebi import state as device_state
from komorebi import tensor as tensor_module
from komorebi import thread as thread_module
from komorebi.errors import (
    CardBusy,
    ChannelError,
    GridTimeout,
    KernelSizeError,
    L1BudgetError,
    ProgramError,
    SlowCodeError,
    StartError,
)
from komorebi.program import KernelSizes, Program, messages_header
from komorebi.run import Run

CONTROL_ALIGN = 64  # the dispatcher stores whole 64-byte lines into the control block
IPI_ACK_DDR = 0x470020  # the X280 runtime's IPI ack word in the cluster's DRAM bank
MESSAGES_DRAM_BYTES = 4 << 30  # the most DRAM a dynamic handler's queues may take
X280_QUEUES = "x280_queues"  # the region of the X280 ranks' queues in their thread
STATE_PAGES = 32  # pages of KMD_STATE_BYTES, interleaved: several in every DRAM bank
NOC_ALIGN = 16  # a NoC write's source and destination share their low 4 address bits
DRAM_BANKS = 8  # interleaved page i lies in bank i % DRAM_BANKS
# The X280 clock: 1750 MHz loses loads under full Tensix load. 1350 is the default; a
# runtime without it gets 1000 (its error: "no PLL solution") unless one was asked for.
X280_MHZ = 1350
X280_FALLBACK_MHZ = 1000
NO_PLL_SOLUTION = "no PLL solution for"
STATE_DUMP_LINES = 8  # 64-byte lines of the kernel's state logged after a failure
CHIP_LOCK = "/dev/shm/TT_UMD_LOCK.CHIP_IN_USE_{card}_PCIe"  # UMD's lock per card
CHIP_LOCK_BYTES, CHIP_LOCK_PID = 56, 52  # its size; the owner PID's byte offset
CBS = 64  # circular buffers of a Tensix core: one unpack-to-DST mode each
PROBE_GAP_S = 0.02  # core 0's pass marker moves every 1,024 passes (about 33 us)
ABORT_GRID_S = 2.0  # an abort's wait for the grid after the stop flag, before the close
REGION_WHOLE = 4 << 20  # a region up to this size is read in one read, else in chunks
REGION_CHUNK = 2 << 20  # of this size, with a log line each
# The runtime's timeout on its waits for the cores, the one at device close included;
# unset, it is 0, which means no timeout. Read when the process first opens a device.
OP_TIMEOUT = "TT_METAL_OPERATION_TIMEOUT_SECONDS"
OP_TIMEOUT_DEFAULT = 30.0  # the runtime's own example value
# A Blackhole Tensix core's L1 and where the runtime's kernel buffer starts in it
# (MEM_L1_SIZE, MEM_MAP_END); the buffer ends where the allocator's L1 base begins.
L1_BYTES = 1536 * 1024
KERNEL_BUFFER_BASE = 41088
KERNEL_BUFFER_DEFAULT = 69 * 1024  # the runtime's own (70,656 B)
KERNEL_BUFFER_ALIGN = 64  # the allocator's base is aligned up to this
GROUP_DEFINE = "-DKM_PROGRAM_GROUP"  # a group's runtime kernels are built apart
GROUP_LINE = re.compile(r"^#define KM_PROGRAM_GROUP (\d+)$", re.M)
# a name the compiler gives a source file's own symbol (trisck.cc.4c8b3e06): the
# build's hash, which no part refers to
UNIT_SYMBOL = re.compile(r"\.cc\.[0-9a-f]+$")
KERNEL_BUFFER_MAX = L1_BYTES - KERNEL_BUFFER_BASE - KERNEL_BUFFER_ALIGN
# A taken branch to code at or above this L1 address costs about 4 cycles more: the
# kernels and the slots go below it, the circular buffers above them.
L1_FAST_CODE = 1 << 20
TENSIX_THREADS = ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2")
# A transfer without a run launches its kernel from this many bytes on: a launch costs
# about 30 ms, as the device API's read of some 76 KiB does (measured on a p150a).
LAUNCH_MIN_BYTES = 80 << 10
TRANSFER_WAIT_S = 1.0  # beyond the bytes at 1 GB/s: a transfer's wait for its words
OPT_DEFINE = "KM_OPT_LEVEL"  # defined to a kernel's level when the program sets one
SPLIT_ISLANDS_MAX = 4  # a card split into islands: a dispatcher an X280 cluster


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
    runtime (with a timeout), then the device it opened is closed."""
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
    return " ".join(word.decode(errors="replace") for word in words if word)[:160]


def aiclk_mhz(card=None):
    """The Tensix clock of the card this process uses, from the driver (MHz)."""
    card = card if card is not None else visible_card()
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    return int(path.read_text().split()[0])


def exit_timeout(environ, log):
    """Sets OP_TIMEOUT to its default in `environ` when unset or empty and logs it; a
    value already there is kept. The timeout in seconds (0: none; None: not a number).
    """
    value = environ.get(OP_TIMEOUT, "")
    if value.strip():
        try:
            return max(float(value), 0.0)
        except ValueError:
            return None  # the runtime rejects it at the open
    environ[OP_TIMEOUT] = f"{OP_TIMEOUT_DEFAULT:g}"
    log(f"{OP_TIMEOUT} unset: set to {OP_TIMEOUT_DEFAULT:g} s (the exit's timeout)")
    return OP_TIMEOUT_DEFAULT


def exit_wait(environ=None):
    """How long the device's close waits for the cores, as a phrase."""
    value = (os.environ if environ is None else environ).get(OP_TIMEOUT, "")
    try:
        seconds = float(value)
    except ValueError:
        seconds = 0.0
    return f"up to {seconds:g} s" if seconds > 0 else "without a timeout"


def worker_l1_size(kernel_bytes):
    """The runtime's worker_l1_size for a kernel buffer of `kernel_bytes` (None: the
    default, nothing passed); a size the runtime would not give exactly is rejected."""
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


def _core_args(program, placed, core):
    """(reader, writer, compute) program arguments of `core`: its group's program's
    when the run holds programs in the runtime's form."""
    if placed is None:
        return program.core_args(core)
    for group in placed.groups:
        if group.first <= core < group.first + group.count and group.programs:
            return placed.programs[group.programs[0]].core_args(core)
    return [], [], []


def kernel_build(program, kernel, defines):
    """A kernel's defines, and its optimisation level when the program sets one. A set
    level also adds a define naming it: the runtime's build cache key leaves the level
    out, and two levels must not share a cached ELF."""
    level = program.kernel_opt(kernel)
    if level is None:
        return {"defines": defines}
    return {"defines": [*defines, (OPT_DEFINE, level)], "opt_level": level}


def slot_area(base, size, used, slow_code):
    """The slots' (address, bytes, buffer index): `size` bytes as the first circular
    buffer, at the L1 base `base`, on an index not in `used`; refused (SlowCodeError)
    when they end above L1_FAST_CODE, unless `slow_code`."""
    size = -(-size // overlay.ALIGN) * overlay.ALIGN
    if base + size > L1_FAST_CODE and not slow_code:
        raise SlowCodeError(base, size, L1_FAST_CODE)
    # below the highest index in use: the kernel config keeps its size
    free = set(range(max(used))) - set(used)
    if not free:
        raise ProgramError(f"no circular buffer index below {max(used)} for the slots")
    return base, size, max(free)


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
    """A DRAM buffer whose bank address holds the dispatcher's state in every
    bank (KMD_STATE_BYTES at least); rejected if it covers the runtime's IPI word."""
    abi_headers = abi.headers()
    rows = _words(session, np.zeros((STATE_PAGES, abi_headers["KMD_STATE_BYTES"] // 4)))
    address = rows.address
    if address <= IPI_ACK_DDR < address + STATE_PAGES * abi_headers["KMD_STATE_BYTES"]:
        raise StartError(f"the kernel state at {address:#x} covers the X280 IPI word")
    return rows


def lim_bases(lim_base, abi_headers=None):
    """The X280 local-memory addresses the kernels are given or derive from the
    runtime's base: the base, every KMD_LIM_* offset from it, the ack stride."""
    abi_headers = abi_headers or abi.headers()
    bases = {"lim_base": lim_base, "KM_ACK_STRIDE": abi_headers["KM_ACK_STRIDE"]}
    for name, offset in abi_headers.constants.items():
        if name.startswith("KMD_LIM_"):
            bases[name] = lim_base + offset
    return bases


def check_bases(bases):
    """Rejects bases that are not multiples of NOC_ALIGN: the runtime's small writes
    go slot to slot and rely on them."""
    bad = {name: address for name, address in bases.items() if address % NOC_ALIGN}
    if bad:
        named = ", ".join(f"{name} {address:#x}" for name, address in bad.items())
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
    for tensor in tensors:
        if not tensor.in_dram:
            raise ValueError("only interleaved DRAM tensors are read by banks")
        page, aligned = tensor.page_bytes, tensor.aligned_page_bytes
        start_page, count = dram.page_range(first, pages, tensor.pages)
        mine = dram.bank_spans(tensor.address, aligned, count, banks, start_page)
        layout.append((len(spans), len(mine), page, aligned, count))
        spans += mine
    chunks = module.read_dram(device, spans)
    return [
        dram.pages_in_order(
            chunks[first_span : first_span + span_count], page, aligned, pages, banks
        )
        for first_span, span_count, page, aligned, pages in layout
    ]


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


def _read_uint32(tensor, device=None, module=None):
    """A row-major 32-bit tensor's words as int64: by banks when it is in DRAM and a
    module is given, else as one buffer read (which waits for the launched cores)."""
    if module is not None and tensor.in_dram:
        (pages,) = read_tensors(device, module, [tensor])
        words = pages.reshape(-1).view(np.uint32).reshape(tuple(tensor.shape))
        return words.astype(np.int64)
    return tensor.read_bytes().view(np.uint32).reshape(tensor.shape).astype(np.int64)


class Session:
    """One card: `card` selects it, `device` uses an open one (`cards`, or `islands` 2
    to 4 beside `card`: an IslandSession); `grid_timeout` bounds every grid wait. As a
    context manager it stops a loaded run at the exit (through the runtime on an
    exception) and closes the device it opened; so do the process's exit and SIGTERM."""

    def __new__(cls, *args, cards=None, islands=None, **options):
        if cards is not None:
            if args or "card" in options or "device" in options:
                raise ValueError("cards= is given alone, without card or device")
            return island_session.IslandSession(cards, islands=islands, **options)
        if islands is not None:
            if args or "card" not in options or "device" in options:
                raise ValueError("islands= is given with card=, without device")
            card = options.pop("card")
            return island_session.IslandSession([card], islands=islands, **options)
        return super().__new__(cls)

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
    ):
        self.card = card
        self.device = device
        self._own_device = device is None
        self.pll_mhz = X280_MHZ if pll_mhz is None else pll_mhz
        self._default_clock = pll_mhz is None  # may fall back to X280_FALLBACK_MHZ
        self.tree = tree
        self.log = log or (lambda message: None)
        self.grid_timeout = grid_timeout
        self.kernel_bytes = kernel_bytes  # None: the runtime's default buffer
        self.slow_code = slow_code  # slots may end above L1_FAST_CODE
        self._worker_l1 = worker_l1_size(kernel_bytes)
        if device is not None and kernel_bytes is not None:
            raise ValueError("kernel_bytes is set when the session opens the device")
        self.run = None
        self.island = None  # (graph, i) of the loaded run: Session.load's `island`
        self.link_area = 0  # its link area's L1 address (Session.load's `link`), or 0
        self.transfer_chunk = None  # a larger transfer chunk asked for, where it fits
        self.tensor_table = 0  # its tensor table's L1 address (load's `tensors`)
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
        if exit_timeout(os.environ, self.log) == 0:
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
        if self._region is None:
            self.device = self._module.Device(self._worker_l1)  # program cache off
        else:
            with _output_seen() as seen:  # the driver's warning, if it gives one
                self.device = self._module.Device(self._worker_l1)
        _hook_exit()
        _LIVE.add(self)
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

    def tensor(self, data, dtype, layout="row", shard_grid=None):
        """A tensor on the card holding `data` (anything np.asarray takes) as `dtype`
        ("bfloat16", "float32", "int32", "uint32", "uint16", "uint8") in `layout`
        ("row" or "tile"); in DRAM, or one row in each core's L1 of `shard_grid`.
        Written before the grid starts, or (DRAM, the host channel) while it runs."""
        self._module = self._module or native.load_module(self.tree)
        raw, shape = formats.to_bytes(data, dtype, layout)
        tensor = tensor_module.empty(
            self._module, self.device, shape, dtype, layout, shard_grid
        )
        self._note_page(tensor)
        if self._transfer_way([tensor], 0, None) is None:
            tensor.buffer.write(raw)
        else:
            self.write(tensor, raw.reshape(tensor.pages, tensor.page_bytes))
        return tensor

    def empty(self, shape, dtype, layout="row", shard_grid=None, address=None):
        """A tensor as tensor() makes it, not written."""
        self._module = self._module or native.load_module(self.tree)
        tensor = tensor_module.empty(
            self._module, self.device, shape, dtype, layout, shard_grid, address
        )
        self._note_page(tensor)
        return tensor

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
        in_l1 = [tensor for tensor in tensors if not tensor.in_dram]
        if in_l1 and self.run is not None and not self.run.stopped:
            raise ChannelError(
                "an L1 tensor is read while a run is loaded through the host channel, "
                "which this session cannot use for it"
            )
        in_dram = [tensor for tensor in tensors if tensor.in_dram]
        banked = iter(read_tensors(self.device, self._module, in_dram, first, pages))
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
        pages = np.ascontiguousarray(pages, np.uint8)
        if pages.ndim != 2 or pages.shape[1] != tensor.page_bytes:
            raise ValueError(f"pages are [n, {tensor.page_bytes}] uint8 rows")
        start, count = dram.page_range(first, len(pages), tensor.pages)
        self._module = self._module or native.load_module(self.tree)
        way = self._transfer_way([tensor], start, count)
        if way is None:
            if count != tensor.pages or (self.run is not None and not self.run.stopped):
                raise ChannelError(
                    "a page range, or a write while a run is loaded, goes through the "
                    "host channel, which this session does not have"
                )
            tensor.buffer.write(pages.reshape(-1))
            return
        aligned = tensor.aligned_page_bytes
        for done, step in channel_module.rounds(count, aligned, self._region):
            rows = self.channel[: step * aligned].reshape(step, aligned)
            rows[:, : tensor.page_bytes] = pages[done : done + step]
            self._transfer(tensor, start + done, step, way, False)

    def _transfer_way(self, tensors, first, pages):
        """How a read of these pages goes: "run" (the live run's cores push them),
        "launch" (the transfer kernel, from LAUNCH_MIN_BYTES on), None (the device).
        An L1 tensor's pages are moved by the cores that hold them."""
        self._map_channel()
        if self.channel is None:
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
        channel_module.check(transfer, count, L1_BYTES)
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
            channel_module.check(transfer, pages, L1_BYTES)
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
            channel_module.check(transfer, count, L1_BYTES)
            words = channel_module.words_view(self.channel, region, count)
            words[:] = 0
            want = channel_module.sequence_words(transfer.nonce, count, page, count)
            run.transfer(handler, (rank + here, count, field), transfer.args(), timeout)
            if not self._words_arrived(words, want, timeout):
                raise ChannelError(self._words_missing(words, want, timeout))
            self.transfer_cycles += channel_module.cycles(words)

    @staticmethod
    def _words_arrived(words, want, timeout):
        """Waits (bounded) until every rank's sequence word is the one expected."""
        deadline = time.monotonic() + timeout
        while not np.array_equal(words[:, :4], want):
            if time.monotonic() > deadline:
                return False
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
        launch = self._module.Launch(self.device)
        args = [
            (x, y, [report.address, int(push), rank, count, *transfer.args()])
            for rank, (x, y) in enumerate(places)
        ]
        include = abi.headers().abi.parents[2]
        source = include.parent / "kernels" / "host_transfer.cpp"
        launch.kernel(str(source), cores, 0, [], [], [str(include)], args, noc=0)
        launch.cb(cores, 0, "uint32", 16, 2 * transfer.chunk + 64)
        launch.run()
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
        settle_microseconds=1.0,
        image=None,
        thread="card",
        island=None,
        link=None,
        tensors=0,
        recover=True,
    ):
        """Launches `program` (or programs in the runtime's form: a list, every core
        holding each, or {program: (first, count)}) on every core, then the dispatcher,
        then the X280 thread (`thread` "card" or "host"); returns a Run. The arguments
        are described in docs/reference/host.md."""
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        self.island = _island(island, self.grid)
        if link is not None and self.island is None:
            raise ValueError("link= is an island's (island= too)")
        self.link_plan = link
        self.link_area = link.area if link is not None else 0
        self.tensor_table = tensors
        placed = None
        if not isinstance(program, Program) or program.form() == "program":
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
        ranks = _x280_ranks(program, owner, placed, on_card)
        depth = depth or program.depth
        sources = program.sources()  # before anything opens: a missing file raises
        if placed is not None:
            sources = programs.runtime_kernels()
        mhz = aiclk_mhz(self.card)
        # the regions go to the card before the grid starts: a tensor write waits
        # for every core to be idle, which a persistent grid never is
        regions = _bank_regions(self, owner, works)
        dispatch = self._open_dispatch(module)
        self.log(f"X280 clock set: {self.pll_mhz} MHz")
        _hook_exit()
        _LIVE.add(self)
        self._starting = dispatch
        dispatch.settle_microseconds = settle_microseconds
        backend, launched = None, False
        try:
            if not dispatch.idle:
                raise StartError("X280 cluster 0 is not idle: no image can start")
            state_tensor = _state_area(self)
            rings = {
                work.cluster: _work_ring(dispatch, work) for work in works if on_card
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
                    program.messages.dispatcher_word()
                    if program.messages is not None
                    else 0
                ),
                programs=_held_fields(placed) if placed and placed.switch else [],
                island=_island_word(self.island),
                link=list(link.line(dispatch)) if link is not None else [],
            )
            check_bases(lim_bases(dispatch.lim_base))
            plan, port_lines = _port_lines(owner, dispatch, rings)
            backend = _Backend(self, dispatch, mhz, trace)
            backend.state_tensor = state_tensor  # kept while the kernel may use it
            if ranks is not None:
                ranks = _x280_words(self, program, dispatch, rings, regions, plan)
            descriptor = backend.describe(
                program, sources, tag, depth, wait, idle, trace, port_lines, placed,
                ranks,
            )  # fmt: skip
            started = time.perf_counter()
            try:
                elves, *build = descriptor.build()
            except Exception as error:
                raise ProgramError(f"{program.name}: build failed: {error}") from error
            program.sizes = kernel_sizes(build)
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
                raise KernelSizeError(program.name, program.sizes, KERNEL_BUFFER_MAX)
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
            _start_loads(run, placed)
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
        self.log("ready")
        if works:
            shared = {name: _host_array(value) for name, value in owner.shared.items()}
            peers = _peers(self, dispatch, images) if on_card else {}
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
                        thread_run.ranks = _x280_line(
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
        rows = _split_rows(graph, self.grid)
        self.link_plans, self.link_plan, self.link_area = links, None, 0
        self.tensor_table = tensors
        image = Path(image) if image else native.image_path(self.tree)
        if not image.is_file() or image.stat().st_size == 0:
            raise StartError(f"dispatcher image {image}: missing or empty")
        module = self._module = native.load_module(self.tree)
        depth = depth or program.depth
        sources = program.sources()
        mhz = aiclk_mhz(self.card)
        dispatches = _Dispatches([self._open_dispatch(module)])
        self.log(f"X280 clock set: {self.pll_mhz} MHz")
        _hook_exit()
        _LIVE.add(self)
        self._starting = dispatches
        backends, launched = [], False
        try:
            for cluster in range(1, len(rows)):
                dispatches.append(module.Dispatch(dispatches[0], cluster))
            states, tags = [], []
            for island, (dispatch, (first_row, count)) in enumerate(
                zip(dispatches, rows)
            ):
                dispatch.settle_microseconds = settle_microseconds
                if not dispatch.idle:
                    raise StartError(f"X280 cluster {island} is not idle")
                states.append(_state_area(self))
                base = first_row * self.grid[0]
                plan = links[island] if links else None
                line = b""
                if plan:
                    line = plan.line(dispatch, dispatches[plan.far], dispatches)
                tags.append(
                    dispatch.prepare(
                        depth, host_depth, event_depth, start, deadline, kernel_idle,
                        states[-1].address,
                        messages_handler=(
                            program.messages.dispatcher_word()
                            if program.messages is not None
                            else 0
                        ),
                        island=_island_word((graph, island)) | base << 32,
                        link=list(line), first_row=first_row, rows=count,
                    )  # fmt: skip
                )
                check_bases(lim_bases(dispatch.lim_base))
            if len({dispatch.lim_base for dispatch in dispatches}) != 1:
                raise StartError(  # a core's words name one LIM address on every one
                    "the islands' dispatchers differ in LIM base: "
                    f"{[hex(dispatch.lim_base) for dispatch in dispatches]}"
                )
            first = _Backend(self, dispatches[0], mhz, trace)
            first.split = (graph, list(dispatches), tags, rows, set())
            first.state_tensor = states[0]
            descriptor = first.describe(
                program, sources, tags[0], depth, wait, idle, trace
            )
            try:
                elves, *build = descriptor.build()
            except Exception as error:
                raise ProgramError(f"{program.name}: build failed: {error}") from error
            program.sizes = kernel_sizes(build)
            if build[-1]:
                raise KernelSizeError(program.name, program.sizes, KERNEL_BUFFER_MAX)
            if insn_check.enabled():
                insn_check.check_kernels(
                    program.name,
                    elves,
                    native.metal_tree(self.tree),
                    native.cache_directory() / "insn-check",
                )
            first.load_overlays(program, descriptor)
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


def _recipe_group(recipe):
    """The group number a runtime kernel's build was made for (KM_PROGRAM_GROUP, in
    its defines or in the header the runtime generates beside its ELF), or None."""
    for word in recipe.defines:
        if word.startswith(GROUP_DEFINE + "="):
            return int(word.split("=", 1)[1])
    header = recipe.elf.parent.parent / "defines_generated.h"
    found = GROUP_LINE.search(header.read_text()) if header.is_file() else None
    return int(found.group(1)) if found else None


def _island(island, grid):
    """Session.load's `island` (graph, i) checked: island i's ranks are this card's
    first cores in core order, its link cores the rest in order, so rank first + c is
    core c and link core k is core ranks + k (the dispatcher's mapping)."""
    if island is None:
        return None
    graph, here = island
    cores = grid[0] * grid[1]
    if not 0 <= here < len(graph.islands):
        raise ValueError(f"island {here} of {len(graph.islands)}")
    if graph.rank_cores[here] + graph.link_cores[here] != tuple(range(cores)):
        raise ValueError(
            f"island {here}: its ranks, then its link cores, are not the {cores} "
            "cores in order"
        )
    return graph, here


def _island_word(island):
    """The dispatcher's island word: first | count << 16 of its ranks; 0 for none."""
    if island is None:
        return 0
    graph, here = island
    return graph.first(here) | graph.count(here) << 16


def _split_rows(graph, grid):
    """[(first row, rows)] of each island of a card split into islands: its cores whole
    rows of the grid in order, its ranks then its link cores (the dispatcher's
    mapping), at most one island an X280 cluster."""
    columns, rows = grid
    out, at = [], 0
    if not 2 <= len(graph.islands) <= SPLIT_ISLANDS_MAX:
        raise ValueError(f"{len(graph.islands)} islands: 2 to {SPLIT_ISLANDS_MAX}")
    for island, ranks in enumerate(graph.rank_cores):
        cores = ranks + graph.link_cores[island]
        if (
            cores != tuple(range(at, at + len(cores)))
            or at % columns
            or len(cores) % columns
        ):
            raise ValueError(
                f"island {island}: its ranks, then its link cores, are not whole rows "
                f"of the {columns} x {rows} grid after the island before it"
            )
        out.append((at // columns, len(cores) // columns))
        at += len(cores)
    if at != columns * rows:
        raise ValueError(f"the islands hold {at} cores of {columns * rows}")
    return out


class _Dispatches(list):
    """The dispatches of a card split into islands, closed together (a load's)."""

    close_failed = ""

    def close(self):
        failed = [d.close_failed for d in self if not d.close()]
        self.close_failed = "; ".join(failed)
        return not failed


def _group_recipes(kernels, count):
    """{target: Recipe} of each group's runtime kernels, by group number: the
    runtime lists a program's kernels in no fixed order."""
    out = [{} for _ in range(count)]
    for _, builds in kernels:
        for row in builds:
            recipe = overlay.Recipe.from_native(row)
            number = _recipe_group(recipe)
            if number is None or not 0 <= number < count:
                raise ProgramError(f"{recipe.elf}: no group number in its build")
            out[number][recipe.target] = recipe
    return out


def _held_fields(placed):
    """A byte per core for the dispatcher's table: the program field it holds, 0xFF
    none."""
    out = [0xFF] * placed.cores
    for group in placed.groups:
        for program_field in group.programs:
            out[group.first : group.first + group.count] = [program_field] * group.count
    return out


def _start_loads(run, placed):
    """In a run whose groups switch, each group loads its program first: the compute
    parts set the hardware up for its formats (the runtime's kernels do not)."""
    handles = [
        run.load(placed.programs[group.programs[0]], (group.first, group.count))
        for group in placed.groups
        if group.programs
    ]
    run.wait(handles, timeout=10.0)


def _common_slots(placed, declared, built):
    """The slots of a run whose groups switch: per part, the declared count (else one a
    program, the reader one more for the service overlay, at most kOverlaySlotsMax) of
    the largest part any program has there."""
    most = overlay.constants()["kOverlaySlotsMax"]
    slots = {}
    for made in built:
        for part, (_, size) in made.slots.items():
            slots[part] = max(slots.get(part, 0), size)
    count = declared.count
    out = {}
    for part, size in slots.items():
        kernel = next(k for k, parts in overlay.KERNEL_PARTS.items() if part in parts)
        wanted = count.get(kernel) if isinstance(count, dict) else count
        service = part == "reader"  # the service overlay's slot beside them
        out[part] = (wanted or min(len(placed.programs) + service, most), size)
    return overlay.Built({}, out)


def _program_cbs(buffers):
    """(index, format, page bytes, pages) of a program's buffers."""
    return [
        (buffer.index, buffer.format, buffer.page_bytes, buffer.pages)
        for buffer in buffers
    ]


def _peers(session, dispatch, images):
    """{cluster: komorebi.thread.Peers} for the threads on the card: each with the
    clusters running its image, their tiles and their banks' DRAM tiles."""
    out, nonce = {}, thread_module.new_check_value() & 0xFFFFFFFF
    for cluster, group in thread_module.peer_groups(images).items():
        tiles, banks = {}, {}
        for peer in group:
            info = dispatch.work_info(peer)
            tiles[peer] = info["virtual_x"] | info["virtual_y"] << 16
            x, y = session.device.dram_core(thread_module.BANK_OF_CLUSTER[peer])
            banks[peer] = int(x) | int(y) << 16
        out[cluster] = thread_module.Peers(nonce, min(group), tiles, banks)
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
    plan = thread_module.port_plan(program.x280_ports["serve"], cores, set(rings))
    lines = {}
    for core, (cluster, _) in plan.items():
        info = dispatch.work_info(cluster)
        tile = info["virtual_x"] | info["virtual_y"] << 16
        lines[core] = (
            tile,
            thread_module.port_box(info["lim_base"], rings[cluster][0], core),
        )
    return plan, _PortLines(
        [int(core_tile) for core_tile in dispatch.core_table()[:cores]], lines
    )


def _x280_ranks(program, owner, placed, on_card):
    """The thread of the program's X280 ranks, its queues' region (X280_QUEUES) added;
    None when the program has none."""
    messages = program.messages
    if messages is None or messages.x280_ranks == 0:
        return None
    work = owner.threads.get(messages.x280_cluster)
    if placed is not None or work is None or not on_card:
        raise ValueError(
            f"{program.name}: X280 ranks need one program with its thread on cluster "
            f"{messages.x280_cluster}, on the card"
        )
    size = messages.queue_region_bytes()
    if work.regions.get(X280_QUEUES, size) != size:
        raise ValueError(f"{program.name}: the thread's region {X280_QUEUES!r} is ours")
    work.regions[X280_QUEUES] = size
    return work


def _x280_words(session, program, dispatch, rings, regions, plan):
    """The X280 ranks' words of km::MessagesArgs (without the inbox): their queues'
    bank address and DRAM tile, their cluster's tile, the cores' credit lines there."""
    messages = program.messages
    cluster = messages.x280_cluster
    bank, queue = regions[thread_module.BANK_OF_CLUSTER[cluster]].address(
        (cluster, X280_QUEUES)
    )
    x, y = session.device.dram_core(bank)
    info = dispatch.work_info(cluster)
    at = thread_module.ranks_at(rings[cluster][0], plan)
    return {
        "x280_queue": queue,
        "x280_bank": int(x) | int(y) << 16,
        "x280_tile": info["virtual_x"] | info["virtual_y"] << 16,
        "x280_credits": (
            info["lim_base"] + at + thread_module.layout()["KMX_RANKS_CREDITS"]
        ),
    }


def _x280_line(program, dispatch, backend, words, wait, mhz):
    """The X280 ranks' line and area for their thread (komorebi.thread.Ranks)."""
    messages = program.messages
    abi_headers, (constants, _) = abi.headers(), messages_header()
    cores = dispatch.grid_x * dispatch.grid_y
    ticks = thread_module.TIMER_HZ
    line = {
        "ranks": messages.x280_ranks,
        "cores": cores,
        "queue": words["x280_queue"],
        "cap": messages.queue_messages,
        "message_bytes": messages.message_bytes,
        "quota": messages.quota(cores),
        "counters": messages.counters,
        "idle": max(1, round(messages.idle_cycles / (mhz * 1e6) * ticks)),
        "inbox": backend.inbox.address,
        "inbox_messages": constants["kMessagesInbox"],
        "lines": dispatch.lim_base + abi_headers["KMD_LIM_MESSAGES"],
        "lines_tile": dispatch.l2_x | dispatch.l2_y << 16,
        "wait": round(wait * ticks),
    }
    tiles = [int(core_tile) for core_tile in dispatch.core_table()[:cores]]
    return thread_module.Ranks(line, tiles)


def _region_bytes(value):
    return int(value) if isinstance(value, int) else np.asarray(value).nbytes


def _region_initial(value):
    """The initial bytes as uint8, or None for a zeroed region."""
    if isinstance(value, int):
        return None
    return np.ascontiguousarray(value).view(np.uint8).reshape(-1)


def _host_array(value):
    """A shared region for the host fallbacks: uint32 words of its initial bytes."""
    size = _region_bytes(value)
    data = np.zeros(-(-size // 4) * 4, dtype=np.uint8)
    initial = _region_initial(value)
    if initial is not None:
        data[:size] = initial
    return data.view(np.uint32)


def _bank_regions(session, program, works):
    """{bank: _Regions} for the threads' own regions and the shared ones."""
    out = {}
    for bank in sorted({work.bank for work in works}):
        mine = [work for work in works if work.bank == bank]
        names = {name for work in mine for name in work.shared}
        shared = {
            name: value for name, value in program.shared.items() if name in names
        }
        out[bank] = _Regions(session, bank, mine, shared)
    return out


def _work_ring(dispatch, work):
    """The thread's ring (RingPlacement) and the kernel's end of it (tile, area)."""
    info = dispatch.work_info(work.cluster)
    return thread_module.work_ring(
        dispatch.lim_base,
        (dispatch.l2_x, dispatch.l2_y),
        info["lim_base"],
        (info["virtual_x"], info["virtual_y"]),
        work.cluster,
    )


def _pages(data, pages):
    """`data` (uint8) zero-padded to `pages` region pages, as [pages, words] uint32."""
    out = np.zeros(pages * thread_module.PAGE, dtype=np.uint8)
    out[: data.size] = data
    return out.view(np.uint32).reshape(pages, -1)


class _Regions:
    """The regions of one DRAM bank in one DRAM tensor whose pages of that bank hold
    them: page 0 the check line, then each thread's own regions (keys (cluster,
    name)), then the shared ones (("shared", name)). Written before the grid starts;
    read(key) only once the grid has stopped, from that bank's pages alone."""

    def __init__(self, session, bank, threads, shared):
        self.session = session
        self.check = thread_module.new_check_value()
        self.bank = bank
        entries = [
            ((thread.cluster, name), thread.region_bytes(name), thread.initial(name))
            for thread in threads
            for name in thread.regions
        ]
        entries += [
            (("shared", name), _region_bytes(value), _region_initial(value))
            for name, value in shared.items()
        ]
        self.placed, pages = {}, 1
        for key, size, _ in entries:
            self.placed[key] = thread_module.Placement(
                key[1], pages, -(-size // thread_module.PAGE), size
            )
            pages += self.placed[key].pages
        self.readable = False  # set by the run once the grid has stopped
        rows = np.zeros((DRAM_BANKS * pages, thread_module.PAGE // 4), dtype=np.uint32)
        mine = rows[self.bank :: DRAM_BANKS]
        mine[0, :2] = [self.check & abi.M32, self.check >> 32]
        for key, _, data in entries:
            placement = self.placed[key]
            if data is not None:
                mine[placement.page : placement.page + placement.pages] = _pages(
                    data, placement.pages
                )
        self.tensor = _words(session, rows)
        self.base = self.tensor.address
        if (
            self.base
            <= thread_module.IPI_ACK_DDR
            < self.base + pages * thread_module.PAGE
        ):
            raise StartError(
                f"the thread's regions at {self.base:#x} cover the IPI word"
            )

    def view(self, thread):
        """The regions `thread` sees, by name."""
        return _ThreadRegions(self, thread)

    def address(self, key):
        """(bank, bank address) of a region, for Tensix kernels."""
        return self.bank, self.base + self.placed[key].page * thread_module.PAGE

    def read(self, key):
        """The region's bytes, as uint8."""
        if not self.readable:
            raise RuntimeError("regions are read after run.stop()")
        placement, session = self.placed[key], self.session
        name = placement.name
        chunk_pages = (
            placement.pages
            if placement.pages * thread_module.PAGE <= REGION_WHOLE
            else REGION_CHUNK // thread_module.PAGE
        )
        spans = dram.bank_chunks(
            self.base,
            thread_module.PAGE,
            self.bank,
            placement.page,
            placement.pages,
            chunk_pages,
        )
        out = np.empty(placement.pages * thread_module.PAGE, np.uint8)
        for offset, span in zip(
            range(0, out.size, chunk_pages * thread_module.PAGE), spans
        ):
            (chunk,) = session._module.read_dram(session.device, [span])
            out[offset : offset + span[2]] = np.asarray(chunk, np.uint8)
            if len(spans) > 1:
                session.log(
                    f"region {name}: {offset + span[2]} of {out.size} bytes read"
                )
        return out[: placement.bytes].copy()


class _ThreadRegions:
    """One thread's regions in its bank's _Regions: its own, then the shared ones it
    names, in id order (table() gives the argument block's region table)."""

    def __init__(self, regions, thread):
        self.all = regions
        self.keys = {name: (thread.cluster, name) for name in thread.regions}
        self.keys |= {name: ("shared", name) for name in thread.shared}
        self.bank, self.base, self.check = regions.bank, regions.base, regions.check

    @property
    def readable(self):
        return self.all.readable

    @readable.setter
    def readable(self, value):
        self.all.readable = value

    def table(self):
        return [
            (self.address(name)[1], self.all.placed[key].bytes)
            for name, key in self.keys.items()
        ]

    def address(self, name):
        """(bank, bank address) of a region, for Tensix kernels."""
        return self.all.address(self.keys[name])

    def read(self, name):
        """The region's bytes, as uint8, after run.stop()."""
        return self.all.read(self.keys[name])


class _TransferBuffer(NamedTuple):
    """The L1 of a run's transfers on every core: two chunks and a word slot."""

    address: int
    chunk: int
    bytes: int


class _Backend:
    """The card side of a Run: the native port, the device tensors, the readback."""

    def __init__(self, session, dispatch, mhz, trace):
        self.session = session
        self.device = session.device
        self.port = dispatch
        self.mhz = mhz
        self.trace_cap = (trace + 3) // 4 * 4 if trace else 0  # whole 64-byte pages
        self.io = []
        self.control = None
        self.control_addr = 0
        self.control_offset = 0
        self.trace_tensor = None
        self.state_tensor = None
        self.inbox = None  # the X280 ranks' messages to each core, in L1
        self.launch = None  # the launched program, kept for reset_core
        self.read = False
        self.close_failed = (
            ""  # the runtime's error message when an abort did not restore
        )
        self.aborted = None  # {words, cores}: what the kernel left at an abort
        self.transfer = None  # a _TransferBuffer: with the host channel, or threads
        # a split card: (graph, dispatches, tags, [(first row, rows)], the islands whose
        # cores returned) of the one launch; `rows` this backend's island's
        self.split = None
        self.rows = None

    def block(self, data):
        """A command's block (uint8 data) in DRAM: a copy at the same address of every
        bank, written while the grid runs; fence() before the command is sent. The
        tensor; free it after the command."""
        words = -(-len(data) // 64) * 16  # whole 64-byte lines: a core reads lines
        tensor = self.session.empty((DRAM_BANKS, words), "uint32")
        padded = np.zeros(4 * words, np.uint8)
        padded[: len(data)] = data
        self.session._module.write_dram(
            self.device, list(range(DRAM_BANKS)), tensor.address, padded
        )
        return tensor

    def fence(self):
        """Returns once every block written so far is in DRAM."""
        self.session._module.dram_barrier(self.device)

    def inspect(self, count):
        """While the kernel runs: the inspect block with its first `count` records, and
        the dispatcher's count of commands it gave up (LIM words, read by the runtime).
        """
        read = self.port.inspect(count)
        return {
            "block": np.asarray(read["block"]).ravel(),
            "given_up": int(read["given_up"]),
        }

    def _control_blocks(self, cores, grid, port_lines=None, area_bytes=0):
        """Zeroed control blocks, one shard per core, padded so that the block can
        start on a 64-byte line: the same L1 address on every core. `port_lines`: the
        KM_X280_TO line of each served core, written once the address is known.
        `area_bytes`: the overlay area after each block (written by load_overlays)."""
        abi_headers = abi.headers()
        words = (abi_headers["KM_CONTROL_BYTES"] + area_bytes + CONTROL_ALIGN) // 4
        self.control = _words(self.session, np.zeros((cores, words)), grid)
        base = self.control.address
        self.control_addr = (base + CONTROL_ALIGN - 1) // CONTROL_ALIGN * CONTROL_ALIGN
        self.control_offset = (self.control_addr - base) // 4
        check_bases({"control block": self.control_addr})
        self.control_rows = np.zeros((cores, words), dtype=np.uint32)
        self._island_rows(cores, grid[0])
        if port_lines is not None:
            rows = self.control_rows
            to_offset = (
                self.control_offset
                + (abi_headers["KM_CONTROL_X280"] + abi_headers["KM_X280_TO"]) // 4
            )
            for core, (tile, box) in port_lines.lines.items():
                rows[core, to_offset + abi_headers["KM_X280_TO_TILE"]] = tile
                rows[core, to_offset + abi_headers["KM_X280_TO_BOX"]] = box
            self.control.write(rows)
        return self.control_addr

    def _island_rows(self, cores, columns):
        """Every core's island area in control_rows: island i of the load's graph,
        else island 0 of 1 with every core a rank (its header, then its table, then
        the link area's and the tensor table's addresses); a split card's islands their
        own rows."""
        abi_headers = abi.headers()
        graph, here = self.session.island or (islands.one(cores, columns), 0)
        parts = [(here, 0, cores)]
        if self.split is not None:
            graph, _, _, rows, _ = self.split
            parts = [
                (i, first * columns, n * columns) for i, (first, n) in enumerate(rows)
            ]
        at = self.control_offset + abi_headers["KM_CONTROL_ISLANDS"] // 4
        line = abi_headers["KM_ISLAND_HEADER_BYTES"] // 4
        plan = getattr(self.session, "link_plan", None)
        plans = getattr(self.session, "link_plans", None) if self.split else None
        for here, base, count in parts:
            plan = plans[here] if plans else plan
            area = plan.area if plans else self.session.link_area
            table = np.frombuffer(graph.table(here), np.uint32)
            header = [here, len(graph.islands), graph.first(here), graph.count(here)]
            header.append(base)
            mine = self.control_rows[base : base + count]
            mine[:, at : at + len(header)] = header
            mine[:, at + line : at + line + len(table)] = table
            if plan is not None:  # the relay's words after the table (km_relay.hpp)
                relay = at + abi_headers["KM_ISLAND_RELAY"]
                mine[:, relay : relay + 3] = plan.relay
                for core, tile in getattr(plan, "relay_tiles", {}).items():
                    self.control_rows[core, relay + 2] = tile
            mine[:, at + abi_headers["KM_ISLAND_LINK_AREA"] // 4] = area
            mine[:, at + abi_headers["KM_ISLAND_TENSORS"] // 4] = (
                self.session.tensor_table
            )
            mine[:, at + abi_headers["KM_ISLAND_FORWARD"] // 4] = getattr(
                plan, "forward", 0
            )

    def _dynamic(self, program, cores, depth, base, grid, x280=None):
        """A dynamic handler's queues in DRAM (zero-filled by the cores) and its
        reader / writer and compute words; status lines in the kernel's LIM; with the
        X280 ranks' words `x280`, each core's inbox in L1 (zeroed)."""
        messages = program.messages
        if messages is None:
            return [], []
        if x280 is not None:
            constants, _ = messages_header()
            words = constants["kMessagesInbox"] * messages.message_bytes // 4
            self.inbox = _words(self.session, np.zeros((cores, words)), grid)
            self.io.append(self.inbox)
            x280 = x280 | {"inbox": self.inbox.address}
        abi_headers = abi.headers()
        messages.check(cores, MESSAGES_DRAM_BYTES, depth)
        page = 32 * messages.message_bytes  # kMessagesPageMessages messages a page
        queue = self.session.empty(
            (cores * messages.queue_messages // 32, page // 4), "int32"
        )
        self.io.append(queue)
        body = 0
        if messages.body_bytes:  # a page of body_bytes a queue position
            bodies = self.session.empty(
                (cores * messages.queue_messages, messages.body_bytes // 4), "int32"
            )
            self.io.append(bodies)
            body = bodies.address
        lines = base + abi_headers["KMD_LIM_MESSAGES"]
        return messages.args(queue.address, lines, body, x280), messages.compute_args(
            cores
        )

    def describe(
        self,
        program,
        sources,
        tag,
        depth,
        wait,
        idle,
        trace,
        port_lines=None,
        placed=None,
        x280=None,
    ):
        """The program descriptor for every core, with the start-up arguments; with
        `placed` (a programs.Plan) the runtime's kernels, per group of cores; `x280`:
        the X280 ranks' words of a dynamic program."""
        abi_headers = abi.headers()
        port = self.port
        grid = (port.grid_x, port.grid_y)
        cores = grid[0] * grid[1]
        area = overlay.constants()["kOverlayAreaBytes"]  # the service overlay's too
        control = self._control_blocks(cores, grid, port_lines, area)
        table = np.zeros((1, abi_headers["KM_CORES_MAX"]), dtype=np.uint64)
        table[0] = port.core_table()
        cores_tensor = _words(self.session, table)
        self.io = [self.control, cores_tensor]
        if self.session.channel is not None or program.threads:
            # also the threads' copies
            chunk = self._transfer_chunk(program, placed, cores)
            self.transfer = self._transfer_buffer(cores, grid, chunk)
        if self.trace_cap:
            self.trace_tensor = _words(
                self.session,
                np.zeros((cores, self.trace_cap * abi_headers["kTraceWords"])),
            )
            self.io.append(self.trace_tensor)
        base = port.lim_base
        groups = abi.column_groups(grid, cores)
        common = {
            "control": control,
            "cores": cores_tensor.address,
            "dispatcher_coordinates": port.l2_x | port.l2_y << 16,
            "ack_base": base + abi_headers["KMD_LIM_ACKS"],
            "ready_base": base + abi_headers["KMD_LIM_READY"],
            "tag": tag,
            "depth": depth,
            "workers": cores,
            "row": grid[0],
            "wait_limit": abi.limit_units(wait, self.mhz),
            "idle_limit": abi.limit_units(idle, self.mhz) if idle else 0,  # 0: none
            "trace": self.trace_tensor.address if self.trace_tensor else 0,
            "trace_cap": self.trace_cap,
        }
        messages_reader_writer_args, messages_compute_args = self._dynamic(
            program, cores, depth, base, grid, x280
        )
        owners = [(0, common, base)] * cores  # per core: its island's first core
        if self.split is not None:  # each island's cores hello its own dispatcher
            owners, groups = self._split_owners(common, grid)
        reader_args, writer_args, compute_args = [], [], []
        for core in range(cores):
            x, y = core % grid[0], core // grid[0]
            leader, rank, peers, slot = groups[core]
            first_core, mine, lim = owners[core]
            start = abi.start_args(
                mine
                | {
                    "me": core,
                    "mailbox": (
                        lim + abi_headers["KMD_LIM_MAILBOX"]
                        if core == first_core
                        else 0
                    ),
                    "leader": leader,
                    "rank": rank,
                    "peers": peers,
                    "ready_slot": slot,
                }
            )
            reader, writer, compute = _core_args(program, placed, core)
            reader_args.append(
                (x, y, start + messages_reader_writer_args + list(reader))
            )
            writer_args.append(
                (x, y, start + messages_reader_writer_args + list(writer))
            )
            limits = [common["wait_limit"], common["idle_limit"]]
            compute_args.append(
                (x, y, [control, core, *limits, *messages_compute_args, *compute])
            )
        launch = self.session._module.Launch(self.device)
        if placed is not None:
            args = (reader_args, writer_args, compute_args)
            self._describe_groups(launch, placed, sources, args, grid)
            return launch
        all_cores = [(0, 0, grid[0] - 1, grid[1] - 1)]
        include = [str(abi.headers().abi.parents[2])]
        defines = [(name, str(value)) for name, value in program.defines.items()]
        if program.messages is not None:
            defines += program.messages.defines()
        self.cb_indices = set()  # every buffer index of the launch, the slots' apart
        for index, fmt, page, pages in self._cbs(program.buffers):
            launch.cb(all_cores, index, fmt, page, page * pages)
            self.cb_indices.add(index)
        kernels = [("reader", reader_args, 0, {}), ("writer", writer_args, 1, {})]
        kernels.append(("compute", compute_args, 2, compute_config(program.compute)))
        for kernel, args, processor, config in kernels:
            launch.kernel(
                str(sources[kernel]),
                all_cores,
                processor,
                program.kernel_compile_args(kernel),
                includes=include,
                args=args,
                **kernel_build(program, kernel, defines),
                **config,
            )
        return launch

    def _split_owners(self, common, grid):
        """A split card's per-core (island's first core, start-up words, LIM base) and
        start-up groups: each island's column leaders report to its own dispatcher."""
        abi_headers = abi.headers()
        _, ports, tags, rows, _ = self.split
        owners, groups = [], {}
        for (first, count), port, tag in zip(rows, ports, tags):
            lim = port.lim_base
            mine = common | {
                "dispatcher_coordinates": port.l2_x | port.l2_y << 16,
                "ack_base": lim + abi_headers["KMD_LIM_ACKS"],
                "ready_base": lim + abi_headers["KMD_LIM_READY"],
                "tag": tag,
            }
            low = first * grid[0]
            owners += [(low, mine, lim)] * (count * grid[0])
            local = abi.column_groups((grid[0], count), count * grid[0])
            for core, (leader, rank, peers, slot) in local.items():
                groups[low + core] = (low + leader, rank, peers, slot)
        return owners, groups

    def _transfer_chunk(self, program, placed, cores):
        """The chunk of the run's transfer buffer: CHUNK_DEFAULT, or the largest page
        of the session's DRAM tensors when two of them take at most half of what a core
        has left after its buffers. A run with overlays, or whose groups switch
        program, keeps the default: it has slots to place."""
        page = self.session.largest_page
        held = placed.programs if placed else [program]
        if (placed and placed.switch) or any(each.overlays for each in held):
            return channel_module.CHUNK_DEFAULT
        buffers = max(
            sum(buffer.page_bytes * buffer.pages for buffer in each.buffers)
            for each in held
        )
        buffers += 2 * abi.headers()["kTokenBytes"] + self.trace_cap * 16
        spare = self.session.l1()[1] - buffers - self.control.nbytes // cores
        asked = self.session.transfer_chunk  # a session over two cards: LINK_CHUNK
        word = abi.headers()["KM_TRANSFER_WORD_BYTES"]
        if asked and asked >= page and 2 * asked + word <= spare // 2:
            return asked
        if page <= channel_module.CHUNK_DEFAULT:
            return channel_module.CHUNK_DEFAULT
        return (
            channel_module.chunk_for(page, spare // 2) or channel_module.CHUNK_DEFAULT
        )

    def _transfer_buffer(self, cores, grid, chunk):
        """The L1 every core moves a transfer's chunks through (a sharded tensor)."""
        size = 2 * chunk + abi.headers()["KM_TRANSFER_WORD_BYTES"]
        tensor = self.session.empty(
            (cores, (size + NOC_ALIGN) // 4), "uint32", shard_grid=grid
        )
        self.io.append(tensor)
        address = -(-tensor.address // NOC_ALIGN) * NOC_ALIGN
        return _TransferBuffer(address, chunk, size + NOC_ALIGN)

    def _transfer_rows(self):
        """The L1 budget table's row of the transfer buffer, if the run has one."""
        if self.transfer is None:
            return []
        return [("host transfer buffer", self.transfer.bytes)]

    def _cbs(self, buffers):
        """(index, format, page bytes, pages) of `buffers` and the runtime's own."""
        return _program_cbs(buffers) + self._runtime_cbs()

    def _runtime_cbs(self):
        abi_headers = abi.headers()
        token_bytes = abi_headers["kTokenBytes"]
        cbs = [
            (abi_headers["kCbTokenCompute"], "uint32", token_bytes, 1),
            (abi_headers["kCbTokenWriter"], "uint32", token_bytes, 1),
        ]
        if self.trace_cap:
            cbs.append((abi_headers["kCbTrace"], "uint32", self.trace_cap * 16, 1))
        return cbs

    def _switch_layouts(self, placed):
        """In a run whose groups switch: each program's buffers placed after the
        runtime's own ({field: layout}), and the end of the largest."""
        _, kernel_free, l1_bytes = self.session._module.kernel_buffer(self.device)
        _, start = programs.buffer_layout(self._runtime_cbs(), l1_bytes - kernel_free)
        self.switch_layouts, end = {}, start
        for program in placed.programs:
            # a group's buffers are launched in index order (joint_buffers)
            ordered = sorted(program.buffers, key=lambda buffer: buffer.index)
            layout, last = programs.buffer_layout(_program_cbs(ordered), start)
            self.switch_layouts[program.program_field] = layout
            end = max(end, last)
        self.switch_base = l1_bytes - kernel_free
        return end

    def _describe_groups(self, launch, placed, sources, args, grid):
        """The runtime's kernels and each group's buffers on its cores: a group's
        kernels are built apart (KM_PROGRAM_GROUP), with its buffers and options."""
        include = [str(abi.headers().abi.parents[2])]
        free = self.session.l1()[1]
        self.group_buffers = []
        self.cb_indices = {  # a switch may give any core any program's buffers
            buffer.index for program in placed.programs for buffer in program.buffers
        }
        align = programs.CB_ALIGN
        if placed.switch:  # the runtime's buffers first, then room for the largest
            reserve = -(-self._switch_layouts(placed) // align) * align
            self.switch_reserve = reserve - self.switch_base
        for number, group in enumerate(placed.groups):
            held = [placed.programs[index] for index in group.programs]
            buffers = programs.joint_buffers(held, free)
            self.group_buffers.append(buffers)
            cores = programs.rectangles(group.first, group.count, grid[0])
            cbs = self._cbs(buffers)
            if placed.switch:
                cbs = self._runtime_cbs() + _program_cbs(buffers)
                _, end = programs.buffer_layout(cbs, self.switch_base)
                pad = reserve - -(-end // align) * align
                if pad > 0:
                    index = min(set(range(32)) - {cb[0] for cb in cbs})
                    cbs.append((index, "uint32", pad, 1))
            for index, fmt, page, pages in cbs:
                launch.cb(cores, index, fmt, page, page * pages)
                self.cb_indices.add(index)
            cb_in, cb_out = programs.start_buffers(buffers)
            defines = [
                ("KM_PROGRAM_GROUP", str(number)),
                ("KM_PROGRAM_CB_IN", str(cb_in)),
                ("KM_PROGRAM_CB_OUT", str(cb_out)),
            ]
            if placed.switch:  # the kernels do not depend on the group's buffers
                defines = [
                    ("KM_PROGRAM_GROUP", str(number)),
                    ("KM_PROGRAM_SWITCH", "1"),
                ]
            first = held[0] if held else None
            if placed.switch and first is None:
                first = placed.programs[0]  # one build for every group's kernels
            mine = range(group.first, group.first + group.count)
            for processor, kernel in enumerate(("reader", "writer", "compute")):
                config = (
                    compute_config(first.compute if first else {})
                    if kernel == "compute"
                    else {}
                )
                build = (
                    kernel_build(first, kernel, defines)
                    if first
                    else {"defines": defines}
                )
                launch.kernel(
                    str(sources[kernel]),
                    cores,
                    processor,
                    [],
                    includes=include,
                    args=[args[processor][core] for core in mine],
                    **build,
                    **config,
                )

    def _slot_buffer(self, descriptor, slot_bytes):
        """Every core's slots as the launch's first circular buffer, at the L1 base,
        so they lie below 1 MB and the buffers above them; returns their address."""
        base = L1_BYTES - self.session.l1()[1]
        address, size, index = slot_area(
            base, slot_bytes, self.cb_indices, self.session.slow_code
        )
        grid = (self.port.grid_x, self.port.grid_y)
        cores = [(0, 0, grid[0] - 1, grid[1] - 1)]
        descriptor.cb(cores, index, "uint32", size, size, first=True)
        # a switch's buffers of every program field move up with the rest
        if getattr(self, "switch_layouts", None):
            for layout in self.switch_layouts.values():
                for key, (at, *rest) in layout.items():
                    layout[key] = (at + size, *rest)
        self.session.log(f"slots: {size:,} B from {address:#x} (buffer {index})")
        return address

    def load_overlays(self, program, descriptor):
        """Builds the program's overlays and the service overlay against the kernels
        just built, puts each part in every DRAM bank, places the slots in L1 (refused,
        with the L1 budget table, when they do not fit), writes each overlay area."""
        session, abi_headers = self.session, abi.headers()
        started = time.perf_counter()
        recipes = {}
        for _, builds in descriptor.recipes():
            for row in builds:
                recipe = overlay.Recipe.from_native(row)
                recipes[recipe.target] = recipe
        built = overlay.build_overlays(
            [*program.overlays.values(), overlay.service()],
            program.overlay_slots,
            recipes,
            native.cache_directory() / "overlays",
            session.log,
        )
        program.overlays_built = built
        grid = (self.port.grid_x, self.port.grid_y)
        cores = grid[0] * grid[1]
        buffers = sum(buffer.page_bytes * buffer.pages for buffer in program.buffers)
        buffers += 2 * abi_headers["kTokenBytes"] + self.trace_cap * 16
        rows = [
            ("circular buffers", buffers),
            ("control block and overlay area", self.control.nbytes // cores),
        ]
        rows += [
            (f"slots: {part}, {count} x {size:,} B", count * size)
            for part, (count, size) in built.slots.items()
        ]
        rows += self._transfer_rows()
        table, fits = overlay.budget(session.l1()[1], rows)
        session.log(f"L1 budget of a core:\n{table}")
        if not fits:
            raise L1BudgetError(program.name, table)
        image_rows, placed = overlay.dram_rows(built)
        image = _words(session, image_rows)
        slot_base = self._slot_buffer(descriptor, built.slot_bytes)
        self.io.append(image)
        area = overlay.area_words(
            built, placed, image.address, image.aligned_page_bytes, slot_base
        )
        at = self.control_offset + abi_headers["KM_CONTROL_BYTES"] // 4
        self.control_rows[:, at : at + len(area)] = area
        self.control.write(self.control_rows)
        session.log(
            f"overlays {[*program.overlays, 'service']} built and placed: slots from "
            f"{slot_base:#x}, {image.nbytes:,} B of DRAM "
            f"({time.perf_counter() - started:.2f} s)"
        )

    def _switch_line(self, area, placed, group):
        """A group's switch line in its overlay area: the program field it holds and
        its buffers (none for a group without one); every number in the table."""
        c = overlay.constants()
        area[c["kOverlayCount"]] = c["kOverlaysMax"]
        line = c["kOverlaySwitch"] // 4
        area[line + c["kSwitchHolding"]] = c["kSwitchNone"]
        area[line + c["kSwitchMask"]] = 0
        if group.programs:
            program_field = group.programs[0]
            area[line + c["kSwitchHolding"]] = program_field
            area[line + c["kSwitchMask"]] = sum(
                1 << index for index in self.switch_layouts[program_field]
            )

    def _switch_records(self, placed, image, placed_rows, symbols):
        """Each program's record in DRAM (its table entries and buffers), and why a
        group cannot switch, if the groups' kernels differ: self.switch for the Run."""
        c = overlay.constants()
        records = {}
        for program in placed.programs:
            number = c["kOverlayService"] - 1 - program.program_field
            table = []
            for part in overlay.PARTS:
                row, size = placed_rows.get((number, part), (None, 0))
                address = image.address + row * image.aligned_page_bytes if size else 0
                table.append((address, size))
            words = programs.record_words(
                program.program_field, self.switch_layouts[program.program_field], table
            )
            records[(program.program_field, "record")] = words.tobytes()
        rows, placed_records = overlay.dram_rows(overlay.Built(records, {}))
        tensor = _words(self.session, rows)
        self.io.append(tensor)
        refused = None
        differ = sorted(
            (target, name)
            for each in symbols[1:]
            for target, names in each.items()
            for name in names.keys() | symbols[0].get(target, {}).keys()
            if names.get(name) != symbols[0].get(target, {}).get(name)
        )
        if differ:
            refused = (
                "the groups' runtime kernels differ (compute options or opt levels; "
                f"{differ[0][0]}'s {differ[0][1]}, {len(differ)} symbols), so a "
                "program's parts would not fit another group's kernels"
            )
        self.switch = {
            "records": {
                key[0]: tensor.address + row * tensor.aligned_page_bytes
                for key, (row, _) in placed_records.items()
            },
            "refused": refused,
        }

    def load_programs(self, placed, descriptor):
        """Builds each program as an overlay against its group's runtime kernels, puts
        the parts in every DRAM bank, places the slots (refused, with the L1 budget
        table, when a group's do not fit) and writes each group's overlay area."""
        session, abi_headers = self.session, abi.headers()
        started = time.perf_counter()
        by_group = _group_recipes(descriptor.recipes(), len(placed.groups))
        grid = (self.port.grid_x, self.port.grid_y)
        cores = grid[0] * grid[1]
        declared = placed.programs[0].overlay_slots
        built, symbols = [], []
        for number, group in enumerate(placed.groups):
            recipes = by_group[number]
            if placed.switch:  # a part runs on any group's kernels: the same globals
                symbols.append(
                    {
                        target: {
                            name: value
                            for name, (value, _, binding) in overlay.elf_symbols(
                                recipe, recipe.elf
                            ).items()
                            if binding != "LOCAL" and not UNIT_SYMBOL.search(name)
                        }
                        for target, recipe in recipes.items()
                    }
                )
            held = [placed.programs[index] for index in group.programs]
            made = overlay.build_overlays(
                [
                    *(
                        programs.program_overlay(program, switch=placed.switch)
                        for program in held
                    ),
                    overlay.service(),
                ],
                declared,
                recipes,
                native.cache_directory() / "overlays",
                session.log,
            )
            for program in held:
                program.overlays_built = made
            built.append(made)
            buffers = sum(
                buffer.page_bytes * buffer.pages
                for buffer in self.group_buffers[number]
            )
            buffers += 2 * abi_headers["kTokenBytes"] + self.trace_cap * 16
            rows = [
                ("circular buffers", buffers),
                ("control block and overlay area", self.control.nbytes // cores),
            ]
            rows += [
                (f"slots: {part}, {count} x {size:,} B", count * size)
                for part, (count, size) in made.slots.items()
            ]
            rows += self._transfer_rows()
            table, fits = overlay.budget(session.l1()[1], rows)
            names = ", ".join(program.name for program in held)
            session.log(f"L1 budget of a core of {names}:\n{table}")
            if not fits:
                raise L1BudgetError(names, table)
        images, stored = {}, []
        service = overlay.constants()["kOverlayService"]
        for made in built:
            mine = {}
            for key, image in made.images.items():
                at = key
                if key[0] == service:  # a copy for each different build of it
                    same = [k for k, v in images.items() if k[:2] == key and v == image]
                    at = same[0] if same else (*key, len(images))
                images[at] = image
                mine[key] = at
            stored.append(mine)
        if placed.switch:  # one set of slots, every program's parts fit any core's
            common = _common_slots(placed, declared, built)
            built = [overlay.Built(made.images, common.slots) for made in built]
            buffers = self.switch_reserve  # the runtime's buffers included
            rows = [
                ("circular buffers (the largest program)", buffers),
                ("control block and overlay area", self.control.nbytes // cores),
            ]
            rows += [
                (f"slots: {part}, {count} x {size:,} B", count * size)
                for part, (count, size) in common.slots.items()
            ]
            rows += self._transfer_rows()
            table, fits = overlay.budget(session.l1()[1], rows)
            session.log(f"L1 budget of a core:\n{table}")
            if not fits:
                raise L1BudgetError(", ".join(p.name for p in placed.programs), table)
        image_rows, placed_rows = overlay.dram_rows(overlay.Built(images, {}))
        image = _words(session, image_rows)
        slot_base = self._slot_buffer(
            descriptor, max(made.slot_bytes for made in built)
        )
        self.io.append(image)
        at = self.control_offset + abi_headers["KM_CONTROL_BYTES"] // 4
        for group, made, keys in zip(placed.groups, built, stored, strict=True):
            mine = {key: placed_rows[at] for key, at in keys.items()}
            area = overlay.area_words(
                made, mine, image.address, image.aligned_page_bytes, slot_base
            )
            if placed.switch:
                self._switch_line(area, placed, group)
            rows = slice(group.first, group.first + group.count)
            self.control_rows[rows, at : at + len(area)] = area
        self.control.write(self.control_rows)
        if placed.switch:
            self._switch_records(placed, image, placed_rows, symbols)
        session.log(
            f"programs {[program.name for program in placed.programs]} built and "
            f"placed: slots from {slot_base:#x}, {image.nbytes:,} B of DRAM "
            f"({time.perf_counter() - started:.2f} s)"
        )

    def finish(self, timeout):
        """Waits for the kernel to finish, stops it, then closes the X280 runtime."""
        try:
            return self.port.finish(timeout)
        finally:
            self.port.close()
            self.session.log("x280 runtime closed")

    def abort(self):
        """The failure path: the stop flag to every core, a timed wait for the grid,
        then the runtime's close, so core 0 is not held behind a grid that runs on.
        Returns whether cluster 0 was restored; `aborted` holds what the kernel left."""
        try:
            self.stop_grid()
            back = self._wait_grid(ABORT_GRID_S)
            self.session.log("grid returned" if back else "grid still running")
        except Exception as error:  # noqa: BLE001 (the close below always runs)
            self.session.log(
                f"stopping the grid raised {type(error).__name__}: {error}"
            )
        cleaned = self.port.close()
        self.close_failed = "" if cleaned else self.port.close_failed
        self.aborted = self.port.aborted()
        why = f": {self.close_failed}" if self.close_failed else ""
        if self.aborted is not None:
            abandoned = self.aborted["x280_cores"][abi.headers()["KMD_H_ABANDONED"]]
            why += f"; {abandoned} commands abandoned"
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
            for line_index, line in enumerate(words.reshape(-1, 8)):
                text = " ".join(f"{int(word):#x}" for word in line)
                self.session.log(f"kernel state +{64 * line_index:#05x}: {text}")
        except Exception as error:  # noqa: BLE001 (a failure path: reported)
            self.session.log(f"kernel state not read ({type(error).__name__}: {error})")

    def kernel_state(self):
        """The first STATE_DUMP_LINES lines of the kernel's state (uint64 words) from
        the first page of each bank: the one whose args and lim words name the LIM
        base is the bank the cluster's Memory Port reaches. None if no page does."""
        module = self.session._module
        banks = module.dram_banks(self.device)
        (pages,) = read_tensors(self.device, module, [self.state_tensor], 0, banks)
        words = np.asarray(pages, np.uint8)[:, : 64 * STATE_DUMP_LINES]
        words = np.ascontiguousarray(words).view("<u8")
        base = self.port.lim_base
        found = [
            line for line in words if line[2] == base and line[3] == base
        ]  # args, lim
        return found[0] if len(found) == 1 else None

    def kernel_report(self):
        """While the kernel runs: whether core 0 still runs its loop (its pass marker
        over PROBE_GAP_S), its state and where, and the runtime's view of each core."""
        probe = self.port.probe(PROBE_GAP_S)
        if not probe["running"]:
            return "no dispatcher is running"
        if probe["failed"]:
            states = probe["x280_states"]
            return f"the dispatcher's {probe['failed']}; X280 cores: {states}"
        result = device_state.result_block(probe["words"])
        before, after = probe["marker_before"], probe["marker_after"]
        doing = "is running its loop" if after != before else "is not running its loop"
        report = (
            f"the dispatcher's core 0 {doing} (pass marker {before} -> {after} "
            f"in {PROBE_GAP_S} s; state {result['state_name']}, "
            f"in {result['where_name']}); "
            f"X280 cores: {probe['x280_states']}"
        )
        self.session.log(report)
        return report

    def stop_grid(self):
        """Writes the stop flag into every core (of this island on a split card)
        through the device API."""
        abi_headers = abi.headers()
        module, grid_x = self.session._module, self.port.grid_x
        for core in self._cores():
            address = self.control_addr + abi_headers["KM_CONTROL_STOP"]
            module.write_l1(self.device, core % grid_x, core // grid_x, address, [1])
        self.session.log("stop flag written by the host")

    def _cores(self):
        """The cores of this backend's island on a split card; else every core."""
        grid_x, grid_y = self.port.grid_x, self.port.grid_y
        if self.rows is None:
            return range(grid_x * grid_y)
        return range(self.rows[0] * grid_x, sum(self.rows) * grid_x)

    def _wait_grid(self, timeout):
        """Whether this backend's cores (_cores) returned within `timeout`."""
        first, rows = self.rows or (0, self.port.grid_y)
        return self.session._module.wait_grid(
            self.device, self.port.grid_x, rows, timeout, first
        )

    def read_back(self, stop_grid=False):
        """After the X280 runtime is closed: waits (with a timeout) for the grid's
        kernels, then reads the control blocks ([cores, KM_CONTROL_BYTES / 4] uint32)
        and the trace. `stop_grid`: the dispatcher was not there to stop the grid, so
        the host first writes the stop flag into every core through the device API."""
        if self.port.is_open:
            raise RuntimeError("close the X280 runtime before reading back")
        abi_headers = abi.headers()
        module, grid_x, grid_y = (
            self.session._module,
            self.port.grid_x,
            self.port.grid_y,
        )
        if stop_grid:
            self.stop_grid()
        started = time.monotonic()
        timeout = self.session.grid_timeout
        if not self._wait_grid(timeout):
            self.session.stuck = True
            raise GridTimeout(
                f"the grid's kernels did not return within {timeout} s after the stop. "
                "The session does not close the device; at the process's exit the "
                f"runtime waits {exit_wait()} for the cores, then puts them in reset. "
                "A thread that is only stuck is recovered by the next open; a kernel "
                "that left a compute unit running is not, and if the next open fails "
                "the card needs a reset"
            )
        words = abi_headers["KM_CONTROL_BYTES"] // 4
        if self.split is not None:  # the other islands may run: no buffer read
            blocks = np.zeros((grid_x * grid_y, words), np.uint32)
            for core in self._cores():
                blocks[core] = module.read_l1(
                    self.device, core % grid_x, core // grid_x, self.control_addr, words
                )
            returned = self.split[4]
            returned.add(self.rows)
            if len(returned) == len(self.split[3]):
                self.device.finish()  # returns at once: every island is done
            self.session.log(
                f"island cores returned ({time.monotonic() - started:.1f} s)"
            )
        else:
            self.device.finish()  # returns at once: every core is done
            self.session.log(f"grid returned ({time.monotonic() - started:.1f} s)")
            rows = _read_uint32(self.control)
            start = self.control_offset
            blocks = rows[:, start : start + words].astype(np.uint32)
        trace = None
        if self.trace_tensor is not None:
            words = _read_uint32(self.trace_tensor, self.device, module).astype(
                np.uint32
            )
            trace = words.reshape(
                words.shape[0], self.trace_cap, abi_headers["kTraceWords"]
            )
        self.session.log("state and trace read back")
        return blocks, trace
