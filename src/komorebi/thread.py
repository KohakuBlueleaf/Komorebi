"""An X280 thread on a work cluster: its description in a Program, the argument block
and region table the host writes, the result block it reads back, and the host
fallback. Nothing here needs a device: the card is reached through `port` and the
regions through `regions` (see ThreadRun).
"""

import functools
import random
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from komorebi import abi, native

TIMER_HZ = 50_000_000  # the X280's timer: kmx_time() ticks per second
PAGE = 4096  # region granularity, bytes; regions are whole pages of one bank
# The DRAM bank each work cluster reaches through its Memory Port (cluster: bank),
# measured on a p150a; the thread's start-up checks it (KMX_ERR_BANK otherwise).
BANK_OF_CLUSTER = {1: 6, 2: 7, 3: 7}
IPI_ACK_DDR = 0x470020  # the X280 runtime's IPI word, at this address in every bank


@functools.cache
def layout():
    """The KMX_* constants of kmx_layout.h."""
    return abi.constants(native.kmx_dir() / "kmx_layout.h")


@dataclass
class Thread:
    """A program's X280 thread: C `sources` linked with kmx, run on `cluster`;
    `regions`: name -> size in bytes or initial contents (ids in dict order);
    `words`: the program's argument words; `deadline`: seconds for the whole thread;
    `host_fallback(regions, words) -> (code, words)`: the same function on the host."""

    sources: list
    regions: dict
    words: list = field(default_factory=list)
    cluster: int = 1
    host_fallback: Callable | None = None
    deadline: float = 10.0
    includes: list = field(default_factory=list)
    defines: dict = field(default_factory=dict)
    prefetch: str = "documented"  # or "keep": the L2 prefetchers as they are

    def __post_init__(self):
        k = layout()
        if self.cluster not in BANK_OF_CLUSTER:
            raise ValueError(f"cluster {self.cluster}: work clusters are 1 to 3")
        if not 1 <= len(self.regions) <= k["KMX_REGIONS_MAX"]:
            raise ValueError(f"1 to {k['KMX_REGIONS_MAX']} regions")
        if len(self.words) > k["KMX_WORDS_MAX"]:
            raise ValueError(f"at most {k['KMX_WORDS_MAX']} words")
        if any(not 0 <= int(w) <= abi.M32 for w in self.words):
            raise ValueError("words are 32-bit unsigned")
        ticks = self.deadline * TIMER_HZ
        if not 0 < ticks <= k["KMX_DEADLINE_MAX"]:
            raise ValueError(f"deadline {self.deadline} s is out of range")
        if self.prefetch not in ("documented", "keep"):
            raise ValueError(f"prefetch={self.prefetch!r}: 'documented' or 'keep'")
        for name in self.regions:
            if self.region_bytes(name) <= 0:
                raise ValueError(f"region {name!r} is empty")

    def region_bytes(self, name):
        value = self.regions[name]
        return int(value) if isinstance(value, int) else np.asarray(value).nbytes

    def initial(self, name):
        """The region's initial bytes as uint8, or None for a zeroed region."""
        value = self.regions[name]
        if isinstance(value, int):
            return None
        return np.ascontiguousarray(value).view(np.uint8).reshape(-1)

    def source_paths(self):
        paths = [Path(s) for s in self.sources]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"no thread source {missing}")
        return paths


class Placement(NamedTuple):
    """Where a region lives: its first page in the bank and its size."""

    name: str
    page: int
    pages: int
    bytes: int


def place(thread):
    """Page 0 is the check line's; each region starts on a page of its own."""
    out, at = [], 1
    for name in thread.regions:
        n = thread.region_bytes(name)
        pages = -(-n // PAGE)
        out.append(Placement(name, at, pages, n))
        at += pages
    return out, at


class RingPlacement(NamedTuple):
    """Where the thread's command ring lives, as its cluster addresses it: command
    slots, the kernel's consumed word, the events, the read-count line."""

    cmd: int
    consumed: int
    events: int
    read: int
    cmd_depth: int
    event_depth: int
    cached: bool = False


def args_block(thread, base, check_value, run_value=0, ring=None):
    """The thread's whole LIM block (KMX_LIM_BYTES): the argument block with the
    region table at bank address `base` (page 0 the check line), the ring's line
    (zero without a `ring`, a RingPlacement), the rest zeroed."""
    k = layout()
    placed, _ = place(thread)
    block = bytearray(k["KMX_LIM_BYTES"])
    flags = k["KMX_FLAG_KEEP_PREFETCH"] if thread.prefetch == "keep" else 0
    head = struct.pack(
        "<QIIQIIQQQ",
        run_value,
        k["KMX_MAGIC"],
        k["KMX_VERSION"],
        int(thread.deadline * TIMER_HZ),
        len(placed),
        len(thread.words),
        base,
        check_value,
        flags,
    )
    block[: len(head)] = head
    for i, p in enumerate(placed):
        at = k["KMX_ARGS_REGIONS"] + 16 * i
        block[at : at + 16] = struct.pack("<QQ", base + p.page * PAGE, p.bytes)
    words = struct.pack(f"<{len(thread.words)}I", *map(int, thread.words))
    at = k["KMX_LIM_ARGS"] + k["KMX_ARGS_WORDS"]
    block[at : at + len(words)] = words
    if ring is not None:
        flags = k["KMX_RING_CACHED"] if ring.cached else 0
        line = struct.pack(
            "<4Q3I5I", *ring[:4], ring.cmd_depth, ring.event_depth, flags, *[0] * 5
        )
        block[k["KMX_LIM_RING"] : k["KMX_LIM_RING"] + 64] = line
    return bytes(block)


def new_check_value():
    """A random nonzero 64-bit value for the check line."""
    return random.getrandbits(64) | 1


class ThreadResult(NamedTuple):
    """The result block: state and library error (names), the thread's code and
    words, its last progress stamp and the times in microseconds. `ran` is False
    when the thread did not return by the timeout (it was stopped)."""

    ran: bool
    state: str
    error: str
    code: int
    words: list
    where: int
    where_us: float
    us: float


def _names(prefix):
    return {
        v: n[len(prefix) :].lower() for n, v in layout().items() if n.startswith(prefix)
    }


def parse_result(raw, ran=True):
    """A ThreadResult from the 256 bytes of the result block."""
    k = layout()
    w = struct.unpack("<64I", bytes(raw)[:256])
    n = min(w[3], k["KMX_RESULT_WORDS_MAX"])
    return ThreadResult(
        ran=ran,
        state=_names("KMX_STATE_").get(w[0], f"unknown {w[0]}"),
        error=_names("KMX_ERR_").get(w[1], f"unknown {w[1]}"),
        code=w[2],
        words=list(w[16 : 16 + n]),
        where=w[4],
        where_us=w[5] / 50.0,
        us=w[6] / 50.0,
    )


class ThreadRun:
    """A program's thread in one run: on the card through `port` (the work_* calls
    of the native module), or through its host fallback. `regions` reads a region's
    bytes from the card; `image` is the thread's image, `base` the regions' bank
    address, `check_value` what the host wrote into the check line."""

    def __init__(
        self, thread, port, regions, image=None, on_card=True, base=0, check_value=1
    ):
        self.thread = thread
        self.port = port
        self.regions = regions
        self.image = image
        self.on_card = on_card
        self.base = base
        self.check_value = check_value
        self.running = False
        self.result = None
        self.arrays = None  # the host fallback's regions, uint32, once it ran
        self.ring = None  # RingPlacement of the command ring, when the kernel has one

    def start(self):
        """Starts the image on the card (no-op for the host fallback)."""
        if not self.on_card:
            if self.thread.host_fallback is None:
                raise ValueError("the thread has no host_fallback")
            return
        block = args_block(self.thread, self.base, self.check_value, ring=self.ring)
        self.port.work_start(self.thread.cluster, str(self.image), block)
        self.running = True

    def wait(self, timeout):
        """Waits (bounded) for the thread to return, stops it, reads the result
        block. A thread still running at the timeout is asked to stop first."""
        if self.result is not None:
            return self.result
        if not self.on_card:
            self.result = self._fallback()
            return self.result
        c, k = self.thread.cluster, layout()
        ran = self.port.work_wait(c, timeout)["ready"]
        if not ran:
            self._ask_stop()
            self.port.work_wait(c, 1.0)
        raw = self.port.work_read(c, k["KMX_LIM_RESULT"], 256)
        self.port.work_stop(c, 2.0)
        self.running = False
        self.result = parse_result(raw, ran=ran)
        return self.result

    def stop(self):
        """Stops a thread still running (any exit path); the result stays unread."""
        if self.running:
            self._ask_stop()
            try:
                self.port.work_wait(self.thread.cluster, 1.0)
            finally:
                self.running = False
                self.port.work_stop(self.thread.cluster, 2.0)

    def read(self, name):
        """A region's bytes (uint8) after the run: read from the card once the grid
        has stopped, or the host fallback's array."""
        if not self.on_card:
            if self.arrays is None:
                raise RuntimeError("the host fallback has not run: thread_result()")
            return self.arrays[name].view(np.uint8)[: self.thread.region_bytes(name)]
        return self.regions.read(name)

    def grid_stopped(self):
        """Called by the run once its grid has stopped: regions can be read."""
        if self.regions is not None:
            self.regions.readable = True

    def _ask_stop(self):
        k = layout()
        line = struct.pack("<I", k["KMX_STOP_WORD"]) + bytes(60)
        self.port.work_write(self.thread.cluster, k["KMX_LIM_STOP"], line)

    def _fallback(self):
        """Runs host_fallback on host copies of the regions' initial contents."""
        self.arrays = {}
        for name in self.thread.regions:
            n = self.thread.region_bytes(name)
            data = np.zeros(-(-n // 4) * 4, dtype=np.uint8)
            initial = self.thread.initial(name)
            if initial is not None:
                data[:n] = initial
            self.arrays[name] = data.view(np.uint32)
        code, words = self.thread.host_fallback(self.arrays, list(self.thread.words))
        words = [int(w) & abi.M32 for w in words]
        return ThreadResult(True, "done", "none", int(code), words, 0, 0.0, 0.0)
