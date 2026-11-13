"""An X280 thread on a work cluster: its description in a Program, the argument block
and region table the host writes, the result block it reads back, and the host
fallback. Nothing here needs a device: the card is reached through `port` and the
regions through `regions` (see ThreadRun).
"""

import functools
import random
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from komorebi import abi, native

TIMER_HZ = 50_000_000  # the X280's timer: kmx_time() ticks per second
PAGE = 4096  # region granularity, bytes; regions are whole pages of one bank
# The DRAM bank each work cluster reaches through its Memory Port (cluster: bank),
# measured on a p150a (clusters 2 and 3 share bank 7); the start-up checks it.
BANK_OF_CLUSTER = {1: 6, 2: 7, 3: 7}
IPI_ACK_DDR = 0x470020  # the X280 runtime's IPI word, at this address in every bank
RING_DEPTH = 16  # the thread's command lines, in the dispatcher's LIM
RING_EVENTS = 64  # its events, in its own LIM after the kmx block


@functools.cache
def layout():
    """The KMX_* constants of kmx_layout.h."""
    return abi.constants(native.kmx_dir() / "kmx_layout.h")


@dataclass
class Thread:
    """A program's X280 thread: C `sources` linked with kmx, run on `cluster`;
    `regions`: name -> size or initial contents (ids in dict order), then `shared`,
    names of program regions; `words`: argument words; `deadline`: seconds in all;
    `host_fallback(regions, words, run) -> (code, words)`: it on the host, on a Run."""

    sources: list
    regions: dict
    words: list = field(default_factory=list)
    cluster: int = 1
    host_fallback: Callable | None = None
    deadline: float = 10.0
    includes: list = field(default_factory=list)
    defines: dict = field(default_factory=dict)
    prefetch: str = "documented"  # or "keep": the L2 prefetchers as they are
    shared: tuple = ()  # names of Program.shared regions, after its own in id order

    def __post_init__(self):
        k = layout()
        if self.cluster not in BANK_OF_CLUSTER:
            raise ValueError(f"cluster {self.cluster}: work clusters are 1 to 3")
        self.shared = tuple(self.shared)
        n = len(self.regions) + len(self.shared)
        if not self.regions or n > k["KMX_REGIONS_MAX"]:
            raise ValueError(
                f"1 to {k['KMX_REGIONS_MAX']} regions, shared ones included"
            )
        if set(self.shared) & set(self.regions) or len(set(self.shared)) < len(
            self.shared
        ):
            raise ValueError("a shared region's name is the thread's own or repeated")
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

    @property
    def bank(self):
        """The DRAM bank of the thread's cluster: every region it reads lives there."""
        return BANK_OF_CLUSTER[self.cluster]

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
    slots, the kernel's consumed word, the events, the read-count line. `remote`
    (x | y << 16): cmd and read are NoC addresses in that tile."""

    cmd: int
    consumed: int
    events: int
    read: int
    cmd_depth: int
    event_depth: int
    cached: bool = False
    remote: int | None = None


class Peers(NamedTuple):
    """The clusters running one image in a program: `nonce` (this load's), `lead`
    (the lowest), `tiles` and `banks`: {cluster: NoC tile x | y << 16} of each cluster
    and of its DRAM bank."""

    nonce: int
    lead: int
    tiles: dict
    banks: dict

    @property
    def mask(self):
        return sum(1 << c for c in self.tiles)


class Ports(NamedTuple):
    """A program's X280 ports as one thread's block names them: the control block
    address, the grid's cores, each core's tile (x | y << 16, by core index), the
    serving X280 cores ({core: (cluster, cpu)}) and the NoC of the windows (0 or 1)."""

    ctl: int
    cores: int
    tiles: list
    serve: dict
    noc: int = 0


def port_plan(serve, cores, clusters):
    """{core: (cluster, cpu)}: `serve` checked against the grid's `cores` and the
    program's thread `clusters`, or (None) cores 0 to cores - 1 dealt in grid order
    over the four X280 cores of each thread, cluster by cluster, about as many each."""
    if serve is None:
        cpus = [(c, h) for c in sorted(clusters) for h in range(4)]
        if not cpus:
            raise ValueError("X280 ports need an X280 thread")
        return {i: cpus[i * len(cpus) // cores] for i in range(cores)}
    for core, (c, h) in serve.items():
        if not 0 <= core < cores:
            raise ValueError(
                f"port of core {core}: the grid has cores 0 to {cores - 1}"
            )
        if c not in clusters:
            raise ValueError(f"port of core {core}: no thread on cluster {c}")
        if not 0 <= h <= 3:
            raise ValueError(f"port of core {core}: X280 core {h} is not 0 to 3")
    return dict(serve)


def ports_at(ring):
    """LIM offset of the ports area in the thread's block: after the block, a remote
    ring's events and the kmx_par_all lines."""
    k = layout()
    n = k["KMX_LIM_BYTES"] + k["KMX_ALL_BYTES"]
    return n + (ring_area_bytes(ring) if ring is not None and ring.remote else 0)


def port_box(lim_base, ring, core):
    """LIM address of `core`'s mailbox line in its serving thread's block."""
    return lim_base + ports_at(ring) + layout()["KMX_PORTS_BOX"] + 64 * core


def peer_groups(images):
    """{cluster: [clusters running the same image]} from {cluster: image path}."""
    by = {}
    for c, image in sorted(images.items()):
        by.setdefault(str(image), []).append(c)
    return {c: group for group in by.values() for c in group}


def ring_bytes():
    """Bytes of one work ring's command lines and read line in the kernel's LIM."""
    return RING_DEPTH * 64 + 64


def work_ring(kernel_lim, kernel_xy, work_lim, work_xy, cluster=1):
    """The ring between work cluster `cluster` and the dispatcher: commands and
    the read line in the kernel's LIM after its layout, cluster c's after those of
    clusters 1 to c - 1 (each side writes the other's LIM and reads its own). Returns
    the thread's RingPlacement and the kernel's (tile, NoC address of the area)."""
    d, k = abi.headers(), layout()
    # after the layout (the host gives no upload region), one ring per cluster
    cmds = d["KMD_LIM_BYTES"] + (cluster - 1) * ring_bytes()
    area = work_lim + k["KMX_LIM_BYTES"]
    x0, y0 = kernel_xy
    ring = RingPlacement(
        kernel_lim + cmds,
        area + d["KMD_WORK_CONSUMED"],
        area + d["KMD_WORK_EVENTS"],
        kernel_lim + cmds + RING_DEPTH * 64,
        RING_DEPTH,
        RING_EVENTS,
        remote=x0 | y0 << 16,
    )
    x1, y1 = work_xy
    return ring, (x1 | y1 << 16, area)


def kernel_work_args(ends):
    """The kernel's work_* arguments for the rings `ends` ({cluster: (tile, NoC
    address of the area)}, from work_ring); none for an empty dict."""
    if not ends:
        return {}
    n = abi.headers()["KMD_WORK_CLUSTERS"]
    nocs, addrs = [0] * n, [0] * n
    for c, (noc, area) in ends.items():
        nocs[c - 1], addrs[c - 1] = noc, area
    return {
        "work_cmds": abi.headers()["KMD_LIM_BYTES"],
        "work_depth": RING_DEPTH,
        "work_event_depth": RING_EVENTS,
        "work_noc": nocs,
        "work_noc_addr": addrs,
    }


def ring_area_bytes(ring):
    """Bytes of the thread's LIM after the kmx block that the ring's events use."""
    return abi.headers()["KMD_WORK_EVENTS"] + ring.event_depth * 16


def args_block(
    thread,
    base,
    check_value,
    run_value=0,
    ring=None,
    table=None,
    peers=None,
    ports=None,
):
    """The thread's LIM block (KMX_LIM_BYTES, a remote ring's events zeroed): argument
    block, check line at bank address `base`, region table (`table`: [(bank address,
    bytes)], else place() from `base`), the lines of `ring` (RingPlacement), `peers`
    (Peers; kmx_par_all lines after) and `ports` (Ports; area after, see ports_area)."""
    k = layout()
    if table is None:
        table = [(base + p.page * PAGE, p.bytes) for p in place(thread)[0]]
    block = bytearray(k["KMX_LIM_BYTES"])
    flags = k["KMX_FLAG_KEEP_PREFETCH"] if thread.prefetch == "keep" else 0
    head = struct.pack(
        "<QIIQIIQQQI",
        run_value,
        k["KMX_MAGIC"],
        k["KMX_VERSION"],
        int(thread.deadline * TIMER_HZ),
        len(table),
        len(thread.words),
        base,
        check_value,
        flags,
        thread.cluster,
    )
    block[: len(head)] = head
    for i, (address, n) in enumerate(table):
        at = k["KMX_ARGS_REGIONS"] + 16 * i
        block[at : at + 16] = struct.pack("<QQ", address, n)
    words = struct.pack(f"<{len(thread.words)}I", *map(int, thread.words))
    at = k["KMX_LIM_ARGS"] + k["KMX_ARGS_WORDS"]
    block[at : at + len(words)] = words
    if ring is not None:
        flags = k["KMX_RING_CACHED"] if ring.cached else 0
        flags |= k["KMX_RING_REMOTE"] if ring.remote is not None else 0
        tile = ring.remote or 0
        line = struct.pack(
            "<4Q3I5I",
            *ring[:4],
            ring.cmd_depth,
            ring.event_depth,
            flags,
            tile,
            *[0] * 4,
        )
        block[k["KMX_LIM_RING"] : k["KMX_LIM_RING"] + 64] = line
        if ring.remote is not None:
            block += bytes(ring_area_bytes(ring))  # its events start zeroed
    if peers is not None:
        tiles = [peers.tiles.get(c, 0) for c in range(4)]
        banks = [peers.banks.get(c, 0) for c in range(4)]
        line = struct.pack(
            "<16I", peers.nonce, peers.lead, peers.mask, len(block), *tiles, *banks,
            *[0] * 4,
        )  # fmt: skip
        block[k["KMX_LIM_PEERS"] : k["KMX_LIM_PEERS"] + 64] = line
    if ports is not None:
        flags = k["KMX_PORTS_NOC1"] if ports.noc == 1 else 0
        line = struct.pack("<4I48x", ports_at(ring), ports.ctl, ports.cores, flags)
        block[k["KMX_LIM_PORTS"] : k["KMX_LIM_PORTS"] + 64] = line
    return bytes(block)


def ports_area(ports):
    """The ports area (Ports): each core's tile and serving X280 core, the mailbox lines
    zeroed. Written before any thread of the program starts: a core may write its
    mailbox line once the first thread submits a command."""
    k = layout()
    area = bytearray(k["KMX_PORTS_BYTES"])
    tiles = struct.pack(f"<{len(ports.tiles)}I", *ports.tiles)
    area[k["KMX_PORTS_TILES"] : k["KMX_PORTS_TILES"] + len(tiles)] = tiles
    for core, (c, h) in ports.serve.items():
        area[k["KMX_PORTS_OWNER"] + core] = c << 2 | h
    return bytes(area)


def new_check_value():
    """A random nonzero 64-bit value for the check line."""
    return random.getrandbits(64) | 1


class ThreadResult(NamedTuple):
    """The result block: state and library error (names), the thread's code and words,
    its last progress marker and the times in microseconds. `ran` is False when the
    thread did not return by the timeout; `stuck` when it did not return after the stop
    flag either, `detail` then the runtime's error message for a failed stop.
    """

    ran: bool
    state: str
    error: str
    code: int
    words: list
    where: int
    where_us: float
    us: float
    stuck: bool = False
    detail: str = ""


def _names(prefix):
    return {
        v: n[len(prefix) :].lower() for n, v in layout().items() if n.startswith(prefix)
    }


def parse_result(raw, ran=True, stuck=False, detail=""):
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
        stuck=stuck,
        detail=detail,
    )


class ThreadRun:
    """A program's thread in one run: on the card through `port` (the work_* calls
    of the native module), or its host fallback through `run`. `regions` reads regions
    from the card and gives their table; `image`: the thread's image; `base`, the check
    line's bank address, holds `check_value`; `shared`: the fallback's shared arrays."""

    def __init__(
        self,
        thread,
        port,
        regions,
        image=None,
        on_card=True,
        base=0,
        check_value=1,
        run=None,
        shared=None,
    ):
        self.thread = thread
        self.shared = shared if shared is not None else {}
        self.port = port
        self.regions = regions
        self.image = image
        self.on_card = on_card
        self.base = base
        self.check_value = check_value
        self.run = run
        self.running = False
        self.result = None
        self.arrays = None  # the host fallback's regions, uint32, once it ran
        self.ring = None  # RingPlacement of the command ring, when the kernel has one
        self.peers = None  # Peers: the clusters running the same image
        self.ports = None  # Ports: the program's X280 ports, when it declares them

    def write_ports(self):
        """Writes the ports area into the cluster's LIM (Ports; before any start)."""
        if self.on_card and self.ports is not None:
            self.port.work_write(
                self.thread.cluster, ports_at(self.ring), ports_area(self.ports)
            )

    def start(self):
        """Starts the image on the card (no-op for the host fallback)."""
        if not self.on_card:
            if self.thread.host_fallback is None:
                raise ValueError("the thread has no host_fallback")
            return
        table = self.regions.table() if hasattr(self.regions, "table") else None
        block = args_block(
            self.thread,
            self.base,
            self.check_value,
            ring=self.ring,
            table=table,
            peers=self.peers,
            ports=self.ports,
        )
        self.port.work_start(self.thread.cluster, str(self.image), block)
        self.running = True

    def wait(self, timeout):
        """Waits (with a timeout) for the thread to return, stops it, reads the result
        block. A thread still running at the timeout is asked to stop first."""
        if self.result is not None:
            return self.result
        if not self.on_card:
            self.result = self._fallback()
            return self.result
        return self.collect(self.port.work_wait(self.thread.cluster, timeout)["ready"])

    def returned(self, timeout):
        """Whether the thread on the card has returned, waiting at most `timeout`."""
        return self.port.work_wait(self.thread.cluster, timeout)["ready"]

    def collect(self, ran):
        """Asks a thread that has not returned (`ran` false) to stop, then stops it
        through the runtime and reads its result block. One that does not return
        after the stop flag either is `stuck`: a core outside the library."""
        if self.result is not None:
            return self.result
        c, k = self.thread.cluster, layout()
        stuck = False
        if not ran:
            self._ask_stop()
            stuck = not self.port.work_wait(c, 1.0).get("ready")
        raw = self.port.work_read(c, k["KMX_LIM_RESULT"], 256)
        stop = self.port.work_stop(c, 2.0) or {}
        self.running = False
        detail = stop.get("failed", "") if stuck else ""
        self.result = parse_result(raw, ran=ran, stuck=stuck, detail=detail)
        return self.result

    def stop(self, timeout=3.0):
        """Stops a thread still running (any exit path) within `timeout`: up to a third
        of it, at most 1 s, for the thread to return, the rest for the runtime's stop.
        The result stays unread."""
        if self.running:
            by = time.monotonic() + timeout
            self._ask_stop()
            try:
                self.port.work_wait(self.thread.cluster, min(1.0, timeout / 3))
            finally:
                self.running = False
                left = max(by - time.monotonic(), 0.0)
                self.port.work_stop(self.thread.cluster, left)

    def read(self, name):
        """A region's bytes (uint8) after the run: read from the card once the grid
        has stopped, or the host fallback's array."""
        if not self.on_card:
            if self.arrays is None:
                raise RuntimeError("the host fallback has not run: thread_result()")
            if name in self.thread.shared:
                return self.shared[name].view(np.uint8)
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
        for name in self.thread.shared:
            self.arrays[name] = self.shared[name]
        words = list(self.thread.words)
        code, words = self.thread.host_fallback(self.arrays, words, self.run)
        words = [int(w) & abi.M32 for w in words]
        return ThreadResult(True, "done", "none", int(code), words, 0, 0.0, 0.0)
