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

from komorebi import abi
from komorebi.device import native

TIMER_HZ = 50_000_000  # the X280's timer: kmx_time() ticks per second
PAGE = 4096  # region granularity, bytes; regions are whole pages of one bank
# The DRAM bank each work cluster reaches through its Memory Port (cluster: bank),
# measured on a p150a (clusters 2 and 3 share bank 7); the start-up checks it.
BANK_OF_CLUSTER = {1: 6, 2: 7, 3: 7}
IPI_ACK_DDR = 0x470020  # the X280 runtime's IPI word, at this address in every bank
RING_DEPTH = 16  # the thread's command lines, in the dispatcher's LIM
UPLOAD_BYTES = 512 * 1024  # the dispatcher's upload region, before the work rings
RING_EVENTS = 64  # its events, in its own LIM after the kmx block


@functools.cache
def layout():
    """The KMX_* constants of kmx_layout.h."""
    return abi.constants(native.kmx_directory() / "kmx_layout.h")


def camel(name):
    """big_step -> BigStep."""
    return "".join(part.capitalize() for part in name.split("_"))


def program_defines(programs):
    """For a thread in a run of several programs: kProgramSum=1u, the program field of
    program sum, and kProgramSumHandlerAdd=2u for its handler add."""
    out = {}
    for program in programs:
        prefix = "kProgram" + camel(program.name)
        named = {prefix: program.program_field}
        for handler, number in program.handlers.items():
            named[f"{prefix}Handler{camel(handler)}"] = number
        for name, value in named.items():
            if not name.isidentifier() or name in out:
                raise ValueError(
                    f"{program.name}: {name!r} is not a C name or is given twice"
                )
            out[name] = f"{value}u"
    return out


@dataclass
class Thread:
    """A program's X280 thread: C `sources` linked with kmx, run on `cluster`;
    `regions`: name -> size or initial contents (ids in dict order), then `shared`
    (program regions); `words`: argument words; `deadline`: seconds for kmx_main;
    `host_fallback(regions, words, run)`; `handlers`: kmx_program's, as Program's."""

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
    handlers: object = field(default_factory=dict)  # name -> number for kmx_program

    def __post_init__(self):
        constants = layout()
        if isinstance(self.handlers, str | Path):
            self.handlers = abi.handlers_from_header(self.handlers)
        user_max = abi.headers()["KM_HANDLER_USER_MAX"]
        for name, number in self.handlers.items():
            if not 1 <= number <= user_max:
                raise ValueError(f"handler {name} = {number}: not in 1..{user_max}")
        if self.cluster not in BANK_OF_CLUSTER:
            raise ValueError(f"cluster {self.cluster}: work clusters are 1 to 3")
        self.shared = tuple(self.shared)
        region_count = len(self.regions) + len(self.shared)
        if not self.regions or region_count > constants["KMX_REGIONS_MAX"]:
            raise ValueError(
                f"1 to {constants['KMX_REGIONS_MAX']} regions, shared ones included"
            )
        if set(self.shared) & set(self.regions) or len(set(self.shared)) < len(
            self.shared
        ):
            raise ValueError("a shared region's name is the thread's own or repeated")
        if len(self.words) > constants["KMX_WORDS_MAX"]:
            raise ValueError(f"at most {constants['KMX_WORDS_MAX']} words")
        if any(not 0 <= int(word) <= abi.M32 for word in self.words):
            raise ValueError("words are 32-bit unsigned")
        ticks = self.deadline * TIMER_HZ
        if not 0 < ticks <= constants["KMX_DEADLINE_MAX"]:
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

    def handler(self, handler):
        """The number of one of kmx_program's handlers, given by name or number."""
        if isinstance(handler, str):
            if handler not in self.handlers:
                raise KeyError(
                    f"the thread on cluster {self.cluster} has no handler "
                    f"{handler!r}"
                )
            return self.handlers[handler]
        return int(handler)

    def handler_defines(self):
        """The handlers as the thread's C sees them: kHandlerBigStep=7u for big_step."""
        return {
            "kHandler"
            + "".join(part.capitalize() for part in name.split("_")): (f"{number}u")
            for name, number in self.handlers.items()
        }

    def all_defines(self, programs=()):
        """Every define of the thread's C: its handlers, its `defines` and, in a run of
        several programs, program_defines of them; a name given twice raises."""
        out = dict(self.handler_defines())
        for name, value in [*self.defines.items(), *program_defines(programs).items()]:
            if name in out:
                raise ValueError(f"the thread's define {name} is given twice")
            out[name] = value
        return out

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
        paths = [Path(source) for source in self.sources]
        missing = [str(path) for path in paths if not path.is_file()]
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
    out, page = [], 1
    for name in thread.regions:
        size = thread.region_bytes(name)
        pages = -(-size // PAGE)
        out.append(Placement(name, page, pages, size))
        page += pages
    return out, page


class RingPlacement(NamedTuple):
    """Where the thread's command ring lives, as its cluster addresses it: command
    slots, the kernel's consumed word, the events, the read-count line. `remote`
    (x | y << 16): cmd and read are NoC addresses in that tile."""

    cmd: int
    consumed: int
    events: int
    read: int
    command_depth: int
    event_depth: int
    cached: bool = False
    remote: int | None = None


def ring_line(ring, buffer, records):
    """The ring's 64-byte line of the LIM block (kmx_ring_descriptor): its addresses,
    depths, flags, the copy buffer and load_records()'s (record 0, step, programs)."""
    constants = layout()
    flags = constants["KMX_RING_CACHED"] if ring.cached else 0
    flags |= constants["KMX_RING_REMOTE"] if ring.remote is not None else 0
    return struct.pack(
        "<4Q3I5I",
        *ring[:4],
        ring.command_depth,
        ring.event_depth,
        flags,
        ring.remote or 0,
        buffer,
        *(records or (0, 0, 0)),
    )


def load_records(switch):
    """(record of program field 0, step, programs) for kmx_load, from a Run's switch
    ({"records": {field: DRAM address}, "refused": why}); None: no switch."""
    if not switch or switch["refused"] or not switch["records"]:
        return None
    records = switch["records"]
    if sorted(records) != list(range(len(records))):
        raise ValueError(f"program records {records}: not one step apart")
    rows = {**records, **{f: a for f, (a, _) in switch.get("spare", {}).items()}}
    step = rows[1] - rows[0] if len(rows) > 1 else 0
    if any(address != records[0] + field * step for field, address in records.items()):
        raise ValueError(f"program records {records}: not one step apart")
    return records[0], step, len(records)


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
        return sum(1 << cluster for cluster in self.tiles)


class Ports(NamedTuple):
    """A program's X280 ports as one thread's block names them: the control block
    address, the grid's cores, each core's tile (x | y << 16, by core index), the
    serving X280 cores ({core: (cluster, x280_core)}) and the NoC of the windows (0 or
    1)."""

    control: int
    cores: int
    tiles: list
    serve: dict
    noc: int = 0


class Ranks(NamedTuple):
    """A dynamic program's X280 ranks as their thread's block names them: the
    kmx_ranks_line words by field name (`at` set by ranks_at) and each core's tile."""

    words: dict
    tiles: list


def port_plan(serve, cores, clusters):
    """{core: (cluster, x280_core)}: `serve` checked against the grid's `cores` and the
    program's thread `clusters`, or (None) cores 0 to cores - 1 dealt in grid order
    over the four X280 cores of each thread, cluster by cluster, about as many each."""
    if serve is None:
        x280_cores = [
            (cluster, x280_core)
            for cluster in sorted(clusters)
            for x280_core in range(4)
        ]
        if not x280_cores:
            raise ValueError("X280 ports need an X280 thread")
        return {i: x280_cores[i * len(x280_cores) // cores] for i in range(cores)}
    for core, (cluster, x280_core) in serve.items():
        if not 0 <= core < cores:
            raise ValueError(
                f"port of core {core}: the grid has cores 0 to {cores - 1}"
            )
        if cluster not in clusters:
            raise ValueError(f"port of core {core}: no thread on cluster {cluster}")
        if not 0 <= x280_core <= 3:
            raise ValueError(
                f"port of core {core}: X280 core {x280_core} is not 0 to 3"
            )
    return dict(serve)


def ports_at(ring):
    """LIM offset of the ports area in the thread's block: after the block, a remote
    ring's events and the kmx_parallel_all lines."""
    constants = layout()
    bytes = constants["KMX_LIM_BYTES"] + constants["KMX_ALL_BYTES"]
    return bytes + (ring_area_bytes(ring) if ring is not None and ring.remote else 0)


def ranks_at(ring, ports):
    """LIM offset of the X280 ranks' area in the thread's block: after the ports area,
    or where it would be."""
    after = layout()["KMX_PORTS_BYTES"] if ports is not None else 0
    return ports_at(ring) + after


def port_box(lim_base, ring, core):
    """LIM address of `core`'s mailbox line in its serving thread's block."""
    return lim_base + ports_at(ring) + layout()["KMX_PORTS_BOX"] + 64 * core


def peer_groups(images):
    """{cluster: [clusters running the same image]} from {cluster: image path}."""
    by_image = {}
    for cluster, image in sorted(images.items()):
        by_image.setdefault(str(image), []).append(cluster)
    return {cluster: group for group in by_image.values() for cluster in group}


def ring_bytes():
    """Bytes of one work ring's command lines and read line in the kernel's LIM."""
    return RING_DEPTH * 64 + 64


def rings_at():
    """Offset in the dispatcher's LIM of the work rings: after its upload region."""
    return abi.headers()["KMD_LIM_BYTES"] + UPLOAD_BYTES


def work_ring(kernel_lim, kernel_coordinates, work_lim, work_coordinates, cluster=1):
    """The ring between work cluster `cluster` and the dispatcher: commands and
    the read line in the kernel's LIM after its layout, cluster c's after those of
    clusters 1 to c - 1 (each side writes the other's LIM and reads its own). Returns
    the thread's RingPlacement and the kernel's (tile, NoC address of the area)."""
    abi_headers, constants = abi.headers(), layout()
    # after the layout and the upload region (streams, graphs), one ring per cluster
    command_lines = rings_at() + (cluster - 1) * ring_bytes()
    area = work_lim + constants["KMX_LIM_BYTES"]
    kernel_x, kernel_y = kernel_coordinates
    ring = RingPlacement(
        kernel_lim + command_lines,
        area + abi_headers["KMD_WORK_CONSUMED"],
        area + abi_headers["KMD_WORK_EVENTS"],
        kernel_lim + command_lines + RING_DEPTH * 64,
        RING_DEPTH,
        RING_EVENTS,
        remote=kernel_x | kernel_y << 16,
    )
    work_x, work_y = work_coordinates
    return ring, (work_x | work_y << 16, area)


def kernel_work_args(ends, beside=False):
    """The kernel's work_* arguments for the rings `ends` ({cluster: (tile, NoC
    address of the area)}, from work_ring); none for an empty dict, unless `beside`:
    another island's thread, so the same layout with no ring in use (address 0)."""
    if not ends and not beside:
        return {}
    count = abi.headers()["KMD_WORK_CLUSTERS"]
    nocs, addrs = [0] * count, [0] * count
    for cluster, (noc, area) in ends.items():
        nocs[cluster - 1], addrs[cluster - 1] = noc, area
    return {
        "work_cmds": rings_at(),
        "work_depth": RING_DEPTH,
        "work_event_depth": RING_EVENTS,
        "work_noc": nocs,
        "work_noc_addr": addrs,
    }


def ring_area_bytes(ring):
    """Bytes of the thread's LIM after the kmx block that the ring uses: the counts, the
    events, the command lines the dispatcher hands the thread (KMD_WORK_IN), run line.
    """
    headers = abi.headers()
    events = -(-ring.event_depth * 16 // 64) * 64
    lines = headers["KMD_WORK_IN_DEPTH"] * 64 + headers["KMD_WORK_RUN_BYTES"]
    return headers["KMD_WORK_EVENTS"] + events + lines


def args_block(
    thread,
    base,
    check_value,
    run_value=0,
    ring=None,
    table=None,
    peers=None,
    ports=None,
    records=None,
    buffer=0,
    ranks=None,
    island=0,
):
    """The thread's LIM block (KMX_LIM_BYTES, a remote ring's events zeroed): argument
    block, check line, region table (`table`, else place()), the lines of `ring`
    (RingPlacement, with `records` and `buffer`), `peers`, `ports` and `ranks` (their
    areas after the block); `island` its island | islands << 16 (0: island 0 of one)."""
    constants = layout()
    if table is None:
        table = [
            (base + placement.page * PAGE, placement.bytes)
            for placement in place(thread)[0]
        ]
    block = bytearray(constants["KMX_LIM_BYTES"])
    flags = constants["KMX_FLAG_KEEP_PREFETCH"] if thread.prefetch == "keep" else 0
    head = struct.pack(
        "<QIIQIIQQQII",
        run_value,
        constants["KMX_MAGIC"],
        constants["KMX_VERSION"],
        int(thread.deadline * TIMER_HZ),
        len(table),
        len(thread.words),
        base,
        check_value,
        flags,
        thread.cluster,
        island,
    )
    block[: len(head)] = head
    for i, (address, size) in enumerate(table):
        offset = constants["KMX_ARGS_REGIONS"] + 16 * i
        block[offset : offset + 16] = struct.pack("<QQ", address, size)
    words = struct.pack(f"<{len(thread.words)}I", *map(int, thread.words))
    offset = constants["KMX_LIM_ARGS"] + constants["KMX_ARGS_WORDS"]
    block[offset : offset + len(words)] = words
    if ring is not None:
        line = ring_line(ring, buffer, records)
        block[constants["KMX_LIM_RING"] : constants["KMX_LIM_RING"] + 64] = line
        if ring.remote is not None:
            block += bytes(ring_area_bytes(ring))  # its events start zeroed
    if peers is not None:
        tiles = [peers.tiles.get(cluster, 0) for cluster in range(4)]
        banks = [peers.banks.get(cluster, 0) for cluster in range(4)]
        line = struct.pack(
            "<16I", peers.nonce, peers.lead, peers.mask, len(block), *tiles, *banks,
            *[0] * 4,
        )  # fmt: skip
        block[constants["KMX_LIM_PEERS"] : constants["KMX_LIM_PEERS"] + 64] = line
    if ports is not None:
        flags = constants["KMX_PORTS_NOC1"] if ports.noc == 1 else 0
        line = struct.pack("<4I48x", ports_at(ring), ports.control, ports.cores, flags)
        block[constants["KMX_LIM_PORTS"] : constants["KMX_LIM_PORTS"] + 64] = line
    if ranks is not None:
        fields = abi.struct_fields(
            native.kmx_directory() / "kmx_layout.h", "kmx_ranks_line"
        )
        words = ranks.words | {"at": ranks_at(ring, ports)}
        line = struct.pack("<16I", *(int(words.get(name, 0)) for name in fields))
        block[constants["KMX_LIM_RANKS"] : constants["KMX_LIM_RANKS"] + 64] = line
    return bytes(block)


def ranks_area(ranks):
    """The X280 ranks' area (Ranks): each core's tile, then the credit lines and the
    inbox places zeroed."""
    constants = layout()
    area = bytearray(constants["KMX_RANKS_BYTES"])
    tiles = struct.pack(f"<{len(ranks.tiles)}I", *ranks.tiles)
    at = constants["KMX_RANKS_TILES"]
    area[at : at + len(tiles)] = tiles
    return bytes(area)


def ports_area(ports):
    """The ports area (Ports): each core's tile and serving X280 core, the mailbox lines
    zeroed. Written before any thread of the program starts: a core may write its
    mailbox line once the first thread submits a command."""
    constants = layout()
    area = bytearray(constants["KMX_PORTS_BYTES"])
    tiles = struct.pack(f"<{len(ports.tiles)}I", *ports.tiles)
    area[constants["KMX_PORTS_TILES"] : constants["KMX_PORTS_TILES"] + len(tiles)] = (
        tiles
    )
    for core, (cluster, x280_core) in ports.serve.items():
        area[constants["KMX_PORTS_OWNER"] + core] = cluster << 2 | x280_core
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
    where_microseconds: float
    microseconds: float
    stuck: bool = False
    detail: str = ""


def _names(prefix):
    return {
        value: name[len(prefix) :].lower()
        for name, value in layout().items()
        if name.startswith(prefix)
    }


def parse_result(raw, ran=True, stuck=False, detail=""):
    """A ThreadResult from the 256 bytes of the result block."""
    constants = layout()
    block = struct.unpack("<64I", bytes(raw)[:256])
    word_count = min(block[3], constants["KMX_RESULT_WORDS_MAX"])
    return ThreadResult(
        ran=ran,
        state=_names("KMX_STATE_").get(block[0], f"unknown {block[0]}"),
        error=_names("KMX_ERROR_").get(block[1], f"unknown {block[1]}"),
        code=block[2],
        words=list(block[16 : 16 + word_count]),
        where=block[4],
        where_microseconds=block[5] / 50.0,
        microseconds=block[6] / 50.0,
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
        self.ranks = None  # Ranks: the dynamic program's X280 ranks on this cluster
        self.records = None  # load_records(): the programs kmx_load can load
        self.buffer = 0  # the cores' L1 for kmx_fetch and kmx_store (0: none)
        self.island = 0  # its island | islands << 16 (0: island 0 of one)

    def write_ports(self):
        """Writes the ports area into the cluster's LIM (Ports; before any start)."""
        if self.on_card and self.ports is not None:
            self.port.work_write(
                self.thread.cluster, ports_at(self.ring), ports_area(self.ports)
            )

    def write_records(self, switch):
        """Raises the programs kmx_load can load to the run's (`switch`, after an
        added class's record is in its row): the ring's line written to the live LIM."""
        if self.records is None or self.ring is None:
            return
        self.records = load_records(switch)
        if self.on_card and self.running:
            line = ring_line(self.ring, self.buffer, self.records)
            self.port.work_write(self.thread.cluster, layout()["KMX_LIM_RING"], line)

    def write_ranks(self):
        """Writes the X280 ranks' area into the cluster's LIM (Ranks; before any
        start)."""
        if self.on_card and self.ranks is not None:
            self.port.work_write(
                self.thread.cluster,
                ranks_at(self.ring, self.ports),
                ranks_area(self.ranks),
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
            records=self.records,
            buffer=self.buffer,
            ranks=self.ranks,
            island=self.island,
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
        cluster, constants = self.thread.cluster, layout()
        stuck = False
        if not ran:
            self._ask_stop()
            stuck = not self.port.work_wait(cluster, 1.0).get("ready")
        raw = self.port.work_read(cluster, constants["KMX_LIM_RESULT"], 256)
        stop = self.port.work_stop(cluster, 2.0) or {}
        self.running = False
        detail = stop.get("failed", "") if stuck else ""
        self.result = parse_result(raw, ran=ran, stuck=stuck, detail=detail)
        return self.result

    def stop(self, timeout=3.0):
        """Stops a thread still running (any exit path) within `timeout`: up to a third
        of it, at most 1 s, for the thread to return, the rest for the runtime's stop.
        The result stays unread."""
        if self.running:
            deadline = time.monotonic() + timeout
            self._ask_stop()
            try:
                self.port.work_wait(self.thread.cluster, min(1.0, timeout / 3))
            finally:
                self.running = False
                left = max(deadline - time.monotonic(), 0.0)
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
        constants = layout()
        line = struct.pack("<I", constants["KMX_STOP_WORD"]) + bytes(60)
        self.port.work_write(self.thread.cluster, constants["KMX_LIM_STOP"], line)

    def _fallback(self):
        """Runs host_fallback on host copies of the regions' initial contents."""
        self.arrays = {}
        for name in self.thread.regions:
            size = self.thread.region_bytes(name)
            data = np.zeros(-(-size // 4) * 4, dtype=np.uint8)
            initial = self.thread.initial(name)
            if initial is not None:
                data[:size] = initial
            self.arrays[name] = data.view(np.uint32)
        for name in self.thread.shared:
            self.arrays[name] = self.shared[name]
        words = list(self.thread.words)
        code, words = self.thread.host_fallback(self.arrays, words, self.run)
        words = [int(word) & abi.M32 for word in words]
        return ThreadResult(True, "done", "none", int(code), words, 0, 0.0, 0.0)
