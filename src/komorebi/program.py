"""A program: its three kernel sources, buffers, start-up arguments, ring depth and
segment numbers. A plain description; nothing here needs a device.
"""

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from komorebi import _env, abi
from komorebi.thread import Thread

KERNELS = ("reader", "compute", "writer")
SEGMENT = re.compile(r"constexpr\s+uint32_t\s+kSeg(\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)u?;")


@dataclass(frozen=True)
class Buffer:
    """A circular buffer on every core: index, data format name, page size, pages."""

    index: int
    format: str
    page_bytes: int
    pages: int


def segments_from_header(path):
    """`constexpr uint32_t kSegName = n;` lines of a header as {"name": n}."""
    found = {}
    for name, value in SEGMENT.findall(Path(path).read_text()):
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        found[snake] = int(value, 0)
    return found


@dataclass
class Program:
    """What the resident worker runs on every core: `kernels` is a directory with
    reader.cpp, compute.cpp and writer.cpp (or a dict of the three), `args(core)` the
    (reader, writer, compute) arguments after the start-up arguments, `segments` a
    dict or a header path."""

    name: str
    kernels: object
    buffers: list = field(default_factory=list)
    args: Callable | None = None
    compile_args: dict = field(default_factory=dict)
    compute: dict = field(default_factory=dict)
    defines: dict = field(default_factory=dict)
    depth: int = 4
    segments: object = field(default_factory=dict)
    threads: dict = field(default_factory=dict)  # work cluster -> Thread
    dyn: "Dynamic | None" = None

    def __post_init__(self):
        if isinstance(self.segments, str | Path):
            self.segments = segments_from_header(self.segments)
        user_max = abi.headers()["KM_SEG_USER_MAX"]
        for name, number in self.segments.items():
            if not 1 <= number <= user_max:
                raise ValueError(f"segment {name} = {number}: not in 1..{user_max}")
        if not 1 <= self.depth <= abi.headers()["KM_RING_SLOTS"]:
            raise ValueError(f"depth {self.depth}: not in 1..KM_RING_SLOTS")
        unknown = set(self.compile_args) - set(KERNELS)
        if unknown:
            raise ValueError(f"compile_args for unknown kernels {sorted(unknown)}")

    def thread(self, cluster=1, **kwargs):
        """Declares the program's X280 thread on work cluster `cluster` (the
        arguments of komorebi.thread.Thread) and returns it. One per program."""
        if self.threads:
            raise ValueError(f"{self.name} already has a thread (one per program)")
        self.threads[cluster] = Thread(cluster=cluster, **kwargs)
        return self.threads[cluster]

    def sources(self):
        """{"reader": path, "compute": path, "writer": path}, each an existing file."""
        if isinstance(self.kernels, dict):
            paths = {k: Path(self.kernels[k]) for k in KERNELS}
        else:
            paths = {k: Path(self.kernels) / f"{k}.cpp" for k in KERNELS}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{self.name}: no kernel source {missing}")
        return paths

    def seg(self, segment):
        """The number of a segment given by name or number."""
        if isinstance(segment, str):
            if segment not in self.segments:
                raise KeyError(f"{self.name} has no segment {segment!r}")
            return self.segments[segment]
        return int(segment)

    def dynamic(self, seg, **options):
        """Makes segment `seg` a dynamic segment (options: those of Dynamic) and adds
        its buffers. Returns the program."""
        if self.dyn is not None:
            raise ValueError(f"{self.name} already has a dynamic segment")
        self.dyn = Dynamic(seg=self.seg(seg), **options)
        self.buffers = [*self.buffers, *self.dyn.buffers()]
        return self

    def core_args(self, core):
        """(reader, writer, compute) program arguments of `core`."""
        if self.args is None:
            return [], [], []
        reader, writer, compute = self.args(core)
        return list(reader), list(writer), list(compute)


@functools.cache
def dyn_header():
    """The constants and the argument order of km_dyn.hpp beside the package."""
    path = _env.include_dir() / "komorebi" / "device" / "km_dyn.hpp"
    fields = abi.struct_fields(path, "DynArgs")
    return abi.constants(path, sizes={"DynArgs": 4 * len(fields)}), fields


@dataclass(frozen=True)
class Dynamic:
    """A dynamic segment: items of `item_bytes` move between the queues of the
    team's cores (`queue_items` each), at most `max_live` alive at once; `bins`
    open bins per writer; a line says idle after `idle_cycles` of idle."""

    seg: int
    item_bytes: int = 64
    max_live: int = 1 << 16
    queue_items: int = 1 << 16
    bins: int = 16
    idle_cycles: int = 20000

    def __post_init__(self):
        k, _ = dyn_header()
        page = k["kDynPageItems"]
        if self.item_bytes % 64 or not 64 <= self.item_bytes <= k["kDynItemMax"]:
            raise ValueError(f"item_bytes {self.item_bytes}: a multiple of 64 to 256")
        q = self.queue_items
        if q < page or q & (q - 1) or q > 1 << 31:
            raise ValueError(f"queue_items {q}: a power of two from {page} to 2**31")
        if not 1 <= self.bins <= k["kDynBinsMax"]:
            raise ValueError(f"bins {self.bins}: not in 1..{k['kDynBinsMax']}")
        if self.max_live < 1 or self.idle_cycles < 1:
            raise ValueError("max_live and idle_cycles must be at least 1")

    def buffers(self):
        """The runtime's buffers of a dynamic segment: packets, outgoing items, bins."""
        k, _ = dyn_header()
        n = k["kDynPacket"]
        return [
            Buffer(k["kCbDynIn"], "uint32", k["kDynHeader"] + n * self.item_bytes, 2),
            Buffer(k["kCbDynOut"], "uint32", self.item_bytes, 2 * n),
            Buffer(k["kCbDynBins"], "uint32", self.bins * n * self.item_bytes, 1),
        ]

    def queue_bytes(self, cores):
        """DRAM bytes of the queues of `cores` cores."""
        return cores * self.queue_items * self.item_bytes

    def check(self, cores, dram_bytes, depth):
        """The launcher's refusals for a team of `cores` cores, `dram_bytes` of DRAM
        given to the queues and a ring depth `depth`; raises ValueError."""
        if self.max_live > self.queue_items:
            raise ValueError(
                f"max_live {self.max_live} > queue_items {self.queue_items}: "
                "every item aimed at one core would not fit its queue"
            )
        if self.queue_bytes(cores) > dram_bytes:
            raise ValueError(
                f"queues of {cores} cores need {self.queue_bytes(cores)} bytes, "
                f"{dram_bytes} given"
            )
        if self.max_live < cores:
            raise ValueError(f"max_live {self.max_live} < {cores} cores: no quota")
        if depth < 2:
            raise ValueError("a dynamic run needs ring depth 2 for its end record")

    def quota(self, cores):
        """Spawns each core starts a run with."""
        return self.max_live // cores

    def args(self, queue, lines):
        """Reader and writer words of km::DynArgs (they precede the program's)."""
        _, fields = dyn_header()
        values = {
            "seg": self.seg,
            "queue": queue,
            "cap": self.queue_items,
            "item_bytes": self.item_bytes,
            "lines": lines,
            "idle": self.idle_cycles,
            "bins": self.bins,
        }
        return [int(values[f]) & abi.M32 for f in fields]

    def compute_args(self, cores):
        """Compute's words (item bytes, quota), before the program's own."""
        return [self.item_bytes, self.quota(cores)]
