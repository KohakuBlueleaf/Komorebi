"""A program: its three kernel sources, buffers, start-up arguments, ring depth and
segment numbers. A plain description; nothing here needs a device.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from komorebi import abi

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

    def core_args(self, core):
        """(reader, writer, compute) program arguments of `core`."""
        if self.args is None:
            return [], [], []
        reader, writer, compute = self.args(core)
        return list(reader), list(writer), list(compute)
