"""A program: its kernel sources (three, or one movement source for both data movers
and a compute source), buffers, start-up arguments, ring depth and handler numbers. A
plain description; nothing here needs a device.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from komorebi import _env, abi
from komorebi.abi import handlers_from_header
from komorebi.errors import ProgramError
from komorebi.movement import check_pairs
from komorebi.overlay import Slots, declare, overlays_from_header
from komorebi.thread import Thread

KERNELS = ("reader", "compute", "writer")
# The movement form: one source built for both data-movement CPUs, and compute.
MOVEMENT_KERNELS = ("movement", "compute")
# The runtime's KernelBuildOptLevel names; a kernel without one keeps the runtime's
# default (O2 for the reader and writer, O3 for the compute kernel).
OPT_LEVELS = ("O0", "O1", "O2", "O3", "Os", "Ofast", "Oz")


@dataclass(frozen=True)
class KernelSizes:
    """A program's build against the kernel buffer: packed text per thread (bytes),
    the kernel config's total (text, arguments, buffer configs) and the buffer."""

    text: dict
    total: int | None
    limit: int


@dataclass(frozen=True)
class Buffer:
    """A circular buffer on every core: index, data format name, page size, pages."""

    index: int
    format: str
    page_bytes: int
    pages: int


@dataclass(frozen=True)
class BufferPair:
    """A logical buffer as two circular buffers, mover A's half (`mover_a`) and mover
    B's (`mover_b`), with one data format and page size; `pages` is both halves' count
    or a (mover A, mover B) pair of counts."""

    mover_a: int
    mover_b: int
    format: str
    page_bytes: int
    pages: object

    def __post_init__(self):
        if self.mover_a == self.mover_b:
            raise ValueError(f"pair ({self.mover_a}, {self.mover_b}): two halves")
        counts = self.counts()
        if len(counts) != 2 or min(counts) < 1:
            raise ValueError(f"pair pages {self.pages!r}: a count or two, each >= 1")

    def counts(self):
        """(mover A's pages, mover B's pages)."""
        if isinstance(self.pages, int):
            return (self.pages, self.pages)
        return tuple(int(count) for count in self.pages)

    def halves(self):
        """The two circular buffers: mover A's, then mover B's."""
        pages_a, pages_b = self.counts()
        return [
            Buffer(self.mover_a, self.format, self.page_bytes, pages_a),
            Buffer(self.mover_b, self.format, self.page_bytes, pages_b),
        ]


@dataclass
class Program:
    """What the persistent kernel runs on every core: `kernels` a directory with
    reader.cpp, compute.cpp and writer.cpp, or movement.cpp and compute.cpp (or a dict
    of them), `args(core)` the (reader, writer, compute) arguments after the start-up
    arguments, `handlers` a dict or a header path, `opt` a level per kernel."""

    name: str
    kernels: object
    buffers: list = field(default_factory=list)
    args: Callable | None = None
    compile_args: dict = field(default_factory=dict)
    compute: dict = field(default_factory=dict)
    defines: dict = field(default_factory=dict)
    depth: int = 4
    handlers: object = field(default_factory=dict)
    threads: dict = field(default_factory=dict)  # work cluster -> Thread
    shared: dict = field(default_factory=dict)  # shared region name -> size or array
    messages: "Dynamic | None" = None
    opt: dict = field(default_factory=dict)  # kernel -> optimisation level
    x280_ports: dict | None = None  # Program.ports(): {"serve": ..., "noc": ...}
    sizes: KernelSizes | None = field(default=None, init=False, compare=False)
    pairs: list = field(default_factory=list, init=False)  # the BufferPairs given
    overlays: dict = field(default_factory=dict, init=False)  # name -> Overlay
    overlay_slots: Slots = field(default_factory=Slots, init=False)
    header: Path | None = field(default=None, init=False)  # the handlers' header
    # the overlays as the last load built them (komorebi.overlay.Built)
    overlays_built: object = field(default=None, init=False, compare=False)

    def __post_init__(self):
        if isinstance(self.handlers, str | Path):
            self.header = Path(self.handlers)
            self.handlers = handlers_from_header(self.handlers)
        user_max = abi.headers()["KM_HANDLER_USER_MAX"]
        for name, number in self.handlers.items():
            if not 1 <= number <= user_max:
                raise ValueError(f"handler {name} = {number}: not in 1..{user_max}")
        if not 1 <= self.depth <= abi.headers()["KM_RING_SLOTS"]:
            raise ValueError(f"depth {self.depth}: not in 1..KM_RING_SLOTS")
        self.pairs = [item for item in self.buffers if isinstance(item, BufferPair)]
        self.buffers = [
            half
            for item in self.buffers
            for half in (item.halves() if isinstance(item, BufferPair) else [item])
        ]
        indices = [buffer.index for buffer in self.buffers]
        for pair in self.pairs:
            for half in (pair.mover_a, pair.mover_b):
                if indices.count(half) > 1:
                    raise ProgramError(
                        f"{self.name}: buffer {half}, a half of a pair, is declared "
                        "twice"
                    )
        names = (*KERNELS, "movement")
        unknown = set(self.compile_args) - set(names)
        if unknown:
            raise ValueError(f"compile_args for unknown kernels {sorted(unknown)}")
        for kernel, level in self.opt.items():
            if kernel not in names:
                raise ProgramError(f"opt for unknown kernel {kernel!r}: not in {names}")
            if level not in OPT_LEVELS:
                raise ProgramError(f"opt {kernel}={level!r}: not one of {OPT_LEVELS}")
        for given in (self.compile_args, self.opt):
            if "movement" in given and {"reader", "writer"} & set(given):
                raise ProgramError(
                    f"{self.name}: 'movement' stands for the reader and the writer; "
                    "give it or them, not both"
                )

    def thread(self, cluster=1, clusters=None, **kwargs):
        """Declares an X280 thread on work cluster `cluster` (the arguments of
        komorebi.thread.Thread) and returns it; with `clusters`, the same thread on
        each of them (a list, one Thread per cluster). One thread per cluster."""
        single = clusters is None
        clusters = (cluster,) if single else tuple(clusters)
        if len(set(clusters)) < len(clusters):
            raise ValueError(f"clusters {clusters}: each at most once")
        for number in clusters:
            if number in self.threads:
                raise ValueError(
                    f"{self.name} already has a thread on cluster {number}"
                )
        made = [Thread(cluster=number, **kwargs) for number in clusters]
        for declared in made:
            self.threads[declared.cluster] = declared
        return made[0] if single else made

    def ports(self, serve=None, noc=0):
        """Declares X280 ports: each served Tensix core gets one X280 core of the
        threads for the run. `serve`: {core: (cluster, x280_core)}, or None for every
        core dealt over the threads' X280 cores in grid order. `noc`: the NoC of the
        X280 cores' stores into the Tensix cores (0 or 1)."""
        if self.x280_ports is not None:
            raise ValueError(f"{self.name} already declares its X280 ports")
        if noc not in (0, 1):
            raise ValueError(f"noc={noc!r}: 0 or 1")
        if serve is not None:
            serve = {
                int(core): (int(cluster), int(x280_core))
                for core, (cluster, x280_core) in serve.items()
            }
            if not serve:
                raise ValueError("serve names no core")
        self.x280_ports = {"serve": serve, "noc": noc}

    def region(self, name, value):
        """Declares a shared region: `value` a size in bytes or an array of initial
        contents. Threads that name it in `shared` see it; all must share its bank."""
        if name in self.shared:
            raise ValueError(f"{self.name} already has a shared region {name!r}")
        size = int(value) if isinstance(value, int) else np.asarray(value).nbytes
        if size <= 0:
            raise ValueError(f"shared region {name!r} is empty")
        self.shared[name] = value

    def check_threads(self):
        """Every shared region a thread names exists, is named by some thread, and all
        threads naming it read one bank (a thread reads only its own bank's regions)."""
        banks = {}
        for thread in self.threads.values():
            for name in thread.shared:
                if name not in self.shared:
                    raise ValueError(f"{self.name}: no shared region {name!r}")
                banks.setdefault(name, set()).add(thread.bank)
        for name in self.shared:
            if name not in banks:
                raise ValueError(f"{self.name}: no thread names shared region {name!r}")
            if len(banks[name]) > 1:
                raise ValueError(
                    f"{self.name}: shared region {name!r} is named by threads on banks "
                    f"{sorted(banks[name])}; a thread reads only its own bank's regions"
                )

    def overlay(self, name, parts, number=None, defines=None):
        """Declares overlay `name`: `parts` a folder with reader.cpp, writer.cpp or
        compute.cpp (any of them) or a dict of them; `number` its number, by default
        kOverlay<Name> of the handlers' header. Returns the program."""
        if name in self.overlays:
            raise ValueError(f"{self.name} already has an overlay {name!r}")
        if number is None:
            numbers = overlays_from_header(self.header) if self.header else {}
            if name not in numbers:
                raise ValueError(
                    f"overlay {name}: no number given and no kOverlay constant for it "
                    "in the handlers' header"
                )
            number = numbers[name]
        made = declare(name, parts, number, defines)
        taken = {o.number: o.name for o in self.overlays.values()}
        if made.number in taken:
            raise ValueError(
                f"overlay {name}: number {made.number} is taken by "
                f"{taken[made.number]}"
            )
        self.overlays[name] = made
        return self

    def slots(self, count=None, bytes=None):
        """Declares the slots of the kernels that have overlay parts: `count` (1 to 4)
        and `bytes` per slot, each an int or a {"reader"|"writer"|"compute": value};
        by default a slot per overlay with a part there, sized by the largest part."""
        self.overlay_slots = Slots(count, bytes)
        return self

    def movement(self):
        """True when one movement source serves both data-movement CPUs: a dict with
        "movement" and "compute", or a directory with movement.cpp and no reader.cpp."""
        if isinstance(self.kernels, dict):
            if "movement" in self.kernels and {"reader", "writer"} & set(self.kernels):
                raise ProgramError(
                    f"{self.name}: kernels name a movement source and reader or writer"
                )
            return "movement" in self.kernels
        folder = Path(self.kernels)
        if (folder / "movement.cpp").is_file():
            if (folder / "reader.cpp").is_file() or (folder / "writer.cpp").is_file():
                raise ProgramError(
                    f"{self.name}: {folder} holds movement.cpp and reader.cpp or "
                    "writer.cpp; keep one form"
                )
            return True
        return False

    def sources(self):
        """{"reader": path, "compute": path, "writer": path}, each an existing file (a
        movement program: its movement source for reader and writer); checks the pairs.
        """
        movement = self.movement()
        names = MOVEMENT_KERNELS if movement else KERNELS
        if isinstance(self.kernels, dict):
            given = {kernel: Path(self.kernels[kernel]) for kernel in names}
        else:
            given = {kernel: Path(self.kernels) / f"{kernel}.cpp" for kernel in names}
        missing = [str(path) for path in given.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{self.name}: no kernel source {missing}")
        if movement:
            paths = {
                "reader": given["movement"],
                "compute": given["compute"],
                "writer": given["movement"],
            }
        else:
            paths = given
        if self.pairs:
            check_pairs(self.name, self.pairs, paths)
        return paths

    def kernel_compile_args(self, kernel):
        """The compile-time arguments of "reader", "writer" or "compute"; a "movement"
        entry serves both data-movement kernels."""
        if kernel in self.compile_args:
            return list(self.compile_args[kernel])
        if kernel == "compute":
            return []
        return list(self.compile_args.get("movement", []))

    def kernel_opt(self, kernel):
        """The optimisation level of "reader", "writer" or "compute", or None."""
        if kernel in self.opt or kernel == "compute":
            return self.opt.get(kernel)
        return self.opt.get("movement")

    def handler(self, handler):
        """The number of a handler given by name or number."""
        if isinstance(handler, str):
            if handler not in self.handlers:
                raise KeyError(f"{self.name} has no handler {handler!r}")
            return self.handlers[handler]
        return int(handler)

    def dynamic(self, handler, *handlers, **options):
        """Makes `handler`, and each of `handlers`, a dynamic handler of one message
        loop (options: those of Dynamic) and adds its buffers. Returns the program."""
        if self.messages is not None:
            raise ValueError(f"{self.name} already has a dynamic handler")
        numbers = tuple(self.handler(each) for each in (handler, *handlers))
        self.messages = Dynamic(handler=numbers[0], handlers=numbers, **options)
        self.buffers = [*self.buffers, *self.messages.buffers()]
        return self

    def core_args(self, core):
        """(reader, writer, compute) program arguments of `core`."""
        if self.args is None:
            return [], [], []
        reader, writer, compute = self.args(core)
        return list(reader), list(writer), list(compute)


@functools.cache
def messages_header():
    """The constants and the argument order of km_messages.hpp beside the package."""
    path = _env.include_directory() / "komorebi" / "device" / "km_messages.hpp"
    fields = abi.struct_fields(path, "MessagesArgs")
    return abi.constants(path, sizes={"MessagesArgs": 4 * len(fields)}), fields


def message_word(position, queue_messages, handler, program=0):
    """Word 0 of the message at queue position `position`, as the writer sets it: the
    lap mark (1 to 65535, round again) in the low 16 bits, handler and program above."""
    constants, _ = messages_header()
    lap = position // queue_messages
    mark = lap % constants["kMessagesLaps"] + 1
    return (
        mark
        | (handler & 0xFF) << constants["kMessagesHandlerShift"]
        | (program & 0xFF) << constants["kMessagesProgramShift"]
    )


@dataclass(frozen=True)
class Dynamic:
    """Dynamic `handlers` (default: `handler` alone): messages of `message_bytes` move
    between the group's queues (`queue_messages` each), at most `max_live` alive; `bins`
    open bins per writer; a line says idle after `idle_cycles`. The last `counters`
    cores of a run's group are counter tiles (layout.counter_tiles)."""

    handler: int
    message_bytes: int = 64
    max_live: int = 1 << 16
    queue_messages: int = 1 << 16
    bins: int = 16
    idle_cycles: int = 20000
    counters: int = 2
    handlers: tuple = ()

    def __post_init__(self):
        constants, _ = messages_header()
        handlers = tuple(int(number) for number in self.handlers) or (self.handler,)
        object.__setattr__(self, "handlers", handlers)
        user_max = abi.headers()["KM_HANDLER_USER_MAX"]
        most = abi.headers()["KM_DYNAMIC_MAX"]
        if self.handler not in handlers or len(set(handlers)) < len(handlers):
            raise ValueError(
                f"dynamic handlers {handlers}: distinct, with {self.handler}"
            )
        if len(handlers) > most or not all(1 <= h <= user_max for h in handlers):
            raise ValueError(
                f"dynamic handlers {handlers}: 1 to {most}, each in 1..{user_max}"
            )
        page = constants["kMessagesPageMessages"]
        if (
            self.message_bytes % 64
            or not 64 <= self.message_bytes <= constants["kMessagesMessageMax"]
        ):
            raise ValueError(
                f"message_bytes {self.message_bytes}: a multiple of 64 to 256"
            )
        queue_size = self.queue_messages
        if queue_size < page or queue_size & (queue_size - 1) or queue_size > 1 << 31:
            raise ValueError(
                f"queue_messages {queue_size}: a power of two from {page} to 2**31"
            )
        if not 1 <= self.bins <= constants["kMessagesBinsMax"]:
            raise ValueError(
                f"bins {self.bins}: not in 1..{constants['kMessagesBinsMax']}"
            )
        most = abi.headers()["KM_COUNTERS_MAX"]
        if not 1 <= self.counters <= most:
            raise ValueError(f"counters {self.counters}: not in 1..{most}")
        if self.max_live < 1 or self.idle_cycles < 1:
            raise ValueError("max_live and idle_cycles must be at least 1")

    def buffers(self):
        """Runtime buffers of a dynamic handler: packets, outgoing messages, bins."""
        constants, _ = messages_header()
        packet = constants["kMessagesPacket"]
        return [
            Buffer(
                constants["kCbMessagesIn"],
                "uint32",
                constants["kMessagesHeader"] + packet * self.message_bytes,
                2,
            ),
            Buffer(
                constants["kCbMessagesOut"], "uint32", self.message_bytes, 2 * packet
            ),
            Buffer(
                constants["kCbMessagesBins"],
                "uint32",
                self.bins * packet * self.message_bytes,
                1,
            ),
        ]

    def queue_bytes(self, cores):
        """DRAM bytes of the queues of `cores` cores."""
        return cores * self.queue_messages * self.message_bytes

    def check(self, cores, dram_bytes, depth):
        """The launcher's rejections for a group of `cores` cores, `dram_bytes` of DRAM
        given to the queues and a ring depth `depth`; raises ValueError."""
        if self.max_live > self.queue_messages:
            raise ValueError(
                f"max_live {self.max_live} > queue_messages {self.queue_messages}: "
                "every message aimed at one core would not fit its queue"
            )
        if self.queue_bytes(cores) > dram_bytes:
            raise ValueError(
                f"queues of {cores} cores need {self.queue_bytes(cores)} bytes, "
                f"{dram_bytes} given"
            )
        if cores <= self.counters:
            raise ValueError(
                f"{cores} cores leave no core beside {self.counters} counter tiles"
            )
        if self.max_live < cores - self.counters:
            raise ValueError(
                f"max_live {self.max_live} < {cores - self.counters} cores: no quota"
            )
        if depth < 2:
            raise ValueError("a dynamic run needs ring depth 2 for its end command")

    def word(self):
        """The dynamic handlers as the kernels and the dispatcher read them, one a byte
        (KM_IS_DYNAMIC); one handler's word is its number."""
        return sum(number << 8 * index for index, number in enumerate(self.handlers))

    def quota(self, cores):
        """Sends each core starts a run with, for a group of `cores` cores."""
        return self.max_live // max(cores - self.counters, 1)

    def args(self, queue, lines):
        """Reader and writer words of km::MessagesArgs (they precede the program's)."""
        _, fields = messages_header()
        values = {
            "handler": self.word(),
            "queue": queue,
            "cap": self.queue_messages,
            "message_bytes": self.message_bytes,
            "lines": lines,
            "idle": self.idle_cycles,
            "bins": self.bins,
            "counters": self.counters,
        }
        return [int(values[field]) & abi.M32 for field in fields]

    def compute_args(self, cores):
        """Compute's words (message bytes, quota, counters) before the program's."""
        return [self.message_bytes, self.quota(cores), self.counters]
