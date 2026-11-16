"""A program: its kernel sources (three, or one movement source for both data movers
and a compute source), buffers, start-up arguments, ring depth and handler numbers. A
plain description; nothing here needs a device.
"""

import functools
import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from komorebi import _env, abi, formats
from komorebi.abi import handlers_from_header
from komorebi.errors import ProgramError
from komorebi.movement import check_pairs
from komorebi.overlay import Overlay, Slots, declare, overlays_from_header
from komorebi.overlay import constants as overlay_constants
from komorebi.thread import Thread

KERNELS = ("reader", "compute", "writer")
# The movement form: one source built for both data-movement CPUs, and compute.
MOVEMENT_KERNELS = ("movement", "compute")
# The runtime's KernelBuildOptLevel names; a kernel without one keeps the runtime's
# default (O2 for the reader and writer, O3 for the compute kernel).
OPT_LEVELS = ("O0", "O1", "O2", "O3", "Os", "Ofast", "Oz")
# Each data-movement kernel's NoC by default: the reader (RISCV_0, mover A) on NoC 0,
# the writer (RISCV_1, mover B) on NoC 1. A program may swap them, not share one.
NOCS = {"reader": 0, "writer": 1}
# The two forms of a kernel source: the kernel form (it defines kernel_main) or a
# program's function in the runtime's form (it defines km_program).
KERNEL_MAIN = re.compile(r"\bkernel_main\s*\(")
KM_PROGRAM = re.compile(r"\bkm_program\s*\(")
COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


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


class Handler(int):
    """A handler number of one program (`program`): what Run.submit, Graph.command and
    a Stream's submit take. As an int it is the number."""

    def __new__(cls, number, program):
        made = super().__new__(cls, number)
        made.program = program
        return made

    @property
    def field(self):
        """The program field of its commands: the program's place in the run."""
        return self.program.program_field

    def __repr__(self):
        return f"Handler({int(self)}, {self.program.name!r})"


# The earlier names of two buffer formats.
FORMAT_ALIASES = {"bfloat8_b": "bfp8_b", "bfloat4_b": "bfp4_b"}


def check_families(name, buffers, compute):
    """One exponent family a kernel (the unpacker and packer have one): a program
    whose buffers are of the A and the B family is refused, naming one of each;
    float32 and the integers go with either. An A buffer needs a 16-bit Dst."""
    first = {}
    for buffer in buffers:
        family = formats.FAMILY.get(FORMAT_ALIASES.get(buffer.format, buffer.format))
        if family is not None:
            first.setdefault(family, buffer)
    if len(first) == 2:
        a, b = first["a"], first["b"]
        raise ProgramError(
            f"{name}: buffer {a.index} ({a.format}) is of the A family (a 5-bit "
            f"exponent) and buffer {b.index} ({b.format}) of the B family (an 8-bit "
            "one): a kernel's buffers are of one family; float32 and the integers "
            "go with either"
        )
    if "a" in first and compute.get("fp32_dest_acc_en"):
        a = first["a"]
        raise ProgramError(
            f"{name}: buffer {a.index} ({a.format}) is of the A family, which the "
            "card reads right only into a 16-bit Dst (in an fp32 Dst each datum "
            "arrives 2^-15 off): fp32_dest_acc_en is off for it"
        )


def source_form(path):
    """ "program" when the source defines km_program (comments aside), else "kernel"
    (the kernel form); ProgramError when it defines km_program and kernel_main."""
    text = COMMENTS.sub("", Path(path).read_text())
    if not KM_PROGRAM.search(text):
        return "kernel"
    if KERNEL_MAIN.search(text):
        raise ProgramError(
            f"{path}: defines kernel_main and km_program; a source is of the kernel "
            "form (kernel_main) or a program's function (km_program), not both"
        )
    return "program"


def broadcast_defines(name, defines, wait):
    """`defines` with KM_BROADCAST_WAIT 1 when `wait` is True (the header's default
    is 0, so a program without it adds nothing); a define that says otherwise raises."""
    given = defines.get("KM_BROADCAST_WAIT")
    if given is not None and (str(given) != "0") != bool(wait):
        raise ProgramError(
            f"{name}: defines KM_BROADCAST_WAIT={given} and broadcast_wait={wait}; "
            "give broadcast_wait alone"
        )
    return {**defines, "KM_BROADCAST_WAIT": 1} if wait else defines


def guard_defines(name, defines, guard):
    """`defines` with KM_GUARD 0 when `guard` is False (the header's default is 1, so
    a program with the guard adds nothing); a define that says otherwise raises."""
    given = defines.get("KM_GUARD")
    if given is not None and (str(given) != "0") != bool(guard):
        raise ProgramError(
            f"{name}: defines KM_GUARD={given} and guard={guard}; give guard alone"
        )
    return defines if guard else {**defines, "KM_GUARD": 0}


def unguarded(programs):
    """The names of those of `programs` built with guard=False."""
    return [program.name for program in programs if not program.guard]


def guard_says(names, recover=True):
    """What a load of programs built with guard=False tells its log, once."""
    return (
        f"guard=False ({', '.join(names)}): commands are not given up at a deadline. "
        "The kernels read no abort word and keep no live words: at the deadline the "
        "dispatcher takes each core as having left and frees the command, a core held "
        "in a wait stays held, and a NoC barrier has no limit at all, so a hang then "
        "costs a card reset. run.inspect refuses these cores"
        + (
            "; the reset lists (recover=True) reset only a core whose own wait limit "
            "ended its kernels"
            if recover
            else ""
        )
    )


@dataclass(eq=False)  # one program is one object: a key of session.load's dict
class Program:
    """What the persistent kernel runs on every core: `kernels` a directory with
    reader.cpp, compute.cpp and writer.cpp, or movement.cpp and compute.cpp (or a dict),
    `args(core)` the (reader, writer, compute) arguments after the start-up ones,
    `handlers` a dict or header path, `opt` per-kernel levels, `noc` a NoC per mover."""

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
    noc: dict = field(default_factory=dict)  # "reader" / "writer" -> 0 or 1
    x280_ports: dict | None = None  # Program.ports(): {"serve": ..., "noc": ...}
    # handlers (names or numbers) whose commands use the cores' ports: run.submit
    # refuses one on cores a command of them still holds unless it is exclusive
    port_handlers: set = field(default_factory=set)
    # True: a word-only broadcast's caster waits at finish for every receiver's
    # clear of VALID (KM_BROADCAST_WAIT 1, km_broadcast.hpp); False (default): none
    broadcast_wait: bool = False
    # False: no deadline, give-up or inspect in its kernels (KM_GUARD 0, km_core.hpp)
    guard: bool = True
    # True: its movers' tt-metal page calls mean its own island's bank (island_names)
    own_island_pages: bool = False
    sizes: KernelSizes | None = field(default=None, init=False, compare=False)
    pairs: list = field(default_factory=list, init=False)  # the BufferPairs given
    overlays: dict = field(default_factory=dict, init=False)  # name -> Overlay
    overlay_slots: Slots = field(default_factory=Slots, init=False)
    header: Path | None = field(default=None, init=False)  # the handlers' header
    # the overlays as the last load built them (komorebi.overlay.Built)
    overlays_built: object = field(default=None, init=False, compare=False)
    # the program field of its commands: its place in the last load (0: the first)
    program_field: int = field(default=0, init=False, compare=False)
    # in a run of programs, what the run adds to its own overlays' numbers
    overlay_shift: int = field(default=0, init=False, compare=False)
    # where its per-core args start in a run of several programs (programs.plan)
    args_at: int = field(default=0, init=False, compare=False)

    def __init_subclass__(cls, compute=None, defines=None, startup=None, **kwargs):
        """A class deriving Program with handler methods is a program written as a
        class: made, it is compiled into an ordinary program (komorebi.dsl);
        `compute` its compute kernel's options by the host API's names, `defines`
        its kernels' defines, `startup` compute's start-up (km.Startup)."""
        super().__init_subclass__(**kwargs)
        importlib.import_module("komorebi.dsl.program").prepare(
            cls, compute, defines, startup
        )

    def __post_init__(self):
        if isinstance(self.handlers, str | Path):
            self.header = Path(self.handlers)
            self.handlers = handlers_from_header(self.handlers)
        user_max = abi.headers()["KM_HANDLER_USER_MAX"]
        for name, number in self.handlers.items():
            if not 1 <= number <= user_max:
                raise ValueError(f"handler {name} = {number}: not in 1..{user_max}")
        self.defines = broadcast_defines(self.name, self.defines, self.broadcast_wait)
        self.defines = guard_defines(self.name, self.defines, self.guard)
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
        unknown = set(self.noc) - set(NOCS)
        if unknown or any(value not in (0, 1) for value in self.noc.values()):
            raise ProgramError(
                f"{self.name}: noc {self.noc!r}: 0 or 1 for 'reader' and 'writer'"
            )
        if self.kernel_noc("reader") == self.kernel_noc("writer"):
            raise ProgramError(
                f"{self.name}: noc {self.noc!r} puts both data-movement kernels on NoC "
                f"{self.kernel_noc('reader')}; each takes its own NoC"
            )
        for given in (self.compile_args, self.opt):
            if "movement" in given and {"reader", "writer"} & set(given):
                raise ProgramError(
                    f"{self.name}: 'movement' stands for the reader and the writer; "
                    "give it or them, not both"
                )
        check_families(self.name, self.buffers, self.compute)

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
        if self.messages is not None and self.messages.parts:
            raise ProgramError(
                f"{self.name}: its handlers are parts, which take the overlay numbers; "
                "it has no overlays of its own"
            )
        if number is None:
            numbers = overlays_from_header(self.header) if self.header else {}
            if name not in numbers:
                raise ValueError(
                    f"overlay {name}: no number given and no kOverlay constant for it "
                    "in the handlers' header"
                )
            number = numbers[name]
        made = declare(name, parts, number, defines)
        if made.number == overlay_constants()["kOverlayService"]:
            raise ValueError(
                f"overlay {name}: number {made.number} is the runtime's service overlay"
            )
        taken = {o.number: o.name for o in self.overlays.values()}
        if made.number in taken:
            raise ValueError(
                f"overlay {name}: number {made.number} is taken by {taken[made.number]}"
            )
        self.overlays[name] = made
        return self

    def ahead(self, name, parts=None, defines=None, compile_args=None, **wanted):
        """Code a run of this program will add, built by the load (and by a build off
        the card) into the parts' cache so the add finds it built: run.addon's `name`,
        `parts`, `defines`, `compile_args`, or a class's handler at a set its first use
        compiles, `program.ahead(Gate.gate, size=4)`; or a class run.addon will add."""
        if isinstance(name, Program):
            if parts is not None or defines or compile_args or wanted:
                raise ProgramError(f"{name.name}: a class ahead takes nothing more")
            kept = getattr(self, "added_ahead_programs", ())
            self.added_ahead_programs = [*kept, name]
            return self
        added = importlib.import_module("komorebi.added")  # it imports this module
        added.ahead(self, name, parts, defines, compile_args, wanted)
        return self

    def slots(self, count=None, bytes=None, service=None):
        """Declares the slots: `count` (1 or more; None: one an overlay with a part
        there, as the L1 holds) and `bytes` per slot, each an int or a {kernel: value},
        and the service overlay's ("own", "shared"; None: the load chooses). Named
        `bytes` also give the kernels the path of run.addon: KM_ADDED_HANDLERS."""
        self.overlay_slots = Slots(count, bytes, service)
        if bytes is None:
            self.defines.pop("KM_ADDED_HANDLERS", None)
        else:
            self.defines["KM_ADDED_HANDLERS"] = "1"
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
        if self.messages is not None and self.messages.parts:
            return message_kernels()
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

    def kernel_noc(self, kernel):
        """The NoC of "reader" or "writer" (in the movement form: mover A's CPU and
        mover B's): the program's `noc`, else NoC 0 and NoC 1."""
        return self.noc.get(kernel, NOCS[kernel])

    def handler(self, handler):
        """The Handler of this program given by name or number: its number, and this
        program, whose place in the run gives its commands' program field."""
        if isinstance(handler, Handler) and handler.program is not self:
            raise ValueError(f"{handler!r} is not a handler of {self.name}")
        if isinstance(handler, str):
            if handler not in self.handlers:
                raise KeyError(f"{self.name} has no handler {handler!r}")
            return Handler(self.handlers[handler], self)
        return Handler(int(handler), self)

    def form(self):
        """ "kernel" (the kernel form: kernel_main) or "program" (the runtime's form:
        each source defines km_program, the runtime kernel_main). ProgramError if
        mixed."""
        paths = self.sources()
        forms = {kernel: source_form(path) for kernel, path in paths.items()}
        if len(set(forms.values())) != 1:
            named = ", ".join(f"{kernel} {forms[kernel]}" for kernel in sorted(forms))
            raise ProgramError(
                f"{self.name}: its sources mix the two forms ({named}): every source "
                "defines kernel_main (the kernel form) or every one km_program"
            )
        form = forms["reader"]
        if form == "program" and self.movement():
            raise ProgramError(
                f"{self.name}: a program in the runtime's form has reader.cpp, "
                "writer.cpp and compute.cpp, not a movement source"
            )
        return form

    def dynamic(self, handler, *handlers, parts=None, **options):
        """Makes `handler`, and each of `handlers`, a dynamic handler of one message
        loop (options: those of Dynamic) and adds its buffers. `parts`: each body as an
        overlay part, a folder of <name>.cpp or {name: source}. Returns the program."""
        if self.messages is not None:
            raise ValueError(f"{self.name} already has a dynamic handler")
        if parts is not None and self.overlays:
            raise ProgramError(
                f"{self.name}: handlers as parts take the overlay numbers; a program "
                "with overlays of its own keeps its handlers in its compute kernel"
            )
        numbers = tuple(int(self.handler(each)) for each in (handler, *handlers))
        messages = Dynamic(
            handler=numbers[0], handlers=numbers, parts=parts is not None, **options
        )
        if parts is not None:
            names = {number: name for name, number in self.handlers.items()}
            for index, number in enumerate(numbers):
                name = names.get(number, str(number))
                if isinstance(parts, dict):
                    source = parts.get(name, parts.get(number))
                    if source is None:
                        raise ValueError(f"{self.name}: no part for handler {name}")
                else:
                    source = Path(parts) / f"{name}.cpp"
                self.overlays[name] = Overlay(
                    name, index + 1, {"compute": Path(source)}, message=True
                )
        self.messages = messages
        self.buffers = [*self.buffers, *self.messages.buffers()]
        return self

    def core_args(self, core):
        """(reader, writer, compute) program arguments of `core`."""
        if self.args is None:
            return [], [], []
        reader, writer, compute = self.args(core)
        return list(reader), list(writer), list(compute)


def message_kernels():
    """{"reader", "compute", "writer"}: the runtime's kernels of a dynamic program
    whose handlers are overlay parts."""
    folder = _env.include_directory().parent / "kernels"
    return {kernel: folder / f"messages_{kernel}.cpp" for kernel in KERNELS}


def direct_declared(direct):
    """`direct=` as a tuple: rank offsets (ints), or (sender, receiver) pairs from a
    list of pairs or a mapping {sender: receivers}. Raises ValueError."""
    if isinstance(direct, dict):
        direct = [(s, r) for s, receivers in direct.items() for r in receivers]
    items = tuple(direct or ())
    if all(isinstance(item, int | np.integer) for item in items):
        if len(set(items)) < len(items):
            raise ValueError(f"direct offsets {items}: an offset is given twice")
        return tuple(int(item) for item in items)
    if not all(len(tuple(item)) == 2 for item in items):
        raise ValueError(
            "direct: rank offsets (ints), (sender, receiver) pairs, or "
            "{sender: receivers}"
        )
    return tuple((int(s), int(r)) for s, r in items)


def direct_layout(receivers, senders, slot_bytes):
    """A core's direct area (km_messages.hpp DirectArea) for its declared receivers
    and senders: its header words (byte offsets) and its size in bytes."""
    constants, _ = messages_header()
    outgoing = constants["kDirectHeader"]
    incoming = outgoing + 4 * receivers
    credit = (incoming + 4 * senders + 15) // 16 * 16
    sent = credit + 16 * receivers
    written = sent + 4 * receivers
    taken = (written + 4 * receivers + 15) // 16 * 16
    slots = taken + 16 * senders
    end = slots + senders * slot_bytes
    header = [receivers, senders, outgoing, incoming, credit, sent, written, taken]
    return [*header, slots], end


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
    """Dynamic `handlers` (default: `handler` alone; `parts`: bodies as overlay parts),
    messages of `message_bytes` (a body of up to `body_bytes`) in queues of
    `queue_messages`, `max_live` alive; `bins` a writer, sent at `flush_age` cycles old
    (0: never); idle: `idle_cycles`; `counters`, `x280_ranks`; `direct` pairs."""

    handler: int
    message_bytes: int = 64
    max_live: int = 1 << 16
    queue_messages: int = 1 << 16
    bins: int = 16
    idle_cycles: int = 20000
    counters: int = 2
    handlers: tuple = ()
    parts: bool = False
    body_bytes: int = 0
    x280_ranks: int = 0
    x280_cluster: int = 1
    direct: object = ()
    stamps: bool = False
    flush_age: int = 1000

    def __post_init__(self):
        constants, _ = messages_header()
        object.__setattr__(self, "direct", direct_declared(self.direct))
        handlers = tuple(int(number) for number in self.handlers) or (self.handler,)
        object.__setattr__(self, "handlers", handlers)
        user_max = abi.headers()["KM_HANDLER_USER_MAX"]
        most = (
            constants["kMessagesHandlersMax"]
            if self.parts
            else abi.headers()["KM_DYNAMIC_MAX"]
        )
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
        if not 0 <= self.flush_age < 1 << 31:
            raise ValueError(f"flush_age {self.flush_age}: cycles, 0 to 2**31 - 1")
        if (
            self.body_bytes % 64
            or not 0 <= self.body_bytes <= constants["kMessagesBodyMax"]
        ):
            raise ValueError(
                f"body_bytes {self.body_bytes}: 0 or a multiple of 64 to "
                f"{constants['kMessagesBodyMax']}"
            )
        most = abi.headers()["KM_X280_RANKS_MAX"]
        if not 0 <= self.x280_ranks <= most:
            raise ValueError(f"x280_ranks {self.x280_ranks}: not in 0..{most}")
        if self.x280_ranks and self.x280_cluster not in (1, 2, 3):
            raise ValueError(f"x280_cluster {self.x280_cluster}: work clusters 1 to 3")
        if self.x280_ranks and self.body_bytes:
            raise ValueError("X280 ranks take no bodies: body_bytes must be 0")
        if self.direct and self.x280_ranks:
            raise ValueError("direct messages are between cores: no X280 ranks")

    def buffers(self):
        """Runtime buffers of a dynamic handler: packets, outgoing messages, bins."""
        constants, _ = messages_header()
        packet = constants["kMessagesPacket"]
        bodies = constants["kMessagesBodyPacket"]  # a packet's bodies, after its lines
        return [
            Buffer(
                constants["kCbMessagesIn"],
                "uint32",
                constants["kMessagesHeader"]
                + packet * self.message_bytes
                + bodies * self.body_bytes,
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
        """DRAM bytes of the queues of `cores` cores, their bodies included."""
        return cores * self.queue_messages * (self.message_bytes + self.body_bytes)

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
        if self.max_live < self.ranks(cores):
            raise ValueError(
                f"max_live {self.max_live} < {self.ranks(cores)} ranks: no quota"
            )
        first_index = abi.headers()["KM_CORES_MAX"] - abi.headers()["KM_X280_RANKS_MAX"]
        if self.x280_ranks and cores > first_index:
            raise ValueError(
                f"X280 ranks take core indices {first_index} on: {cores} cores"
            )
        if depth < 2:
            raise ValueError("a dynamic run needs ring depth 2 for its end command")
        self.direct_pairs(cores - self.counters)

    def direct_pairs(self, ranks):
        """The declared (sender, receiver) pairs of a run of `ranks` ranks: offsets d
        give (r, (r + d) mod ranks) for every rank r. Raises ValueError."""
        if not self.direct:
            return []
        if isinstance(self.direct[0], int):
            pairs = [(r, (r + d) % ranks) for d in self.direct for r in range(ranks)]
        else:
            pairs = list(self.direct)
        if len(set(pairs)) < len(pairs):
            raise ValueError(f"direct: a pair is declared twice (ranks {ranks})")
        for sender, receiver in pairs:
            if not (0 <= sender < ranks and 0 <= receiver < ranks):
                raise ValueError(
                    f"direct: pair ({sender}, {receiver}) is not two of the "
                    f"{ranks} message ranks"
                )
        return pairs

    def direct_rows(self, cores):
        """Each core's direct area as words (km::DirectArea): its tables of declared
        receivers and senders, credit lines, counts and slots; one row a core."""
        pairs = self.direct_pairs(cores - self.counters)
        receivers = [[] for _ in range(cores)]
        senders = [[] for _ in range(cores)]
        for sender, receiver in pairs:
            receivers[sender].append(receiver)
            senders[receiver].append(sender)
        layouts = [
            direct_layout(len(receivers[core]), len(senders[core]), self.slot_bytes())
            for core in range(cores)
        ]
        rows = np.zeros((cores, max(end for _, end in layouts) // 4), np.uint32)
        for core, (header, _) in enumerate(layouts):
            rows[core, : len(header)] = header
            outgoing, incoming = header[2] // 4, header[3] // 4
            for index, receiver in enumerate(receivers[core]):  # its slot there / 16
                slot = layouts[receiver][0][8] + senders[receiver].index(core) * (
                    self.slot_bytes()
                )
                rows[core, outgoing + index] = receiver | slot // 16 << 16
            for index, sender in enumerate(senders[core]):  # its credit line there / 16
                credit = layouts[sender][0][4] + 16 * receivers[sender].index(core)
                rows[core, incoming + index] = sender | credit // 16 << 16
        return rows

    def slot_bytes(self):
        """A direct slot: a line and, in a program with bodies, its body."""
        return self.message_bytes + self.body_bytes

    def direct_bytes(self, cores):
        """L1 bytes of a core's direct area (the largest core's): 0 without direct."""
        return self.direct_rows(cores).shape[1] * 4 if self.direct else 0

    def word(self):
        """The handlers a RUN command may name (the first KM_DYNAMIC_MAX) as the
        kernels and the dispatcher read them, one a byte; one handler's is its number.
        """
        runs = self.handlers[: abi.headers()["KM_DYNAMIC_MAX"]]
        return sum(number << 8 * index for index, number in enumerate(runs))

    def defines(self):
        """The defines of the runtime's kernels of handlers as parts: the program's
        handlers in order (handler i is overlay i + 1)."""
        stamps = [("KM_MESSAGES_STAMPS", "1")] if self.stamps else []
        if not self.parts:
            return stamps
        handlers = ",".join(str(number) for number in self.handlers)
        return [("KM_MESSAGE_PARTS", "1"), ("KM_MESSAGE_HANDLERS", handlers)] + stamps

    def ranks(self, cores):
        """The ranks of a group of `cores` cores: its cores without the counter tiles,
        then the X280 ranks."""
        return cores - self.counters + self.x280_ranks

    def quota(self, cores):
        """Sends each rank starts a run with, for a group of `cores` cores."""
        return self.max_live // max(self.ranks(cores), 1)

    def queue_region_bytes(self):
        """Bytes of the X280 ranks' queues in their cluster's bank (0: none)."""
        return self.x280_ranks * self.queue_messages * self.message_bytes

    def dispatcher_word(self):
        """The dispatcher's messages_handler: word() and the X280 ranks above it."""
        ranks = self.x280_cluster | self.x280_ranks << 4 if self.x280_ranks else 0
        return self.word() | ranks << 32

    def args(self, queue, lines, body=0, x280=None, direct=0, stamps=0):
        """Reader and writer words of km::MessagesArgs (they precede the program's);
        `body`: bodies' DRAM address; `x280`: the X280 ranks' words by field name;
        `direct`, `stamps`: the direct area's and the stamps' L1 addresses."""
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
            "body": body,
            "body_bytes": self.body_bytes,
            "x280": self.x280_ranks,
            "direct": direct,
            "stamps": stamps,
            "age": self.flush_age,
        }
        values |= x280 or {}
        return [int(values.get(field, 0)) & abi.M32 for field in fields]

    def compute_args(self, cores, direct=0):
        """Compute's words (message and body bytes, quota, counters, X280 ranks, the
        direct area's L1 address) before the program's."""
        constants, _ = messages_header()
        sizes = (
            self.message_bytes | self.body_bytes << constants["kMessagesCArgBodyShift"]
        )
        return [sizes, self.quota(cores), self.counters, self.x280_ranks, direct]
