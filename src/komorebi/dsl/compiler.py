"""The compiler of a program written as a class, level 1: each handler's body, read
with `ast`, becomes the nodes of the movement source (both movers) and of the compute
kernel; streams become buffer pairs, parameters command words, a refusal names the
line. `komorebi.dsl.program` writes the files and makes the Program.
"""

import ast
import copy
import importlib
import inspect
import keyword
import linecache
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from komorebi import abi
from komorebi.dsl import front
from komorebi.dsl.merge_token import tokened
from komorebi.dsl.names import NAMES, kind_of, spelling
from komorebi.dsl.nodes import Block, Glue, Group, Line, has_line
from komorebi.dsl.tile import trace, unit_program
from komorebi.dsl.tile.ir import (
    A_DTYPES,
    BF16,
    BLOCK_B,
    F32,
    KernelIR,
    PlanError,
    TensorParam,
)
from komorebi.dsl.tile.rows import chunk_program, row_program
from komorebi.dsl.tile.steps import step_program
from komorebi.dsl.tile.unit import UnitError
from komorebi.errors import ProgramError
from komorebi.program.movement import Pattern

# Words of a command a handler's arguments may take (a0 to a9).
COMMAND_WORDS = 10
# Default patterns, the reference's measured best on the whole grid for reads; a write
# of whole blocks goes by mover B's NoC 1, as the classic writer's.
READ_PATTERN = "AAAB"
WRITE_PATTERN = "B"
# Items a read half holds (double buffering on each mover and one more each), and a
# write half.
READ_ITEMS = 4
WRITE_ITEMS = 2
# A reduce's arrival buffer: one slot for every other core of the largest group a
# reduce by owner takes on a 120-core grid (119 cores), one tile a slot.
REDUCE_SLOTS = 118
# Blocks a broadcast holds in flight when the plan does not say.
BROADCAST_WINDOW = 2
# Circular buffer indices: reads from 0, writes from 16, the others from 8 (24 and up
# are the runtime's).
FIRST_READ, FIRST_OTHER, FIRST_WRITE, LAST_INDEX = 0, 8, 16, 25
HANDOVER_WORDS = 16  # a body's words of the other mover: one 64 B page
REGION_PAGE = 4096  # an X280 thread's region: whole pages of one bank (thread.PAGE)
PAGE_BYTES = {"bfloat16": 2048, "float32": 4096}
TILE_ELEMENTS = 32 * 32
# Every local the generated code declares starts with GENERATED_PREFIX, which a
# parameter or a variable may not; the runtime's names bodies use are taken too.
GENERATED_PREFIX = "km_"
GENERATED = {"command", "movement", "compute"}
# The runtime's where names (komorebi.session.state) of each kind of wait.
WAIT_DATA, WAIT_INPUT, WAIT_OUTPUT = "data", "cinput", "coutput"
DST_HALF = 4  # fp32 tiles in DST's half a tile function's calls take (half sync)
DST_FULL = 8  # fp32 tiles in all of DST (full sync: sums wider than a half)
# A reduce of n tiles by tree: a slot of n tiles for each round (km::kTreeRounds).
TREE_ROUNDS = 7
WAIT_PORT, WAIT_EMIT = "port", "emit"
WAIT_VALID, WAIT_DONE = "broadcastvalid", "broadcastdone"
WAIT_FREE = "broadcastfree"
# A broadcast's mover: along a row on mover A (NoC 0), down a column on mover B (NoC 1).
ALONG = {"row": "A", "column": "B"}
# A tile function's stamps page: the sequence, the calls, 16 empty spans (a wall-clock
# pair around one tensix_sync), then each call's span from word 18.
SPANS_EMPTY, SPANS_FIRST = 16, 18
# A stamped handler of steps: the page's last words, for unpack, math and pack: the
# span, the waits for inputs and the waits for room (TIMED_WAITS sorts each wait).
CPU_WORDS = 9
TIMED_WAITS = (  # (the call's start, the counter, an expression or a statement)
    ("compute.wait_front(", "km_waited_in", True),
    ("km::broadcast_receive(compute, ", "km_waited_in", True),
    ("compute.reserve_back(", "km_waited_out", True),
    ("tile_regs_acquire();", "km_waited_out", False),
    ("tile_regs_wait();", "km_waited_in", False),
)
TILE_FORMATS = {"float32": F32, "bfloat16": BF16, **{b: b for b in BLOCK_B}}
# An inline body's CPU (on=): a mover, one compute CPU, or compute's three; the #if of
# each one CPU's code.
COMPUTE_CPUS = ("unpack", "math", "pack")
# What compute's three CPUs run together in a body: a word of pack's alone must not
# steer it (Compiler.pack_words_steer).
STEERED = re.compile(
    r"\btile_regs_\w+\s*\(|\bcopy_tile\s*\(|\bmatmul\w*\s*\(|\b(?!pack_)\w+_tile\s*\(|"
    r"\b(?:return|break|continue|goto)\b"
)
BODY_CPUS = ("mover_a", "mover_b", *COMPUTE_CPUS)
CPU_GUARDS = {
    "mover_a": "defined(COMPILE_FOR_BRISC)",
    "mover_b": "!defined(COMPILE_FOR_BRISC)",
    **{cpu: f"defined(TRISC_{cpu.upper()})" for cpu in COMPUTE_CPUS},
}


class Refused(ProgramError):
    """A line the compiler does not take: the line and what to write instead."""


@dataclass
class Param:
    """A handler parameter: its kind ("tensor", "int", "pattern") and command word."""

    name: str
    kind: str
    word: int


@dataclass
class Buffer:
    """A buffer of the program: a pair (`pair` True: indices a, b) or one buffer
    (index a), its format, page bytes, pages (a pair: each half's), the line."""

    name: str
    role: str  # "read", "write", "arrival", "broadcast", "field"
    index_a: int
    index_b: int | None
    dtype: str
    page_bytes: int
    pages: tuple
    block: int = 1
    what: str = ""
    patterns: dict = field(default_factory=dict)  # constant name -> letters


@dataclass
class Move:
    """A line that moves pages: its handler, line and text, the tensor, and how
    (a read stream, a write stream, a page write, a fill)."""

    handler: str
    line: int
    text: str
    tensor: str
    how: str


@dataclass
class Wait:
    """A wait of the generated code: handler, CPU, the runtime's where, Python line."""

    handler: str
    cpu: str
    where: str
    line: int
    text: str


@dataclass
class Value:
    """What a Python name holds in a handler: `kind` and what the C++ needs of it."""

    kind: str  # int, pattern, partition, range, tensor, write, tile, reduced, broadcast
    cxx: str = ""
    first: str = "0"  # partition: the range's first page
    count: str = ""  # range: pages
    tensor: str = ""  # range: the tensor parameter
    buffer: Buffer | None = None
    pattern: str = ""
    dst: int = 0
    packed: bool = False
    pushed: bool = False
    mover: str = ""  # broadcast: the mover that runs it
    window: int = 0  # broadcast: blocks in flight
    length: int | None = None  # broadcast: cores a band (None: the whole side)
    unread: tuple | None = None  # broadcast: (C++ of its unread word, names it reads)
    loop: dict | None = None  # broadcast: the range loop its blocks count
    waited: bool = False  # a written tile: mover B has waited for its page
    popped: bool = False  # a written tile: mover B has freed its page
    words: dict | None = None  # a message: its class's named words
    width: int = 1  # a streamed row, or a tile function's result: its tiles
    batch: tuple | None = None  # a batched call's result: (its loop, DST a call)
    # a range of a rectangle of tiles: (rows, columns, the tensor's tile columns),
    # C++ texts but the last; rows and columns ints where the build knows them
    rect: tuple | None = None
    # a block a broadcast brought: each part's (first slot page, slot rows, pitch,
    # transposed), pitch the slot's pages a row (ints, or None where a word)
    parts: tuple | None = None
    # a rectangle's most rows and columns where the build knows them (a side's length,
    # or n of `c:min(c + n, e)`); its sides themselves may be words
    bounds: tuple | None = None
    sides: tuple | None = None  # a rectangle's rows (C++) and columns (int or C++)
    items: tuple | None = None  # a range of items: (bytes an item, items a page)
    summed: str = ""  # a reduce of a field of sums: the field
    letters: str = ""  # a write stream's stated or default pattern letters
    # a write stream's move outside a loop, or a stream loop's move (kind "moved"):
    # its line, rewritten when a mover's body takes it
    move: Line | None = None
    ended: tuple | None = (
        None  # a "moved" of a loop inside an if or a loop: (kind, line)
    )
    # a page's table of entries (kind "entries") or one entry of it (kind "entry"):
    # its class, buffers, words (entries_table)
    table: dict | None = None
    depth: int = 0  # a write stream: the loops around its line
    # a once-only object (a tile of DST, a reduce's sum, a block a broadcast brought):
    # (the range loops around the line that made it, that line), None where not kept
    made_at: tuple | None = None

    def first_text(self, name):
        """A partition's first page: its range's first page plus its start."""
        if self.first == "0":
            return f"{name}.start"
        return f"({self.first} + {name}.start)"


@dataclass
class Compiled:
    """The compiled class: handlers' numbers and words, buffers, the kernels' nodes per
    handler, includes, waits, the tensors' order of compile arguments."""

    name: str
    source: str
    handlers: dict = field(default_factory=dict)  # name -> number
    # overlay handler -> its overlay number; broadcast handler -> its km.Broadcast
    overlays: dict = field(default_factory=dict)
    broadcast_lines: dict = field(default_factory=dict)
    rect_lines: dict = field(default_factory=dict)  # handler -> self.x/y/core read at
    # mover -> [(handler, its broadcast's line)] in the class's order: a core holds
    # each one's arguments after the last's (kArgBroadcast + kBroadcastArgs i)
    mover_broadcasts: dict = field(default_factory=dict)
    # handler -> (the size of its reduce's groups, the reduce's line)
    reduce_groups: dict = field(default_factory=dict)
    port_handlers: set = field(default_factory=set)  # handlers whose cores use ports
    stamps: dict = field(default_factory=dict)  # overlay handler -> (tensor, word)
    dynamic: str | None = None  # the handler whose lines are a message loop
    # The other dynamic handlers, each its message loop alone: name -> method; their
    # loops are cases of the dynamic handler's loop, by the handler a message names.
    folded: dict = field(default_factory=dict)
    results: set = field(default_factory=set)  # handlers whose leader returns a word
    messages: dict = field(default_factory=dict)  # km.Message class name -> words
    words: dict = field(default_factory=dict)  # name -> [(parameter, kind)]
    lines: dict = field(default_factory=dict)  # name -> its def line
    buffers: list = field(default_factory=list)
    movement: dict = field(default_factory=dict)  # handler -> nodes
    compute: dict = field(default_factory=dict)  # handler -> nodes
    tensors: list = field(default_factory=list)  # tensor parameter names, in order
    regions: list = field(default_factory=list)  # region parameter names
    includes: set = field(default_factory=set)  # "port", "broadcast", "fill", "add"
    # The #include lines of inline bodies, by kernel ("movement", "compute").
    body_includes: dict = field(default_factory=lambda: {"movement": [], "compute": []})
    waits: list = field(default_factory=list)
    unpack_fp32: set = field(default_factory=set)  # buffer names compute copies fp32
    summed_fields: set = field(default_factory=set)  # fields of sums: compute packs
    broadcasts: list = field(default_factory=list)  # (along, buffer, window)
    rect_sized: bool = False  # a handler's rectangle given at each submit
    rect_plain: bool = False  # a handler reads the make's rectangle
    rect_broadcasts: set = field(default_factory=set)  # (mover, index) computed so
    constants: dict = field(default_factory=dict)  # plan names used -> value
    rectangle: bool = False  # self.x or self.y used
    fields: dict = field(default_factory=dict)  # name -> front.Field
    own: dict = field(default_factory=dict)  # an own field -> its push (front.Own)
    pushers: dict = field(default_factory=dict)  # own field -> {CPU: (handler, line)}
    # field -> (its producer: "fill" or "product", handler, line, the line's text)
    producers: dict = field(default_factory=dict)
    word_fields: dict = field(
        default_factory=dict
    )  # a field of km.u32 words: name -> n
    fills: list = field(default_factory=list)  # (field, count text, handler, line)
    moves: list = field(default_factory=list)  # Move of every line that moves pages
    # Each tile function's unit program by its C++ name, and each call of one.
    units: dict = field(default_factory=dict)
    tile_calls: list = field(default_factory=list)
    # Each product `acc += f(block, self.w)`: (handler, line, text, function, columns,
    # block tiles).
    products: list = field(default_factory=list)
    fidelity: str | None = None  # the products' (one a compute kernel), when given
    # Each product's placement: (handler, line, text, left format, right format,
    # "none" | "plain" | "turned", ("field" | "tensor", W's name) or None).
    placements: list = field(default_factory=list)
    # field -> (the handlers that fill it, those that read it and do not fill it)
    field_users: dict = field(default_factory=dict)
    full_dst: bool = False  # a row-form tile function: DST in full-sync mode
    # Each value an inline body or a line hands between compute CPUs by mailbox.
    handovers: list = field(default_factory=list)
    compute_options: dict = field(default_factory=dict)  # the class's compute=
    entries: dict = field(default_factory=dict)  # km.Entry class name -> words
    # The inline bodies at a kernel's start or end (at=): (at, kernel) -> texts.
    kernel_bodies: dict = field(default_factory=dict)
    kernel_body_lines: list = field(default_factory=list)  # their lines, in order
    reserved: set = field(default_factory=set)  # indices left to other programs
    islands: tuple = (1, 0)  # the session's (islands, an island's most ranks)
    startup: object = None  # the class's km.Startup, or None (the default start-up)
    # A handler with constant parameters -> [(its set of values, the handler built
    # for them)] (handler_variants).
    variants: dict = field(default_factory=dict)
    # The class's constants (km.constant) by name: this program's values; a symbol's
    # only where the plan gives it (declared_constants).
    class_constants: dict = field(default_factory=dict)
    declared: dict = field(default_factory=dict)  # name -> its front.Constant
    chain: int = 0  # the chain's walk: its handler number (0: none in this build)


def handler_methods(cls):
    """The class's handler methods in the order of their definitions: its parents'
    first, a subclass's handler of a parent's name in that one's place."""
    found = {}
    for klass in reversed(cls.__mro__):
        own = [
            value
            for value in vars(klass).values()
            if isinstance(value, front.HandlerMethod)
        ]
        for method in sorted(own, key=lambda m: m.function.__code__.co_firstlineno):
            found[method.name] = method
    return list(found.values())


def class_annotations(cls):
    """The annotations of the class and its parents, a subclass's over its
    parents'."""
    merged = {}
    for klass in reversed(cls.__mro__):
        merged.update(inspect.get_annotations(klass))
    return merged


def compile_class(
    cls,
    plan,
    tensors,
    l1=None,
    constants=None,
    handlers=None,
    reserved=(),
    islands=(1, 0),
):
    """Compiles each handler of `cls`; `plan` sizes, `tensors` an example of each tensor
    parameter, `l1` the bytes for buffers, `constants` their values, `handlers` the sets
    (handler_variants), `reserved` indices left to others, `islands` the session's
    (islands, an island's most ranks): a tree's slots."""
    methods = handler_methods(cls)
    compiled = Compiled(cls.__name__, inspect.getsourcefile(methods[0].function))
    compiled.reserved = set(reserved)
    compiled.islands = tuple(islands)
    declared_constants(compiled, cls, constants)
    compiled.fields = {
        name: front.Field(
            value.format,
            tuple(shape_size(compiled, name, n) for n in value.shape),
        )
        for name, value in class_annotations(cls).items()
        if isinstance(value, front.Field)
    }
    compiled.own = {
        name: inspect.getattr_static(cls, name).push
        for name in compiled.fields
        if isinstance(inspect.getattr_static(cls, name, None), front.Own)
    }
    for name, value in list(compiled.fields.items()):
        if word_field(value) and name in compiled.own:
            raise field_refused(
                compiled,
                name,
                "a field of words is the host's, not a body's own",
                "declare it of whole tiles, km.L1[format, n, 32, 32] = "
                f"km.{spelling('own')}()",
            )
        if word_field(value):  # words in each core's row of an L1 tensor
            compiled.word_fields[name] = value.shape[0]
            del compiled.fields[name]
    compiled.field_users = field_users(methods, compiled.fields)
    dynamic = [m for m in methods if arrive_of(source_of(m.function)[0])]
    if len(dynamic) > 1:
        loops = [m for m in dynamic if not loop_alone(m)]
        if len(loops) > 1:
            definition = source_of(loops[1].function)[0]
            raise Refused(
                f"{cls.__name__}.{loops[1].name}, line "
                f"{definition.lineno}: {loops[0].name} has lines "
                "besides its message loop, and so has this one: the RUN command's "
                "handler has them, a handler a message names is its loop alone; "
                f"keep the seeding and the totals in {loops[0].name}"
            )
        run = loops[0] if loops else dynamic[0]
        compiled.folded = {m.name: m for m in dynamic if m is not run}
    for number, method in handler_variants(compiled, methods, handlers):
        compiled.handlers[method.name] = number
        if method.overlay:
            compiled.overlays[method.name] = len(compiled.overlays) + 1
        names = handler_names(method)
        HandlerCompiler(compiled, method, names, plan, tensors).run()
    listed_words(compiled, methods)
    bind_fields(compiled, plan, l1)
    compiled.compute_options = class_compute(cls, compiled)
    compiled.startup = getattr(cls, "startup_given", None)
    return compiled


def declared_constants(compiled, cls, constants):
    """The class's declared constants and this program's values of them on
    `compiled` (a symbol's where the plan or the make gives it)."""
    compiled.declared = front.constants_of(cls)
    compiled.class_constants = dict(constants or {})


def prefetch_checked(handler, node, value):
    """A read stream's `prefetch=`: True or False as written (the chain's walk reads
    it, komorebi.dsl.chain_calls); anything else refused at its line."""
    if not (isinstance(value, ast.Constant) and isinstance(value.value, bool)):
        handler.refuse(
            node,
            f"{spelling('prefetch')}= is True or False as written",
            f"write {spelling('prefetch')}=False (read once the input is marked) or "
            "True (proven unwritten by the chain, or refused)",
        )


def listed_words(compiled, methods):
    """A handler whose sets are listed: its sets' words on it too (its parameters but
    the constants, one for every set), so that `Class.handler.args(...)` gives a
    call's words before the handler for its constants is picked."""
    for method in methods:
        built = [variant for _, variant in compiled.variants.get(method.name, [])]
        if not built or built[0] is method:
            continue
        first = built[0]
        if all(v.words == first.words and v.widths == first.widths for v in built):
            method.words, method.widths = first.words, first.widths
            method.returns = first.returns


def handler_variants(compiled, methods, handlers):
    """(number, method) of each handler built: a handler without constant
    parameters, or with none of its sets listed, itself (its defaults), numbered as
    the class numbers it; each listed set of one a handler of its own, named by its
    values, the first at the handler's number and the others after the class's."""
    handlers = handlers or {}
    built, after = [], len(methods) + 1
    for number, method in enumerate(methods, start=1):
        parameters = front.constant_parameters(method.function)
        sets = handlers.get(method.name)
        if not parameters or not sets:
            method.constant_set = {n: default for n, (_, default) in parameters.items()}
            if parameters:
                compiled.variants[method.name] = [(method.constant_set, method)]
            built.append((number, method))
            continue
        compiled.variants[method.name] = []
        for index, values in enumerate(sets):
            variant = copy.copy(method)
            variant.constant_set = dict(values)
            variant.name = f"{method.name}_" + "_".join(
                f"{n}{variant_word(v)}" for n, v in values.items()
            )
            variant.number = number if index == 0 else after
            after += index > 0
            compiled.variants[method.name].append((variant.constant_set, variant))
            built.append((variant.number, variant))
    user_max = abi.headers()["KM_HANDLER_USER_MAX"]
    if after - 1 > user_max:
        raise Refused(
            f"{compiled.name}: {after - 1} handlers with the listed sets of their "
            f"constant parameters, {user_max} most"
        )
    return built


def variant_word(value):
    """A constant parameter's value as a part of a C++ name."""
    return str(value).replace(".", "p").replace("-", "m")


@dataclass
class MergedPart:
    """One class of a merged program (komorebi.dsl.merged): the class, its plan, its
    example tensors, its constants' values, its handlers [(merged number, method)],
    the prefix of its tensors that take an alias of their own."""

    cls: type
    plan: dict
    tensors: dict
    constants: dict
    methods: list
    prefix: str


def compile_merged(name, parts, l1=None):
    """Compiles the handlers of several classes as ONE program: each handler's lines
    read its own class's constants, plan and tensors; a field of one name in two
    classes is one field when declared alike, else refused naming both. Returns the
    Compiled and the example tensors by the names its kernels give them."""
    methods = [method for part in parts for _, method in part.methods]
    compiled = Compiled(name, inspect.getsourcefile(methods[0].function))
    merged_fields(compiled, parts)
    compiled.field_users = field_users(methods, compiled.fields)
    for method in methods:
        if arrive_of(source_of(method.function)[0]):
            raise Refused(
                f"{name}: {method.owner.__name__}.{method.name} is a dynamic handler, "
                "whose message loop is its program's own kernels; not merged"
            )
    plan, tensors, order = {}, {}, []
    for part in parts:
        for key, value in part.plan.items():
            if plan.setdefault(key, value) != value:
                raise Refused(
                    f"{name}: the plan's {key} is {plan[key]!r} in one class and "
                    f"{value!r} in {part.cls.__name__}; one value a merged program"
                )
        declared_constants(compiled, part.cls, part.constants)
        compiled.tensors = []  # this class's tensor parameters
        for number, method in part.methods:
            compiled.handlers[method.name] = number
            if method.overlay:
                compiled.overlays[method.name] = len(compiled.overlays) + 1
            HandlerCompiler(compiled, method, handler_names(method), part.plan,
                            part.tensors).run()  # fmt: skip
        aliases = {}
        for tensor in compiled.tensors:
            example = part.tensors.get(tensor)
            if tensor in tensors and tensors[tensor] is not example:
                aliases[tensor] = f"{part.prefix}_{tensor}"
            key = aliases.get(tensor, tensor)
            if key not in tensors:
                tensors[key] = example
                order.append(key)
        for _, method in part.methods:
            merged_aliases(compiled, method.name, aliases)
    compiled.tensors = order
    compiled.declared, compiled.class_constants = {}, {}
    bind_fields(compiled, plan, l1)
    options = [(part.cls, class_compute(part.cls, compiled)) for part in parts]
    for cls, given in options[1:]:
        if given != options[0][1]:
            raise Refused(
                f"{name}: compute options {options[0][1]} of "
                f"{options[0][0].__name__} and {given} of {cls.__name__}: one compute "
                "kernel takes one set"
            )
    compiled.compute_options = options[0][1]
    startups = {id(getattr(part.cls, "startup_given", None)) for part in parts}
    if len(startups) > 1:
        raise Refused(f"{name}: the classes give different start-ups; give one")
    compiled.startup = getattr(parts[0].cls, "startup_given", None)
    return compiled, tensors


def merged_fields(compiled, parts):
    """The fields of every class of a merged program, each shape by its own class's
    constants; one name in two classes one field when declared alike."""
    owners = {}
    for part in parts:
        declared_constants(compiled, part.cls, part.constants)
        for field_name, value in class_annotations(part.cls).items():
            if not isinstance(value, front.Field):
                continue
            made = front.Field(
                value.format,
                tuple(shape_size(compiled, field_name, n) for n in value.shape),
            )
            own = inspect.getattr_static(part.cls, field_name, None)
            push = own.push if isinstance(own, front.Own) else "none"
            if field_name in owners and owners[field_name][1:] != (made, push):
                raise Refused(
                    f"{compiled.name}: field {field_name} is {owners[field_name][1]} "
                    f"in {owners[field_name][0]} and {made} in {part.cls.__name__}; "
                    "a field of one name is one field: declare it alike, or rename "
                    "one"
                )
            owners[field_name] = (part.cls.__name__, made, push)
            compiled.fields[field_name] = made
            if isinstance(own, front.Own):
                compiled.own[field_name] = own.push
    for field_name, value in list(compiled.fields.items()):
        if word_field(value):  # as compile_class: words in each core's row
            if field_name in compiled.own:
                raise field_refused(
                    compiled,
                    field_name,
                    "a field of words is the host's, not a body's own",
                    "declare it of whole tiles",
                )
            compiled.word_fields[field_name] = value.shape[0]
            del compiled.fields[field_name]


def merged_aliases(compiled, handler, aliases):
    """A merged handler whose tensors another class's share a name with: its movement
    lines in a block of their own that names each by its own alias."""
    if not aliases:
        return
    lines = [
        Line(f"using {camel_of(tensor)} [[maybe_unused]] = {camel_of(alias)};")
        for tensor, alias in aliases.items()
    ]
    compiled.movement[handler] = [
        Block("{", "}", [*lines, *compiled.movement[handler]])
    ]


def camel_of(name):
    """A tensor parameter's C++ type name (komorebi.dsl.program.camel)."""
    return "".join(part.capitalize() for part in name.split("_"))


def handler_names(method):
    """The names a handler's lines read: its module's, and its constant parameters'
    values as module constants."""
    if not method.constant_set:
        return method.function.__globals__
    return {**method.function.__globals__, **method.constant_set}


def shape_size(compiled, field_name, size):
    """One size of a field's shape: an int; a constant of the class, this program's
    value; a symbol, its name (the plan's or the build's)."""
    if not isinstance(size, front.Constant):
        return size
    if isinstance(size, front.Symbol):  # as before constants: the plan's by name
        return size.name
    try:
        value = size.value(compiled.class_constants)
    except KeyError as missing:
        raise field_refused(
            compiled,
            field_name,
            f"its size {size!r} needs {missing.args[0]}, which the build solves "
            "after the shapes",
            f"give {missing.args[0]} in the {spelling('plan')}",
        ) from None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise field_refused(
            compiled,
            field_name,
            f"its size {size!r} is {value!r}",
            "a size is an int of 1 or more",
        )
    return value


def class_compute(cls, compiled):
    """The class's compute options (`compute=`): the host API's names, each refused
    by name when unknown or when the class sets it from its lines."""
    given = getattr(cls, "compute_given", {})
    try:
        line = inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        line = "?"
    where = f"{cls.__name__}, line {line}: compute option"
    for key, value in given.items():
        if key == "unpack_to_dest_mode":
            raise Refused(
                f"{where} {key!r}: the class sets it from its float32 buffers; "
                "drop it"
            )
        if key == "math_fidelity":
            if value not in front.FIDELITIES:
                raise Refused(
                    f"{where} {key!r} = {value!r}: not a fidelity; give one of "
                    f"{', '.join(front.FIDELITIES)}"
                )
            if compiled.fidelity is not None and compiled.fidelity != value:
                raise Refused(
                    f"{where} {key!r} = {value!r}: the products ask "
                    f"{compiled.fidelity}; give one fidelity"
                )
            continue
        # The session's flags, looked up when used: the session imports this module
        flags = importlib.import_module("komorebi.session.session").COMPUTE_FLAGS
        if key not in flags:
            raise Refused(
                f"{where} {key!r}: the host API has no such option; give "
                f"math_fidelity or one of {', '.join(flags)}"
            )
        if not isinstance(value, bool):
            raise Refused(f"{where} {key!r} = {value!r}: a flag; give True or False")
        if key == "fp32_dest_acc_en" and not value:
            raise Refused(
                f"{where} {key!r} = False: the class's tiles are fp32 in DST (its "
                "sums, products and tile functions); drop it"
            )
        if key == "dst_full_sync_en" and not value and compiled.full_dst:
            raise Refused(
                f"{where} {key!r} = False: the class's tiles take DST past its half "
                f"({DST_HALF} fp32 tiles), in full sync; drop it"
            )
    return dict(given)


def field_users(methods, fields):
    """field -> (the handlers whose lines fill it, the handlers that read it and do
    not fill it), by `self.f.fill(...)` and the other `self.f` of each handler."""
    users = {name: ([], []) for name in fields}
    for method in methods:
        definition = source_of(method.function)[0]
        fills, reads = set(), set()
        for node in ast.walk(definition):
            if not (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in fields
            ):
                continue
            reads.add(node.attr)
        for node in ast.walk(definition):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == spelling("fill")
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
                and node.func.value.attr in fields
            ):
                fills.add(node.func.value.attr)
        for name in fills:
            users[name][0].append(method.name)
        for name in reads - fills:
            users[name][1].append(method.name)
    return users


def arrive_of(definition):
    """The `for message in self.arrive():` loop of a handler's def, or None."""
    for node in ast.walk(definition):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and isinstance(node.iter.func.value, ast.Name)
            and node.iter.func.value.id == "self"
            and node.iter.func.attr == spelling("arrive_messages")
        ):
            return node
    return None


def loop_alone(method):
    """Whether a dynamic handler is its message loop alone (no words, nothing
    before or after the loop but a docstring)."""
    definition = source_of(method.function)[0]
    body = [
        n
        for n in definition.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    return (
        len(definition.args.args) == 1
        and len(body) == 1
        and body[0] is arrive_of(definition)
    )


def word_field(value):
    """Whether a field is words the host reads (`km.L1[km.u32, n]`, n under a tile's
    1,024), not tiles."""
    return (
        value.format.dtype == "uint32"
        and len(value.shape) == 1
        and isinstance(value.shape[0], int)
        and 0 < value.shape[0] < TILE_ELEMENTS
    )


# The formats a product's operand may be (each side its own), by exponent family:
# float32 is held as tf32 and goes with either.
PRODUCT_FAMILIES = {
    "b": ("bfloat16", "bfp8_b", "bfp4_b", "bfp2_b"),
    "a": ("float16", "bfp8", "bfp4", "bfp2"),
}


def product_refusal(left, right):
    """None, or why a product of a `left` tile with a `right` tile is refused: each
    side a matrix format (float32 or one of PRODUCT_FAMILIES), both of one family."""
    matrix = ("float32", *PRODUCT_FAMILIES["b"], *PRODUCT_FAMILIES["a"])
    for side, dtype in (("left", left), ("right", right)):
        if dtype not in matrix:
            return (
                f"a product's {side} operand is {dtype}: the matrix unit takes "
                f"{', '.join(matrix)}"
            )
    families = {
        f for f, names in PRODUCT_FAMILIES.items() for d in (left, right) if d in names
    }
    if len(families) > 1:
        return (
            f"a product of {left} and {right}: its operands are of one exponent "
            "family (bfloat16 and bfp*_b, or float16 and bfp*; float32 goes with "
            "either)"
        )
    return None


def marked_operand(node):
    """An operand of `@`: its name and its marker ("srcb", "srca"; None: unmarked)."""
    sides = {spelling("srcb"): "srcb", spelling("srca"): "srca"}
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "km"
        and node.func.attr in sides
        and len(node.args) == 1
        and not node.keywords
    ):
        return getattr(node.args[0], "id", None), sides[node.func.attr]
    return getattr(node, "id", None), None


def product_placement(function):
    """A product tile function's placement: "none" (no marker; the left operand in
    SrcB, the right in SrcA), "plain" (marked so) or "turned" (marked the other way:
    the right operand in SrcB); a str of why when both are marked for one side."""
    value = source_of(function)[0].body[-1].value
    left, right = marked_operand(value.left)[1], marked_operand(value.right)[1]
    if left is not None and left == right:
        return (
            f"both operands are marked km.{spelling(left)}: a product has one of each"
        )
    if left is None and right is None:
        return "none"
    return "turned" if left == "srca" or right == "srcb" else "plain"


def turned_tiles(tiles, first):
    """The C++ that transposes DST tiles first.. (`tiles` of them) in place: the
    matrix unit's transposed product's sums to and from the plain layout."""
    return Line(
        "transpose_dest_init<DST_ACCUM_MODE, true>();\n"
        + "\n".join(
            f"transpose_dest<DST_ACCUM_MODE, true>({first + j});" for j in range(tiles)
        )
    )


SCALAR = "scalar"  # a tile function's argument that is a float word, not a tile


def scalar_words(unit, scalars):
    """The call's run-time scalar arguments after `dst`: each float word's bits, in
    the unit's order (`unit.scalars`: `<parameter>_bits`)."""
    return "".join(
        f", __builtin_bit_cast(uint32_t, {scalars[name.removesuffix('_bits')]})"
        for name in unit.scalars
    )


def timed_waits(nodes):
    """A stamped handler's compute lines with each wait (TIMED_WAITS) timed into
    its CPU's counter, in place."""
    for node in nodes:
        if isinstance(node, (Line, Glue)):
            node.text = timed_text(node.text)
        elif isinstance(node, Block):
            node.opening = timed_text(node.opening)
            timed_waits(node.children)
        elif isinstance(node, Group):
            timed_waits(node.children)


def timed_text(text):
    """`text` with its waits wrapped in KM_WAITED / KM_WAITED_DO."""
    for start, counter, expression in TIMED_WAITS:
        out, at = [], 0
        while (found := text.find(start, at)) >= 0:
            out.append(text[at:found])
            if not expression:
                out.append(f"KM_WAITED_DO({counter}, {start[:-1]});")
                at = found + len(start)
                continue
            depth, opened = 0, found + start.index("(")  # the call's own parenthesis
            for end in range(opened, len(text)):  # its close
                depth += {"(": 1, ")": -1}.get(text[end], 0)
                if depth == 0:
                    break
            out.append(f"KM_WAITED({counter}, {text[found : end + 1]})")
            at = end + 1
        text = "".join(out) + text[at:]
    return text


def slot_part(part):
    """A cast part's place in the slot: (first page, rows, pitch, transposed), the
    rows and pitch ints where the build knows them (None where a word)."""
    pages, pitch = part["pages"], part["pitch"]
    sides = pages.sides if pages.sides is not None else ("1", pages.count)
    bounds = pages.bounds or (None, None)
    rows, columns = (
        int(side) if str(side).isdigit() else most for side, most in zip(sides, bounds)
    )
    if part["transposed"]:  # a column of the rectangle a row of the slot
        rows, columns = columns, (pitch or rows)
    elif pitch is not None:
        columns = pitch
    return part["offset"], rows, columns, part["transposed"]


def tile_format_refusal(formats):
    """None, or why a tile function is refused tiles of `formats`: float32 and the B
    family run on an fp32 Dst, the A family on a 16-bit one (tile_dest); a mix of
    the two is refused (an fp32 Dst reads the A family 2^-15 off, a 16-bit Dst cuts
    float32 to fp16)."""
    if all(f in TILE_FORMATS for f in formats) or all(f in A_DTYPES for f in formats):
        return None
    a = [f for f in formats if f in A_DTYPES]
    other = [f for f in formats if f not in A_DTYPES]
    return (
        f"a tile function takes tiles of float32 or the B family ("
        f"{', '.join(TILE_FORMATS)}), run on an fp32 Dst, or of the A family ("
        f"{', '.join(A_DTYPES)}), run on a 16-bit Dst; given {', '.join(a)} with "
        f"{', '.join(other)}: an fp32 Dst reads the A family 2^-15 off, and a 16-bit "
        "Dst holds no float32"
    )


class SrcaSwitch:
    """A tile call's SrcA format set again before each unpack into it (a product's w,
    each copied operand) when those are not of one format: the call's setup sets it
    once, for the first of them, and the unpacker reads every later tile as that."""

    def __init__(self, products, copied):
        read = [w.buffer.dtype for _, w in products] + [c.buffer.dtype for c in copied]
        self.on = len(set(read)) > 1

    def product(self, a, w):
        if not self.on:
            return []
        return [f"KM_UNPACK_FORMATS({w.buffer.name}.mover_a, {a.buffer.name}.mover_a);"]

    def copy(self, item):
        if not self.on:
            return []
        return [f"KM_UNPACK_FORMAT_A({item.buffer.name}.mover_a);"]


def tile_dest(formats):
    """The Dst a tile function on tiles of `formats` runs on: "fp16a" for the A
    family, else "fp32"."""
    return "fp16a" if all(f in A_DTYPES for f in formats) else "fp32"


def field_refused(compiled, name, why, instead):
    """A refusal of field `name`: the class, the field, its line in the source."""
    lines = Path(compiled.source).read_text().split("\n")
    pattern = re.compile(rf"^\s*{name}\s*:")
    found = [n for n, line in enumerate(lines, start=1) if pattern.match(line)]
    where = f", line {found[0]}, `{lines[found[0] - 1].strip()}`" if found else ""
    return Refused(f"{compiled.name}.{name}{where}: {why}; {instead}")


def bind_fields(compiled, plan, l1):
    """Each field a buffer of whole tiles; its shape's symbols from the plan, and one
    symbol left unknown solved as the largest that fits the L1 the others leave."""
    unknown, user = set(), {}
    for name, value in compiled.fields.items():
        symbols = [n for n in value.shape if isinstance(n, str) and n not in plan]
        if len(symbols) != len(set(symbols)):
            raise field_refused(
                compiled, name, "a symbol twice in one shape", "give it in the plan"
            )
        unknown |= set(symbols)
        for symbol in symbols:
            user.setdefault(symbol, name)
    if len(unknown) > 1:
        raise field_refused(
            compiled,
            user[sorted(unknown)[1]],
            f"the fields leave {', '.join(sorted(unknown))} unknown and the build "
            "solves one",
            f"give the others in the {spelling('plan')}",
        )
    solved = dict(plan)
    if unknown:
        symbol = unknown.pop()
        if l1 is None:
            raise field_refused(
                compiled,
                user[symbol],
                f"{symbol} is solved against the L1 the buffers leave",
                "make the program with session=session (or l1=session.l1()[1])",
            )
        solved[symbol] = solve(compiled, symbol, plan, l1, user[symbol])
    for name, value in compiled.fields.items():
        elements = field_elements(value, solved)
        if elements % TILE_ELEMENTS:
            raise field_refused(
                compiled,
                name,
                f"{elements} items are not whole tiles ({TILE_ELEMENTS} a tile)",
                "give sizes whose product is a multiple of 1,024",
            )
        page_bytes = value.format.tile_bytes
        shape = " x ".join(
            f"{n} = {solved[n]}" if isinstance(n, str) else str(n) for n in value.shape
        )
        buffer = Buffer(
            f"kField{camel(name)}",
            "field",
            next_index(compiled),
            None,
            value.format.dtype,
            page_bytes,
            (elements // TILE_ELEMENTS,),
            1,
            f"field {name}: {value.format.dtype}, {shape}"
            + (
                "; own: used by bodies, not filled by a line"
                if name in compiled.own
                else ""
            ),
        )
        compiled.buffers.append(buffer)
        if name in compiled.own and value.format.dtype == "float32":
            compiled.unpack_fp32.add(buffer.name)  # unpacked to DST as float32
        for n in value.shape:
            if isinstance(n, str):
                compiled.constants[n] = solved[n]
    for name, count, handler, line in compiled.fills:
        tiles = field_elements(compiled.fields[name], solved) // TILE_ELEMENTS
        if isinstance(count, tuple):  # a rectangle: rows of the field's columns
            shape = compiled.fields[name].shape
            columns = solved.get(shape[1], shape[1]) if len(shape) == 4 else None
            if columns == count[1]:
                continue
            lines = Path(compiled.source).read_text().split("\n")
            raise Refused(
                f"{compiled.name}.{handler}, line {line}, `{lines[line - 1].strip()}`: "
                f"a fill of a rectangle of {count[1]} columns fills rows of the "
                f"field's, and the field is not rows of {count[1]} tiles; declare it "
                f"km.L1[format, rows, {count[1]}, 32, 32]"
            )
        if count not in (
            str(tiles),
            *(n for n in compiled.fields[name].shape if solved.get(n) == tiles),
        ):
            lines = Path(compiled.source).read_text().split("\n")
            raise Refused(
                f"{compiled.name}.{handler}, line {line}, `{lines[line - 1].strip()}`: "
                f"a fill reads the field's {tiles} tiles, and {count} is not known to "
                f"be {tiles}; write the count as {tiles} or as the field's symbol"
            )


def stream_units(read):
    """What a read stream's blocks count: (its pages, its block) and, for a rectangle
    of constant columns, (its rows, its rows a block): streams in step share one."""
    pages = read["pages"]
    units = {(pages.count, read["block"])}
    sides = pages.sides
    if (
        sides is not None
        and isinstance(sides[1], int)
        and read["block"] % sides[1] == 0
    ):
        units.add((sides[0], read["block"] // sides[1]))
    return units


def side_fits(side, bound, most):
    """Whether a rectangle's side (an int, or C++ text) writes a field's side of
    `most` tiles: the same constant, or a word the build knows is at most it."""
    if isinstance(side, int) or side.isdigit():
        return int(side) == most
    return bound is not None and bound <= most


def field_elements(value, solved):
    return math.prod(solved[n] if isinstance(n, str) else n for n in value.shape)


def solve(compiled, symbol, plan, l1, name):
    """The largest value of `symbol` for which every field is whole tiles and the
    fields and buffers fit `l1` bytes."""
    used = sum(sum(b.pages) * b.page_bytes for b in compiled.buffers)
    fixed, per = 0, 0  # without the symbol, and per unit of it: 1/1,024ths of bytes
    step = 1
    for value in compiled.fields.values():
        items = field_elements(value, {**plan, symbol: 1})
        rest = items * value.format.tile_bytes
        if symbol in value.shape:
            per += rest
            step = math.lcm(step, TILE_ELEMENTS // math.gcd(items, TILE_ELEMENTS))
        else:
            fixed += rest
    room = (l1 - used) * TILE_ELEMENTS - fixed
    value = room // per // step * step if per and room > 0 else 0
    if value < 1:
        raise field_refused(
            compiled,
            name,
            f"no room for {symbol}: the buffers take {used} B of {l1} B",
            "give smaller blocks, or a smaller field",
        )
    return value


def next_index(compiled):
    taken = {b.index_a for b in compiled.buffers} | {
        b.index_b for b in compiled.buffers
    }
    taken |= compiled.reserved
    index = FIRST_OTHER
    while index in taken:
        index += 1
    if index > LAST_INDEX:
        raise Refused(f"{compiled.name}: more buffers and fields than the indices")
    return index


def source_of(function):
    """The function's def, parsed, with its file's line numbers."""
    lines, first = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    ast.increment_lineno(tree, first - 1)
    return tree.body[0], lines, first


class HandlerCompiler:
    """One handler: its parameters, then each statement into the nodes of both
    kernels."""

    def __init__(self, compiled, method, names, plan, tensors):
        self.compiled = compiled
        self.method = method
        method.widths = {}  # tensor parameter -> its rectangles' tile width
        self.names = names
        self.plan = plan
        self.tensors = tensors
        self.definition, self.source_lines, self.first_line = source_of(method.function)
        self.scope = {}
        self.params = []
        self.counts = {}  # tensor parameter -> its page-count word name
        self.putters = []  # the movers ("A", "B") that put items: waited at the end
        self.handed = {}  # a body's node id -> the words it takes of the other mover
        self.drained = {}  # a drained stream's name -> its field, end line, fills
        self.reads = 0
        self.writes = 0
        self.arrivals = 0
        self.dst_next = 0
        self.dst_live = None  # the accumulating tile's name while DST is held
        self.carried = set()  # tiles written or reduced in a loop deeper than zeros
        self.dst_pair = None  # two DST sums held together (a second zeros), by name
        self.compute_waits = False
        self.leader_writes = set()  # reduced values written by the leader
        self.reduced = []
        self.in_leader = False
        self.loops = []  # the range loops around the statement, innermost last
        self.cast_flat = set()  # the broadcasts that cast once outside every loop
        self.movers = {}  # mover -> the broadcast it runs
        self.allow_words = False  # a body's word is read by the return only
        self.field_line = False  # a word field's words are read in a line setting one
        self.closing = []  # compute's lines after the handler's (a product's field)
        # {block a cast outside a loop brought: its mover}, whose finish (the wait for
        # compute to free the block) is at the cast
        self.finished_casts = {}
        self.opening = []  # compute's lines before them (a field's sums count)
        self.opening_b = []  # mover B's (the same count)
        self.summed = {}  # a field of sums -> its counter's C++ name
        self.filled = {}  # field -> (its fill's node, its block) until a body takes it
        self.fill_loops = {}  # field -> (the range loops around its fill, its line)
        self.fill_loop_of = {}  # field -> the range loop its stream fill stands in
        self.leader_condition = None  # the C++ of `self.rank == self.leader`, uses
        self.masked = set()  # fields last written from sums started from a row
        self.waited_fields = set()  # fields compute waited for (a product's, once)
        self.stepped = {}  # a step's state field -> its pages (a state tile a page)
        self.packed = {}  # a field compute writes -> its flag: pages pushed, not freed
        self.held_blocks = {}  # a statement list -> blocks taken there by products
        self.leader_text = "0"  # self.leader: rank 0, or a reduce group's first rank
        self.stream_loops = []  # the numbers of the read loops around the statement
        # Per read loop around the statement: its tile function, its other tile
        # lines, and the lines compute runs once before its blocks.
        self.loop_states = []
        self.expression_calls = {}  # id of a tile expression's call: (call, function)
        self.blocks = []  # the statement lists around the statement, innermost last
        self.body_moves = []  # the movers' lines of an inline body's write stream
        self.held = {}  # a statement list -> its read streams bodies take (held_stream)
        self.tables = 0  # the handler's tables of entries (entries_table)
        self.assigned = self.assigned_names()
        self.ended = {}  # a name first assigned in an if or a loop -> (kind, line)
        self.dynamic = self.arrives()

    # The source -------------------------------------------------------------------

    def text(self, node):
        """The node's source on one line: a compound statement's head, else all of
        it, each line stripped."""
        body = getattr(node, "body", None)
        last = None
        if isinstance(body, list) and body and body[0].lineno > node.lineno:
            last = body[0].lineno - 1
        last = last or getattr(node, "end_lineno", None) or node.lineno
        lines = self.source_lines[
            node.lineno - self.first_line : last - self.first_line + 1
        ]
        text = " ".join(
            line.strip() for line in lines if not line.strip().startswith("#")
        )
        for _ in range(2):  # `a, )` closes twice: its space, then its comma
            text = re.sub(r"([(\[]) | ([)\]])|,([)\]])", r"\1\2\3", text)
        return text

    def refuse(self, node, why, instead):
        text = self.text(node)
        if (node.end_lineno or node.lineno) - node.lineno > 2 and not hasattr(
            node, "body"
        ):  # a long call (an inline body): its first line
            text = self.source_lines[node.lineno - self.first_line].strip() + " ..."
        raise Refused(
            f"{self.compiled.name}.{self.method.name}, line {node.lineno}, "
            f"`{text}`: {why}; {instead}"
        )

    def same_loops(self, node, name, value, using, why, instead):
        """Refuses `using` a once-only object (`value.made_at`: a tile of DST, a
        reduce's sum, a cast block) in other loops than the line that made it: a loop
        around the one line and not the other runs it a turn more, and the movers'
        and compute's counts differ."""
        if value.made_at is None:
            return
        loops, line = value.made_at
        if loops == len(self.loops):
            return
        where = (
            f"outside the loop {using} stands in"
            if loops < len(self.loops)
            else f"inside a loop that has ended where {using} stands"
        )
        self.refuse(node, f"{name} was made at line {line}, {where}: {why}", instead)

    def once_text(self, value):
        """The reason and the right forms a refusal of a tile or a reduce's sum used
        in other loops than the line that made it gives."""
        if value.kind == "tile":
            return (
                "a tile holds DST from its zeros to its one write or reduce, which "
                "frees it, so the next turn of a loop would add into a freed tile "
                "nothing zeroed",
                "make it in the loop that writes it (`total = km.tile.zeros("
                "km.fp32)` as its first line there), or add into it in the loop and "
                "write it after the loop",
            )
        return (
            "a reduce makes one sum each time it runs, and a write or a body frees "
            "it, so a turn that follows would wait for a sum that is gone",
            "write it where it is reduced, in the same loop (under the leader's "
            "`if`), or reduce and write both outside the loop",
        )

    def carries(self, node, value):
        """True when `value` is a running total: a tile of DST made by zeros outside
        the range loop whose body writes or reduces it at `node` (a page a turn, the
        sum so far); its tiles go through a carry buffer there and are back in DST at
        the end of that loop's body."""
        if (
            value is None
            or value.kind != "tile"
            or value.made_at is None
            or value.batch is not None
            or getattr(value, "state", None) is not None
            or getattr(value, "row_of", None) is not None
            or self.dst_live != value.cxx
        ):
            return False
        if value.made_at[0] >= len(self.loops):
            return False
        if node not in self.loops[-1].get("body", ()) or self.loops[-1].get("back"):
            self.refuse(
                node,
                f"{value.cxx} is a running total (made at line {value.made_at[1]}, "
                "outside this loop): it is written or reduced once a turn, as a line "
                "of the range loop's own body",
                "move the line out of the if or inner loop into the range loop's "
                "body, or make the tile in the loop",
            )
        self.carried.add(value.cxx)
        return True

    def carry_buffer(self, node, value):
        """The fp32 buffer a running total's tiles pass through (compute's alone:
        pack writes it, unpack reads it back to DST unchanged)."""
        buffer = self.out_buffer(
            "float32",
            PAGE_BYTES["float32"],
            f"line {node.lineno}: the running total {value.cxx}, carried",
            value.width,
            f"kCarry{camel(value.cxx)}",
        )
        buffer.pages = (value.width,)
        self.compiled.unpack_fp32.add(buffer.name)
        return buffer

    def carry_lines(self, buffer, value, pop=True):
        """(the pack of the total's DST tiles into `buffer`, the read of them back
        into the same DST tiles after a new acquire, freed when `pop`)."""
        name, dst, width = buffer.name, value.dst, value.width
        packs = "".join(f"    pack_tile({dst + t}, {name});\n" for t in range(width))
        saved = (
            f"if (km_ok && compute.reserve_back({name}, {width})) {{  // the total\n"
            f"    KM_PACK_FORMAT({name});\n"
            f"{packs}"
            f"    cb_push_back({name}, {width});\n"
            "}\n"
        )
        copies = "".join(
            f"    copy_tile({name}, {t}, {dst + t});\n" for t in range(width)
        )
        loaded = (
            f"km_ok = km_ok && compute.wait_front({name}, {width});\n"
            "tile_regs_acquire();\n"
            "if (km_ok) {  // the total back in DST\n"
            f"    KM_UNPACK_FORMAT_A({name});\n"
            f"    copy_init({name});\n"
            f"{copies}" + (f"    cb_pop_front({name}, {width});\n" if pop else "") + "}"
        )
        return saved, loaded

    def carried_write(self, node, value, out):
        """A running total's page write: its tiles packed into `out` and the carry,
        DST released, then read back from the carry into the same tiles."""
        buffer = self.carry_buffer(node, value)
        saved, loaded = self.carry_lines(buffer, value)
        self.loops[-1]["back"] = (value, loaded)
        self.dst_live = None
        packs = "".join(
            f"    pack_tile({value.dst + t}, {out.name});\n" for t in range(value.width)
        )
        return (
            f"// {value.cxx} is a running total: written and carried\n"
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"if (km_ok && compute.reserve_back({out.name}, {value.width})) {{\n"
            f"    KM_PACK_FORMAT({out.name});\n"
            f"{packs}"
            f"    cb_push_back({out.name}, {value.width});\n"
            "}\n"
            f"{saved}"
            "tile_regs_release();"
        )

    def carried_reduce(self, node, value):
        """A running total's reduce: (before it, the tiles carried and read back with
        the carry kept; after the reduce's pack, the own total read back again and the
        carry freed), so the reduce's sums never enter the total."""
        buffer = self.carry_buffer(node, value)
        saved, kept = self.carry_lines(buffer, value, pop=False)
        _, loaded = self.carry_lines(buffer, value)
        before = (
            f"// {value.cxx} is a running total: carried before the reduce\n"
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"{saved}"
            "tile_regs_release();\n"
            f"{kept}"
        )
        self.loops[-1]["back"] = (value, loaded)
        self.dst_live = None
        return before

    def carried_back(self, loop):
        """A running total written or reduced in `loop`'s body: read back from its
        carry into its DST tiles as the body's last line, for the next turn."""
        back = loop.pop("back", None)
        if back is None:
            return []
        value, loaded = back
        if self.dst_live is not None:
            self.refuse(
                loop["body"][-1],
                f"{self.dst_live} holds DST at the end of the turn, where the running "
                f"total {value.cxx} comes back to it",
                f"write or reduce {self.dst_live} before the turn ends",
            )
        self.dst_live = value.cxx
        return [Line(f"// {value.cxx}, the running total, back in DST\n{loaded}")]

    def carried_end(self):
        """A running total still in DST at the handler's end (no line after its loop
        wrote it): DST released."""
        if self.dst_live is None or self.dst_live not in self.carried:
            return []
        return [
            Line(
                f"// {self.dst_live}, the running total, left: DST released\n"
                "tile_regs_commit();\n"
                "tile_regs_wait();\n"
                "tile_regs_release();"
            )
        ]

    def assigned_names(self):
        """Names assigned more than once in the handler (their C++ is not const)."""
        counts = {}
        for node in ast.walk(self.definition):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target, node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    counts[target.id] = counts.get(target.id, 0) + 1
        return {name for name, count in counts.items() if count > 1}

    # The handler -------------------------------------------------------------------

    def run(self):
        name = self.method.name
        self.compiled.lines[name] = self.definition.lineno
        if name in self.compiled.folded:  # its loop is a case of the dynamic one's
            self.compiled.words[name] = []
            self.compiled.movement[name] = []
            self.compiled.compute[name] = []
            return
        movement = [
            Glue(
                "const uint32_t km_rank = movement.me() - km::first(command);",
                {"km_rank"},
            )
        ]
        compute = [
            Glue(
                "const uint32_t km_rank = compute.me() - km::first(command);",
                {"km_rank"},
            )
        ]
        if self.dynamic:  # the ranks of the message loop: no counter tiles
            self.dynamic_start(compute)
        for nodes, me in ((movement, "movement"), (compute, "compute")):
            if not (self.dynamic and nodes is compute):
                nodes.append(
                    Glue("const uint32_t km_size = km::count(command);", {"km_size"})
                )
            for axis in ("x", "y"):
                nodes.append(
                    Glue(
                        f"const uint32_t km_self_{axis} "
                        f"= {me}.arg(kArg{axis.upper()});",
                        {f"km_self_{axis}"},
                    )
                )
            nodes.append(
                Glue(f"const uint32_t km_self_core = {me}.me();", {"km_self_core"})
            )
            for side in ("columns", "rows"):
                nodes.append(
                    Glue(
                        f"const uint32_t km_rect_{side} "
                        f"= {me}.arg(kArg{side.title()});",
                        {f"km_rect_{side}"},
                    )
                )
            nodes.append(
                Glue(
                    f"const km::Group km_group = km::group_of(command, {me}.me());",
                    {"km_group"},
                )
            )
        for index, field_ in enumerate(self.compiled.word_fields):
            where = f"km::kMessagesCArgs + {index}" if self.dynamic else str(index)
            compute.append(
                Glue(
                    f"volatile tt_l1_ptr uint32_t* km_field_{field_} = "
                    f"km::l1(compute.arg({where}));",
                    {f"km_field_{field_}"},
                )
            )
        self.rect_param = None
        self.bind_parameters(movement, compute)
        self.rect_glue(movement, compute)
        spans = None
        if self.method.stamps is not None and not self.method.overlay:
            spans = self.tile_stamps()
        elif self.method.stamps is not None:
            self.stamps()
        if not self.dynamic:
            compute.append(Glue("bool km_ok = true;", {"km_ok"}))
        if spans is not None:
            compute.append(spans[0])
        self.block_words, self.stream_word = {}, None
        marks = (len(movement), len(compute))
        self.rect_read = False
        body_m, body_c = self.statements(self.definition.body)
        if self.rect_param is not None and not (self.rect_read or self.movers):
            self.refuse(
                self.definition,
                "its rectangle is given at each submit, and it reads no self.x, "
                "self.y or broadcast",
                f"leave rect out of km.{spelling('up_to')}",
            )
        compute[marks[1] : marks[1]] = self.block_lines()
        movement[marks[0] : marks[0]] = self.block_lines()
        movement += self.opening_b + body_m + self.put_waits()
        compute += self.opening + body_c + self.carried_end()
        if spans is not None:
            if not any(call[0] == name for call in self.compiled.tile_calls):
                self.refuse(
                    self.definition.decorator_list[0],
                    f"stamps={self.method.stamps!r} of a resident handler are the "
                    "cycles of its tile function calls, and it calls none",
                    "call a tile function, or write @km.handler(overlay=True, "
                    'stamps="t")',
                )
            if self.calls_steps():
                timed_waits(body_c)
            compute.append(spans[1])
            movement.append(spans[2])
        for kept, filled in self.filled.items():
            if filled is not None and not (
                kept in self.waited_fields or self.kept_field(kept)
            ):
                self.refuse(
                    filled[0],
                    f"self.{kept} is filled and no km.cxx body takes it",
                    f"take it: km.cxx(..., takes=[self.{kept}])",
                )
        for value in self.reduced:
            if value not in self.leader_writes:
                self.refuse(
                    self.definition,
                    f"the reduce's {value} is not written by the leader",
                    f"write it under `if self.{spelling('rank')} == "
                    f"self.{spelling('leader')}:`",
                )
        self.compiled.movement[name] = movement
        self.compiled.compute[name] = compute + self.closing

    def stamps(self):
        """`stamps="t"`: the overlay's resident case on each mover writes the
        command's sequence, the overlay, its load and call cycles and whether it was
        held to page 2 x core + mover of tensor parameter t (mover A 0, B 1)."""
        stamps = self.method.stamps
        decorator = self.definition.decorator_list[0]
        found = [p for p in self.params if p.name == stamps and p.kind == "tensor"]
        if not self.method.overlay or not found:
            self.refuse(
                decorator,
                f"stamps={stamps!r} names no tensor parameter of an overlay handler",
                'write @km.handler(overlay=True, stamps="t") with t: km.Tensor',
            )
        dtype, page_bytes = self.tensor_format(decorator, stamps)
        if page_bytes < 20 or page_bytes % 4:
            self.refuse(
                decorator,
                f"a stamps page is five words; {stamps}'s page is {page_bytes} B",
                f"make {stamps} a tensor of uint32 pages of at least five words",
            )
        buffer = self.other_buffer(
            "kStamps", "word", dtype, page_bytes, 2, "stamps: a page a mover"
        )  # fmt: skip
        buffer.page_bytes = max(buffer.page_bytes, page_bytes)
        self.compiled.stamps[self.method.name] = (stamps, found[0].word)
        self.moved(decorator, stamps, "page write")

    def calls_steps(self):
        """Whether the handler calls a step (`f(x, self.s[r])`, f a tile function, a
        field indexed second): its stamps then add the three CPUs' words."""
        return any(
            isinstance(node, ast.Call)
            and self.tile_function_of(node) is not None
            and len(node.args) == 2
            and isinstance(node.args[1], ast.Subscript)
            and self.field_name(node.args[1].value) is not None
            for node in ast.walk(self.definition)
        )

    def tile_stamps(self):
        """`stamps="t"` of a resident handler: the math CPU's cycles of each tile
        function call to page `rank` of tensor parameter t (the sequence, the calls,
        the empty spans, each call's span); compute's opening and closing lines and
        mover B's write of the page."""
        stamps = self.method.stamps
        decorator = self.definition.decorator_list[0]
        found = [p for p in self.params if p.name == stamps and p.kind == "tensor"]
        if not found:
            self.refuse(
                decorator,
                f"stamps={stamps!r} names no tensor parameter",
                f'write @km.handler(stamps="t") with t: km.{spelling("tensor")}',
            )
        dtype, page_bytes = self.tensor_format(decorator, stamps)
        if page_bytes < 4 * (SPANS_FIRST + 1) or page_bytes % 4:
            self.refuse(
                decorator,
                f"a stamps page holds {SPANS_FIRST + 1} words or more; {stamps}'s "
                f"page is {page_bytes} B",
                f"make {stamps} a tensor of uint32 pages of {SPANS_FIRST + 1} words "
                "or more",
            )
        buffer = self.other_buffer(
            "kSpans", "word", dtype, page_bytes, 1, "stamps: the math CPU's spans"
        )  # fmt: skip
        steps = self.calls_steps()  # then the three CPUs' words end the page
        least = SPANS_FIRST + 1 + CPU_WORDS
        if steps and page_bytes < 4 * least:
            self.refuse(
                decorator,
                f"a stamps page of a handler of steps holds {least} words or more "
                f"(the compute CPUs' {CPU_WORDS} at its end); {stamps}'s page is "
                f"{page_bytes} B",
                f"make {stamps} a tensor of uint32 pages of {least} words or more",
            )
        self.spans_most = page_bytes // 4 - SPANS_FIRST - (CPU_WORDS if steps else 0)
        cpus = page_bytes // 4 - CPU_WORDS
        self.moved(decorator, stamps, "page write")
        self.wait(decorator, "mover B", WAIT_DATA)
        waits = (
            "\n// each compute CPU: its span and its waits (CPU_WORDS)\n"
            "#define KM_WAITED(counter, call) ({ const uint32_t km_w0 = km::wall(); "
            "const bool km_r = (call); counter += km::wall() - km_w0; km_r; })\n"
            "#define KM_WAITED_DO(counter, call) do { const uint32_t km_w0 = "
            "km::wall(); call; counter += km::wall() - km_w0; } while (0)\n"
            "uint32_t km_waited_in = 0;\n"
            "uint32_t km_waited_out = 0;\n"
            "const uint32_t km_opened = km::wall();"
            if steps
            else ""
        )
        opening = Group(
            f"stamps: the math CPU's cycles of each tile function call, to {stamps} "
            "page rank",
            [
                Line(
                    "const uint32_t km_spans_page "
                    f"= get_tile_address({buffer.name}, 0);\n"
                    "MATH(volatile tt_l1_ptr uint32_t* km_spans "
                    "= km::l1(km_spans_page));\n"
                    "MATH(km_spans[0] = command.sequence);\n"
                    "MATH(km_spans[1] = 0);\n"
                    "for (uint32_t km_sync = 0; "
                    f"km_sync < {SPANS_EMPTY}; km_sync++) {{\n"
                    "    MATH(const uint32_t km_called = km::wall());\n"
                    "    MATH(tensix_sync());\n"
                    "    MATH(km_spans[2 + km_sync] = km::wall() - km_called);\n"
                    "}" + waits
                )
            ],
        )
        words = "\n".join(
            f"{cpu}(km::l1(km_spans_page)[{cpus + 3 * at} + {k}] = {value});"
            for at, cpu in enumerate(("UNPACK", "MATH", "PACK"))
            for k, value in enumerate(
                ("km::wall() - km_opened", "km_waited_in", "km_waited_out")
            )
        )
        closing = Group(
            f"stamps: the page to mover B, for {stamps} page rank",
            [
                *([Line(words)] if steps else []),
                Line(
                    f"if (km_ok && compute.reserve_back({buffer.name}, 1)) {{\n"
                    f"    cb_push_back({buffer.name}, 1);\n"
                    "}",
                    {"km_ok"},
                ),
            ],
        )
        tensor = f"{camel(stamps)}({stamps})"
        write = Group(
            f"stamps: compute's page to {stamps} page rank",
            [
                Line(
                    f"if (!movement.wait_front({buffer.name}, 1) ||\n"
                    f"    !km::write_pages(movement, command, {tensor}, km_rank, 1,\n"
                    f"                     get_read_ptr({buffer.name}))) {{\n"
                    "    return false;\n"
                    "}\n"
                    "noc_async_write_barrier();\n"
                    f"cb_pop_front({buffer.name}, 1);",
                    {"km_rank", stamps},
                    mover="B",
                )
            ],
        )
        self.compiled.includes.add("ask")
        return opening, closing, write

    def arrives(self):
        """Whether the handler has `for message in self.arrive():` (it is dynamic:
        compute's lines alone, the runtime's movers carry the messages)."""
        for node in ast.walk(self.definition):
            if isinstance(node, ast.For) and self.self_call(
                node.iter, "arrive_messages"
            ):
                return True
        return False

    def self_call(self, node, kind):
        """Whether `node` is the call `self.<kind's spelling>(...)`."""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == spelling(kind)
        )

    def dynamic_start(self, compute):
        """A dynamic handler: one in a class, no moves; its message loop's state, the
        rank and size of the loop's ranks."""
        if self.compiled.dynamic is not None:
            self.refuse(
                self.definition,
                f"{self.compiled.dynamic} is the class's dynamic handler already",
                "make one class for each dynamic handler",
            )
        self.compiled.dynamic = self.method.name
        for node in ast.walk(self.definition):
            moving = isinstance(node, ast.Call) and (
                self.construct(node) in ("stream", "reduce", "all_reduce", "broadcast")
                or isinstance(node.func, ast.Attribute)
                and node.func.attr in (spelling("push"), spelling("fill"))
                or any(k.arg == spelling("on") for k in node.keywords)
            )
            if moving:
                self.refuse(
                    node,
                    "a dynamic handler is compute's lines alone: the runtime's movers "
                    "carry its messages",
                    "move pages in a handler of another class",
                )
        compute.insert(
            0, Glue("km::Messages km_messages(compute, command);", {"km_messages"})
        )
        compute[1] = Glue(
            "const uint32_t km_rank = km_messages.rank();", {"km_rank"}, {"km_messages"}
        )
        compute.append(
            Glue(
                "const uint32_t km_size = km_messages.count();",
                {"km_size"},
                {"km_messages"},
            )
        )

    def message_class(self, node, call):
        """The km.Message class `call` names, or None."""
        if not isinstance(call, ast.Call):
            return None
        func = call.func
        found = self.names.get(func.id) if isinstance(func, ast.Name) else None
        if isinstance(found, type) and issubclass(found, front.Message):
            return found
        return None

    def made_message(self, node, name, cls, call):
        """`message = Visit(id=..., ...)`: a message of named words, the others 0."""
        if not self.dynamic:
            self.refuse(node, "a message is a dynamic handler's", "send it from one")
        if call.args or any(k.arg not in cls.words for k in call.keywords):
            self.refuse(
                node,
                f"a {cls.__name__} is made of its named words",
                f"write {cls.__name__}({', '.join(f'{w}=...' for w in cls.words)})",
            )
        self.messages_of(cls)
        self.scope[name] = Value("message", name, words=cls.words)
        text = [f"uint32_t {name}[kMessageWords] = {{}};"]
        uses = set()
        for given in call.keywords:
            word = f"k{cls.__name__}{camel(given.arg)}"
            text.append(f"{name}[{word}] = {self.integer(given.value)};")
            uses |= self.uses(given.value)
        return [], [Line("\n".join(text), uses)]

    def messages_of(self, cls):
        """The class's message words, for the header."""
        known = self.compiled.messages.setdefault(cls.__name__, cls.words)
        if known != cls.words:
            self.refuse(self.definition, f"two {cls.__name__}s", "name one")

    def message_word(self, node):
        """The C++ of `message.word` (a word of a message), or None."""
        base = node.value
        if not (isinstance(base, ast.Name) and base.id in self.scope):
            return None
        value = self.scope[base.id]
        if value.kind != "message":
            return None
        if node.attr not in value.words:
            self.refuse(
                node,
                f"{base.id} has no word {node.attr}",
                f"use one of {', '.join(value.words)}",
            )
        cls = next(c for c, w in self.compiled.messages.items() if w is value.words)
        return f"{base.id}[k{cls}{camel(node.attr)}]"

    def arrive_loop(self, node):
        """`for message in self.arrive():` each message given to this core, in packets
        (copied from the queue), each then forwarded, consumed or left."""
        if not isinstance(node.target, ast.Name):
            self.refuse(node, "one message name", "write for message in self.arrive()")
        args = node.iter.args
        cls = None
        if len(args) == 1:
            cls = self.message_class(node, ast.Call(func=args[0], args=[], keywords=[]))
        elif not args:
            known = [
                v for v in self.names.values()
                if isinstance(v, type) and issubclass(v, front.Message)
                and v is not front.Message
            ]  # fmt: skip
            cls = known[0] if len(known) == 1 else None
        if cls is None:
            self.refuse(
                node,
                "the loop's message class is not known",
                "write self.arrive(Visit), Visit a km.Message of the module",
            )
        name = node.target.id
        self.check_name(node, name)
        self.messages_of(cls)
        self.scope[name] = Value("message", name, words=cls.words)
        _, body = self.statements(node.body)
        self.scope.pop(name)
        if self.compiled.folded:
            body = [self.handler_cases(name, cls, body)]
        copy = (
            "const volatile tt_l1_ptr uint32_t* km_incoming = km_packet[km_arrived];\n"
            f"uint32_t {name}[kMessageWords] = {{}};  // word 0: the runtime's\n"
            "for (uint32_t km_word = 1; km_word < kMessageWords; km_word++) {\n"
            f"    {name}[km_word] = km_incoming[km_word];\n"
            "}"
        )
        inner = Block(
            "for (uint32_t km_arrived = 0; "
            "km_arrived < km_packet.count; km_arrived++) {",
            "}",
            [Line(copy), *body],
        )
        outer = Block(
            "while (km::Packet km_packet = km_messages.next()) {",
            "}",
            [inner, Line("km_messages.done(km_packet);")],
            {"km_messages"},
        )
        return [], [outer]

    def handler_cases(self, name, cls, body):
        """The message loop's body of a class of several dynamic handlers: a case of
        each, by the handler the message names (its own loop's, compiled here with its
        own lines and names); a message of no handler of the class is consumed."""
        cases = [
            Block(
                f"case kHandler{camel(self.method.name)}: {{",
                "}",
                [*body, Line("break;")],
            )
        ]
        for folded in self.compiled.folded.values():
            saved = (
                self.method,
                self.definition,
                self.source_lines,
                self.first_line,
                self.scope,
            )
            self.method = folded
            self.definition, self.source_lines, self.first_line = source_of(
                folded.function
            )
            self.scope = {}
            try:
                loop = arrive_of(self.definition)
                if not isinstance(loop.target, ast.Name):
                    self.refuse(
                        loop, "one message name", "write for m in self.arrive()"
                    )
                self.check_name(loop, loop.target.id)
                self.scope[loop.target.id] = Value("message", name, words=cls.words)
                if loop.target.id != name:
                    self.refuse(
                        loop,
                        f"the loops of one class's dynamic handlers name their "
                        f"message alike ({name})",
                        f"write for {name} in self.arrive()",
                    )
                _, lines = self.statements(loop.body)
            finally:
                (
                    self.method,
                    self.definition,
                    self.source_lines,
                    self.first_line,
                    self.scope,
                ) = saved
            cases.append(
                Block(
                    f"case kHandler{camel(folded.name)}: {{",
                    "}",
                    [*lines, Line("break;")],
                )
            )
        cases.append(Line("default:\n    km_messages.consume();", {"km_messages"}))
        return Block("switch (km_messages.handler()) {", "}", cases, {"km_messages"})

    def message_call(self, node, call, kind):
        """self.send(rank, message), self.forward(rank, message), self.consume()."""
        if not self.dynamic:
            self.refuse(node, "a message call is a dynamic handler's", "use one")
        if kind == "consume":
            if call.args or call.keywords:
                self.refuse(node, "consume() takes nothing", "write self.consume()")
            return [], [Line("km_messages.consume();", {"km_messages"})]
        arguments, uses = self.message_arguments(node, call, kind)
        return [], [Line(f"km_messages.{kind}({arguments});", uses)]

    def direct_call(self, node, name, call):
        """`went = self.direct(rank, message)`: 1 when the message went into its slot
        at rank (a declared pair), 0 when the slot was busy and nothing was sent."""
        if not self.dynamic:
            self.refuse(node, "a message call is a dynamic handler's", "use one")
        arguments, uses = self.message_arguments(node, call, "direct_message")
        text = f"(km_messages.direct({arguments}) ? 1u : 0u)"
        if name in self.scope:
            if self.scope[name].kind != "int" or name not in self.assigned:
                self.refuse(node, f"{name} is already defined", "use a new name")
            return [], [Line(f"{name} = {text};", uses | {name})]
        self.scope[name] = Value("int", name)
        declared = "uint32_t" if name in self.assigned else "const uint32_t"
        # a line, not glue: the call sends, so it stays when the name goes unread
        return [], [Line(f"{declared} {name} = {text};", uses)]

    def direct_unnamed(self, node):
        """self.direct(...) anywhere but as a name's value: refused, by its bool."""
        self.refuse(
            node,
            "self.direct(rank, message) says whether the message went (0: its slot "
            "is busy and nothing was sent), so its value is a name's",
            "write went = self.direct(rank, message), then forward it when went == 0",
        )

    def message_arguments(self, node, call, kind):
        """The C++ arguments of self.<kind>(rank, message[, to=]) and their uses."""
        to = [k for k in call.keywords if k.arg == spelling("to")]
        if len(call.args) != 2 or len(call.keywords) != len(to) or len(to) > 1:
            self.refuse(
                node,
                "a rank and a message (and the handler there, to=)",
                f"write self.{spelling(kind)}(rank, m) or "
                f"self.{spelling(kind)}(rank, m, to={self.compiled.name}.handler)",
            )
        message = call.args[1]
        if not (
            isinstance(message, ast.Name)
            and self.scope.get(message.id, Value("")).kind == "message"
        ):
            self.refuse(node, "the second is a message", "give a message's name")
        rank = self.integer(call.args[0])
        uses = self.uses(call.args[0]) | {"km_messages", message.id}
        handler = f", {self.handler_named(node, to[0].value)}" if to else ""
        return f"{rank}, {message.id}{handler}", uses

    def handler_named(self, node, value):
        """The C++ of `to=Class.handler`: a dynamic handler of this class."""
        named = (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == self.compiled.name
            and value.attr in (*self.compiled.folded, self.compiled.dynamic)
        )
        if not named:
            self.refuse(
                node,
                f"to= names a dynamic handler of {self.compiled.name}",
                f"write to={self.compiled.name}.<handler>, a handler with "
                "`for message in self.arrive():`",
            )
        return f"kHandler{camel(value.attr)}"

    def seeding_branch(self, node):
        """`if self.seeding():` the first messages, then the loop is told they are
        all sent."""
        if node.orelse:
            self.refuse(node, "seeding has no else", "write the if alone")
        _, body = self.statements(node.body)
        return [], [
            Block(
                "if (km_messages.seeding()) {",
                "}",
                [*body, Line("km_messages.seeded();")],
                {"km_messages"},
            )
        ]

    def bind_parameters(self, movement, compute):
        """Each parameter after self is a command word: a tensor's reference word, an
        int, a pattern's word; a tensor whose pages are counted gets a word more."""
        args = self.definition.args
        constants = self.constant_parameters(args)
        if (
            args.vararg
            or args.kwarg
            or args.kwonlyargs
            or (
                len(args.defaults)
                > len(constants) + len(front.sized_parameters(self.method.function))
            )
        ):
            self.refuse(
                self.definition,
                "a handler's parameters are plain, one a command word",
                "write `def name(self, a: km.Tensor, b: int, ...)`",
            )
        parameters = [p for p in args.args[1:] if p.arg not in constants]
        words = []
        for parameter in parameters:
            word = len(words)  # a rectangle takes four
            kind = self.parameter_kind(parameter)
            self.check_name(parameter, parameter.arg)
            param = Param(parameter.arg, kind, word)
            self.params.append(param)
            words.append((parameter.arg, kind))
            if kind == "tensor":
                self.scope[param.name] = Value("tensor", param.name)
                if param.name not in self.compiled.tensors:
                    self.compiled.tensors.append(param.name)
                text = f"const uint32_t {param.name} = command.a{word};"
            elif kind == "region":
                text = self.bind_region(parameter, word)
            elif kind == "pattern":
                self.scope[param.name] = Value("pattern", param.name)
                text = (
                    f"const km::Pattern {param.name} = "
                    f"km::Pattern::of(command.a{word});"
                )
            elif kind == "rect":
                text = self.bind_rect(param.name, word, words)
            elif kind == "float":
                self.scope[param.name] = Value("float", param.name)
                text = (
                    f"const float {param.name} = "
                    f"__builtin_bit_cast(float, command.a{word});"
                )
            else:
                self.scope[param.name] = Value("int", param.name)
                text = f"const uint32_t {param.name} = command.a{word};"
            for nodes in (movement, compute):
                nodes.append(Glue(text, {param.name}))
        if front.rect_given(self.method.function) is not None:  # km.UPTO(rect=)
            word = len(words)
            words.append(("rect", "rect"))
            text = self.bind_rect("rect", word, words, "km_rect_given")
            for nodes in (movement, compute):
                nodes.append(Glue(text, {"km_rect_given"}))
        counted = self.counted_tensors()
        for tensor in counted:
            word = len(words)
            words.append((tensor, "pages"))
            self.counts[tensor] = f"{tensor}_pages"
            for nodes in (movement, compute):
                nodes.append(
                    Glue(
                        f"const uint32_t {tensor}_pages = command.a{word};",
                        {f"{tensor}_pages"},
                    )
                )
        self.column_words = {}
        sized = front.sized_parameters(self.method.function)
        for tensor in [n for n, kind in words if kind == "tensor" and n in sized]:
            word = len(words)  # its tile columns, the tensor's at each submit
            words.append((tensor, "columns"))
            self.column_words[tensor] = f"{tensor}_columns"
            for nodes in (movement, compute):
                nodes.append(
                    Glue(
                        f"const uint32_t {tensor}_columns = command.a{word};",
                        {f"{tensor}_columns"},
                    )
                )
        for name in [n for n, kind in words if kind == "region"]:  # its bank's word
            word = len(words)
            words.append((name, "bank"))
            for nodes in (movement, compute):
                nodes.append(Glue(f"{name}.bank = command.a{word};", {name}))
        if len(words) > COMMAND_WORDS:
            self.refuse(
                self.definition,
                f"{len(words)} command words ({', '.join(n for n, _ in words)}); a "
                f"command has {COMMAND_WORDS}",
                "pass fewer arguments, or a tensor of them",
            )
        self.method.words = words
        self.compiled.words[self.method.name] = words

    def constant_parameters(self, args):
        """The handler's constant parameters (km.constant[int] = n): each with its
        default, after the command words."""
        constants = front.constant_parameters(self.method.function)
        defaulted = [p.arg for p in args.args[len(args.args) - len(args.defaults) :]]
        for parameter in args.args[1:]:
            if parameter.arg in constants and parameter.arg not in defaulted:
                self.refuse(
                    parameter,
                    f"{parameter.arg} is a constant parameter: the handler is built "
                    "for its default when no set of it is listed",
                    f"give it a default, {parameter.arg}: "
                    f"km.{spelling('constant')}[int] = n",
                )
        sized = front.sized_parameters(self.method.function)
        if any(name not in constants and name not in sized for name in defaulted):
            return {}  # a default of a command word: refused by bind_parameters
        return constants

    def parameter_kind(self, parameter):
        annotation = parameter.annotation
        if annotation is None:
            self.refuse(
                self.definition,
                f"parameter {parameter.arg} has no type",
                f"write `{parameter.arg}: km.{spelling('tensor')}`, `: int` or "
                f"`: km.{spelling('pattern')}`",
            )
        if isinstance(annotation, ast.Name) and annotation.id in ("int", "float"):
            return annotation.id
        most = front.sized_parameters(self.method.function).get(parameter.arg)
        if isinstance(annotation, ast.Name) and isinstance(most, tuple):
            return "rect"  # a rectangle given at each submit: four words
        if isinstance(annotation, ast.Subscript) and isinstance(most, int):
            return "int"  # a constant int km.UPTO gives at each submit: a word
        if isinstance(annotation, ast.Attribute):
            kind = kind_of(annotation.attr, ("tensor", "pattern", "region"))
            if kind:
                return kind
        self.refuse(
            self.definition,
            f"parameter {parameter.arg}: a type the command words cannot carry",
            f"a parameter is km.{spelling('tensor')}, int, float or "
            f"km.{spelling('pattern')}",
        )

    def bind_rect(self, name, word, words, text=None):
        """A rectangle given at each submit: its four words (x0, y0, columns, rows)
        as an array (named `text` in C++, else `name`), three more words after its
        place (`words` the handler's)."""
        text = text or name
        if self.rect_param is not None:
            self.refuse(
                self.definition,
                f"two rectangles given at each submit ({self.rect_param}, {name})",
                "give one",
            )
        self.rect_param = text
        self.compiled.rect_sized = True
        words += [(name, "rect words")] * 3
        places = ", ".join(f"command.a{word + i}" for i in range(4))
        return f"const uint32_t {text}[4] = {{{places}}};"

    def rect_glue(self, movement, compute):
        """A handler with a rectangle given at each submit: self.x, self.y, the
        rectangle's sides and whether the core is in it computed on the core
        (km_rect.hpp) from its words and the grid's (program arguments)."""
        if self.rect_param is None:
            return
        for nodes, me in ((movement, "movement"), (compute, "compute")):
            nodes.append(
                Glue(
                    "uint32_t km_grid_words[kGridWords];\n"
                    "for (uint32_t i = 0; i < kGridWords; i++) {\n"
                    f"    km_grid_words[i] = {me}.arg(kArgGrid + i);\n"
                    "}\n"
                    "const km::RectGrid km_grid = km::rect_grid(km_grid_words);",
                    {"km_grid"},
                )
            )
            nodes.append(
                Glue(
                    f"const km::RectPlace km_place = km::rect_place({me}.me(), "
                    f"km_grid.width, {self.rect_param});",
                    {"km_place"},
                    {"km_grid", self.rect_param},
                )
            )
            fields = {"km_self_x": "x", "km_self_y": "y"}
            fields |= {"km_rect_columns": "columns", "km_rect_rows": "rows"}
            nodes[:] = [  # the program arguments' place words: here the command's
                n
                for n in nodes
                if not (isinstance(n, Glue) and n.defines <= set(fields))
            ]
            for value, field_ in fields.items():
                nodes.append(
                    Glue(
                        f"const uint32_t {value} = km_place.{field_};",
                        {value},
                        {"km_place"},
                    )
                )

    def included_test(self, me):
        """The C++ of `the core is outside the rectangle`."""
        if self.rect_param is not None:
            return "km_place.included == 0", {"km_place"}
        return f"{me}.arg(kArgIncluded) == 0", set()

    def bind_region(self, parameter, word):
        """A region parameter: a tensor of the handler's lines in pages of the thread's
        granularity, its bank address this word and its bank a word after the rest."""
        name = parameter.arg
        example = self.tensors.get(name)
        islands = {
            len(t.placement.parts)
            for t in self.tensors.values()
            if hasattr(t, "placement") and hasattr(t.placement, "parts")
        }
        if islands and max(islands) > 1:
            island = getattr(example, "island", 0)
            self.refuse(
                parameter,
                f"{name} is a region of island {island}'s DRAM, and the program is "
                f"made with island tensors: another island's core would read it",
                "a region is read by its own island's cores only; make the program "
                "with this island's tensors",
            )
        self.scope[name] = Value("tensor", name)
        if name not in self.compiled.regions:
            self.compiled.regions.append(name)
        return f"km::Region {name}{{command.a{word}, 0}};"

    def counted_tensors(self):
        """Tensor parameters whose `pages` the handler uses whole (not called)."""
        tensors = {p.name for p in self.params if p.kind == "tensor"}
        called = set()
        for node in ast.walk(self.definition):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.add(id(node.func))
        found = []
        for node in ast.walk(self.definition):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == spelling("pages")
                and isinstance(node.value, ast.Name)
                and node.value.id in tensors
                and id(node) not in called
                and node.value.id not in found
            ):
                found.append(node.value.id)
        return found

    def check_name(self, node, name):
        if (
            name in GENERATED
            or name.startswith(GENERATED_PREFIX)
            or keyword.iskeyword(name)
            or name in CXX_KEYWORDS
        ):
            self.refuse(
                node,
                f"the name {name} is taken by the generated C++ (its own names start "
                f"{GENERATED_PREFIX}, and {', '.join(sorted(GENERATED))} are the "
                "runtime's)",
                "give it another name",
            )

    # Statements --------------------------------------------------------------------

    def statements(self, body):
        """The movement nodes and the compute nodes of a list of statements."""
        movement, compute = [], []
        self.blocks.append(id(body))
        for statement in body:
            m, c = self.statement(statement)
            comment = f"line {statement.lineno}: {self.text(statement)}"
            if statement.end_lineno - statement.lineno > 2 and not hasattr(
                statement, "body"
            ):  # a long call (an inline body): its first line
                first = self.source_lines[statement.lineno - self.first_line].strip()
                comment = f"line {statement.lineno}: {first} ..."
            if m:
                movement.append(Group(comment, m))
            if c:
                compute.append(Group(comment, c))
        compute += self.held_ends(body)
        compute += self.held_block_ends(body)
        self.blocks.pop()
        return movement, compute

    def statement(self, node):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            return [], []  # a docstring
        if isinstance(node, ast.Pass):
            return [], []
        if isinstance(node, ast.Assign):
            return self.assign(node)
        if isinstance(node, ast.AugAssign):
            return self.augmented(node)
        if isinstance(node, ast.For):
            return self.loop(node)
        if isinstance(node, ast.With):
            return self.drained_with(node)
        if isinstance(node, ast.If):
            return self.branch(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return self.call_statement(node, node.value)
        if isinstance(node, ast.Return):
            return self.returned(node)
        self.refuse(
            node,
            "not a line of a handler at level 1",
            "write a move, an int, a loop over a stream or a range, an if, or "
            f"km.{spelling('cxx')}(...)",
        )

    def assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            if self.tile_function_of(node.value) is not None:
                return self.tile_results(node, node.targets[0], node.value)
            return self.read_words(node, node.targets[0], node.value)
        if len(node.targets) == 1 and self.word_name(node.targets[0]):
            return self.set_field_word(node, node.targets[0])
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
            return self.set_word(node, node.targets[0], self.integer(node.value))
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self.refuse(node, "one name a line", "assign one name")
        name = node.targets[0].id
        self.check_name(node, name)
        self.not_parameter(node, name)
        value = node.value
        made = self.message_class(node, value)
        if made is not None:
            return self.made_message(node, name, made, value)
        value = self.tile_expression(node, value) or value
        if self.step_of_field(value):
            return self.started_from_field(node, name, value)
        if self.tile_function_of(value) is not None:
            return self.tile_call(node, value, name)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == spelling("read")
            and not value.args
            and not value.keywords
            and self.is_range(value.func.value)
        ):
            return self.read_word(node, name, self.range_of(value.func.value))
        if self.entries_call(value) is not None:
            return self.entries_made(node, name, value)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == spelling("bump")
            and self.field_name(value.func.value)
        ):
            return self.bump_line(node, name, value)
        if self.self_call(value, "direct_message"):
            return self.direct_call(node, name, value)
        call = self.construct(value)
        if call == "partition":
            return self.partition(node, name, value)
        if call == "zeros":
            return self.zeros(node, name, value)
        if call == "stream" and self.field_name(value.args[0] if value.args else None):
            return self.drained_stream(node, name, value)
        if call == "stream" and not self.write_keyword(value):
            return self.held_stream(node, name, value)
        if call == "stream":
            return self.write_stream(node, name, value)
        if call == "reduce":
            return self.reduce(node, name, value)
        if call == "all_reduce":
            return self.reduce(node, name, value, everywhere=True)
        if call == "broadcast":
            return self.broadcast(node, name, value)
        if call == "cxx":
            return self.inline(node, value, name)
        if call:
            self.refuse(
                node,
                f"km.{spelling(call)} gives no value to name",
                "write it on a line of its own",
            )
        if self.is_range(value):
            self.scope[name] = self.range_of(value)
            return [], []
        text = self.integer(value)
        declared = "uint32_t" if name in self.assigned else "const uint32_t"
        if name in self.scope:
            if self.scope[name].kind != "int" or name not in self.assigned:
                self.refuse(node, f"{name} is already defined", "use a new name")
            line = f"{name} = {text};"
            uses = self.uses(value) | {name}
            return [Line(line, uses)], [Line(line, uses)]
        self.scope[name] = Value("int", name)
        glue = f"{declared} {name} = {text};"
        uses = self.uses(value)
        return [Glue(glue, {name}, uses)], [Glue(glue, {name}, uses)]

    def augmented(self, node):
        target = node.target
        if self.word_name(target):
            op = OPERATORS.get(type(node.op))
            if op is None:
                self.refuse(node, "an operator level 1 has no C++ for", "use + - *")
            return self.set_field_word(node, target, op)
        field_ = self.field_name(target)
        if field_ is not None and self.tile_function_of(node.value) is not None:
            return self.field_product(node, field_, node.value)
        if isinstance(target, ast.Attribute):
            op = OPERATORS.get(type(node.op))
            if op is None:
                self.refuse(node, "an operator level 1 has no C++ for", "use + - *")
            return self.set_word(node, target, self.integer(node.value), op)
        if isinstance(target, ast.Name):
            self.not_parameter(node, target.id)
            self.not_ended(node, target.id)
        if not isinstance(target, ast.Name) or target.id not in self.scope:
            self.refuse(node, "an unknown name", "assign it first")
        value = self.scope[target.id]
        if value.kind == "tile" and self.tile_function_of(node.value) is not None:
            return self.product(node, value, node.value)
        if value.kind == "tile":
            return self.accumulate(node, value)
        if value.kind == "word" and value.mover in ("A", "B"):
            return self.word_sum(node, value)
        if value.kind != "int":
            self.refuse(node, f"{target.id} is a {value.kind}", "add to an int")
        op = OPERATORS.get(type(node.op))
        if op is None:
            self.refuse(node, "an operator level 1 has no C++ for", "use + - * etc.")
        text = f"{target.id} {op}= {self.integer(node.value)};"
        uses = self.uses(node.value) | {target.id}
        return [Line(text, uses)], [Line(text, uses)]

    def word_name(self, node):
        """The word field `node` indexes (`self.f[i]`), or None."""
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
            and node.value.attr in self.compiled.word_fields
        ):
            return node.value.attr
        return None

    def field_word(self, node):
        """The C++ of `self.f[i]`, word i of word field f in this core's L1 (its
        index checked when it is a constant)."""
        name = self.word_name(node)
        index = self.integer(node.slice)
        count = self.compiled.word_fields[name]
        if index.rstrip("u").isdigit() and int(index.rstrip("u")) >= count:
            self.refuse(
                node,
                f"self.{name} has {count} words, 0 to {count - 1}",
                "index one of them",
            )
        return f"km_field_{name}[{index}]", self.uses(node.slice) | {f"km_field_{name}"}

    def set_field_word(self, node, target, op=""):
        """`self.f[i] = value` (or `+=` and the others): a word the host reads after
        the run, written by the unpack CPU alone (one write, not three); the value
        may read the field's words."""
        word, uses = self.field_word(target)
        self.field_line = True
        try:
            value = self.integer(node.value)
        finally:
            self.field_line = False
        uses |= self.uses(node.value)
        uses |= {
            f"km_field_{n}"
            for n in self.compiled.word_fields
            if self.reads_word(node, n)
        }
        return [], [Line(f"UNPACK(({word} {op}= {value}));", uses)]

    def reads_word(self, node, name):
        """Whether the value of `node` reads word field `name`."""
        return any(self.word_name(item) == name for item in ast.walk(node.value))

    def set_word(self, node, target, text, op=""):
        """`message.word = value` (or `+=` and the others): a word of a message."""
        word = self.message_word(target)
        if word is None:
            self.refuse(node, "an attribute set is a message's word", "set one")
        uses = self.uses(node.value) | {target.value.id}
        return [], [Line(f"{word} {op}= {text};", uses)]

    def construct(self, node):
        """The construct a call is (a key of the names table), or None."""
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                return kind_of(func.attr, ("partition", "result"))
            if (
                isinstance(func.value, ast.Attribute)
                and func.value.attr == spelling("tile")
                and func.attr == spelling("zeros")
            ):
                return "zeros"
            if isinstance(func.value, ast.Name) and func.value.id in ("km", "self"):
                return kind_of(
                    func.attr, ("stream", "reduce", "all_reduce", "broadcast", "cxx")
                )
            value = isinstance(func.value, ast.Name) and self.scope.get(func.value.id)
            if value and value.kind == "broadcast":
                return kind_of(func.attr, ("partition",))
        return None

    # Integers ----------------------------------------------------------------------

    def integer(self, node):
        """The C++ of an integer expression (glue)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            if isinstance(node.value, bool) or node.value < 0:
                self.refuse(node, "a word is a non-negative int", "write one")
            return f"{node.value}u" if node.value > 0x7FFFFFFF else str(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.scope:
                value = self.scope[node.id]
                if value.kind == "word" and self.allow_words:
                    return node.id
                if value.kind in ("streamed", "tile"):
                    self.refuse(
                        node,
                        f"{node.id} is a tile: tile arithmetic is a tile function's",
                        "write it in a @km.tile_function and call that on the "
                        "stream's tiles",
                    )
                if value.kind == "float":
                    self.refuse(
                        node,
                        f"{node.id} is a float word (a float32's bits), and this is "
                        "the handler's integer arithmetic (a slice bound, a range, a "
                        "count)",
                        f"pass an int word for it, or use {node.id} in a tile "
                        "expression, a tile function's call or a body",
                    )
                if value.kind != "int":
                    self.refuse(
                        node, f"{node.id} is a {value.kind}, not an int", "use an int"
                    )
                return node.id
            self.not_ended(node, node.id)
            return str(self.constant(node, node.id))
        if isinstance(node, ast.Attribute):
            return self.message_word(node) or self.attribute(node)
        if self.word_name(node):
            if not self.field_line:
                self.refuse(
                    node,
                    f"self.{self.word_name(node)} is words the host reads, written "
                    "by one CPU: its words are read in a line that sets one",
                    f"keep the count in an int, or write self.{self.word_name(node)}"
                    "[i] += value",
                )
            return self.field_word(node)[0]
        if isinstance(node, ast.BinOp):
            op = OPERATORS.get(type(node.op))
            if op is None:
                self.refuse(node, "an operator level 1 has no C++ for", "use + - * //")
            return f"({self.operand(node.left)} {op} {self.operand(node.right)})"
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = COMPARISONS.get(type(node.ops[0]))
            if op is None:
                self.refuse(node, "a comparison level 1 has no C++ for", "use == < >")
            left, right = node.left, node.comparators[0]
            mirror = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
            for one, other, said in ((left, right, op), (right, left, mirror.get(op))):
                if (
                    isinstance(one, ast.BinOp)
                    and isinstance(one.op, ast.Sub)
                    and isinstance(other, ast.Constant)
                    and other.value == 0
                ):
                    self.refuse(
                        node,
                        "words are unsigned: a subtraction below 0 wraps to a large "
                        "word, never below 0",
                        f"compare the two: write {ast.unparse(one.left)} {said or op} "
                        f"{ast.unparse(one.right)}",
                    )
            return f"{self.operand(left)} {op} {self.operand(right)}"
        if isinstance(node, ast.BoolOp):
            truths = (ast.Compare, ast.BoolOp)
            for value in node.values:
                if not isinstance(value, truths) and not (
                    isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)
                ):
                    word = "or" if isinstance(node.op, ast.Or) else "and"
                    self.refuse(
                        node,
                        f"`{word}` of words is a word in Python ({ast.unparse(value)} "
                        "itself), and a truth in C++",
                        f"write {spelling('max')}(a, 1), or an if",
                    )
            op = " && " if isinstance(node.op, ast.And) else " || "
            return "(" + op.join(self.integer(v) for v in node.values) + ")"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return f"!({self.integer(node.operand)})"
        if self.self_call(node, "direct_message"):
            self.direct_unnamed(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in (spelling("min"), spelling("max"))
            and node.func.id not in self.scope
        ):  # the smaller (larger) of ints, as words
            if len(node.args) < 2 or node.keywords:
                self.refuse(
                    node,
                    f"{node.func.id}(...) of level 1 takes two ints or more",
                    f"write {node.func.id}(a, b)",
                )
            op = "<" if node.func.id == spelling("min") else ">"
            text = self.operand(node.args[0])
            for arg in node.args[1:]:
                other = self.operand(arg)
                text = f"({text} {op} {other} ? {text} : {other})"
            return text
        self.refuse(node, "not an integer of level 1", "write ints, names, + - * //")

    def operand(self, node):
        """integer() of an operator's operand: a comparison in parentheses, which C++
        binds looser than arithmetic (`a + (b < c)`)."""
        text = self.integer(node)
        return f"({text})" if isinstance(node, ast.Compare) else text

    def constant(self, node, name):
        """A name of the module: an int constant, or the plan's value for it."""
        if name in self.plan:
            value = self.plan[name]
        elif name in self.names and isinstance(self.names[name], int):
            value = self.names[name]
        else:
            self.refuse(node, f"{name} is not known here", "define it, or pass it")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            self.refuse(node, f"{name} = {value!r}: not a word", "make it an int")
        self.compiled.constants[name] = value
        return value

    def constant_int(self, node):
        """An int known at build: a literal, a module constant or a plan value."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if self.is_class_constant(node):
            return self.class_constant(node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self.constant(node, node.value)  # a symbol: the plan's
        if isinstance(node, ast.Name) and node.id not in self.scope:
            return self.constant(node, node.id)
        self.word_given(node, "a size the build needs")
        self.refuse(node, "a size the build needs", "give an int, or a plan name")

    def sized_block(self, node):
        """A stream's block= : a constant, or a parameter given at each submit
        (a handler parameter given at each submit), whose most sizes the buffer
        (stream_word)."""
        sized = front.sized_parameters(self.method.function)
        if isinstance(node, ast.Name) and node.id in sized:
            self.stream_word = node.id
            return sized[node.id]
        return self.constant_int(node)

    def block_word(self, node, buffer):
        """A stream's pair whose block is given at each submit: the handler's
        `<pair>Block` a word, its constant the most (the pair's pages) elsewhere; a
        pair one handler streams with two blocks refused."""
        word, self.stream_word = self.stream_word, None
        if self.block_words.setdefault(buffer.name, word) != word:
            self.refuse(
                node,
                f"pair {buffer.name} is streamed with block {word or 'a constant'} "
                f"and with {self.block_words[buffer.name] or 'a constant'} in one "
                "handler",
                "give the two streams one block, or tensors of other formats",
            )

    def block_lines(self):
        """Each pair's block given at each submit, as `<pair>Block` over the
        constant (the most), in the handler's movement and compute."""
        return [
            Line(
                f"[[maybe_unused]] const uint32_t {name}Block = {word};  // "
                "given at each submit: a divisor of the constant's",
                {word},
            )
            for name, word in self.block_words.items()
            if word is not None
        ]

    def attribute(self, node):
        base = node.value
        if self.is_class_constant(node):
            return str(self.class_constant(node))
        if isinstance(base, ast.Name) and base.id == "self":
            kind = kind_of(node.attr, ("rank", "size", "leader", "x", "y", "core"))
            if kind == "core":
                where = f"line {node.lineno}"
                self.compiled.rect_lines.setdefault(self.method.name, where)
                return "km_self_core"
            if kind == "rank":
                return "km_rank"
            if kind == "size":
                return "km_size"
            if kind == "leader":
                return self.leader_text
            if kind in ("x", "y"):
                where = f"line {node.lineno}"
                self.compiled.rect_lines.setdefault(self.method.name, where)
                self.compiled.rectangle = True
                self.compiled.rect_plain |= self.rect_param is None
                self.rect_read = True
                return f"km_self_{kind}"
            self.field_use(node)
            self.refuse(node, f"self.{node.attr} is not known", "use self.rank")
        if isinstance(base, ast.Name) and base.id in self.scope:
            value = self.scope[base.id]
            if value.kind in ("entry", "entries"):
                return self.entries_word(node, value)
            if value.kind == "partition":
                if node.attr == "start":
                    return value.first_text(value.cxx)
                if node.attr == "count":
                    return f"{value.cxx}.len"
            if value.kind == "range" and node.attr in ("first", "count"):
                return value.first if node.attr == "first" else value.count
            if value.kind == "tensor" and node.attr == spelling("pages"):
                return self.counts[base.id]
        self.refuse(node, "an attribute level 1 does not know", "use .start, .count")

    def is_class_constant(self, node):
        """Whether `node` is self.X of a constant the class declares."""
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in self.compiled.declared
        )

    def class_constant(self, node):
        """self.X of a constant of the class: this program's value, lowered as a
        module constant is (a non-negative int)."""
        name = node.attr
        if name not in self.compiled.class_constants:
            self.refuse(
                node,
                f"self.{name} is a symbol the build solves after the handlers' lines",
                f"give {name} in the {spelling('plan')}, or as {name}= at the make",
            )
        value = self.compiled.class_constants[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            self.refuse(node, f"self.{name} = {value!r}: not a word", "make it an int")
        return value

    def word_given(self, node, what):
        """Refuses a command word where `what` needs a constant: the tier's error."""
        if isinstance(node, ast.Name) and any(p.name == node.id for p in self.params):
            self.refuse(
                node,
                f"{node.id} is a word of the command, known only when the command "
                f"runs, and {what} is fixed when the program is built: it needs a "
                "constant of the class",
                f"declare {node.id.upper()} = km.{spelling('constant')}(n) in the "
                f"class and write self.{node.id.upper()}",
            )

    def stated_words(self, node, what):
        """word_given of each `n` a rectangle's side states as its most (r:r + n,
        r:min(r + n, e))."""
        if not isinstance(node, ast.Subscript):
            return
        sides = node.slice.elts if isinstance(node.slice, ast.Tuple) else []
        for side in sides:
            if not isinstance(side, ast.Slice) or side.upper is None:
                continue
            upper = side.upper
            ends = upper.args if isinstance(upper, ast.Call) else [upper]
            for end in ends:
                if isinstance(end, ast.BinOp) and isinstance(end.op, ast.Add):
                    self.word_given(end.right, what)

    def uses(self, node):
        """The names an expression reads (scope names and generated ones)."""
        found = set()
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and item.id in self.scope:
                value = self.scope[item.id]
                found.add(value.cxx or item.id)
            if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name):
                if item.value.id == "self":
                    kind = kind_of(item.attr, ("rank", "size", "x", "y", "core"))
                    if kind in ("x", "y", "core"):
                        found.add(f"km_self_{kind}")
                    elif kind:
                        found.add(f"km_{kind}")
                if item.value.id in self.counts and item.attr == spelling("pages"):
                    found.add(self.counts[item.value.id])
                entry = self.scope.get(item.value.id)
                if entry is not None and entry.kind in ("entry", "entries"):
                    found.add(self.entries_word(item, entry))
        return found

    # Ranges, partitions ------------------------------------------------------------

    def is_range(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return node.func.attr in (
                spelling("pages"),
                spelling("items"),
            ) and self.is_tensor(node.func.value)
        if isinstance(node, ast.Subscript):
            return self.is_tensor(node.value) or self.is_tiles(node.value)
        if isinstance(node, ast.Attribute):
            return node.attr == spelling("pages") and self.is_tensor(node.value)
        if isinstance(node, ast.Name):
            return self.scope.get(node.id, Value("")).kind == "range"
        return False

    def is_tiles(self, node):
        """`tensor.tiles`, a tensor's tiles as a grid."""
        return (
            isinstance(node, ast.Attribute)
            and node.attr == spelling("tiles")
            and self.is_tensor(node.value)
        )

    def rectangle(self, node):
        """`t.tiles[r0:r1, c0:c1]`: the range of the rectangle's tiles, row-major (a
        row: one run of pages; rows: runs of its columns, a row of t apart);
        `t.tiles[r0:r1:s, c0:c1]` its rows s rows of t apart (stepped_side)."""
        tensor = node.value.value.id
        example = self.tensors.get(tensor)
        sides = node.slice.elts if isinstance(node.slice, ast.Tuple) else []
        if (
            len(sides) != 2
            or not all(isinstance(side, ast.Slice) for side in sides)
            or sides[1].step is not None
            or not all(
                side.lower is not None and side.upper is not None for side in sides
            )
        ):
            self.refuse(
                node,
                "a rectangle of tiles is two slices, rows then columns, each with "
                "its start and end, a step on the rows alone",
                f"write {tensor}.{spelling('tiles')}[r0:r1, c0:c1] or [r0:r1:s, "
                "c0:c1]",
            )
        if (
            not front.is_tensor(example)
            or example.layout != "tile"
            or len(example.shape) < 2
        ):
            self.refuse(
                node,
                f"{tensor}'s tiles: the program is made without an example tensor "
                "in tiles of at least two dimensions",
                f'make it with {tensor}=a tensor of the shape, layout "tile"',
            )
        columns_of = getattr(self, "column_words", {}).get(tensor)
        if columns_of is None:
            columns_of = example.shape[-1] // 32
            self.method.widths[tensor] = columns_of  # the host's check at submit
        texts, lengths, bounds = [], [], []
        for side in sides:
            low, high = self.integer(side.lower), self.integer(side.upper)
            texts.append(low)
            known = self.side_length(side)
            lengths.append((known, f"(({high}) - ({low}))"))
            bounds.append(known if known is not None else self.side_bound(side))
        step = self.row_step(node, sides[0], columns_of)
        if step is not None:  # rows s apart: their count, its most
            lengths[0], bounds[0] = self.stepped_side(sides[0])
        rows, columns = lengths
        first = f"(({texts[0]}) * {columns_of} + ({texts[1]}))"
        if rows[0] is not None and columns[0] is not None:
            count = str(rows[0] * columns[0])
        elif rows[0] == 1:  # one row: its columns
            count = columns[1]
        else:
            count = (
                f"({rows[1] if rows[0] is None else rows[0]} * "
                f"{columns[1] if columns[0] is None else columns[0]})"
            )
        used = {columns_of} if isinstance(columns_of, str) else set()
        for side in sides:
            used |= self.uses(side.lower) | self.uses(side.upper)
        for part in (sides[0].step,) if step is not None else ():
            used |= self.uses(part)
        if step is None and (rows[0] == 1 or columns[0] == columns_of):  # one run
            # (a width given at each submit is never known to equal a side)
            rect = None
        else:  # rows of columns; either side a word, read at run time
            rect = (
                rows[1] if rows[0] is None else str(rows[0]),
                columns[1] if columns[0] is None else columns[0],
                columns_of if step is None else step,
            )
        return Value(
            "range",
            first=first,
            count=count,
            tensor=tensor,
            cxx=" ".join(sorted(used)),
            rect=rect,
            bounds=tuple(bounds),
            sides=(
                rows[1] if rows[0] is None else str(rows[0]),
                columns[1] if columns[0] is None else columns[0],
            ),
        )

    def side_bound(self, side):
        """The most a slice whose length is a word can be, where the build knows it:
        n of `c:min(c + n, e)` (either order of min's ints), else None."""
        upper = side.upper
        if not (
            isinstance(upper, ast.Call)
            and isinstance(upper.func, ast.Name)
            and upper.func.id == spelling("min")
        ):
            return None
        found = [
            self.known_int(arg.right)
            for arg in upper.args
            if isinstance(arg, ast.BinOp)
            and isinstance(arg.op, ast.Add)
            and ast.dump(arg.left) == ast.dump(side.lower)
        ]
        known = [n for n in found if n is not None]
        return min(known) if known else None

    def row_step(self, node, side, columns_of):
        """A rectangle's rows' step, s of `r0:r1:s` (an int or a word): the tensor
        pages from one of its rows to the next (s x t's columns), or None for no step
        or a step of 1."""
        if side.step is None or self.known_int(side.step) == 1:
            return None
        known = self.known_int(side.step)
        if known is not None and known < 1:
            self.refuse(node, "a rectangle's rows' step is 1 or more", "give s >= 1")
        if known is not None and isinstance(columns_of, int):
            return known * columns_of
        if known is not None:  # the tensor's width given at each submit
            return f"({known} * {columns_of})"
        return f"(({self.integer(side.step)}) * {columns_of})"

    def stepped_side(self, side):
        """The rows of `r0:r1:s`, (their count where the build knows it, its C++),
        and their most where the build knows it: r1 = r0 + n * s (n rows), r1 =
        min(r0 + n * s, e) (at most n), or a step known and r1 - r0 known or bound
        (the rows of that many)."""
        low, high = self.integer(side.lower), self.integer(side.upper)
        step = self.integer(side.step)
        text = f"(((({high}) - ({low})) + ({step}) - 1) / ({step}))"

        def rows_of(end):  # n of r0 + n * s (either order of the product), or None
            if not (
                isinstance(end, ast.BinOp)
                and isinstance(end.op, ast.Add)
                and ast.dump(end.left) == ast.dump(side.lower)
                and isinstance(end.right, ast.BinOp)
                and isinstance(end.right.op, ast.Mult)
            ):
                return None
            factors = (end.right.left, end.right.right)
            for factor, other in (factors, factors[::-1]):
                if ast.dump(other) == ast.dump(side.step):
                    return self.known_int(factor)
            return None

        known = rows_of(side.upper)
        if known is not None:
            return (known, text), known
        upper = side.upper
        if (
            isinstance(upper, ast.Call)
            and isinstance(upper.func, ast.Name)
            and upper.func.id == spelling("min")
        ):
            found = [n for n in map(rows_of, upper.args) if n is not None]
            if found:
                return (None, text), min(found)
        each = self.known_int(side.step)
        length = self.side_length(side)
        if each is not None and length is not None:
            return (-(-length // each), text), -(-length // each)
        bound = self.side_bound(side)
        if each is not None and bound is not None:
            return (None, text), -(-bound // each)
        return (None, text), None

    def side_length(self, side):
        """A slice's length where the build knows it: both ends ints, or `a:a + n`."""
        ends = []
        for end in (side.lower, side.upper):
            try:
                ends.append(ast.literal_eval(end))
            except ValueError:
                ends.append(self.known_int(end))
        if all(isinstance(end, int) for end in ends):
            return ends[1] - ends[0]
        upper = side.upper
        if (
            isinstance(upper, ast.BinOp)
            and isinstance(upper.op, ast.Add)
            and ast.dump(upper.left) == ast.dump(side.lower)
        ):
            return self.known_int(upper.right)
        return None

    def known_int(self, node):
        """An int the build knows (a literal, a module constant, a plan value), or
        None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if self.is_class_constant(node):
            return self.class_constant(node)
        if isinstance(node, ast.Name) and node.id not in self.scope:
            value = self.plan.get(node.id, self.names.get(node.id))
            return value if isinstance(value, int) else None
        return None

    def is_tensor(self, node):
        return (
            isinstance(node, ast.Name)
            and self.scope.get(node.id, Value("")).kind == "tensor"
        )

    def range_of(self, node):
        """A page range: tensor.pages, tensor.pages(first, count), tensor[part]."""
        if isinstance(node, ast.Name):
            return self.scope[node.id]
        if isinstance(node, ast.Attribute):
            tensor = node.value.id
            return Value(
                "range", first="0", count=self.counts[tensor], tensor=tensor, cxx=tensor
            )
        if isinstance(node, ast.Call) and node.func.attr == spelling("items"):
            return self.items_of(node)
        if isinstance(node, ast.Call):
            tensor = node.func.value.id
            if len(node.args) != 2 or node.keywords:
                self.refuse(node, "pages(first, count)", "give both")
            first, count = (self.integer(arg) for arg in node.args)
            used = self.uses(node.args[0]) | self.uses(node.args[1])
            return Value(
                "range",
                first=first,
                count=count,
                tensor=tensor,
                cxx=" ".join(sorted(used)),
            )
        if self.is_tiles(node.value):
            return self.rectangle(node)
        tensor = node.value.id
        part = node.slice
        if isinstance(part, ast.Name) and part.id in self.scope:
            value = self.scope[part.id]
            if value.kind == "partition":
                return Value(
                    "range",
                    first=value.first_text(value.cxx),
                    count=f"{value.cxx}.len",
                    tensor=tensor,
                    cxx=part.id,
                )
        self.refuse(
            node,
            "a tensor is in DRAM: a slice is not a value",
            f"take a part: {tensor}[self.{spelling('partition')}(...)], or use a "
            f"stream or {spelling('read')}()",
        )

    def items_of(self, node):
        """`t.items(first, count, size=B)`: items of B bytes packed in t's pages, a
        whole number a page; a stream moves them as pages of B bytes."""
        tensor = node.func.value.id
        if not node.args:  # first=, count=, size=, step=: the keyword form
            return self.stepped_items(node, tensor)
        sizes = [k.value for k in node.keywords if k.arg == spelling("item_size")]
        if len(node.args) != 2 or len(sizes) != 1 or len(node.keywords) != 1:
            self.refuse(
                node,
                f"{spelling('items')}(first, count, {spelling('item_size')}=bytes)",
                "give first, count and the bytes an item",
            )
        size = self.constant_int(sizes[0])
        _, page_bytes = self.tensor_format(node, tensor)
        if size < 1 or page_bytes % size:
            self.refuse(
                node,
                f"items of {size} B in pages of {page_bytes} B: a page holds a whole "
                "number of items",
                f"give a {spelling('item_size')} that divides {page_bytes}",
            )
        first, count = (self.integer(arg) for arg in node.args)
        used = self.uses(node.args[0]) | self.uses(node.args[1])
        return Value(
            "range",
            first=first,
            count=count,
            tensor=tensor,
            cxx=" ".join(sorted(used)),
            items=(size, page_bytes // size),
        )

    def stepped_items(self, node, tensor):
        """`t.items(first=f, count=n, size=B, step=s)`: items f, f + s, ... (n of
        them), a `km::Walk` of one item a row and a row pitch of s items."""
        names = ("entries_first", "entries_count", "item_size", "item_step")
        given = {k.arg: k.value for k in node.keywords}
        words = [spelling(n) for n in names]
        if set(given) - set(words) or not set(words[:3]) <= set(given):
            self.refuse(
                node,
                f"{spelling('items')}({', '.join(w + '=' for w in words)}): the first "
                "item, their count, the bytes an item and the items from one to the "
                f"next ({spelling('item_step')}= 1 if left out)",
                "give first=, count= and size= (and step=), and no other keyword",
            )
        size = self.constant_int(given[words[2]])
        _, page_bytes = self.tensor_format(node, tensor)
        if size < 1 or page_bytes % size:
            self.refuse(
                node,
                f"items of {size} B in pages of {page_bytes} B: a page holds a whole "
                "number of items",
                f"give a {spelling('item_size')} that divides {page_bytes}",
            )
        bounded = {}
        for word in (words[1], words[3]):
            if word not in given:
                continue
            value = given[word]
            known = self.known_int(value)
            if known is None and not self.is_word(value):
                self.refuse(
                    node,
                    f"{word}= {ast.unparse(value)}: the build bounds an int or a word "
                    "of the handler, not this",
                    f"give {word}= an int or a word (a parameter, an entry's word)",
                )
            if known is not None and known < 1:
                self.refuse(node, f"{word}= {known}: 1 or more", f"give {word}= >= 1")
            bounded[word] = known
        first = self.integer(given[words[0]])
        count = self.integer(given[words[1]])
        step = self.integer(given[words[3]]) if words[3] in given else "1"
        used = set()
        for value in given.values():
            used |= self.uses(value)
        rect = None if step == "1" else (count, 1, step)
        return Value(
            "range",
            first=first,
            count=count,
            tensor=tensor,
            cxx=" ".join(sorted(used)),
            items=(size, page_bytes // size),
            rect=rect,
            bounds=(bounded[words[1]], 1),
            sides=(count, 1),
        )

    def is_word(self, node):
        """Whether `node` is a name or attribute the handler holds as an int word."""
        if isinstance(node, ast.Name):
            value = self.scope.get(node.id)
            return value is not None and value.kind in ("int", "word")
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (
                getattr(self.scope.get(node.value.id), "kind", None)
                in ("entry", "entries")
                or node.value.id == "self"
            )
        )

    def partition(self, node, name, call):
        """`part = self.partition(pages)`: this core's part by rank; `part =
        b.partition(pages)`: by its place along broadcast b's row or column."""
        index, length = "km_rank", "km_size"
        if isinstance(call.func.value, ast.Name) and call.func.value.id != "self":
            along = self.scope[call.func.value.id].pattern
            index, length = (
                ("km_self_x", "km_rect_columns")
                if along == "row"
                else ("km_self_y", "km_rect_rows")
            )
        if len(call.args) != 1 or call.keywords:
            self.refuse(node, "partition(pages) takes one range", "give one")
        argument = call.args[0]
        if self.is_range(argument):
            pages = self.range_of(argument)
            self.pages_only(node, pages)
            if pages.rect is not None and str(pages.rect[0]) != "1":
                self.refuse(
                    node,
                    "a part is a run of pages, and a rectangle of rows is not one",
                    f"partition one row ({spelling('tiles')}[r:r + 1, c0:c1]), whole "
                    "rows, or a page range",
                )
            first, count = pages.first, pages.count
        else:
            first, count = "0", self.integer(argument)
        self.scope[name] = Value("partition", name, first=first)
        text = (
            f"const km::Partition {name} = km::partition({count}, {index}, {length});"
        )
        uses = text_uses(count, self) | {index, length}
        return [Glue(text, {name}, uses)], [Glue(text, {name}, uses)]

    # Tiles -------------------------------------------------------------------------

    def zeros(self, node, name, call):
        fmt = call.args[0] if call.args else None
        if not (
            isinstance(fmt, ast.Attribute) and kind_of(fmt.attr, ("fp32",)) == "fp32"
        ):
            self.refuse(
                node,
                "a tile accumulates in DST in fp32",
                f"write km.{spelling('tile')}.{spelling('zeros')}(km.fp32)",
            )
        count = self.constant_int(call.args[1]) if len(call.args) > 1 else 1
        second = self.dst_live is not None and self.second_sums(node, name, count)
        if self.dst_live is not None and not second:
            self.refuse(
                node,
                f"{self.dst_live} still holds DST; level 1 keeps one tile there",
                "write it, or reduce it, first",
            )
        if not 1 <= count <= DST_FULL - self.dst_next:
            self.refuse(
                node,
                f"{count} tiles of sums: DST holds {DST_FULL} fp32 tiles",
                f"write km.{spelling('tile')}.{spelling('zeros')}(km.fp32, n), n up "
                f"to {DST_FULL - self.dst_next}",
            )
        if self.dst_next + count > DST_HALF:  # past a half: DST in full-sync mode
            self.compiled.full_dst = True
        dst = self.dst_next
        self.dst_next += count
        if second:
            self.dst_pair = (self.dst_live, name)
        self.dst_live = name
        self.scope[name] = Value(
            "tile", name, dst=dst, width=count, made_at=(len(self.loops), node.lineno)
        )
        self.compiled.includes |= {"fill"}
        fills = "\n".join(f"fill_tile({dst + t}, 0.0f);" for t in range(count))
        if second:  # DST is held already: the second sums' tiles zeroed after the first
            return [], [Line(f"fill_tile_init();\n{fills}")]
        text = f"tile_regs_acquire();\nfill_tile_init();\n{fills}"
        return [], [Line(text)]

    def second_sums(self, node, name, count):
        """Whether `name = km.tile.zeros(km.fp32, count)` is a second set of DST sums
        held beside the live one (two sums a tile function of both writes: `pages.
        write(f(a, b))`); refused by name when they would not fit DST together."""
        first = self.scope.get(self.dst_live)
        if (
            self.dst_pair is not None
            or first is None
            or first.kind != "tile"
            or first.made_at is None
            or first.made_at[0] != len(self.loops)
            or first.batch is not None
            or getattr(first, "state", None) is not None
            or getattr(first, "row_of", None) is not None
            or first.cxx in self.carried
        ):
            return False
        if first.dst + first.width + count > DST_FULL:
            self.refuse(
                node,
                f"{first.cxx} ({first.width} tiles) and {name} ({count} tiles) take "
                f"{first.dst + first.width + count} of DST's {DST_FULL} fp32 tiles",
                f"make each {DST_FULL // 2} or fewer",
            )
        return True

    def scratch(self):
        """The DST tile a copy lands in before an add: the one after the sums."""
        return self.dst_next

    def product(self, node, value, call):
        """`acc += f(block, self.w)`, f a tile function `return a @ b`: the library's
        product (matmul_block) of each tile of the loop's block by the field's row of
        that K tile (acc's width of tiles) added into acc's DST tiles; the field (K
        rows, K-major) held from its fill to the handler's end."""
        function = self.tile_function_of(call)
        where = function_line(function)
        if not is_product(function.function):
            self.refuse(
                node,
                f"acc += f(...) adds a product, and the tile function {function.name} "
                f"({where}) is not `return a @ b`",
                f"write {function.name} as `return a @ b`, a the stream's block, b a "
                "field",
            )
        held = len(call.args) == 2 and self.held_row(call.args[0])
        arrived = held and self.arrived_operand(node, call.args[1])
        if arrived:
            return self.held_block_product(node, value, function, held, arrived)
        left = len(call.args) == 2 and self.field_name(call.args[0])
        rows = left and self.arrived_rows(node, call.args[1])
        if rows:
            return self.field_block_product(node, value, function, left, rows)
        if not self.loop_states or len(call.args) != 2 or call.keywords:
            self.refuse(
                node,
                "a product takes a block of the loop's stream and a field",
                f"write acc += self.{function.name}(block, self.w) in "
                f"`for block in km.{spelling('stream')}(...)`",
            )
        state = self.loop_states[-1]
        item = isinstance(call.args[0], ast.Name) and self.scope.get(call.args[0].id)
        left = self.field_name(call.args[0])
        right = isinstance(call.args[1], ast.Name) and self.scope.get(call.args[1].id)
        if left is not None and right and self.in_this_loop(right):
            return self.field_left_product(node, value, function, left, right)
        field_ = self.field_name(call.args[1])
        other = isinstance(call.args[1], ast.Name) and self.scope.get(call.args[1].id)
        if item and other and self.in_this_loop(item) and self.in_this_loop(other):
            return self.stream_product(node, value, function, item, other)
        arrived = self.arrived_operand(node, call.args[1])
        if item and self.in_this_loop(item) and arrived is not None:
            return self.broadcast_product(node, value, function, item, arrived)
        if not item or item.kind != "streamed" or field_ is None:
            self.refuse(
                node,
                f"`{ast.unparse(call.args[0])}` is not the loop's tile or "
                f"`{ast.unparse(call.args[1])}` is not a field",
                f"write acc += self.{function.name}(block, self.w)",
            )
        stored = self.compiled.fields[field_]
        why = product_refusal(item.buffer.dtype, stored.format.dtype)
        if why is not None:
            self.refuse(
                node,
                why,
                "stream a tensor of a matrix format and declare the field in one of "
                "its family",
            )
        columns = stored.shape[1] if len(stored.shape) == 4 else None
        if columns != value.width:
            target = ast.unparse(node.target)
            self.refuse(
                node,
                f"self.{field_} is K rows of {columns} tiles and {target} holds "
                f"{value.width}",
                f"declare self.{field_}: km.L1[km.bf16, K, {value.width}, 32, 32] or "
                f"make {target} km.{spelling('tile')}.{spelling('zeros')}"
                f"(km.fp32, {columns})",
            )
        if state["other"] is not None or state["function"] not in (None, "product"):
            self.refuse(
                node,
                "this loop has other tile lines, and a product holds the matrix unit's "
                "setup for the loop",
                "keep the loop to the product",
            )
        fillers = self.compiled.field_users.get(field_, ([], []))[0]
        if (
            self.filled.get(field_) is None
            and field_ not in self.waited_fields
            and not (fillers and self.method.name not in fillers)
        ):
            self.refuse(
                node,
                f"self.{field_} is not filled above in this handler, and no other "
                "handler of the class fills it",
                f"write self.{field_}.fill(km.{spelling('stream')}(...)) before, or in "
                "a handler of its own",
            )
        filled_in = self.fill_loops.get(field_, (0, 0))
        if filled_in[0] > 0 and self.fill_turn_loop(field_) is None:
            self.refuse(
                node,
                f"self.{field_} is filled in a loop (line {filled_in[1]}) and a "
                "product frees it once, at the handler's end: the next turn's fill "
                "would wait for room that is never freed",
                f"fill it before the loops, once (self.{field_}.fill(km."
                f"{spelling('stream')}(...))), or take it with a body in the loop "
                "that fills it, which frees it each turn",
            )
        fidelity = function.fidelity or "HiFi4"
        if self.compiled.fidelity not in (None, fidelity):
            self.refuse(
                node,
                f"{function.name}'s fidelity is {fidelity} and another product's "
                f"{self.compiled.fidelity}: the fidelity is the compute kernel's",
                "give the class's products one fidelity",
            )
        self.compiled.fidelity = fidelity
        weight = ("field", field_)
        left, right = item.buffer.dtype, stored.format.dtype
        turned = self.placed(node, state, function, left, right, weight)
        number = self.stream_loops[-1]
        buffer = f"kField{camel(field_)}"
        block = item.buffer.name + "Block"
        if turned:
            return self.turned_product(node, value, function, item, field_, columns)
        if state["function"] is None:
            state["function"] = "product"
            waited = ""
            if field_ not in self.waited_fields:  # once a handler
                self.waited_fields.add(field_)
                self.wait(node, "compute", WAIT_INPUT)
                waited = (
                    f"if (!compute.wait_front({buffer}, {buffer}Pages)) {{\n"
                    "    return;\n}\n"
                )
                if not self.kept_field(field_):
                    self.field_freed(field_, buffer)
            state["inits"].append(
                Line(
                    f"{waited}"
                    "// the zeros' fills (vector unit) done before the matrix unit "
                    "adds\n"
                    "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
                    f"KM_UNPACK_FORMATS({buffer}, {item.buffer.name}.mover_a);\n"
                    f"matmul_block_init({item.buffer.name}.mover_a, {buffer}, 0, "
                    f"{columns}, 1, {block});"
                )
            )
        state["inits"] += self.k_bound(node, state, field_, stored.shape[0], block)
        self.compiled.includes.add("matmul")
        self.compiled.products.append(
            (
                self.method.name,
                node.lineno,
                self.text(node),
                function,
                columns,
                block,
                fidelity,
            )
        )
        text = (
            f"matmul_block({item.cxx}, {buffer}, {item.first}, (km_block{number} * "
            f"{block} + {item.first}) * {columns}, {value.dst}, 0, {columns}, 1, "
            f"{block});"
        )
        return [], [Line(text)]

    def field_left_product(self, node, value, function, field_, item):
        """`acc += f(self.a, wb)`, self.a a row of K tiles compute wrote, wb a block
        of the loop's stream of K rows of acc's width: row k by tile k of the field,
        a row of products a step into acc's DST; the field waited for before the
        loop, freed by its next write (or the handler's end)."""
        state, number = self.loop_states[-1], self.stream_loops[-1]
        width = value.width
        if field_ not in self.packed:
            self.refuse(
                node,
                f"self.{field_} is not written above in this handler: a product's left "
                "field is one compute writes",
                f"write self.{field_}.write(acc) before this loop",
            )
        if state["other"] is not None or state["function"] not in (None, "product"):
            self.refuse(
                node,
                "this loop has other tile lines, and a product holds the matrix unit's "
                "setup for the loop",
                "keep the loop to the product",
            )
        if len(self.loop_states[-1]["tensors"]) != 1:
            self.refuse(
                node,
                "a product with a field on the left reads one stream in its loop",
                f"write `for wb in km.{spelling('stream')}(w.tiles[0:K, 0:n], "
                "block=...)`",
            )
        stored = self.compiled.fields[field_]
        k = math.prod(stored.shape[:-2])
        buffer, w = f"kField{camel(field_)}", item.buffer.name
        rows = state["count"]  # the loop's tiles, as written
        if (
            item.buffer.block % width
            or rows is not None
            and str(rows).isdigit()
            and int(rows) != k * width
        ):
            self.refuse(
                node,
                f"self.{field_} is a row of {k} tile(s) and acc {width}: the stream "
                f"is {k} rows of {width} tiles, a block of whole rows",
                f"stream w.tiles[0:{k}, c:c + {width}] with a block of n x {width}",
            )
        why = product_refusal(stored.format.dtype, item.buffer.dtype)
        if why is not None:
            self.refuse(
                node, why, "declare the field in a matrix format of the stream's"
            )
        fidelity = self.product_fidelity(node, function)
        if self.placed(
            node,
            state,
            function,
            stored.format.dtype,
            item.buffer.dtype,
            ("tensor", state["tensors"][w]),
        ):
            self.refuse(
                node,
                f"{function.name} turns the product, and a field's product on the left "
                "is built plain",
                "drop the marker",
            )
        self.wait(node, "compute", WAIT_INPUT)
        state["function"], state["step"] = "product", width
        state["inits"].append(
            Line(
                f"km_ok = km_ok && compute.wait_front({buffer}, {buffer}Pages);\n"
                "// the zeros' fills (vector unit) done before the matrix unit adds\n"
                "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
                f"KM_UNPACK_FORMATS({w}.mover_a, {buffer});\n"
                f"matmul_block_init({buffer}, {w}.mover_a, 0, {width}, 1, {k});",
                {"km_ok"},
            )
        )
        self.compiled.includes.add("matmul")
        self.compiled.products.append(
            (self.method.name, node.lineno, self.text(node), function, width, str(k))
            + (fidelity,)
        )
        row = f"(km_block{number} * {w}Block + {item.first}) / {width}"
        text = (
            f"matmul_block({buffer}, {item.cxx}, {row}, {item.first}, {value.dst}, 0, "
            f"{width}, 1, {k});"
        )
        return [], [Line(text)]

    def arrived_operand(self, node, arg):
        """`w`, `kv[i]` or either `[:, c0:c0 + n]`, w a block a broadcast brought:
        (its Value, the part's slot geometry, the first column's C++, the columns or
        None for the part's); None when `arg` names no such block."""
        window = None
        if (
            isinstance(arg, ast.Subscript)
            and isinstance(arg.slice, ast.Tuple)
            and len(arg.slice.elts) == 2
        ):
            window, arg = arg.slice.elts, arg.value
        part = None
        if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
            part, arg = arg.slice, arg.value
        value = isinstance(arg, ast.Name) and self.scope.get(arg.id)
        if not value or value.kind != "arrived":
            return None
        drawn = f"write {arg.id}[i][:, c0:c0 + n], i its part in the cast"
        if part is None and len(value.parts) > 1:
            self.refuse(
                node,
                f"{arg.id} is a cast of {len(value.parts)} parts: a product reads one",
                drawn,
            )
        index = 0 if part is None else self.known_int(part)
        if index is None or not 0 <= index < len(value.parts):
            self.refuse(node, f"{arg.id} has parts 0 .. {len(value.parts) - 1}", drawn)
        first, columns = "0", None
        if window is not None:
            rows, cols = window
            if not (
                isinstance(rows, ast.Slice)
                and rows.lower is rows.upper is rows.step is None
                and isinstance(cols, ast.Slice)
                and cols.step is None
                and cols.lower is not None
                and isinstance(cols.upper, ast.BinOp)
                and isinstance(cols.upper.op, ast.Add)
                and ast.dump(cols.upper.left) == ast.dump(cols.lower)
            ):
                self.refuse(node, "a block's columns are a window c0:c0 + n", drawn)
            first = self.integer(cols.lower)
            columns = self.known_int(cols.upper.right)
            if columns is None:
                self.refuse(
                    node,
                    "a window's width is an int the build knows",
                    "write c0:c0 + n, n a constant",
                )
        return value, value.parts[index], first, columns, arg.id

    def broadcast_product(self, node, value, function, item, arrived):
        """`acc += f(xb, w)` (or `w[i][:, c0:c0 + n]`), w a broadcast's block brought
        above the loop: tile t of block k by K row k x block + t of w's part (columns
        from c0, acc's width; transposed where the cast turned it) into acc's DST;
        w waited for before the loop's blocks, freed after them."""
        block_value, (offset, rows, pitch, transposed), first, columns, name = arrived
        state, number = self.loop_states[-1], self.stream_loops[-1]
        mover = self.finished_casts.get(name)
        if mover is not None and any(
            letters is None or mover in letters for letters in state["letters"]
        ):
            other = "A" if mover == "B" else "B"
            self.refuse(
                node,
                f"{name} was cast outside a loop, so mover {mover} waits at the cast "
                "for compute to free the block, and compute frees it after this "
                f"loop, some of whose stream's blocks mover {mover} moves",
                f'read the loop\'s stream on mover {other}: pattern="{other}"',
            )
        width = value.width
        if columns is None:
            columns = pitch
        if columns != width:
            self.refuse(
                node,
                f"the block's part is {columns or 'a word of'} columns and "
                f"{ast.unparse(node.target)} holds {width}",
                f"take a window of {width}: w[:, c0:c0 + {width}]",
            )
        if pitch is None:
            self.refuse(
                node,
                "the block's rows are a word apart in its slot",
                "cast a rectangle whose columns the build knows",
            )
        if first.isdigit() and int(first) + width > pitch:
            self.refuse(
                node,
                f"columns {first} .. {int(first) + width - 1} of a part of {pitch}",
                f"take a window inside the part's {pitch} columns",
            )
        if state["other"] is not None or state["function"] not in (None, "product"):
            self.refuse(
                node,
                "this loop has other tile lines, and a product holds the matrix unit's "
                "setup for the loop",
                "keep the loop to the product",
            )
        buffer = block_value.buffer
        why = product_refusal(item.buffer.dtype, buffer.dtype)
        if why is not None:
            self.refuse(node, why, "cast tiles of a matrix format of the stream's")
        fidelity = self.product_fidelity(node, function)
        if self.placed(
            node,
            state,
            function,
            item.buffer.dtype,
            buffer.dtype,
            ("tensor", block_value.tensor),
        ):
            self.refuse(
                node,
                f"{function.name} turns the product, and a broadcast block's product "
                "is built plain",
                "drop the marker",
            )
        held = getattr(block_value, "held", False)  # waited at its cast instead
        if not held:
            self.arrived_once(node, name, block_value)
        x, w = f"{item.buffer.name}.mover_a", buffer.name
        block = item.buffer.name + "Block"
        flip = "true" if transposed else "false"
        self.wait(node, "compute", WAIT_INPUT)
        if state["function"] is None:
            state["function"] = "product"
        received = (
            ""
            if held
            else f"km_ok = km_ok && km::broadcast_receive(compute, {w}, {w}Pages);\n"
        )
        state["inits"].append(
            Line(
                f"{received}"
                "// the zeros' fills (vector unit) done before the matrix unit adds\n"
                "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
                f"KM_UNPACK_FORMATS({w}, {x});\n"
                f"matmul_block_init({x}, {w}, {flip}, {width}, 1, {block});",
                {"km_ok"},
            )
        )
        if not held:
            state["after"].append(
                Line(f"if (km_ok) {{\n    km::broadcast_done({w}, {w}Pages);\n}}")
            )
        self.compiled.includes.add("matmul")
        self.compiled.products.append(
            (self.method.name, node.lineno, self.text(node), function, width, block)
            + (fidelity,)
        )
        row = f"(km_block{number} * {block} + {item.first})"
        text = (
            f"matmul_block({item.cxx}, {w}, {item.first}, {offset} + {row} * {pitch} "
            f"+ {first}, {value.dst}, {flip}, {width}, 1, {block});"
        )
        return [], [Line(text, text_uses(first, self))]

    def placed(self, node, state, function, left, right, weight):
        """Whether the product is turned (product_placement): refused when both
        operands are marked for one side, or the loop's other product is placed the
        other way; noted for the build's report of both placements."""
        placement = product_placement(function.function)
        if placement not in ("none", "plain", "turned"):
            self.refuse(
                node,
                f"{function.name}: {placement}",
                f"mark one operand: `return a @ km.{spelling('srcb')}(b)`",
            )
        self.compiled.placements.append(
            (
                self.method.name,
                node.lineno,
                self.text(node),
                left,
                right,
                placement,
                weight,
            )
        )
        turned = placement == "turned"
        if state.setdefault("turned", turned) != turned:
            self.refuse(
                node,
                "this loop's products are placed two ways, and a loop holds the "
                "matrix unit's setup for one",
                "mark both products' operands alike",
            )
        return turned

    def turned_product(self, node, value, function, item, field_, columns):
        """The field product turned: W's tiles (stored transposed) into SrcB, the
        block's tile read transposed into SrcA, acc's tiles transposed before the
        loop and after it, so DST sums (X W)^T between and X W outside."""
        state, number = self.loop_states[-1], self.stream_loops[-1]
        buffer = f"kField{camel(field_)}"
        block = item.buffer.name + "Block"
        x = f"{item.buffer.name}.mover_a"
        turn = turned_tiles(value.width, value.dst)
        if state["function"] is None:
            state["function"] = "product"
            waited = ""
            if field_ not in self.waited_fields:  # once a handler
                self.waited_fields.add(field_)
                self.wait(node, "compute", WAIT_INPUT)
                waited = (
                    f"if (!compute.wait_front({buffer}, {buffer}Pages)) {{\n"
                    "    return;\n}\n"
                )
                if not self.kept_field(field_):
                    self.field_freed(field_, buffer)
            state["inits"].append(
                Line(
                    f"{waited}"
                    "// the zeros' fills (vector unit) done before the matrix unit "
                    "adds\n"
                    "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));"
                )
            )
        state["inits"] += [
            turn,
            Line(
                f"KM_UNPACK_FORMATS({x}, {buffer});\n"
                f"matmul_block_init({buffer}, {x}, 1, 1, {columns}, 1);"
            ),
        ]
        state["after"].append(turn)
        stored = self.compiled.fields[field_]
        state["inits"] += self.k_bound(node, state, field_, stored.shape[0], block)
        self.compiled.includes |= {"matmul", "transpose_dest"}
        self.compiled.products.append(
            (
                self.method.name,
                node.lineno,
                self.text(node),
                function,
                columns,
                block,
                self.compiled.fidelity,
            )
        )
        text = (
            f"matmul_block({buffer}, {item.cxx}, (km_block{number} * {block} + "
            f"{item.first}) * {columns}, {item.first}, {value.dst}, 1, 1, "
            f"{columns}, 1);"
        )
        return [], [Line(text)]

    def k_bound(self, node, state, field_, rows, block):
        """A product's K tiles (the loop's stream, its repeats) against the field's K
        rows: refused at build when both are numbers; else one compare a command
        before the loop, a give-up (kGaveBadItem) in place of a read past the field."""
        count, repeat = state["count"], state["repeat"]
        if count.isdigit() and repeat.isdigit():
            size = state["block"]
            tiles = (int(count) + size - 1) // size * (int(repeat) - 1) * size
            tiles += int(count)
            if tiles > rows:
                self.refuse(
                    node,
                    f"self.{field_} is {rows} K rows, and the loop's stream (line "
                    f"{state['line']}) is {tiles} K tiles: a tile of the stream "
                    f"takes a row of self.{field_}",
                    f"stream at most {rows} tiles, or declare self.{field_} with "
                    f"{tiles} rows",
                )
            return []
        tiles = count
        if repeat != "1":
            tiles = (
                f"({count} + {block} - 1) / {block} * ({repeat} - 1) * {block} + "
                f"{count}"
            )
        uses = text_uses(count, self) | text_uses(repeat, self)
        return [
            Line(
                f"// the stream's K tiles against self.{field_}'s {rows} rows\n"
                f"if ({tiles} > {rows}) {{\n"
                "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
                "    return;\n}",
                uses,
            )
        ]

    def in_this_loop(self, item):
        """Whether `item` is a block of the innermost stream loop's streams."""
        return (
            item.kind == "streamed"
            and bool(self.stream_loops)
            and item.first == f"km_tile{self.stream_loops[-1]}"
            and item.buffer is not None
        )

    def product_fidelity(self, node, function):
        """The product's fidelity, one for the class's products (the compute
        kernel's)."""
        fidelity = function.fidelity or "HiFi4"
        if self.compiled.fidelity not in (None, fidelity):
            self.refuse(
                node,
                f"{function.name}'s fidelity is {fidelity} and another product's "
                f"{self.compiled.fidelity}: the fidelity is the compute kernel's",
                "give the class's products one fidelity",
            )
        self.compiled.fidelity = fidelity
        return fidelity

    def stream_operands(self, node, function, a, b, columns, target):
        """Two streams' blocks as a product's operands: bfloat16, b's block k rows
        of `columns` for each of a's k tiles (a: one row of tiles a block)."""
        why = product_refusal(a.buffer.dtype, b.buffer.dtype)
        if why is not None:
            self.refuse(node, why, "stream tensors of matrix formats of one family")
        if a.width != 1 or b.buffer.block != a.buffer.block * columns:
            self.refuse(
                node,
                f"{target} is {columns} tiles wide: each block of the second stream "
                f"is the first's {a.buffer.block} K tiles by {columns} columns "
                f"({a.buffer.block * columns} tiles), and it is {b.buffer.block}",
                f"stream w.{spelling('tiles')}[k0:k1, c:c + {columns}] with "
                f"{spelling('block')}={a.buffer.block * columns} beside x's "
                f"{spelling('block')}={a.buffer.block}",
            )
        state = self.loop_states[-1]
        if a.cxx != state["first"]:
            self.refuse(
                node,
                "the loop counts the first stream's tiles, and the product's first "
                "operand is not the first stream's block",
                "zip x's stream first, then w's",
            )
        if state["other"] is not None or state["function"] not in (None, "product"):
            self.refuse(
                node,
                "this loop has other tile lines, and a product holds the matrix unit's "
                "setup for the loop",
                "keep the loop to the product",
            )
        self.product_fidelity(node, function)
        self.compiled.includes.add("matmul")

    def stream_product(self, node, value, function, a, b):
        """`acc += f(xb, wb)` in `for xb, wb in zip(km.Stream(x..), km.Stream(w..))`:
        each tile k of xb by row k of wb (acc's width of tiles) added into acc's DST
        tiles by the library's product; each W block used as it arrives and freed."""
        target = ast.unparse(node.target)
        self.stream_operands(node, function, a, b, value.width, target)
        state = self.loop_states[-1]
        block = f"{a.buffer.name}Block"
        weight = ("tensor", state["tensors"][b.buffer.name])
        left, right = a.buffer.dtype, b.buffer.dtype
        if self.placed(node, state, function, left, right, weight):
            return self.turned_stream_product(node, value, function, a, b)
        if state["function"] is None:
            state["function"] = "product"
            state["inits"].append(
                Line(
                    "// the zeros' fills (vector unit) done before the matrix unit "
                    "adds\n"
                    "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
                    f"KM_UNPACK_FORMATS({b.buffer.name}.mover_a, "
                    f"{a.buffer.name}.mover_a);\n"
                    f"matmul_block_init({a.buffer.name}.mover_a, {b.buffer.name}"
                    f".mover_a, 0, {value.width}, 1, {block});"
                )
            )
        self.compiled.products.append(
            (
                self.method.name,
                node.lineno,
                self.text(node),
                function,
                value.width,
                block,
                self.compiled.fidelity,
            )
        )
        text = (
            f"matmul_block({a.cxx}, {b.cxx}, {a.first}, {a.first} * {value.width}, "
            f"{value.dst}, 0, {value.width}, 1, {block});"
        )
        return [], [Line(text)]

    def turned_stream_product(self, node, value, function, a, b):
        """The stream product turned: each W block (its tiles stored transposed)
        into SrcB, X's tile read transposed into SrcA, acc's tiles transposed before
        the loop and after it (turned_product's)."""
        state = self.loop_states[-1]
        x, w = f"{a.buffer.name}.mover_a", f"{b.buffer.name}.mover_a"
        turn = turned_tiles(value.width, value.dst)
        if state["function"] is None:
            state["function"] = "product"
            state["inits"].append(
                Line(
                    "// the zeros' fills (vector unit) done before the matrix unit "
                    "adds\n"
                    "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));"
                )
            )
        state["inits"] += [
            turn,
            Line(
                f"KM_UNPACK_FORMATS({x}, {w});\n"
                f"matmul_block_init({w}, {x}, 1, 1, {value.width}, 1);"
            ),
        ]
        state["after"].append(turn)
        self.compiled.includes.add("transpose_dest")
        self.compiled.products.append(
            (
                self.method.name,
                node.lineno,
                self.text(node),
                function,
                value.width,
                f"{a.buffer.name}Block",
                self.compiled.fidelity,
            )
        )
        text = (
            f"matmul_block({b.cxx}, {a.cxx}, {a.first} * {value.width}, {a.first}, "
            f"{value.dst}, 1, 1, {value.width}, 1);"
        )
        return [], [Line(text)]

    def field_product(self, node, field_, call):
        """`self.part += f(a, b)`, f `return a @ b`, a and b blocks two broadcasts
        brought (oh x kc and kc x ow tiles), self.part an fp32 field of oh x ow tiles:
        the library's product by subblocks of DST, each packed into the field with
        the packer's L1 accumulation (overwriting at the first sum after a write)."""
        function = self.tile_function_of(call)
        where = function_line(function)
        if not is_product(function.function):
            self.refuse(
                node,
                f"self.{field_} += f(...) adds a product, and the tile function "
                f"{function.name} ({where}) is not `return a @ b`",
                f"write {function.name} as `return a @ b`",
            )
        items = [
            isinstance(arg, ast.Name) and self.scope.get(arg.id) for arg in call.args
        ]
        if (
            len(items) == 2
            and not call.keywords
            and all(item and self.in_this_loop(item) for item in items)
        ):
            return self.stream_field_product(node, field_, function, *items)
        blocks, casts = [], []
        for arg in call.args:
            item = isinstance(arg, ast.Name) and self.scope.get(arg.id)
            if not item or item.kind != "arrived":
                self.refuse(
                    node,
                    f"`{ast.unparse(arg)}` is not a block a broadcast brought: a sum "
                    "in a field adds products of two broadcasts' blocks",
                    "write `if b.sender(): v = b.cast(t.tiles[...].read())`, `else: "
                    "v = b.arrive()` for each operand",
                )
            self.arrived_loops(node, arg.id, item)
            blocks.append(item.buffer)
            casts.append(item.tensor)
        if len(blocks) != 2 or call.keywords:
            self.refuse(node, "a product of two blocks", "write self.f += g(a, b)")
        stored = self.compiled.fields[field_]
        if stored.format.dtype != "float32" or len(stored.shape) != 4:
            self.refuse(
                node,
                f"self.{field_} holds sums: fp32 tiles, rows x columns of them",
                f"declare self.{field_}: km.L1[km.fp32, rows, columns, 32, 32]",
            )
        why = product_refusal(blocks[0].dtype, blocks[1].dtype)
        if why is not None:
            self.refuse(node, why, "cast tiles of matrix formats of one family")
        left, right = blocks[0].dtype, blocks[1].dtype
        turned = self.placed(node, {}, function, left, right, ("tensor", casts[1]))
        rows, columns = self.field_sides(node, field_)
        a, b = blocks
        inner = a.block // rows if rows and a.block % rows == 0 else 0
        if inner == 0 or b.block != inner * columns:
            self.refuse(
                node,
                f"self.{field_} is {rows} x {columns} tiles; the blocks are {a.block} "
                f"and {b.block} tiles, not {rows} x k and k x {columns}",
                "cast blocks of the field's rows and columns",
            )
        fidelity = function.fidelity or "HiFi4"
        if self.compiled.fidelity not in (None, fidelity):
            self.refuse(
                node,
                f"{function.name}'s fidelity is {fidelity} and another product's "
                f"{self.compiled.fidelity}: the fidelity is the compute kernel's",
                "give the class's products one fidelity",
            )
        self.compiled.fidelity = fidelity
        width = max(w for w in range(1, DST_HALF + 1) if columns % w == 0)
        height = max(
            h for h in range(1, DST_HALF // width + 1) if rows % h == 0
        )  # a subblock: height x width tiles of DST
        buffer = f"kField{camel(field_)}"
        row, column, k = (
            f"km_{n}{node.lineno}" for n in ("sub_row", "sub_column", "k")
        )
        self.producer(node, field_, "product")
        sums = self.summed.get(field_)
        if sums is None:
            sums = f"km_sums_{field_}"
            self.summed[field_] = sums
            self.compiled.summed_fields.add(f"kField{camel(field_)}")
            self.opening.append(Line(f"uint32_t {sums} = 0;  // self.{field_}'s"))
            self.opening.append(Line(sums_room_line(field_)))
            self.opening_b.append(  # mover B counts the same sums: its write's
                Line(f"uint32_t {sums} = 0;  // self.{field_}'s", mover="B")
            )
        self.compiled.includes |= {"matmul", "pack"}
        self.compiled.unpack_fp32.add(buffer)
        self.compiled.products.append(
            (
                self.method.name,
                node.lineno,
                self.text(node),
                function,
                rows * columns,
                inner,
                fidelity,
            )
        )
        self.wait(node, "compute", WAIT_INPUT)
        init = (
            f"    KM_UNPACK_FORMATS({b.name}, {a.name});\n"
            f"    matmul_block_init({a.name}, {b.name}, 0, {width}, {height}, "
            f"{inner});\n"
        )
        calls = (
            f"                matmul_block({a.name}, {b.name}, {row} * {inner} + {k}, "
            f"{k} * {columns} + {column}, 0, 0, {width}, {height}, {inner});\n"
        )
        turn = ""
        if turned:
            init, calls, turn = self.turned_blocks(
                a, b, (row, column, k), inner, columns, (height, width)
            )
        room = sums_room(field_)
        packs = (
            f"            if ({room}) {{\n"
            + "\n".join(
                f"                pack_tile<true>({i * width + j}, {buffer}, ({row} + "
                f"{i}) * {columns} + {column} + {j});"
                for i in range(height)
                for j in range(width)
            )
            + "\n            }"
        )
        text = (
            "km_ok = km_ok && km::broadcast_receive(compute, "
            f"{a.name}, {a.name}Pages) &&\n"
            f"    km::broadcast_receive(compute, {b.name}, {b.name}Pages);\n"
            f"{sums_reserve(sums, room, buffer)}"
            "if (km_ok) {\n"
            f"{init}"
            f"    KM_PACK_FORMAT({buffer});\n"
            f"    if ({room}) {{\n"
            f"        pack_reconfig_l1_acc({sums} > 0 ? 1 : 0);\n"
            "    }\n"
            f"    for (uint32_t {row} = 0; {row} < {rows}; {row} += {height}) {{\n"
            f"        for (uint32_t {column} = 0; {column} < {columns}; {column} += "
            f"{width}) {{\n"
            "            tile_regs_acquire();\n"
            f"            for (uint32_t {k} = 0; {k} < {inner}; {k}++) {{\n"
            f"{calls}"
            "            }\n"
            f"{turn}"
            "            tile_regs_commit();\n"
            "            tile_regs_wait();\n"
            f"{packs}\n"
            "            tile_regs_release();\n"
            "        }\n"
            "    }\n"
            f"    if ({room}) {{\n"
            "        pack_reconfig_l1_acc(0);\n"
            "    }\n"
            f"    {sums}++;\n"
            f"    km::broadcast_done({a.name}, {a.name}Pages);\n"
            f"    km::broadcast_done({b.name}, {b.name}Pages);\n"
            "}"
        )
        return [Line(f"{sums}++;", mover="B")], [Line(text, {"km_ok"})]

    def turned_blocks(self, a, b, names, inner, columns, sides):
        """The broadcasts' product turned: its setup, a K tile's calls (W's tiles into
        SrcB, each row of X's subblock read transposed: DST i * width .. hold row i's
        transposed sums) and the lines after the K tiles (DST turned back, the setup
        again)."""
        row, column, k = names
        height, width = sides
        init = (
            f"KM_UNPACK_FORMATS({a.name}, {b.name});\n"
            f"matmul_block_init({b.name}, {a.name}, 1, 1, {width}, 1);"
        )
        calls = "".join(
            f"                matmul_block({b.name}, {a.name}, {k} * {columns} + "
            f"{column}, ({row} + {i}) * {inner} + {k}, {i * width}, 1, 1, {width}, "
            "1);\n"
            for i in range(height)
        )
        self.compiled.includes.add("transpose_dest")
        lines = turned_tiles(height * width, 0).text.split("\n") + init.split("\n")[1:]
        turn = "".join(f"            {line}\n" for line in lines)
        return "".join(f"    {line}\n" for line in init.split("\n")), calls, turn

    def sums_field(self, node, field_):
        """A field of sums: fp32, rows x columns of tiles; its count of sums (a
        name of compute's, 0: the next sum overwrites)."""
        stored = self.compiled.fields[field_]
        if stored.format.dtype != "float32" or len(stored.shape) != 4:
            self.refuse(
                node,
                f"self.{field_} holds sums: fp32 tiles, rows x columns of them",
                f"declare self.{field_}: km.L1[km.fp32, rows, columns, 32, 32]",
            )
        self.producer(node, field_, "product")
        sums = self.summed.get(field_)
        if sums is None:
            sums = f"km_sums_{field_}"
            self.summed[field_] = sums
            self.compiled.summed_fields.add(f"kField{camel(field_)}")
            self.opening.append(Line(f"uint32_t {sums} = 0;  // self.{field_}'s"))
            self.opening.append(Line(sums_room_line(field_)))
            self.opening_b.append(  # mover B counts the same sums: its write's
                Line(f"uint32_t {sums} = 0;  // self.{field_}'s", mover="B")
            )
        return sums

    def stream_field_product(self, node, field_, function, a, b):
        """`self.part += f(xb, wb)` in a zip of two streams, self.part an fp32 field of
        1 x n tiles: each block's product (its valid K tiles) by subblocks of DST,
        packed into the field with the packer's L1 accumulation; each W block used
        as it arrives and freed. The field is as wide as L1 holds, not as DST."""
        sums = self.sums_field(node, field_)
        rows, columns = self.field_sides(node, field_)
        if rows != 1:
            self.refuse(
                node,
                f"self.{field_} is {rows} rows of tiles; a stream's block is one row "
                "of X's tiles",
                f"declare self.{field_}: km.L1[km.fp32, 1, n, 32, 32]",
            )
        self.stream_operands(node, function, a, b, columns, f"self.{field_}")
        state = self.loop_states[-1]
        state["whole"] = True  # the product takes the block, not its tiles
        # mover B counts the loop's blocks into each field the loop adds to
        state.setdefault("sums", []).append(sums)
        buffer = f"kField{camel(field_)}"
        inner = a.buffer.block
        width = max(w for w in range(1, DST_HALF + 1) if columns % w == 0)
        valid = f"km_valid{self.stream_loops[-1]}"
        column, k = (f"km_{n}{node.lineno}" for n in ("sub_column", "k"))
        weight = ("tensor", state["tensors"][b.buffer.name])
        left, right = a.buffer.dtype, b.buffer.dtype
        turned = self.placed(node, state, function, left, right, weight)
        call = (
            f"matmul_block({a.cxx}, {b.cxx}, {k}, {k} * {columns} + {column}, 0, 0, "
            f"{width}, 1, {inner});"
        )
        init = (
            f"KM_UNPACK_FORMATS({b.buffer.name}.mover_a, {a.buffer.name}.mover_a);\n"
            f"matmul_block_init({a.buffer.name}.mover_a, {b.buffer.name}.mover_a, 0, "
            f"{width}, 1, {inner});"
        )
        turn = ""
        if turned:
            call, init, turn = self.turned_sums(
                a, b, f"{k} * {columns} + {column}", k, width
            )
        if state["function"] is None:
            state["function"] = "product"
            state["product_init"] = init
            state["inits"].append(Line(f"{init}\nKM_PACK_FORMAT({buffer});"))
        own, restore = product_setups(state.get("product_init"), init)
        self.compiled.includes |= {"pack"}
        self.compiled.unpack_fp32.add(buffer)
        self.compiled.products.append(
            (
                self.method.name,
                node.lineno,
                self.text(node),
                function,
                columns,
                inner,
                self.compiled.fidelity,
            )
        )
        room = sums_room(field_)
        packs = (
            f"        if ({room}) {{\n"
            + "\n".join(
                f"            pack_tile<true>({j}, {buffer}, {column} + {j});"
                for j in range(width)
            )
            + "\n        }"
        )
        text = (
            f"{sums_reserve(sums, room, buffer)}"
            "if (km_ok) {\n"
            f"{own}"
            f"    if ({room}) {{\n"
            f"        pack_reconfig_l1_acc({sums} > 0 ? 1 : 0);\n"
            "    }\n"
            f"    for (uint32_t {column} = 0; {column} < {columns}; {column} += "
            f"{width}) {{\n"
            "        tile_regs_acquire();\n"
            f"        for (uint32_t {k} = 0; {k} < {valid}; {k}++) {{\n"
            f"            {call}\n"
            "        }\n"
            f"{turn}"
            "        tile_regs_commit();\n"
            "        tile_regs_wait();\n"
            f"{packs}\n"
            "        tile_regs_release();\n"
            "    }\n"
            f"    if ({room}) {{\n"
            "        pack_reconfig_l1_acc(0);\n"
            "    }\n"
            f"{restore}"
            f"    {sums}++;\n"
            "}"
        )
        return [], [Line(text, {"km_ok", valid})]

    def turned_sums(self, a, b, w_tile, x_tile, width):
        """The field sums' product turned: its call (W's tile `w_tile` into SrcB, X's
        `x_tile` read transposed into SrcA, DST tiles 0.. the transposed sums), its
        setup, and the lines after a subblock's K tiles: its DST tiles transposed
        back, the setup again."""
        x, w = f"{a.buffer.name}.mover_a", f"{b.buffer.name}.mover_a"
        call = (
            f"matmul_block({b.cxx}, {a.cxx}, {w_tile}, {x_tile}, 0, 1, 1, {width}, 1);"
        )
        init = (
            f"KM_UNPACK_FORMATS({x}, {w});\n"
            f"matmul_block_init({w}, {x}, 1, 1, {width}, 1);"
        )
        self.compiled.includes.add("transpose_dest")
        lines = turned_tiles(width, 0).text.split("\n") + init.split("\n")[1:]
        return call, init, "".join(f"        {line}\n" for line in lines)

    def field_sides(self, node, field_):
        """A field of tiles' rows and columns, its symbols the plan's."""
        sides = []
        for side in self.compiled.fields[field_].shape[:2]:
            if isinstance(side, str):
                if not isinstance(self.plan.get(side), int):
                    self.refuse(
                        node,
                        f"self.{field_}'s {side} is the plan's, and the plan has none",
                        f"make the program with {spelling('plan')}={{{side!r}: n}}",
                    )
                side = self.plan[side]
            sides.append(side)
        return tuple(sides)

    def accumulate(self, node, value):
        """`total += tile` inside a stream loop, or `total += block` of a broadcast."""
        source = node.value
        if not isinstance(source, ast.Name) or source.id not in self.scope:
            self.refuse(node, "add a tile of a stream", "write total += tile")
        item = self.scope[source.id]
        self.other_tile_line(node)
        if item.kind == "streamed":
            self.compiled.includes |= {"add"}
            text = add_tile(item.cxx, item.first, value.dst, self.scratch())
            return [], [Line(text, {item.cxx, item.first})]
        if item.kind == "block":
            self.compiled.includes |= {"add"}
            loop = (
                f"for (uint32_t {item.first} = 0; {item.first} < {item.count}; "
                f"{item.first}++) {{"
            )
            body = add_tile(item.cxx, item.first, value.dst, self.scratch())
            return [], [Block(loop, "}", [Line(body)], {item.cxx})]
        self.refuse(node, f"{source.id} is a {item.kind}", "add a tile of a stream")

    # Streams -----------------------------------------------------------------------

    def stream_options(self, node, call, write):
        """(range, block, pattern C++, pattern letters or None, repeat C++); a read's
        tiles a row (`row=`) as self.row_width."""
        if len(call.args) != 1 or not self.is_range(call.args[0]):
            self.refuse(
                node,
                "a stream of a page range",
                "write km.Stream(tensor[part], block=8)",
            )
        pages = self.range_of(call.args[0])
        block, pattern, letters, repeat = None, None, None, "1"
        self.row_width = 1
        self.stream_word = None  # a block given at each submit: its parameter
        for keyword_ in call.keywords:
            key = keyword_.arg
            if key == spelling("block"):
                block = self.sized_block(keyword_.value)
            elif key == spelling("row") and not write:
                self.row_width = self.constant_int(keyword_.value)
            elif key == "pattern":
                pattern, letters = self.pattern(keyword_.value)
            elif key == spelling("repeat"):
                repeat = self.integer(keyword_.value)
            elif key == spelling("write_stream"):
                continue
            elif key == spelling("prefetch") and not write:
                prefetch_checked(self, node, keyword_.value)
            else:
                self.refuse(node, f"a stream has no {key}=", "use block=, pattern=")
        block = self.row_width if block is None else block
        if block < 1 or block % self.row_width and self.row_width % block:
            self.refuse(
                node,
                "a block of no pages, or not of whole rows nor a part of one",
                "give block= a count, a multiple of row= or a part of it",
            )
        if self.stream_word is not None and (
            self.row_width > 1 or pages.rect is not None
        ):
            self.refuse(
                node,
                f"a block given at each submit ({self.stream_word}) of rows of a "
                "rectangle or of row= is not built",
                "give block= a constant, or stream one run of pages",
            )
        if pattern is None and letters is None:  # unstated: the default
            letters = WRITE_PATTERN if write else READ_PATTERN
        if pages.rect is not None:
            columns = pages.rect[1]
            if isinstance(columns, str):
                self.refuse(
                    node,
                    "a stream's blocks are whole rows of its rectangle, and this "
                    "rectangle's columns are a word",
                    f"stream {pages.tensor}.{spelling('tiles')}[r0:r1, c:c + n], n a "
                    "constant, or one row of it",
                )
            if block % columns:
                self.refuse(
                    node,
                    f"a block of a rectangle of {columns} columns is whole rows of it",
                    f"give block= a multiple of {columns}",
                )
            pages.rect = (*pages.rect[:3], block)
        return pages, block, pattern, letters, repeat

    def pattern(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                Pattern(node.value)
            except ValueError as error:
                self.refuse(node, str(error), "write letters A and B")
            return None, node.value
        if isinstance(node, ast.Name) and self.scope.get(node.id, Value("")).kind == (
            "pattern"
        ):
            return node.id, None
        self.refuse(
            node, "a pattern is letters or a pattern parameter", 'write pattern="AB"'
        )

    def tensor_format(self, node, tensor):
        if tensor in self.compiled.regions:  # a thread's region: words, in its pages
            return "uint32", REGION_PAGE
        example = self.tensors.get(tensor)
        if not front.is_tensor(example):
            self.refuse(
                node,
                f"the program is made without an example tensor for {tensor}",
                f"make it with {tensor}=a_tensor (its format and page size)",
            )
        return example.dtype, example.page_bytes

    def pair(self, role, number, dtype, page_bytes, block, items, what):
        """The program's pair `number` of `role`, made or grown."""
        name = f"k{role.capitalize()}{number}"
        for buffer in self.compiled.buffers:
            if buffer.name == name:
                if (buffer.dtype, buffer.page_bytes, buffer.block) != (
                    dtype,
                    page_bytes,
                    block,
                ):
                    break
                return buffer
        else:
            first = FIRST_READ if role == "read" else FIRST_WRITE
            taken = {b.index_a for b in self.compiled.buffers} | {
                b.index_b for b in self.compiled.buffers
            }
            taken |= self.compiled.reserved
            index = first
            while index in taken or index + 1 in taken:
                index += 2
            buffer = Buffer(
                name,
                role,
                index,
                index + 1,
                dtype,
                page_bytes,
                (items * block, items * block),
                block,
                what,
            )
            if index + 1 > LAST_INDEX:
                raise Refused(f"{self.compiled.name}: more pairs than buffers")
            self.compiled.buffers.append(buffer)
            return buffer
        return self.pair(role, f"{number}b", dtype, page_bytes, block, items, what)

    def pattern_cxx(self, buffer, pattern, letters):
        """The C++ of a stream's pattern: a parameter's, or the pair's constant."""
        if pattern is not None:
            return pattern
        constant = f"{buffer.name}Pattern"
        known = buffer.patterns.get(constant)
        if known is not None and known != letters:
            constant = f"{buffer.name}Pattern{len(buffer.patterns)}"
        buffer.patterns[constant] = letters
        return constant

    def stream_cxx(self, kind, variable, pages, buffer, pattern, repeat):
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        if pages.items is not None:  # items of a page: an item a buffer page
            tensor = f"km::items({tensor}, {pages.items[0]}, {pages.items[1]})"
        direction = "km::kMoveRead" if kind == "read" else "km::kMoveWrite"
        if pages.rect is not None:  # blocks of whole rows of a rectangle
            rows, columns, row_step, block = pages.rect
            per = block // columns  # rows a block
            blocks = f"(({rows}) + {per - 1}) / {per}"
            between = block_step(per, row_step)
            walk = (
                f"km::Walk{{{pages.first}, {blocks}, {between}, {per}, "
                f"{row_step}, {columns}, 1, {columns}, {block}, ({rows}) - ({blocks} "
                f"- 1) * {per}, {columns}}}"
            )
            return (
                f"// the rectangle's rows, {columns} tiles each, {row_step} apart\n"
                f"const km::Stream {variable}(\n"
                f"    {direction}, {tensor}, {walk}, {repeat}, "
                f"{buffer.name}, {pattern}\n"
                ");"
            )
        return (
            f"const km::Stream {variable}(\n"
            f"    {direction}, {tensor}, {pages.first}, "
            f"{pages.count}, {buffer.name}Block, {repeat}, {buffer.name}, {pattern}\n"
            ");"
        )

    def items_alone(self, node, pages):
        """Refuses what items do not take: a stream of rows of them."""
        if self.row_width > 1:
            self.refuse(
                node,
                "items are not tiles: a stream of them has no rows",
                f"drop {spelling('row')}=",
            )

    def pages_only(self, node, pages):
        """Refuses a range of items where a line takes pages."""
        if pages.items is not None:
            self.refuse(
                node,
                f"{pages.tensor}.{spelling('items')}(...) is a stream's: this line "
                "moves whole pages",
                f"write {pages.tensor}.{spelling('pages')}(first, count), or stream "
                "the items",
            )

    def write_stream(self, node, name, call):
        written = [k for k in call.keywords if k.arg == spelling("write_stream")]
        if not written or not (
            isinstance(written[0].value, ast.Constant) and written[0].value.value
        ):
            self.refuse(
                node,
                "a stream to iterate goes in a for line",
                f"write `for tile in km.{spelling('stream')}(...)`, or "
                "write=True for a write stream",
            )
        pages, block, pattern, letters, repeat = self.stream_options(node, call, True)
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        if pages.items is not None:  # the buffer's pages are the items
            page_bytes = pages.items[0]
        self.moved(node, pages.tensor, "write stream")
        number = self.writes
        self.writes += 1
        buffer = self.pair(
            "write",
            number,
            dtype,
            page_bytes,
            block,
            WRITE_ITEMS,
            f"line {node.lineno}: {self.text(node)}",
        )
        self.block_word(node, buffer)
        pattern_text = self.pattern_cxx(buffer, pattern, letters)
        variable = f"km_write{number}"
        self.scope[name] = Value(
            "write",
            variable,
            count=pages.count,
            buffer=buffer,
            pattern=pattern_text,
            letters=letters or "",
        )
        text = self.stream_cxx("write", variable, pages, buffer, pattern_text, repeat)
        uses = range_uses(pages, self) | pattern_uses(pattern)
        self.wait(node, "mover A", WAIT_DATA)
        self.wait(node, "mover B", WAIT_DATA)
        self.scope[name].depth = len(self.loops)
        fed = f"km_fed_{name}"  # compute's blocks of it, when a loop's bodies feed it
        return [Glue(text, {variable}, uses)], [Glue(f"uint32_t {fed} = 0;", {fed})]

    def loop(self, node):
        if node.orelse:
            self.refuse(node, "a loop with an else", "write the else after it")
        iterator = node.iter
        self.held_not_iterated(node, iterator)
        if self.self_call(iterator, "arrive_messages"):
            return self.arrive_loop(node)
        if (
            isinstance(iterator, ast.Name)
            and self.scope.get(iterator.id, Value("")).kind == "arrived"
        ):
            return self.block_loop(node, self.scope[iterator.id])
        if (
            isinstance(iterator, ast.Name)
            and self.scope.get(iterator.id, Value("")).kind == "drained"
        ):
            return self.drain_loop(node, iterator.id)
        if self.construct(iterator) == "stream":
            return self.read_loop(node, iterator)
        streams = self.zipped(node)
        if streams is not None:
            return self.read_loop(node, None, streams)
        if self.entries_call(iterator) is not None or self.entries_named(iterator):
            return self.entries_loop(node)
        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range"
        ):
            return self.range_loop(node, iterator)
        self.refuse(
            node,
            "a loop of level 1 is over a stream or a range",
            f"write `for tile in km.{spelling('stream')}(...)` or "
            "`for i in range(n)`",
        )

    def range_loop(self, node, call):
        if not isinstance(node.target, ast.Name):
            self.refuse(node, "one loop name", "write for i in range(n)")
        name = node.target.id
        self.check_name(node, name)
        if not 1 <= len(call.args) <= 2:
            self.refuse(node, "range(n) or range(first, n)", "give one or two")
        first = self.integer(call.args[0]) if len(call.args) == 2 else "0"
        end = self.integer(call.args[-1])
        uses = set().union(*(self.uses(arg) for arg in call.args))
        self.scope[name] = Value("int", name)
        loop = {"name": name, "first": first, "end": end, "casts": []}
        loop["body"] = node.body  # a running total's line stands in it (carries)
        self.loops.append(loop)
        before = set(self.scope)
        body_m, body_c = self.statements(node.body)
        body_c += loop.pop("frees", [])
        body_c += self.carried_back(loop)
        self.scoped("loop", node, before)
        self.loops.pop()
        self.scope.pop(name)
        head = f"for (uint32_t {name} = {first}; {name} < {end}; {name}++) {{"
        head_c = head
        if self.compute_waits:
            head_c = (
                f"for (uint32_t {name} = {first}; km_ok && {name} < {end}; {name}++) {{"
            )
        movement = [Block(head, "}", body_m, uses)] if body_m else []
        compute = [Block(head_c, "}", body_c, uses | {"km_ok"})] if body_c else []
        compute += loop.pop("after", [])  # a sum held across the loop, freed
        for broadcast in loop["casts"]:
            m, c = self.broadcast_end(broadcast, end, uses)
            movement += m
            compute += c
        m, c = self.counted_ends(loop)
        movement, compute = movement + m, compute + c
        movement += self.fed_moves(node, loop, body_m)
        return movement, compute

    def zipped(self, node):
        """The streams of `for a, b in zip(km.Stream(..), km.Stream(..)):`, each with
        its name, or None when the loop is not over a zip of streams."""
        iterator = node.iter
        if not (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == spelling("zip")
            and iterator.args
            and all(self.construct(arg) == "stream" for arg in iterator.args)
        ):
            return None
        target = node.target
        if (
            iterator.keywords
            or not isinstance(target, ast.Tuple)
            or len(target.elts) != len(iterator.args)
            or not all(isinstance(name, ast.Name) for name in target.elts)
        ):
            self.refuse(
                node,
                "streams read in step name one tile each",
                f"write `for a, b in {spelling('zip')}(km.{spelling('stream')}(...), "
                f"km.{spelling('stream')}(...))`",
            )
        return [(name.id, call) for name, call in zip(target.elts, iterator.args)]

    def read_loop(self, node, call, streams=None):
        """A loop over one read stream, or over several read in step (`streams`:
        each tile's name and its stream's call): one move of them and of the write
        streams the loop pushes to; compute takes block k of each from its half."""
        if streams is None:
            if not isinstance(node.target, ast.Name):
                self.refuse(node, "one name a tile", "write for tile in km.Stream(...)")
            streams = [(node.target.id, call)]
        number = self.reads  # the loop's own names: block, tile, valid, out, room
        block_var, tile, valid = (
            f"km_block{number}",
            f"km_tile{number}",
            f"km_valid{number}",
        )
        reads = []
        for name, stream_call in streams:
            self.check_name(node, name)
            pages, block, pattern, letters, repeat = self.stream_options(
                node, stream_call, False
            )
            if self.row_width > 1 and len(streams) > 1:
                self.refuse(
                    node,
                    "a stream of rows is read alone",
                    f"write `for row in km.{spelling('stream')}(..., row=2)`",
                )
            dtype, page_bytes = self.tensor_format(node, pages.tensor)
            if pages.items is not None:  # the buffer's pages are the items
                page_bytes = pages.items[0]
                self.items_alone(node, pages)
            index = self.reads
            self.reads += 1
            buffer = self.pair(
                "read",
                index,
                dtype,
                page_bytes,
                block,
                READ_ITEMS,
                f"line {node.lineno}: {self.text(node)}",
            )
            word = self.stream_word
            self.block_word(node, buffer)
            pattern_text = self.pattern_cxx(buffer, pattern, letters)
            if dtype == "float32" and pages.items is None:
                self.compiled.unpack_fp32.add(buffer.name)
            self.moved(node, pages.tensor, "read stream")
            reads.append(
                {
                    "name": name,
                    "variable": f"km_read{index}",
                    "half": f"km_half{index}",
                    "pages": pages,
                    "block": block,
                    "pattern": pattern,
                    "pattern_text": pattern_text,
                    "letters": letters,
                    "repeat": repeat,
                    "buffer": buffer,
                    "width": self.row_width,
                    "word": word,
                }
            )
        first = reads[0]
        rows_in_step = False  # blocks of whole rows, as many blocks: a product's
        for other in reads[1:]:
            if (
                other["block"],
                other["pages"].count,
                other["repeat"],
                other["word"],
            ) == (first["block"], first["pages"].count, first["repeat"], first["word"]):
                continue
            if other["word"] or first["word"]:
                self.refuse(
                    node,
                    "streams read in step, one with a block given at each submit, "
                    "have one count and the same block parameter",
                    "give each stream block= the same parameter",
                )
            if stream_units(other) & stream_units(first) and other["repeat"] == (
                first["repeat"]
            ):
                rows_in_step = True
                continue
            self.refuse(
                node,
                "streams read in step have one count, block and repeat (or, for a "
                "product, a row of the second's rectangle a tile of the first's)",
                "give each the same count of pages, block= and repeat=, or stream "
                f"x.{spelling('tiles')}[0:1, k0:k1] with {spelling('block')}=b beside "
                f"w.{spelling('tiles')}[k0:k1, c:c + n] with {spelling('block')}=b * n",
            )
        for read in reads:
            self.scope[read["name"]] = Value(
                "streamed",
                read["half"],
                first=tile,
                buffer=read["buffer"],
                width=read["width"],
            )
        before = set(self.scope)
        writes_before = [v for v in self.scope.values() if v.kind == "write"]
        pushed_before = {id(v): v.pushed for v in writes_before}
        self.stream_loops.append(number)
        self.loop_states.append(
            {
                "function": None,
                "other": None,
                "inits": [],
                "after": [],
                "block": reads[0]["block"],
                "first": reads[0]["half"],
                "count": reads[0]["pages"].count,  # the loop's tiles, as written
                "tensors": {r["buffer"].name: r["pages"].tensor for r in reads},
                "letters": [r["letters"] for r in reads],  # None: a pattern word
                "repeat": reads[0]["repeat"],
                "most": held_most(reads[0]["pages"]),  # the most pages, where known
                "line": node.lineno,
            }
        )
        self.held_cast_movers(node, [r["letters"] for r in reads])
        body_m, body_c = self.statements(node.body)
        state = self.loop_states.pop()
        self.results_pushed(node, state)
        self.stream_loops.pop()
        if first["block"] < first["width"]:
            self.row_passes_read(node, first, state, number)
        if body_m:
            self.refuse(
                node.body[0],
                "inside a stream loop only compute's lines",
                "move the line before or after the loop",
            )
        if rows_in_step and (state["function"] != "product" or state["other"]):
            self.refuse(
                node,
                "streams in step with blocks of different sizes are a product's "
                "operands: a tile of one is a row of the other",
                "keep the loop to `acc += self.f(a, b)`, f `return a @ b`",
            )
        self.scoped("loop", node, before)
        for read in reads:
            self.scope.pop(read["name"], None)
        pushed = [v for v in writes_before if v.pushed and not pushed_before[id(v)]]
        # The movers: these streams and the write streams they feed, in one move.
        movement = []
        for read in reads:
            stream_uses = (
                range_uses(read["pages"], self)
                | pattern_uses(read["pattern"])
                | text_uses(read["repeat"], self)
            )
            movement.append(
                Glue(
                    self.stream_cxx(
                        "read",
                        read["variable"],
                        read["pages"],
                        read["buffer"],
                        read["pattern_text"],
                        read["repeat"],
                    ),
                    {read["variable"]},
                    stream_uses,
                )
            )
        variables = [read["variable"] for read in reads] + [v.cxx for v in pushed]
        moved = ", ".join(variables)
        move = Line(
            f"if (!movement.move({moved})) {{\n    return false;\n}}", set(variables)
        )
        movement.append(move)
        movement.extend(  # a sum a block into each field, as compute counts them
            Line(
                f"{sums} += ({first['pages'].count} + "
                f"{first['buffer'].name}Block - 1) / {first['buffer'].name}Block;",
                range_uses(first["pages"], self),
                mover="B",
            )
            for sums in state.get("sums", ())
        )
        for read in reads:  # after the loop: what each mover moved (a mover's body)
            self.scope[read["name"]] = Value("moved", f"km_moved{number}", move=move)
        self.wait(node, "mover A", WAIT_DATA)
        self.wait(node, "mover B", WAIT_DATA)
        self.wait(node, "compute", WAIT_INPUT)
        # Compute: block k of each stream from the half its owner filled; its tiles.
        self.compute_waits = True
        buffer, pages, repeat = first["buffer"], first["pages"], first["repeat"]
        blocks = f"({pages.count} + {buffer.name}Block - 1) / {buffer.name}Block"
        per_turn = f"km_blocks{number}"
        count_uses = text_uses(pages.count, self)
        compute = []
        checked = set()
        for read in reads:
            if read["pattern"] is not None and read["pattern"] not in checked:
                checked.add(read["pattern"])
                compute.append(
                    Line(
                        f"if (!{read['pattern']}.valid()) {{\n    return;\n}}",
                        {read["pattern"]},
                    )
                )
        compute.append(
            Glue(f"const uint32_t {per_turn} = {blocks};", {per_turn}, count_uses)
        )
        compute += state["inits"]
        total = per_turn if repeat == "1" else f"{per_turn} * {repeat}"
        inner = []
        for read in reads:
            half, name = read["half"], read["buffer"].name
            inner.append(
                Line(
                    f"const uint32_t {half} = km::block_half({name}, "
                    f"{read['pattern_text']}, {block_var});\n"
                    f"km_ok = compute.wait_front({half}, {name}Block);\n"
                    f"if (!km_ok) {{\n    break;\n}}",
                    uses=pattern_uses(read["pattern"]),
                )
            )
        inner.append(
            Glue(
                f"const uint32_t {valid} = {block_var} % {per_turn} + 1 == "
                f"{per_turn} ? {pages.count} - ({per_turn} - 1) * "
                f"{buffer.name}Block : {buffer.name}Block;",
                {valid},
                count_uses,
            )
        )
        for write in pushed:
            out, room = out_names(number, write, state)
            # a row read in passes: its blocks written in the last pass alone
            written, when = state.get("written", (block_var, ""))
            inner.append(
                Line(
                    f"const uint32_t {out} = km::block_half({write.buffer.name}, "
                    f"{write.pattern}, {written});\n"
                    f"const bool {room} = {when}compute.reserve_back({out}, "
                    f"{write.buffer.name}Block);",
                    uses=pattern_uses(write.pattern),
                )
            )
            self.wait(node, "compute", WAIT_OUTPUT)
        if pushed and state["function"] is None and not state.get("body_pushed"):
            inner.append(
                Line(
                    f"KM_UNPACK_FORMAT_A({first['half']});\n"
                    f"copy_init({first['half']});"
                )
            )
        width = state.get("step", first["width"])
        step = f"{tile}++" if width == 1 else f"{tile} += {width}"
        loop = f"for (uint32_t {tile} = 0; {tile} < {valid}; {step}) {{"
        if body_c and state.get("whole"):  # the body takes the block whole
            inner += body_c
        elif body_c:
            inner.append(Block(loop, "}", body_c, {valid}))
        for write in pushed:
            out, room = out_names(number, write, state)
            inner.append(
                Line(
                    f"if ({room}) {{\n    cb_push_back({out}, "
                    f"{write.buffer.name}Block);\n}}"
                )
            )
        inner += [
            Line(f"cb_pop_front({read['half']}, {read['buffer'].name}Block);")
            for read in reads
        ]
        head = (
            f"for (uint32_t {block_var} = 0; km_ok && {block_var} < {total}; "
            f"{block_var}++) {{"
        )
        uses = {per_turn, "km_ok"} | text_uses(repeat, self)
        compute.append(Block(head, "}", inner, uses))
        compute += state["after"]
        return movement, compute

    def call_statement(self, node, call):
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == spelling("push"):
            return self.push(node, call)
        if isinstance(func, ast.Attribute) and func.attr == spelling("write"):
            return self.write(node, call)
        if isinstance(func, ast.Attribute) and func.attr == spelling("put"):
            return self.put_line(node, call)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("stream_end")
            and isinstance(func.value, ast.Name)
            and getattr(self.scope.get(func.value.id), "kind", None) == "drained"
        ):
            return self.drained_end(node, func.value.id)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("fill")
            and self.field_name(func.value)
        ):
            return self.fill(node, call, self.field_name(func.value))
        if (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("zero")
            and self.field_name(func.value)
        ):
            return self.zero(node, call, self.field_name(func.value))
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
            self.field_use(func.value)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("drain")
            and self.construct(func.value) == "stream"
        ):
            return self.drain(node, call)
        for kind in ("send", "forward", "consume"):
            if self.self_call(call, kind):
                return self.message_call(node, call, kind)
        if self.self_call(call, "direct_message"):
            self.direct_unnamed(node)
        construct = self.construct(call)
        if construct == "cxx":
            return self.inline(node, call)
        if construct == "reduce":
            return self.reduce(node, None, call)
        if construct == "all_reduce":
            return self.reduce(node, None, call, everywhere=True)
        if construct == "result":
            self.refuse(
                node,
                "a command's result word is the handler's return",
                "write `return word` last under `if self.rank == self.leader:`",
            )
        self.refuse(
            node, "a call level 1 does not know", "write a move, or an inline body"
        )

    # Tile functions ----------------------------------------------------------------

    def tile_function_of(self, node):
        """The tile function a call is (`self.f(...)` of a @km.tile_function of the
        class, or `f(...)` of the module's), or None."""
        if not isinstance(node, ast.Call):
            return None
        if id(node) in self.expression_calls:
            return self.expression_calls[id(node)][1]
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            for owner in inspect.getmro(self.method.owner):
                found = vars(owner).get(func.attr)
                if found is not None:
                    return found if isinstance(found, front.TileFunction) else None
            return None
        if isinstance(func, ast.Name) and func.id not in self.scope:
            found = self.names.get(func.id)
            if isinstance(found, front.TileFunction):
                return found
            if (
                inspect.isfunction(found)
                and node.args
                and all(
                    isinstance(arg, ast.Name)
                    and self.scope.get(arg.id, Value("")).kind == "streamed"
                    for arg in node.args
                )
            ):  # a plain function of a stream's tiles
                return front.TileFunction(found)
        return None

    def other_tile_line(self, node):
        """A tile line of compute that is not a tile function's call: refused in a
        loop that calls one (its setup holds the vector unit's registers)."""
        if not self.loop_states:
            return
        state = self.loop_states[-1]
        if state["function"] is not None:
            self.refuse(
                node,
                f"this loop calls the tile function {state['function']}, whose setup "
                "holds the vector unit for the loop",
                "write the loop's tile arithmetic in that tile function",
            )
        state["other"] = node

    def tile_expression(self, node, value):
        """`out.push(x * y)`, `t = km.silu(x) * y`: an expression of the loop's tiles,
        numbers and km's tile arithmetic, made a tile function of those tiles (its
        parameters in the order the expression names them) and returned as that
        function's call; None when `value` names no tile of the loop."""
        if not isinstance(value, (ast.BinOp, ast.UnaryOp, ast.Call)):
            return None
        if self.tile_function_of(value) is not None:
            return None
        arithmetic = {spelling(k) for k in NAMES if k.startswith("math_")}
        for call in ast.walk(value):
            if isinstance(call, ast.Call) and not (
                (isinstance(call.func, ast.Name) and call.func.id == "abs")
                or (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "km"
                    and call.func.attr in arithmetic
                )
            ):
                return None  # a call of no tile arithmetic: not a tile expression
        names = [n for n in ast.walk(value) if isinstance(n, ast.Name)]
        tiles = []  # the function's parameters: the loop's tiles and float words
        for n in names:
            kind = self.scope.get(n.id, Value("")).kind
            if kind in ("streamed", "float") and n.id not in tiles:
                tiles.append(n.id)
        if not any(self.scope[t].kind == "streamed" for t in tiles):
            return None
        for n in names:
            kind = self.scope.get(n.id, Value("")).kind if n.id in self.scope else None
            if kind == "int":
                self.refuse(
                    node,
                    f"`{n.id}` is an int word, and a tile expression's run-time "
                    "scalar is a float word (a float32's bits, loaded by the unit "
                    "program's setup)",
                    f"annotate it `{n.id}: float` and pass the number",
                )
            if n.id == "self" or kind not in (None, "streamed", "float"):
                what = "self" if n.id == "self" else f"`{n.id}` ({kind})"
                self.refuse(
                    node,
                    f"a tile expression reads the loop's tiles, numbers, module "
                    f"constants and km's tile arithmetic, and {what} is none of them: "
                    "the unit program's numbers are immediates built into its code",
                    "write a number, a module constant or a float word there",
                )
        text = ast.unparse(value)
        name = f"tile_line_{node.lineno}"
        filename = (
            f"<the tile expression of {Path(self.compiled.source).name}, "
            f"line {node.lineno}>"
        )
        source = f"def {name}({', '.join(tiles)}): return {text}\n"
        lines = ["\n"] * (node.lineno - 1) + [source]
        linecache.cache[filename] = (len(source), None, lines, filename)
        names_of = dict(self.names)
        exec(compile("".join(lines), filename, "exec"), names_of)
        function = front.TileFunction(names_of[name])
        call = ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()),
            args=[ast.Name(id=t, ctx=ast.Load()) for t in tiles],
            keywords=[],
        )
        ast.copy_location(call, value)
        self.expression_calls[id(call)] = (call, function)
        return call

    def tile_call(self, node, call, name):
        """`t = self.f(a, b)` (or pushed): a, b the loop's stream tiles (or a row of
        `row=` tiles and fields held as resident rows), copied into DST as f's unit
        program reads them, the emitted body on the math CPU, the result in DST 0 (a
        row: 0 .. width - 1). Returns (no movement, compute's lines)."""
        function = self.tile_function_of(call)
        self.held_not_traced(node, call, function)
        traced_placement = self.traced_placement(node, function)
        if not self.loop_states:
            self.refuse(
                node,
                f"the tile function {function.name} is called on a stream's tiles, "
                "in the stream's loop",
                f"call it inside `for tile in km.{spelling('stream')}(...)`",
            )
        state = self.loop_states[-1]
        if state["other"] is not None:
            self.refuse(
                node,
                f"this loop has other tile lines (line {state['other'].lineno}), and "
                "a tile function's setup holds the vector unit for the loop",
                "write the loop's tile arithmetic in the tile function",
            )
        if call.keywords or not call.args:
            self.refuse(node, "a tile function takes tiles", "call f(tile, ...)")
        if function.fidelity is not None and matrix_product(function.function) is None:
            self.refuse(
                node,
                f"fidelity= is a product's (the matrix unit's), and {function.name} "
                "is the vector unit's",
                f"drop fidelity= from {function.name}",
            )
        operands = []  # (Value of a stream tile, or None, field name or None)
        scalars = {}  # {the function's parameter: the float word passed for it}
        names = list(inspect.signature(function.function).parameters)
        for arg in call.args:
            field = self.field_name(arg)
            if field is not None:
                operands.append((None, field))
                continue
            if self.scalar_argument(node, arg):
                scalars[names[len(operands)]] = arg.id
                operands.append((None, None))
                continue
            item = isinstance(arg, ast.Name) and self.scope.get(arg.id)
            if (
                not item
                or item.kind != "streamed"
                or not item.cxx.startswith("km_half")
            ):
                self.refuse(
                    node,
                    f"a tile function takes the tiles of the loop's streams, and "
                    f"`{ast.unparse(arg)}` is not one",
                    "pass the loop's tiles: f(a, b) in `for a, b in "
                    f"{spelling('zip')}(km.{spelling('stream')}(...), ...)`",
                )
            operands.append((item, None))
        streamed = [item for item, _ in operands if item is not None]
        width = max(item.width for item in streamed) if streamed else 1
        fields = [field for _, field in operands if field is not None]
        if self.dst_live is not None or self.dst_next != 0:
            self.refuse(
                node,
                "a tile function's result is DST 0, which another tile holds",
                "write or push that tile first",
            )
        if scalars and width > 1:
            self.refuse(
                node,
                f"{', '.join(scalars.values())}: a run-time scalar of a tile function "
                "called on a row: the row form's numbers are immediates",
                "call it on a stream's tiles, one tile a side",
            )
        formats = []
        for item, field in operands:
            if item is not None:
                formats.append(item.buffer.dtype)
            elif field is not None:
                formats.append(self.compiled.fields[field].format.dtype)
            else:
                formats.append(SCALAR)
        why = tile_format_refusal([f for f in formats if f != SCALAR])
        if why is not None:
            self.refuse(node, f"{function.name}: {why}", "give it tiles of those")
        unit, cxx = self.tile_unit(node, function, formats, width, fields)
        if width == 1 and fields:  # each field's parameter
            given = {f: names[i] for i, (_, f) in enumerate(operands) if f is not None}
            self.fields_by_row(node, function, unit, given)
        if unit.row_operands or unit.col_operands:  # unpacked broadcast (unary_bcast)
            self.compiled.includes.add("bcast")
        if state["function"] not in (None, cxx):
            self.refuse(
                node,
                f"this loop calls the tile function {state['function']} already; a "
                "loop calls one",
                "make one tile function of the two",
            )
        parameters = list(inspect.signature(function.function).parameters)
        products = [  # each product's operands: (a's tile, w's tile)
            (operands[parameters.index(p.a)][0], operands[parameters.index(p.w)][0])
            for p in unit.products
        ]
        for (a, w), p in zip(products, unit.products):
            why = product_refusal(a.buffer.dtype, w.buffer.dtype)
            if why is not None:
                self.refuse(
                    node,
                    f"{why}, in {function.name}",
                    "stream tensors of matrix formats of one family into it",
                )
            self.compiled.placements.append(
                (
                    self.method.name,
                    node.lineno,
                    self.text(node),
                    a.buffer.dtype,
                    w.buffer.dtype,
                    traced_placement.get((p.a, p.w), "none"),
                    None,
                )
            )
        if products:
            fidelity = unit.fidelity
            if self.compiled.fidelity not in (None, fidelity):
                self.refuse(
                    node,
                    f"{function.name}'s fidelity is {fidelity} and another product's "
                    f"{self.compiled.fidelity}: the fidelity is the compute kernel's",
                    "give the class's products one fidelity",
                )
            self.compiled.fidelity = fidelity
            self.compiled.includes.add("matmul_custom")
        if state["function"] is None:
            state["function"] = cxx
            for field in fields:
                state["inits"].append(self.held_field(node, field))
                buffer = f"kField{camel(field)}"
                if field not in self.packed:  # compute's own: its next write frees it
                    state["after"].append(
                        Line(f"cb_pop_front({buffer}, {buffer}Pages);")
                    )
            init = []
            if any(operand is not None for operand in unit.operands):
                init += [
                    f"KM_UNPACK_FORMAT_A({streamed[0].buffer.name}.mover_a);",
                    f"copy_init({streamed[0].buffer.name}.mover_a);",
                ]
            if products:  # the first product's buffers: a into SrcB, w into SrcA
                a, w = (item.buffer.name + ".mover_a" for item in products[0])
                init += [
                    f"KM_UNPACK_FORMATS({w}, {a});",
                    f"mm_no_mop_init_short({a}, {w}, false, 1, 1, 1);",
                ]
            init.append(f"MATH((km_tile::{cxx}_init()));  // {function.name}'s setup")
            state["inits"].append(Line("\n".join(init)))
        timed = self.method.stamps is not None and not self.method.overlay
        self.results_named(node, function, unit, name)
        count = max(len(unit.operands), unit.results)  # DST tiles a call
        slots = DST_HALF // count if width == 1 else 1
        copied = [
            operands[parameters.index(operand)][0]
            for operand in unit.operands
            if operand is not None and operands[parameters.index(operand)][0]
        ]
        switch = SrcaSwitch(products, copied)
        number = self.stream_loops[-1]
        if width > 2 and state["block"] < width:  # the row read in passes
            return self.row_passes(node, function, unit, cxx, streamed[0], fields, name)
        if width > 2:
            return self.row_chunks(node, function, unit, cxx, streamed[0], fields, name)
        lines = ["tile_regs_acquire();"]
        if slots > 1:  # the calls of `slots` tiles an acquire, each its own DST tiles
            state["step"] = slots
            batch, slot = f"km_batch{number}", f"km_slot{number}"
            left = f"km_valid{number} - km_tile{number}"
            lines = [
                f"const uint32_t {batch} = {left} < {slots} ? {left} : {slots};",
                "tile_regs_acquire();",
                f"for (uint32_t {slot} = 0; {slot} < {batch}; {slot}++) {{",
            ]
            for a, w in products:  # unpacked in the unit's issue order
                lines += [f"    {line}" for line in switch.product(a, w)]
                lines.append(
                    f"    UNPACK((llk_unpack_AB_matmul({a.cxx}, {w.cxx}, {a.first} + "
                    f"{slot}, {w.first} + {slot}, 1, 1, 1)));"
                )
            for dst, operand in enumerate(unit.operands):
                if operand is None:  # a product's Dst tile
                    continue
                item, field = operands[parameters.index(operand)]
                if dst in unit.row_operands or dst in unit.col_operands:
                    lines += [
                        f"    {line}"
                        for line in row_broadcast(
                            item,
                            field,
                            slot,
                            f"{count} * {slot} + {dst}",
                            copied,
                            "COL" if dst in unit.col_operands else "ROW",
                        )
                    ]
                    continue
                lines += [f"    {line}" for line in switch.copy(item)]
                lines.append(
                    f"    copy_tile({item.cxx}, {item.first} + {slot}, "
                    f"{count} * {slot} + {dst});"
                )
            words = scalar_words(unit, scalars)
            call_ = [f"MATH((km_tile::{cxx}({count} * {slot}{words})));"]
            lines += ["    " + line for line in self.timed_call(call_, timed)] + ["}"]
        elif width == 1:
            for a, w in products:  # unpacked in the unit's issue order
                lines += switch.product(a, w)
                lines.append(
                    f"UNPACK((llk_unpack_AB_matmul({a.cxx}, {w.cxx}, {a.first}, "
                    f"{w.first}, 1, 1, 1)));"
                )
            for dst, operand in enumerate(unit.operands):
                if operand is None:  # a product's Dst tile
                    continue
                item, field = operands[parameters.index(operand)]
                if dst in unit.row_operands or dst in unit.col_operands:
                    along = "COL" if dst in unit.col_operands else "ROW"
                    lines += row_broadcast(item, field, None, dst, copied, along)
                    continue
                lines += switch.copy(item)
                lines.append(f"copy_tile({item.cxx}, {item.first}, {dst});")
        else:  # the row form: the row in DST 0 .. width - 1, the field's tile after
            item = streamed[0]
            lines += [
                f"copy_tile({item.cxx}, {item.first} + {t}, {t});" for t in range(width)
            ]
            for k, field in enumerate(fields):  # resident row k: DST 2 width + k
                lines.append(f"copy_tile(kField{camel(field)}, 0, {2 * width + k});")
        if slots == 1:
            words = scalar_words(unit, scalars)
            lines += self.timed_call([f"MATH((km_tile::{cxx}(0{words})));"], timed)
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, self.text(node), function, cxx)
        )
        self.dst_next = 1
        held = "<results>" if isinstance(name, list) else name or "<tile function>"
        self.dst_live = held
        self.scope[held] = Value(
            "tile",
            held,
            dst=0,
            width=width,
            batch=(number, count) if slots > 1 else None,
        )
        return [], [Line("\n".join(lines), uses=set(scalars.values()))]

    def scalar_argument(self, node, arg):
        """Whether a tile function's argument is a float word (a run-time scalar);
        an int word is refused, naming the float annotation."""
        value = isinstance(arg, ast.Name) and self.scope.get(arg.id)
        if value and value.kind == "int":
            self.refuse(
                node,
                f"`{arg.id}` is an int word, and a tile function's run-time scalar is "
                "a float word (a float32's bits, loaded by the unit program's setup)",
                f"annotate it `{arg.id}: float` and pass the number",
            )
        return bool(value) and value.kind == "float"

    def traced_placement(self, node, function):
        """Each product's placement ("none" or "plain") in a traced tile function, by
        its operands' names; refused at the call's line when a marker turns one (the
        unit's vector code reads the products in DST in one body, and the transpose
        back is a library call) or marks both operands for one side."""
        tree = source_of(function.function)[0]
        placements = {}
        for product in ast.walk(tree):
            if not (
                isinstance(product, ast.BinOp) and isinstance(product.op, ast.MatMult)
            ):
                continue
            left = marked_operand(product.left)[1]
            right = marked_operand(product.right)[1]
            where = (
                f"{function.name}'s `{ast.unparse(product)}` (line {product.lineno})"
            )
            if left is not None and left == right:
                self.refuse(
                    node,
                    f"{where}: both operands are marked km.{spelling(left)}",
                    "mark one operand",
                )
            if left == "srca" or right == "srcb":
                self.refuse(
                    node,
                    f"{where} turns the product (its right operand in SrcB), and a "
                    "tile function the tile compiler traces runs its vector code on "
                    "the product in DST in one body: the turned product would leave "
                    "(a @ w)^T there, and DST's transpose back is a library call the "
                    "body's words do not hold",
                    "mark it the plain way (`km.srcb(a) @ w`), or make the product "
                    "a tile function `return a @ km.srcb(w)` of its own that a "
                    "handler's product line adds",
                )
            names = (marked_operand(product.left)[0], marked_operand(product.right)[0])
            placements[names] = "none" if left is None and right is None else "plain"
        return placements

    def step_calls(self, cxx):
        """A step's setup and body on the math CPU, each line indented once; with
        stamps (a resident handler), each timed by timed_call: two spans a step,
        its setup's and its body's."""
        timed = self.method.stamps is not None and not self.method.overlay
        lines = [
            *self.timed_call([f"MATH((km_tile::{cxx}_init()));"], timed),
            *self.timed_call([f"MATH((km_tile::{cxx}(0)));"], timed),
        ]
        if timed:  # each span's start in its own scope
            lines = [
                *(["{", *("    " + x for x in lines[:7]), "}"]),
                *(["{", *("    " + x for x in lines[7:]), "}"]),
            ]
        return "".join(f"    {line}\n" for line in lines)

    def timed_call(self, call, timed):
        """A call's lines; `timed` (stamps): between two tensix_sync(), its span on
        the math CPU's wall clock to the stamps page."""
        if not timed:
            return call
        last = self.spans_most - 1
        return [
            "MATH(tensix_sync());",
            "MATH(const uint32_t km_called = km::wall());",
            *call,
            "MATH(tensix_sync());",
            f"MATH(km_spans[{SPANS_FIRST} + (km_spans[1] < {last} ? km_spans[1] : "
            f"{last})] = km::wall() - km_called);",
            "MATH(km_spans[1] += 1);",
            'MATH(asm volatile("fence" ::: "memory"));',
        ]

    def held_field(self, node, field):
        """A field a tile function holds for its loop: filled above in the loop's
        block, waited for once before the blocks (and freed after them)."""
        filled = self.filled.get(field)
        if field in self.packed:  # a row's per-row value compute wrote above
            filled = (None, self.blocks[-2]) if len(self.blocks) > 1 else None
        if filled is None or len(self.blocks) < 2 or filled[1] != self.blocks[-2]:
            self.refuse(
                node,
                f"self.{field} is not filled before this loop, in its block: a field "
                "is filled for the loop whose tile function holds it",
                f"write self.{field}.fill(km.{spelling('stream')}(...)) before the "
                "loop",
            )
        self.filled[field] = None
        buffer = f"kField{camel(field)}"
        if self.compiled.fields[field].format.dtype == "float32":
            self.compiled.unpack_fp32.add(buffer)
        self.wait(node, "compute", WAIT_INPUT)
        return Line(
            f"if (!compute.wait_front({buffer}, {buffer}Pages)) {{\n    return;\n}}"
        )

    def tile_unit(self, node, function, formats, width=1, fields=(), tiles=None):
        """The unit program of `function` on tiles of `formats` (traced, its rules
        applied, emitted at the write; `width` > 1: the row form over rows of that
        many tiles, `fields` its resident rows) and its C++ name; refused by name and
        line when the tile compiler does not lower it."""
        made = (function.function, tuple(formats), width)
        if tiles is not None:  # operands already in Dst, apart (a unit of its own)
            made += (tuple(sorted(tiles.items())),)
        for cxx, (key, unit) in self.compiled.units.items():
            if key == made:
                return unit, cxx
        where = function_line(function)
        parameters = list(inspect.signature(function.function).parameters)
        if len(parameters) != len(formats):
            self.refuse(
                node,
                f"the tile function {function.name} ({where}) takes "
                f"{len(parameters)} tiles, {len(formats)} given",
                "pass one tile a parameter",
            )
        product = matrix_product(function.function)
        if product is not None and width > 1:
            self.refuse(
                node,
                f"the tile function {function.name}, line {product}, has a matrix "
                "product (`@`) and is called on a row: a product of one K step of "
                "32 x 32 tiles a side is lowered, a K loop over a row is not",
                "call it on a stream's tiles, one tile a side, or make the row's "
                "product with `acc += self.f(block, self.w)`",
            )
        cxx = function.name
        while cxx in self.compiled.units:
            cxx += "_"
        dest = tile_dest(formats)
        if dest != "fp32" and width > 1:
            self.refuse(
                node,
                f"the tile function {function.name} ({where}) takes tiles of "
                f"{', '.join(sorted(set(formats)))} (the A family, a 16-bit Dst) and "
                "is called on a row: the row form keeps each row's value and its sums "
                "in Dst between its levels, which a 16-bit Dst would cut to fp16",
                "call it on a stream's tiles, one tile a side, or stream tensors of "
                "the B family",
            )
        kinds = {**TILE_FORMATS, **{f: f for f in A_DTYPES}}
        tensors = [
            TensorParam(p, False, kinds[f])
            for p, f in zip(parameters, formats)
            if f != SCALAR
        ]
        values = [  # the run-time scalars: float words of the call
            trace.ValueParam(p, "float")
            for p, f in zip(parameters, formats)
            if f == SCALAR
        ]
        result = TensorParam("result", True, F32)
        ir = KernelIR(cxx, [*tensors, result], values, {})
        held = len(parameters) - len(fields)  # the fields are the last parameters
        try:
            with trace.Tracer(ir):
                proxies = [trace.TensorProxy(t) for t in tensors]
                out = trace.TensorProxy(result)
                if width == 1:
                    for t in trace.grid(trace.num_tiles(proxies[0])):
                        loads = iter([trace.load(p.tiles[t]) for p in proxies])
                        given = iter(values)
                        args = [next(given if f == SCALAR else loads) for f in formats]
                        stored = function.function(*args)
                        if isinstance(stored, tuple):  # its results: DST 0, 1, ...
                            traced_results(ir, out, t, stored)
                        else:
                            trace.store(out.tiles[t], stored)
                else:
                    rows = [trace.resident(p.tiles[0, :]) for p in proxies[held:]]
                    for t in trace.grid(trace.num_tile_rows(proxies[0])):
                        loads = [trace.load(p.tiles[t, :]) for p in proxies[:held]]
                        value = function.function(*loads, *rows)
                        if isinstance(value, tuple) and width > 2:  # chunk form's
                            traced_results(ir, out, (t, slice(None)), value)
                            continue
                        if isinstance(value, tuple):
                            raise PlanError(
                                f"{len(value)} results; a row of 2 tiles stores one "
                                "(several results are the one-tile form's and a "
                                "wide row's)"
                            )
                        trace.store(out.tiles[t, :], value)
            if width == 1:
                fidelity = function.fidelity or "HiFi4"  # its products'
                unit = unit_program(
                    ir, dest, load_macro=True, fidelity=fidelity, tiles=tiles
                )
            else:
                unit = (  # a wide row: in chunks of 2 tiles (row_chunks)
                    chunk_program(ir, width)
                    if width > 2
                    else row_program(ir, width, width, 2 * width)
                )
                self.compiled.full_dst = True
        except (PlanError, UnitError) as error:
            self.refuse(
                node,
                f"the tile function {function.name} ({where}) is not lowered by the "
                f"tile compiler: {error}",
                "write this body inline: km.cxx(..., gives=km.Tile)",
            )
        self.compiled.units[cxx] = (made, unit)
        self.compiled.body_includes["compute"].append(f'#include "tile_{cxx}.hpp"')
        return unit, cxx

    def push(self, node, call):
        target = call.func.value
        if not isinstance(target, ast.Name) or target.id not in self.scope:
            self.refuse(node, "push into a write stream", "write out.push(tile)")
        stream = self.scope[target.id]
        if stream.kind != "write":
            self.refuse(
                node,
                f"{target.id} is not a write stream",
                "open it with km.Stream(..., write=True)",
            )
        pushed = call.args and self.tile_expression(node, call.args[0])
        if len(call.args) == 1 and pushed:
            call.args[0] = pushed
        if len(call.args) == 1 and self.tile_function_of(call.args[0]) is not None:
            _, made = self.tile_call(node, call.args[0], None)
            item = self.scope.pop("<tile function>")
            if item.kind == "chunked":  # a wide row's call packs its own chunks
                return self.push_chunked(node, stream, target.id, made)
            _, pushed = self.push_made(node, stream, item, target.id)
            return [], made + pushed
        if len(call.args) == 1 and isinstance(call.args[0], ast.BinOp):
            self.refuse(
                node,
                "a tile expression is a tile function's",
                "write it in a @km.tile_function and push its call: "
                "out.push(self.f(tile))",
            )
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            self.refuse(node, "push one tile of the loop", "write out.push(tile)")
        item = self.scope.get(call.args[0].id)
        if item is not None and item.kind == "tile" and self.dst_live == item.cxx:
            return self.push_made(node, stream, item, target.id)
        if item is not None and item.kind == "result":
            return self.push_results(node, stream, item, target.id)
        self.other_tile_line(node)
        if (
            item is None
            or item.kind != "streamed"
            or not item.cxx.startswith("km_half")
        ):
            self.refuse(node, "push a tile of the stream loop", "write out.push(tile)")
        if self.dst_live is not None:
            self.refuse(node, f"{self.dst_live} holds DST", "write it first")
        if stream.pushed:
            self.refuse(
                node, "a write stream fed by two loops", "open a write stream for each"
            )
        self.loop_fed(node, target.id, stream)
        stream.pushed = True
        number = self.stream_loops[-1]
        text = (
            "tile_regs_acquire();\n"
            f"copy_tile({item.cxx}, {item.first}, 0);\n"
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"if (km_room{number}) {{\n"
            f"    KM_PACK_FORMAT(km_out{number});\n"
            f"    pack_tile<true>(0, km_out{number}, {item.first});\n"
            "}\n"
            "tile_regs_release();"
        )
        return [], [Line(text)]

    def loop_fed(self, node, name, stream):
        """A write stream a read loop feeds: a block of its pages a block of the
        loop's, and its count the loop's by the same expression."""
        state = self.loop_states[-1]
        block, count = stream.buffer.block, stream.count
        if block != state["block"] or count != state["count"]:
            self.refuse(
                node,
                f"{name} is {count} pages in blocks of {block}, and the loop's stream "
                f"(line {state['line']}) {state['count']} in blocks of "
                f"{state['block']}: a block of the loop's is a block of {name}'s",
                f"open {name} with the loop's count, {state['count']}, and block="
                f"{state['block']}",
            )

    def push_made(self, node, stream, item, name):
        """`out.push(t)`, t the tile a body made in DST 0 in a stream loop: packed into
        the write stream at the loop tile's place."""
        if not self.stream_loops:
            self.refuse(
                node, "a pushed tile is pushed in a stream loop", "push it there"
            )
        if stream.pushed:
            self.refuse(
                node, "a write stream fed by two loops", "open a write stream for each"
            )
        self.loop_fed(node, name, stream)
        stream.pushed = True
        number = self.stream_loops[-1]
        self.dst_live, self.dst_next = None, 0
        item.kind = "pushed"
        packs = [f"    pack_tile<true>(0, km_out{number}, km_tile{number});"]
        if item.batch is not None:  # each call's DST 0 to its tile's place
            slot, count = f"km_slot{number}", item.batch[1]
            packs = [
                f"    for (uint32_t {slot} = 0; "
                f"{slot} < km_batch{number}; {slot}++) {{\n"
                f"        pack_tile<true>({count} * {slot}, km_out{number}, "
                f"km_tile{number} + {slot});\n    }}"
            ]
        if item.width > 1:  # a row: DST 0 .. width - 1 to the row's places
            packs = [
                f"    pack_tile<true>({t}, km_out{number}, km_tile{number} + {t});"
                for t in range(item.width)
            ]
        text = (
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"if (km_room{number}) {{\n    KM_PACK_FORMAT(km_out{number});\n"
            + "\n".join(packs)
            + "\n}\n"
            "tile_regs_release();"
        )
        return [], [Line(text)]

    def row_chunks(self, node, function, unit, cxx, item, fields, name):
        """A pushed call on a row of more than 2 tiles: one acquire for the sums'
        passes over its chunks of 2 (DST 0, 1) and the per-row values (DST 2, packed
        to a page of their own); then an acquire a chunk: the chunk, those values and
        each field's chunk (DST 4 + k) in, the body, the chunk packed in place."""
        if name is not None and not isinstance(name, list):
            self.refuse(
                node,
                f"a call of {function.name} on a row of {item.width} tiles writes the "
                "row in chunks, pushed as it goes",
                f"push the call: `out.push({function.name}(row, ...))`",
            )
        if self.method.stamps is not None and not self.method.overlay:
            self.refuse(
                node,
                "stamps time one call a tile or a row, and a wide row's call is a "
                "call a chunk",
                "time it without stamps=, or on rows of 2 tiles",
            )
        rows = chunk_rows_buffer(self.compiled, cxx, node)
        number = self.stream_loops[-1]
        chunk, tile, first = f"km_chunk{number}", f"km_tile{number}", item.first
        count = item.width // 2
        loop = f"for (uint32_t {chunk} = 0; {chunk} < {count}; {chunk}++) {{"
        copy = [
            f"    copy_tile({item.cxx}, {first} + 2 * {chunk}, 0);",
            f"    copy_tile({item.cxx}, {first} + 2 * {chunk} + 1, 1);",
        ]
        lines = ["tile_regs_acquire();", f"MATH((km_tile::{cxx}_start(0)));"]
        last = f"copy_tile({item.cxx}, {first} + {item.width - 1}, 0);"  # an odd row's
        for part in unit.parts:
            if part.startswith("sum") and part.endswith("_one"):
                lines += [last, f"MATH((km_tile::{cxx}_{part}(0)));"]
            elif part.startswith("sum"):
                call = f"    MATH((km_tile::{cxx}_{part}(0)));"
                lines += [loop, *copy, call, "}"]
            elif part.startswith("finish"):
                lines.append(f"MATH((km_tile::{cxx}_{part}(0)));")
        back = f"{item.buffer.name}.mover_a"
        value = self.loop_states[-1].get("value")
        if value is not None and "value" not in unit.parts:
            self.refuse(
                node,
                f"{function.name}'s result reads the row; a field takes a per-row "
                "value alone",
                f"return the value broadcast: `return s.col()` in {function.name}",
            )
        if value is not None:  # the per-row value's tile into a field, no writes
            self.compiled.tile_calls.append(
                (self.method.name, node.lineno, self.text(node), function, cxx)
            )
            return [], [Line("\n".join([*lines, *value_packed(cxx, *value)]))]
        head = lines
        lines = [
            "tile_regs_commit();",
            "tile_regs_wait();",
            f"cb_reserve_back({rows}, 1);",
            f"KM_PACK_FORMAT({rows});",
            f"pack_tile(2, {rows});",
            f"cb_push_back({rows}, 1);",
            "tile_regs_release();",
            f"cb_wait_front({rows}, 1);",
            loop,
            "    tile_regs_acquire();",
            *copy,
            f"    KM_UNPACK_FORMAT_A({rows});",
            f"    copy_init({rows});",
            f"    copy_tile({rows}, 0, 2);",
            *[f"    {x}" for x in field_copies(self.compiled, fields, rows, chunk)],
            f"    KM_UNPACK_FORMAT_A({back});",
            f"    copy_init({back});",
            f"    MATH((km_tile::{cxx}(0)));",
            "    tile_regs_commit();",
            "    tile_regs_wait();",
            "@packs",
            "    tile_regs_release();",
            "}",
        ]
        body_ = lines

        def text(outs):  # the call's lines, each result packed into its stream
            lines = [*head]
            for x in body_:
                lines += (
                    chunk_packs(unit, outs, tile, chunk, "    ")
                    if x == "@packs"
                    else [x]
                )
            if item.width % 2:  # the last tile: a chunk of one
                at = lines.index(loop, lines.index(f"cb_wait_front({rows}, 1);"))
                body = [x[4:] for x in lines[at + 1 : -1] if "+ 1" not in x[4:]]
                body = [
                    x.replace(f"{first} + 2 * {chunk}", f"{first} + {item.width - 1}")
                    .replace(f"{tile} + 2 * {chunk}", f"{tile} + {item.width - 1}")
                    .replace(f", {chunk}, ", f", {count}, ")
                    .replace(f"km_tile::{cxx}(0)", f"km_tile::{cxx}_apply_one(0)")
                    for x in body
                ]
                lines += body
            lines.append(f"cb_pop_front({rows}, 1);")
            return "\n".join(lines)

        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, self.text(node), function, cxx)
        )
        if isinstance(name, list):  # several results: packed at the last push
            self.loop_states[-1]["wide"] = text
            self.scope["<results>"] = Value("tile", "<results>", width=item.width)
            return [], []
        self.scope["<tile function>"] = Value("chunked", cxx, width=item.width)
        return [], [Line(text([(f"km_out{number}", f"km_room{number}")]))]

    def row_passes_read(self, node, read, state, number):
        """A row longer than its stream's block: the range read once a pass of the
        row function's (`state["passes"]`), its blocks written in the last."""
        passes = state.get("passes")
        if passes is None:
            self.refuse(
                node,
                f"a block of {read['block']} tiles is a part of a row of "
                f"{read['width']}, and a row read in parts is a row function's",
                "push a row function's call on the row, or give block= the row",
            )
        if read["repeat"] != "1":
            self.refuse(
                node,
                "a row read in parts is read again once a pass, by itself",
                f"drop {spelling('repeat')}=",
            )
        read["repeat"] = str(passes)
        per_turn = f"km_blocks{number}"
        state["written"] = (
            f"km_block{number} % {per_turn}",
            f"km_block{number} / {per_turn} == {passes - 1} && ",
        )

    def row_passes(self, node, function, unit, cxx, item, fields, name):
        """A pushed call on a row longer than its block: the range read once a sum
        level, then for the writes. A sum pass holds DST a row, adds its chunks (DST
        0, 1) and packs the values' and sums' tiles (DST 2, 3) to a page pair of the
        row's, unpacked at the next pass; the last writes as on a row in one block."""
        state = self.loop_states[-1]
        block, count = state["block"], known_count(state["count"])
        if count is None and state.get("most"):  # a word: the per-row pages' most
            count = state["most"]
        refused = (
            "a wide row read in parts is whole chunks of 2 a block, and its rows "
            "a constant count or a stated most"
        )
        if block % 2 or count is None or count % item.width:
            self.refuse(
                node,
                refused,
                "give block= an even part of the row, and the rows a count or a most: "
                f"x.{spelling('tiles')}[r:min(r + MOST, r + rows), 0:W]",
            )
        self.row_chunks(node, function, unit, cxx, item, fields, name)  # its checks
        number = self.stream_loops[-1]
        levels = sum(1 for p in unit.parts if p.startswith("sum"))
        pair = 2 if levels > 1 else 1  # one sum pass: the values' tile alone a row
        rows = chunk_rows_buffer(self.compiled, cxx, node, pair * count // item.width)
        value = state.get("value")  # a per-row value into a field: no write pass
        state["passes"] = levels + (value is None)
        at, part = f"km_pass{number}", f"km_part{number}"
        per_turn = f"km_blocks{number}"
        chunk, tile, first = f"km_chunk{number}", f"km_tile{number}", item.first
        loop = f"for (uint32_t {chunk} = 0; {chunk} < {block // 2}; {chunk}++) {{"
        copy = [
            f"    copy_tile({item.cxx}, {first} + 2 * {chunk}, 0);",
            f"    copy_tile({item.cxx}, {first} + 2 * {chunk} + 1, 1);",
        ]
        back = f"{item.buffer.name}.mover_a"
        last = item.width // block - 1

        def by_pass(name_):  # the level's part of the pass this block is in
            return " else ".join(
                f"if ({at} == {k}) {{\n    MATH((km_tile::{cxx}_{name_}{k}(0)));\n}}"
                for k in range(levels)
            ).split("\n")

        lines = [
            f"const uint32_t {at} = km_block{number} / {per_turn};",
            f"const uint32_t {part} = km_block{number} % {per_turn} % {last + 1};",
            f"if ({at} < {levels}) {{",
            f"    if ({part} == 0) {{",
            "        tile_regs_acquire();",
            f"        if ({at} == 0) {{",
            f"            MATH((km_tile::{cxx}_start(0)));",
            "        } else {",
            f"            cb_wait_front({rows}, 2);",
            f"            KM_UNPACK_FORMAT_A({rows});",
            f"            copy_init({rows});",
            f"            copy_tile({rows}, 0, 2);",
            f"            copy_tile({rows}, 1, 3);",
            f"            cb_pop_front({rows}, 2);",
            f"            KM_UNPACK_FORMAT_A({back});",
            f"            copy_init({back});",
            "        }",
            "    }",
            f"    {loop}",
            *[f"    {x}" for x in copy],
            *[f"        {x}" for x in by_pass("sum")],
            "    }",
            f"    if ({part} == {last}) {{",
            *[f"        {x}" for x in by_pass("finish")],
            *(
                []
                if value is None
                else [
                    f"        if ({at} == {levels - 1}) {{",
                    *[f"            {x}" for x in value_packed(cxx, *value)],
                    "        } else {",
                ]
            ),
            "        tile_regs_commit();",
            "        tile_regs_wait();",
            f"        cb_reserve_back({rows}, {pair});",
            f"        KM_PACK_FORMAT({rows});",
            f"        pack_tile(2, {rows});",
            *([f"        pack_tile(3, {rows});"] if pair == 2 else []),
            f"        cb_push_back({rows}, {pair});",
            "        tile_regs_release();",
            *([] if value is None else ["        }"]),
            "    }",
            "} else {",
            f"    if ({part} == 0) {{",
            f"        cb_wait_front({rows}, {pair});",
            "    }",
            f"    {loop}",
            "        tile_regs_acquire();",
            *[f"    {x}" for x in copy],
            f"        KM_UNPACK_FORMAT_A({rows});",
            f"        copy_init({rows});",
            f"        copy_tile({rows}, 0, 2);",
            *[
                f"        {x}"
                for x in field_copies(
                    self.compiled, fields, rows, f"{part} * {block // 2} + {chunk}"
                )
            ],
            f"        KM_UNPACK_FORMAT_A({back});",
            f"        copy_init({back});",
            f"        MATH((km_tile::{cxx}(0)));",
            "        tile_regs_commit();",
            "        tile_regs_wait();",
            "@packs",
            "        tile_regs_release();",
            "    }",
            f"    if ({part} == {last}) {{",
            f"        cb_pop_front({rows}, {pair});",
            "    }",
            "}",
        ]
        if value is not None:  # no write pass: the else above never runs
            lines = lines[: lines.index("} else {")] + ["}"]

        def text(outs):  # each result packed into its stream in the last pass
            packs = chunk_packs(unit, outs, tile, chunk, "        ")
            return "\n".join(
                y for x in lines for y in (packs if x == "@packs" else [x])
            )

        if isinstance(name, list):  # several results: packed at the last push
            state["wide"] = text
            return [], []
        return [], [Line(text([(f"km_out{number}", f"km_room{number}")]))]

    def push_chunked(self, node, stream, name, made):
        """`out.push(f(row, ...))` of a wide row: the call packed its chunks."""
        if stream.pushed:
            self.refuse(
                node, "a write stream fed by two loops", "open a write stream for each"
            )
        self.loop_fed(node, name, stream)
        stream.pushed = True
        return [], made

    def tile_results(self, node, targets, call):
        """`s, y = self.f(a, b)`, f returning several tiles: the call as `tile_call`
        makes it, result k in DST k of the call, each name pushed on its own."""
        names = [t.id if isinstance(t, ast.Name) else None for t in targets.elts]
        if None in names:
            self.refuse(node, "a tile function's results are names", "write a, b = ...")
        for name in names:
            self.check_name(node, name)
            self.not_parameter(node, name)
        _, made = self.tile_call(node, call, names)
        group = self.scope.pop("<results>")
        state = self.loop_states[-1]
        state["results"] = {"names": names, "packs": {}, "line": node.lineno}
        state["results"]["wide"] = state.pop("wide", None)  # a wide row's lines
        for k, name in enumerate(names):
            self.scope[name] = Value("result", name, dst=k, batch=group.batch)
        return [], made

    def fields_by_row(self, node, function, unit, given):
        """A field in the one-tile form is a plain row, read by `.row()` alone, or a
        plain column, read by `.col()` alone (`given`: each field's parameter)."""
        kept = {*unit.row_operands, *unit.col_operands}
        row = {unit.operands[k] for k in kept}
        whole = {o for k, o in enumerate(unit.operands) if k not in kept}
        for field_, parameter in given.items():
            if parameter not in row or parameter in whole:
                self.refuse(
                    node,
                    f"self.{field_} in a call on a stream's tiles is a row (row 0 of "
                    f"its tile) read by `{parameter}.row()` alone, or a column (column "
                    f"0) read by `{parameter}.col()` alone",
                    f"write `{parameter}.row()` or `{parameter}.col()` in "
                    f"{function.name}, or call it on a row of a stream of rows",
                )

    def results_named(self, node, function, unit, name):
        """A call of a function of several results names each (`a, b = f(...)`)."""
        count = len(name) if isinstance(name, list) else 1
        if count != unit.results:
            self.refuse(
                node,
                f"the tile function {function.name} returns {unit.results} tile(s), "
                f"and the line takes {count}",
                "name each result: `a, b = f(...)`, then push each",
            )

    def push_results(self, node, stream, item, name):
        """`out.push(y)`, y a result of a call of several: its DST tile packed into
        the write stream; the commit, the packs and the release at the last push of
        the call's results."""
        state = self.loop_states[-1] if self.loop_states else {}
        group = state.get("results")
        if group is None or item.cxx not in group["names"]:
            self.refuse(
                node,
                f"{item.cxx} is a result of a tile function's call, pushed in its loop",
                "push it in the loop of the call",
            )
        if stream.pushed:
            self.refuse(
                node, "a write stream fed by two loops", "open a write stream for each"
            )
        self.loop_fed(node, name, stream)
        stream.pushed = True
        state.setdefault("own_out", set()).add(stream.cxx)
        number = self.stream_loops[-1]
        out, room = out_names(number, stream, state)
        if group["wide"] is not None:  # a wide row's: its chunks packed in its loop
            group["packs"][item.cxx] = (out, room)
            item.kind = "pushed"
            if len(group["packs"]) < len(group["names"]):
                return [], []
            return [], [
                Line(group["wide"]([group["packs"][n] for n in group["names"]]))
            ]
        pack = f"    pack_tile<true>({item.dst}, {out}, km_tile{number});"
        if item.batch is not None:  # each call's DST k to its tile's place
            slot, count = f"km_slot{number}", item.batch[1]
            pack = (
                f"    for (uint32_t {slot} = 0; {slot} < km_batch{number}; {slot}++) "
                f"{{\n        pack_tile<true>({count} * {slot} + {item.dst}, {out}, "
                f"km_tile{number} + {slot});\n    }}"
            )
        group["packs"][
            item.cxx
        ] = f"if ({room}) {{\n    KM_PACK_FORMAT({out});\n{pack}\n}}"
        item.kind = "pushed"
        if len(group["packs"]) < len(group["names"]):
            return [], []
        packs = [group["packs"][n] for n in group["names"]]
        self.dst_live, self.dst_next = None, 0
        text = "\n".join(["tile_regs_commit();", "tile_regs_wait();", *packs])
        return [], [Line(text + "\ntile_regs_release();")]

    def results_pushed(self, node, state):
        """A loop that calls a function of several results pushes each."""
        group = state.get("results")
        if group is None or len(group["packs"]) == len(group["names"]):
            return
        left = [n for n in group["names"] if n not in group["packs"]]
        self.refuse(
            node,
            f"the results {', '.join(group['names'])} of the call at line "
            f"{group['line']} are pushed in its loop, and {', '.join(left)} is not",
            f"push {left[0]} into a write stream in the loop",
        )

    def write(self, node, call):
        target = call.func.value
        if self.field_name(target) is not None:
            return self.write_field(node, call, self.field_name(target))
        if isinstance(target, ast.Subscript) and self.field_name(target.value):
            return self.write_row(
                node, call, self.field_name(target.value), target.slice
            )
        if not self.is_range(target):
            self.refuse(
                node,
                "write to a page range",
                "write tensor.pages(first, 1).write(tile)",
            )
        pages = self.range_of(target)
        self.pages_only(node, pages)
        summed = call.args[0] if len(call.args) == 1 else None
        if self.field_name(summed) in self.summed:
            return self.write_sums(node, pages, self.field_name(summed))
        if (
            self.tile_function_of(summed) is not None
            and summed.args
            and all(self.field_name(arg) in self.summed for arg in summed.args)
        ):
            return self.write_function_of_sums(node, pages, summed)
        if self.tile_function_of(summed) is not None and self.dst_pair_of(summed):
            return self.write_function_of_pair(node, call, summed)
        if self.tile_function_of(summed) is not None and self.dst_sums_of(summed):
            return self.write_function_of_dst(node, call, summed)
        if self.tile_function_of(summed) is not None and self.step_of_dst(summed):
            return self.write_step_of_dst(node, call, summed)
        if self.field_name(summed) in self.stepped:
            return self.write_state(node, pages, self.field_name(summed))
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            self.refuse(node, "write one tile", "write pages.write(total)")
        name = call.args[0].id
        value = self.scope.get(name)
        if value is not None and value.kind == "word":
            return self.write_word(node, pages, value)
        self.not_paired(node, name)
        carried = self.carries(node, value)
        held_sum = self.sum_held(value)
        if (
            value is not None
            and value.kind in ("tile", "reduced")
            and not value.popped
            and not carried
            and not (held_sum and not value.summed)
        ):
            self.same_loops(node, name, value, "this write", *self.once_text(value))
        if value is not None and value.kind == "reduced" and value.summed:
            if not self.in_leader or value.popped:
                self.refuse(
                    node,
                    f"{name} is the reduce's sum of self.{value.summed}, which only "
                    "the leader holds, once",
                    f"write it once under `if self.{spelling('rank')} == "
                    f"self.{spelling('leader')}:`",
                )
            value.popped = True
            self.leader_writes.add(name)
            return self.write_sums(node, pages, value.summed, pushed=True)
        if value is None or value.kind not in ("tile", "reduced") or value.popped:
            self.refuse(
                node,
                f"{name} is not a tile of compute (or is written already)",
                "write a tile made by km.tile.zeros and added to, once",
            )
        width = value.width
        if pages.rect is not None:
            self.refuse(
                node,
                "a page write of a rectangle is one row of it",
                f"write {pages.tensor}.{spelling('tiles')}[r:r + 1, c0:c1].write(...)",
            )
        if pages.count != str(width):
            self.refuse(
                node,
                f"{name} is {width} tile(s), {width} page(s)",
                f"write pages(first, {width})",
            )
        self.compiled.includes.add("pack")
        compute = []
        converted = False
        if value.kind == "tile":
            dtype, page_bytes = self.tensor_format(node, pages.tensor)
            buffer = self.out_buffer(
                dtype, page_bytes, f"line {node.lineno}: {self.text(node)}", width
            )
            if carried:
                compute.append(Line(self.carried_write(node, value, buffer), {"km_ok"}))
                value.buffer = buffer
                self.wait(node, "compute", WAIT_OUTPUT)
                self.wait(node, "compute", WAIT_INPUT)
                movement, _ = self.write_moved(node, pages, value, buffer, width)
                return movement, compute
            compute.append(Line(pack(value.dst, buffer.name, width), {"km_ok"}))
            if getattr(value, "state", None) is not None:  # started from a field
                self.state_packed(compute, *value.state)
            self.row_given_back(compute, value)
            self.dst_live = None
            value.kind = "written"
            value.buffer = buffer
            self.wait(node, "compute", WAIT_OUTPUT)
        else:
            if not self.in_leader:
                self.refuse(
                    node,
                    f"{name} is the reduce's sum, which only the leader holds",
                    f"write it under `if self.{spelling('rank')} == "
                    f"self.{spelling('leader')}:`",
                )
            buffer = value.buffer
            self.leader_writes.add(name)
            dtype, page_bytes = self.tensor_format(node, pages.tensor)
            if held_sum:
                buffer = self.out_buffer(
                    dtype, page_bytes, f"line {node.lineno}: {self.text(node)}", width
                )
                compute.append(self.held_sum_copy(value, buffer, width))
                converted = True
            elif dtype != buffer.dtype:  # the sum packed again in the tensor's format
                source = buffer
                buffer = self.out_buffer(
                    dtype, page_bytes, f"line {node.lineno}: {self.text(node)}", width
                )
                self.compiled.unpack_fp32.add(source.name)
                copies = "\n".join(
                    f"copy_tile({source.name}, {t}, {t});" for t in range(width)
                )
                again = f"{copies}\ncb_pop_front({source.name}, {width});\n" + pack(
                    0, buffer.name, width
                )
                body = "\n".join("    " + line for line in again.split("\n"))
                compute.append(
                    Line(
                        f"// the sum in {source.dtype}, packed again in {dtype}\n"
                        "km_ok = km_ok && compute.wait_front("
                        f"{source.name}, {width});\n"
                        "if (km_ok) {\n"
                        f"    KM_UNPACK_FORMAT_A({source.name});\n"
                        f"    copy_init({source.name});\n"
                        "    tile_regs_acquire();\n"
                        f"{body}\n}}",
                        {"km_ok"},
                    )
                )
                converted = True
        return self.write_moved(node, pages, value, buffer, width, converted, compute)

    def write_moved(
        self, node, pages, value, buffer, width, converted=False, compute=()
    ):
        """Mover B's page write of `value`'s `width` pages from `buffer`, and
        `compute`'s lines with it."""
        # A page write: mover B waits for compute's page, writes it, frees it once
        # sent (as a hand writer does); the runtime settles writes before it reports.
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        write = (
            f"km::write_pages(movement, command, {tensor}, {pages.first}, {width}, "
            f"get_read_ptr({buffer.name}))"
        )
        if value.waited:
            opening = f"if (!{write}) {{"
        elif converted:
            opening = (
                f"if (!movement.wait_front({buffer.name}, {width}) ||\n"
                f"    !{write}) {{"
            )
        else:
            opening = self.page_wait(value, width)[:-3] + f" ||\n    !{write}) {{"
        text = (
            f"{opening}\n"
            "    return false;\n"
            "}\n"
            "noc_async_writes_flushed();\n"
            f"cb_pop_front({buffer.name}, {width});"
        )
        if value.cxx not in self.carried:  # a running total stays live
            value.waited = value.popped = True
        self.moved(node, pages.tensor, "page write")
        movement = [Line(text, range_uses(pages, self), mover="B")]
        self.compiled.includes.add("ask")
        self.wait(node, "mover B", WAIT_DATA)
        return movement, list(compute)

    def write_field(self, node, call, field_):
        """`self.f.write(acc)`, acc DST sums (or `f(acc)`, `f(acc, self.s[r])` a
        function of them): acc's tiles packed into the field's pages for products
        that take it as their left side. A write frees the last one's pages (their
        readers are done), the handler's end frees the last."""
        written = call.args[0] if len(call.args) == 1 else None
        if self.tile_function_of(written) is not None and self.row_written(written):
            return self.write_row_value(node, field_, written)
        if self.tile_function_of(written) is not None and self.dst_sums_of(written):
            return self.write_function_of_dst(node, call, written)
        if self.tile_function_of(written) is not None and self.step_of_dst(written):
            return self.write_step_of_dst(node, call, written)
        value = isinstance(written, ast.Name) and self.scope.get(written.id)
        if not value or value.kind != "tile" or value.popped:
            self.refuse(
                node,
                f"self.{field_} takes a tile of compute's DST sums (or a tile function "
                "of them), once",
                f"write self.{field_}.write(acc), acc made by km.tile.zeros and added "
                "to",
            )
        self.same_loops(node, written.id, value, "this write", *self.once_text(value))
        stored = self.compiled.fields[field_]
        pages = math.prod(stored.shape[:-2])
        fillers = self.compiled.field_users.get(field_, ([], []))[0]
        if (
            fillers
            or field_ in self.compiled.own
            or field_ in self.summed
            or field_ in self.stepped
            or len(stored.shape) < 3
        ):
            self.refuse(
                node,
                f"self.{field_} is filled, summed into or a body's: compute writes "
                "a field of its own",
                f"declare a field for the write: km.L1[km.bf16, {value.width}, 32, 32]",
            )
        if pages != value.width:
            self.refuse(
                node,
                f"self.{field_} holds {pages} tile(s) and {written.id} is "
                f"{value.width}",
                f"declare self.{field_}: km.L1[km.bf16, {value.width}, 32, 32]",
            )
        self.producer(node, field_, "product")
        if getattr(value, "row_of", None) is not None:  # sums started from a mask
            self.masked.add(field_)
        else:
            self.masked.discard(field_)
        buffer, held = f"kField{camel(field_)}", f"km_packed_{field_}"
        self.compiled.summed_fields.add(buffer)  # pack writes it: one of its formats
        if field_ not in self.packed:
            self.packed[field_] = held
            self.opening.append(
                Line(
                    f"bool {held} = false;  // self.{field_}'s pages pushed, not freed"
                )
            )
            self.closing.append(
                Line(f"if ({held}) {{\n    cb_pop_front({buffer}, {buffer}Pages);\n}}")
            )
        self.compiled.includes.add("pack")
        self.wait(node, "compute", WAIT_OUTPUT)
        value.kind, value.popped = "written", True
        self.dst_live = None
        state = getattr(value, "state", None)
        text = (
            f"if ({held}) {{  // the last write's pages: their products are done\n"
            f"    cb_pop_front({buffer}, {buffer}Pages);\n"
            "}\n"
            f"{pack(value.dst, buffer, value.width)}\n"
            f"{held} = km_ok;"
        )
        compute = [Line(text, {"km_ok"})]
        if state is not None:
            self.state_packed(compute, *state)
        self.row_given_back(compute, value)
        return [], compute

    def row_written(self, call):
        """Whether `call` is a tile function's call on a row of more than 2 tiles."""
        first = call.args[0] if call.args else None
        value = isinstance(first, ast.Name) and self.scope.get(first.id)
        return bool(value) and value.kind == "streamed" and (value.width or 1) > 2

    def write_row_value(self, node, field_, written):
        """`self.part.write(f(row))`, f's result a per-row value (`s.col()`): the
        row's sum passes, then the value's tile packed into the field, one tile of
        compute's own, for a later loop's `.col()` (a write frees the last one)."""
        stored = self.compiled.fields[field_]
        if (
            self.compiled.field_users.get(field_, ([], []))[0]
            or field_ in self.compiled.own
            or math.prod(stored.shape[:-2]) != 1
            or stored.format.dtype != "float32"
        ):
            self.refuse(
                node,
                f"self.{field_} takes a row's per-row value: one float32 tile of "
                "compute's own",
                f"declare self.{field_}: km.L1[km.fp32, 1, 32, 32], not filled",
            )
        self.producer(node, field_, "product")
        buffer, held = f"kField{camel(field_)}", f"km_packed_{field_}"
        self.compiled.summed_fields.add(buffer)
        if field_ not in self.packed:
            self.packed[field_] = held
            self.opening.append(Line(f"bool {held} = false;  // self.{field_} pushed"))
            self.closing.append(
                Line(f"if ({held}) {{\n    cb_pop_front({buffer}, {buffer}Pages);\n}}")
            )
        self.compiled.includes.add("pack")
        self.wait(node, "compute", WAIT_OUTPUT)
        self.loop_states[-1]["value"] = (buffer, held)
        _, made = self.tile_call(node, written, None)
        self.scope.pop("<tile function>", None)
        return [], made

    def write_sums(self, node, pages, field_, pushed=False):
        """`t.tiles[r0:r1, c0:c1].write(self.part)`, self.part a field of sums: its
        tiles copied into DST, packed in t's format a row at a time, each row of the
        rectangle written by mover B; the field's pages freed, the next sum starts.
        `pushed`: a reduce's sum, the field handed on already."""
        rows, columns = self.field_sides(node, field_)
        rect = pages.rect
        bounds = pages.bounds or (None, None)
        if (
            rect is not None
        ):  # rows of the rectangle; a side a word: at most the field's
            sides = rect[:2]
        elif pages.count == str(rows * columns):
            sides = (str(rows), columns)
        else:  # one row
            sides = ("1", pages.count)
        fits = all(
            side_fits(side, bound, most)
            for side, bound, most in zip(sides, bounds, (rows, columns), strict=True)
        )
        if not fits:
            self.refuse(
                node,
                f"self.{field_} is {rows} x {columns} tiles; write a rectangle of them "
                "(a side a word: the build must know it is at most the field's)",
                f"write {pages.tensor}.{spelling('tiles')}[r:r + {rows}, c:c + "
                f"{columns}].write(self.{field_}), or c:{spelling('min')}(c + "
                f"{columns}, e)",
            )
        written_rows, written_columns = sides
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        name = f"kOutRows{columns}"  # its rows' pages stay whole: one width a buffer
        held = [b.pages[0] for b in self.compiled.buffers if b.name == name]
        out = self.out_buffer(
            dtype, page_bytes, f"line {node.lineno}: {self.text(node)}", columns, name
        )
        # a row packed while the one before is written (one row: the row alone)
        out.pages = (max(held + [columns * min(rows, 2)]),)
        buffer, sums = f"kField{camel(field_)}", self.summed[field_]
        room, out_room = sums_room(field_), f"km_room{node.lineno}"
        step = max(w for w in range(1, DST_HALF + 1) if columns % w == 0)
        tile = f"km_tile{node.lineno}"
        copies = "\n".join(
            f"        copy_tile({buffer}, {tile} + {j}, {j});" for j in range(step)
        )
        packs = "\n".join(
            f"            pack_tile({j}, {out.name});" for j in range(step)
        )
        taken = (
            f"km_ok = km_ok && compute.wait_front({buffer}, {buffer}Pages);\n"
            "if (km_ok) {\n"
            if pushed
            else f"if (km_ok && {sums} > 0) {{\n"
            f"    if ({room}) {{  // pack's: its pages held\n"
            f"        cb_push_back({buffer}, {buffer}Pages);\n"
            "    }\n"
            f"    km_ok = compute.wait_front({buffer}, {buffer}Pages);\n"
            "}\n"
            f"if (km_ok && {sums} > 0) {{\n"
        )
        compute = (
            f"// self.{field_}'s sums, packed again in {dtype}\n"
            f"{taken}"
            f"    KM_UNPACK_FORMAT_A({buffer});\n"
            f"    copy_init({buffer});\n"
            f"    bool {out_room} = true;  // pack's alone: it guards the packs only\n"
            f"    for (uint32_t {tile} = 0; km_ok && "
            f"{tile} < {rows * columns}; {tile} += "
            f"{step}) {{\n"
            "        tile_regs_acquire();\n"
            f"{copies}\n"
            "        tile_regs_commit();\n"
            "        tile_regs_wait();\n"
            f"        {out_room} = {out_room} && compute.reserve_back({out.name}, "
            f"{step});\n"
            f"        if ({out_room}) {{\n"
            f"            KM_PACK_FORMAT({out.name});\n"
            f"{packs}\n"
            f"            cb_push_back({out.name}, {step});\n"
            "        }\n"
            "        tile_regs_release();\n"
            "    }\n"
            f"    cb_pop_front({buffer}, {buffer}Pages);\n"
            "}\n"
            f"{sums} = 0;"
        )
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        row = f"km_row{node.lineno}"
        row_step = rect[2] if rect is not None else columns
        written = (  # the rectangle's rows of the field's, where they are a word
            ""
            if str(written_rows) == str(rows)
            else f"{row} < ({written_rows}) &&\n        "
        )
        movement = (
            f"for (uint32_t {row} = 0; {row} < {rows}; {row}++) {{\n"
            f"    if (!movement.wait_front({out.name}, {columns}) ||\n"
            f"        {'(' if written else ''}{written}!km::write_pages(movement, "
            f"command, {tensor}, {pages.first} + {row} * {row_step}, "
            f"{written_columns}, get_read_ptr({out.name})){')' if written else ''}) "
            "{\n"
            "        return false;\n"
            "    }\n"
            "    noc_async_writes_flushed();\n"
            f"    cb_pop_front({out.name}, {columns});\n"
            "}"
        )
        if not pushed:  # under compute's condition of the pack: a sum was made
            inner = "\n".join("    " + line for line in movement.split("\n"))
            movement = f"if ({sums} > 0) {{\n{inner}\n}}\n{sums} = 0;"
        self.compiled.includes |= {"pack", "ask"}
        self.moved(node, pages.tensor, "page write")
        self.wait(node, "compute", WAIT_OUTPUT)
        self.wait(node, "mover B", WAIT_DATA)
        return [Line(movement, range_uses(pages, self), mover="B")], [
            Line(compute, {"km_ok"})
        ]

    def dst_sums_of(self, call):
        """The name of the DST sums a tile function's call takes (`f(acc)`, acc a
        km.tile.zeros sum compute holds), or None."""
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            return None
        value = self.scope.get(call.args[0].id)
        if value is None or value.kind != "tile" or value.batch is not None:
            return None
        return call.args[0].id

    def write_function_of_dst(self, node, write, call):
        """`pages.write(f(acc))`, acc DST sums (a product's over K, added in the
        loop): f's body run on each of acc's DST tiles in place on the math CPU, then
        acc written as `pages.write(acc)` writes it."""
        function = self.tile_function_of(call)
        name = self.dst_sums_of(call)
        value = self.scope[name]
        unit, cxx = self.tile_unit(node, function, ["float32"])
        if unit.products:
            self.refuse(
                node,
                f"{function.name} has a matrix product (`@`), and its operand here is "
                f"{name}, sums in DST that the matrix unit does not read",
                f"add the product in the loop and write {function.name}({name}) of "
                "its sums",
            )
        calls = "\n".join(
            f"MATH((km_tile::{cxx}({value.dst + k})));" for k in range(value.width)
        )
        body = f"MATH((km_tile::{cxx}_init()));  // {function.name}'s setup\n{calls}"
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, self.text(node), function, cxx)
        )
        write.args[0] = ast.Name(id=name, ctx=ast.Load())
        movement, compute = self.write(node, write)
        return movement, [Line(body), *compute]

    def dst_pair_of(self, call):
        """The two DST sums held together that a tile function's call takes (`f(a,
        b)`, a and b the pair of a second zeros, each once), or None."""
        if (
            self.dst_pair is None
            or len(call.args) != 2
            or call.keywords
            or not all(isinstance(arg, ast.Name) for arg in call.args)
        ):
            return None
        names = tuple(arg.id for arg in call.args)
        return names if set(names) == set(self.dst_pair) else None

    def not_paired(self, node, name):
        """Refuses a write or a reduce of one of two DST sums held together: its
        pack releases DST, and the other's sums with it."""
        if self.dst_pair is not None and name in self.dst_pair:
            a, b = self.dst_pair
            self.refuse(
                node,
                f"{a} and {b} hold DST together, and writing {name} alone releases "
                "the other's sums",
                f"write a tile function of both: pages.write(f({a}, {b}))",
            )

    def write_function_of_pair(self, node, write, call):
        """`pages.write(f(a, b))`, a and b DST sums held together (a second zeros):
        f's body run on DST tile t of the first of them with the other operand read
        its distance apart, for each of their tiles, then the first's tiles written as
        `pages.write(acc)` writes them (DST released)."""
        function = self.tile_function_of(call)
        values = [self.scope[arg.id] for arg in call.args]
        if values[0].width != values[1].width:
            self.refuse(
                node,
                f"{function.name} takes {call.args[0].id} ({values[0].width} tiles) "
                f"and {call.args[1].id} ({values[1].width} tiles): tile t of each",
                "make the two sums one width",
            )
        for value in values:
            self.same_loops(
                node, value.cxx, value, "this write", *self.once_text(value)
            )
        base = min(values, key=lambda v: v.dst)
        parameters = list(inspect.signature(function.function).parameters)
        tiles = {p: v.dst - base.dst for p, v in zip(parameters, values, strict=False)}
        unit, cxx = self.tile_unit(node, function, ["float32", "float32"], tiles=tiles)
        if unit.products:
            self.refuse(
                node,
                f"{function.name} has a matrix product (`@`), and its operands here "
                "are sums in DST that the matrix unit does not read",
                f"add the products in the loops and write {function.name} of the sums",
            )
        calls = "\n".join(
            f"MATH((km_tile::{cxx}({base.dst + k})));" for k in range(base.width)
        )
        body = f"MATH((km_tile::{cxx}_init()));  // {function.name}'s setup\n{calls}"
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, self.text(node), function, cxx)
        )
        other = values[1] if base is values[0] else values[0]
        other.kind = "written"  # its sums end with the write of the first's
        self.dst_pair = None
        self.dst_live = base.cxx
        write.args[0] = ast.Name(id=base.cxx, ctx=ast.Load())
        movement, compute = self.write(node, write)
        return movement, [Line(body), *compute]

    def step_of_dst(self, call):
        """(acc, field, index) of a step function's call `f(acc, self.s[r])`, acc DST
        sums and self.s a field, or None."""
        if len(call.args) != 2 or call.keywords:
            return None
        acc, slot = call.args
        value = isinstance(acc, ast.Name) and self.scope.get(acc.id)
        if not value or value.kind != "tile" or value.batch is not None:
            return None
        field_ = isinstance(slot, ast.Subscript) and self.field_name(slot.value)
        return (acc.id, field_, slot.slice) if field_ else None

    def write_step_of_dst(self, node, write, call):
        """`pages.write(f(acc, self.s[r]))`, f a step-form function of acc (1 to 3 DST
        tiles) and state r of the fp32 field self.s: the state into the DST tile after
        the row, f, the row written as acc is and the state packed back into page r,
        ordered by a token pack pushes and unpack takes (primed at the handler's start).
        """
        function = self.tile_function_of(call)
        name, field_, index = self.step_of_dst(call)
        value = self.scope[name]
        width = value.width
        if value.dst != 0 or not 1 <= width <= 3:
            self.refuse(
                node,
                f"{name} is {width} tile(s) of DST from tile {value.dst}; a step takes "
                "a row of 1 to 3 tiles from DST 0 and its state in the tile after them",
                "sum the row in km.tile.zeros(km.fp32, n), n at most 3",
            )
        carried = getattr(value, "state", None)
        if carried is not None:
            return self.step_on_carried(node, write, call, value, carried)
        buffer, token, slot, bound = self.step_state(node, field_, index)
        unit, cxx = self.step_unit(node, function, width)
        before = (
            f"{bound}"
            f"// {function.name}'s step: self.{field_}[{ast.unparse(index)}] into "
            f"DST {width}\n"
            f"{self.state_taken(buffer, token, slot, width)}"
            f"{self.step_calls(cxx)}"
            "}"
        )
        self.wait(node, "compute", WAIT_INPUT)
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, self.text(node), function, cxx)
        )
        write.args[0] = ast.Name(id=name, ctx=ast.Load())
        movement, compute = self.write(node, write)
        self.state_packed(compute, buffer, token, slot, width, index)
        return movement, [Line(before, {"km_ok"} | self.uses(index)), *compute]

    def step_on_carried(self, node, write, call, value, carried):
        """`pages.write(f(acc, self.s[r]))`, acc started from a row of a field with
        state r (`acc = g(self.m[i], self.s[r])`): state r is in DST after acc's
        tiles already, so f runs on it there, and acc's write packs it back."""
        function = self.tile_function_of(call)
        name, field_, index = self.step_of_dst(call)
        if carried[0] != f"kField{camel(field_)}" or ast.dump(carried[4]) != ast.dump(
            index
        ):
            self.refuse(
                node,
                f"{name} holds a state of its start in DST, and this step takes "
                f"self.{field_}[{ast.unparse(index)}]",
                f"step on the state {name} started with",
            )
        unit, cxx = self.step_unit(node, function, value.width)
        before = (
            f"// {function.name}'s step on the state in DST {value.width}\n"
            "if (km_ok) {\n"
            f"{self.step_calls(cxx)}"
            "}"
        )
        self.wait(node, "compute", WAIT_INPUT)
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, self.text(node), function, cxx)
        )
        write.args[0] = ast.Name(id=name, ctx=ast.Load())
        movement, compute = self.write(node, write)
        return movement, [Line(before, {"km_ok"}), *compute]

    def step_state(self, node, field_, index, width=None):
        """State `index` of a step's field (or row `index` of a field of rows of
        `width` tiles): (its buffer, its token, the index's C++, a run-time bound
        check or ""); at the field's first use its token buffer, the token primed at
        the handler's start and taken at its end."""
        stored = self.compiled.fields[field_]
        if width is None:
            pages = stored.shape[0] if len(stored.shape) == 3 else None
            if stored.format.dtype != "float32" or not isinstance(pages, int):
                self.refuse(
                    node,
                    f"self.{field_} holds a step's states: fp32 tiles, a page each",
                    f"declare self.{field_}: km.L1[km.fp32, states, 32, 32]",
                )
            items, what = pages, "states"
        else:
            shape = stored.shape
            if stored.format.dtype != "float32" or len(shape) != 4 or shape[1] != width:
                self.refuse(
                    node,
                    f"self.{field_} holds rows of {width} fp32 tile(s), a row an index",
                    f"declare self.{field_}: km.L1[km.fp32, rows, {width}, 32, 32]",
                )
            items, what = shape[0], "rows"
            pages = items * width
        buffer, token = f"kField{camel(field_)}", f"kStepToken{camel(field_)}"
        filled_in = self.fill_loops.get(field_, (0, 0))
        if filled_in[0] > 0:
            self.refuse(
                node,
                f"self.{field_} is filled in a loop (line {filled_in[1]}) and its "
                f"{what} are freed once, at the handler's end: the next turn's fill "
                "would wait for room that is never freed",
                f"fill it before the loops, once (self.{field_}.fill(km."
                f"{spelling('stream')}(...)))",
            )
        if field_ not in self.stepped:
            if self.filled.get(field_) is None:
                self.refuse(
                    node,
                    f"self.{field_} is not filled above in this handler: a step's "
                    f"{what} start from a fill",
                    f"write self.{field_}.fill(km.{spelling('stream')}(t.pages(first, "
                    f"{pages}))) before the loop",
                )
            self.filled[field_] = None
            self.stepped[field_] = pages
            self.other_buffer(
                token,
                "token",
                "float32",
                16,
                1,
                f"line {node.lineno}: the {what}' token",
            )
            self.compiled.summed_fields.add(buffer)  # pack writes the states
            self.compiled.unpack_fp32.add(buffer)
            self.opening.append(
                Line(
                    f"if (compute.reserve_back({token}, 1)) {{  // self.{field_}'s "
                    f"{what}: the fill's\n"
                    f"    cb_push_back({token}, 1);\n}}"
                )
            )
            self.closing.append(
                Line(
                    f"if (km_ok && compute.wait_front({token}, 1)) {{\n"
                    f"    cb_pop_front({token}, 1);\n}}\n"
                    f"cb_pop_front({buffer}, {buffer}Pages);",
                    {"km_ok"},
                )
            )
        slot = self.integer(index)
        known = slot.rstrip("u").isdigit()
        if known and not int(slot.rstrip("u")) < items:
            self.refuse(
                node,
                f"self.{field_} holds {items} {what}, 0 .. {items - 1}",
                f"index it below {items}",
            )
        bound = (
            ""
            if known
            else (
                f"if (!(({slot}) < {items})) {{  // self.{field_}'s {what}\n"
                "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
                "    return;\n}\n"
            )
        )
        return buffer, token, slot, bound

    @staticmethod
    def state_taken(buffer, token, slot, dst):
        """The token taken and state `slot` copied into DST `dst` (an open block)."""
        return (
            f"km_ok = km_ok && compute.wait_front({buffer}, {buffer}Pages) &&\n"
            f"    compute.wait_front({token}, 1);\n"
            "if (km_ok) {\n"
            f"    cb_pop_front({token}, 1);\n"
            f"    KM_UNPACK_FORMAT_A({buffer});\n"
            f"    copy_init({buffer});\n"
            f"    copy_tile({buffer}, {slot}, {dst});\n"
        )

    def state_packed(self, compute, buffer, token, slot, dst, index):
        """DST `dst` packed into state `slot` and the token pushed, before the last
        compute line's DST release (after the row's packs)."""
        packed = (
            f"if (km_ok && compute.reserve_back({token}, 1)) {{  // pack's alone\n"
            f"    KM_PACK_FORMAT({buffer});\n"
            f"    pack_tile<true>({dst}, {buffer}, {slot});\n"
            f"    cb_push_back({token}, 1);\n"
            "}\n"
            "tile_regs_release();"
        )
        last = compute[-1]
        at = last.text.rindex("tile_regs_release();")
        last.text = (
            last.text[:at] + packed + last.text[at + len("tile_regs_release();") :]
        )
        last.uses = last.uses | self.uses(index)

    def row_given_back(self, compute, value):
        """Sums started from a row of a field (started_from_row) written elsewhere
        leave the row as it was: its token pushed back before the last compute line's
        DST release, so the row's next read finds it."""
        row_of = getattr(value, "row_of", None)
        if row_of is None:
            return
        given = (
            f"if (km_ok && compute.reserve_back({row_of[2]}, 1)) {{  // the row as "
            "it was\n"
            f"    cb_push_back({row_of[2]}, 1);\n"
            "}\n"
            "tile_regs_release();"
        )
        last = compute[-1]
        at = last.text.rindex("tile_regs_release();")
        last.text = (
            last.text[:at] + given + last.text[at + len("tile_regs_release();") :]
        )

    def step_of_field(self, call):
        """(field, state field, index) of `f(self.o, self.s[r])`, f a tile function,
        self.s a field indexed; else None."""
        if self.tile_function_of(call) is None or len(call.args) != 2:
            return None
        row, slot = call.args
        index = None
        if isinstance(row, ast.Subscript):  # a row of a field of rows
            row, index = row.value, row.slice
        field_ = self.field_name(row)
        state = isinstance(slot, ast.Subscript) and self.field_name(slot.value)
        return (field_, state, slot.slice, index) if field_ and state else None

    def started_from_field(self, node, name, call):
        """`acc = f(self.o, self.s[r])`, f a step-form function, self.o a field
        compute writes (1 to 3 tiles): DST sums that start from the field's tiles in
        DST 0 .., state r in the DST tile after them, f's body run on both; products
        then add into acc, and acc's write packs the state back into page r."""
        function = self.tile_function_of(call)
        field_, state, index, row = self.step_of_field(call)
        if row is not None:
            return self.started_from_row(node, name, call, field_, state, index, row)
        if self.dst_live is not None:
            self.refuse(
                node,
                f"{self.dst_live} still holds DST; a step's row starts at DST 0",
                "write it first",
            )
        if field_ not in self.packed:
            self.refuse(
                node,
                f"self.{field_} is not written above in this handler: sums start from "
                "a field compute writes",
                f"write self.{field_}.write(acc) before, e.g. of km.tile.zeros",
            )
        stored = self.compiled.fields[field_]
        width = math.prod(stored.shape[:-2])
        if not 1 <= width <= 3 or stored.format.dtype != "float32":
            self.refuse(
                node,
                f"self.{field_} is {width} {stored.format.dtype} tile(s); a step's row "
                "is 1 to 3 fp32 tiles",
                f"declare self.{field_}: km.L1[km.fp32, n, 32, 32], n at most 3",
            )
        buffer, token, slot, bound = self.step_state(node, state, index)
        unit, cxx = self.step_unit(node, function, width)
        source = f"kField{camel(field_)}"
        self.compiled.unpack_fp32.add(source)
        copies = "".join(f"    copy_tile({source}, {t}, {t});\n" for t in range(width))
        text = (
            f"{bound}"
            f"// {function.name}'s step: self.{field_}'s tiles into DST 0 .. "
            f"{width - 1}, self.{state}[{ast.unparse(index)}] into DST {width}\n"
            f"km_ok = km_ok && compute.wait_front({source}, {source}Pages);\n"
            "tile_regs_acquire();\n"
            f"{self.state_taken(buffer, token, slot, width)}"
            f"    KM_UNPACK_FORMAT_A({source});\n"
            f"    copy_init({source});\n"
            f"{copies}"
            f"{self.step_calls(cxx)}"
            "}"
        )
        self.wait(node, "compute", WAIT_INPUT)
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, self.text(node), function, cxx)
        )
        self.dst_next = max(self.dst_next, width + 1)  # the state's tile too
        if self.dst_next > DST_HALF:
            self.compiled.full_dst = True
        self.dst_live = name
        value = Value(
            "tile", name, dst=0, width=width, made_at=(len(self.loops), node.lineno)
        )
        value.state = (buffer, token, slot, width, index)  # packed at acc's write
        self.scope[name] = value
        return [], [Line(text, {"km_ok"} | self.uses(index))]

    def step_unit(self, node, function, width):
        """The step form's unit of `function(row, state)` over a row of `width` DST
        tiles and the state tile after them: it returns (the row, the state's
        per-row values) or (the row, the values, a word of word_max)."""
        made = (function.function, ("step",), width)
        for cxx, (key, unit) in self.compiled.units.items():
            if key == made:
                return unit, cxx
        where = function_line(function)
        drawn = "return (row, (m, l, ...)) or (row, (m, l), km.word_max(...))"
        if len(inspect.signature(function.function).parameters) != 2:
            self.refuse(
                node,
                f"the step function {function.name} ({where}) takes a row and its "
                "state",
                f"def {function.name}(row, state): ... {drawn}",
            )
        cxx = function.name
        while cxx in self.compiled.units:
            cxx += "_"
        tensors = [
            TensorParam(n, out, F32)
            for n, out in (
                ("row", False),
                ("state", False),
                ("result", True),
                ("state_out", True),
            )
        ]
        ir = KernelIR(cxx, tensors, [], {})
        try:
            with trace.Tracer(ir):
                rows, states, results, outs = (trace.TensorProxy(t) for t in tensors)
                for t in trace.grid(trace.num_tile_rows(rows)):
                    got = function.function(
                        trace.load(rows.tiles[t, :]), trace.load_state(states.tiles[t])
                    )
                    if not isinstance(got, tuple) or len(got) not in (2, 3):
                        self.refuse(
                            node,
                            f"the step function {function.name} ({where}) returns "
                            "the row and its state",
                            drawn,
                        )
                    trace.store(results.tiles[t, :], got[0])
                    word = got[2] if len(got) == 3 else None
                    trace.store_state(outs.tiles[t], got[1], word=word)
            fidelity = function.fidelity or "HiFi4"
            unit = step_program(ir, width, width, fidelity=fidelity, matrix_sum=False)
        except (PlanError, UnitError) as error:
            self.refuse(
                node,
                f"the step function {function.name} ({where}) is not lowered by the "
                f"tile compiler: {error}",
                "write this body inline: km.cxx(..., gives=km.Tile)",
            )
        self.compiled.units[cxx] = (made, unit)
        self.compiled.body_includes["compute"].append(f'#include "tile_{cxx}.hpp"')
        return unit, cxx

    def write_state(self, node, pages, field_):
        """`t.pages(first, n).write(self.s)`, self.s a step's n states: after the last
        step's token, each state copied into DST 0 and packed in t's format, written
        by mover B a page at a time."""
        count = self.stepped[field_]
        if pages.rect is not None or pages.count != str(count):
            self.refuse(
                node,
                f"self.{field_} is {count} states, a page each",
                f"write {pages.tensor}.{spelling('pages')}(first, {count})"
                f".write(self.{field_})",
            )
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        out = self.out_buffer(
            dtype, page_bytes, f"line {node.lineno}: {self.text(node)}"
        )
        buffer, token = f"kField{camel(field_)}", f"kStepToken{camel(field_)}"
        tile = f"km_state{node.lineno}"
        packed = "\n".join("        " + line for line in pack(0, out.name).split("\n"))
        compute = (
            f"// self.{field_}'s states, after the last step's\n"
            f"km_ok = km_ok && compute.wait_front({token}, 1);\n"
            "if (km_ok) {\n"
            f"    KM_UNPACK_FORMAT_A({buffer});\n"
            f"    copy_init({buffer});\n"
            f"    for (uint32_t {tile} = 0; {tile} < {count}; {tile}++) {{\n"
            "        tile_regs_acquire();\n"
            f"        copy_tile({buffer}, {tile}, 0);\n"
            f"{packed}\n"
            "    }\n"
            "}"
        )
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        movement = (
            f"for (uint32_t {tile} = 0; {tile} < {count}; {tile}++) {{\n"
            f"    if (!movement.wait_front({out.name}, 1) ||\n"
            f"        !km::write_pages(movement, command, {tensor}, {pages.first} + "
            f"{tile}, 1, get_read_ptr({out.name}))) {{\n"
            "        return false;\n"
            "    }\n"
            "    noc_async_writes_flushed();\n"
            f"    cb_pop_front({out.name}, 1);\n"
            "}"
        )
        self.compiled.includes |= {"pack", "ask"}
        self.moved(node, pages.tensor, "page write")
        self.wait(node, "compute", WAIT_OUTPUT)
        self.wait(node, "mover B", WAIT_DATA)
        return [Line(movement, range_uses(pages, self), mover="B")], [
            Line(compute, {"km_ok"})
        ]

    def write_function_of_sums(self, node, pages, call):
        """`t.tiles[r0:r1, c0:c1].write(f(self.h, self.g))`, f a tile function of
        fields of sums of one shape: each tile's fields copied into DST k (the unit's
        operand order), f's body, DST 0 packed in t's format and written a row at a
        time by mover B as `write(self.h)` is; the fields freed, the next sums start."""
        function = self.tile_function_of(call)
        fields = [self.field_name(arg) for arg in call.args]
        sides = {self.field_sides(node, f) for f in fields}
        if len(sides) > 1 or len(set(fields)) != len(fields):
            self.refuse(
                node,
                f"{function.name} takes fields of sums of one shape, each once, and "
                f"is given {', '.join('self.' + f for f in fields)}",
                "give it fields of one shape",
            )
        ((rows, columns),) = sides
        formats = [self.compiled.fields[f].format.dtype for f in fields]
        why = tile_format_refusal(formats)
        if why is not None:
            self.refuse(node, f"{function.name}: {why}", "give it fields of those")
        unit, cxx = self.tile_unit(node, function, formats)
        if unit.products:
            self.refuse(
                node,
                f"{function.name} has a matrix product (`@`), and its operands here "
                "are fields of sums in L1, which the unit reads from DST",
                f"add the product into a field (self.f += self.p(a, b)) and write "
                f"{function.name} of the fields",
            )
        rect = pages.rect
        bounds = pages.bounds or (None, None)
        written = rect[:2] if rect is not None else ("1", pages.count)
        fits = all(
            side_fits(side, bound, most)
            for side, bound, most in zip(written, bounds, (rows, columns), strict=True)
        )
        if not fits:
            self.refuse(
                node,
                f"the fields are {rows} x {columns} tiles; write a rectangle of them",
                f"write {pages.tensor}.{spelling('tiles')}[r:r + {rows}, c:c + "
                f"{columns}].write({function.name}(...))",
            )
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        name = f"kOutRows{columns}"
        held = [b.pages[0] for b in self.compiled.buffers if b.name == name]
        out = self.out_buffer(
            dtype, page_bytes, f"line {node.lineno}: {self.text(node)}", columns, name
        )
        out.pages = (max(held + [columns * min(rows, 2)]),)
        parameters = list(inspect.signature(function.function).parameters)
        buffers = {f: f"kField{camel(f)}" for f in fields}
        counts = [self.summed[f] for f in fields]
        made = " && ".join(f"{sums} > 0" for sums in counts)
        tile, out_room = f"km_tile{node.lineno}", f"km_room{node.lineno}"
        lines = [f"// {self.text(node)}: {function.name} of the fields' sums"]
        for f in fields:
            buffer, sums = buffers[f], self.summed[f]
            lines += [
                f"if (km_ok && {sums} > 0) {{",
                f"    if ({sums_room(f)}) {{  // pack's: its pages held",
                f"        cb_push_back({buffer}, {buffer}Pages);",
                "    }",
                f"    km_ok = compute.wait_front({buffer}, {buffer}Pages);",
                "}",
            ]
        first = buffers[fields[0]]
        lines += [
            f"if (km_ok && {made}) {{",
            f"    KM_UNPACK_FORMAT_A({first});",
            f"    copy_init({first});",
            f"    MATH((km_tile::{cxx}_init()));  // {function.name}'s setup",
            f"    bool {out_room} = true;  // pack's alone: it guards the packs only",
            f"    for (uint32_t {tile} = 0; km_ok && {tile} < {rows * columns}; "
            f"{tile}++) {{",
            "        tile_regs_acquire();",
        ]
        for dst, operand in enumerate(unit.operands):
            buffer = buffers[fields[parameters.index(operand)]]
            lines.append(f"        copy_tile({buffer}, {tile}, {dst});")
        lines += [
            f"        MATH((km_tile::{cxx}(0)));",
            "        tile_regs_commit();",
            "        tile_regs_wait();",
            f"        {out_room} = {out_room} && compute.reserve_back({out.name}, 1);",
            f"        if ({out_room}) {{",
            f"            KM_PACK_FORMAT({out.name});",
            f"            pack_tile(0, {out.name});",
            f"            cb_push_back({out.name}, 1);",
            "        }",
            "        tile_regs_release();",
            "    }",
            *[f"    cb_pop_front({b}, {b}Pages);" for b in buffers.values()],
            "}",
            *[f"{sums} = 0;" for sums in counts],
        ]
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        row = f"km_row{node.lineno}"
        row_step = rect[2] if rect is not None else columns
        whole = str(written[0]) == str(rows)
        part = "" if whole else f"{row} < ({written[0]}) &&\n    "
        movement = [
            f"if ({made}) {{",
            f"    for (uint32_t {row} = 0; {row} < {rows}; {row}++) {{",
            f"        if (!movement.wait_front({out.name}, {columns}) ||",
            f"            {'(' if part else ''}{part}!km::write_pages(movement, "
            f"command, {tensor}, {pages.first} + {row} * {row_step}, {written[1]}, "
            f"get_read_ptr({out.name})){')' if part else ''}) {{",
            "            return false;",
            "        }",
            "        noc_async_writes_flushed();",
            f"        cb_pop_front({out.name}, {columns});",
            "    }",
            "}",
            *[f"{sums} = 0;" for sums in counts],
        ]
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, self.text(node), function, cxx)
        )
        self.compiled.includes |= {"pack", "ask"}
        self.moved(node, pages.tensor, "page write")
        self.wait(node, "compute", WAIT_OUTPUT)
        self.wait(node, "mover B", WAIT_DATA)
        return [Line("\n".join(movement), range_uses(pages, self), mover="B")], [
            Line("\n".join(lines), {"km_ok"})
        ]

    def read_word(self, node, name, pages):
        """`word = pages.read()` of one page: its first word, on mover B, from its own
        L1 page (read, settled, the L1 cache invalidated)."""
        self.pages_only(node, pages)
        if pages.count != "1":
            self.refuse(node, "a word is one page", "read pages(first, 1)")
        self.check_name(node, name)
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        buffer = self.other_buffer(
            "kWordB", "word", dtype, page_bytes, 1, f"line {node.lineno}: "
            f"{self.text(node)} (mover B's word page)",
        )  # fmt: skip
        buffer.page_bytes = max(buffer.page_bytes, page_bytes)
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        text = (
            f"uint32_t {name} = 0;\n"
            f"if (!km::read_pages(movement, command, {tensor}, {pages.first}, 1,\n"
            f"                    get_write_ptr({buffer.name}))) {{\n"
            "    return false;\n"
            "}\n"
            "noc_async_read_barrier();\n"
            "invalidate_l1_cache();\n"
            f"{name} = km::l1(get_write_ptr({buffer.name}))[0];"
        )  # fmt: skip
        self.moved(node, pages.tensor, "word read")
        self.compiled.includes.add("ask")
        self.scope[name] = Value("word", name, mover="B")
        return [Line(text, range_uses(pages, self), mover="B")], []

    def read_words(self, node, target, value):
        """`a, b = pages.read()[0:2]` of one page: its words 0 and 1 as ints of
        every kernel. Mover A reads the page into a page it hands compute (unpack
        takes the words and hands them to math and pack by mailbox); mover B reads
        its own."""
        call = value.value if isinstance(value, ast.Subscript) else None
        part = value.slice if call is not None else None
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == spelling("read")
            and not call.args
            and not call.keywords
            and self.is_range(call.func.value)
            and isinstance(part, ast.Slice)
            and part.step is None
            and part.upper is not None
            and all(isinstance(name, ast.Name) for name in target.elts)
        ):
            self.refuse(
                node,
                "names are given a page's words, by a constant slice",
                f"write `first, count = t.{spelling('pages')}(i, 1)."
                f"{spelling('read')}()[0:2]`",
            )
        lower = 0 if part.lower is None else self.constant_int(part.lower)
        upper = self.constant_int(part.upper)
        names = [name.id for name in target.elts]
        pages = self.range_of(call.func.value)
        self.pages_only(node, pages)
        if pages.count != "1":
            self.refuse(node, "words are of one page", "read pages(first, 1)")
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        if not 0 <= lower < upper <= page_bytes // 4 or upper - lower != len(names):
            self.refuse(
                node,
                f"words {lower} to {upper - 1} of a page of {page_bytes // 4} words "
                f"for {len(names)} names",
                "give a slice of the page with a name each word",
            )
        for name in names:
            self.check_name(node, name)
            self.scope[name] = Value("int", name)
        what = f"line {node.lineno}: {self.text(node)}"
        handed = self.other_buffer(
            "kWords", "word", dtype, page_bytes, 1, f"{what} (mover A's, to compute)"
        )
        own = self.other_buffer(
            "kWordB", "word", dtype, page_bytes, 1, f"{what} (mover B's word page)"
        )
        for buffer in (handed, own):
            buffer.page_bytes = max(buffer.page_bytes, page_bytes)
        declared = "\n".join(f"[[maybe_unused]] uint32_t {name} = 0;" for name in names)
        taken = "\n".join(
            f"    {name} = km_page_words[{lower + n}];" for n, name in enumerate(names)
        )
        tensor = f"{camel(pages.tensor)}({pages.tensor})"

        def mover(buffer, reserve):
            opening = (
                f"if (!movement.reserve({buffer}, 1) ||\n    " if reserve else ("if (")
            )
            return (
                f"{declared}\n"
                f"{opening}!km::read_pages(movement, command, {tensor}, "
                f"{pages.first}, 1,\n                    get_write_ptr({buffer}))) {{\n"
                "    return false;\n"
                "}\n"
                "noc_async_read_barrier();\n"
                "invalidate_l1_cache();\n"
                "{\n"
                "    volatile tt_l1_ptr uint32_t* km_page_words =\n"
                f"        km::l1(get_write_ptr({buffer}));\n"
                f"{taken}\n"
                "}"
            )

        uses = range_uses(pages, self)
        movement = [
            Line(mover("kWords", True) + "\ncb_push_back(kWords, 1);", uses, mover="A"),
            Glue(mover("kWordB", False), set(names), uses, mover="B"),
        ]
        hands = "\n".join(
            f"    compute.hand(ckernel::ThreadId::{cpu}ThreadId, {name});"
            for name in names
            for cpu in ("Math", "Pack")
        )
        heard = "\n".join(
            f"{name} = compute.handed(ckernel::ThreadId::UnpackThreadId);"
            for name in names
        )
        compute = [
            Line(
                f"{declared}\n"
                "if (!compute.wait_front(kWords, 1)) {\n"
                "    return;\n"
                "}\n"
                "#if defined(TRISC_UNPACK)\n"
                "{\n"
                "    const volatile uint32_t* km_page_words =\n"
                "        reinterpret_cast<const volatile uint32_t*>(\n"
                "            get_tile_l1_byte_address(kWords, 0)\n"
                "        );\n"
                '    asm volatile("fence" ::: "memory");\n'
                f"{taken}\n"
                f"{hands}\n"
                "}\n"
                "#else\n"
                f"{heard}\n"
                "#endif\n"
                "cb_pop_front(kWords, 1);"
            )
        ]
        self.moved(node, pages.tensor, "word read")
        self.compiled.includes.add("ask")
        self.handover(node, ", ".join(names), "unpack", ("math", "pack"))
        self.wait(node, "mover A", WAIT_DATA)
        self.wait(node, "mover B", WAIT_DATA)
        self.wait(node, "compute", WAIT_INPUT)
        return movement, compute

    def handover(self, node, what, sender, receivers):
        """A value handed between compute CPUs by mailbox, for the build's report."""
        self.compiled.handovers.append(
            f"{self.method.name}, line {node.lineno}: {what} from {sender} to "
            f"{' and '.join(receivers)} by mailbox"
        )

    def word_sum(self, node, value):
        """`word += x` (or another operator) on a mover's word: its mover's line; x
        reads ints and that mover's words."""
        op = OPERATORS.get(type(node.op))
        if op is None:
            self.refuse(node, "an operator level 1 has no C++ for", "use + - * etc.")
        for item in ast.walk(node.value):
            other = isinstance(item, ast.Name) and self.scope.get(item.id)
            if other and other.kind == "word" and other.mover != value.mover:
                self.refuse(
                    node,
                    f"{item.id} is another CPU's word",
                    "add words of one mover",
                )
        self.allow_words = True
        text = f"{value.cxx} {op}= {self.integer(node.value)};"
        self.allow_words = False
        uses = self.uses(node.value) | {value.cxx}
        return [Line(text, uses, mover=value.mover)], []

    def write_word(self, node, pages, value):
        """`pages.write(word)`: a mover's word as one page (the word first, zeros
        after) from its own L1 page, written as a page write is, settled."""
        if value.mover not in ("A", "B"):
            self.refuse(
                node,
                f"{value.cxx} is compute's word; a page of words is a mover's",
                'make the word on="mover_a" or on="mover_b"',
            )
        if pages.count != "1":
            self.refuse(node, "a word is one page", "write pages(first, 1)")
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        if page_bytes % 4:
            self.refuse(
                node, f"a page of {page_bytes} B is not whole words", "use words"
            )
        buffer = self.other_buffer(
            f"kWord{value.mover}", "word", dtype, page_bytes, 1, f"line {node.lineno}: "
            f"{self.text(node)} (mover {value.mover}'s word page)",
        )  # fmt: skip
        buffer.page_bytes = max(buffer.page_bytes, page_bytes)
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        text = (
            "{\n"
            "    volatile tt_l1_ptr uint32_t* km_word_page =\n"
            f"        km::l1(get_write_ptr({buffer.name}));\n"
            f"    km_word_page[0] = {value.cxx};\n"
            "    for (uint32_t km_rest = 1; "
            f"km_rest < {page_bytes // 4}; km_rest++) {{\n"
            "        km_word_page[km_rest] = 0;\n"
            "    }\n"
            f"    if (!km::write_pages(movement, command, {tensor}, {pages.first}, 1,\n"
            f"                         get_write_ptr({buffer.name}))) {{\n"
            "        return false;\n"
            "    }\n"
            "    noc_async_write_barrier();\n"
            "}"
        )  # fmt: skip
        self.moved(node, pages.tensor, "page write")
        self.compiled.includes.add("ask")
        uses = range_uses(pages, self) | {value.cxx}
        return [Line(text, uses, mover=value.mover)], []

    def out_buffer(self, dtype, page_bytes, what, width=1, name=None):
        """The buffer of the page(s) compute packs and mover B writes: the program's
        next one of this handler (handlers share them), two writes of `width`; or the
        one `name`d (rows of a field's sums: rows of one width a buffer)."""
        if name is None:
            name = f"kOut{self.writes}"
            self.writes += 1
        for buffer in self.compiled.buffers:
            if buffer.name == name and (buffer.dtype, buffer.page_bytes) == (
                dtype,
                page_bytes,
            ):
                buffer.pages = (max(buffer.pages[0], WRITE_ITEMS * width),)
                return buffer
        while any(b.name == name for b in self.compiled.buffers):
            name += "b"
        taken = {b.index_a for b in self.compiled.buffers} | {
            b.index_b for b in self.compiled.buffers
        }
        taken |= self.compiled.reserved
        index = FIRST_WRITE
        while index in taken:
            index += 1
        if index > LAST_INDEX:
            raise Refused(f"{self.compiled.name}: more buffers than the indices")
        buffer = Buffer(
            name, "out", index, None, dtype, page_bytes, (WRITE_ITEMS * width,), 1, what
        )
        self.compiled.buffers.append(buffer)
        return buffer

    def branch(self, node):
        if self.self_call(node.test, "seeding"):
            return self.seeding_branch(node)
        broadcast = self.broadcast_branch(node)
        if broadcast is not None:
            return broadcast
        condition = self.integer(node.test)
        uses = self.uses(node.test)
        live = self.dst_live
        leader, self.in_leader = self.in_leader, self.in_leader or self.leader_test(
            node
        )
        if self.leader_test(node):
            self.leader_condition = (condition, uses)  # a held sum's free (sum_held)
        before = set(self.scope)
        then_m, then_c = self.statements(node.body)
        self.scoped("if", node, before)
        self.in_leader = leader
        else_m, else_c = self.statements(node.orelse)
        self.scoped("if", node, before)
        if self.dst_live != live:
            self.refuse(
                node,
                "a tile held in DST on one branch only",
                "write the tile after the if",
            )
        movement, compute = [], []
        for body_then, body_else, out in (
            (then_m, else_m, movement),
            (then_c, else_c, compute),
        ):
            if body_then:
                out.append(Block(f"if ({condition}) {{", "}", body_then, uses))
                if body_else:
                    out.append(Block("else {", "}", body_else))
            elif body_else:
                out.append(Block(f"if (!({condition})) {{", "}", body_else, uses))
        return movement, compute

    def scoped(self, kind, node, before):
        """The names first assigned inside an if or a loop end with it, as C++'s
        do: a line after it that reads one is refused (not_ended). Ranges are the
        build's alone and stay."""
        for name in set(self.scope) - before:
            if getattr(self.scope[name], "kind", None) in ("int", "word", "partition"):
                self.scope.pop(name)
                self.ended[name] = (kind, node.lineno)
            elif getattr(self.scope[name], "kind", None) == "moved":
                self.scope[name].ended = self.scope[name].ended or (kind, node.lineno)

    def not_ended(self, node, name):
        if name in self.ended and name not in self.scope:
            kind, line = self.ended[name]
            self.refuse(
                node,
                f"{name} is first assigned inside the {kind} at line {line}, and "
                f"ends with it",
                f"assign it before the {kind}",
            )

    def not_parameter(self, node, name):
        if any(param.name == name for param in self.params):
            self.refuse(
                node,
                f"{name} is a parameter, a command word: it is not assigned again",
                f"give the new value a name of its own: n = {name} + 1",
            )

    def leader_test(self, node):
        """True when the if's test is `self.rank == self.leader` (either way round)."""
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
        ):
            return False
        sides = {
            side.attr
            for side in (test.left, test.comparators[0])
            if isinstance(side, ast.Attribute)
            and isinstance(side.value, ast.Name)
            and side.value.id == "self"
        }
        return sides == {spelling("rank"), spelling("leader")}

    def put_line(self, node, call):
        """`t.items(size=B).put(self.f, at=i)`: B bytes of own field f (or of its item
        k, `self.f[k]`) at item i of t, by a mover (on=, mover B unstated); unwaited,
        the mover's puts waited for at the handler's end (km::put_bytes, put_wait)."""
        items = call.func.value
        if not (
            isinstance(items, ast.Call)
            and isinstance(items.func, ast.Attribute)
            and items.func.attr == spelling("items")
            and isinstance(items.func.value, ast.Name)
            and getattr(self.scope.get(items.func.value.id), "kind", None) == "tensor"
            and not items.args
            and [k.arg for k in items.keywords] == [spelling("item_size")]
        ):
            self.refuse(
                node,
                f"a put is `t.{spelling('items')}({spelling('item_size')}=B)."
                f"{spelling('put')}(source, {spelling('put_at')}=i)`, t a tensor or a "
                "region parameter",
                "give the tensor's items by their size alone",
            )
        tensor = items.func.value.id
        size = self.constant_int(items.keywords[0].value)
        _, page_bytes = self.tensor_format(node, tensor)
        if size < 4 or size % 4 or page_bytes % size:
            self.refuse(
                node,
                f"items of {size} B in pages of {page_bytes} B: a put is whole words, "
                "a page a whole number of items",
                f"give a {spelling('item_size')} of 4 B steps dividing {page_bytes}",
            )
        keywords = {k.arg: k.value for k in call.keywords}
        if (
            len(call.args) != 1
            or set(keywords) - {spelling("put_at"), "on"}
            or (spelling("put_at") not in keywords)
        ):
            self.refuse(
                node,
                f"{spelling('put')}(source, {spelling('put_at')}=i), on= a mover",
                "give the source, the item and (if not mover B) the mover",
            )
        on = keywords.get("on")
        on = (
            on.value
            if isinstance(on, ast.Constant)
            else "mover_b" if on is None else ""
        )
        mover = {"mover_a": "A", "mover_b": "B"}.get(on)
        if mover is None:
            self.refuse(
                node,
                "a put is a mover's NoC write",
                'write on="mover_a" or on="mover_b" (mover B when not given)',
            )
        self.mover_word(node, call.args[0], mover)
        source, uses = self.put_source(node, call.args[0], size)
        at = keywords[spelling("put_at")]
        index = self.mover_integer(node, at, mover)
        uses |= self.uses(at)
        per = page_bytes // size
        if mover not in self.putters:
            self.putters.append(mover)
        text = (
            f"if (!km::put_bytes(movement, command, {camel(tensor)}({tensor}), "
            f"({index}) / {per}, (({index}) % {per}) * {size}, {size}, {source})) {{\n"
            "    return false;\n}"
        )
        return [Line(text, uses | {tensor}, mover=mover)], []

    def put_source(self, node, source, size):
        """A put's source: an own field's L1 start, or item k of it (`self.f[k]`)."""
        item = None
        if (
            isinstance(source, ast.Name)
            and getattr(self.scope.get(source.id), "kind", None) == "page"
        ):  # the page a mover reads of a stream
            return source.id, {source.id}
        if isinstance(source, ast.Subscript):
            source, item = source.value, source.slice
        field_ = self.field_name(source)
        if field_ not in self.compiled.own:
            self.refuse(
                node,
                "a put's source is an own field (km.own(): the bodies' buffer), whole "
                "or an item of it",
                f"put self.f or self.f[k], f declared = km.{spelling('own')}(...)",
            )
        address = f"get_write_ptr(kField{camel(field_)})"
        if item is None:
            return address, set()
        return f"{address} + ({self.integer(item)}) * {size}", self.uses(item)

    def mover_integer(self, node, value, mover):
        """The C++ of an int of a line on `mover`: a word of that mover by its name,
        else an int of every CPU; a word of another CPU refused."""
        self.mover_word(node, value, mover)
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            page = self.scope.get(value.value.id)
            if page is not None and page.kind == "page":
                return f"{value.value.id}_words[{self.integer(value.slice)}]"
        if (
            isinstance(value, ast.Name)
            and getattr(self.scope.get(value.id), "kind", None) == "word"
        ):
            return value.id
        return self.integer(value)

    def mover_word(self, node, value, mover):
        """Refuses a word of another CPU where a line runs on `mover`."""
        for item in ast.walk(value):
            if not isinstance(item, ast.Name) or item.id not in self.scope:
                continue
            known = self.scope[item.id]
            if known.kind in ("word", "page") and known.mover != mover:
                where = {"A": "mover A", "B": "mover B"}.get(known.mover, known.mover)
                self.refuse(
                    node,
                    f"{item.id} is a word of {where}, and this line runs on mover "
                    f"{mover}",
                    "give it a word of that mover, or an int of every CPU",
                )

    def put_waits(self):
        """The handler's end on each mover that put items: its puts waited for."""
        return [
            Line(
                "if (!km::put_wait(movement, command)) {\n    return false;\n}",
                set(),
                mover=mover,
            )
            for mover in self.putters
        ]

    def bump_line(self, node, name, call):
        """`slot = self.f.bump(i)`: word i of field f, then one more, on the CPU that
        owns f's words: an own field's push= mover, or a word field's unpack CPU."""
        field_ = self.field_name(call.func.value)
        if len(call.args) != 1 or call.keywords:
            self.refuse(node, f"{spelling('bump')}(i): one word's index", "give one")
        self.check_name(node, name)
        uses = self.uses(call.args[0])
        if field_ in self.compiled.word_fields:
            index = self.integer(call.args[0])
            self.scope[name] = Value("word", name, mover="unpack")
            word = f"km_field_{field_}[{index}]"
            text = f"uint32_t {name} = 0;\nUNPACK(({name} = {word}++));"
            return [], [Line(text, uses | {f"km_field_{field_}"})]
        mover = {"mover_a": "A", "mover_b": "B"}.get(self.compiled.own.get(field_))
        if field_ not in self.compiled.own or mover is None:
            self.refuse(
                node,
                f"self.{field_}'s words are no one CPU's: a bump runs on the CPU that "
                "owns them",
                f"declare it km.{spelling('own')}(push=\"mover_a\") or "
                '(push="mover_b"), or a field of km.u32 words (unpack\'s)',
            )
        index = self.mover_integer(node, call.args[0], mover)
        self.scope[name] = Value("word", name, mover=mover)
        text = (
            f"uint32_t {name} = 0;\n{{\n    volatile tt_l1_ptr uint32_t* km_bumped = "
            f"km::l1(get_write_ptr(kField{camel(field_)}));\n    {name} = "
            f"km_bumped[{index}]++;\n}}"
        )
        return [Line(text, uses, mover=mover)], []

    def field_name(self, node):
        """The field `node` is (`self.name`), or None."""
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in self.compiled.fields
        ):
            return node.attr
        return None

    def fill(self, node, call, field):
        """`self.f.fill(km.Stream(t.pages(first, n)))`: mover A reads the n pages into
        the field's L1 and hands it on; one km.cxx body takes it, then it is free."""
        if field in self.compiled.own:
            self.refuse(
                node,
                f"self.{field} is an own field: only bodies use it, no line fills it",
                f"declare it without km.{spelling('own')}() to fill it from a stream",
            )
        stream = call.args[0] if len(call.args) == 1 and not call.keywords else None
        if not (
            self.construct(stream) == "stream"
            and len(stream.args) == 1
            and not stream.keywords
            and self.is_range(stream.args[0])
        ):
            self.refuse(
                node,
                "a field is filled from a stream of a page range",
                f"write self.{field}.fill(km.Stream(tensor.pages(first, count)))",
            )
        if self.filled.get(field) is not None:
            self.refuse(
                node,
                f"self.{field} is filled above and no body took it",
                "fill it once, before the body that takes it",
            )
        pages = self.range_of(stream.args[0])
        self.pages_only(node, pages)
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        value = self.compiled.fields[field]
        if (dtype, page_bytes) != (value.format.dtype, value.format.tile_bytes):
            self.refuse(
                node,
                f"{pages.tensor} has {dtype} pages and self.{field} "
                f"{value.format.dtype} tiles",
                "fill a field from a tensor of its format",
            )
        buffer, pages_name = f"kField{camel(field)}", f"kField{camel(field)}Pages"
        pitch = None  # a row's tiles in the field: the rectangle's columns, or the
        if pages.rect is not None and isinstance(pages.rect[1], str):  # field's
            pitch = self.field_sides(node, field)[1]
            if pages.bounds[1] is None or pages.bounds[1] > pitch:
                self.stated_words(stream.args[0], "a fill of a field")
                self.refuse(
                    node,
                    f"the rectangle's columns are a word and self.{field}'s rows are "
                    f"{pitch} tiles: the build must know they are at most {pitch}",
                    f"write {pages.tensor}.{spelling('tiles')}[r0:r1, "
                    f"c:{spelling('min')}(c + {pitch}, e)]",
                )
        count = pages.count if pages.rect is None else ("rect", pitch or pages.rect[1])
        self.producer(node, field, "fill")
        self.compiled.fills.append((field, count, self.method.name, node.lineno))
        self.compiled.includes.add("ask")
        self.filled[field] = (node, self.blocks[-1])
        self.fill_loops[field] = (len(self.loops), node.lineno)
        self.fill_loop_of[field] = self.loops[-1] if self.loops else None
        self.moved(node, pages.tensor, "fill")
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        text = (
            f"if (!movement.reserve({buffer}, {pages_name}) ||\n"
            f"    !km::read_pages(movement, command, {tensor}, {pages.first}, "
            f"{pages_name}, get_write_ptr({buffer}))) {{\n"
            "    return false;\n"
            "}\n"
            "noc_async_read_barrier();\n"
            f"cb_push_back({buffer}, {pages_name});"
        )
        if pages.rect is not None:  # each row of the rectangle, its run of pages
            rows, columns, row_step = pages.rect[:3]
            row = f"km_row_{field}"
            text = (
                f"// the rectangle's rows, at most the field's\n"
                f"if (({rows}) * {pitch or columns} > {pages_name} ||\n"
                f"    !movement.reserve({buffer}, {pages_name})) {{\n"
                "    return false;\n"
                "}\n"
                f"for (uint32_t {row} = 0; {row} < ({rows}); {row}++) {{\n"
                f"    if (!km::read_pages(movement, command, {tensor}, {pages.first} + "
                f"{row} * {row_step}, {columns}, get_write_ptr({buffer}) + {row} * "
                f"{pitch or columns} * {page_bytes})) {{\n"
                "        return false;\n"
                "    }\n"
                "}\n"
                "noc_async_read_barrier();\n"
                f"cb_push_back({buffer}, {pages_name});"
            )
        self.wait(node, "mover A", WAIT_DATA)
        compute = []
        if self.kept_field(field):
            held = f"km_held_{field}"
            self.wait(node, "compute", WAIT_INPUT)
            compute.append(
                Line(
                    f"// self.{field} stays between commands: the last fill freed "
                    "before this one\n"
                    f"static bool {held} = false;\n"
                    f"if ({held}) {{\n"
                    f"    if (!compute.wait_front({buffer}, {pages_name})) {{\n"
                    "        return;\n    }\n"
                    f"    cb_pop_front({buffer}, {pages_name});\n"
                    "}\n"
                    f"{held} = true;"
                )
            )
        return [Line(text, range_uses(pages, self), mover="A")], compute

    def zero(self, node, call, field):
        """`self.f.zero()`: compute packs a zero tile into each of the field's pages and
        pushes them, as a fill would (no DRAM read); its lines then read it as a filled
        field."""
        if field in self.compiled.own:
            self.refuse(
                node,
                f"self.{field} is an own field: only bodies use it, no line zeroes it",
                f"declare it without km.{spelling('own')}() to zero it",
            )
        if call.args or call.keywords:
            self.refuse(node, "a zero takes no arguments", f"write self.{field}.zero()")
        if self.filled.get(field) is not None:
            self.refuse(
                node,
                f"self.{field} is filled above and no body took it",
                "zero it once, before the lines that read it",
            )
        if self.dst_live is not None:
            self.refuse(
                node,
                f"{self.dst_live} still holds DST, and a zero packs DST 0",
                "write it first",
            )
        buffer, pages = f"kField{camel(field)}", f"kField{camel(field)}Pages"
        self.producer(node, field, "zero")
        self.compiled.includes |= {"fill", "pack"}
        self.compiled.summed_fields.add(buffer)  # pack writes it: its pack format
        self.filled[field] = (node, self.blocks[-1])
        self.fill_loops[field] = (len(self.loops), node.lineno)
        self.wait(node, "compute", WAIT_OUTPUT)
        text = (
            f"// self.{field} zeroed on the core: a zero tile packed into each page\n"
            "tile_regs_acquire();\n"
            "fill_tile_init();\n"
            "fill_tile(0, 0.0f);\n"
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"if (km_ok && compute.reserve_back({buffer}, {pages})) {{\n"
            f"    KM_PACK_FORMAT({buffer});\n"
            f"    for (uint32_t km_page = 0; km_page < {pages}; km_page++) {{\n"
            f"        pack_tile<true>(0, {buffer}, km_page);\n"
            "    }\n"
            f"    cb_push_back({buffer}, {pages});\n"
            "}\n"
            "tile_regs_release();"
        )
        return [], [Line(text, {"km_ok"})]

    def producer(self, node, field, kind):
        """A field's pages are pushed by one producer: mover A's fill, or compute's
        adds of a product, in every handler of the class."""
        other = self.compiled.producers.setdefault(
            field, (kind, self.method.name, node.lineno, self.text(node))
        )
        if other[0] != kind:
            _, handler, line, text = other
            pusher = "mover A" if other[0] == "fill" else "compute"
            self.refuse(
                node,
                f"self.{field} is pushed by {pusher}"
                f" too ({handler}, line {line}, `{text}`): a buffer has one producer",
                "fill it, or add products into it, not both: give each its own field",
            )

    def sum_held(self, value):
        """True when `value` is a reduce's sum made outside the range loop a line
        writes it or a body takes it in, under the leader's if: the sum is held
        (waited for each turn, never freed in it) and freed after that loop."""
        return (
            value is not None
            and value.kind == "reduced"
            and not value.popped
            and value.made_at is not None
            and value.made_at[0] < len(self.loops)
            and self.in_leader
            and self.leader_condition is not None
        )

    def sum_freed_after(self, value, freed):
        """A held sum's free: after the outermost loop the sum was made outside of,
        under the leader's condition (only the leader holds the sum)."""
        condition, uses = self.leader_condition
        loop = self.loops[value.made_at[0]]
        loop.setdefault("after", []).append(
            Line(
                f"if ({condition}) {{  // {value.cxx}, held across the loop, freed\n"
                f"    {freed}\n}}",
                set(uses),
            )
        )

    def held_sum_copy(self, value, buffer, width):
        """A turn's write of a held sum: its tiles copied (fp32, exact) from the sum's
        buffer into DST and packed into the write's own `buffer`, the sum kept for
        the next turn."""
        source = value.buffer
        self.compiled.unpack_fp32.add(source.name)
        copies = "\n".join(f"copy_tile({source.name}, {t}, {t});" for t in range(width))
        again = f"{copies}\n" + pack(0, buffer.name, width)
        body = "\n".join("    " + line for line in again.split("\n"))
        self.sum_freed_after(value, f"cb_pop_front({source.name}, {width});")
        return Line(
            f"// {value.cxx}, the sum held across the loop, copied for this turn\n"
            f"km_ok = km_ok && compute.wait_front({source.name}, {width});\n"
            "if (km_ok) {\n"
            f"    KM_UNPACK_FORMAT_A({source.name});\n"
            f"    copy_init({source.name});\n"
            "    tile_regs_acquire();\n"
            f"{body}\n}}",
            {"km_ok"},
        )

    def fill_turn_loop(self, field):
        """The range loop around a product of `field` in which the field is filled
        each turn (its fill a line of that loop, the loop still open, the field read
        by this handler alone), or None."""
        loop = self.fill_loop_of.get(field)
        if loop is None or self.kept_field(field):
            return None
        return loop if any(open_ is loop for open_ in self.loops) else None

    def field_freed(self, field, buffer):
        """A product's field freed once it is read: at the handler's end, or for a
        field filled each turn at the end of that turn, so the next fill finds room."""
        pop = f"cb_pop_front({buffer}, {buffer}Pages);"
        loop = self.fill_turn_loop(field)
        if loop is None:
            self.closing.append(Line(pop))
        else:
            freed = f"// self.{field}, this turn's fill, freed for the next\n{pop}"
            loop.setdefault("frees", []).append(Line(freed))

    def kept_field(self, field):
        """Whether a field is filled by one handler and read by another: it stays in
        L1 between commands (its next fill frees it)."""
        fillers, readers = self.compiled.field_users.get(field, ([], []))
        return bool(fillers and readers)

    def take_field(self, node, field, where, bound, before, after):
        """A field an inline body takes: compute's three its buffer index, one compute
        CPU its L1 address (unpack's, handed on), mover B its address once mover A
        filled it. Freed after the body when filled here; a kept field (another
        handler reads it) stays, also in its filler (its next fill frees it)."""
        stays = self.kept_field(field)
        kept = stays and field not in self.filled
        if not kept and (field not in self.filled or self.filled[field] is None):
            self.refuse(
                node,
                f"self.{field} is not filled above in this handler (or a body took "
                "it), and no other handler keeps it filled",
                f"write self.{field}.fill(km.Stream(...)) before this body, or in "
                "a handler of its own",
            )
        if not kept and self.filled[field][1] != self.blocks[-1]:
            self.refuse(
                node,
                f"self.{field} is filled in another block: every core that fills it "
                "must free it",
                "take it in the block of its fill (or fill it inside this one)",
            )
        if where == "A":
            self.refuse(
                node,
                f"self.{field} is filled by mover A and handed on",
                'take it in a body on compute or on="mover_b"',
            )
        if not kept:
            self.filled[field] = None
        buffer, pages_name = f"kField{camel(field)}", f"kField{camel(field)}Pages"
        if where == "B":
            before.append(
                Line(
                    f"if (!movement.wait_front({buffer}, {pages_name})) {{\n"
                    "    return false;\n}",
                    mover="B",
                )
            )
            bound.append(f"const uint32_t {field} = get_read_ptr({buffer});")
            if not stays:
                after.append(Line(f"cb_pop_front({buffer}, {pages_name});", mover="B"))
            self.wait(node, "mover B", WAIT_DATA)
            return
        before.append(
            Line(
                f"if (!compute.wait_front({buffer}, {pages_name})) {{\n"
                "    return;\n}"
            )
        )
        if where == "compute":
            bound.append(f"const uint32_t {field} = {buffer};")
        else:
            address = f"get_tile_l1_byte_address({buffer}, 0)"
            self.cpu_address(
                node, field, address, where, bound, before, after, not stays
            )
        if not stays:
            after.append(Line(f"cb_pop_front({buffer}, {pages_name});"))
        self.wait(node, "compute", WAIT_INPUT)

    def take_constant(self, item, bound):
        """`self.K` of a constant of the class (not a field), taken: the body's
        `constexpr uint32_t K`, this program's value, on any CPU. False for any other
        name."""
        if not (
            isinstance(item, ast.Attribute)
            and isinstance(item.value, ast.Name)
            and item.value.id == "self"
            and item.attr not in self.compiled.fields
            and item.attr in self.compiled.class_constants
        ):
            return False
        bound.append(f"constexpr uint32_t {item.attr} = {self.class_constant(item)};")
        return True

    def take_own(self, node, field, where, text, bound):
        """An own field a body takes: its buffer index by its name, on any CPU, no wait
        and no free. The CPU whose bodies push it (cb_push_back of the name: a mover's,
        or compute's pack CPU) is one: the field's push=, or the one the bodies show."""
        bound.append(f"const uint32_t {field} = kField{camel(field)};")
        if not re.search(rf"\bcb_push_back\s*\(\s*{field}\b", text):
            return
        cpu = {"A": "mover_a", "B": "mover_b"}.get(where, "pack")
        declared = self.compiled.own[field]
        own = f"km.{spelling('own')}"
        if declared is None:
            self.refuse(
                node,
                f"self.{field} is {own}(push=None): written in place, never pushed, "
                "and this body pushes it",
                f"drop the push, or declare {own}(push={cpu!r})",
            )
        if declared != "found" and declared != cpu:
            self.refuse(
                node,
                f"self.{field} is pushed by {declared} ({own}(push={declared!r})), "
                f"and this body pushes it on {cpu}: a buffer has one producer",
                f"push it on {declared} alone",
            )
        pushers = self.compiled.pushers.setdefault(field, {})
        others = {c: at for c, at in pushers.items() if c != cpu}
        if others:
            (other, (handler, line)), *_ = others.items()
            self.refuse(
                node,
                f"self.{field} is pushed on {other} too ({handler}, line {line}): a "
                "buffer has one producer",
                f"push it on one CPU, and say which: {own}(push={other!r})",
            )
        pushers.setdefault(cpu, (self.method.name, node.lineno))

    def field_use(self, node):
        """Refuses any other use of a field: `@`, slices and `.columns` of a field
        are tile expressions of level 2."""
        if self.field_name(node):
            self.refuse(
                node,
                f"self.{node.attr} is a field: a line fills it from a stream and a "
                "km.cxx body takes it; `@`, slices and `.columns` of a field are "
                "level 2",
                f"write self.{node.attr}.fill(km.Stream(...)) and "
                f"km.cxx(..., takes=[self.{node.attr}])",
            )

    def moved(self, node, tensor, how):
        self.compiled.moves.append(
            Move(self.method.name, node.lineno, self.text(node), tensor, how)
        )
        mover = {"page write": "B", "fill": "A"}.get(how)
        if mover is not None:
            self.held_cast_movers(node, [mover], how)

    def wait(self, node, cpu, where):
        self.compiled.waits.append(
            Wait(self.method.name, cpu, where, node.lineno, self.text(node))
        )

    # Reduce ------------------------------------------------------------------------

    def reduce(self, node, name, call, everywhere=False):
        """`whole = km.reduce(total, form=form)`: total's n tiles summed over the
        group (by owner or by tree) at its rank 0, a slot of n tiles a sender;
        `km.reduce(self.part, ...)`: a field of sums, the same in L1. `everywhere`
        (km.all_reduce): the sum then cast to every rank of the group (km_tree.hpp)."""
        field_ = self.field_name(call.args[0]) if len(call.args) == 1 else None
        merging = any(keyword_.arg == spelling("combine") for keyword_ in call.keywords)
        if everywhere:
            return self.all_reduce(node, name, call, field_, merging)
        if name is None and not merging:
            self.refuse(
                node,
                "a reduce's sum is a value: it is assigned",
                f"write whole = km.reduce(...), or merge a state in place with "
                f"{spelling('combine')}=",
            )
        if field_ is None and merging:
            self.refuse(
                node,
                f"a reduce with {spelling('combine')}= merges a state written in "
                "place: an own field of fp32 tiles",
                f"write km.reduce(self.state, ..., {spelling('combine')}=MERGE)",
            )
        tile = self.reduced_tile(node, call, "reduce") if field_ is None else None
        form, form_uses, size, group, uses_group = self.reduce_keywords(node, call)
        combine = next(
            (k.value for k in call.keywords if k.arg == spelling("combine")), None
        )
        if combine is not None or name is None:
            return self.merge_reduce(
                node,
                name,
                field_,
                (form, form_uses),
                size,
                (group, uses_group),
                combine,
            )
        if field_ is not None:
            return self.field_reduce(
                node, name, field_, (form, form_uses), size, (group, uses_group)
            )
        return self.tile_reduce(
            node, name, call, tile, (form, form_uses), size, (group, uses_group)
        )

    def reduced_tile(self, node, call, kind):
        """The DST tile `km.<kind>(total)` sums, checked."""
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            self.refuse(
                node,
                f"{spelling(kind)} a tile, or a field of sums",
                f"write km.{spelling(kind)}(total, form=form) or "
                f"km.{spelling(kind)}(self.part, form=form)",
            )
        tile = self.scope.get(call.args[0].id)
        self.not_paired(node, call.args[0].id)
        if tile is None or tile.kind != "tile" or self.dst_live != call.args[0].id:
            self.refuse(
                node,
                f"{spelling(kind)} the tile held in DST",
                f"write km.{spelling(kind)}(total, form=form)",
            )
        if not self.carries(node, tile):
            self.same_loops(
                node,
                call.args[0].id,
                tile,
                f"this {spelling(kind)}",
                *self.once_text(tile),
            )
        return tile

    def all_reduce(self, node, name, call, field_, merging):
        """`whole = km.all_reduce(total)`: reduce's sum, cast back by rank 0 to every
        rank of the group in the same command (km_tree.hpp); `km.all_reduce(self.part)`
        of a field compute wrote: every rank's field then holds the group's sum."""
        kind = spelling("all_reduce")
        if merging:
            self.refuse(
                node,
                f"km.{kind} adds; a merge of states is km.reduce's "
                f"{spelling('combine')}=",
                f"write km.reduce(self.state, ..., {spelling('combine')}=MERGE)",
            )
        if field_ is not None and name is not None:
            self.refuse(
                node,
                f"km.{kind}(self.{field_}) leaves the sum in self.{field_} at every "
                "rank: it gives no value",
                f"write km.{kind}(self.{field_}) on a line of its own",
            )
        if field_ is None and name is None:
            self.refuse(
                node,
                f"km.{kind}'s sum is a value: it is assigned",
                f"write whole = km.{kind}(total), or km.{kind}(self.part)",
            )
        tile = self.reduced_tile(node, call, "all_reduce") if field_ is None else None
        form, form_uses, size, group, uses_group = self.reduce_keywords(node, call)
        if field_ is not None:
            return self.field_all_reduce(
                node, field_, (form, form_uses), size, (group, uses_group)
            )
        return self.tile_reduce(
            node,
            name,
            call,
            tile,
            (form, form_uses),
            size,
            (group, uses_group),
            everywhere=True,
        )

    def field_all_reduce(self, node, field_, form, size, group):
        """`km.all_reduce(self.part)` of a field compute wrote: compute adds the
        arrivals to it in DST and packs the sum for mover B's send (rank 0: its cast
        back, then freed); rank 0 packs it into its field, each other rank the sum cast
        back, which a later loop's tile function then takes at every rank."""
        form, form_uses = form
        group, uses_group = group
        kind = spelling("all_reduce")
        held = self.packed.get(field_)
        stored = self.compiled.fields[field_]
        if held is None or stored.format.dtype != "float32":
            self.refuse(
                node,
                f"km.{kind}(self.{field_}) sums a float32 field compute wrote above",
                f"write self.{field_}.write(...) before it, self.{field_} declared "
                "km.L1[km.fp32, ...]",
            )
        tiles = math.prod(stored.shape[:-2])
        if tiles + 1 > DST_HALF:
            self.refuse(
                node,
                f"self.{field_}'s {tiles} tiles and the arrivals' tile are more than "
                f"DST's {DST_HALF}",
                f"all-reduce a field of up to {DST_HALF - 1} tiles",
            )
        if form == "km::kReduceOwner":
            if size is None:
                self.refuse(
                    node,
                    f"a reduce of self.{field_} by owner holds a slot of its {tiles} "
                    "tiles for each other core of the group, and the group's size is "
                    "the command's",
                    f'give {spelling("group")}=S, or reduce by tree: form="tree"',
                )
            slots = max(1, size - 1)
        elif form == "km::kReduceTree":
            slots = self.tree_slots(size)
        else:
            self.refuse(
                node,
                f"an all-reduce of self.{field_} sizes its slots by the form",
                'write form="owner" or form="tree"',
            )
        number = self.arrivals
        self.arrivals += 1
        arrival = self.arrival_buffer(
            number, slots, tiles, f"line {node.lineno}: {self.text(node)}"
        )
        buffer = f"kField{camel(field_)}"
        out = self.out_buffer(
            "float32",
            PAGE_BYTES["float32"],
            f"line {node.lineno}: {self.text(node)}",
            tiles,
        ).name
        self.compiled.unpack_fp32 |= {arrival.name, buffer}
        self.compiled.includes |= {"port", "tree", "add", "pack"}
        self.compiled.port_handlers.add(self.method.name)
        uses = uses_group | form_uses
        moving = group.format(cpu="movement")
        computing = group.format(cpu="compute")
        movement = [
            Line(
                f"if (!km::all_reduce_arrive(movement, command, {arrival.name}, "
                f"{moving}, {form})) {{\n    return false;\n}}",
                uses,
                mover="A",
            ),
            Line(
                f"if (!km::all_reduce_emit(movement, command, {arrival.name}, "
                f"{moving}, {form}, {out}, {tiles})) {{\n    return false;\n}}\n"
                f"if (km::all_reduce_keeps({moving})) {{  // the cast's, kept by it\n"
                f"    cb_pop_front({out}, {tiles});\n}}",
                uses,
                mover="B",
            ),
        ]
        adds = "\n".join(
            add_tile(
                f"{arrival.name}.cb",
                f"km::slot_page({arrival.name}, km_slot)" + (f" + {t}" if t else ""),
                t,
                tiles,
            )
            for t in range(tiles)
        )
        adds = "\n".join("    " + line for line in adds.split("\n"))
        taken = "".join(f"    copy_tile({buffer}, {t}, {t});\n" for t in range(tiles))
        cast = "".join(
            f"    copy_tile({arrival.name}.cb, {t}, {t});\n" for t in range(tiles)
        )
        packs = "".join(f"    pack_tile({t}, {out});\n" for t in range(tiles))
        # another island's leader casts the sum on from the out buffer as well
        keeps = (
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"if (km_ok && compute.reserve_back({buffer}, {buffer}Pages)) {{\n"
            f"    KM_PACK_FORMAT({buffer});\n"
            + "".join(f"    pack_tile({t}, {buffer});\n" for t in range(tiles))
            + f"    cb_push_back({buffer}, {buffer}Pages);\n"
            "}\n"
            f"if (km_ok && km::all_reduce_keeps({computing}) &&\n"
            f"    compute.reserve_back({out}, {tiles})) {{\n"
            f"    KM_PACK_FORMAT({out});\n{packs}"
            f"    cb_push_back({out}, {tiles});\n"
            "}\n"
            "tile_regs_release();"
        )
        packed_keeps = "\n".join("    " + line for line in keeps.split("\n"))
        root_packs = "".join(f"    pack_tile({t}, {buffer});\n" for t in range(tiles))
        text = (
            f"// self.{field_} summed over the group, the sum at every rank\n"
            f"km_ok = km_ok && compute.wait_front({buffer}, {buffer}Pages);\n"
            "tile_regs_acquire();\n"
            "if (km_ok) {\n"
            f"    KM_UNPACK_FORMAT_A({buffer});\n"
            f"    copy_init({buffer});\n{taken}"
            f"    cb_pop_front({buffer}, {buffer}Pages);\n"
            "}\n"
            f"{held} = false;\n"
            f"km_ok = km_ok && km::reduce(compute, {arrival.name}, {computing}, "
            f"{form}, [&](uint32_t km_slot) {{\n{adds}\n}});\n"
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"if (km_ok && compute.reserve_back({out}, {tiles})) {{  // for mover B\n"
            f"    KM_PACK_FORMAT({out});\n{packs}"
            f"    cb_push_back({out}, {tiles});\n"
            "}\n"
            f"if (km_ok && km::reduce_send({computing}, {form}).root &&\n"
            f"    compute.reserve_back({buffer}, {buffer}Pages)) {{  // rank 0's\n"
            f"    KM_PACK_FORMAT({buffer});\n{root_packs}"
            f"    cb_push_back({buffer}, {buffer}Pages);\n"
            "}\n"
            "tile_regs_release();\n"
            f"km_ok = km_ok && km::all_reduce_take(compute, {arrival.name}, "
            f"{computing}, [&] {{\n"
            "    tile_regs_acquire();\n"
            f"    KM_UNPACK_FORMAT_A({arrival.name}.cb);\n"
            f"    copy_init({arrival.name}.cb);\n{cast}{packed_keeps}\n"
            "});\n"
            f"{held} = km_ok;"
        )
        self.compute_waits = True
        self.wait(node, "mover A", WAIT_PORT)
        self.wait(node, "mover B", WAIT_EMIT)
        self.wait(node, "compute", WAIT_INPUT)
        return movement, [Line(text, uses | {"km_ok"})]

    def reduce_keywords(self, node, call):
        """A reduce's form= and group=: (form, its uses, size or None, the group's
        C++ text, its uses); group= sets the leader to each group's rank 0."""
        form, size = None, None
        for keyword_ in call.keywords:
            if keyword_.arg == spelling("combine"):
                continue
            if keyword_.arg == spelling("group"):
                size = self.constant_int(keyword_.value)
                if size < 1:
                    self.refuse(
                        node,
                        "a reduce's groups are of 1 rank or more",
                        "give group= a size",
                    )
                continue
            if keyword_.arg != spelling("form"):
                self.refuse(
                    node,
                    f"a reduce has no {keyword_.arg}=",
                    f"use form=, group= and {spelling('combine')}=",
                )
            value = keyword_.value
            if isinstance(value, ast.Constant) and value.value in ("owner", "tree"):
                form = (
                    "km::kReduceOwner" if value.value == "owner" else "km::kReduceTree"
                )
                form_uses = set()
            else:
                form = self.integer(value)
                form_uses = self.uses(value)
        if form is None:
            form, form_uses = "km::kReduceTree", set()
        group = "km_group"
        uses_group = {"km_group"}
        if size is not None:  # consecutive ranks in groups of `size`, rank 0 owns
            if self.method.name in self.compiled.reduce_groups:
                self.refuse(
                    node,
                    "a handler reduces in groups once",
                    "reduce the handler's sums in one line",
                )
            self.compiled.reduce_groups[self.method.name] = (
                size,
                f"line {node.lineno}, `{self.text(node)}`",
            )
            group = f"km::digit_group(command, {{cpu}}.me(), 1, {size})"
            uses_group = set()
            self.leader_text = f"(km_rank - km_rank % {size})"
        return form, form_uses, size, group, uses_group

    def tile_reduce(self, node, name, call, tile, form, size, group, everywhere=False):
        """`whole = km.reduce(total)` of DST tiles (reduce's); `everywhere`: rank 0
        casts the sum back and every other rank packs it into the same buffer."""
        form, form_uses = form
        group, uses_group = group
        width = tile.width
        slots = REDUCE_SLOTS  # by owner, or a form the command chooses
        if width > 1 and form != "km::kReduceTree":
            self.refuse(
                node,
                f"a reduce of {width} tiles by owner holds a slot of them for each "
                "other core of the group",
                'reduce them by tree: form="tree"',
            )
        if form == "km::kReduceTree":  # a tree's: one slot a round
            slots = self.tree_slots(size)
        if tile.dst + width + 1 > DST_FULL:  # the sums and a tile for the arrivals
            self.refuse(
                node,
                f"{call.args[0].id}'s {width} tiles and the arrivals' tile are more "
                f"than DST's {DST_FULL}",
                f"reduce up to {DST_FULL - 1 - tile.dst} tiles",
            )
        if tile.dst + width + 1 > DST_HALF:
            self.compiled.full_dst = True
        number = self.arrivals
        self.arrivals += 1
        arrival = self.arrival_buffer(
            number, slots, width, f"line {node.lineno}: {self.text(node)}"
        )
        self.compiled.unpack_fp32.add(arrival.name)
        out = self.out_buffer(
            "float32",
            PAGE_BYTES["float32"],
            f"line {node.lineno}: {self.text(node)}",
            width,
        )
        self.compiled.includes |= {"port", "add", "pack"}
        self.compiled.port_handlers.add(self.method.name)
        uses = uses_group | form_uses
        moving = group.format(cpu="movement")
        call_ = "km::all_reduce" if everywhere else "km::reduce"
        movement = [
            Line(
                f"if (!{call_}_arrive(movement, command, {arrival.name}, {moving}, "
                f"{form})) {{\n    return false;\n}}",
                uses,
                mover="A",
            ),
            Line(
                f"if (!{call_}_emit(movement, command, {arrival.name}, {moving}, "
                f"{form}, {out.name}, {width})) {{\n    return false;\n}}",
                uses,
                mover="B",
            ),
        ]
        if everywhere:
            self.compiled.includes.add("tree")
        combine = "\n".join(
            add_tile(
                f"{arrival.name}.cb",
                f"km::slot_page({arrival.name}, km_slot)" + (f" + {t}" if t else ""),
                tile.dst + t,
                tile.dst + width,
            )
            for t in range(width)
        )
        body = "\n".join("    " + line for line in combine.split("\n"))
        compute = [
            Line(
                f"km_ok = km_ok && km::reduce(compute, {arrival.name}, "
                f"{group.format(cpu='compute')}, {form}, "
                f"[&](uint32_t km_slot) {{\n{body}\n}});",
                uses | {"km_ok"},
            ),
            Line(pack(tile.dst, out.name, width), {"km_ok"}),
        ]
        if everywhere:  # the sum cast back: slot 0's tiles packed as rank 0's were
            copies = "".join(
                f"    copy_tile({arrival.name}.cb, {t}, {tile.dst + t});\n"
                for t in range(width)
            )
            cast = (
                "tile_regs_acquire();\n"
                f"KM_UNPACK_FORMAT_A({arrival.name}.cb);\n"
                f"copy_init({arrival.name}.cb);\n{copies}"
                + pack(tile.dst, out.name, width)
            )
            inner = "\n".join("    " + line for line in cast.split("\n"))
            compute.append(
                Line(
                    f"km_ok = km_ok && km::all_reduce_take(compute, {arrival.name}, "
                    f"{group.format(cpu='compute')}, [&] {{\n{inner}\n}});",
                    uses | {"km_ok"},
                )
            )
        self.compute_waits = True
        if tile.cxx in self.carried and self.dst_live == tile.cxx:
            compute = [Line(self.carried_reduce(node, tile), {"km_ok"}), *compute]
            self.scope[name] = Value(
                "reduced",
                name,
                buffer=out,
                width=width,
                made_at=(len(self.loops), node.lineno),
            )
            if not everywhere:  # its sum is the leader's alone
                self.reduced.append(name)
            self.wait(node, "mover A", WAIT_PORT)
            self.wait(node, "mover B", WAIT_EMIT)
            self.wait(node, "compute", WAIT_INPUT)
            return movement, compute
        self.dst_live = None
        tile.kind = "written"
        self.scope[name] = Value(
            "reduced",
            name,
            buffer=out,
            width=width,
            made_at=(len(self.loops), node.lineno),
        )
        if not everywhere:
            self.reduced.append(name)
        self.wait(node, "mover A", WAIT_PORT)
        self.wait(node, "mover B", WAIT_EMIT)
        self.wait(node, "compute", WAIT_INPUT)
        return movement, compute

    def merge_reduce(self, node, name, field_, form, size, group, combine):
        """`km.reduce(self.state, form="tree", group=S, combine=MERGE)`: an own fp32
        field written in place, merged up the group: the MERGE body on compute's three
        for each arrival slot (`arrival_buffer`, `arrival_index`, the field by name); a
        sender pushes the state for mover B's send, the root's stays in place."""
        combine_ = spelling("combine")
        if combine is None:
            self.refuse(
                node,
                "a reduce's sum is a value: it is assigned",
                f"write whole = km.reduce(...), or merge a state in place with "
                f"{combine_}=",
            )
        if name is not None:
            self.refuse(
                node,
                f"a reduce with {combine_}= leaves the merged state in place, at the "
                "group's rank 0: it gives no value",
                f"write km.reduce(self.state, ..., {combine_}=MERGE) on its own line",
            )
        own = f"km.{spelling('own')}(push=None)"
        stored = self.compiled.fields.get(field_) if field_ else None
        if (
            stored is None
            or field_ not in self.compiled.own
            or self.compiled.own[field_] is not None
        ):
            self.refuse(
                node,
                f"{combine_}= merges a state written in place: an own field of fp32 "
                "tiles that only bodies write",
                f"declare it state: km.L1[km.fp32, n, 32, 32] = {own}",
            )
        if stored.format.dtype != "float32":
            self.refuse(
                node,
                f"self.{field_} is {stored.format.dtype}: a merged state is fp32 "
                "tiles, as the arrival's slots are",
                f"declare self.{field_}: km.L1[km.fp32, ...] = {own}",
            )
        tiles = 1
        for side in stored.shape[:-2]:
            tiles *= side
        merge = tokened(  # a token between pack and unpack, or refused here
            self,
            node,
            field_,
            textwrap.dedent(
                self.body_text(node, ast.Call(func=None, args=[combine], keywords=[]))
            ).strip("\n"),
        )
        form, form_uses = form
        group, uses_group = group
        if form == "km::kReduceOwner":
            if size is None:
                self.refuse(
                    node,
                    f"a merge by owner holds a slot of self.{field_}'s {tiles} tiles "
                    "for each other core of the group, and the group's size is the "
                    "command's",
                    f'give {spelling("group")}=S, or merge by tree: form="tree"',
                )
            slots = max(1, size - 1)
        elif form == "km::kReduceTree":
            slots = self.tree_slots(size)
        else:
            self.refuse(
                node,
                f"a merge of self.{field_} sizes its slots by the form",
                'write form="owner" or form="tree"',
            )
        number = self.arrivals
        self.arrivals += 1
        arrival = self.arrival_buffer(
            number, slots, tiles, f"line {node.lineno}: {self.text(node)}"
        )
        self.compiled.unpack_fp32.add(arrival.name)
        self.compiled.unpack_fp32.add(f"kField{camel(field_)}")
        buffer = f"kField{camel(field_)}"
        self.compiled.includes |= {"port", "pack"}
        self.compiled.port_handlers.add(self.method.name)
        uses = uses_group | form_uses
        moving = group.format(cpu="movement")
        movement = [
            Line(
                f"if (!km::reduce_arrive(movement, command, {arrival.name}, {moving}, "
                f"{form})) {{\n    return false;\n}}",
                uses,
                mover="A",
            ),
            Line(
                f"if (!km::reduce_emit(movement, command, {arrival.name}, {moving}, "
                f"{form}, {buffer}, {buffer}Pages)) {{\n    return false;\n}}",
                uses,
                mover="B",
            ),
        ]
        start = combine.lineno + 1 if isinstance(combine, ast.Constant) else node.lineno
        body = (
            f"const uint32_t arrival{spelling('tile_buffer')} = {arrival.name}.cb;\n"
            f"const uint32_t arrival{spelling('tile_index')} = "
            f"km::slot_page({arrival.name}, km_slot);\n"
            f"const uint32_t {field_} = {buffer};\n"
            f'#line {start} "{self.compiled.source}"\n{merge}\n#line RESUME'
        )
        inner = "\n".join("    " + line for line in body.split("\n"))
        compute_group = group.format(cpu="compute")
        compute = [
            Line(
                f"// self.{field_} merged with the group's: each arrival by the "
                f"{combine_} body\n"
                f"km_ok = km_ok && km::reduce(compute, {arrival.name}, "
                f"{compute_group}, {form}, [&](uint32_t km_slot) {{\n{inner}\n}});\n"
                f"if (km_ok && !km::reduce_send({compute_group}, {form}).root) {{  "
                "// to mover B's send\n"
                f"    const bool km_room_{field_} = compute.reserve_back({buffer}, "
                f"{buffer}Pages);\n"
                f"    if (km_room_{field_}) {{\n"
                f"        cb_push_back({buffer}, {buffer}Pages);\n"
                "    }\n"
                "}",
                uses | {"km_ok"},
            ),
        ]
        self.compute_waits = True
        self.wait(node, "mover A", WAIT_PORT)
        self.wait(node, "mover B", WAIT_EMIT)
        self.wait(node, "compute", WAIT_INPUT)
        return movement, compute

    def field_reduce(self, node, name, field_, form, size, group):
        """`km.reduce(self.part, form=form, group=S)` of a field of sums, added at the
        group's rank 0 in L1: mover A takes the arrivals (a slot a sender or a round),
        compute packs each onto the field with L1 accumulation, mover B sends the
        field on. Its size: the field and the slots in L1, not DST."""
        form, form_uses = form
        group, uses_group = group
        sums = self.summed.get(field_)
        if sums is None:
            self.refuse(
                node,
                f"self.{field_} holds no sums above in this handler",
                f"add products into it first: self.{field_} += self.f(a, b)",
            )
        rows, columns = self.field_sides(node, field_)
        tiles = rows * columns
        if form == "km::kReduceOwner":
            if size is None:
                self.refuse(
                    node,
                    f"a reduce of self.{field_} by owner holds a slot of its {tiles} "
                    "tiles for each other core of the group, and the group's size is "
                    "the command's",
                    f'give {spelling("group")}=S, or reduce by tree: form="tree"',
                )
            slots = max(1, size - 1)
        elif form == "km::kReduceTree":
            slots = self.tree_slots(size)
        else:
            self.refuse(
                node,
                f"a reduce of self.{field_} sizes its slots by the form",
                'write form="owner" or form="tree"',
            )
        number = self.arrivals
        self.arrivals += 1
        arrival = self.arrival_buffer(
            number, slots, tiles, f"line {node.lineno}: {self.text(node)}"
        )
        self.compiled.unpack_fp32.add(arrival.name)
        buffer = f"kField{camel(field_)}"
        self.compiled.includes |= {"port", "pack"}
        self.compiled.port_handlers.add(self.method.name)
        uses = uses_group | form_uses
        moving = group.format(cpu="movement")
        movement = [
            Line(
                f"if (!km::reduce_arrive(movement, command, {arrival.name}, {moving}, "
                f"{form})) {{\n    return false;\n}}",
                uses,
                mover="A",
            ),
            Line(
                f"if (!km::reduce_emit(movement, command, {arrival.name}, {moving}, "
                f"{form}, {buffer}, {buffer}Pages)) {{\n    return false;\n}}",
                uses,
                mover="B",
            ),
        ]
        tile = f"km_tile{node.lineno}"
        room = sums_room(field_)
        combine = (
            f"KM_UNPACK_FORMAT_A({arrival.name}.cb);\n"
            f"copy_init({arrival.name}.cb);\n"
            f"KM_PACK_FORMAT({buffer});\n"
            f"if ({room}) {{\n"
            f"    pack_reconfig_l1_acc({sums} > 0 ? 1 : 0);\n"
            "}\n"
            f"for (uint32_t {tile} = 0; {tile} < {tiles}; {tile}++) {{\n"
            "    tile_regs_acquire();\n"
            f"    copy_tile({arrival.name}.cb, km::slot_page({arrival.name}, "
            f"km_slot) + {tile}, 0);\n"
            "    tile_regs_commit();\n"
            "    tile_regs_wait();\n"
            f"    if ({room}) {{\n"
            f"        pack_tile<true>(0, {buffer}, {tile});\n"
            "    }\n"
            "    tile_regs_release();\n"
            "}\n"
            f"if ({room}) {{\n"
            "    pack_reconfig_l1_acc(0);\n"
            "}\n"
            f"{sums}++;"
        )
        body = "\n".join("    " + line for line in combine.split("\n"))
        compute = [
            Line(
                f"// self.{field_}'s sums and the group's: each arrival added into it\n"
                f"{sums_reserve(sums, room, buffer)}"
                f"km_ok = km_ok && km::reduce(compute, {arrival.name}, "
                f"{group.format(cpu='compute')}, {form}, "
                f"[&](uint32_t km_slot) {{\n{body}\n}});\n"
                f"if (km_ok && {room}) {{  // to mover B's send, or to the leader's "
                "write\n"
                f"    cb_push_back({buffer}, {buffer}Pages);\n"
                "}\n"
                + sent_field_acked(buffer, group.format(cpu="compute"), form)
                + f"{sums} = 0;",
                uses | {"km_ok"},
            ),
        ]
        self.compute_waits = True
        self.scope[name] = Value(
            "reduced",
            name,
            buffer=None,
            width=tiles,
            summed=field_,
            made_at=(len(self.loops), node.lineno),
        )
        self.reduced.append(name)
        self.wait(node, "mover A", WAIT_PORT)
        self.wait(node, "mover B", WAIT_EMIT)
        self.wait(node, "compute", WAIT_INPUT)
        return movement, compute

    def take_handed(self, node, name, value, where, bound, before, after):
        """A word one mover's body gave, taken by a body on the other mover: the
        giver writes the body's words into a page of the hand-over buffer and pushes
        it there; the taker waits for the page, reads them, and frees it after."""
        parts = self.handed.setdefault(id(node), {"words": []})
        words = parts["words"]
        if len(words) == HANDOVER_WORDS:
            self.refuse(
                node,
                f"a hand-over between the movers holds {HANDOVER_WORDS} words a body, "
                f"and this body takes {name} after {len(words)}",
                f"take at most {HANDOVER_WORDS} words of the other mover in one body",
            )
        if words and parts["giver"] != value.mover:
            self.refuse(node, "a body's hand-over is from one mover", "take one's")
        words.append(name)
        buffer = self.other_buffer(
            "kHandover",
            "word",
            "uint32",
            4 * HANDOVER_WORDS,
            1,
            f"line {node.lineno}: a hand-over of mover {value.mover}'s words",
        ).name
        page = f"km_handed{node.lineno}"
        stores = "".join(
            f"\nkm::l1({page})[{n}] = {word};" for n, word in enumerate(words)
        )
        giving = (
            f"if (!movement.reserve({buffer}, 1)) {{\n    return false;\n}}\n"
            f"const uint32_t {page} = get_write_ptr({buffer});{stores}\n"
            f"cb_push_back({buffer}, 1);"
        )
        if len(words) == 1:
            parts["giver"] = value.mover
            parts["line"] = Line(giving, set(words), mover=value.mover)
            before.append(parts["line"])
            before.append(
                Line(
                    f"if (!movement.wait_front({buffer}, 1)) {{\n    return false;\n}}"
                    f"\ninvalidate_l1_cache();\nconst uint32_t {page} = "
                    f"get_read_ptr({buffer});",
                    mover=where,
                )
            )
            after.append(Line(f"cb_pop_front({buffer}, 1);", mover=where))
        else:
            parts["line"].text = giving
            parts["line"].uses |= {name}
        bound.append(f"const uint32_t {name} = km::l1({page})[{len(words) - 1}];")
        self.compiled.handovers.append(
            f"{self.method.name}, line {node.lineno}: {name} from mover {value.mover} "
            f"to mover {where} by the hand-over buffer"
        )
        return set()

    def tree_slots(self, size):
        """A tree reduce's arrival slots: a round each over a group of `size`; the
        command's group (None): TREE_ROUNDS, and over islands the root part's rounds
        and a slot each other part (km::reduce_slots), whichever is more."""
        if size is not None:
            return max(1, (size - 1).bit_length())
        parts, most = self.compiled.islands
        if parts <= 1:
            return TREE_ROUNDS
        return max(TREE_ROUNDS, max(1, (most - 1).bit_length()) + parts - 1)

    def arrival_buffer(self, number, slots, per, what):
        """kArrival<number>, which the handlers' reduces of that number share: a slot
        of the largest part (`per` pages) and as many slots as the most a reduce takes.
        """
        name = f"kArrival{number}"
        found = next((b for b in self.compiled.buffers if b.name == name), None)
        if found is not None:
            per, slots = max(per, found.block), max(
                slots, found.pages[0] // found.block
            )
            found.block, found.pages = per, (slots * per,)
            return found
        arrival = self.other_buffer(
            name, "arrival", "float32", PAGE_BYTES["float32"], slots * per, what
        )
        arrival.block = per
        return arrival

    def other_buffer(self, name, role, dtype, page_bytes, pages, what):
        for buffer in self.compiled.buffers:
            if buffer.name == name:
                return buffer
        taken = {b.index_a for b in self.compiled.buffers} | {
            b.index_b for b in self.compiled.buffers
        }
        taken |= self.compiled.reserved
        index = FIRST_OTHER
        while index in taken:
            index += 1
        if index >= FIRST_WRITE and role != "field":
            raise Refused(f"{self.compiled.name}: more buffers than the indices")
        buffer = Buffer(name, role, index, None, dtype, page_bytes, (pages,), 1, what)
        self.compiled.buffers.append(buffer)
        return buffer

    # Broadcasts --------------------------------------------------------------------

    def broadcast(self, node, name, call):
        """`b = km.Broadcast(self, along="row")`: the broadcast on its mover (a row on
        mover A, a column on mover B); a core it leaves out skips the handler."""
        if self.method.overlay:
            self.refuse(
                node,
                "a broadcast's handler is resident: its included test ends the "
                "command in the kernel",
                f"drop {spelling('overlay')}=True from this handler",
            )
        if node not in self.definition.body or self.dst_live is not None:
            self.refuse(
                node,
                "a broadcast is made at the top of the handler",
                "make it before the tiles and loops",
            )
        if not (
            len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "self"
        ):
            self.refuse(
                node,
                "a broadcast of this core",
                f'write km.{spelling("broadcast")}(self, along="row")',
            )
        along, window, placed = None, BROADCAST_WINDOW, {}
        for keyword_ in call.keywords:
            if keyword_.arg == spelling("along"):
                value = keyword_.value
                if not (isinstance(value, ast.Constant) and value.value in ALONG):
                    self.refuse(
                        node,
                        "along= is a row or a column",
                        'write along="row" or along="column"',
                    )
                along = value.value
            elif keyword_.arg == spelling("window"):
                window = self.constant_int(keyword_.value)
            elif keyword_.arg in (
                spelling("broadcast_mover"),
                spelling("broadcast_length"),
                spelling("broadcast_unread"),
            ):
                placed[keyword_.arg] = keyword_.value
            else:
                self.refuse(
                    node,
                    f"a broadcast has no {keyword_.arg}=",
                    "use along=, window=, mover=, length= and unread=",
                )
        if along is None:
            self.refuse(
                node,
                "a broadcast goes along a row or a column",
                'write along="row" or along="column"',
            )
        mover, length = self.broadcast_place(node, along, placed)
        if mover in self.movers:
            self.refuse(
                node,
                f"{self.movers[mover]} already runs on mover {mover}; each mover runs "
                "one broadcast (rows on mover A, columns on mover B unless mover= "
                "says)",
                "make one broadcast a mover: give the other mover=",
            )
        self.movers[mover] = name
        index = self.class_mover(node, mover)
        self.compiled.broadcast_lines.setdefault(
            self.method.name, f"line {node.lineno}, `{self.text(node)}`"
        )
        self.compiled.rectangle = True
        self.compiled.rect_plain |= self.rect_param is None
        self.compiled.includes |= {"broadcast"}
        self.compute_waits = True
        self.scope[name] = Value(
            "broadcast", name, mover=mover, window=window, pattern=along, length=length
        )
        self.scope[name].index = index  # its arguments' place among its mover's
        unread = placed.get(spelling("broadcast_unread"))
        if unread is not None:  # a word of the handler: its C++, the names it reads
            self.scope[name].unread = (self.integer(unread), self.uses(unread))
        arg = "kArgBroadcast" + (f" + km::kBroadcastArgs * {index}" if index else "")
        if self.rect_param is not None:  # its arguments computed on the core
            arg = self.rect_broadcast(name, along, mover, window, length, index)
        movement = Line(
            f"km::Broadcast {name}(movement, command, {arg});\n"
            f"if (!{name}.ok()) {{\n    return false;\n}}\n"
            f"if (!{name}.included()) {{\n    return true;\n}}",
            {"km_grid", self.rect_param} if self.rect_param else set(),
            mover=mover,
        )
        counter = f"km_blocks_{name}"  # kept where casts count over loops
        counted = [
            Glue(f"uint32_t {counter} = 0;", {counter}, mover=mover),
            Glue(f"uint32_t {counter} = 0;", {counter}),
        ]
        compute = []
        if len(self.movers) == 1:
            test, uses = self.included_test("compute")
            compute.append(Line(f"if ({test}) {{\n    break;\n}}", uses))
            return [movement, counted[0], self.other_mover_skips(mover)], [
                *compute,
                counted[1],
            ]
        return [movement, counted[0]], [counted[1]]

    def rect_broadcast(self, name, along, mover, window, length, index):
        """A broadcast's arguments on a rectangle given at each submit: the build's
        (its number, NoC, buffer, window, pages), the command's group, and rank,
        receivers, sender and corners computed on the core (km::rect_broadcast)."""
        args, place = f"km_args_{name}", f"{mover}{index}"
        self.compiled.rect_broadcasts.add((mover, index))
        return (
            f"[&]() {{\n"
            f"    km::BroadcastArgs {args}{{}};\n"
            f"    {args}.broadcast = kRectId{place};\n"
            f"    {args}.group = command.group;\n"
            f"    {args}.noc = {0 if mover == 'A' else 1}u;\n"
            f"    {args}.cb = kRectCb{place};\n"
            f"    {args}.window = kRectWindow{place};\n"
            f"    {args}.pages = kRectPages{place};\n"
            f"    {args}.next = km::kBroadcastNoNext;\n"
            f"    km::rect_broadcast({args}, movement.me(), km_grid, "
            f"{self.rect_param}, {'true' if along == 'row' else 'false'}, "
            f"{length or 0}u, km::kBroadcastNone);\n"
            f"    return {args};\n"
            f"}}()"
        )

    def class_mover(self, node, mover):
        """The place of this handler's broadcast among its mover's in the class: a
        core holds each one's arguments after the last's. A core has KM_BROADCASTS
        broadcast areas (a broadcast's number picks one): a class past them is
        refused, naming the broadcasts it has."""
        line = f"line {node.lineno}, `{self.text(node)}`"
        held = self.compiled.mover_broadcasts.setdefault(mover, [])
        for index, (handler, _) in enumerate(held):
            if handler == self.method.name:  # the handler made again (a variant)
                return index
        areas = abi.headers()["KM_BROADCASTS"]
        others = [b for each in self.compiled.mover_broadcasts.values() for b in each]
        if len(others) >= areas:
            named = "; ".join(f"{h}, {at}" for h, at in others)
            self.refuse(
                node,
                f"the class runs {len(others)} broadcasts already ({named}); a core "
                f"has {areas} broadcast areas, one a broadcast of the class",
                "make this handler's broadcast in a class of its own",
            )
        held.append((self.method.name, line))
        return len(held) - 1

    def broadcast_place(self, node, along, placed):
        """A broadcast's mover and band: `mover="A"` (NoC 0: a column's sender at its
        top, a row's at its left) or `"B"` (NoC 1: bottom, right), by default its
        direction's (rows A, columns B); `length=L` bands of L cores, each with its
        sender (an int the build knows), by default the rectangle's whole side."""
        mover, length = ALONG[along], None
        given = placed.get(spelling("broadcast_mover"))
        if given is not None:
            if not (isinstance(given, ast.Constant) and given.value in ("A", "B")):
                self.refuse(
                    node,
                    'a broadcast\'s mover is "A" (NoC 0) or "B" (NoC 1)',
                    'write mover="A" or mover="B"',
                )
            mover = given.value
        given = placed.get(spelling("broadcast_length"))
        if given is not None:
            length = self.known_int(given)
            if length is None or length < 1:
                self.refuse(
                    node,
                    "a broadcast's length is cores a band, an int the build knows "
                    "(a constant or a plan value) of 1 or more",
                    "write length=L, L a module constant or a plan value",
                )
        return mover, length

    def other_mover_skips(self, mover):
        """The mover a handler's first broadcast does not run skips the handler on a
        core the rectangle leaves out, as that broadcast's mover and compute do."""
        test, uses = self.included_test("movement")
        return Line(
            f"if ({test}) {{\n    return true;\n}}",
            uses,
            mover="B" if mover == "A" else "A",
        )

    def broadcast_branch(self, node):
        """`if b.sender(): v = b.cast(pages.read())`, `else: v = b.arrive()`: on the
        sender the block read into its slot and cast, on the others its arrival."""
        test = node.test
        if not (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Attribute)
            and test.func.attr == spelling("sender")
            and isinstance(test.func.value, ast.Name)
            and self.scope.get(test.func.value.id, Value("")).kind == "broadcast"
        ):
            return None
        name = test.func.value.id
        broadcast = self.scope[name]
        drawn = (
            f"write `if {name}.{spelling('sender')}(): v = {name}.{spelling('cast')}"
            f"(t.pages(first, n).{spelling('read')}())`, `else: v = {name}."
            f"{spelling('arrive')}()`"
        )
        cast, arrive = (
            branch[0] if len(branch) == 1 else None
            for branch in (node.body, node.orelse)
        )
        if not (
            isinstance(cast, ast.Assign)
            and isinstance(arrive, ast.Assign)
            and len(cast.targets) == len(arrive.targets) == 1
            and isinstance(cast.targets[0], ast.Name)
            and isinstance(arrive.targets[0], ast.Name)
            and cast.targets[0].id == arrive.targets[0].id
            and self.is_cast(cast.value, name)
            and self.is_call_of(arrive.value, name, "arrive", 0)
        ):
            self.refuse(node, "the sender casts a block, the others take it", drawn)
        value = cast.targets[0].id
        self.check_name(cast, value)
        parts, count, dtype, page_bytes = self.cast_parts(cast, drawn)
        buffer = broadcast.buffer
        if buffer is None:
            number = len(self.compiled.broadcasts)
            buffer = self.other_buffer(
                f"kBroadcast{number}",
                "broadcast",
                dtype,
                page_bytes,
                broadcast.window * count,
                f"line {node.lineno}: {self.text(node)}",
            )
            buffer.block = count
            broadcast.buffer = buffer
            self.compiled.broadcasts.append(
                (
                    broadcast.pattern,
                    buffer,
                    broadcast.window,
                    broadcast.mover,
                    broadcast.length,
                    broadcast.index,
                )
            )
            if dtype == "float32":
                self.compiled.unpack_fp32.add(buffer.name)
        elif (buffer.dtype, buffer.block) != (dtype, count):
            self.refuse(
                cast,
                f"{name} casts blocks of {buffer.block} {buffer.dtype} "
                "pages already",
                "cast the same pages each time",
            )
        counted = self.counted_cast(node, name, broadcast)  # None: the one loop's
        if name in self.cast_flat:
            self.refuse(
                node,
                f"{name} casts outside a loop already: its one block was finished "
                "there, and another cast would be block 0 again",
                "cast once outside a loop, or every time in one loop",
            )
        if counted is not None:
            block = counted
        elif self.loops:
            loop = self.loops[-1]
            if loop["first"] != "0":
                self.refuse(
                    node,
                    "a broadcast's blocks count from 0",
                    f"write for {loop['name']} in range(n)",
                )
            if broadcast.loop is not None and broadcast.loop is not loop:
                self.refuse(node, f"{name} casts in two loops", "cast in one loop")
            broadcast.loop = loop
            if name not in loop["casts"]:
                loop["casts"].append(name)
            block = loop["name"]
        elif broadcast.loop is not None:
            self.refuse(node, f"{name} casts in a loop already", "cast in one loop")
        else:
            block = "0"
            self.cast_flat.add(name)
        address = f"km_address{buffer.index_a}"
        read_text = self.cast_reads(parts, address, page_bytes)
        text = (
            f"if ({name}.sender()) {{\n"
            f"    const uint32_t {address} = {name}.reserve({block});\n"
            f"    if ({address} == 0 || !{read_text} || "
            f"!{name}.cast({block})) {{\n"
            "        return false;\n"
            "    }\n"
            f"}} else if (!{name}.arrive({block})) {{\n"
            "    return false;\n"
            "}"
        )
        uses = {name} | text_uses(block, self)
        for part in parts:
            uses |= range_uses(part["pages"], self)
        if broadcast.unread is not None:  # each core's slot pushed as it is, or cast
            text, uses = self.unread_push(broadcast, text, uses)
        tensors = ",".join(sorted({part["pages"].tensor for part in parts}))
        self.scope[value] = Value(
            "arrived",
            buffer.name,
            buffer=buffer,
            tensor=tensors,
            parts=tuple(slot_part(part) for part in parts),
            made_at=(len(self.loops), node.lineno),
        )
        self.scope[value].held = self.taken_by_field_products(
            value
        ) or self.taken_in_a_loop(node, value)
        self.scope[value].sides = tuple(  # each part's rectangle (rows, columns)
            part["pages"].sides or ("1", part["pages"].count) for part in parts
        )
        self.wait(node, f"mover {broadcast.mover}", WAIT_FREE)
        self.wait(node, f"mover {broadcast.mover}", WAIT_DONE)
        self.wait(node, f"mover {broadcast.mover}", WAIT_VALID)
        movement = [Line(text, uses, mover=broadcast.mover)]
        compute = []
        if counted is not None:  # the next block's number, on its mover and compute
            movement.append(Line(f"{counted}++;", {counted}, mover=broadcast.mover))
            compute.append(Line(f"{counted}++;", {counted}))
        if self.scope[value].held:  # products take it: waited here, freed at the end
            w = buffer.name
            self.wait(node, "compute", WAIT_INPUT)
            compute.append(
                Line(
                    f"km_ok = km_ok && km::broadcast_receive(compute, {w}, {w}Pages);",
                    {"km_ok"},
                )
            )
            self.held_blocks.setdefault(self.blocks[-1], []).append((value, w))
        if not self.loops:
            self.finished_casts[value] = broadcast.mover
            movement += self.end_outside_loop(name)
        return movement, compute

    def end_outside_loop(self, name):
        """The ends of a cast outside any loop (one block): the mover's finish, which
        stays at the cast, and compute's padding pop, which goes to compute's end
        (`closing`) after the block's last use: the finish pushes the padding only once
        compute popped the block, and the pop waits for it."""
        movement_end, compute_end = self.broadcast_end(name, "1", set())
        self.closing += compute_end
        return movement_end

    def unread_push(self, broadcast, text, uses):
        """`unread=word`: when the word is not 0, each core's mover takes the block's
        slot and pushes it as it is (no read, no cast, no arrival); else the cast."""
        word, reads = broadcast.unread
        buffer = broadcast.buffer.name
        text = (
            f"if ({word}) {{\n"
            f"    if (!movement.reserve({buffer}, {buffer}Pages)) {{\n"
            "        return false;\n"
            "    }\n"
            f"    cb_push_back({buffer}, {buffer}Pages);\n"
            f"}} else {text}"
        )
        return text, uses | reads

    def unread_end(self, broadcast, blocks, uses, finish):
        """After the last block of a broadcast pushed unread: its mover pads the slots
        to the window, as the finish does; else the finish."""
        word, reads = broadcast.unread
        buffer = broadcast.buffer.name
        pad = f"km_pad_{broadcast.cxx}"
        text = (
            f"if ({word}) {{\n"
            f"    const uint32_t {pad} = ({buffer}Window - ({blocks}) % "
            f"{buffer}Window) % {buffer}Window * {buffer}Pages;\n"
            f"    if ({pad} != 0) {{\n"
            f"        if (!movement.reserve({buffer}, {pad})) {{\n"
            "            return false;\n"
            "        }\n"
            f"        cb_push_back({buffer}, {pad});\n"
            "    }\n"
            f"}} else {finish.text}"
        )
        return Line(text, finish.uses | uses | reads, mover=broadcast.mover)

    def counted_cast(self, node, name, broadcast):
        """A cast inside more than one loop, or a loop not from 0: its blocks counted
        over every loop around it by a word of its mover and of compute, the finish
        after the outermost loop with the count (None: one loop from 0 counts them)."""
        if not self.loops or len(self.loops) == 1 and self.loops[0]["first"] == "0":
            return None
        outer = self.loops[0]
        if broadcast.loop is not None and broadcast.loop is not outer:
            self.refuse(
                node,
                f"{name} casts in two loops of the handler's top",
                "cast in one loop, at any depth",
            )
        broadcast.loop = outer
        counted = outer.setdefault("counted", [])
        if name not in counted:
            counted.append(name)
        return f"km_blocks_{name}"

    def counted_ends(self, loop):
        """After the outermost loop around counted casts: each broadcast's finish and
        compute's padding, by the blocks counted."""
        movement, compute = [], []
        for name in loop.get("counted", ()):
            counter = f"km_blocks_{name}"
            m, c = self.broadcast_end(name, counter, {counter})
            movement += m
            compute += c
        return movement, compute

    def is_call_of(self, node, name, kind, arguments):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == spelling(kind)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
            and len(node.args) == arguments
            and not node.keywords
        )

    def is_cast(self, node, name):
        """`name.cast(part, ...)`: one part or more, no keywords."""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == spelling("cast")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
            and len(node.args) >= 1
            and not node.keywords
        )

    def cast_parts(self, cast, drawn):
        """A cast's parts in the slot in order, each `t.pages(first, n)` (n an int),
        `t.tiles[...]` (each side's most known) or `t.tiles[...].T` (its columns in
        turn), `.read()` or not: the parts (range, slot pages, pitch, first slot
        page, transposed), the block's pages, its format and page size."""
        parts, count, formats = [], 0, set()
        for read in cast.value.args:
            source = read
            if (
                isinstance(read, ast.Call)
                and isinstance(read.func, ast.Attribute)
                and read.func.attr == spelling("read")
                and not read.args
                and not read.keywords
            ):
                source = read.func.value
            transposed = (
                isinstance(source, ast.Attribute)
                and source.attr == spelling("transposed")
                and isinstance(source.value, ast.Subscript)
                and self.is_tiles(source.value.value)
            )
            if transposed:
                source = source.value
            if not (
                isinstance(source, ast.Call)
                and self.is_range(source)
                and len(source.args) == 2
                or isinstance(source, ast.Subscript)
                and self.is_tiles(source.value)
            ):
                self.refuse(cast, "a cast sends pages read from a tensor", drawn)
            pages = self.range_of(source)
            self.pages_only(cast, pages)
            pitch = None  # a row's pages in the slot, where the columns are a word
            if isinstance(source, ast.Subscript):
                if pages.count.isdigit():
                    most = int(pages.count)
                elif None in pages.bounds:
                    self.stated_words(source, "a cast's slot")
                    self.refuse(
                        cast,
                        "a cast's slot holds the rectangle's most rows and columns, "
                        "and the build knows no most for a side here",
                        f"write each side r:r + m or r:{spelling('min')}(r + m, e), m "
                        "a constant; rows s apart r:r + m * s:s or r:"
                        f"{spelling('min')}(r + m * s, e):s",
                    )
                else:  # the slot of the most; the rows at the most columns' pitch
                    most = pages.bounds[0] * pages.bounds[1]
                    pitch = pages.bounds[1]
                if transposed:  # a column of the rectangle a row of the slot
                    rows = pages.bounds[0] if None not in pages.bounds else None
                    pitch = rows or int(pages.sides[0])
            else:
                most = self.constant_int(source.args[1])
            formats.add(self.tensor_format(cast, pages.tensor))
            self.moved(cast, pages.tensor, "broadcast's read")
            parts.append(
                {
                    "pages": pages,
                    "most": most,
                    "pitch": pitch,
                    "offset": count,
                    "transposed": transposed,
                }
            )
            count += most
        if len(formats) > 1:
            self.refuse(
                cast,
                "a cast's block is of one format and page size, and its parts are "
                f"of {len(formats)}",
                "cast parts of tensors of one format",
            )
        dtype, page_bytes = formats.pop()
        return parts, count, dtype, page_bytes

    def cast_reads(self, parts, address, page_bytes):
        """The C++ that reads a cast's parts into the slot at `address`, each at its
        first slot page: one part's read as it is, several's joined by &&."""
        reads = []
        for part in parts:
            pages, pitch = part["pages"], part["pitch"]
            at = address
            if part["offset"]:
                at = f"{address} + {part['offset'] * page_bytes}"
            tensor = f"{camel(pages.tensor)}({pages.tensor})"
            if part["transposed"]:  # each column of the rectangle, a page a row
                rows, columns = pages.sides
                row_step = (
                    pages.rect[2] if pages.rect is not None else
                    self.method.widths[pages.tensor]
                )  # fmt: skip
                reads.append(
                    "[&]() {\n"
                    f"        for (uint32_t km_column = 0; km_column < {columns}; "
                    "km_column++) {\n"
                    f"            for (uint32_t km_row = 0; km_row < {rows}; "
                    "km_row++) {\n"
                    "                if (!km::read_pages(movement, command, "
                    f"{tensor}, {pages.first} + km_row * {row_step} + km_column, 1, "
                    f"{at} + (km_column * {pitch} + km_row) * {page_bytes})) {{\n"
                    "                    return false;\n"
                    "                }\n"
                    "            }\n"
                    "        }\n"
                    "        return true;\n"
                    "    }()"
                )
            elif pages.rect is not None:  # the rectangle's rows into the slot, in order
                rows, columns, row_step = pages.rect[:3]
                pitch = pitch or columns
                reads.append(
                    "[&]() {\n"
                    f"        for (uint32_t km_row = 0; km_row < {rows}; km_row++) {{\n"
                    "            if (!km::read_pages(movement, command, "
                    f"{tensor}, {pages.first} + km_row * {row_step}, "
                    f"{columns}, {at} + km_row * {pitch * page_bytes})) {{\n"
                    "                return false;\n"
                    "            }\n"
                    "        }\n"
                    "        return true;\n"
                    "    }()"
                )
            else:
                count = part["most"] if pitch is None else pages.count
                reads.append(
                    f"km::read_pages(movement, command, {tensor}, "
                    f"{pages.first}, {count}, {at})"
                )
        return reads[0] if len(reads) == 1 else "(" + " && ".join(reads) + ")"

    def broadcast_end(self, name, blocks, uses):
        """After a broadcast's last block: its mover's finish, compute's padding."""
        broadcast = self.scope[name]
        buffer = broadcast.buffer.name
        movement = Line(
            f"if (!{name}.finish()) {{\n    return false;\n}}",
            {name},
            mover=broadcast.mover,
        )
        if broadcast.unread is not None:
            movement = self.unread_end(broadcast, blocks, uses, movement)
        compute = Line(
            f"km_ok = km_ok && km::broadcast_end(compute, {buffer}, {buffer}Pages, "
            f"{buffer}Window, {blocks});",
            uses | {"km_ok"},
        )
        return [movement], [compute]

    def drained_stream(self, node, name, call):
        """`out = km.Stream(self.f, mover="B")`: pages of own field f that compute's
        bodies fill and the line pushes, drained by the mover named (drain_loop), up
        to the end compute states (drained_end)."""
        field_ = self.field_name(call.args[0])
        keywords = {k.arg: k.value for k in call.keywords}
        named = keywords.get("mover")
        mover = named.value if isinstance(named, ast.Constant) else None
        if mover not in ("A", "B") or set(keywords) != {"mover"} or len(call.args) != 1:
            self.refuse(
                node,
                "a stream of an own field's pages is compute's to fill and one "
                "mover's to drain: mover= names it",
                f'write `{name} = km.{spelling("stream")}(self.f, mover="B")`',
            )
        if field_ not in self.compiled.own or self.compiled.own[field_] in (
            "mover_a",
            "mover_b",
        ):
            self.refuse(
                node,
                f"self.{field_} is no own field compute fills (a mover pushes it, or "
                "it is filled by a line)",
                f'declare it km.{spelling("own")}(push="pack")',
            )
        self.check_name(node, name)
        self.scope[name] = Value("drained", name, mover=mover)
        self.drained[name] = {
            "field": field_,
            "ended": None,
            "taken": False,
            "loops": (len(self.loops), node.lineno),
        }
        self.drained[name]["within"] = False
        pushed = Line(f"uint32_t km_pushed_{name} = 0;  // the pages pushed")
        return [], [pushed]

    def take_drained(self, node, name, where, bound, before, after):
        """A drained stream a compute body fills: its page reserved before the body,
        pushed after it and counted for the end."""
        state = self.drained[name]
        if where in ("A", "B"):
            self.refuse(
                node,
                f"{name}'s pages are compute's to fill; mover {self.scope[name].mover} "
                f"drains them",
                f"take it on compute, and drain it with `for page in {name}:`",
            )
        if state["ended"] is not None:
            self.refuse(
                node,
                f"{name} ended at line {state['ended']}: no page follows",
                "take it before its end",
            )
        state["taken"] = True
        buffer = f"kField{camel(state['field'])}"
        room = f"km_room_{name}{node.lineno}"
        before.append(Line(f"const bool {room} = compute.reserve_back({buffer}, 1);"))
        bound.append(
            f"const uint32_t {name} = {room} ? "
            f"get_local_cb_interface({buffer}).fifo_wr_ptr << 4 : 0;"
        )
        bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {buffer};")
        after.append(
            Line(
                f"if ({room}) {{\n    cb_push_back({buffer}, 1);\n"
                f"    ++km_pushed_{name};\n}}"
            )
        )
        return set()

    def end_buffer(self, node, name):
        """The page of a drained stream's end: the count of pages compute pushed."""
        return self.other_buffer(
            f"kEnd{camel(name)}",
            "word",
            "uint32",
            16,
            1,
            f"line {node.lineno}: the end of {name} (pack's, to mover "
            f"{self.scope[name].mover})",
        ).name

    def drained_loops(self, node, name, what):
        """A drained stream's make, end and drain stand in the same loops: compute
        pushes one end (a count) where the end line runs, the mover reads one where
        the drain runs, and a stream made in a loop is a new stream each turn."""
        loops, line = self.drained[name]["loops"]
        if loops == len(self.loops):
            return
        once = ""
        if loops > len(self.loops):  # made each turn, drained after: all turns' pages
            once = (
                f"; for every turn's pages drained once, make {name} before the loop "
                f"({name} = km.{spelling('stream')}(...)), its body lines in the loop, "
                f"and {name}.end() and the drain after it"
            )
        self.refuse(
            node,
            f"{name} was made at line {line} in {_loops(loops)} and {what} stands in "
            f"{_loops(len(self.loops))}: the end's count is pushed where the end line "
            "runs and read where the drain runs, so a turn's pages or a count are "
            "lost or waited for for good",
            f"make, end and drain {name} in the same loop (a with block per turn, and "
            f"`for page in {name}:` after it in that loop), or all three outside the "
            f"loops{once}",
        )

    def drained_end(self, node, name):
        """`out.end()`: compute's line saying no page follows, a hand-over of the
        count of pages pushed to the mover that drains them."""
        state = self.drained[name]
        if state["within"]:
            self.refuse(
                node,
                f"{name} ends where its with block closes",
                f"drop {name}.{spelling('stream_end')}(), or the with",
            )
        if state["ended"] is not None:
            self.refuse(
                node,
                f"{name} ended at line {state['ended']} already",
                "end it once",
            )
        self.drained_loops(node, name, "its end")
        state["ended"] = node.lineno
        end = self.end_buffer(node, name)
        text = (
            f"if (compute.reserve_back({end}, 1)) {{\n"
            f"    PACK((reinterpret_cast<volatile uint32_t*>(get_local_cb_interface("
            f"{end}).fifo_wr_ptr << 4)[0] = km_pushed_{name}));\n"
            f"    cb_push_back({end}, 1);\n}}"
        )
        return [], [Line(text)]

    def drained_with(self, node):
        """`with km.Stream(self.f, mover="B") as out:`: the stream, its block's
        lines, and its end where the block closes."""
        item = node.items[0] if len(node.items) == 1 else None
        if (
            item is None
            or not isinstance(item.optional_vars, ast.Name)
            or self.construct(item.context_expr) != "stream"
            or not self.field_name(
                item.context_expr.args[0] if item.context_expr.args else None
            )
        ):
            self.refuse(
                node,
                "a with block is a drained stream's, ended where it closes",
                f'write `with km.{spelling("stream")}(self.f, mover="B") as out:`',
            )
        name = item.optional_vars.id
        movement, compute = self.drained_stream(node, name, item.context_expr)
        self.drained[name]["within"] = True
        body_m, body_c = self.statements(node.body)
        self.drained[name]["within"] = False
        end_m, end_c = self.drained_end(node, name)
        self.drained[name]["ended"] = node.end_lineno
        return movement + body_m + end_m, compute + body_c + end_c

    def drain_loop(self, node, stream):
        """`for page in out:` the mover's loop over a drained stream's pages, a page
        a turn, until the count compute's end handed over."""
        state = self.drained[stream]
        mover = self.scope[stream].mover
        if state["ended"] is None:
            self.refuse(
                node,
                f"no line ends {stream} above: the mover would wait for pages forever",
                f"end it: {stream}.{spelling('stream_end')}() after the loop that "
                "fills it, or fill it in a with block",
            )
        if not state["taken"]:
            self.refuse(
                node,
                f"nothing pushes {stream}'s pages: no compute body takes it",
                f"fill it: km.{spelling('cxx')}(..., takes=[{stream}])",
            )
        self.drained_loops(node, stream, "its drain")
        if not isinstance(node.target, ast.Name):
            self.refuse(node, "one name a page", f"write `for page in {stream}:`")
        buffer = f"kField{camel(state['field'])}"
        end = self.end_buffer(node, stream)
        taken, total = f"km_taken_{stream}", f"km_total_{stream}"
        opening = (
            f"if (movement.poll([&] {{\n    if ({total} == 0xFFFFFFFFu && "
            f"cb_pages_available_at_front({end}, 1)) {{\n        "
            f"invalidate_l1_cache();\n        {total} = km::l1(get_read_ptr({end}))[0];"
            f"\n        cb_pop_front({end}, 1);\n    }}\n    return {taken} == {total} "
            f"|| cb_pages_available_at_front({buffer}, 1);\n}}, "
            "movement.args().wait_limit) != km::kGaveNone) {\n    return false;\n}\n"
            f"if ({taken} == {total}) {{\n    break;\n}}"
        )
        counts = Line(
            f"uint32_t {taken} = 0, {total} = 0xFFFFFFFFu;  // the end's count",
            mover=mover,
        )
        loop = self.page_loop(
            node, node.target.id, state["field"], mover, opening, f"++{taken};", ""
        )
        return [counts, *loop[0]], []

    def page_loop(self, node, name, field_, mover, opening, closing, _):
        """A mover's loop over pages of own field f, a page a turn: `opening` waits
        for one (or breaks), the body's lines (the mover's alone), the page freed
        once its writes read it, then `closing`; `name` the page's L1 address."""
        self.check_name(node, name)
        self.scope[name] = Value("page", name, mover=mover)
        body_m, body_c = self.statements(node.body)
        self.scope.pop(name)

        def movers(nodes, inherited=None):  # the mover of each line of the nodes
            for item in nodes:
                held = getattr(item, "mover", None) or inherited
                if isinstance(item, Line):
                    yield held
                else:
                    yield from movers(getattr(item, "children", []), held)

        if any(has_line(n) for n in body_c) or set(movers(body_m)) - {mover}:
            self.refuse(
                node.body[0],
                f"inside a stream mover {mover} drains, mover {mover}'s lines alone",
                "move the other CPU's lines before or after the loop",
            )
        self.wait(node, f"mover {mover}", WAIT_DATA)
        buffer = f"kField{camel(field_)}"
        loop = Block(
            "for (;;) {",
            "}",
            [
                Line(
                    f"{opening}\ninvalidate_l1_cache();\nconst uint32_t {name} = "
                    f"get_read_ptr({buffer});\nvolatile tt_l1_ptr uint32_t* "
                    f"{name}_words = km::l1({name});",
                    mover=mover,
                ),
                *body_m,
                Line(
                    "noc_async_writes_flushed();  // the page read by its writes\n"
                    f"cb_pop_front({buffer}, 1);\n{closing}",
                    mover=mover,
                ),
            ],
            mover=mover,
        )
        return [loop], []

    def block_loop(self, node, value):
        """`for tile in v:` over a block a broadcast brought: compute waits for it,
        takes its pages in order, frees its slot."""
        if not isinstance(node.target, ast.Name):
            self.refuse(node, "one name a tile", "write for tile in block")
        held = getattr(value, "held", False)  # waited at its cast, freed after
        if not held:
            self.arrived_once(node, node.iter.id, value)
        buffer = value.buffer.name
        page = f"km_page{value.buffer.index_a}"
        self.check_name(node, node.target.id)
        self.scope[node.target.id] = Value("streamed", buffer, first=page)
        body_m, body_c = self.statements(node.body)
        self.scope.pop(node.target.id)
        if body_m:
            self.refuse(
                node.body[0],
                "inside a block's loop only compute's lines",
                "move the line before or after the loop",
            )
        self.wait(node, "compute", WAIT_INPUT)
        loop = Block(
            f"for (uint32_t {page} = 0; {page} < {buffer}Pages; {page}++) {{",
            "}",
            body_c,
        )
        if held:  # read in place: the cast's lines wait for it and free it
            return [], [Block("if (km_ok) {", "}", [loop])]
        compute = [
            Line(
                "km_ok = km_ok && km::broadcast_receive(compute, "
                f"{buffer}, {buffer}Pages);",
                {"km_ok"},
            ),
            Block(
                "if (km_ok) {",
                "}",
                [loop, Line(f"km::broadcast_done({buffer}, " f"{buffer}Pages);")],
            ),
        ]
        return [], compute

    # Inline bodies and results -----------------------------------------------------

    def inline(self, node, call, name=None):
        """`km.cxx(text, takes=[...], gives=..., on=..., include=[...])`: the C++ as
        written, on its CPU (a mover, one compute CPU, or compute's three), the names
        it takes bound before it; `name = km.cxx(..., gives=int)` the word the body
        sets, `gives=km.Tile` the tile it leaves in DST 0 (compute)."""
        text = textwrap.dedent(self.body_text(node, call)).strip("\n")
        included = 0  # its first lines that are #include lines: the kernel's top
        lines = text.split("\n")
        while included < len(lines) and lines[included].lstrip().startswith("#include"):
            included += 1
        if any(line.lstrip().startswith("#include") for line in lines[included:]):
            self.refuse(
                node,
                "a body's #include lines come first (they go to its kernel's top)",
                "put them on the body's first lines",
            )
        text = "\n".join(lines[included:])
        on, takes, gives, includes, at = "compute", [], None, [], None
        stated = {}  # reads= and writes=: the tensors it reads or writes, stated
        for keyword_ in call.keywords:
            value = keyword_.value
            if keyword_.arg in (spelling("body_reads"), spelling("body_writes")):
                stated[keyword_.arg] = value
            elif keyword_.arg == spelling("at"):
                at = self.kernel_at(node, value)
            elif keyword_.arg == spelling("on"):
                if not (isinstance(value, ast.Constant) and value.value in BODY_CPUS):
                    self.refuse(
                        node,
                        "on= names a mover or a compute CPU (a body without it is "
                        "compute's three)",
                        'write on="mover_a", "mover_b", "unpack", "math" or "pack"',
                    )
                on = value.value
            elif keyword_.arg == spelling("takes"):
                if not isinstance(value, ast.List):
                    self.refuse(node, "takes= is a list of names", "write takes=[a]")
                takes = list(value.elts)
            elif keyword_.arg == spelling("include"):
                if not (
                    isinstance(value, ast.List)
                    and all(
                        isinstance(item, ast.Constant) and isinstance(item.value, str)
                        for item in value.elts
                    )
                ):
                    self.refuse(
                        node,
                        "include= is a list of header paths",
                        'write include=["../kernels/walk.hpp"]',
                    )
                includes = [item.value for item in value.elts]
            elif keyword_.arg == spelling("gives"):
                if isinstance(value, ast.Name) and value.id == "int":
                    gives = "int"
                elif isinstance(value, ast.Attribute) and value.attr == spelling(
                    "tile_value"
                ):
                    gives = "tile"
                else:
                    self.refuse(
                        node,
                        "a body gives a word or a tile",
                        f"write gives=int or gives=km.{spelling('tile_value')}",
                    )
            else:
                self.refuse(
                    node,
                    f"km.cxx has no {keyword_.arg}=",
                    f"use takes=, gives=, on=, include=, at=, "
                    f"{spelling('body_reads')}= and {spelling('body_writes')}=",
                )
        self.stated_tensors(node, stated, takes)
        if at is not None:
            self.kernel_takes(node, takes, gives, name)
        if (gives is None) != (name is None):
            self.refuse(
                node,
                "a body that gives a value is assigned, one that gives none is not",
                "write `x = km.cxx(..., gives=int)`, or km.cxx(...) on its own line",
            )
        mover = {"mover_a": "A", "mover_b": "B"}.get(on)
        cpu = on if on in COMPUTE_CPUS else None  # one compute CPU, or None
        if gives == "tile" and on != "compute":
            self.refuse(
                node,
                "a tile is compute's three CPUs'",
                "drop on= for a body giving a tile",
            )
        where = mover or cpu or "compute"
        uses, bound, before, after = set(), [], [], []
        for item in takes:
            if self.take_constant(item, bound):
                continue
            field = self.field_name(item)
            if field in self.compiled.own:
                self.take_own(node, field, where, text, bound)
                continue
            if field:
                self.take_field(node, field, where, bound, before, after)
                continue
            uses |= self.take(node, item, where, bound, before, after)
        if where == "compute":
            self.pack_words_steer(node, text, takes)
        kernel = "compute" if mover is None else "movement"
        guard = CPU_GUARDS.get(on)
        entries = [line.strip() for line in lines[:included]]
        for path in includes:  # the body's CPU's alone
            entry = f'#include "{path}"'
            entries.append(entry if guard is None else f"#if {guard}\n{entry}\n#endif")
        for entry in entries:
            if entry not in self.compiled.body_includes[kernel]:
                self.compiled.body_includes[kernel].append(entry)
        start = call.args[0].lineno + 1 + included
        if not isinstance(call.args[0], ast.Constant):  # the module's: the call's line
            start = node.lineno
        body = f'#line {start} "{self.compiled.source}"\n{text}\n#line RESUME'
        if bound:
            inner = "\n".join("    " + line for line in [*bound, *body.split("\n")])
            body = "{\n" + inner + "\n}"
        if gives == "int":
            self.check_name(node, name)
            body = f"uint32_t {name} = 0;\n{body}"
            self.scope[name] = Value("word", name, mover=where)
        elif gives == "tile":
            self.check_name(node, name)
            self.other_tile_line(node)
            if self.dst_live is not None or self.dst_next != 0:
                self.refuse(
                    node,
                    "a body's tile is DST 0, which another tile holds",
                    "write or reduce that tile first",
                )
            self.dst_next = 1
            self.dst_live = name
            self.scope[name] = Value("tile", name, dst=0)
            body = f"tile_regs_acquire();\n{body}"
        if cpu is not None:  # its CPU's alone
            body = f"#if {guard}\n{body}\n#endif"
        if at is not None:  # at the kernel's start or end, not in the handler
            if mover is not None:
                body = f"#if {guard}\n{body}\n#endif"
            self.compiled.kernel_bodies.setdefault((at, kernel), []).append(body)
            self.compiled.kernel_body_lines.append((at, node.lineno))
            return [], []
        lines = [*before, Line(body, uses, mover=mover), *after]
        moves, self.body_moves = self.body_moves, []
        if mover is None:
            return moves, lines
        return lines, []

    def stated_tensors(self, node, stated, takes):
        """`reads=[t]` and `writes=[t]` of a body: tensor parameters it takes, whose
        pages its own C++ reads or writes (a chain's description of the call; the
        build cannot check the C++ against them)."""
        taken = {item.id for item in takes if isinstance(item, ast.Name)}
        for keyword_, value in stated.items():
            if not isinstance(value, ast.List) or not all(
                isinstance(item, ast.Name) for item in value.elts
            ):
                self.refuse(
                    node,
                    f"{keyword_}= is a list of the tensors the body {keyword_}",
                    f"write {keyword_}=[t]",
                )
            for item in value.elts:
                if getattr(self.scope.get(item.id), "kind", None) != "tensor":
                    self.refuse(
                        node,
                        f"{item.id} in {keyword_}= is not a tensor parameter: "
                        f"{keyword_}= names the tensors whose pages the body "
                        f"{keyword_} in DRAM",
                        f"name tensor parameters in {keyword_}=",
                    )
                if item.id not in taken:
                    self.refuse(
                        node,
                        f"{item.id} is in {keyword_}= and not in "
                        f"{spelling('takes')}=: a body states what it {keyword_} of "
                        "what it takes",
                        f"take it: {spelling('takes')}=[..., {item.id}]",
                    )

    def pack_words_steer(self, node, text, takes):
        """A written page's word in a body on compute is pack's (0 once pack's reserve
        gave up, the address on unpack and math): it may guard the packs only, and is
        refused where it steers a section, a copy, a product or a way out (pack would
        leave unpack and math held inside the unit)."""
        words = [
            item.id
            for item in takes
            if isinstance(item, ast.Name)
            and getattr(self.scope.get(item.id), "kind", None) in ("write", "drained")
        ]
        for word in words:
            mentions = re.compile(rf"\b{re.escape(word)}\b")
            heads = []
            for line in text.splitlines():
                code = line.split("//")[0]
                steered = any(mentions.search(head) for head in heads)
                shared = STEERED.search(code)
                if shared and (steered or mentions.search(code)):
                    self.refuse(
                        node,
                        f"{word} is pack's word (0 once pack's reserve gave up, the "
                        f"page on unpack and math): `{code.strip()}` would run on some "
                        "of compute's CPUs and not on others",
                        f"guard only pack_tile (and the packer's lines) with {word}; "
                        "keep tile_regs_*, copies, products and the way out on all "
                        "three",
                    )
                for char in code:
                    if char == "{":
                        heads.append(code)
                    elif char == "}" and heads:
                        heads.pop()

    def kernel_at(self, node, value):
        """`at="start"` or `at="end"`: a body at its kernel's start (before the
        commands, after the start-up) or end (after the run)."""
        if not (isinstance(value, ast.Constant) and value.value in ("start", "end")):
            self.refuse(
                node,
                "at= is the kernel's start or its end",
                'write at="start" or at="end"',
            )
        return value.value

    def kernel_takes(self, node, takes, gives, name):
        """A body at a kernel's start or end takes the class's own fields alone (no
        handler's value is made there) and gives nothing."""
        if gives is not None or name is not None:
            self.refuse(
                node,
                "a body at a kernel's start or end gives nothing to the handler",
                "drop gives= and the name",
            )
        for item in takes:
            if self.field_name(item) not in self.compiled.own:
                self.refuse(
                    node,
                    "a body at a kernel's start or end takes the class's own fields "
                    "alone: no handler's value is made there",
                    f"take fields declared km.{spelling('own')}(), or drop at=",
                )

    def body_text(self, node, call):
        """A body's text: a string, a string constant of the class's module, or a sum
        of them (one text for the bodies of several handlers)."""

        def text_of(part):
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                return part.value
            if isinstance(part, ast.Name) and isinstance(self.names.get(part.id), str):
                return self.names[part.id]
            if isinstance(part, ast.BinOp) and isinstance(part.op, ast.Add):
                return text_of(part.left) + text_of(part.right)
            return None

        text = text_of(call.args[0]) if call.args else None
        if text is None:
            self.refuse(
                node,
                "an inline body is a string, a string constant of the module or a "
                "sum of them",
                'write km.cxx("..."), or km.cxx(BODY) with BODY = "..." at module '
                "level",
            )
        return text

    def take(self, node, item, where, bound, before, after):
        """A name a body takes on `where` (mover "A" or "B", "compute", or a compute
        CPU), as the table of docs/reference/classes.md ("Inline bodies") says: by
        name, a stream tile's buffer or L1 address, a write stream's page on pack, a
        tensor on a mover, a written tile on mover B."""
        if isinstance(item, ast.Name):
            self.not_ended(node, item.id)
        if not isinstance(item, ast.Name) or item.id not in self.scope:
            self.refuse(
                node,
                "takes= names the handler's values and the class's constants",
                "take names, or self.K of a constant",
            )
        value = self.scope[item.id]
        if value.kind in ("int", "partition", "pattern", "float"):
            return {item.id}  # a float word: a C++ float of that name
        if value.kind == "page" and where == value.mover:  # its L1 address
            return {item.id}
        if value.kind == "drained":
            return self.take_drained(node, item.id, where, bound, before, after)
        if value.kind == "page":
            self.refuse(
                node,
                f"{item.id} is a page mover {value.mover} reads",
                f'take it in a body on="mover_{value.mover.lower()}"',
            )
        if value.kind in ("entry", "entries"):
            return self.take_entry(node, item.id, value, bound)
        if value.kind == "arrived":
            return self.take_arrived(node, item.id, value, where, bound, before, after)
        if value.kind == "held":
            return self.take_held(node, item.id, value, where, bound)
        if value.kind == "streamed" and where == "compute":  # its buffer and place
            buffer, index = spelling("tile_buffer"), spelling("tile_index")
            bound.append(f"const uint32_t {item.id}{buffer} = {value.cxx};")
            bound.append(f"const uint32_t {item.id}{index} = {value.first};")
            return set()
        if value.kind == "streamed" and where in COMPUTE_CPUS:
            address = f"get_tile_l1_byte_address({value.cxx}, {value.first})"
            self.cpu_address(node, item.id, address, where, bound, before, after, True)
            return set()
        if value.kind == "write" and where in ("pack", "compute"):
            return self.take_write(node, item.id, value, bound, before, after)
        if value.kind == "write" and where == "B":
            self.take_page(node, item.id, value, bound, before, after)
            return set()
        if value.kind == "write":
            self.refuse(
                node,
                f"{item.id} is a write stream: its pages are pack's to fill",
                'take it in a body on="pack" (and after it, on="mover_b")',
            )
        if value.kind == "moved" and where in ("A", "B"):
            self.take_moved(node, item.id, value, bound)
            return set()
        if value.kind == "moved":
            self.refuse(
                node,
                f"{item.id} after its loop is what a mover moved",
                'take it in a body on="mover_a" or on="mover_b"',
            )
        if value.kind == "tensor" and where in ("A", "B"):
            word = f"km_word_{item.id}"
            kind = "km::Region" if item.id in self.compiled.regions else "uint32_t"
            bound.append(f"const {kind} {word} = {item.id};")
            bound.append(f"const {camel(item.id)} {item.id}({word});")
            return {item.id}
        if value.kind == "tensor":
            self.refuse(
                node,
                f"{item.id} is a tensor, in DRAM: the movers' to move",
                'take it in a body on="mover_a" or on="mover_b"',
            )
        if value.kind == "word" and (
            value.mover == where or value.mover == "compute" and where in COMPUTE_CPUS
        ):
            return {item.id}
        if value.kind == "word" and {value.mover, where} == {"A", "B"}:
            return self.take_handed(node, item.id, value, where, bound, before, after)
        if value.kind in ("written", "reduced") and value.buffer is not None:
            if where != "B":
                self.refuse(
                    node,
                    f"{item.id} is packed for mover B's write",
                    'take it in a body on="mover_b"',
                )
            if value.popped:
                self.refuse(
                    node,
                    f"{item.id} was written and its page freed above",
                    "take it before the line that writes it",
                )
            if not value.waited:
                before.append(Line(self.page_wait(value) + "\n    return false;\n}",
                                   mover="B"))  # fmt: skip
                value.waited = True
            bound.append(f"const uint32_t {item.id} = {value.buffer.name};")
            return set()
        if value.kind == "reduced" and value.summed:
            return self.take_sum(node, item.id, value, where, bound, before, after)
        self.refuse(
            node,
            f"{item.id} is a {value.kind} here: a body takes ints, partitions, "
            "patterns, words of its CPU, stream tiles, write streams (on pack), "
            "tensors (on a mover) and written tiles",
            "pass those",
        )

    def take_sum(self, node, name, value, where, bound, before, after):
        """A reduce's sum of a field, taken by a body of compute's three at the root:
        the field's buffer and first tile bound (the group's sum, fp32), waited for
        before the body and freed after it, as the hand's operation on the sum."""
        field_ = value.summed
        if not self.in_leader or value.popped:
            self.refuse(
                node,
                f"{name} is the reduce's sum of self.{field_}, which only the leader "
                "holds, once",
                f"take it once in a body under `if self.{spelling('rank')} == "
                f"self.{spelling('leader')}:`",
            )
        held = self.sum_held(value)
        if not held:
            self.same_loops(node, name, value, "this body", *self.once_text(value))
        if where != "compute":
            self.refuse(
                node,
                f"{name} is the group's sum in compute's L1 (fp32 tiles of "
                f"self.{field_}): a body of compute's three takes it",
                f"drop {spelling('on')}= from this body",
            )
        value.popped = True
        self.leader_writes.add(name)
        buffer = f"kField{camel(field_)}"
        before.append(
            Line(
                f"if (!compute.wait_front({buffer}, {buffer}Pages)) {{  // {name}\n"
                "    return;\n}"
            )
        )
        bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {buffer};")
        bound.append(f"const uint32_t {name}{spelling('tile_index')} = 0;")
        freed = f"cb_pop_front({buffer}, {buffer}Pages);  // {name} freed"
        if held:
            self.sum_freed_after(value, freed)
        else:
            after.append(Line(freed))
        self.wait(node, "compute", WAIT_INPUT)
        return set()

    def cpu_address(self, node, name, address, cpu, bound, before, after, release):
        """An L1 address unpack knows, for a body on one compute CPU: unpack's own;
        math or pack are handed it before the body and, with `release`, hand back a
        release after it (unpack frees nothing before)."""
        if cpu == "unpack":
            bound.append(f"const uint32_t {name} = {address};")
            return
        thread = f"ckernel::ThreadId::{cpu.capitalize()}ThreadId"
        before.append(
            Line(
                "#if defined(TRISC_UNPACK)\n"
                f"compute.hand({thread}, {address});\n"
                "#endif"
            )
        )
        bound.append(
            f"const uint32_t {name} = "
            "compute.handed(ckernel::ThreadId::UnpackThreadId);"
        )
        self.handover(node, f"{name}'s L1 address", "unpack", (cpu,))
        if not release:
            return
        after.append(
            Line(
                f"// {cpu} is done with {name}: unpack may free it\n"
                f"#if {CPU_GUARDS[cpu]}\n"
                "compute.hand(ckernel::ThreadId::UnpackThreadId, 1);\n"
                "#elif defined(TRISC_UNPACK)\n"
                f"(void)compute.handed({thread});\n"
                "#endif"
            )
        )
        self.handover(node, f"the release of {name}", cpu, ("unpack",))

    def taken_by_field_products(self, name):
        """Whether products with a field on the left take block `name`
        (`f(self.p, name[i][r0:r0 + n, :])`): it is then waited for at its cast and
        freed at the end of the cast's lines, for any number of them."""
        for node in ast.walk(self.definition):
            if (
                isinstance(node, ast.Call)
                and len(node.args) == 2
                and (
                    self.field_name(node.args[0]) is not None
                    or self.held_row(node.args[0]) is not None
                )
                and any(
                    isinstance(n, ast.Name) and n.id == name
                    for n in ast.walk(node.args[1])
                )
            ):
                return True
        return False

    def taken_in_a_loop(self, cast, name):
        """Whether block `name`, cast at `cast`, is taken in a range loop the cast does
        not stand in (a block held across turns: a weight cast once and read each turn):
        it is then waited for at its cast and freed at the end of the cast's lines,
        and every take reads it in place."""
        for loop in ast.walk(self.definition):
            if (
                not isinstance(loop, ast.For)
                or not isinstance(loop.iter, ast.Call)
                or not isinstance(loop.iter.func, ast.Name)
                or loop.iter.func.id != "range"
                or any(n is cast for n in ast.walk(loop))
            ):
                continue  # a range loop the cast does not stand in
            if any(
                isinstance(n, ast.Name) and n.id == name
                for line in loop.body
                for n in ast.walk(line)
            ):
                return True
        return False

    def held_cast_movers(self, node, letters, how="stream"):
        """Refuses a stream (or a page write, a fill) on the mover of a block cast
        outside a loop that products take: the mover waits at the cast for compute
        to free the block, which compute does at the end of the cast's lines, after
        this line."""
        for ends in self.held_blocks.values():
            for name, _ in ends:
                mover = self.finished_casts.get(name)
                if mover is not None and any(
                    each is None or mover in each for each in letters
                ):
                    other = "A" if mover == "B" else "B"
                    self.refuse(
                        node,
                        f"{name} was cast outside a loop and products take it, so "
                        f"mover {mover} waits at the cast for compute to free it, "
                        f"which compute does after this {how}, which mover {mover} "
                        "moves",
                        (
                            f'read the stream on mover {other}: pattern="{other}"'
                            if how == "stream"
                            else f'cast it on mover {other} (mover="{other}"), or in '
                            "the loop around this line"
                        ),
                    )

    def held_row(self, arg):
        """(the held stream's Value, the row's C++) of `qs[r]`, qs a held stream;
        else None."""
        if not (isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name)):
            return None
        value = self.scope.get(arg.value.id)
        if value is None or value.kind != "held":
            return None
        return value, arg.slice

    def held_block_product(self, node, value, function, held, arrived):
        """`acc += f(qs[r], w)` (or `w[i][:, c0:c0 + n]`), qs a held stream (its
        rows of k tiles) and w a block a broadcast brought that products take: tile
        j of qs's row r by row j of w's part (acc's width of columns, transposed
        where the cast turned it) into acc's DST, outside a stream loop."""
        stream, row = held
        block_value, (offset, _, pitch, transposed), first, columns, name = arrived
        width = value.width
        if self.loop_states:
            self.refuse(
                node,
                "a product of a held stream's row runs on its own, and this stream "
                "loop holds the matrix unit's setup for its own products",
                "write it after the loop",
            )
        k = stream.columns
        if k is None:
            self.refuse(
                node,
                f"{ast.unparse(node.value.args[0])}'s stream has rows of tiles the "
                "build does not know",
                "stream a rectangle whose columns are a constant: t.tiles[r:..., 0:k]",
            )
        if columns is None:
            columns = pitch
        if columns != width or pitch is None:
            self.refuse(
                node,
                f"the block's part is {columns or 'a word of'} columns and "
                f"{ast.unparse(node.target)} holds {width}",
                f"take a window of {width}: w[i][:, c0:c0 + {width}]",
            )
        if not getattr(block_value, "held", False):
            self.refuse(
                node,
                f"{name} is not waited for at its cast",
                "cast it in the lines around this product",
            )
        buffer = block_value.buffer
        left = stream.buffer.dtype
        why = product_refusal(left, buffer.dtype)
        if why is not None:
            self.refuse(node, why, "cast tiles of a matrix format of the stream's")
        fidelity = self.product_fidelity(node, function)
        if self.placed(
            node, {}, function, left, buffer.dtype, ("tensor", block_value.tensor)
        ):
            self.refuse(
                node,
                f"{function.name} turns the product, and a held stream's product is "
                "built plain",
                "drop the marker",
            )
        most = stream.buffer.block // k
        line = self.integer(row)
        bound = (
            ""
            if line.rstrip("u").isdigit()
            else (
                f"if (!(({line}) < {most})) {{  // the held block's rows\n"
                "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
                "    return;\n}\n"
            )
        )
        if line.rstrip("u").isdigit() and not int(line.rstrip("u")) < most:
            self.refuse(
                node,
                f"the held block has {most} rows, 0 .. {most - 1}",
                f"index it below {most}",
            )
        self.compiled.includes.add("matmul")
        self.compiled.products.append(
            (self.method.name, node.lineno, self.text(node), function, width, str(k))
            + (fidelity,)
        )
        if getattr(value, "row_of", None) is None:  # a mask's -inf covers the tail
            part = arrived[1]
            bound += self.window_inside(block_value, part, first, width, "columns")
        q, w = stream.cxx, buffer.name
        flip = "true" if transposed else "false"
        calls = "\n".join(
            f"    matmul_block({q}, {w}, ({line}) * {k} + {j}, {offset} + {j} * "
            f"{pitch} + {first}, {value.dst}, {flip}, {width}, 1, {k});"
            for j in range(k)
        )
        text = (
            f"{bound}"
            "if (km_ok) {\n"
            "    // the sums' setup (vector unit) done before the matrix unit adds\n"
            "    MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
            f"    KM_UNPACK_FORMATS({w}, {q});\n"
            f"    matmul_block_init({q}, {w}, {flip}, {width}, 1, {k});\n"
            f"{calls}\n"
            "}"
        )
        uses = {"km_ok"} | self.uses(row) | text_uses(first, self)
        return [], [Line(text, uses | text_uses(bound, self))]

    def window_inside(self, block_value, part, first, width, along):
        """A run-time check that a window of `width` from `first` along the part's
        rows ("rows") or columns ("columns") of the slot stays inside the cast's
        rectangle (a short block leaves the slot's tail holding an earlier block):
        "" where the build knows it, else a give-up past it."""
        index = block_value.parts.index(part)
        rows, columns = block_value.sides[index]
        if part[3]:  # turned: the slot's rows are the rectangle's columns
            rows, columns = columns, rows
        valid = str(rows if along == "rows" else columns)
        if valid.isdigit() and str(first).isdigit():
            return ""
        return (
            f"if (!(({first}) + {width} <= ({valid}))) {{  // the window inside the "
            "cast's rectangle\n"
            "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
            "    return;\n}\n"
        )

    def held_block_ends(self, body):
        """compute's frees of the blocks products took (taken_by_field_products),
        cast in `body`, at its end."""
        ends = self.held_blocks.pop(id(body), [])
        nodes = [
            Line(f"if (km_ok) {{\n    km::broadcast_done({w}, {w}Pages);\n}}")
            for _, w in ends
        ]
        if not nodes:
            return []
        return [Group("the end of the cast's lines: the products' blocks freed", nodes)]

    def arrived_rows(self, node, arg):
        """`w[i][r0:r0 + n, :]` (or `w[r0:r0 + n, :]`), w a block a broadcast brought:
        (its Value, the part's slot geometry, r0's C++, n); None when `arg` is not."""
        if not (
            isinstance(arg, ast.Subscript)
            and isinstance(arg.slice, ast.Tuple)
            and len(arg.slice.elts) == 2
        ):
            return None
        (rows, cols), inner = arg.slice.elts, arg.value
        part = None
        if isinstance(inner, ast.Subscript) and isinstance(inner.value, ast.Name):
            part, inner = inner.slice, inner.value
        value = isinstance(inner, ast.Name) and self.scope.get(inner.id)
        if not value or value.kind != "arrived":
            return None
        drawn = f"write {inner.id}[i][r0:r0 + n, :], n the field's tiles"
        whole = (
            isinstance(cols, ast.Slice)
            and cols.lower is cols.upper is cols.step is None
        )
        sliced = isinstance(rows, ast.Slice) and rows.step is None
        known = (  # a:b, both ints the build knows
            sliced
            and None not in (rows.lower, rows.upper)
            and None not in (self.known_int(rows.lower), self.known_int(rows.upper))
        )
        if not whole or not (
            known
            or sliced
            and rows.lower is not None
            and isinstance(rows.upper, ast.BinOp)
            and isinstance(rows.upper.op, ast.Add)
            and ast.dump(rows.upper.left) == ast.dump(rows.lower)
        ):
            self.refuse(
                node, "a product's field on the left takes rows r0:r0 + n", drawn
            )
        index = 0 if part is None else self.known_int(part)
        if index is None or not 0 <= index < len(value.parts):
            self.refuse(
                node, f"{inner.id} has parts 0 .. {len(value.parts) - 1}", drawn
            )
        if known:
            low, high = self.known_int(rows.lower), self.known_int(rows.upper)
            return value, value.parts[index], str(low), high - low, inner.id
        count = self.known_int(rows.upper.right)
        if count is None:
            self.refuse(node, "the rows' count is an int the build knows", drawn)
        return value, value.parts[index], self.integer(rows.lower), count, inner.id

    def field_block_product(self, node, value, function, field_, rows):
        """`acc += f(self.p, w[i][r0:r0 + n, :])`, self.p a row of n tiles compute
        wrote, w a block a broadcast brought (products take it: waited at its cast):
        tile k of the field by row r0 + k of w's part, acc's width a row, into acc's
        DST; outside a stream loop, any number a block."""
        block_value, (offset, _, pitch, transposed), first, count, name = rows
        width = value.width
        if self.loop_states:
            self.refuse(
                node,
                "a product of a field and a block runs on its own, and this stream "
                "loop holds the matrix unit's setup for its own products",
                "write it after the loop",
            )
        if field_ not in self.packed:
            self.refuse(
                node,
                f"self.{field_} is not written above in this handler: a product's left "
                "field is one compute writes",
                f"write self.{field_}.write(acc) before",
            )
        stored = self.compiled.fields[field_]
        k = math.prod(stored.shape[:-2])
        if count != k:
            self.refuse(
                node,
                f"self.{field_} is {k} tile(s) and the block's rows {count}",
                f"take rows r0:r0 + {k}",
            )
        if transposed or pitch != width:
            self.refuse(
                node,
                f"the block's part is {pitch or 'a word of'} columns"
                f"{' (cast turned)' if transposed else ''} and "
                f"{ast.unparse(node.target)} holds {width}",
                f"cast a part of {width} columns, not turned",
            )
        buffer = block_value.buffer
        why = product_refusal(stored.format.dtype, buffer.dtype)
        if why is not None:
            self.refuse(node, why, "declare the field in a matrix format of the cast's")
        fidelity = self.product_fidelity(node, function)
        if self.placed(
            node,
            {},
            function,
            stored.format.dtype,
            buffer.dtype,
            ("tensor", block_value.tensor),
        ):
            self.refuse(
                node,
                f"{function.name} turns the product, and a field's product on the left "
                "is built plain",
                "drop the marker",
            )
        self.compiled.includes.add("matmul")
        self.compiled.products.append(
            (self.method.name, node.lineno, self.text(node), function, width, str(k))
            + (fidelity,)
        )
        field_buffer, w = f"kField{camel(field_)}", buffer.name
        bound = ""
        if field_ not in self.masked:  # a masked P is zero on the tail
            bound = self.window_inside(block_value, rows[1], first, k, "rows")
        self.wait(node, "compute", WAIT_INPUT)
        calls = "\n".join(
            f"    matmul_block({field_buffer}, {w}, {t}, {offset} + ({first} + {t}) * "
            f"{pitch}, {value.dst}, 0, {width}, 1, {k});"
            for t in range(k)
        )
        text = (
            f"{bound}"
            f"km_ok = km_ok && compute.wait_front({field_buffer}, "
            f"{field_buffer}Pages);\n"
            "if (km_ok) {\n"
            "    // the sums' setup (vector unit) done before the matrix unit adds\n"
            "    MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
            f"    KM_UNPACK_FORMATS({w}, {field_buffer});\n"
            f"    matmul_block_init({field_buffer}, {w}, 0, {width}, 1, {k});\n"
            f"{calls}\n"
            "}"
        )
        uses = {"km_ok"} | text_uses(first, self) | text_uses(bound, self)
        return [], [Line(text, uses)]

    def started_from_row(self, node, name, call, field_, state, index, row):
        """`acc = f(self.o[i], self.s[r])`: started_from_field from row i of a field
        of rows filled above (n x w fp32 tiles), read in place by index after its
        token, as a step's states are; acc's write `self.o[j].write(acc)` packs it
        back in place and pushes the token."""
        function = self.tile_function_of(call)
        if self.dst_live is not None:
            self.refuse(
                node,
                f"{self.dst_live} still holds DST; a step's row starts at DST 0",
                "write it first",
            )
        stored = self.compiled.fields[field_]
        width = stored.shape[1] if len(stored.shape) == 4 else 0
        if not 1 <= width <= 3:
            self.refuse(
                node,
                f"self.{field_}'s rows are {width or 'not'} tiles; a step's row is 1 "
                "to 3 fp32 tiles",
                f"declare self.{field_}: km.L1[km.fp32, rows, n, 32, 32], n at most 3",
            )
        source, token_o, line_o, bound_o = self.step_state(node, field_, row, width)
        buffer, token, slot, bound = self.step_state(node, state, index)
        unit, cxx = self.step_unit(node, function, width)
        base = f"({line_o}) * {width}"
        copies = "".join(  # tile 0 copied with the token's take
            f"    copy_tile({source}, {base} + {t}, {t});\n" for t in range(1, width)
        )
        text = (
            f"{bound_o}{bound}"
            f"// {function.name}'s step: self.{field_}[{ast.unparse(row)}] into DST 0 "
            f".. {width - 1}, self.{state}[{ast.unparse(index)}] into DST {width}\n"
            "tile_regs_acquire();\n"
            f"{self.state_taken(buffer, token, slot, width)}"
            "}\n"
            f"{self.state_taken(source, token_o, f'{base}', 0)}"
            f"{copies}"
            f"{self.step_calls(cxx)}"
            "}"
        )
        self.wait(node, "compute", WAIT_INPUT)
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, self.text(node), function, cxx)
        )
        self.dst_next = max(self.dst_next, width + 1)
        if self.dst_next > DST_HALF:
            self.compiled.full_dst = True
        self.dst_live = name
        value = Value(
            "tile", name, dst=0, width=width, made_at=(len(self.loops), node.lineno)
        )
        value.state = (buffer, token, slot, width, index)
        value.row_of = (field_, source, token_o)  # written back in place
        self.scope[name] = value
        uses = {"km_ok"} | self.uses(index) | self.uses(row)
        return [], [Line(text, uses)]

    def write_row(self, node, call, field_, row):
        """`self.o[j].write(acc)`, acc started from a row of self.o: its tiles packed
        into row j in place, the token pushed, then the state packed (acc's)."""
        written = call.args[0] if len(call.args) == 1 else None
        value = isinstance(written, ast.Name) and self.scope.get(written.id)
        if (
            not value
            or value.kind != "tile"
            or getattr(value, "row_of", (None,))[0] != field_
        ):
            self.refuse(
                node,
                f"a row of self.{field_} takes sums started from one of its rows "
                "(their token orders its reads and writes)",
                f"write acc = f(self.{field_}[i], self.s[r]) above, then "
                f"self.{field_}[j].write(acc)",
            )
        self.same_loops(node, written.id, value, "this write", *self.once_text(value))
        _, source, token_o = value.row_of
        line = self.integer(row)
        items = self.compiled.fields[field_].shape[0]
        width = value.width
        bound = (
            ""
            if line.rstrip("u").isdigit()
            else (
                f"if (!(({line}) < {items})) {{  // self.{field_}'s rows\n"
                "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
                "    return;\n}\n"
            )
        )
        if line.rstrip("u").isdigit() and not int(line.rstrip("u")) < items:
            self.refuse(
                node,
                f"self.{field_} holds {items} rows, 0 .. {items - 1}",
                f"index it below {items}",
            )
        packs = "".join(
            f"    pack_tile<true>({value.dst + t}, {source}, ({line}) * {width} + "
            f"{t});\n"
            for t in range(width)
        )
        text = (
            f"{bound}"
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"if (km_ok && compute.reserve_back({token_o}, 1)) {{  // pack's alone\n"
            f"    KM_PACK_FORMAT({source});\n"
            f"{packs}"
            f"    cb_push_back({token_o}, 1);\n"
            "}\n"
            "tile_regs_release();"
        )
        self.compiled.includes.add("pack")
        value.kind, value.popped = "written", True
        self.dst_live = None
        compute = [Line(text, {"km_ok"} | self.uses(row))]
        self.state_packed(compute, *value.state)
        return [], compute

    def arrived_loops(self, node, name, value):
        """A block a broadcast brought is taken in the loops its cast stands in: the
        movers cast a block each time the cast runs, compute takes one each time the
        take runs."""
        self.same_loops(
            node,
            name,
            value,
            "this take",
            "a cast makes one block each time its line runs and a take frees one each "
            "time its line runs, so the movers' blocks and compute's differ",
            "cast it in the loop that takes it (a block a turn), or take it once, "
            "outside the loops",
        )

    def arrived_once(self, node, name, value):
        """A block a broadcast brought is taken once: a body or a loop over it."""
        self.arrived_loops(node, name, value)
        if value.popped:
            self.refuse(
                node,
                "this block is taken above (a body or a loop over it), and freed",
                "take it once each block",
            )
        value.popped = True

    def take_arrived(self, node, name, value, where, bound, before, after):
        """A block a broadcast brought, taken whole by a body on compute's three: the
        block waited for before it, `name_buffer` its buffer and `name` its first page
        there (0), freed after it."""
        if where != "compute":
            self.refuse(
                node,
                f"{name} is a block of compute's three CPUs: each waits for it and "
                "frees it",
                "take it in a body on compute's three (no on=)",
            )
        held_in_loop = getattr(value, "held", False) and not (
            self.taken_by_field_products(name)
        )
        if getattr(value, "held", False) and not held_in_loop:
            self.refuse(
                node,
                f"{name} is taken by products of a field (waited at its cast, freed at "
                "the end of its lines), and a body takes a block whole",
                "take it in products only, or in a body only",
            )
        buffer = value.buffer.name
        if held_in_loop:  # read in place: the cast's lines wait for it and free it
            bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {buffer};")
            bound.append(f"const uint32_t {name} = 0;")
            return set()
        self.arrived_once(node, name, value)
        self.wait(node, "compute", WAIT_INPUT)
        before.append(
            Line(
                f"km_ok = km_ok && km::broadcast_receive(compute, {buffer}, "
                f"{buffer}Pages);",
                {"km_ok"},
            )
        )
        bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {buffer};")
        bound.append(f"const uint32_t {name} = 0;")
        after.append(
            Line(f"if (km_ok) {{\n    km::broadcast_done({buffer}, {buffer}Pages);\n}}")
        )
        return set()

    def take_write(self, node, name, value, bound, before, after):
        """A write stream's page for a body on pack: in a stream loop the loop's
        page (reserved with the block, pushed after it); outside one, the stream's
        one block, reserved, pushed and moved at this line. The page's L1 address,
        0 when compute gave up."""
        if value.pushed:
            self.refuse(
                node,
                f"{name} is fed above (a push or another body)",
                "open a write stream for each",
            )
        value.pushed = True
        buffer = value.buffer.name
        if self.stream_loops:
            state = self.loop_states[-1]
            if value.buffer.block != state["block"]:
                self.refuse(
                    node,
                    f"{name}'s block is {value.buffer.block} page(s), the loop's "
                    f"{state['block']}",
                    "give the write stream the loop's block=",
                )
            self.loop_fed(node, name, value)
            state["body_pushed"] = True
            number = self.stream_loops[-1]
            out, room, tile = f"km_out{number}", f"km_room{number}", f"km_tile{number}"
            ring = f"get_local_cb_interface({out})"
            bound.append(
                f"const uint32_t {name} = {room} ? ({ring}.fifo_wr_ptr + {tile} * "
                f"{ring}.fifo_page_size) << 4 : 0;"
            )
            bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {out};")
            return set()
        if value.depth < len(self.loops):  # a block each turn of the loops inside
            return self.take_fed(node, name, value, bound, before, after)
        if value.count != str(value.buffer.block):
            self.refuse(
                node,
                f"{name} outside a stream loop is one block: {value.count} page(s), "
                f"its block {value.buffer.block}",
                "give the write stream one block, or fill it in a stream loop",
            )
        out, room = f"km_out_{name}", f"km_room_{name}"
        given = value.pattern if value.pattern in self.scope else None
        before.append(
            Line(
                f"const uint32_t {out} = "
                f"km::block_half({buffer}, {value.pattern}, 0);\n"
                f"const bool {room} = compute.reserve_back({out}, {buffer}Block);",
                uses=pattern_uses(given),
            )
        )
        bound.append(
            f"const uint32_t {name} = {room} ? "
            f"get_local_cb_interface({out}).fifo_wr_ptr << 4 : 0;"
        )
        bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {out};")
        after.append(
            Line(f"if ({room}) {{\n    cb_push_back({out}, {buffer}Block);\n}}")
        )
        value.move = Line(
            f"if (!movement.move({value.cxx})) {{\n    return false;\n}}", {value.cxx}
        )
        self.body_moves.append(value.move)
        self.wait(node, "compute", WAIT_OUTPUT)
        self.wait(node, "mover B", WAIT_DATA)
        return set()

    def take_fed(self, node, name, value, bound, before, after):
        """A write stream made outside the loops around a body on compute or pack:
        one block of it each time the body runs (compute counts them), reserved
        before the body and pushed after it; the movers move the whole stream after
        the outermost of those loops (fed_moves)."""
        if value.move is not None and not value.table:
            self.refuse(
                node,
                f"{name} is fed above outside a loop",
                "open a write stream for each",
            )
        buffer, fed = value.buffer.name, f"km_fed_{name}"
        out, room = f"km_out_{name}", f"km_room_{name}"
        given = value.pattern if value.pattern in self.scope else None
        before.append(
            Line(
                f"const uint32_t {out} = km::block_half({buffer}, {value.pattern}, "
                f"{fed});\n"
                f"const bool {room} = compute.reserve_back({out}, {buffer}Block);",
                uses=pattern_uses(given) | {fed},
            )
        )
        bound.append(
            f"const uint32_t {name} = {room} ? "
            f"get_local_cb_interface({out}).fifo_wr_ptr << 4 : 0;"
        )
        bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {out};")
        after.append(
            Line(
                f"if ({room}) {{\n    cb_push_back({out}, {buffer}Block);\n}}\n"
                f"{fed}++;",
                {fed},
            )
        )
        if value.move is None:
            value.move = Line(
                f"if (!movement.move({value.cxx})) {{\n    return false;\n}}",
                {value.cxx},
            )
            value.table = {"fed": True}
            self.loops[value.depth].setdefault("moves", []).append(value.move)
        self.wait(node, "compute", WAIT_OUTPUT)
        self.wait(node, "mover B", WAIT_DATA)
        return set()

    def fed_moves(self, node, loop, body_m):
        """The moves of the write streams a loop's bodies feed (take_fed), after the
        loop; refused when the loop has movers' lines too (each would wait on the
        other)."""
        moves = loop.get("moves", [])
        if moves and body_m:
            self.refuse(
                node,
                "a write stream fed a block a turn is moved after the loop, and this "
                "loop has movers' lines too: the movers would wait on each other",
                "feed it in a loop of compute's lines alone",
            )
        return list(moves)

    def write_keyword(self, call):
        """True when a stream's call has write= (a write stream's, or refused there)."""
        return any(k.arg == spelling("write_stream") for k in call.keywords)

    def held_stream(self, node, name, call):
        """`q = km.Stream(t.tiles[...], block=b)` outside a for line: a read stream of
        one block that bodies take whole. The movers move it at this line; compute
        waits for its block here and frees it at the end of the lines it stands in
        (held_ends), so the loops between may hold movers' lines too."""
        pages, block, pattern, letters, repeat = self.stream_options(node, call, False)
        if self.row_width != 1 or repeat != "1" or pages.items is not None:
            self.refuse(
                node,
                "a stream bodies take is one block of pages, read once",
                f"drop {spelling('row')}=, {spelling('repeat')}= and items, or "
                f"write `for tile in km.{spelling('stream')}(...)`",
            )
        most = held_most(pages)
        if most is None or most > block:
            known = "unknown to the build" if most is None else most
            self.refuse(
                node,
                f"a stream bodies take is one block, and its pages are at most "
                f"{known}, its block {block}",
                f"give {spelling('block')}= its most pages (a side r:min(r + n, e), or "
                "rows r:min(r + n * s, e):s), or iterate it in a for line",
            )
        if not (letters or "").startswith("A"):
            self.refuse(
                node,
                "a stream bodies take is mover A's block (an empty one is pushed "
                "by mover A)",
                'drop pattern=, or give one that starts with "A"',
            )
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        index = self.reads
        self.reads += 1
        buffer = self.pair(
            "read",
            index,
            dtype,
            page_bytes,
            block,
            1,  # one block a half: the block is held until the end of its lines
            f"line {node.lineno}: {self.text(node)}",
        )
        if self.stream_word is not None:
            self.refuse(
                node,
                f"a block bodies take, given at each submit ({self.stream_word}), is "
                "not built",
                "give block= its most pages, a constant",
            )
        pattern_text = self.pattern_cxx(buffer, pattern, letters)
        if dtype == "float32":
            self.compiled.unpack_fp32.add(buffer.name)
        self.moved(node, pages.tensor, "read stream")
        variable, half = f"km_read{index}", f"km_held_{name}"
        stream_uses = range_uses(pages, self) | pattern_uses(pattern)
        movement = [
            Glue(
                self.stream_cxx("read", variable, pages, buffer, pattern_text, "1"),
                {variable},
                stream_uses,
            ),
            *held_move(variable, half, buffer.name, pages.count),
        ]
        for line in movement[1:]:
            line.uses |= text_uses(pages.count, self)
        self.wait(node, "mover A", WAIT_DATA)
        self.wait(node, "mover B", WAIT_DATA)
        self.wait(node, "compute", WAIT_INPUT)
        self.compute_waits = True
        compute = []
        if pattern is not None:
            compute.append(
                Line(f"if (!{pattern}.valid()) {{\n    return;\n}}", {pattern})
            )
        compute.append(
            Line(
                f"const uint32_t {half} = km::block_half({buffer.name}, "
                f"{pattern_text}, 0);\n"
                f"km_ok = km_ok && compute.wait_front({half}, {buffer.name}Block);",
                {"km_ok"} | pattern_uses(pattern),
            )
        )
        self.scope[name] = Value("held", half, buffer=buffer)
        bounds = getattr(pages, "bounds", None)  # a row's tiles, for products
        self.scope[name].columns = (
            bounds[1] if bounds and isinstance(bounds[1], int) else None
        )
        self.held.setdefault(self.blocks[-1], []).append((name, node))
        return movement, compute

    def held_ends(self, body):
        """compute's frees of the held streams made in `body` (held_stream), at its
        end; their names end with it."""
        nodes = []
        for name, node in self.held.pop(id(body), []):
            value = self.scope.pop(name)
            self.ended[name] = ("lines around its stream", node.lineno)
            buffer = value.buffer.name
            nodes.append(
                Line(
                    f"if (km_ok) {{\n    cb_pop_front({value.cxx}, {buffer}Block);\n}}",
                    {"km_ok"},
                )
            )
        if not nodes:
            return []
        return [Group("the end of the held streams' lines: their blocks freed", nodes)]

    def held_not_iterated(self, node, iterator):
        """Refuses a loop over a held stream: it is one block, taken or iterated."""
        if isinstance(iterator, ast.Name) and (
            self.scope.get(iterator.id, Value("")).kind == "held"
        ):
            self.refuse(
                node,
                f"{iterator.id} is a read stream that bodies take, one block: it is "
                "not iterated too",
                f"iterate a stream of its own, `for tile in km.{spelling('stream')}"
                "(...)`, or take this one in a body",
            )

    def held_not_traced(self, node, call, function):
        """Refuses a tile function's call on a held stream: its operands are a
        stream loop's tiles, and a held block has no loop of its tiles."""
        for arg in call.args:
            if isinstance(arg, ast.Name) and (
                self.scope.get(arg.id, Value("")).kind == "held"
            ):
                self.refuse(
                    node,
                    f"{arg.id} is a read stream that bodies take, one block: the tile "
                    f"function {function.name} takes a stream loop's tiles",
                    f"take {arg.id} in a body, or iterate its pages with "
                    f"`for tile in km.{spelling('stream')}(...)`",
                )

    def take_held(self, node, name, value, where, bound):
        """A held stream's block for a body on compute's three: `name_buffer` its
        buffer, `name` its first page there (0)."""
        if where != "compute":
            self.refuse(
                node,
                f"{name} is a block of compute's three CPUs: each waits for it and "
                "frees it",
                "take it in a body on compute's three (no on=)",
            )
        bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {value.cxx};")
        bound.append(f"const uint32_t {name} = 0;")
        return set()

    def take_page(self, node, name, value, bound, before, after):
        """A write stream's page for a body on mover B, after pack's body filled it
        outside a stream loop: mover B waits for the page, the body gets its L1
        address, and the movers write it after this body, not at pack's line."""
        if value.move is None or self.stream_loops:
            self.refuse(
                node,
                f"{name}'s page is mover B's after pack's body fills it, outside a "
                "stream loop",
                f"take {name} on pack first, then on mover B, outside the loop",
            )
        if not value.letters.startswith("B"):
            self.refuse(
                node,
                f"{name}'s block is not mover B's to write",
                'give the write stream pattern="B"',
            )
        if value.waited:
            self.refuse(
                node,
                f"{name}'s page is taken on mover B above",
                "take it in one body on mover B",
            )
        value.waited = True
        buffer, half = value.buffer.name, f"km_out_{name}"
        given = value.pattern if value.pattern in self.scope else None
        before.append(
            Line(
                f"const uint32_t {half} = "
                f"km::block_half({buffer}, {value.pattern}, 0);\n"
                f"if (!movement.wait_front({half}, {buffer}Block)) {{\n"
                "    return false;\n}\ninvalidate_l1_cache();",
                uses=pattern_uses(given),
                mover="B",
            )
        )
        bound.append(f"const uint32_t {name} = get_read_ptr({half});")
        after.append(Line(value.move.text, set(value.move.uses)))  # both movers
        value.move.text = f"// {name}: written after mover B's body below"

    def take_moved(self, node, name, value, bound):
        """A stream loop's name after the loop, for a body on a mover: what that mover
        moved in the loop's move (km::Moved), `name_tiles` and `name_cycles`."""
        moved, move = value.cxx, value.move
        if value.ended is not None:  # its km::Moved is declared inside the if or loop
            kind, line = value.ended
            self.refuse(
                node,
                f"{name} after its loop is what a mover moved, and the loop is inside "
                f"the {kind} at line {line}: what it moved ends with the {kind}",
                f"take it in a body inside the {kind}, after the loop",
            )
        if not move.text.startswith("km::Moved"):
            move.text = f"km::Moved {moved} = {{0, 0}};\n" + move.text.replace(
                "movement.move(", f"movement.move({moved}, ", 1
            )
        bound.append(f"const uint32_t {name}_tiles = {moved}.tiles;")
        bound.append(f"const uint32_t {name}_cycles = {moved}.cycles;")

    def page_wait(self, value, pages=1):
        """The opening of mover B's wait for the page(s) compute packs of `value`."""
        if value.kind == "reduced":
            return f"if (!km::reduce_result(movement, {value.buffer.name}, {pages})) {{"
        return f"if (!movement.wait_front({value.buffer.name}, {pages})) {{"

    def returned(self, node):
        """`return word` inside `if self.rank == self.leader:`: the command's result
        word, set by the leader's mover B."""
        last = self.definition.body[-1]
        if not (
            self.in_leader
            and isinstance(last, ast.If)
            and last.body
            and last.body[-1] is node
            and node.value is not None
        ):
            self.refuse(
                node,
                "a command's result word is the leader's, its last line",
                f"write `return word` last under `if self.{spelling('rank')} == "
                f"self.{spelling('leader')}:`, the handler's last line",
            )
        self.allow_words = True
        text = self.integer(node.value)
        self.allow_words = False
        self.method.returns = True
        self.compiled.results.add(self.method.name)
        return [Line(f"movement.result({text});", self.uses(node.value), mover="B")], []

    def drain(self, node, call):
        """`km.Stream(pages, ...).drain()`: the stream read and dropped, as a loop with
        an empty body."""
        loop = ast.For(
            target=ast.Name("_", ast.Store()),
            iter=call.func.value,
            body=[ast.Pass()],
            orelse=[],
        )
        ast.copy_location(loop, node)
        loop.end_lineno = node.end_lineno
        loop.body[0].lineno = loop.body[0].end_lineno = node.lineno
        return self.read_loop(loop, call.func.value)

    # Entries: a page's words as a table, on every CPU -------------------------------

    def entries_call(self, node):
        """`node` when it is `<a page range>.entries(...)`, else None."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == spelling("entries")
            and self.is_range(node.func.value)
        ):
            return node
        return None

    def entries_named(self, node):
        """The table of entries `node` names (`jobs` or `jobs[a:b]`), or None."""
        base = node.value if isinstance(node, ast.Subscript) else node
        if isinstance(base, ast.Name):
            value = self.scope.get(base.id)
            if value is not None and value.kind == "entries":
                return value
        return None

    def entries_table(self, node, call):
        """`pages.entries(Job, first=F, size=S, count=C)` (or fixed=N for count=):
        the one page read on every CPU, mover A's pushed to compute, mover B's its
        own; (the table's Value, movement nodes, compute nodes, compute's pop)."""
        if self.dynamic:
            self.refuse(
                node,
                "a dynamic handler is compute's lines alone: no mover reads a table",
                "read the table in a handler of another class",
            )
        cls = None
        if len(call.args) == 1 and isinstance(call.args[0], ast.Name):
            found = self.names.get(call.args[0].id)
            if (
                isinstance(found, type)
                and issubclass(found, front.Entry)
                and found is not front.Entry
            ):
                cls = found
        keys = {k.arg: k.value for k in call.keywords}
        first_key, size_key, count_key, fixed_key = (
            spelling(f"entries_{kind}") for kind in ("first", "size", "count", "fixed")
        )
        if (
            cls is None
            or set(keys) - {first_key, size_key, count_key, fixed_key}
            or size_key not in keys
            or (count_key in keys) == (fixed_key in keys)
        ):
            self.refuse(
                node,
                f"entries name their class (a km.{spelling('entry')} of the module), "
                f"their {size_key}= and either the word of their count ({count_key}=) "
                f"or their number ({fixed_key}=)",
                f"write {spelling('entries')}(Job, {first_key}=8, {size_key}=8, "
                f"{count_key}=0)",
            )
        size = self.constant_int(keys[size_key])
        first = self.constant_int(keys[first_key]) if first_key in keys else 0
        pages = self.range_of(call.func.value)
        self.pages_only(node, pages)
        if pages.count != "1":
            self.refuse(node, "a table of entries is one page", "read pages(i, 1)")
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        words = page_bytes // 4
        if size < len(cls.words):
            self.refuse(
                node,
                f"{cls.__name__} names {len(cls.words)} words, and an entry is "
                f"{size}",
                f"give {size_key}= at least {len(cls.words)}",
            )
        most = (words - first) // size if words > first else 0
        if most < 1:
            self.refuse(
                node,
                f"a page of {words} words holds no entry of {size} from word {first}",
                "give a first word and a size inside the page",
            )
        if count_key in keys:
            word = self.constant_int(keys[count_key])
            if word >= words or first <= word < first + most * size:
                self.refuse(
                    node,
                    f"the count's word {word} is not a word of the page before or "
                    f"after its entries (words {first} to {first + most * size - 1} "
                    f"of {words})",
                    f"give {count_key}= a word outside the entries",
                )
        else:
            fixed = self.constant_int(keys[fixed_key])
            if not 1 <= fixed <= most:
                self.refuse(
                    node,
                    f"{fixed} entries in a page that holds {most}",
                    f"give {fixed_key}= 1 to {most}",
                )
        known = self.compiled.entries.setdefault(cls.__name__, cls.words)
        if known != cls.words:
            self.refuse(node, f"two entry classes named {cls.__name__}", "name one")
        number = self.tables
        self.tables += 1
        name = f"km_entries{number}"
        what = f"line {node.lineno}: {self.text(node)}"
        pushed = self.other_buffer(
            f"kEntries{number}", "word", dtype, page_bytes, 1, f"{what} (mover A's, "
            "to compute)",
        )  # fmt: skip
        own = self.other_buffer(
            f"kEntries{number}B", "word", dtype, page_bytes, 1, f"{what} (mover B's)"
        )
        for buffer in (pushed, own):
            buffer.page_bytes = max(buffer.page_bytes, page_bytes)
        tensor = f"{camel(pages.tensor)}({pages.tensor})"

        def read(buffer, reserve):
            opening = f"!movement.reserve({buffer}, 1) ||\n    " if reserve else ""
            return (
                f"if ({opening}!km::read_pages(movement, command, {tensor}, "
                f"{pages.first}, 1,\n                    get_write_ptr({buffer}))) "
                "{\n    return false;\n}\n"
                "noc_async_read_barrier();\n"
                "invalidate_l1_cache();\n"
                f"const volatile tt_l1_ptr uint32_t* {name} =\n"
                f"    km::l1(get_write_ptr({buffer}));"
            )

        uses = range_uses(pages, self)
        movement = [
            Line(
                read(pushed.name, True) + f"\ncb_push_back({pushed.name}, 1);",
                uses,
                mover="A",
            ),
            Glue(read(own.name, False), {name}, uses, mover="B"),
        ]
        compute = [
            Line(
                f"if (!compute.wait_front({pushed.name}, 1)) {{\n    return;\n}}\n"
                f"const volatile tt_l1_ptr uint32_t* {name} =\n"
                f"    km::l1(get_tile_address({pushed.name}, 0));\n"
                "invalidate_l1_cache();"
            )
        ]
        if count_key in keys:
            count = f"{name}_count"
            text = (
                f"const uint32_t {count} =\n"
                f"    {name}[{word}] <= {most} ? {name}[{word}] : 0;"
            )
            movement.append(Glue(text, {count}, {name}))
            compute.append(Glue(text, {count}, {name}))
            self.scope[count] = Value("int", count)  # generated: read by text_uses
        else:
            count = str(fixed)
        self.moved(node, pages.tensor, "table read")
        self.compiled.includes.add("ask")
        self.wait(node, "mover A", WAIT_DATA)
        self.wait(node, "mover B", WAIT_DATA)
        self.wait(node, "compute", WAIT_INPUT)
        table = {"class": cls, "first": first, "size": size, "count": count}
        pop = Line(f"cb_pop_front({pushed.name}, 1);")
        return Value("entries", name, table=table), movement, compute, pop

    def entries_made(self, node, name, call):
        """`jobs = pages.entries(...)` at the handler's top: the table, read at this
        line on every CPU and kept to the handler's end (compute frees its page)."""
        if len(self.blocks) != 1:
            self.refuse(
                node,
                "a named table is read at the handler's top and kept to its end",
                f"name it before the loops and ifs, or loop over "
                f"pages.{spelling('entries')}(...) itself",
            )
        self.check_name(node, name)
        value, movement, compute, pop = self.entries_table(node, call)
        self.scope[name] = value
        self.closing.append(pop)
        return movement, compute

    def entries_loop(self, node):
        """`for e in pages.entries(...)`, `for e in jobs` or `for e in jobs[a:b]`:
        the same loop on each CPU, its entry's words ints of each (`e.word`); a
        slice's ends clamped to the count, as Python's; one entry known at build is
        a block of no loop."""
        iterator = node.iter
        call = self.entries_call(iterator)
        lower = upper = pop = None
        if call is not None:
            table, movement, compute, pop = self.entries_table(node, call)
        else:
            table = self.entries_named(iterator)
            movement, compute = [], []
            if isinstance(iterator, ast.Subscript):
                part = iterator.slice
                if not isinstance(part, ast.Slice) or part.step is not None:
                    self.refuse(
                        node,
                        "a part of a table is a slice of its entries, with no step",
                        f"write for e in {iterator.value.id}[a:b]",
                    )
                lower, upper = part.lower, part.upper
        if not isinstance(node.target, ast.Name):
            self.refuse(node, "one name an entry", "write for e in table")
        name = node.target.id
        self.check_name(node, name)
        if name in self.scope:
            self.refuse(node, f"{name} is already defined", "use a new name")
        info, source = table.table, table.cxx
        count = info["count"]
        uses = {source} | text_uses(count, self)

        def clamped(bound):
            text = self.integer(bound)
            uses.update(self.uses(bound))
            return f"({text} < {count} ? {text} : {count})"

        low = "0" if lower is None else clamped(lower)
        high = count if upper is None else clamped(upper)
        index = f"km_{name}_at"
        cls = info["class"]
        self.scope[name] = Value("entry", name, table=info)
        for word in cls.words:  # generated: read by text_uses in ranges' C++
            self.scope[f"km_{name}_{word}"] = Value("int", f"km_{name}_{word}")
        loop = {"name": index, "first": low, "end": high, "casts": []}
        self.loops.append(loop)
        before = set(self.scope)
        body_m, body_c = self.statements(node.body)
        self.scoped("loop", node, before)
        self.loops.pop()
        self.scope.pop(name)
        for word in cls.words:
            self.scope.pop(f"km_{name}_{word}")
        mover = None
        if call is not None:  # a table read at this line: only where it is used
            movement, compute, pop, mover = self.entries_users(
                movement, compute, pop, body_m, body_c
            )
        alone = count == "1" and lower is None and upper is None
        offset = (
            str(info["first"])
            if alone
            else (f"{info['first']} + {info['size']} * {index}")
        )
        entry = [
            Glue(
                f"const volatile tt_l1_ptr uint32_t* {name} = {source} + {offset};",
                {name},
                {source},
            )
        ]
        entry += [
            Glue(
                f"const uint32_t km_{name}_{word} = "
                f"{name}[k{cls.__name__}{camel(word)}];",
                {f"km_{name}_{word}"},
                {name},
            )
            for word in cls.words
        ]
        head = head_c = "{"
        if not alone:
            head = f"for (uint32_t {index} = {low}; {index} < {high}; {index}++) {{"
            head_c = head
            if self.compute_waits:
                head_c = (
                    f"for (uint32_t {index} = {low}; km_ok && {index} < {high}; "
                    f"{index}++) {{"
                )
        if body_m:
            movement.append(Block(head, "}", [*entry, *body_m], uses, mover=mover))
        if body_c:
            compute.append(Block(head_c, "}", [*entry, *body_c], uses | {"km_ok"}))
        if pop is not None:
            compute.append(pop)
        for broadcast in loop["casts"]:
            m, c = self.broadcast_end(broadcast, high, uses)
            movement += m
            compute += c
        m, c = self.counted_ends(loop)
        movement, compute = movement + m, compute + c
        movement += self.fed_moves(node, loop, body_m)
        return movement, compute

    def entries_users(self, movement, compute, pop, body_m, body_c):
        """A table's read kept on the CPUs whose lines in its loop use it: mover B
        reads its own copy, mover A one for itself and compute; (movement, compute,
        compute's pop, the loop's mover: "A", "B" or None for both)."""

        def mover_has_line(nodes, mover, inherited=None):
            for node in nodes:
                own = node.mover or inherited
                if isinstance(node, Line) and own in (None, mover):
                    return True
                if (
                    isinstance(node, Block | Group)
                    and own in (None, mover)
                    and mover_has_line(node.children, mover, own)
                ):
                    return True
            return False

        on_compute = any(has_line(node) for node in body_c)
        on_a = on_compute or mover_has_line(body_m, "A")
        on_b = mover_has_line(body_m, "B")
        if on_a == on_b:  # both movers (or no line at all): the read as it is
            return movement, compute, pop, None
        mover = "A" if on_a else "B"
        kept = []
        for node in movement:
            if node.mover not in (None, mover):
                continue
            node.mover = mover
            kept.append(node)
        if not on_compute:
            compute, pop = [], None
        return kept, compute, pop, mover

    def entries_word(self, node, value):
        """The C++ of `e.word` (a word of an entry) or `table.count`."""
        cls = value.table["class"]
        if value.kind == "entries":
            if node.attr != spelling("entries_count"):
                self.refuse(
                    node,
                    f"a table has its {spelling('entries_count')} and its entries",
                    f"write {node.value.id}.{spelling('entries_count')}, or loop "
                    f"over {node.value.id}",
                )
            return value.table["count"]
        if node.attr not in cls.words:
            self.refuse(
                node,
                f"{node.value.id}, a {cls.__name__}, has no word {node.attr}",
                f"use one of {', '.join(cls.words)}",
            )
        return f"km_{node.value.id}_{node.attr}"

    def take_entry(self, node, name, value, bound):
        """An entry a body takes: each of its words as `name_word` (a table: its
        count as `name_count`)."""
        if value.kind == "entries":
            pairs = [(f"{name}_{spelling('entries_count')}", value.table["count"])]
        else:
            pairs = [
                (f"{name}_{word}", f"km_{name}_{word}")
                for word in value.table["class"].words
            ]
        for given, text in pairs:
            if given in self.scope:
                self.refuse(
                    node,
                    f"the body's {given} is a name of the handler too",
                    "rename one",
                )
            bound.append(f"const uint32_t {given} = {text};")
        return {text for _, text in pairs} | {value.cxx}


def _loops(count):
    """A loop depth in words, for a refusal."""
    return "no loop" if count == 0 else "1 loop" if count == 1 else f"{count} loops"


def function_line(function):
    """`line N of file` of a tile function's def."""
    code = function.function.__code__
    return f"line {code.co_firstlineno} of {Path(code.co_filename).name}"


def is_product(function):
    """Whether a tile function is `return a @ b` of its two parameters, either one
    marked km.srcb(...) or km.srca(...)."""
    tree = source_of(function)[0]
    body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    names = [a.arg for a in tree.args.args]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or len(names) != 2:
        return False
    value = body[0].value
    return (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.MatMult)
        and [marked_operand(value.left)[0], marked_operand(value.right)[0]] == names
    )


def traced_results(ir, out, t, values):  # t: a tile, or (a row, slice(None))
    """Each result of a tile function's trace stored into its own output tensor (the
    first into `out`, the others added to the IR as result1, result2, ...)."""
    for k, value in enumerate(values):
        if k == 0:
            trace.store(out.tiles[t], value)
            continue
        result = TensorParam(f"result{k}", True, F32)
        ir.tensors.append(result)
        trace.store(trace.TensorProxy(result).tiles[t], value)


def chunk_packs(unit, outs, tile, chunk, indent):
    """A wide row's chunk packed: result k's two Dst tiles from unit.chunk_results[k]
    into its stream's half (outs[k]: its out and room)."""
    lines = []
    for base, (out, room) in zip(unit.chunk_results, outs):
        lines += [
            f"if ({room}) {{",
            f"    KM_PACK_FORMAT({out});",
            f"    pack_tile<true>({base}, {out}, {tile} + 2 * {chunk});",
            f"    pack_tile<true>({base + 1}, {out}, {tile} + 2 * {chunk} + 1);",
            "}",
        ]
    return [indent + x for x in lines]


def value_packed(cxx, buffer, held):
    """A row's per-row value (its part `value`, Dst 0) packed into its field: the
    last write's page freed first."""
    return [
        f"if ({held}) {{",
        f"    cb_pop_front({buffer}, {buffer}Pages);",
        "}",
        f"MATH((km_tile::{cxx}_value(0)));",
        *pack(0, buffer).split("\n"),
        f"{held} = km_ok;",
    ]


def field_copies(compiled, fields, rows, index):
    """A wide row's field tiles into DST 4 + k, tile `index` of each: a field of
    another format than the values' fp32 page unpacked in its own setup (a bf16
    field: half the L1), the page's again after."""
    lines, now = [], rows
    for k, f in enumerate(fields):
        cb = f"kField{camel(f)}"
        own = cb if compiled.fields[f].format.dtype != "float32" else rows
        if own != now:
            lines += [f"KM_UNPACK_FORMAT_A({own});", f"copy_init({own});"]
            now = own
        lines.append(f"copy_tile({cb}, {index}, {4 + k});")
    return lines


def chunk_rows_buffer(compiled, cxx, node, pages=1):
    """The page a wide row's per-row values go through between its two parts (one
    a tile function), or `pages` (a row read in passes: a pair of the values' and
    the sums' tiles a row): pushed by the pack CPU, read back by the unpacker, fp32."""
    name = f"kChunkRows{camel(cxx)}"
    for buffer in compiled.buffers:  # a row read in passes holds its rows' pairs
        if buffer.name == name and buffer.pages[0] < pages:
            buffer.pages = (pages,)
    if all(b.name != name for b in compiled.buffers):
        compiled.buffers.append(
            Buffer(
                name,
                "field",
                next_index(compiled),
                None,
                "float32",
                4096,
                (pages,),
                1,
                f"the per-row values of {cxx}'s wide rows (line {node.lineno})",
            )
        )
        compiled.unpack_fp32.add(name)
    return name


def row_broadcast(item, field, slot, dst, copied, along="ROW"):
    """The unpack of a per-column operand into DST tile `dst`: row 0 of its tile (a
    field's, or the loop tile of `item`) copied to every row (a per-row operand,
    `along` "COL": column 0 to every column), then the copy's setup again for the
    call's next tiles."""
    if field is not None:
        cb, tile = f"kField{camel(field)}", "0"
        formats = cb
    else:  # the block's half; the formats of half A (the pair's one format)
        cb, formats = item.cxx, f"{item.buffer.name}.mover_a"
        tile = item.first if slot is None else f"{item.first} + {slot}"
    lines = [  # a 16-bit tile goes through SrcB (an fp32 one unpacked to DST)
        f"KM_UNPACK_FORMATS({formats}, {formats});",
        f"unary_bcast_init<BroadcastType::{along}>({cb});",
        f"unary_bcast<BroadcastType::{along}>({cb}, {tile}, {dst});",
        f"unary_bcast_uninit<BroadcastType::{along}>({cb});",
    ]
    if copied:
        back = f"{copied[0].buffer.name}.mover_a"
        lines += [f"KM_UNPACK_FORMAT_A({back});", f"copy_init({back});"]
    return lines


def out_names(number, write, state):
    """The C++ names of a loop's output half and room for write stream `write`: one
    a loop, or the stream's own when results of one call go to several streams."""
    if write.cxx in state.get("own_out", ()):
        return f"{write.cxx}_out", f"{write.cxx}_room"
    return f"km_out{number}", f"km_room{number}"


def matrix_product(function):
    """The line of the first `@` in a function's source, or None."""
    tree, _, _ = source_of(function)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return node.lineno
    return None


def held_move(variable, half, buffer, count):
    """The movers' lines of a held stream: its one block moved, or with no pages an
    empty block pushed unread by mover A, block 0's (compute waits for one block a
    stream, as a hand reader pushes a job's Q block of no rows)."""
    return [
        Line(
            f"if (({count}) == 0) {{\n"
            f"    if (!movement.reserve({buffer}.mover_a, {buffer}Block)) {{\n"
            "        return false;\n"
            "    }\n"
            f"    cb_push_back({buffer}.mover_a, {buffer}Block);\n"
            f"}} else if (!movement.move({variable})) {{\n"
            "    return false;\n"
            "}",
            {variable},
            mover="A",
        ),
        Line(
            f"if (({count}) != 0 && !movement.move({variable})) {{\n"
            "    return false;\n"
            "}",
            {variable},
            mover="B",
        ),
    ]


def known_count(text):
    """A count's C++ text of integer constants (`(2 * 32)`) as an int, else None."""
    ops = {ast.Add: int.__add__, ast.Sub: int.__sub__, ast.Mult: int.__mul__}
    ops[ast.Div] = int.__floordiv__

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](value(node.left), value(node.right))
        raise ValueError(node)

    try:
        return value(ast.parse(str(text), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None


def held_most(pages):
    """The most pages of a range where the build knows it: a count known, or a
    rectangle's sides' most (Value.bounds), else None."""
    if pages.count.isdigit():
        return int(pages.count)
    bounds = getattr(pages, "bounds", None)
    if bounds and all(isinstance(side, int) for side in bounds):
        return bounds[0] * bounds[1]
    return None


def block_step(rows, row_step):
    """The tensor pages from one block of `rows` rows to the next: an int, or C++ for a
    row step that is a word."""
    return rows * row_step if isinstance(row_step, int) else f"{rows} * {row_step}"


def range_uses(pages, compiler, tensor=True):
    """The names a range's C++ reads (`tensor`: its tensor's word too)."""
    found = {pages.tensor} if pages.tensor and tensor else set()
    if pages.rect is not None:  # a rectangle's rows and their step
        found |= text_uses(f"{pages.rect[0]} {pages.rect[2]}", compiler)
    return found | text_uses(pages.first, compiler) | text_uses(pages.count, compiler)


def text_uses(text, compiler):
    """The names of the handler a piece of generated C++ reads."""
    found = set()
    for name in re.findall(r"[A-Za-z_]\w*", text):
        if name in compiler.scope or name in GENERATED_VALUES:
            found.add(name)
        if name in compiler.counts.values():
            found.add(name)
        if name in getattr(compiler, "column_words", {}).values():  # a width word
            found.add(name)
    return found


GENERATED_VALUES = (
    "km_rank",
    "km_size",
    "km_group",
    "km_self_x",
    "km_self_y",
    "km_self_core",
    "km_rect_columns",
    "km_rect_rows",
)


def pattern_uses(pattern):
    return {pattern} if pattern else set()


def add_tile(buffer, page, dst, scratch):
    """DST dst += page `page` of buffer (copied to DST scratch, added by the SFPU)."""
    return (
        f"KM_UNPACK_FORMAT_A({buffer});\n"
        f"copy_init({buffer});\n"
        f"copy_tile({buffer}, {page}, {scratch});\n"
        "add_binary_tile_init();\n"
        f"add_binary_tile({dst}, {scratch}, {dst});"
    )


def pack(dst, half, width=1):
    """DST tiles `dst` .. `dst + width - 1` packed into pages of `half`; DST
    released."""
    packs = "".join(f"    pack_tile({dst + t}, {half});\n" for t in range(width))
    return (
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"if (km_ok && compute.reserve_back({half}, {width})) {{\n"
        f"    KM_PACK_FORMAT({half});\n"
        f"{packs}"
        f"    cb_push_back({half}, {width});\n"
        "}\n"
        "tile_regs_release();"
    )


def camel(name):
    """The C++ type name of a tensor parameter (source: Source)."""
    return "".join(part.capitalize() for part in name.split("_"))


OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.FloorDiv: "/",
    ast.Mod: "%",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
}
COMPARISONS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}
CXX_KEYWORDS = {
    "auto",
    "bool",
    "case",
    "char",
    "const",
    "default",
    "delete",
    "do",
    "double",
    "enum",
    "float",
    "goto",
    "int",
    "long",
    "new",
    "private",
    "public",
    "register",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "template",
    "this",
    "throw",
    "typedef",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "km",
    "uint32_t",
}


def product_setups(first, init):
    """A loop's later product of sums whose setup (`init`: SrcA's and SrcB's formats,
    the block's shape, a turn) is not the loop's first product's (`first`, set once
    before the blocks): the lines that set its own before its K tiles and the first's
    again after them; none when they are one."""
    if first is None or first == init:
        return "", ""
    own = "".join(f"    {line}\n" for line in init.split("\n"))
    return own, "".join(f"    {line}\n" for line in first.split("\n"))


def sums_room(field_):
    """The name of pack's own word that it holds a field of sums' pages: only pack's
    reserve decides it (unpack's and math's is always true), so it guards the pack
    lines alone and never a branch all three compute threads take."""
    return f"km_room_sums_{field_}"


def sums_room_line(field_):
    """A field of sums' room, declared beside its count of sums."""
    return f"bool {sums_room(field_)} = true;  // pack holds self.{field_}'s pages"


def sums_reserve(sums, room, buffer):
    """A new sum holds the field's pages: pack's reserve sets the room, and a give-up
    leaves it false for the command, so the three threads still run the sections
    and the destination's hand-over together while pack packs nothing."""
    return (
        f"if (km_ok && {sums} == 0) {{  // a new sum: the field's pages held\n"
        f"    {room} = {room} && compute.reserve_back({buffer}, {buffer}Pages);\n"
        "}\n"
    )


def sent_field_acked(buffer, group, form):
    """A field of sums is emptied by compute's unpack on the group's root and by mover
    B's send elsewhere; unpack keeps its own count of the pages taken, so a core that
    sent counts mover B's pop there too, or its next turn as root rewinds the count
    (the hand's K split does the same). The field is one whole item of its buffer."""
    return (
        f"if (km_ok && !km::reduce_send({group}, {form}).root) {{  // mover B pops it\n"
        f"    UNPACK((get_local_cb_interface({buffer}).tiles_acked += "
        f"{buffer}Pages));\n"
        "}\n"
    )
