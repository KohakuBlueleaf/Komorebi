"""The compiler of a program written as a class, level 1: each handler's body, read
with `ast`, becomes the nodes of the movement source (both movers) and of the compute
kernel; streams become buffer pairs, parameters command words, a refusal names the
line. `komorebi.dsl.program` writes the files and makes the Program.
"""

import ast
import importlib
import inspect
import keyword
import math
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from komorebi.dsl import front
from komorebi.dsl.names import kind_of, spelling
from komorebi.dsl.nodes import Block, Glue, Group, Line
from komorebi.dsl.tile import trace, unit_program
from komorebi.dsl.tile.ir import BF16, BLOCK_B, F32, KernelIR, PlanError, TensorParam
from komorebi.dsl.tile.rows import row_program
from komorebi.dsl.tile.unit import UnitError
from komorebi.errors import ProgramError
from komorebi.movement import Pattern

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
PAGE_BYTES = {"bfloat16": 2048, "float32": 4096}
TILE_ELEMENTS = 32 * 32
# Every local the generated code declares starts with GENERATED_PREFIX, which a
# parameter or a variable may not; the runtime's names bodies use are taken too.
GENERATED_PREFIX = "km_"
GENERATED = {"command", "movement", "compute"}
# The runtime's where names (komorebi.state) of each kind of wait.
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
TILE_FORMATS = {"float32": F32, "bfloat16": BF16, **{b: b for b in BLOCK_B}}
# An inline body's CPU (on=): a mover, one compute CPU, or compute's three; the #if of
# each one CPU's code.
COMPUTE_CPUS = ("unpack", "math", "pack")
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
    # handler -> (the size of its reduce's groups, the reduce's line)
    reduce_groups: dict = field(default_factory=dict)
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
    includes: set = field(default_factory=set)  # "port", "broadcast", "fill", "add"
    # The #include lines of inline bodies, by kernel ("movement", "compute").
    body_includes: dict = field(default_factory=lambda: {"movement": [], "compute": []})
    waits: list = field(default_factory=list)
    unpack_fp32: set = field(default_factory=set)  # buffer names compute copies fp32
    summed_fields: set = field(default_factory=set)  # fields of sums: compute packs
    broadcasts: list = field(default_factory=list)  # (along, buffer, window)
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
    startup: object = None  # the class's km.Startup, or None (the default start-up)


def handler_methods(cls):
    """The class's handler methods in the order of their definitions."""
    found = [
        value for value in vars(cls).values() if isinstance(value, front.HandlerMethod)
    ]
    return sorted(found, key=lambda method: method.function.__code__.co_firstlineno)


def compile_class(cls, plan, tensors, l1=None):
    """Compiles each handler of `cls`; `plan` the program's sizes by name, `tensors`
    an example tensor of each tensor parameter (its format and page), `l1` the bytes
    left for buffers (a field's unknown symbol takes what the others leave)."""
    methods = handler_methods(cls)
    compiled = Compiled(cls.__name__, inspect.getsourcefile(methods[0].function))
    compiled.fields = {
        name: front.Field(
            value.format,
            tuple(n.name if isinstance(n, front.Symbol) else n for n in value.shape),
        )
        for name, value in inspect.get_annotations(cls).items()
        if isinstance(value, front.Field)
    }
    compiled.own = {
        name: cls.__dict__[name].push
        for name in compiled.fields
        if isinstance(cls.__dict__.get(name), front.Own)
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
    for number, method in enumerate(methods, start=1):
        compiled.handlers[method.name] = number
        if method.overlay:
            compiled.overlays[method.name] = len(compiled.overlays) + 1
        names = method.function.__globals__
        HandlerCompiler(compiled, method, names, plan, tensors).run()
    bind_fields(compiled, plan, l1)
    compiled.compute_options = class_compute(cls, compiled)
    compiled.startup = getattr(cls, "startup_given", None)
    return compiled


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
        flags = importlib.import_module("komorebi.session").COMPUTE_FLAGS
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


def tile_format_refusal(formats):
    """None, or why a tile function is refused tiles of `formats`: its code runs on
    an fp32 Dst, which reads the A family wrong (the card: each datum 2^-15 off)."""
    wrong = [f for f in formats if f not in TILE_FORMATS]
    if not wrong:
        return None
    return (
        f"a tile function takes tiles of float32 or the B family ("
        f"{', '.join(TILE_FORMATS)}), not {', '.join(wrong)}: its code runs on an "
        "fp32 Dst, which the card fills wrong from the A family"
    )


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
        self.reads = 0
        self.writes = 0
        self.arrivals = 0
        self.dst_next = 0
        self.dst_live = None  # the accumulating tile's name while DST is held
        self.compute_waits = False
        self.leader_writes = set()  # reduced values written by the leader
        self.reduced = []
        self.in_leader = False
        self.loops = []  # the range loops around the statement, innermost last
        self.movers = {}  # mover -> the broadcast it runs
        self.allow_words = False  # a body's word is read by the return only
        self.field_line = False  # a word field's words are read in a line setting one
        self.closing = []  # compute's lines after the handler's (a product's field)
        self.opening = []  # compute's lines before them (a field's sums count)
        self.opening_b = []  # mover B's (the same count)
        self.summed = {}  # a field of sums -> its counter's C++ name
        self.filled = {}  # field -> (its fill's node, its block) until a body takes it
        self.waited_fields = set()  # fields compute waited for (a product's, once)
        self.leader_text = "0"  # self.leader: rank 0, or a reduce group's first rank
        self.stream_loops = []  # the numbers of the read loops around the statement
        # Per read loop around the statement: its tile function, its other tile
        # lines, and the lines compute runs once before its blocks.
        self.loop_states = []
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
        self.bind_parameters(movement, compute)
        spans = None
        if self.method.stamps is not None and not self.method.overlay:
            spans = self.tile_stamps()
        elif self.method.stamps is not None:
            self.stamps()
        if not self.dynamic:
            compute.append(Glue("bool km_ok = true;", {"km_ok"}))
        if spans is not None:
            compute.append(spans[0])
        body_m, body_c = self.statements(self.definition.body)
        movement += self.opening_b + body_m
        compute += self.opening + body_c
        if spans is not None:
            if not any(call[0] == name for call in self.compiled.tile_calls):
                self.refuse(
                    self.definition.decorator_list[0],
                    f"stamps={self.method.stamps!r} of a resident handler are the "
                    "cycles of its tile function calls, and it calls none",
                    "call a tile function, or write @km.handler(overlay=True, "
                    'stamps="t")',
                )
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
        self.spans_most = page_bytes // 4 - SPANS_FIRST
        self.moved(decorator, stamps, "page write")
        self.wait(decorator, "mover B", WAIT_DATA)
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
                    "}"
                )
            ],
        )
        closing = Group(
            f"stamps: the page to mover B, for {stamps} page rank",
            [
                Line(
                    f"if (km_ok && compute.reserve_back({buffer.name}, 1)) {{\n"
                    f"    cb_push_back({buffer.name}, 1);\n"
                    "}",
                    {"km_ok"},
                )
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
                self.construct(node) in ("stream", "reduce", "broadcast")
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
        return [], [Line(f"km_messages.{kind}({rank}, {message.id}{handler});", uses)]

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
        if args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
            self.refuse(
                self.definition,
                "a handler's parameters are plain, one a command word",
                "write `def name(self, a: km.Tensor, b: int, ...)`",
            )
        parameters = args.args[1:]
        words = []
        for word, parameter in enumerate(parameters):
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
            elif kind == "pattern":
                self.scope[param.name] = Value("pattern", param.name)
                text = (
                    f"const km::Pattern {param.name} = "
                    f"km::Pattern::of(command.a{word});"
                )
            else:
                self.scope[param.name] = Value("int", param.name)
                text = f"const uint32_t {param.name} = command.a{word};"
            for nodes in (movement, compute):
                nodes.append(Glue(text, {param.name}))
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
        if len(words) > COMMAND_WORDS:
            self.refuse(
                self.definition,
                f"{len(words)} command words ({', '.join(n for n, _ in words)}); a "
                f"command has {COMMAND_WORDS}",
                "pass fewer arguments, or a tensor of them",
            )
        self.method.words = words
        self.compiled.words[self.method.name] = words

    def parameter_kind(self, parameter):
        annotation = parameter.annotation
        if annotation is None:
            self.refuse(
                self.definition,
                f"parameter {parameter.arg} has no type",
                f"write `{parameter.arg}: km.{spelling('tensor')}`, `: int` or "
                f"`: km.{spelling('pattern')}`",
            )
        if isinstance(annotation, ast.Name) and annotation.id == "int":
            return "int"
        if isinstance(annotation, ast.Attribute):
            kind = kind_of(annotation.attr, ("tensor", "pattern"))
            if kind:
                return kind
        self.refuse(
            self.definition,
            f"parameter {parameter.arg}: a type the command words cannot carry",
            f"a parameter is km.{spelling('tensor')}, int or km.{spelling('pattern')}",
        )

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
        call = self.construct(value)
        if call == "partition":
            return self.partition(node, name, value)
        if call == "zeros":
            return self.zeros(node, name, value)
        if call == "stream" and not self.write_keyword(value):
            return self.held_stream(node, name, value)
        if call == "stream":
            return self.write_stream(node, name, value)
        if call == "reduce":
            return self.reduce(node, name, value)
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
                return kind_of(func.attr, ("stream", "reduce", "broadcast", "cxx"))
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
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self.constant(node, node.value)  # a symbol: the plan's
        if isinstance(node, ast.Name) and node.id not in self.scope:
            return self.constant(node, node.id)
        self.refuse(node, "a size the build needs", "give an int, or a plan name")

    def attribute(self, node):
        base = node.value
        if isinstance(base, ast.Name) and base.id == "self":
            kind = kind_of(node.attr, ("rank", "size", "leader", "x", "y", "core"))
            if kind == "core":
                return "km_self_core"
            if kind == "rank":
                return "km_rank"
            if kind == "size":
                return "km_size"
            if kind == "leader":
                return self.leader_text
            if kind in ("x", "y"):
                self.compiled.rectangle = True
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
        used = set()
        for side in sides:
            used |= self.uses(side.lower) | self.uses(side.upper)
        for part in (sides[0].step,) if step is not None else ():
            used |= self.uses(part)
        if step is None and (rows[0] == 1 or columns[0] == columns_of):  # one run
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
        if known is not None:
            return known * columns_of
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
        if self.dst_live is not None:
            self.refuse(
                node,
                f"{self.dst_live} still holds DST; level 1 keeps one tile there",
                "write it, or reduce it, first",
            )
        count = self.constant_int(call.args[1]) if len(call.args) > 1 else 1
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
        self.dst_live = name
        self.scope[name] = Value("tile", name, dst=dst, width=count)
        self.compiled.includes |= {"fill"}
        fills = "\n".join(f"fill_tile({dst + t}, 0.0f);" for t in range(count))
        text = f"tile_regs_acquire();\nfill_tile_init();\n{fills}"
        return [], [Line(text)]

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
        if not self.loop_states or len(call.args) != 2 or call.keywords:
            self.refuse(
                node,
                "a product takes a block of the loop's stream and a field",
                f"write acc += self.{function.name}(block, self.w) in "
                f"`for block in km.{spelling('stream')}(...)`",
            )
        state = self.loop_states[-1]
        item = isinstance(call.args[0], ast.Name) and self.scope.get(call.args[0].id)
        field_ = self.field_name(call.args[1])
        other = isinstance(call.args[1], ast.Name) and self.scope.get(call.args[1].id)
        if item and other and self.in_this_loop(item) and self.in_this_loop(other):
            return self.stream_product(node, value, function, item, other)
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
                    self.closing.append(Line(f"cb_pop_front({buffer}, {buffer}Pages);"))
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
                    self.closing.append(Line(f"cb_pop_front({buffer}, {buffer}Pages);"))
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
        blocks = []
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
            blocks.append(item.buffer)
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
        if self.placed(node, {}, function, left, right, None):
            self.refuse(
                node,
                f"{function.name} turns the product (its right operand in SrcB), and "
                "a product of two broadcasts' blocks is built in the plain placement "
                "alone",
                "drop the marker, or mark it the plain way: "
                f"`return km.{spelling('srcb')}(a) @ b`",
            )
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
        packs = "\n".join(
            f"        pack_tile<true>({i * width + j}, {buffer}, ({row} + {i}) * "
            f"{columns} + {column} + {j});"
            for i in range(height)
            for j in range(width)
        )
        text = (
            "km_ok = km_ok && km::broadcast_receive(compute, "
            f"{a.name}, {a.name}Pages) &&\n"
            f"    km::broadcast_receive(compute, {b.name}, {b.name}Pages);\n"
            f"if (km_ok && {sums} == 0) {{  // a new sum: the field's pages held\n"
            f"    km_ok = compute.reserve_back({buffer}, {buffer}Pages);\n"
            "}\n"
            "if (km_ok) {\n"
            f"    KM_UNPACK_FORMATS({b.name}, {a.name});\n"
            f"    matmul_block_init({a.name}, {b.name}, 0, {width}, {height}, "
            f"{inner});\n"
            f"    KM_PACK_FORMAT({buffer});\n"
            f"    pack_reconfig_l1_acc({sums} > 0 ? 1 : 0);\n"
            f"    for (uint32_t {row} = 0; {row} < {rows}; {row} += {height}) {{\n"
            f"        for (uint32_t {column} = 0; {column} < {columns}; {column} += "
            f"{width}) {{\n"
            "            tile_regs_acquire();\n"
            f"            for (uint32_t {k} = 0; {k} < {inner}; {k}++) {{\n"
            f"                matmul_block({a.name}, {b.name}, {row} * {inner} + {k}, "
            f"{k} * {columns} + {column}, 0, 0, {width}, {height}, {inner});\n"
            "            }\n"
            "            tile_regs_commit();\n"
            "            tile_regs_wait();\n"
            + "\n".join("    " + line for line in packs.split("\n"))
            + "\n            tile_regs_release();\n"
            "        }\n"
            "    }\n"
            "    pack_reconfig_l1_acc(0);\n"
            f"    {sums}++;\n"
            f"    km::broadcast_done({a.name}, {a.name}Pages);\n"
            f"    km::broadcast_done({b.name}, {b.name}Pages);\n"
            "}"
        )
        return [Line(f"{sums}++;", mover="B")], [Line(text, {"km_ok"})]

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
        state["sums"] = sums  # mover B counts the loop's blocks into it
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
            state["inits"].append(Line(f"{init}\nKM_PACK_FORMAT({buffer});"))
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
        packs = "\n".join(
            f"        pack_tile<true>({j}, {buffer}, {column} + {j});"
            for j in range(width)
        )
        text = (
            f"if (km_ok && {sums} == 0) {{  // a new sum: the field's pages held\n"
            f"    km_ok = compute.reserve_back({buffer}, {buffer}Pages);\n"
            "}\n"
            "if (km_ok) {\n"
            f"    pack_reconfig_l1_acc({sums} > 0 ? 1 : 0);\n"
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
            "    pack_reconfig_l1_acc(0);\n"
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
        for keyword_ in call.keywords:
            key = keyword_.arg
            if key == spelling("block"):
                block = self.constant_int(keyword_.value)
            elif key == spelling("row") and not write:
                self.row_width = self.constant_int(keyword_.value)
            elif key == "pattern":
                pattern, letters = self.pattern(keyword_.value)
            elif key == spelling("repeat"):
                repeat = self.integer(keyword_.value)
            elif key == spelling("write_stream"):
                continue
            else:
                self.refuse(node, f"a stream has no {key}=", "use block=, pattern=")
        block = self.row_width if block is None else block
        if block < 1 or block % self.row_width:
            self.refuse(
                node,
                "a block of no pages, or not of whole rows",
                "give block= a count, a multiple of row=",
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
        self.loops.append(loop)
        before = set(self.scope)
        body_m, body_c = self.statements(node.body)
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
                    "repeat": repeat,
                    "buffer": buffer,
                    "width": self.row_width,
                }
            )
        first = reads[0]
        rows_in_step = False  # blocks of whole rows, as many blocks: a product's
        for other in reads[1:]:
            if (other["block"], other["pages"].count, other["repeat"]) == (
                first["block"],
                first["pages"].count,
                first["repeat"],
            ):
                continue
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
                "repeat": reads[0]["repeat"],
                "line": node.lineno,
            }
        )
        body_m, body_c = self.statements(node.body)
        state = self.loop_states.pop()
        self.stream_loops.pop()
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
        if state.get("sums"):  # a sum a block: mover B counts them as compute does
            movement.append(
                Line(
                    f"{state['sums']} += ({first['pages'].count} + "
                    f"{first['buffer'].name}Block - 1) / {first['buffer'].name}Block;",
                    range_uses(first["pages"], self),
                    mover="B",
                )
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
            out, room = f"km_out{number}", f"km_room{number}"
            inner.append(
                Line(
                    f"const uint32_t {out} = km::block_half({write.buffer.name}, "
                    f"{write.pattern}, {block_var});\n"
                    f"const bool {room} = compute.reserve_back({out}, "
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
            out, room = f"km_out{number}", f"km_room{number}"
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
        if (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("fill")
            and self.field_name(func.value)
        ):
            return self.fill(node, call, self.field_name(func.value))
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
        construct = self.construct(call)
        if construct == "cxx":
            return self.inline(node, call)
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

    def tile_call(self, node, call, name):
        """`t = self.f(a, b)` (or pushed): a, b the loop's stream tiles (or a row of
        `row=` tiles and fields held as resident rows), copied into DST as f's unit
        program reads them, the emitted body on the math CPU, the result in DST 0 (a
        row: 0 .. width - 1). Returns (no movement, compute's lines)."""
        function = self.tile_function_of(call)
        self.held_not_traced(node, call, function)
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
        for arg in call.args:
            field = self.field_name(arg)
            if field is not None:
                operands.append((None, field))
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
        if fields and width == 1:
            self.refuse(
                node,
                f"self.{fields[0]} is a resident row of the row form: the function "
                "is called on a row of a stream of rows",
                f"read the stream with row=: `for row in km.{spelling('stream')}"
                "(..., row=2)`",
            )
        if self.dst_live is not None or self.dst_next != 0:
            self.refuse(
                node,
                "a tile function's result is DST 0, which another tile holds",
                "write or push that tile first",
            )
        formats = []
        for item, field in operands:
            if item is not None:
                formats.append(item.buffer.dtype)
            else:
                formats.append(self.compiled.fields[field].format.dtype)
        why = tile_format_refusal(formats)
        if why is not None:
            self.refuse(node, f"{function.name}: {why}", "give it tiles of those")
        unit, cxx = self.tile_unit(node, function, formats, width, fields)
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
        for a, w in products:
            why = product_refusal(a.buffer.dtype, w.buffer.dtype)
            if why is not None:
                self.refuse(
                    node,
                    f"{why}, in {function.name}",
                    "stream tensors of matrix formats of one family into it",
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
                state["after"].append(Line(f"cb_pop_front({buffer}, {buffer}Pages);"))
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
        count = len(unit.operands)
        slots = DST_HALF // count if width == 1 else 1
        number = self.stream_loops[-1]
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
                lines.append(
                    f"    UNPACK((llk_unpack_AB_matmul({a.cxx}, {w.cxx}, {a.first} + "
                    f"{slot}, {w.first} + {slot}, 1, 1, 1)));"
                )
            for dst, operand in enumerate(unit.operands):
                if operand is None:  # a product's Dst tile
                    continue
                item, _ = operands[parameters.index(operand)]
                lines.append(
                    f"    copy_tile({item.cxx}, {item.first} + {slot}, "
                    f"{count} * {slot} + {dst});"
                )
            call_ = [f"MATH((km_tile::{cxx}({count} * {slot})));"]
            lines += ["    " + line for line in self.timed_call(call_, timed)] + ["}"]
        elif width == 1:
            for a, w in products:  # unpacked in the unit's issue order
                lines.append(
                    f"UNPACK((llk_unpack_AB_matmul({a.cxx}, {w.cxx}, {a.first}, "
                    f"{w.first}, 1, 1, 1)));"
                )
            for dst, operand in enumerate(unit.operands):
                if operand is None:  # a product's Dst tile
                    continue
                item, _ = operands[parameters.index(operand)]
                lines.append(f"copy_tile({item.cxx}, {item.first}, {dst});")
        else:  # the row form: the row in DST 0 .. width - 1, the field's tile after
            item = streamed[0]
            lines += [
                f"copy_tile({item.cxx}, {item.first} + {t}, {t});" for t in range(width)
            ]
            for field in fields:
                lines.append(f"copy_tile(kField{camel(field)}, 0, {2 * width});")
        if slots == 1:
            lines += self.timed_call([f"MATH((km_tile::{cxx}(0)));"], timed)
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, self.text(node), function, cxx)
        )
        self.dst_next = 1
        held = name or "<tile function>"
        self.dst_live = held
        self.scope[held] = Value(
            "tile",
            held,
            dst=0,
            width=width,
            batch=(number, count) if slots > 1 else None,
        )
        return [], [Line("\n".join(lines))]

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

    def tile_unit(self, node, function, formats, width=1, fields=()):
        """The unit program of `function` on tiles of `formats` (traced, its rules
        applied, emitted at the write; `width` > 1: the row form over rows of that
        many tiles, `fields` its resident rows) and its C++ name; refused by name and
        line when the tile compiler does not lower it."""
        made = (function.function, tuple(formats), width)
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
        tensors = [
            TensorParam(p, False, TILE_FORMATS[f]) for p, f in zip(parameters, formats)
        ]
        result = TensorParam("result", True, F32)
        ir = KernelIR(cxx, [*tensors, result], [], {})
        held = len(parameters) - len(fields)  # the fields are the last parameters
        try:
            with trace.Tracer(ir):
                proxies = [trace.TensorProxy(t) for t in tensors]
                out = trace.TensorProxy(result)
                if width == 1:
                    for t in trace.grid(trace.num_tiles(proxies[0])):
                        loads = [trace.load(p.tiles[t]) for p in proxies]
                        trace.store(out.tiles[t], function.function(*loads))
                else:
                    rows = [trace.resident(p.tiles[0, :]) for p in proxies[held:]]
                    for t in trace.grid(trace.num_tile_rows(proxies[0])):
                        loads = [trace.load(p.tiles[t, :]) for p in proxies[:held]]
                        value = function.function(*loads, *rows)
                        trace.store(out.tiles[t, :], value)
            if width == 1:
                fidelity = function.fidelity or "HiFi4"  # its products'
                unit = unit_program(ir, load_macro=True, fidelity=fidelity)
            else:
                unit = row_program(ir, width, width, 2 * width)
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
        if len(call.args) == 1 and self.tile_function_of(call.args[0]) is not None:
            _, made = self.tile_call(node, call.args[0], None)
            _, pushed = self.push_made(
                node, stream, self.scope.pop("<tile function>"), target.id
            )
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

    def write(self, node, call):
        target = call.func.value
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
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            self.refuse(node, "write one tile", "write pages.write(total)")
        name = call.args[0].id
        value = self.scope.get(name)
        if value is not None and value.kind == "word":
            return self.write_word(node, pages, value)
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
            compute.append(Line(pack(value.dst, buffer.name, width), {"km_ok"}))
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
            if dtype != buffer.dtype:  # the sum packed again in the tensor's format
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
        value.waited = value.popped = True
        self.moved(node, pages.tensor, "page write")
        movement = [Line(text, range_uses(pages, self), mover="B")]
        self.compiled.includes.add("ask")
        self.wait(node, "mover B", WAIT_DATA)
        return movement, compute

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
            f"    cb_push_back({buffer}, {buffer}Pages);\n"
            f"    km_ok = compute.wait_front({buffer}, {buffer}Pages);\n"
            "}\n"
            f"if (km_ok && {sums} > 0) {{\n"
        )
        compute = (
            f"// self.{field_}'s sums, packed again in {dtype}\n"
            f"{taken}"
            f"    KM_UNPACK_FORMAT_A({buffer});\n"
            f"    copy_init({buffer});\n"
            f"    for (uint32_t {tile} = 0; km_ok && "
            f"{tile} < {rows * columns}; {tile} += "
            f"{step}) {{\n"
            "        tile_regs_acquire();\n"
            f"{copies}\n"
            "        tile_regs_commit();\n"
            "        tile_regs_wait();\n"
            f"        km_ok = compute.reserve_back({out.name}, {step});\n"
            "        if (km_ok) {\n"
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

    def producer(self, node, field, kind):
        """A field's pages are pushed by one producer: mover A's fill, or compute's
        adds of a product, in every handler of the class."""
        other = self.compiled.producers.setdefault(
            field, (kind, self.method.name, node.lineno, self.text(node))
        )
        if other[0] != kind:
            _, handler, line, text = other
            pusher = "mover A" if kind == "product" else "compute"
            self.refuse(
                node,
                f"self.{field} is pushed by {pusher}"
                f" too ({handler}, line {line}, `{text}`): a buffer has one producer",
                "fill it, or add products into it, not both: give each its own field",
            )

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

    def wait(self, node, cpu, where):
        self.compiled.waits.append(
            Wait(self.method.name, cpu, where, node.lineno, self.text(node))
        )

    # Reduce ------------------------------------------------------------------------

    def reduce(self, node, name, call):
        """`whole = km.reduce(total, form=form)`: total's n tiles summed over the
        group (by owner or by tree) at its rank 0, a slot of n tiles a sender;
        `km.reduce(self.part, ...)`: a field of sums, the same in L1."""
        field_ = self.field_name(call.args[0]) if len(call.args) == 1 else None
        if field_ is None:
            if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
                self.refuse(
                    node,
                    "reduce a tile, or a field of sums",
                    "write km.reduce(total, form=form) or km.reduce(self.part, "
                    "form=form)",
                )
            tile = self.scope.get(call.args[0].id)
            if tile is None or tile.kind != "tile" or self.dst_live != call.args[0].id:
                self.refuse(
                    node,
                    "reduce the tile held in DST",
                    "write km.reduce(total, form=form)",
                )
        form, size = None, None
        for keyword_ in call.keywords:
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
                    node, f"a reduce has no {keyword_.arg}=", "use form= and group="
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
        if field_ is not None:
            return self.field_reduce(
                node, name, field_, (form, form_uses), size, (group, uses_group)
            )
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
            slots = TREE_ROUNDS if size is None else max(1, (size - 1).bit_length())
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
        arrival = self.other_buffer(
            f"kArrival{number}",
            "arrival",
            "float32",
            PAGE_BYTES["float32"],
            slots * width,
            f"line {node.lineno}: {self.text(node)}",
        )
        arrival.block = width
        self.compiled.unpack_fp32.add(arrival.name)
        out = self.out_buffer(
            "float32",
            PAGE_BYTES["float32"],
            f"line {node.lineno}: {self.text(node)}",
            width,
        )
        self.compiled.includes |= {"port", "add", "pack"}
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
                f"{form}, {out.name}, {width})) {{\n    return false;\n}}",
                uses,
                mover="B",
            ),
        ]
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
        self.compute_waits = True
        self.dst_live = None
        tile.kind = "written"
        self.scope[name] = Value("reduced", name, buffer=out, width=width)
        self.reduced.append(name)
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
            slots = TREE_ROUNDS if size is None else max(1, (size - 1).bit_length())
        else:
            self.refuse(
                node,
                f"a reduce of self.{field_} sizes its slots by the form",
                'write form="owner" or form="tree"',
            )
        number = self.arrivals
        self.arrivals += 1
        arrival = self.other_buffer(
            f"kArrival{number}",
            "arrival",
            "float32",
            PAGE_BYTES["float32"],
            slots * tiles,
            f"line {node.lineno}: {self.text(node)}",
        )
        arrival.block = tiles
        self.compiled.unpack_fp32.add(arrival.name)
        buffer = f"kField{camel(field_)}"
        self.compiled.includes |= {"port", "pack"}
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
        combine = (
            f"KM_UNPACK_FORMAT_A({arrival.name}.cb);\n"
            f"copy_init({arrival.name}.cb);\n"
            f"KM_PACK_FORMAT({buffer});\n"
            f"pack_reconfig_l1_acc({sums} > 0 ? 1 : 0);\n"
            f"for (uint32_t {tile} = 0; {tile} < {tiles}; {tile}++) {{\n"
            "    tile_regs_acquire();\n"
            f"    copy_tile({arrival.name}.cb, km::slot_page({arrival.name}, slot) + "
            f"{tile}, 0);\n"
            "    tile_regs_commit();\n"
            "    tile_regs_wait();\n"
            f"    pack_tile<true>(0, {buffer}, {tile});\n"
            "    tile_regs_release();\n"
            "}\n"
            "pack_reconfig_l1_acc(0);\n"
            f"{sums}++;"
        )
        body = "\n".join("    " + line for line in combine.split("\n"))
        compute = [
            Line(
                f"// self.{field_}'s sums and the group's: each arrival added into it\n"
                f"if (km_ok && {sums} == 0) {{  // no product: the first arrival "
                "overwrites\n"
                f"    km_ok = compute.reserve_back({buffer}, {buffer}Pages);\n"
                "}\n"
                f"km_ok = km_ok && km::reduce(compute, {arrival.name}, "
                f"{group.format(cpu='compute')}, {form}, "
                f"[&](uint32_t km_slot) {{\n{body}\n}});\n"
                f"if (km_ok) {{  // to mover B's send, or to the leader's write\n"
                f"    cb_push_back({buffer}, {buffer}Pages);\n"
                "}\n"
                f"{sums} = 0;",
                uses | {"km_ok"},
            ),
        ]
        self.compute_waits = True
        self.scope[name] = Value(
            "reduced", name, buffer=None, width=tiles, summed=field_
        )
        self.reduced.append(name)
        self.wait(node, "mover A", WAIT_PORT)
        self.wait(node, "mover B", WAIT_EMIT)
        self.wait(node, "compute", WAIT_INPUT)
        return movement, compute

    def other_buffer(self, name, role, dtype, page_bytes, pages, what):
        for buffer in self.compiled.buffers:
            if buffer.name == name:
                return buffer
        taken = {b.index_a for b in self.compiled.buffers} | {
            b.index_b for b in self.compiled.buffers
        }
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
        self.compiled.broadcast_lines.setdefault(
            self.method.name, f"line {node.lineno}, `{self.text(node)}`"
        )
        self.compiled.rectangle = True
        self.compiled.includes |= {"broadcast"}
        self.compute_waits = True
        self.scope[name] = Value(
            "broadcast", name, mover=mover, window=window, pattern=along, length=length
        )
        unread = placed.get(spelling("broadcast_unread"))
        if unread is not None:  # a word of the handler: its C++, the names it reads
            self.scope[name].unread = (self.integer(unread), self.uses(unread))
        movement = Line(
            f"km::Broadcast {name}(movement, command, kArgBroadcast);\n"
            f"if (!{name}.ok()) {{\n    return false;\n}}\n"
            f"if (!{name}.included()) {{\n    return true;\n}}",
            mover=mover,
        )
        counter = f"km_blocks_{name}"  # kept where casts count over loops
        counted = [
            Glue(f"uint32_t {counter} = 0;", {counter}, mover=mover),
            Glue(f"uint32_t {counter} = 0;", {counter}),
        ]
        compute = []
        if len(self.movers) == 1:
            compute.append(Line("if (compute.arg(kArgIncluded) == 0) {\n    break;\n}"))
            return [movement, counted[0], self.other_mover_skips(mover)], [
                *compute,
                counted[1],
            ]
        return [movement, counted[0]], [counted[1]]

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
        return Line(
            "if (movement.arg(kArgIncluded) == 0) {\n    return true;\n}",
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
        self.scope[value] = Value("arrived", buffer.name, buffer=buffer)
        self.wait(node, f"mover {broadcast.mover}", WAIT_FREE)
        self.wait(node, f"mover {broadcast.mover}", WAIT_DONE)
        self.wait(node, f"mover {broadcast.mover}", WAIT_VALID)
        movement = [Line(text, uses, mover=broadcast.mover)]
        compute = []
        if counted is not None:  # the next block's number, on its mover and compute
            movement.append(Line(f"{counted}++;", {counted}, mover=broadcast.mover))
            compute.append(Line(f"{counted}++;", {counted}))
        if not self.loops:
            movement_end, compute = self.broadcast_end(name, "1", set())
            movement += movement_end
        return movement, compute

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

    def block_loop(self, node, value):
        """`for tile in v:` over a block a broadcast brought: compute waits for it,
        takes its pages in order, frees its slot."""
        if not isinstance(node.target, ast.Name):
            self.refuse(node, "one name a tile", "write for tile in block")
        self.arrived_once(node, value)
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
        for keyword_ in call.keywords:
            value = keyword_.value
            if keyword_.arg == spelling("at"):
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
                    "use takes=, gives=, on=, include= and at=",
                )
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
            field = self.field_name(item)
            if field in self.compiled.own:
                self.take_own(node, field, where, text, bound)
                continue
            if field:
                self.take_field(node, field, where, bound, before, after)
                continue
            uses |= self.take(node, item, where, bound, before, after)
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
            return [], []
        lines = [*before, Line(body, uses, mover=mover), *after]
        moves, self.body_moves = self.body_moves, []
        if mover is None:
            return moves, lines
        return lines, []

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
            self.refuse(node, "takes= names the handler's values", "take names")
        value = self.scope[item.id]
        if value.kind in ("int", "partition", "pattern"):
            return {item.id}
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
            bound.append(f"const uint32_t {word} = {item.id};")
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
        self.refuse(
            node,
            f"{item.id} is a {value.kind} here: a body takes ints, partitions, "
            "patterns, words of its CPU, stream tiles, write streams (on pack), "
            "tensors (on a mover) and written tiles",
            "pass those",
        )

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

    def arrived_once(self, node, value):
        """A block a broadcast brought is taken once: a body or a loop over it."""
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
        self.arrived_once(node, value)
        buffer = value.buffer.name
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
            movement.append(Block(head, "}", [*entry, *body_m], uses))
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
