"""The data of the class compiler: a handler's parameters, the program's buffers, the
lines that move pages and wait, what a name holds, and the compiled class."""

from dataclasses import dataclass, field
from typing import NamedTuple

from komorebi.dsl.nodes import Line
from komorebi.errors import ProgramError


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


class StreamOptions(NamedTuple):
    """A km.Stream's options: its range (a Value), block, pattern C++ (or None),
    pattern letters (or None), repeat C++, a read's tiles a row, its block's word."""

    pages: object
    block: int
    pattern: str | None
    letters: str | None
    repeat: str
    row: int
    word: str | None


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
    index: int | None = None  # a broadcast: its arguments' place among its mover's
    held: bool = False  # a block a broadcast brought: waited at its cast, freed after
    # a tile carried in a field (packed at acc's write): (buffer, token, slot, width,
    # index); a field written back in place: (field, its source, token)
    state: tuple | None = None
    row_of: tuple | None = None
    columns: int | None = None  # a held row: its tiles where the build knows them

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
    rect_lines: dict = field(default_factory=dict)  # handler -> self.x/y read at
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
    # Set by komorebi.dsl.program and komorebi.dsl.merged after the compile: a direct
    # input's landing area (tiles, page bytes, declared) and its (handler, place, pair)
    direct_area: tuple | None = None
    direct_in: tuple | None = None
    chain_stamps: bool = False  # the chain's stamps (program.stamps_made)
    handler_sources: dict = field(default_factory=dict)  # merged: handler -> source
