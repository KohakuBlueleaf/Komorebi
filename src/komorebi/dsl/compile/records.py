"""The data of the class compiler: a handler's parameters, the program's buffers, the
lines that move pages and wait, what a name holds, and the compiled class."""

from dataclasses import dataclass, field
from typing import NamedTuple

from komorebi.dsl.front.nodes import Line
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
    pers: list = field(default_factory=list)  # an arrival's reduces' part pages
    handlers: list = field(
        default_factory=list
    )  # a pair's streaming handlers (or None)
    padded: bool = False  # a read pair handlers of other blocks share (KM_RING_PADDED)


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
    tiles: bool = False  # a range of t.tiles[...]: first and count in tiles
    columns_first: int | None = None  # its first column, where the build knows it
    items: tuple | None = None  # a range of items: (bytes an item, items a page)
    summed: str = ""  # a reduce of a field of sums: the field
    everywhere: bool = False  # a reduce's sum every rank holds (km.all_reduce's)
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
class CompiledText:
    """The kernels' text: what komorebi.dsl.program writes the sources from (the
    nodes of each handler, buffers, includes, the compute options and the tile
    functions' units, the messages' and entries' words, the start-up)."""

    name: str
    source: str
    overlays: dict = field(default_factory=dict)  # overlay handler -> its number
    stamps: dict = field(default_factory=dict)  # overlay handler -> (tensor, word)
    dynamic: str | None = None  # the handler whose lines are a message loop
    messages: dict = field(default_factory=dict)  # km.Message class name -> words
    buffers: list = field(default_factory=list)
    movement: dict = field(default_factory=dict)  # handler -> nodes
    compute: dict = field(default_factory=dict)  # handler -> nodes
    includes: set = field(default_factory=set)  # "port", "broadcast", "fill", "add"
    # The #include lines of inline bodies, by kernel ("movement", "compute").
    body_includes: dict = field(default_factory=lambda: {"movement": [], "compute": []})
    unpack_fp32: set = field(default_factory=set)  # buffer names compute copies fp32
    summed_fields: set = field(default_factory=set)  # fields of sums: compute packs
    word_fields: dict = field(
        default_factory=dict
    )  # a field of km.u32 words: name -> n
    # Each tile function's unit program by its C++ name.
    units: dict = field(default_factory=dict)
    fidelity: str | None = None  # the products' (one a compute kernel), when given
    full_dst: bool = False  # a row-form tile function: DST in full-sync mode
    compute_options: dict = field(default_factory=dict)  # the class's compute=
    entries: dict = field(default_factory=dict)  # km.Entry class name -> words
    # The inline bodies at a kernel's start or end (at=): (at, kernel) -> texts.
    kernel_bodies: dict = field(default_factory=dict)
    kernel_body_lines: list = field(default_factory=list)  # their lines, in order
    startup: object = None  # the class's km.Startup, or None (the default start-up)
    # A handler with constant parameters -> [(its set of values, the handler built
    # for them)] (handler_variants).
    variants: dict = field(default_factory=dict)


@dataclass
class ThreadFacts:
    """The handlers' interface, which the X280 thread program (class_thread) and the
    class's files call by: their numbers, words, results, lines, tensors and regions."""

    handlers: dict = field(default_factory=dict)  # name -> number
    results: set = field(default_factory=set)  # handlers whose leader returns a word
    words: dict = field(default_factory=dict)  # name -> [(parameter, kind)]
    lines: dict = field(default_factory=dict)  # name -> its def line
    tensors: list = field(default_factory=list)  # tensor parameter names, in order
    regions: list = field(default_factory=list)  # region parameter names


@dataclass
class HostFacts:
    """What the host places and submits by: broadcasts, rectangles, reduce groups,
    ports, the plan's constants, the fields' layout, the islands, and what
    komorebi.dsl.program and komorebi.dsl.merged set after the compile."""

    # broadcast handler -> its km.Broadcast
    broadcast_lines: dict = field(default_factory=dict)
    # mover -> [(handler, its broadcast's line)] in the class's order: a core holds
    # each one's arguments after the last's (kArgBroadcast + kBroadcastArgs i)
    mover_broadcasts: dict = field(default_factory=dict)
    # handler -> (the size of its reduce's groups, the reduce's line)
    reduce_groups: dict = field(default_factory=dict)
    port_handlers: set = field(default_factory=set)  # handlers whose cores use ports
    broadcasts: list = field(default_factory=list)  # (along, buffer, window)
    rect_sized: bool = False  # a handler's rectangle given at each submit
    rect_plain: bool = False  # a handler reads the make's rectangle
    rect_broadcasts: set = field(default_factory=set)  # (mover, index) computed so
    constants: dict = field(default_factory=dict)  # plan names used -> value
    rectangle: bool = False  # self.x or self.y used
    fields: dict = field(default_factory=dict)  # name -> front.Field
    islands: tuple = (1, 0)  # the session's (islands, an island's most ranks)
    chain: int = 0  # the chain's walk: its handler number (0: none in this build)
    # Set by komorebi.dsl.program and komorebi.dsl.merged after compile: a direct
    # input's landing area (tiles, page bytes, declared) and its (handler, place, pair)
    direct_area: tuple | None = None
    direct_in: tuple | None = None
    chain_stamps: bool = False  # the chain's stamps (program.stamps_made)
    handler_sources: dict = field(default_factory=dict)  # merged: handler -> source


@dataclass
class Reports:
    """What the build reports and checks after the compile: waits, moves, fills, tile
    function calls, products and their placements, hand-overs, fields' users."""

    waits: list = field(default_factory=list)
    fills: list = field(default_factory=list)  # (field, count text, handler, line)
    moves: list = field(default_factory=list)  # Move of every line that moves pages
    # (handler, line) -> [(stream variable, tensor, prefetch= or None)] of each loop
    # over read streams in step: a chain build holds each stream by its tensor.
    zips: dict = field(default_factory=dict)
    tile_calls: list = field(default_factory=list)  # each call of a tile function
    # Each product `acc += f(block, self.w)`: (handler, line, text, function, columns,
    # block tiles).
    products: list = field(default_factory=list)
    # Each product's placement: (handler, line, text, left format, right format,
    # "none" | "plain" | "turned", ("field" | "tensor", W's name) or None).
    placements: list = field(default_factory=list)
    # field -> (the handlers that fill it, those that read it and do not fill it)
    field_users: dict = field(default_factory=dict)
    # Each value an inline body or a line hands between compute CPUs by mailbox.
    handovers: list = field(default_factory=list)
    # (handler, tensor parameter) -> the tiles a DRAM page of its example
    page_tiles: dict = field(default_factory=dict)


@dataclass
class CompilerOnly:
    """What only the compile reads: the folded message loops, own fields and their
    pushes, fields' producers, reserved indices, the class's constants."""

    # The other dynamic handlers, each its message loop alone: name -> method; their
    # loops are cases of the dynamic handler's loop, by the handler a message names.
    folded: dict = field(default_factory=dict)
    own: dict = field(default_factory=dict)  # an own field -> its push (front.Own)
    pushers: dict = field(default_factory=dict)  # own field -> {CPU: (handler, line)}
    # field -> (its producer: "fill" or "product", handler, line, the line's text)
    producers: dict = field(default_factory=dict)
    reserved: set = field(default_factory=set)  # indices left to other programs
    share_pairs: bool = False  # handlers of other blocks may share a read pair
    # The class's constants (km.constant) by name: this program's values; a symbol's
    # only where the plan gives it (declared_constants).
    class_constants: dict = field(default_factory=dict)
    declared: dict = field(default_factory=dict)  # name -> its front.Constant


@dataclass
class Compiled(CompilerOnly, Reports, HostFacts, ThreadFacts, CompiledText):
    """The compiled class: one object of five declared parts, each field in the part
    of the code that reads it (CompiledText, ThreadFacts, HostFacts, Reports,
    CompilerOnly); the parts' fields come in that order (name and source first)."""
