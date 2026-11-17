"""The state of one handler's compile: its inputs, where its lines are, its names, and
what its lines so far leave open (loops, DST, fields, the lines before and after)."""

import ast
from collections.abc import Callable
from dataclasses import dataclass, field

from komorebi.dsl.compile.records import Compiled
from komorebi.dsl.compile.source import Where


@dataclass(eq=False)
class Frame:
    """One handler's compile: the class compiled so far, the handler (a
    front.HandlerMethod), the names its lines read, the plan, the example tensors;
    then the state its lines leave for the lines after them."""

    compiled: Compiled
    method: object
    names: dict
    plan: dict
    tensors: dict
    # the lowering of a list of statements to (movement, compute) nodes: the
    # statements module's entry bound to this frame, which the families recurse into
    statements: Callable | None = None
    definition: ast.FunctionDef | None = None  # the handler's def, parsed
    where: Where | None = None  # its lines in its file
    assigned: set = field(default_factory=set)  # names assigned twice: not const
    dynamic: bool = False  # its lines are a message loop
    # What the compile returns, which the driver sets on the front.HandlerMethod:
    words: list | None = None  # the command words, [(parameter, kind)]
    widths: dict = field(default_factory=dict)  # tensor parameter -> tile width
    returns: bool = False  # its leader returns a word
    scope: dict = field(default_factory=dict)
    params: list = field(default_factory=list)
    # tensor parameter -> its page-count word name
    counts: dict = field(default_factory=dict)
    # the movers ("A", "B") that put items: waited at the end
    putters: list = field(default_factory=list)
    # a body's node id -> the words it takes of the other mover
    handed: dict = field(default_factory=dict)
    # a drained stream's name -> its field, end line, fills
    drained: dict = field(default_factory=dict)
    reads: int = 0
    writes: int = 0
    arrivals: int = 0
    dst_next: int = 0
    dst_live: str | None = None  # the accumulating tile's name while DST is held
    # tiles written or reduced in a loop deeper than zeros
    carried: set = field(default_factory=set)
    # two DST sums held together (a second zeros), by name
    dst_pair: tuple | None = None
    compute_waits: bool = False
    # reduced values written by the leader
    leader_writes: set = field(default_factory=set)
    reduced: list = field(default_factory=list)
    in_leader: bool = False
    # the range loops around the statement, innermost last
    loops: list = field(default_factory=list)
    # the broadcasts that cast once outside every loop
    cast_flat: set = field(default_factory=set)
    movers: dict = field(default_factory=dict)  # mover -> the broadcast it runs
    allow_words: bool = False  # a body's word is read by the return only
    field_line: bool = False  # a word field's words are read in a line setting one
    # compute's lines after the handler's (a product's field)
    closing: list = field(default_factory=list)
    # {block a cast outside a loop brought: its mover}, whose finish (the wait for
    # compute to free the block) is at the cast
    finished_casts: dict = field(default_factory=dict)
    # {such a block: its finish's nodes at the cast, None under a condition}, and
    # the blocks whose finish goes to the handler's end (handler_buffers)
    cast_finishes: dict = field(default_factory=dict)
    deferred_casts: list = field(default_factory=list)
    # compute's lines before them (a field's sums count)
    opening: list = field(default_factory=list)
    opening_b: list = field(default_factory=list)  # mover B's (the same count)
    # a field of sums -> its counter's C++ name
    summed: dict = field(default_factory=dict)
    # field -> (its fill's node, its block) until a body takes it
    filled: dict = field(default_factory=dict)
    # field -> (the range loops around its fill, its line)
    fill_loops: dict = field(default_factory=dict)
    # field -> the range loop its stream fill stands in
    fill_loop_of: dict = field(default_factory=dict)
    leader_condition: tuple | None = None  # the C++ of `self.rank == self.leader`, uses
    # fields last written from sums started from a row
    masked: set = field(default_factory=set)
    # fields compute waited for (a product's, once)
    waited_fields: set = field(default_factory=set)
    # a step's state field -> its pages (a state tile a page)
    stepped: dict = field(default_factory=dict)
    # a field compute writes -> its flag: pages pushed, not freed
    packed: dict = field(default_factory=dict)
    # a statement list -> blocks taken there by products
    held_blocks: dict = field(default_factory=dict)
    leader_text: str = "0"  # self.leader: rank 0, or a reduce group's first rank
    # the numbers of the read loops around the statement
    stream_loops: list = field(default_factory=list)
    # Per read loop around the statement: its tile function, its other tile
    # lines, and the lines compute runs once before its blocks.
    loop_states: list = field(default_factory=list)
    # id of a tile expression's call: (call, function)
    expression_calls: dict = field(default_factory=dict)
    # the statement lists around the statement, innermost last
    blocks: list = field(default_factory=list)
    # the movers' lines of an inline body's write stream
    body_moves: list = field(default_factory=list)
    # a statement list -> its read streams bodies take (held_stream)
    held: dict = field(default_factory=dict)
    tables: int = 0  # the handler's tables of entries (entries_table)
    rect_param: str | None = None  # the handler's rectangle parameter (rect_glue)
    rect_read: bool = False  # the lines read self.x, self.y or self.core
    # buffer -> its block's submit word (None: a constant)
    block_words: dict = field(default_factory=dict)
    # a sized tensor parameter -> its tile columns' word
    column_words: dict = field(default_factory=dict)
    spans_most: int | None = None  # a stamps page: the spans it holds (tile_stamps)
    # a name first assigned in an if or a loop -> (kind, line)
    ended: dict = field(default_factory=dict)
