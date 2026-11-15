"""The compiler of a program written as a class, level 1: each handler's body, read
with `ast`, becomes the nodes of the movement source (both movers) and of the compute
kernel; streams become buffer pairs, parameters command words, a refusal names the
line. `komorebi.dsl.program` writes the files and makes the Program.
"""

import ast
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
# Names the generated code takes for itself: a parameter or a variable may not.
GENERATED = {"command", "movement", "compute", "rank", "size", "group", "ok"}
GENERATED |= {"self_x", "self_y", "rect_columns", "rect_rows"}
GENERATED_NUMBERED = re.compile(
    r"(block|half|valid|tile|read|write|out|room|page|address)\d+"
)
# The runtime's where names (komorebi.state) of each kind of wait.
WAIT_DATA, WAIT_INPUT, WAIT_OUTPUT = "data", "cinput", "coutput"
WAIT_PORT, WAIT_EMIT = "port", "emit"
WAIT_VALID, WAIT_DONE = "broadcastvalid", "broadcastdone"
WAIT_FREE = "broadcastfree"
# A broadcast's mover: along a row on mover A (NoC 0), down a column on mover B (NoC 1).
ALONG = {"row": "A", "column": "B"}


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
    loop: dict | None = None  # broadcast: the range loop its blocks count
    waited: bool = False  # a written tile: mover B has waited for its page
    popped: bool = False  # a written tile: mover B has freed its page

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
    overlays: dict = field(
        default_factory=dict
    )  # overlay handler -> its overlay number
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
    broadcasts: list = field(default_factory=list)  # (along, buffer, window)
    constants: dict = field(default_factory=dict)  # plan names used -> value
    rectangle: bool = False  # self.x or self.y used
    fields: dict = field(default_factory=dict)  # name -> front.Field
    fills: list = field(default_factory=list)  # (field, count text, handler, line)
    moves: list = field(default_factory=list)  # Move of every line that moves pages


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
    for number, method in enumerate(methods, start=1):
        compiled.handlers[method.name] = number
        if method.overlay:
            compiled.overlays[method.name] = len(compiled.overlays) + 1
        names = method.function.__globals__
        HandlerCompiler(compiled, method, names, plan, tensors).run()
    bind_fields(compiled, plan, l1)
    return compiled


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
                "make the program with l1=session.l1()[1]",
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
        page_bytes = TILE_ELEMENTS * value.format.itemsize
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
            f"field {name}: {value.format.dtype}, {shape}",
        )
        compiled.buffers.append(buffer)
        for n in value.shape:
            if isinstance(n, str):
                compiled.constants[n] = solved[n]
    for name, count, handler, line in compiled.fills:
        tiles = field_elements(compiled.fields[name], solved) // TILE_ELEMENTS
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


def field_elements(value, solved):
    return math.prod(solved[n] if isinstance(n, str) else n for n in value.shape)


def solve(compiled, symbol, plan, l1, name):
    """The largest value of `symbol` for which every field is whole tiles and the
    fields and buffers fit `l1` bytes."""
    used = sum(sum(b.pages) * b.page_bytes for b in compiled.buffers)
    fixed, per = 0, 0  # bytes without the symbol, and bytes per unit of it
    step = 1
    for value in compiled.fields.values():
        rest = field_elements(value, {**plan, symbol: 1}) * value.format.itemsize
        if symbol in value.shape:
            per += rest
            items = rest // value.format.itemsize
            step = math.lcm(step, TILE_ELEMENTS // math.gcd(items, TILE_ELEMENTS))
        else:
            fixed += rest
    room = l1 - used - fixed
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
        self.filled = {}  # field -> (its fill's node, its block) until a body takes it
        self.stream_loops = []  # the numbers of the read loops around the statement
        self.blocks = []  # the statement lists around the statement, innermost last
        self.assigned = self.assigned_names()

    # The source -------------------------------------------------------------------

    def text(self, node):
        """The node's source on one line: a compound statement's head, else all of
        it, each line stripped."""
        body = getattr(node, "body", None)
        last = body[0].lineno - 1 if isinstance(body, list) and body else None
        last = last or getattr(node, "end_lineno", None) or node.lineno
        lines = self.source_lines[
            node.lineno - self.first_line : last - self.first_line + 1
        ]
        text = " ".join(
            line.strip() for line in lines if not line.strip().startswith("#")
        )
        return re.sub(r"([(\[]) | ([)\]])|,([)\]])", r"\1\2\3", text)

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
        movement = [
            Glue("const uint32_t rank = movement.me() - km::first(command);", {"rank"})
        ]
        compute = [
            Glue("const uint32_t rank = compute.me() - km::first(command);", {"rank"})
        ]
        for nodes, me in ((movement, "movement"), (compute, "compute")):
            nodes.append(Glue("const uint32_t size = km::count(command);", {"size"}))
            for axis in ("x", "y"):
                nodes.append(
                    Glue(
                        f"const uint32_t self_{axis} = {me}.arg(kArg{axis.upper()});",
                        {f"self_{axis}"},
                    )
                )
            for side in ("columns", "rows"):
                nodes.append(
                    Glue(
                        f"const uint32_t rect_{side} = {me}.arg(kArg{side.title()});",
                        {f"rect_{side}"},
                    )
                )
            nodes.append(
                Glue(
                    f"const km::Group group = km::group_of(command, {me}.me());",
                    {"group"},
                )
            )
        self.bind_parameters(movement, compute)
        body_m, body_c = self.statements(self.definition.body)
        movement += body_m
        compute += body_c
        compute.insert(len(compute) - len(body_c), Glue("bool ok = true;", {"ok"}))
        for kept, filled in self.filled.items():
            if filled is not None:
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
        self.compiled.compute[name] = compute

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
            or GENERATED_NUMBERED.fullmatch(name)
            or keyword.iskeyword(name)
            or name in CXX_KEYWORDS
        ):
            self.refuse(
                node,
                f"the name {name} is taken by the generated C++",
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
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self.refuse(node, "one name a line", "assign one name")
        name = node.targets[0].id
        self.check_name(node, name)
        value = node.value
        call = self.construct(value)
        if call == "partition":
            return self.partition(node, name, value)
        if call == "zeros":
            return self.zeros(node, name, value)
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
        if not isinstance(target, ast.Name) or target.id not in self.scope:
            self.refuse(node, "an unknown name", "assign it first")
        value = self.scope[target.id]
        if value.kind == "tile":
            return self.accumulate(node, value)
        if value.kind != "int":
            self.refuse(node, f"{target.id} is a {value.kind}", "add to an int")
        op = OPERATORS.get(type(node.op))
        if op is None:
            self.refuse(node, "an operator level 1 has no C++ for", "use + - * etc.")
        text = f"{target.id} {op}= {self.integer(node.value)};"
        uses = self.uses(node.value) | {target.id}
        return [Line(text, uses)], [Line(text, uses)]

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
            return str(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.scope:
                value = self.scope[node.id]
                if value.kind == "word" and self.allow_words:
                    return node.id
                if value.kind != "int":
                    self.refuse(
                        node, f"{node.id} is a {value.kind}, not an int", "use an int"
                    )
                return node.id
            return str(self.constant(node, node.id))
        if isinstance(node, ast.Attribute):
            return self.attribute(node)
        if isinstance(node, ast.BinOp):
            op = OPERATORS.get(type(node.op))
            if op is None:
                self.refuse(node, "an operator level 1 has no C++ for", "use + - * //")
            return f"({self.integer(node.left)} {op} {self.integer(node.right)})"
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = COMPARISONS.get(type(node.ops[0]))
            if op is None:
                self.refuse(node, "a comparison level 1 has no C++ for", "use == < >")
            left, right = node.left, node.comparators[0]
            return f"{self.integer(left)} {op} {self.integer(right)}"
        if isinstance(node, ast.BoolOp):
            op = " && " if isinstance(node.op, ast.And) else " || "
            return "(" + op.join(self.integer(v) for v in node.values) + ")"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return f"!({self.integer(node.operand)})"
        self.refuse(node, "not an integer of level 1", "write ints, names, + - * //")

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
            kind = kind_of(node.attr, ("rank", "size", "leader", "x", "y"))
            if kind == "rank":
                return "rank"
            if kind == "size":
                return "size"
            if kind == "leader":
                return "0"
            if kind in ("x", "y"):
                self.compiled.rectangle = True
                return f"self_{kind}"
            self.field_use(node)
            self.refuse(node, f"self.{node.attr} is not known", "use self.rank")
        if isinstance(base, ast.Name) and base.id in self.scope:
            value = self.scope[base.id]
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
                    kind = kind_of(item.attr, ("rank", "size", "x", "y"))
                    if kind in ("x", "y"):
                        found.add(f"self_{kind}")
                    elif kind:
                        found.add(kind)
                if item.value.id in self.counts and item.attr == spelling("pages"):
                    found.add(self.counts[item.value.id])
        return found

    # Ranges, partitions ------------------------------------------------------------

    def is_range(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return node.func.attr == spelling("pages") and self.is_tensor(
                node.func.value
            )
        if isinstance(node, ast.Subscript):
            return self.is_tensor(node.value)
        if isinstance(node, ast.Attribute):
            return node.attr == spelling("pages") and self.is_tensor(node.value)
        if isinstance(node, ast.Name):
            return self.scope.get(node.id, Value("")).kind == "range"
        return False

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

    def partition(self, node, name, call):
        """`part = self.partition(pages)`: this core's part by rank; `part =
        b.partition(pages)`: by its place along broadcast b's row or column."""
        index, length = "rank", "size"
        if isinstance(call.func.value, ast.Name) and call.func.value.id != "self":
            along = self.scope[call.func.value.id].pattern
            index, length = (
                ("self_x", "rect_columns")
                if along == "row"
                else ("self_y", "rect_rows")
            )
        if len(call.args) != 1 or call.keywords:
            self.refuse(node, "partition(pages) takes one range", "give one")
        argument = call.args[0]
        if self.is_range(argument):
            pages = self.range_of(argument)
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
        dst = self.dst_next
        self.dst_next += 1
        self.dst_live = name
        self.scope[name] = Value("tile", name, dst=dst)
        self.compiled.includes |= {"fill"}
        text = f"tile_regs_acquire();\nfill_tile_init();\nfill_tile({dst}, 0.0f);"
        return [], [Line(text)]

    def scratch(self):
        """The DST tile a copy lands in before an add: the one after the sums."""
        return self.dst_next

    def accumulate(self, node, value):
        """`total += tile` inside a stream loop, or `total += block` of a broadcast."""
        source = node.value
        if not isinstance(source, ast.Name) or source.id not in self.scope:
            self.refuse(node, "add a tile of a stream", "write total += tile")
        item = self.scope[source.id]
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
        """(range, block, pattern C++, pattern letters or None, repeat C++)."""
        if len(call.args) != 1 or not self.is_range(call.args[0]):
            self.refuse(
                node,
                "a stream of a page range",
                "write km.Stream(tensor[part], block=8)",
            )
        pages = self.range_of(call.args[0])
        block, pattern, letters, repeat = 1, None, None, "1"
        for keyword_ in call.keywords:
            key = keyword_.arg
            if key == spelling("block"):
                block = self.constant_int(keyword_.value)
            elif key == "pattern":
                pattern, letters = self.pattern(keyword_.value)
            elif key == spelling("repeat"):
                repeat = self.integer(keyword_.value)
            elif key == spelling("write_stream"):
                continue
            else:
                self.refuse(node, f"a stream has no {key}=", "use block=, pattern=")
        if block < 1:
            self.refuse(node, "a block of no pages", "give block= a count")
        if pattern is None:
            letters = WRITE_PATTERN if write else READ_PATTERN
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
        tensor_type = camel(pages.tensor)
        direction = "km::kMoveRead" if kind == "read" else "km::kMoveWrite"
        return (
            f"const km::Stream {variable}(\n"
            f"    {direction}, {tensor_type}({pages.tensor}), {pages.first}, "
            f"{pages.count}, {buffer.name}Block, {repeat}, {buffer.name}, {pattern}\n"
            ");"
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
        variable = f"write{number}"
        self.scope[name] = Value("write", variable, buffer=buffer, pattern=pattern_text)
        text = self.stream_cxx("write", variable, pages, buffer, pattern_text, repeat)
        uses = range_uses(pages, self) | pattern_uses(pattern)
        self.wait(node, "mover A", WAIT_DATA)
        self.wait(node, "mover B", WAIT_DATA)
        return [Glue(text, {variable}, uses)], []

    def loop(self, node):
        if node.orelse:
            self.refuse(node, "a loop with an else", "write the else after it")
        iterator = node.iter
        if (
            isinstance(iterator, ast.Name)
            and self.scope.get(iterator.id, Value("")).kind == "arrived"
        ):
            return self.block_loop(node, self.scope[iterator.id])
        if self.construct(iterator) == "stream":
            return self.read_loop(node, iterator)
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
        body_m, body_c = self.statements(node.body)
        self.loops.pop()
        self.scope.pop(name)
        head = f"for (uint32_t {name} = {first}; {name} < {end}; {name}++) {{"
        head_c = head
        if self.compute_waits:
            head_c = (
                f"for (uint32_t {name} = {first}; ok && {name} < {end}; {name}++) {{"
            )
        movement = [Block(head, "}", body_m, uses)] if body_m else []
        compute = [Block(head_c, "}", body_c, uses | {"ok"})] if body_c else []
        for broadcast in loop["casts"]:
            m, c = self.broadcast_end(broadcast, end, uses)
            movement += m
            compute += c
        return movement, compute

    def read_loop(self, node, call):
        if not isinstance(node.target, ast.Name):
            self.refuse(node, "one name a tile", "write for tile in km.Stream(...)")
        pages, block, pattern, letters, repeat = self.stream_options(node, call, False)
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        number = self.reads
        self.reads += 1
        buffer = self.pair(
            "read",
            number,
            dtype,
            page_bytes,
            block,
            READ_ITEMS,
            f"line {node.lineno}: {self.text(node)}",
        )
        pattern_text = self.pattern_cxx(buffer, pattern, letters)
        if dtype == "float32":
            self.compiled.unpack_fp32.add(buffer.name)
        self.moved(node, pages.tensor, "read stream")
        variable = f"read{number}"
        half, tile, valid = f"half{number}", f"tile{number}", f"valid{number}"
        block_var = f"block{number}"
        self.scope[node.target.id] = Value("streamed", half, first=tile)
        before = set(self.scope)
        writes_before = [v for v in self.scope.values() if v.kind == "write"]
        pushed_before = {id(v): v.pushed for v in writes_before}
        self.stream_loops.append(number)
        body_m, body_c = self.statements(node.body)
        self.stream_loops.pop()
        if body_m:
            self.refuse(
                node.body[0],
                "inside a stream loop only compute's lines",
                "move the line before or after the loop",
            )
        for name in set(self.scope) - before:
            self.scope.pop(name)
        self.scope.pop(node.target.id, None)
        pushed = [v for v in writes_before if v.pushed and not pushed_before[id(v)]]
        # The movers: this stream and the write streams it feeds, in one move.
        stream_uses = (
            range_uses(pages, self) | pattern_uses(pattern) | text_uses(repeat, self)
        )
        streams = ", ".join([variable] + [v.cxx for v in pushed])
        movement = [
            Glue(
                self.stream_cxx("read", variable, pages, buffer, pattern_text, repeat),
                {variable},
                stream_uses,
            ),
            Line(
                f"if (!movement.move({streams})) {{\n    return false;\n}}",
                {variable} | {v.cxx for v in pushed},
            ),
        ]
        self.wait(node, "mover A", WAIT_DATA)
        self.wait(node, "mover B", WAIT_DATA)
        self.wait(node, "compute", WAIT_INPUT)
        # Compute: block k from the half its owner filled; the tiles of the block.
        self.compute_waits = True
        blocks = f"({pages.count} + {buffer.name}Block - 1) / {buffer.name}Block"
        per_turn = f"blocks{number}"
        count_uses = text_uses(pages.count, self)
        compute = []
        if pattern is not None:
            compute.append(
                Line(f"if (!{pattern}.valid()) {{\n    return;\n}}", {pattern})
            )
        compute.append(
            Glue(f"const uint32_t {per_turn} = {blocks};", {per_turn}, count_uses)
        )
        total = per_turn if repeat == "1" else f"{per_turn} * {repeat}"
        inner = [
            Line(
                f"const uint32_t {half} = km::block_half({buffer.name}, "
                f"{pattern_text}, {block_var});\n"
                f"ok = compute.wait_front({half}, {buffer.name}Block);\n"
                f"if (!ok) {{\n    break;\n}}",
                uses=pattern_uses(pattern),
            ),
            Glue(
                f"const uint32_t {valid} = {block_var} % {per_turn} + 1 == "
                f"{per_turn} ? {pages.count} - ({per_turn} - 1) * "
                f"{buffer.name}Block : {buffer.name}Block;",
                {valid},
                count_uses,
            ),
        ]
        for write in pushed:
            out, room = f"out{number}", f"room{number}"
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
        if pushed:
            inner.append(Line(f"copy_init({half});"))
        loop = f"for (uint32_t {tile} = 0; {tile} < {valid}; {tile}++) {{"
        if body_c:
            inner.append(Block(loop, "}", body_c, {valid}))
        for write in pushed:
            out, room = f"out{number}", f"room{number}"
            inner.append(
                Line(
                    f"if ({room}) {{\n    cb_push_back({out}, "
                    f"{write.buffer.name}Block);\n}}"
                )
            )
        inner.append(Line(f"cb_pop_front({half}, {buffer.name}Block);"))
        head = (
            f"for (uint32_t {block_var} = 0; ok && {block_var} < {total}; "
            f"{block_var}++) {{"
        )
        uses = {per_turn, "ok"} | text_uses(repeat, self)
        compute.append(Block(head, "}", inner, uses))
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
        if len(call.args) == 1 and isinstance(call.args[0], ast.BinOp):
            self.refuse(
                node,
                "a tile expression (`tile * factors`, `block @ self.weight`) is "
                "level 2",
                "push the loop's tile, or make the tile in a km.cxx body "
                "(gives=km.Tile)",
            )
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            self.refuse(node, "push one tile of the loop", "write out.push(tile)")
        item = self.scope.get(call.args[0].id)
        if item is not None and item.kind == "tile" and self.dst_live == item.cxx:
            return self.push_made(node, stream, item)
        if item is None or item.kind != "streamed" or not item.cxx.startswith("half"):
            self.refuse(node, "push a tile of the stream loop", "write out.push(tile)")
        if self.dst_live is not None:
            self.refuse(node, f"{self.dst_live} holds DST", "write it first")
        if stream.pushed:
            self.refuse(
                node, "a write stream fed by two loops", "open a write stream for each"
            )
        stream.pushed = True
        number = item.cxx.removeprefix("half")
        text = (
            "tile_regs_acquire();\n"
            f"copy_tile({item.cxx}, {item.first}, 0);\n"
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"if (room{number}) {{\n"
            f"    pack_tile<true>(0, out{number}, {item.first});\n"
            "}\n"
            "tile_regs_release();"
        )
        return [], [Line(text)]

    def push_made(self, node, stream, item):
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
        stream.pushed = True
        number = self.stream_loops[-1]
        self.dst_live, self.dst_next = None, 0
        item.kind = "pushed"
        text = (
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"if (room{number}) {{\n"
            f"    pack_tile<true>(0, out{number}, tile{number});\n"
            "}\n"
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
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            self.refuse(node, "write one tile", "write pages.write(total)")
        name = call.args[0].id
        value = self.scope.get(name)
        if value is None or value.kind not in ("tile", "reduced") or value.popped:
            self.refuse(
                node,
                f"{name} is not a tile of compute (or is written already)",
                "write a tile made by km.tile.zeros and added to, once",
            )
        if pages.count != "1":
            self.refuse(node, "a tile is one page", "write pages(first, 1)")
        self.compiled.includes.add("pack")
        compute = []
        if value.kind == "tile":
            dtype, page_bytes = self.tensor_format(node, pages.tensor)
            buffer = self.out_buffer(
                dtype, page_bytes, f"line {node.lineno}: {self.text(node)}"
            )
            compute.append(Line(pack(value.dst, buffer.name), {"ok"}))
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
        # A page write: mover B waits for compute's page, writes it, frees it once
        # sent (as a hand writer does); the runtime settles writes before it reports.
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        write = (
            f"km::write_pages(movement, command, {tensor}, {pages.first}, 1, "
            f"get_read_ptr({buffer.name}))"
        )
        if value.waited:
            opening = f"if (!{write}) {{"
        else:
            opening = self.page_wait(value)[:-3] + f" ||\n    !{write}) {{"
        text = (
            f"{opening}\n"
            "    return false;\n"
            "}\n"
            "noc_async_writes_flushed();\n"
            f"cb_pop_front({buffer.name}, 1);"
        )
        value.waited = value.popped = True
        self.moved(node, pages.tensor, "page write")
        movement = [Line(text, range_uses(pages, self), mover="B")]
        self.compiled.includes.add("ask")
        self.wait(node, "mover B", WAIT_DATA)
        return movement, compute

    def out_buffer(self, dtype, page_bytes, what):
        """The buffer of one page compute packs and mover B writes: the program's
        next one of this handler (handlers share them), two pages."""
        number = self.writes
        self.writes += 1
        name = f"kOut{number}"
        for buffer in self.compiled.buffers:
            if buffer.name == name and (buffer.dtype, buffer.page_bytes) == (
                dtype,
                page_bytes,
            ):
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
            name, "out", index, None, dtype, page_bytes, (WRITE_ITEMS,), 1, what
        )
        self.compiled.buffers.append(buffer)
        return buffer

    def branch(self, node):
        broadcast = self.broadcast_branch(node)
        if broadcast is not None:
            return broadcast
        condition = self.integer(node.test)
        uses = self.uses(node.test)
        live = self.dst_live
        leader, self.in_leader = self.in_leader, self.in_leader or self.leader_test(
            node
        )
        then_m, then_c = self.statements(node.body)
        self.in_leader = leader
        else_m, else_c = self.statements(node.orelse)
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
        dtype, page_bytes = self.tensor_format(node, pages.tensor)
        value = self.compiled.fields[field]
        if (dtype, page_bytes) != (value.format.dtype, 1024 * value.format.itemsize):
            self.refuse(
                node,
                f"{pages.tensor} has {dtype} pages and self.{field} "
                f"{value.format.dtype} tiles",
                "fill a field from a tensor of its format",
            )
        buffer, pages_name = f"kField{camel(field)}", f"kField{camel(field)}Pages"
        self.compiled.fills.append((field, pages.count, self.method.name, node.lineno))
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
        self.wait(node, "mover A", WAIT_DATA)
        return [Line(text, range_uses(pages, self), mover="A")], []

    def take_field(self, node, field, mover, bound, before, after):
        """A field an inline body takes: compute gets its buffer index, mover B its
        L1 address, once mover A filled it; it is freed after the body."""
        if field not in self.filled or self.filled[field] is None:
            self.refuse(
                node,
                f"self.{field} is not filled above in this handler (or a body took "
                "it): a field is filled for one body, in the handler that takes it",
                f"write self.{field}.fill(km.Stream(...)) before this body",
            )
        if self.filled[field][1] != self.blocks[-1]:
            self.refuse(
                node,
                f"self.{field} is filled in another block: every core that fills it "
                "must free it",
                "take it in the block of its fill (or fill it inside this one)",
            )
        if mover == "A":
            self.refuse(
                node,
                f"self.{field} is filled by mover A and handed on",
                'take it in a body on compute or on="mover_b"',
            )
        self.filled[field] = None
        buffer, pages_name = f"kField{camel(field)}", f"kField{camel(field)}Pages"
        if mover is None:
            before.append(
                Line(
                    f"if (!compute.wait_front({buffer}, {pages_name})) {{\n"
                    "    return;\n}"
                )
            )
            bound.append(f"const uint32_t {field} = {buffer};")
            after.append(Line(f"cb_pop_front({buffer}, {pages_name});"))
            self.wait(node, "compute", WAIT_INPUT)
        else:
            before.append(
                Line(
                    f"if (!movement.wait_front({buffer}, {pages_name})) {{\n"
                    "    return false;\n}",
                    mover="B",
                )
            )
            bound.append(f"const uint32_t {field} = get_read_ptr({buffer});")
            after.append(Line(f"cb_pop_front({buffer}, {pages_name});", mover="B"))
            self.wait(node, "mover B", WAIT_DATA)

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
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            self.refuse(node, "reduce one tile", "write km.reduce(total, form=form)")
        tile = self.scope.get(call.args[0].id)
        if tile is None or tile.kind != "tile" or self.dst_live != call.args[0].id:
            self.refuse(
                node, "reduce the tile held in DST", "write km.reduce(total, form=form)"
            )
        form = None
        for keyword_ in call.keywords:
            if keyword_.arg != spelling("form"):
                self.refuse(node, f"a reduce has no {keyword_.arg}=", "use form=")
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
        number = self.arrivals
        self.arrivals += 1
        arrival = self.other_buffer(
            f"kArrival{number}",
            "arrival",
            "float32",
            PAGE_BYTES["float32"],
            REDUCE_SLOTS,
            f"line {node.lineno}: {self.text(node)}",
        )
        self.compiled.unpack_fp32.add(arrival.name)
        out = self.out_buffer(
            "float32", PAGE_BYTES["float32"], f"line {node.lineno}: {self.text(node)}"
        )
        self.compiled.includes |= {"port", "add", "pack"}
        uses = {"group"} | form_uses
        movement = [
            Line(
                f"if (!km::reduce_arrive(movement, command, {arrival.name}, group, "
                f"{form})) {{\n    return false;\n}}",
                uses,
                mover="A",
            ),
            Line(
                f"if (!km::reduce_emit(movement, command, {arrival.name}, group, "
                f"{form}, {out.name}, 1)) {{\n    return false;\n}}",
                uses,
                mover="B",
            ),
        ]
        combine = add_tile(
            f"{arrival.name}.cb",
            f"km::slot_page({arrival.name}, slot)",
            tile.dst,
            self.scratch(),
        )
        body = "\n".join("    " + line for line in combine.split("\n"))
        compute = [
            Line(
                f"ok = ok && km::reduce(compute, {arrival.name}, group, {form}, "
                f"[&](uint32_t slot) {{\n{body}\n}});",
                uses | {"ok"},
            ),
            Line(pack(tile.dst, out.name), {"ok"}),
        ]
        self.compute_waits = True
        self.dst_live = None
        tile.kind = "written"
        self.scope[name] = Value("reduced", name, buffer=out)
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
        along, window = None, BROADCAST_WINDOW
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
            else:
                self.refuse(
                    node,
                    f"a broadcast has no {keyword_.arg}=",
                    "use along= and window=",
                )
        if along is None:
            self.refuse(
                node,
                "a broadcast goes along a row or a column",
                'write along="row" or along="column"',
            )
        mover = ALONG[along]
        if mover in self.movers:
            self.refuse(
                node,
                f"{self.movers[mover]} already goes along a {along}; each mover runs "
                "one broadcast (rows on mover A, columns on mover B)",
                "make one broadcast along a row and one along a column",
            )
        self.movers[mover] = name
        self.compiled.rectangle = True
        self.compiled.includes |= {"broadcast"}
        self.compute_waits = True
        self.scope[name] = Value(
            "broadcast", name, mover=mover, window=window, pattern=along
        )
        movement = Line(
            f"km::Broadcast {name}(movement, command, kArgBroadcast);\n"
            f"if (!{name}.ok()) {{\n    return false;\n}}\n"
            f"if (!{name}.included()) {{\n    return true;\n}}",
            mover=mover,
        )
        compute = []
        if len(self.movers) == 1:
            compute.append(Line("if (compute.arg(kArgIncluded) == 0) {\n    break;\n}"))
        return [movement], compute

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
            and self.is_call_of(cast.value, name, "cast", 1)
            and self.is_call_of(arrive.value, name, "arrive", 0)
        ):
            self.refuse(node, "the sender casts a block, the others take it", drawn)
        read = cast.value.args[0]
        if not (
            isinstance(read, ast.Call)
            and isinstance(read.func, ast.Attribute)
            and read.func.attr == spelling("read")
            and not read.args
            and isinstance(read.func.value, ast.Call)
            and self.is_range(read.func.value)
            and len(read.func.value.args) == 2
        ):
            self.refuse(cast, "a cast sends pages read from a tensor", drawn)
        value = cast.targets[0].id
        self.check_name(cast, value)
        pages = self.range_of(read.func.value)
        count = self.constant_int(read.func.value.args[1])
        dtype, page_bytes = self.tensor_format(cast, pages.tensor)
        self.moved(cast, pages.tensor, "broadcast's read")
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
                (broadcast.pattern, buffer, broadcast.window)
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
        if self.loops:
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
        address = f"address{buffer.index_a}"
        tensor = camel(pages.tensor)
        text = (
            f"if ({name}.sender()) {{\n"
            f"    const uint32_t {address} = {name}.reserve({block});\n"
            f"    if ({address} == 0 || !km::read_pages(movement, command, "
            f"{tensor}({pages.tensor}), {pages.first}, {count}, {address}) || "
            f"!{name}.cast({block})) {{\n"
            "        return false;\n"
            "    }\n"
            f"}} else if (!{name}.arrive({block})) {{\n"
            "    return false;\n"
            "}"
        )
        uses = range_uses(pages, self) | {name} | text_uses(block, self)
        self.scope[value] = Value("arrived", buffer.name, buffer=buffer)
        self.wait(node, f"mover {broadcast.mover}", WAIT_FREE)
        self.wait(node, f"mover {broadcast.mover}", WAIT_DONE)
        self.wait(node, f"mover {broadcast.mover}", WAIT_VALID)
        movement = [Line(text, uses, mover=broadcast.mover)]
        compute = []
        if not self.loops:
            movement_end, compute = self.broadcast_end(name, "1", set())
            movement += movement_end
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

    def broadcast_end(self, name, blocks, uses):
        """After a broadcast's last block: its mover's finish, compute's padding."""
        broadcast = self.scope[name]
        buffer = broadcast.buffer.name
        movement = Line(
            f"if (!{name}.finish()) {{\n    return false;\n}}",
            {name},
            mover=broadcast.mover,
        )
        compute = Line(
            f"ok = ok && km::broadcast_end(compute, {buffer}, {buffer}Pages, "
            f"{buffer}Window, {blocks});",
            uses | {"ok"},
        )
        return [movement], [compute]

    def block_loop(self, node, value):
        """`for tile in v:` over a block a broadcast brought: compute waits for it,
        takes its pages in order, frees its slot."""
        if not isinstance(node.target, ast.Name):
            self.refuse(node, "one name a tile", "write for tile in block")
        buffer = value.buffer.name
        page = f"page{value.buffer.index_a}"
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
                f"ok = ok && km::broadcast_receive(compute, {buffer}, {buffer}Pages);",
                {"ok"},
            ),
            Block(
                "if (ok) {",
                "}",
                [loop, Line(f"km::broadcast_done({buffer}, " f"{buffer}Pages);")],
            ),
        ]
        return [], compute

    # Inline bodies and results -----------------------------------------------------

    def inline(self, node, call, name=None):
        """`km.cxx(text, takes=[...], gives=..., on=...)`: the C++ as written, on its
        CPU, the names it takes bound before it; `name = km.cxx(..., gives=int)` the
        word the body sets, `gives=km.Tile` the tile it leaves in DST 0 (compute)."""
        if not call.args or not isinstance(call.args[0], ast.Constant):
            self.refuse(node, "an inline body is a string", 'write km.cxx("...")')
        text = textwrap.dedent(call.args[0].value).strip("\n")
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
        on, takes, gives = "compute", [], None
        for keyword_ in call.keywords:
            value = keyword_.value
            if keyword_.arg == spelling("on"):
                if not (
                    isinstance(value, ast.Constant)
                    and value.value in ("mover_a", "mover_b")
                ):
                    self.refuse(
                        node,
                        "on= names a mover (a body without it is compute's)",
                        'write on="mover_a" or on="mover_b"',
                    )
                on = value.value
            elif keyword_.arg == spelling("takes"):
                if not isinstance(value, ast.List):
                    self.refuse(node, "takes= is a list of names", "write takes=[a]")
                takes = list(value.elts)
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
                    "use takes=, gives= and on=",
                )
        if (gives is None) != (name is None):
            self.refuse(
                node,
                "a body that gives a value is assigned, one that gives none is not",
                "write `x = km.cxx(..., gives=int)`, or km.cxx(...) on its own line",
            )
        mover = None if on == "compute" else ("A" if on == "mover_a" else "B")
        if gives == "tile" and mover is not None:
            self.refuse(
                node, "a tile is compute's", "drop on= for a body giving a tile"
            )
        uses, bound, before, after = set(), [], [], []
        for item in takes:
            field = self.field_name(item)
            if field:
                self.take_field(node, field, mover, bound, before, after)
                continue
            uses |= self.take(node, item, mover, bound, before)
        kernel = "compute" if mover is None else "movement"
        for line in lines[:included]:
            if line.strip() not in self.compiled.body_includes[kernel]:
                self.compiled.body_includes[kernel].append(line.strip())
        start = call.args[0].lineno + 1 + included
        body = f'#line {start} "{self.compiled.source}"\n{text}\n#line RESUME'
        if bound:
            inner = "\n".join("    " + line for line in [*bound, *body.split("\n")])
            body = "{\n" + inner + "\n}"
        if gives == "int":
            self.check_name(node, name)
            body = f"uint32_t {name} = 0;\n{body}"
            self.scope[name] = Value("word", name, mover=mover or "compute")
        elif gives == "tile":
            self.check_name(node, name)
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
        lines = [*before, Line(body, uses, mover=mover), *after]
        if mover is None:
            return [], lines
        return lines, []

    def take(self, node, item, mover, bound, before):
        """A name an inline body takes: an int, partition or pattern by its own name;
        a tile written to a page (on mover B) as its buffer, once compute packed it."""
        if not isinstance(item, ast.Name) or item.id not in self.scope:
            self.refuse(node, "takes= names the handler's values", "take names")
        value = self.scope[item.id]
        if value.kind in ("int", "partition", "pattern"):
            return {item.id}
        if value.kind == "streamed" and mover is None:  # its buffer and place
            buffer, index = spelling("tile_buffer"), spelling("tile_index")
            bound.append(f"const uint32_t {item.id}{buffer} = {value.cxx};")
            bound.append(f"const uint32_t {item.id}{index} = {value.first};")
            return set()
        if value.kind == "word" and value.mover == (mover or "compute"):
            return {item.id}
        if value.kind in ("written", "reduced") and value.buffer is not None:
            if mover != "B":
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
            f"{item.id} is a {value.kind}: a body takes ints, partitions, patterns, "
            "words and written tiles",
            "pass those",
        )

    def page_wait(self, value):
        """The opening of mover B's wait for the page compute packs of `value`."""
        if value.kind == "reduced":
            return f"if (!km::reduce_result(movement, {value.buffer.name}, 1)) {{"
        return f"if (!movement.wait_front({value.buffer.name}, 1)) {{"

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


def range_uses(pages, compiler, tensor=True):
    """The names a range's C++ reads (`tensor`: its tensor's word too)."""
    found = {pages.tensor} if pages.tensor and tensor else set()
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
    "rank",
    "size",
    "group",
    "self_x",
    "self_y",
    "rect_columns",
    "rect_rows",
)


def pattern_uses(pattern):
    return {pattern} if pattern else set()


def add_tile(buffer, page, dst, scratch):
    """DST dst += page `page` of buffer (copied to DST scratch, added by the SFPU)."""
    return (
        f"copy_init({buffer});\n"
        f"copy_tile({buffer}, {page}, {scratch});\n"
        "add_binary_tile_init();\n"
        f"add_binary_tile({dst}, {scratch}, {dst});"
    )


def pack(dst, half):
    """DST tile `dst` packed into one page of `half`; DST released."""
    return (
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"if (ok && compute.reserve_back({half}, 1)) {{\n"
        f"    pack_tile({dst}, {half});\n"
        f"    cb_push_back({half}, 1);\n"
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
