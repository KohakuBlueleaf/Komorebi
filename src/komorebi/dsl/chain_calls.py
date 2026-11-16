"""A chain stated with a class's handlers: each call (`Steps.scale(x, y)` in
`run.chain`'s list, or a line of a `km.chain` method) made the framework's call
(`komorebi.chain.Call`), its description derived from its handler's lines.
"""

import ast
import inspect
from dataclasses import dataclass, field
from typing import NamedTuple

from komorebi.chain import calls as kc
from komorebi.dsl import compiler, first_use, front
from komorebi.dsl.names import spelling
from komorebi.errors import ProgramError

TREE = 1  # km::kReduceTree: a sum up a tree, the chain's credits' form


class Undescribed(Exception):
    """A handler line whose reads or writes a chain cannot be told."""


@dataclass
class Access:
    """A line that reads or writes a tensor parameter: its pages from `first`, `count`
    of them (expressions), shared over the group's ranks (self.partition's), a
    rectangle's rows and columns, or under `if self.rank == self.leader:`."""

    kind: str  # "read" or "write"
    tensor: str
    line: int
    first: ast.expr = None
    count: ast.expr = None
    shared: bool = False
    rect: tuple = None  # (r0, r1, c0, c1) of t.tiles[r0:r1, c0:c1]
    # under a leader's if: True (rank 0 alone), "reduce" (self.leader after the
    # reduce: each group's root) or the G of `self.rank % G == 0`
    leader: object = False
    prefetch: bool = None
    stream: bool = False  # a write stream's (movement.move writes it to DRAM)
    over: tuple = ()  # (name, range's arguments) of the range loops around it


@dataclass
class Lines:
    """What a handler's lines read and write, its sum, its names; or why not."""

    accesses: list = field(default_factory=list)
    reduce: tuple = None  # (line, form expression or None, group size or None)
    names: dict = field(default_factory=dict)  # name -> its expression
    loops: set = field(default_factory=set)  # names of loop variables
    body_reads: list = field(default_factory=list)  # (tensor, line): reads= of a body
    body_writes: list = field(default_factory=list)  # (tensor, line): writes=
    refused: str = None


def handler_lines(method):
    """The reads, writes and sum of a handler, read from its lines (cached on it)."""
    cached = getattr(method, "chain_lines", None)
    if cached is None:
        cached = LineReader(method).read()
        method.chain_lines = cached
    return cached


def streamed(access):
    """A stream's access, marked when it writes (its pages leave by the stream)."""
    access.stream = access.kind == "write"
    return access


class LineReader:
    """Reads a handler's statements for what each core reads and writes."""

    def __init__(self, method):
        self.method = method
        self.definition, _, _ = compiler.source_of(method.function)
        self.tensors = [p for p, kind in method.words or () if kind == "tensor"]
        self.lines = Lines()
        self.partitions = {}  # name -> (first, count) of self.partition(...)
        self.streams = {}  # name -> the Access of a stream made on a line of its own
        self.over = []  # (name, range's arguments) of the range loops open here

    def read(self):
        try:
            self.body(self.definition.body, False)
            self.uncovered()
        except Undescribed as error:
            self.lines.refused = str(error)
        return self.lines

    def refuse(self, node, why):
        raise Undescribed(f"line {node.lineno}: {why}")

    def body(self, statements, leader):
        for node in statements:
            self.statement(node, leader)

    def statement(self, node, leader):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                self.assign(node, target.id, node.value, leader)
                return
            self.reads_in(node, node.value, leader)
            return
        if isinstance(node, ast.AugAssign | ast.AnnAssign | ast.Pass | ast.Return):
            return
        if isinstance(node, ast.For):
            self.loop(node, leader)
            return
        if isinstance(node, ast.If):
            root = self.leader_of(node.test)
            if root is not None:
                self.body(node.body, root)
                if self.touches(node.orelse):
                    self.refuse(node, "a read or write in the else of a leader's if")
                return
            if self.touches(node.body) or self.touches(node.orelse):
                self.refuse(
                    node,
                    "a read or write under an if on the command's words: which cores "
                    "read or write it is not known before the command runs",
                )
            return
        if isinstance(node, ast.With):
            self.body(node.body, leader)
            return
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self.expression(node, node.value, leader)
            return
        if self.touches([node]):
            self.refuse(node, "a read or write of a form the chain does not describe")

    def assign(self, node, name, value, leader):
        if self.is_self_call(value, "partition"):
            tensor, first, count = self.pages(node, value.args[0])
            self.partitions[name] = (first, count)
            return
        if self.is_km_call(value, "stream"):
            self.streams[name] = self.stream(node, value, leader)
            return
        if self.is_km_call(value, "reduce"):
            self.reduction(node, value)
            return
        if self.is_km_call(value, "cxx"):
            self.body_takes(node, value)
            return
        if self.touches_expression(value):
            self.reads_in(node, value, leader)
            return
        self.lines.names[name] = value

    def loop(self, node, leader):
        source = node.iter
        if self.is_km_call(source, "stream"):
            self.stream(node, source, leader)
        elif isinstance(source, ast.Call) and getattr(source.func, "id", None) == "zip":
            for one in source.args:
                if not self.is_km_call(one, "stream"):
                    self.refuse(node, "a zip of something other than streams")
                self.stream(node, one, leader)
        elif isinstance(source, ast.Call) and getattr(source.func, "id", None) == (
            "range"
        ):
            if isinstance(node.target, ast.Name) and not source.keywords:
                self.over.append((node.target.id, tuple(source.args)))
        elif self.touches_expression(source):
            self.reads_in(node, source, leader)
        for name in ast.walk(node.target):
            if isinstance(name, ast.Name):
                self.lines.loops.add(name.id)
        opened = len(self.over)
        self.body(node.body, leader)
        if (
            self.over
            and opened
            and self.over[opened - 1][0] == getattr(node.target, "id", None)
        ):
            del self.over[opened - 1 :]

    def expression(self, node, call, leader):
        func = call.func
        method = func.attr if isinstance(func, ast.Attribute) else None
        if method == spelling("write") and self.is_pages(func.value):
            tensor, first, count = self.pages(node, func.value)
            self.add("write", tensor, node, first, count, leader=leader)
            return
        if method == spelling("write") and self.is_rectangle(func.value):
            rows, columns = self.rectangle(node, func.value.slice)
            access = self.add("write", func.value.value.value.id, node, leader=leader)
            access.rect = (*rows, *columns)
            return
        if method == spelling("drain") and self.is_km_call(func.value, "stream"):
            self.stream(node, func.value, leader)
            return
        filled = call.args and self.is_km_call(call.args[0], "stream")
        if method == spelling("fill") and filled:
            self.stream(node, call.args[0], leader)
            return
        if self.is_km_call(call, "cxx"):
            self.body_takes(node, call)
            return
        if self.is_km_call(call, "reduce"):  # a state merged in place (combine=)
            self.reduction(node, call)
            return
        if isinstance(func, ast.Attribute) and func.attr in ("push", "end"):
            return
        if self.touches_expression(call):
            self.refuse(node, "a read or write of a form the chain does not describe")

    def body_takes(self, node, call):
        """A body's tensors: each stated in reads= (an outside input of the call) or
        writes= (written by the call), else refused."""
        stated = {spelling("body_reads"): set(), spelling("body_writes"): set()}
        for keyword in call.keywords:
            if keyword.arg in stated and isinstance(keyword.value, ast.List):
                stated[keyword.arg] = {
                    item.id for item in keyword.value.elts if isinstance(item, ast.Name)
                }
        reads, writes = stated.values()
        for keyword in call.keywords:
            if keyword.arg != spelling("takes"):
                continue
            if not isinstance(keyword.value, ast.List):
                continue
            for taken in keyword.value.elts:
                if not (isinstance(taken, ast.Name) and taken.id in self.tensors):
                    continue
                if taken.id in reads:
                    self.lines.body_reads.append((taken.id, node.lineno))
                if taken.id in writes:
                    self.lines.body_writes.append((taken.id, node.lineno))
                if taken.id not in reads | writes:
                    self.refuse(
                        node,
                        f"an inline body takes tensor {taken.id}: what it reads or "
                        "writes is the body's own C++, which the build does not read; "
                        f"state it at the body: {spelling('body_reads')}=[{taken.id}] "
                        "(an outside input of the call) or "
                        f"{spelling('body_writes')}=[{taken.id}] (written by the call)",
                    )

    def stream(self, node, call, leader):
        """A km.Stream line: a read or (write=True) a write of its pages."""
        write, prefetch = False, None
        for keyword in call.keywords:
            if keyword.arg == spelling("write_stream"):
                write = bool(getattr(keyword.value, "value", False))
            if keyword.arg == spelling("prefetch"):
                prefetch = keyword.value.value
        if len(call.args) != 1:
            self.refuse(node, "a stream of a form the chain does not describe")
        source = call.args[0]
        kind = "write" if write else "read"
        if isinstance(source, ast.Subscript) and isinstance(source.value, ast.Name):
            part = source.slice
            if isinstance(part, ast.Name) and part.id in self.partitions:
                first, count = self.partitions[part.id]
                return streamed(
                    self.add(
                        kind,
                        source.value.id,
                        node,
                        first,
                        count,
                        True,
                        leader,
                        prefetch,
                    )
                )
        if (
            isinstance(source, ast.Subscript)
            and isinstance(source.value, ast.Attribute)
            and source.value.attr == spelling("tiles")
            and isinstance(source.value.value, ast.Name)
        ):
            rows, columns = self.rectangle(node, source.slice)
            access = self.add(kind, source.value.value.id, node, leader=leader)
            access.rect, access.prefetch = (*rows, *columns), prefetch
            return streamed(access)
        if self.is_pages(source):
            tensor, first, count = self.pages(node, source)
            return streamed(
                self.add(kind, tensor, node, first, count, False, leader, prefetch)
            )
        if isinstance(source, ast.Attribute) and source.attr not in self.tensors:
            return None  # a field's pages (km.Stream(self.f, ...)): no tensor
        self.refuse(node, "a stream of a form the chain does not describe")
        return None

    def is_rectangle(self, node):
        """Whether `node` is a tensor parameter's rectangle, `t.tiles[r0:r1, c0:c1]`."""
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == spelling("tiles")
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in self.tensors
        )

    def rectangle(self, node, index):
        if not (isinstance(index, ast.Tuple) and len(index.elts) == 2):
            self.refuse(node, "a rectangle of rows and columns, t.tiles[r0:r1, c0:c1]")
        sides = []
        for side in index.elts:
            if not isinstance(side, ast.Slice) or side.step is not None:
                self.refuse(node, "a rectangle with a step is not described")
            sides.append((side.lower, side.upper))
        return sides

    def reduction(self, node, call):
        form, size = None, None
        for keyword in call.keywords:
            if keyword.arg == spelling("form"):
                form = keyword.value
            elif keyword.arg == spelling("group"):
                size = keyword.value
        if self.lines.reduce is not None:
            self.refuse(node, "a second reduce")
        self.lines.reduce = (node.lineno, form, size)

    def reads_in(self, node, value, leader):
        """A page's words read (`t.pages(i, 1).read()`, `.entries(...)`)."""
        for inner in ast.walk(value):
            if self.is_pages(inner):
                tensor, first, count = self.pages(node, inner)
                self.add("read", tensor, node, first, count, leader=leader)
                return
        self.refuse(node, "a read of a form the chain does not describe")

    def add(
        self,
        kind,
        tensor,
        node,
        first=None,
        count=None,
        shared=False,
        leader=False,
        prefetch=None,
    ):
        access = Access(kind, tensor, node.lineno, first, count, shared)
        access.leader, access.prefetch = leader, prefetch
        access.over = tuple(self.over)
        self.lines.accesses.append(access)
        return access

    def uncovered(self):
        """Every use of a tensor parameter's pages is a line read above."""
        covered = {access.line for access in self.lines.accesses}
        for node in ast.walk(self.definition):
            if isinstance(node, ast.stmt) and node is not self.definition:
                if isinstance(node, ast.For | ast.If | ast.With):
                    continue
                uncovered = node.lineno not in covered
                if (
                    uncovered
                    and self.touches([node])
                    and not self.is_partition_line(node)
                ):
                    self.refuse(
                        node, "a read or write of a form the chain does not describe"
                    )

    def is_partition_line(self, node):
        return isinstance(node, ast.Assign) and self.is_self_call(
            node.value, "partition"
        )

    # forms

    def is_pages(self, node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == spelling("pages")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.tensors
        )

    def pages(self, node, call):
        if not self.is_pages(call) or len(call.args) != 2:
            self.refuse(node, "a page range of a tensor, t.pages(first, count)")
        return call.func.value.id, call.args[0], call.args[1]

    def is_self_call(self, node, kind):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == spelling(kind)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        )

    def is_km_call(self, node, kind):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == spelling(kind)
            and isinstance(node.func.value, ast.Name)
        )

    def leader_of(self, test):
        """Whose if it is: rank 0's (True), each group's root's ("reduce" for
        self.leader after the reduce, else the G of `self.rank % G == 0`), or None."""
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
        ):
            return None
        sides = [test.left, test.comparators[0]]
        names = [getattr(side, "attr", None) for side in sides]
        zero = [isinstance(side, ast.Constant) and side.value == 0 for side in sides]
        if names in (["rank", "leader"], ["leader", "rank"]):
            return True if self.lines.reduce is None else "reduce"
        if names[0] == "rank" and zero[1]:
            return True
        left = sides[0]
        if (
            zero[1]
            and isinstance(left, ast.BinOp)
            and isinstance(left.op, ast.Mod)
            and getattr(left.left, "attr", None) == spelling("rank")
        ):
            return left.right
        return None

    def touches(self, statements):
        return any(self.touches_expression(node) for node in statements)

    def touches_expression(self, node):
        for inner in ast.walk(node):
            named = isinstance(inner, ast.Attribute | ast.Subscript) and isinstance(
                inner.value, ast.Name
            )
            if not named or inner.value.id not in self.tensors:
                continue
            taken = (spelling("pages"), spelling("tiles"), "items")
            if isinstance(inner, ast.Subscript) or inner.attr in taken:
                return True
        return False


class Evaluator:
    """The value of a handler's int expression for one core of one call: the
    call's words, the class's and the handler's constants, the core's rank."""

    def __init__(self, method, lines, values, program, rank, size):
        self.method, self.lines, self.values = method, lines, values
        self.program, self.rank, self.size = program, rank, size
        self.globals = compiler.handler_names(method)
        self.bound = {}  # a range loop's name -> its value (rectangle)

    def value(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.Name):
            return self.name(node)
        if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == (
            "self"
        ):
            return self.attribute(node)
        if isinstance(node, ast.BinOp):
            left, right = self.value(node.left), self.value(node.right)
            operations = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.FloorDiv: lambda a, b: a // b,
                ast.Mod: lambda a, b: a % b,
            }
            for kind, operation in operations.items():
                if isinstance(node.op, kind):
                    return operation(left, right)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in (
            "min",
            "max",
        ):
            numbers = [self.value(arg) for arg in node.args]
            return min(numbers) if node.func.id == "min" else max(numbers)
        raise Undescribed(
            f"line {node.lineno}: {ast.unparse(node)} is not a number the call's "
            "words and the class's constants give"
        )

    def name(self, node):
        name = node.id
        if name in self.bound:
            return self.bound[name]
        if name in self.lines.loops:
            raise Undescribed(
                f"line {node.lineno}: what it reads or writes changes with the "
                f"loop's {name}"
            )
        if name in self.values and isinstance(self.values[name], int):
            return int(self.values[name])
        if isinstance(self.values.get(name), front.AtSubmit):
            raise Undescribed(
                f"line {node.lineno}: what it reads or writes changes with {name}, "
                f"given at each submit ({self.values[name]!r}); pass {name} itself"
            )
        if name in self.lines.names:
            return self.value(self.lines.names[name])
        value = self.globals.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise Undescribed(
            f"line {node.lineno}: {name} is not a word of the call or a constant"
        )

    def attribute(self, node):
        if node.attr == spelling("rank"):
            return self.rank
        if node.attr == spelling("size"):
            return self.size
        if node.attr == spelling("leader"):
            return 0
        values = getattr(self.program, "constant_values", None) or {}
        if node.attr in values:
            return values[node.attr]
        raise Undescribed(
            f"line {node.lineno}: self.{node.attr} is not known before the command"
        )


def rectangle_of(evaluator, access):
    """(r0, r1, c0, c1) of a rectangle access for one core; inside range loops of a
    constant count, the rectangle all the loops' values make together (refused
    when they make no rectangle)."""
    used = {
        n.id for side in access.rect for n in ast.walk(side) if isinstance(n, ast.Name)
    }
    loops = [(name, args) for name, args in access.over if name in used]
    if not loops:
        return tuple(evaluator.value(side) for side in access.rect)
    cells = set()

    def walk(at):
        if at == len(loops):
            r0, r1, c0, c1 = (evaluator.value(side) for side in access.rect)
            cells.update((r, c) for r in range(r0, r1) for c in range(c0, c1))
            return
        name, args = loops[at]
        for value in range(*(evaluator.value(arg) for arg in args)):
            evaluator.bound[name] = value
            walk(at + 1)
        evaluator.bound.pop(name, None)

    walk(0)
    if not cells:
        raise Undescribed(f"line {access.line}: a loop of no turns reads or writes")
    rows, columns = {r for r, _ in cells}, {c for _, c in cells}
    box = (min(rows), max(rows) + 1, min(columns), max(columns) + 1)
    if len(cells) != (box[1] - box[0]) * (box[3] - box[2]):
        raise Undescribed(
            f"line {access.line}: {access.tensor}'s rectangles over the loop's "
            f"{loops[0][0]} make no rectangle together"
        )
    return box


@dataclass
class Described:
    """A call's description from its handler's lines (tensors by their object)."""

    handler: object
    words: list
    group: tuple
    size: int
    width: int
    output: object = None
    reads: dict = field(default_factory=dict)  # tensor -> (k, ranges, lines)
    sends: bool = False
    prefetch: dict = field(default_factory=dict)  # tensor -> [(line, True/False)]
    marks: dict = field(default_factory=dict)  # tensor -> "past" or "direct"
    name: str = None
    tensor_bits: list = field(default_factory=list)  # (tensor, its parameter's bit)
    mask_word: int = None
    outputs: list = field(default_factory=list)  # its other outputs: (tensor, width)
    body_reads: dict = field(default_factory=dict)  # tensor id -> (tensor, lines)
    body_writes: dict = field(default_factory=dict)  # tensor id -> (tensor, lines)
    unranged: dict = field(default_factory=dict)  # tensor -> why it is no range
    program: object = None  # the program whose handler it calls
    streamed: int = None  # the line of a write stream of its output, if any
    covering: dict = field(default_factory=dict)  # tensor id -> line of a cover range
    rows: dict = field(default_factory=dict)  # output id -> (r0, r1) its roots write
    row_reads: dict = field(default_factory=dict)  # tensor id -> RowRead (columns)
    in_place: bool = False  # its handler's direct input lands in its stream's halves
    submitted: tuple = None  # (name, its word's place 0 to 9): km.at_submit's word
    word_tensors: dict = field(default_factory=dict)  # a tensor word's place -> tensor


def describe(run, call, group):
    """A call's description: its handler built for its constants, its words, and
    what each core reads and writes, from its handler's lines; refused
    (ChainRefused) when a line cannot be described, saying which and why."""
    values, marks = [], {}
    places = call.method.places()
    for place, value in zip(places, call.values):
        if isinstance(value, front.Marked):
            marks[id(value.tensor)] = (value.how, value.tensor, place)
            value = value.tensor
        values.append(value)
    named = {}
    for name, value in call.named.items():
        if isinstance(value, front.Marked):
            marks[id(value.tensor)] = (value.how, value.tensor, name)
            value = value.tensor
        named[name] = value
    at = {
        place: value
        for place, value in [*zip(places, values), *named.items()]
        if isinstance(value, front.AtSubmit)
    }
    given_at = dict(zip(places, values))  # km.at_submit's object kept: a read of it
    given_at.update(named)  # refuses (Evaluator)
    values = [0 if isinstance(v, front.AtSubmit) else v for v in values]
    named = {k: 0 if isinstance(v, front.AtSubmit) else v for k, v in named.items()}
    program = getattr(call.handler, "program", None)
    if program is None:
        program = run.programs[run._handler(call.handler)[1]]
    handler, method = merged_call(program, call.handler, call.method)
    if front.constant_parameters(method.function):  # the handler built for them
        given_constants = method.constants_given(values, named)
        number, method = first_use.constant_handler(program, method, given_constants)
        handler = front.ClassHandler(number, program, method)
    label = f"{type(program).__name__}.{call.method.name}"
    submitted = submitted_word(handler, method, at, label)
    words = handler.args(*values, **named)
    if not getattr(program, "chain_handler", 0):
        raise kc.ChainRefused(
            f"{type(program).__name__}.{method.name}: its program was made without "
            f"the chain's walk; make it with {spelling('chain')}=True (or give the "
            f"class a km.{spelling('chain')} method)"
        )
    given = given_at
    group = tuple(call.cores) if call.cores is not None else tuple(group)
    lines = handler_lines(method)
    if lines.refused is not None:
        raise kc.ChainRefused(
            f"{label}, {lines.refused}; it still runs as a command (run.submit)"
        )
    try:
        described = derived(method, lines, given, program, group, label)
    except Undescribed as error:
        raise kc.ChainRefused(
            f"{label}, {error}; it still runs as a command (run.submit)"
        ) from None
    described.handler, described.words = handler, words
    described.program = program
    tensors = [p for p, kind in method.words if kind == "tensor"]
    described.tensor_bits = [
        (id(given[p]), 1 << i) for i, p in enumerate(tensors) if p in given
    ]
    described.mask_word = len(method.words) if len(method.words) < 10 else None
    for key, (how, _tensor, place) in marks.items():
        if how == "direct" and place not in method.direct:
            raise kc.ChainRefused(
                f"{label}: km.{spelling('direct')}({place}) marks {place}, which the "
                f"handler does not declare (km.handler({spelling('direct')}=(...)) "
                f"names {', '.join(method.direct) or 'none'})"
            )
        described.marks[key] = how
    described.in_place = bool(getattr(method, "in_place", False))
    described.submitted = submitted
    described.word_tensors = {
        place: (
            given[parameter].tensor
            if isinstance(given[parameter], front.Marked)
            else given[parameter]
        )
        for place, (parameter, kind) in enumerate(method.words)
        if kind == "tensor" and parameter in given
    }
    return described


def submitted_word(handler, method, at, label):
    """(name, its word's place) of a call's km.at_submit argument, or None: one a
    call, at an int word of the handler (not a constant, a tensor or a pattern)."""
    if not at:
        return None
    if len(at) > 1:
        raise kc.ChainRefused(
            f"{label}: km.{spelling('at_submit')} at {', '.join(at)}: one word a "
            "call is given at submit"
        )
    ((place, value),) = at.items()
    if place in front.sized_parameters(method.function):
        raise kc.ChainRefused(
            f"{label}: {place} is a size given at each command, whose most a "
            "chain's per-submit word is not checked against: give it a value in the "
            "call"
        )
    if place in front.constant_parameters(method.function):
        raise kc.ChainRefused(
            f"{label}: {place} is a constant, fixed when the handler is built; "
            f"km.{spelling('at_submit')} gives a word"
        )
    places = [p for p, _kind in handler.method.words]
    kind = dict(handler.method.words).get(place)
    if kind in ("tensor", "pages", "columns", "region", "bank", "pattern", "float",
                None):  # fmt: skip
        raise kc.ChainRefused(
            f"{label}: {place} is a {kind or 'parameter with no word'}; "
            f"km.{spelling('at_submit')} gives an int word"
        )
    return value.name, places.index(place)


def merged_call(program, handler, method):
    """(handler, method) of a call: a merged class's (komorebi.dsl.merged) copy of a
    handler named through one of its classes (Scales.scale), its words the merged
    program's; else as given."""
    if getattr(program, "merged_sources", None) is None:
        return handler, method
    if not isinstance(handler, front.HandlerMethod):
        return handler, method
    mine = program.handler(handler)
    return mine, mine.method


def derived(method, lines, given, program, group, label):
    """The description of a call on `group` from the reads and writes of `lines`."""
    first, count = group
    size, sends = 1, False
    if lines.reduce is not None:
        line, form, stated = lines.reduce
        evaluator = Evaluator(method, lines, given, program, 0, count)
        if form is not None:
            value = (
                form.value if isinstance(form, ast.Constant) else evaluator.value(form)
            )
            if value not in ("tree", TREE):
                raise Undescribed(
                    f'line {line}: a chain\'s sums go up a tree (form="tree"), and '
                    "this reduce is by owner"
                )
        size = count if stated is None else evaluator.value(stated)
        sends = size > 1
    if count % size:
        raise Undescribed(f"{count} cores do not divide into groups of {size}")
    described = Described(None, None, group, size, 0, name=label)
    described.sends = sends
    writes = [a for a in lines.accesses if a.kind == "write"]
    outputs = list(dict.fromkeys(a.tensor for a in writes))  # in the lines' order
    for number, name in enumerate(outputs):
        mine = [a for a in writes if a.tensor == name]
        try:
            width = output_width(method, lines, mine, given, program, group, size)
        except Undescribed:
            if not any(
                several_rows(method, lines, a, given, program, group) for a in mine
            ):
                raise
            width, rows = row_width(method, lines, mine, given, program, group, size)
            described.rows[id(given[name])] = rows
        if number == 0:
            described.output, described.width = given[name], width
            described.streamed = next((a.line for a in mine if a.stream), None)
            continue
        if id(given[name]) in {id(t) for t, _ in described.outputs} | {
            id(described.output)
        }:
            raise Undescribed(
                f"line {mine[0].line}: {name} is the same tensor as another output"
            )
        described.outputs.append((given[name], width))
    for kind, held in (
        ("body_reads", described.body_reads),
        ("body_writes", described.body_writes),
    ):
        for name, line in getattr(lines, kind):
            entry = held.setdefault(id(given[name]), (given[name], []))
            entry[1].append(line)
    for access in (a for a in lines.accesses if a.kind == "read"):
        tensor = given[access.tensor]
        if access.rect is not None:
            row_read(described, access, tensor, method, lines, given, program, size)
        if unranged_read(described, access, tensor, method, lines, given, program):
            continue
        k, ranges = read_shape(method, lines, access, given, program, group, size)
        if getattr(ranges, "covering", False):
            described.covering.setdefault(id(tensor), access.line)
        held = described.reads.setdefault(id(tensor), [tensor, None, [], []])
        if k is not None:
            if held[1] is not None or held[2]:
                raise Undescribed(
                    f"line {access.line}: {access.tensor} read twice as a share of K"
                )
            held[1] = k
        else:
            held[2] += [r for r in ranges if r not in held[2]]
        held[3].append((access.line, access.prefetch))
        if held[1] is not None and held[2]:
            raise Undescribed(
                f"line {access.line}: {access.tensor} read both as a share of K and "
                "by ranges"
            )
        if len(held[2]) > 3:
            raise Undescribed(
                f"line {access.line}: {access.tensor} read in more than three ranges"
            )
    return described


def rectangles(method, lines, access, given, program, group):
    """(r0, r1, c0, c1) of a rectangle access for each core of `group`, by rank."""
    out = []
    for rank in range(group[1]):
        evaluator = Evaluator(method, lines, given, program, rank, group[1])
        out.append(rectangle_of(evaluator, access))
    return out


def several_rows(method, lines, access, given, program, group):
    """Whether a rectangle access is, on some core, several rows not whole ones."""
    if access.rect is None:
        return False
    columns = given[access.tensor].shape[-1] // 32
    return any(
        r1 - r0 > 1 and (c0, c1) != (0, columns)
        for r0, r1, c0, c1 in rectangles(method, lines, access, given, program, group)
    )


class RowRead(NamedTuple):
    """A rectangle read in column space: (K share or None, ranges of columns), each
    rank's rows, the line."""

    k: object
    ranges: tuple
    rows: list
    line: int


def column_spans(method, lines, access, given, program, group):
    """(first column, columns) of a rectangle access for each core, by rank."""
    return [
        (c0, c1 - c0)
        for _, _, c0, c1 in rectangles(method, lines, access, given, program, group)
    ]


def row_read(described, access, tensor, method, lines, given, program, size):
    """A rectangle read's column form (RowRead), kept beside its page form: used when
    a call of the chain writes the tensor in rows (placed)."""
    group = described.group
    try:
        k, ranges = read_shape(
            method, lines, access, given, program, group, size, column_spans
        )
    except Undescribed:
        return
    rows = [(r0, r1) for r0, r1, _, _ in rectangles(
        method, lines, access, given, program, group
    )]  # fmt: skip
    held = described.row_reads.get(id(tensor))
    if held is not None:  # a second read: kept when it is the same form
        if (held.k, held.ranges) != (k, tuple(ranges or ())):
            described.row_reads[id(tensor)] = None
        return
    described.row_reads[id(tensor)] = RowRead(k, tuple(ranges or ()), rows, access.line)


def row_width(method, lines, writes, given, program, group, size):
    """(width in columns, (r0, r1)) of an output each group's root writes as rows
    r0 .. r1 - 1 of its block of columns g x width on (in one write or a piece a
    row); refused when the roots' rows differ or the block has a gap."""
    cells = {}  # group -> {(row, column)}
    for access in writes:
        if access.rect is None or (size > 1 and not access.leader):
            raise Undescribed(
                f"line {access.line}: {access.tensor} is written in rows; every write "
                "of it is a rectangle, by each group's root"
            )
        for rank, (r0, r1, c0, c1) in enumerate(
            rectangles(method, lines, access, given, program, group)
        ):
            if rank % size == 0:
                mine = cells.setdefault(rank // size, set())
                mine |= {(r, c) for r in range(r0, r1) for c in range(c0, c1)}
    width, rows, line = None, None, writes[0].line
    for index, mine in sorted(cells.items()):
        found_rows = sorted({r for r, _ in mine})
        found_columns = sorted({c for _, c in mine})
        width = len(found_columns) if width is None else width
        here = (found_rows[0], found_rows[-1] + 1)
        if rows is not None and here != rows:
            raise Undescribed(
                f"line {line}: the root of group {index} writes {writes[0].tensor} "
                f"rows {here[0]} to {here[1] - 1}, the root of group 0 rows "
                f"{rows[0]} to {rows[1] - 1}; a chain's call in rows writes the same "
                "rows a root"
            )
        rows = here
        whole = {
            (r, c)
            for r in range(*rows)
            for c in range(index * width, (index + 1) * width)
        }
        if mine != whole:
            raise Undescribed(
                f"line {line}: group {index} writes {writes[0].tensor} cells other "
                f"than rows {rows[0]} to {rows[1] - 1} x columns {index * width} to "
                f"{(index + 1) * width - 1}; a chain's call writes block g of its "
                "output, columns g x width on, over the same rows"
            )
    return width or 0, rows


def spans(method, lines, access, given, program, group):
    """(first page, count) of `access` for each core of `group`, by rank."""
    first, count = group
    out = []
    for rank in range(count):
        evaluator = Evaluator(method, lines, given, program, rank, count)
        if access.rect is not None:
            r0, r1, c0, c1 = rectangle_of(evaluator, access)
            columns = given[access.tensor].shape[-1] // 32
            if r1 - r0 == 1:
                out.append((r0 * columns + c0, c1 - c0))
            elif (c0, c1) == (0, columns):
                out.append((r0 * columns, (r1 - r0) * columns))
            else:
                raise Undescribed(
                    f"line {access.line}: a rectangle of several rows, not whole "
                    "ones, is not one range of tiles"
                )
        else:
            out.append((evaluator.value(access.first), evaluator.value(access.count)))
    return out


def unranged_read(described, access, tensor, method, lines, given, program):
    """A read of a rectangle of several rows, not whole ones (a weight's columns,
    `w.tiles[0:K, c:c + n]`): no range of tiles. Kept as an outside read, which no
    call of the chain may write (placed refuses it there); False for other reads."""
    if access.rect is None:
        return False
    group = described.group
    for rank in range(group[1]):
        evaluator = Evaluator(method, lines, given, program, rank, group[1])
        r0, r1, c0, c1 = rectangle_of(evaluator, access)
        columns = tensor.shape[-1] // 32
        if r1 - r0 > 1 and (c0, c1) != (0, columns):
            described.unranged[id(tensor)] = (
                f"line {access.line} reads {access.tensor} as a rectangle of several "
                "rows, not whole ones, which is not one range of tiles"
            )
            described.reads.setdefault(id(tensor), [tensor, None, [], []])
            described.reads[id(tensor)][3].append((access.line, access.prefetch))
            return True
    return False


def roots_of(method, lines, access, given, program, group, size):
    """Whether a line under a leader's if is each group's root's: self.leader after
    the reduce, or `self.rank % G == 0` with G the call's group size."""
    if access.leader == "reduce":
        return True
    if isinstance(access.leader, ast.expr):
        evaluator = Evaluator(method, lines, given, program, 0, group[1])
        if evaluator.value(access.leader) != size:
            raise Undescribed(
                f"line {access.line}: self.rank % G == 0 with G not the call's group "
                f"of {size}"
            )
        return True
    return False


def k_share(found, size):
    """(base, K) when each group's ranks read the same K tiles shared out as
    km::partition shares them, else None."""
    shares = set()
    for at in range(0, len(found), size):
        spans = found[at : at + size]
        base, total = spans[0][0], sum(count for _, count in spans)
        for rank, (page, count) in enumerate(spans):
            start, length = kc.partition(total, rank, size)
            if count != length or (length and page != base + start):
                return None
        shares.add((base, total))
    return shares.pop() if len(shares) == 1 else None


def output_width(method, lines, writes, given, program, group, size):
    """The tiles each group's root writes: block g of the output, g x width on (in
    one write or in several pieces that together make the block)."""
    pieces = {}  # group -> [(page, count, line)] of its root's writes
    for access in writes:
        roots = access.leader and roots_of(
            method, lines, access, given, program, group, size
        )
        if size > 1 and not access.leader:
            raise Undescribed(
                f"line {access.line}: every rank of a group of {size} writes "
                f"{access.tensor}; in a chain only a group's root writes its block "
                "(under if self.rank == self.leader)"
            )
        for rank, (page, count) in enumerate(
            spans(method, lines, access, given, program, group)
        ):
            if rank % size:
                continue
            index = rank // size
            if access.leader and not roots and group[1] // size > 1:
                raise Undescribed(
                    f"line {access.line}: the leader alone writes {access.tensor}, "
                    "and the call has several groups"
                )
            pieces.setdefault(index, []).append((page, count, access))
            if access.leader and not roots:
                break
    width = None
    for index, found in sorted(pieces.items()):
        found.sort(key=lambda piece: piece[0])
        page, access = found[0][0], found[0][2]
        count = sum(piece[1] for piece in found)
        apart = [
            later
            for (at, more, _), (after, _, later) in zip(found, found[1:])
            if after != at + more
        ]
        width = count if width is None else width
        if apart or count != width or page != index * width:
            raise Undescribed(
                f"line {(apart or [access])[0].line}: group {index} writes "
                f"{access.tensor} pages {page} to {found[-1][0] + found[-1][1] - 1}"
                f"{' with a gap or an overlap' if apart else ''}; a chain's call "
                f"writes block g of its output, tiles g x {width} to g x {width} + "
                f"{width - 1}"
            )
    return width or 0


def read_shape(method, lines, access, given, program, group, size, spans_of=None):
    """(K range, None) of a read shared over a group's ranks, or (None, ranges);
    `spans_of`: the spans in another space (column_spans), else pages."""
    found = (spans_of or spans)(method, lines, access, given, program, group)
    roots = access.leader and roots_of(
        method, lines, access, given, program, group, size
    )
    if roots:
        found = [found[g * size] for g in range(group[1] // size) for _ in range(size)]
    elif access.leader:
        if group[1] // size > 1:
            raise Undescribed(
                f"line {access.line}: the leader alone reads {access.tensor}, and the "
                "call has several groups"
            )
        return None, [kc.Range(found[0][0], 0, found[0][1], size > 1)]
    if access.shared:
        if size != group[1]:
            raise Undescribed(
                f"line {access.line}: self.partition shares {access.tensor} over the "
                f"command's {group[1]} cores, and the call sums in groups of {size}"
            )
        if len(set(found)) != 1:
            raise Undescribed(
                f"line {access.line}: the shared range of {access.tensor} changes "
                "with the rank"
            )
        return found[0], None
    firsts = [found[g * size] for g in range(group[1] // size)]
    for rank, span in enumerate(found):
        if span != firsts[rank // size]:
            share = None if roots else k_share(found, size)
            if share is not None:
                return share, None
            raise Undescribed(
                f"line {access.line}: rank {rank % size} of group {rank // size} reads "
                f"{access.tensor} pages other than its root's; a chain's range is a "
                "group's"
            )
    base, width = firsts[0]
    step = firsts[1][0] - base if len(firsts) > 1 else 0
    if step >= 0 and all(
        (page, count) == (base + step * index, width)
        for index, (page, count) in enumerate(firsts)
    ):
        return None, [kc.Range(base, step, width, bool(roots) and size > 1)]
    cover = covering_range(firsts, access)
    out = Ranges([cover._replace(root_only=bool(roots) and size > 1)])
    out.covering = True
    return None, out


class Ranges(list):
    """A read's ranges; `covering`: a range that holds each group's evaluated one."""

    covering = False


def covering_range(firsts, access):
    """The narrowest base + step x index range (step 0 or a step between two groups)
    that holds each group's evaluated range (page, count): its marks wait for every
    block it meets, the device keeping to the blocks written. Refused when none fits
    a range word."""
    steps = {0} | {
        later[0] - first[0]
        for first, later in zip(firsts, firsts[1:])
        if later[0] >= first[0]
    }
    best = None
    for step in sorted(steps):
        base = min(page - step * index for index, (page, _) in enumerate(firsts))
        width = max(
            page + count - base - step * index
            for index, (page, count) in enumerate(firsts)
        )
        try:
            fits = base >= 0 and kc.Range(base, step, width).word() is not None
        except kc.ChainRefused:
            fits = False
        if fits and (best is None or width < best.width):
            best = kc.Range(base, step, width)
    if best is None:
        raise Undescribed(
            f"line {access.line}: {access.tensor}'s groups read "
            f"{', '.join(f'{p} to {p + c - 1}' for p, c in firsts[:4])}...: no range "
            "base + step x index holds them all"
        )
    return best


def tensor_name(tensor, names):
    """A tensor's name in the chain's refusals: its parameter where it was first
    met, with its address."""
    return names.setdefault(id(tensor), f"tensor at {getattr(tensor, 'address', 0):#x}")


def chain_calls(run, calls, group, prefetch=True):
    """The framework's calls of a chain stated with handler calls (front.HandlerCall;
    a kc.Call as it is): each described, its inputs placed (the call before's output,
    km.past's, an outside tensor), refused naming the call, the tensor and the line;
    each record's mask word for prefetch."""
    described = []
    names = {}
    for index, call in enumerate(calls):
        if isinstance(call, kc.Call):
            raise kc.ChainRefused(
                f"call {index} ({call.name or call.handler}): a chain.Call among a "
                "class's handler calls; its tensors are names the class's calls do not "
                "share, so no input or prefetch between them can be checked (state "
                "every call one way)"
            )
        if not isinstance(call, front.HandlerCall):
            raise kc.ChainRefused(
                f"call {index}: {call!r} is neither a call of a class's handler nor a "
                "chain.Call"
            )
        one = describe(run, call, group)
        for place, value in zip(call.method.places(), call.values):
            tensor = value.tensor if isinstance(value, front.Marked) else value
            if front.is_tensor(tensor):
                names.setdefault(
                    id(tensor), f"{place} ({getattr(tensor, 'address', 0):#x})"
                )
        described.append(one)
    written = {id(one.output) for one in described if one.output is not None}
    written |= {id(tensor) for one in described for tensor, _ in one.outputs}
    written |= {key for one in described for key in one.body_writes}
    calls = [
        placed(described, index, one, names, written, prefetch)
        for index, one in enumerate(described)
    ]
    direct_checked(described, calls, names)
    made = Calls(calls)
    made.word_tensors = [one.word_tensors for one in described]
    for index, call in enumerate(calls):
        if call.direct is not None:  # the call before's output: in no DRAM
            label = f"call {index - 1} ({described[index - 1].name}) of a chain"
            made.direct.append((described[index - 1].output, label))
    made.written = [
        one.output
        for index, one in enumerate(described)
        if one.output is not None
        and not (index + 1 < len(calls) and calls[index + 1].direct is not None)
    ]
    return made


class Calls(list):
    """A chain's calls, with the tensors it sends direct ((tensor, the call that sends
    it): no DRAM copy) and those its calls write to DRAM."""

    def __init__(self, calls):
        super().__init__(calls)
        self.direct, self.written = [], []
        self.word_tensors = []  # each call's tensor words: place -> tensor


def direct_checked(described, calls, names):
    """A chain's direct input: one in a chain (one landing area a core), not from an
    overlay handler (its part writes its block to DRAM), and its tensor read by no
    later call (it has no DRAM copy)."""
    sent = [index for index, call in enumerate(calls) if call.direct is not None]
    if len(sent) > 1:
        raise kc.ChainRefused(
            f"calls {sent[0]} and {sent[1]} ({calls[sent[0]].name}, "
            f"{calls[sent[1]].name}) both take a direct input: a core has one landing "
            "area, which a later call's roots could write while an earlier reader "
            "still reads it (send one through DRAM)"
        )
    for index in sent:
        before = described[index - 1]
        stream_refused(before, index, calls[index], names)
        line = described[index].covering.get(id(before.output))
        if line is not None:
            raise kc.ChainRefused(
                f"call {index} ({calls[index].name}): its direct input "
                f"{tensor_name(before.output, names)} is read at line {line} by groups "
                "whose ranges are not base + step x index; the roots send each core "
                "the range it reads, which needs that form (drop km.direct)"
            )
        method = getattr(before.handler, "method", before.handler)
        if method.name in (getattr(before.program, "overlays", None) or {}):
            raise kc.ChainRefused(
                f"call {index} ({calls[index].name}): a direct input from call "
                f"{index - 1} ({before.name}), an overlay handler: its part writes its "
                "block to DRAM (make it resident, or drop km.direct)"
            )
        bytes_ = getattr(before.output, "page_bytes", None)
        if bytes_ is not None and bytes_ != calls[index].direct.tile_bytes:
            raise kc.ChainRefused(
                f"call {index} ({calls[index].name}): its direct input "
                f"{tensor_name(before.output, names)} has tiles of {bytes_} bytes, its "
                f"landing area tiles of {calls[index].direct.tile_bytes}"
            )
        held = getattr(described[index].program, "direct_in", None)
        for earlier in range(index if held else 0):
            handler = described[earlier].handler
            if getattr(handler, "method", handler).name == held[0]:
                raise kc.ChainRefused(
                    f"call {index} ({calls[index].name}): its direct input lands in "
                    f"place, in the halves of {held[0]}'s stream, which call {earlier} "
                    f"({calls[earlier].name}) may still be reading when the roots send "
                    "(drop km.direct, or call it once before in another chain)"
                )
        key = id(before.output)
        for later in range(index + 1, len(described)):
            if key in described[later].reads or key in described[later].body_reads:
                raise kc.ChainRefused(
                    f"call {later} ({calls[later].name}): reads "
                    f"{tensor_name(before.output, names)}, which call {index} "
                    f"({calls[index].name}) takes direct: it has no DRAM copy (drop "
                    "km.direct, or read it in this chain's direct call only)"
                )


def stream_refused(before, index, call, names):
    """A direct input whose roots write it by a write stream: refused (a stream's
    blocks go to DRAM by movement.move, so the roots would send nothing)."""
    if before.streamed is None:
        return
    raise kc.ChainRefused(
        f"call {index} ({call.name}): its direct input "
        f"{tensor_name(before.output, names)} is written by call {index - 1} "
        f"({before.name}) through a write stream (line {before.streamed}), whose "
        "blocks go to DRAM, not to the next cores; write the block with "
        f".{spelling('pages')}(...).write(...) or "
        f".{spelling('tiles')}[...].write(...), which a root sends as written, or drop "
        "km.direct"
    )


def rows_input(described, at, one, key, name, label, mark):
    """(k, ranges) in columns of a read of a tensor call `at` writes in rows: each
    rank's rows inside the rows its roots write; refused when read as pages, or as a
    direct input (not built)."""
    writer = described[at]
    r0, r1 = writer.rows[key]
    read = one.row_reads.get(key)
    if read is None:
        lines = ", ".join(str(line) for line, _ in one.reads[key][3])
        raise kc.ChainRefused(
            f"{label}: line {lines} reads {name} as pages or in two forms, and call "
            f"{at} ({writer.name}) writes it in rows {r0} to {r1 - 1}, its ranges in "
            f"tile columns (written in rows); read it as .{spelling('tiles')}[r0:r1, "
            "c0:c1], one form a call"
        )
    if mark == "direct":
        direct = f"km.{spelling('direct')}({name})"
        if one.in_place:
            why = (
                "an in-place input (km.handler(in_place=True)) over rows is not "
                f"designed: drop in_place= and {direct}"
            )
        elif writer.streamed is not None:
            why = (
                f"its output leaves call {at} by a write stream (line "
                f"{writer.streamed}); a direct input of several rows is not built: "
                f"drop {direct}"
            )
        else:
            why = f"a direct input of several rows is not built: drop {direct}"
        raise kc.ChainRefused(
            f"{label}: {direct}, written in rows {r0} to {r1 - 1} by call {at} "
            f"({writer.name}): {why}; the call then waits for the marks"
        )
    for core, (low, high) in enumerate(read.rows):
        outside = [row for row in range(low, high) if not r0 <= row < r1]
        if outside:
            raise kc.ChainRefused(
                f"{label}: line {read.line}, core {core} reads row {outside[0]} of "
                f"{name}, and call {at} ({writer.name}) writes rows {r0} to {r1 - 1}"
            )
    one.unranged.pop(key, None)  # a range of columns, rows proven
    return read.k, read.ranges


def placed(described, index, one, names, written, prefetch):
    """Call `index` as a kc.Call: its input the call before's output, its other
    reads earlier (km.past, proven by the framework) or outside."""
    label = f"call {index} ({one.name})"
    before = described[index - 1] if index else None
    before_output = before.output if before is not None else None
    before_others = {id(t) for t, _ in before.outputs} if before is not None else set()
    call_input, k, ranges, reads, direct = None, None, (), [], None
    others = []  # reads of the call before's other outputs: (key, k, ranges)
    body_conflicts(described, index, one, names, label)
    for key, (tensor, read_k, read_ranges, read_lines) in one.reads.items():
        name = tensor_name(tensor, names)
        writers = [at for at in range(index) if writes_key(described[at], key)]
        mark = one.marks.get(key)
        if writers and key in described[writers[-1]].rows:
            read_k, read_ranges = rows_input(
                described, writers[-1], one, key, name, label, mark
            )
        for line, stated in read_lines:
            if stated is True and key in written:
                raise kc.ChainRefused(
                    f"{label}: line {line} reads {name} with prefetch=True, and a call "
                    "of the chain writes it: it is read only once marked"
                )
        if before_output is not None and key == id(before_output):
            if key in one.unranged:
                raise kc.ChainRefused(
                    f"{label}: {one.unranged[key]}; it is the call before's output, "
                    "read by ranges of tiles"
                )
            if mark == "past":
                raise kc.ChainRefused(
                    f"{label}: km.{spelling('past')} marks {name}, the call before's "
                    "output, which the chain marks itself"
                )
            call_input, k, ranges = name, read_k, tuple(read_ranges)
            if mark == "direct":
                direct = True
                k, ranges = direct_share(one, k, ranges)
            continue
        if key in one.unranged and writers:
            raise kc.ChainRefused(
                f"{label}: {one.unranged[key]}; a call of the chain writes it, and "
                "only a range's tiles are proven complete"
            )
        if key in before_others and mark is None:  # proven by the input's marks
            others.append((key, read_k, tuple(read_ranges)))
            reads.append(kc.Read(name, read_k, tuple(read_ranges), True))
            continue
        if mark == "direct":
            raise kc.ChainRefused(
                f"{label}: km.{spelling('direct')}({name}): a direct input is the call "
                "before's output"
            )
        if writers and mark != "past":
            writer = described[writers[-1]]
            raise kc.ChainRefused(
                f"{label}: reads {name}, written by call {writers[-1]} "
                f"({writer.name}) further back; mark it km.{spelling('past')}({name}) "
                "at the call, so the build proves it complete or refuses"
            )
        reads.append(kc.Read(name, read_k, tuple(read_ranges), bool(writers)))
    for key, (tensor, _lines) in one.body_reads.items():
        if key not in one.reads:  # an outside input: nothing of the chain writes it
            reads.append(kc.Read(tensor_name(tensor, names), None, (), False))
    if others and not covered(before, one, k, ranges, others):
        # every block of the call before: its marks tell of each output's block
        whole = before.width * (before.group[1] // before.size)
        call_input = tensor_name(before_output, names)
        k, ranges = None, (kc.Range(0, 0, whole),)
    for key, how in one.marks.items():
        if key not in one.reads:
            raise kc.ChainRefused(
                f"{label}: km.{how} marks a tensor its handler does not read"
            )
    words = list(one.words)
    if one.submitted is not None and one.submitted[1] >= len(words):
        raise kc.ChainRefused(f"{label}: the per-submit word's place is past a9")
    if one.mask_word is not None:
        mask = 0xFFFFFFFF if not prefetch else 0
        for key, bit in one.tensor_bits:
            mask |= bit if key in written else 0
        words += [0] * (one.mask_word - len(words)) + [mask]
    output = tensor_name(one.output, names) if one.output is not None else None
    return kc.Call(
        one.handler,
        tuple(words),
        one.group,
        one.size,
        one.width,
        output,
        call_input,
        k,
        ranges,
        one.sends,
        tuple(reads),
        direct_area(one) if direct else None,
        name=one.name,
        row=stamp_row(one, index),
        outputs=tuple((tensor_name(t, names), width) for t, width in one.outputs),
        word=one.submitted,
    )


def writes_key(one, key):
    """Whether described call `one` writes the tensor of id `key` (an output)."""
    outputs = [one.output, *(tensor for tensor, _ in one.outputs)]
    return any(tensor is not None and id(tensor) == key for tensor in outputs)


def body_conflicts(described, index, one, names, label):
    """A tensor a body writes (writes=) is read by no other call of the chain, and a
    body's outside input (reads=) is written by no call: neither can be proven."""
    for key in set(one.reads) | set(one.body_reads):
        for at, other in enumerate(described):
            if at != index and key in other.body_writes:
                tensor, lines = other.body_writes[key]
                raise kc.ChainRefused(
                    f"{label}: reads {tensor_name(tensor, names)}, which a body of "
                    f"call {at} ({other.name}) writes ({spelling('body_writes')}=, "
                    f"line {lines[0]}): what a body writes is not described, so no "
                    "read of it in the chain is proven; read it in a later chain or "
                    "command"
                )
    for key, (tensor, lines) in one.body_reads.items():
        for at, other in enumerate(described):
            if at != index and writes_key(other, key):
                raise kc.ChainRefused(
                    f"{label}: line {lines[0]}'s body reads "
                    f"{tensor_name(tensor, names)} ({spelling('body_reads')}=), an "
                    f"outside input of the chain, and call {at} ({other.name}) writes "
                    "it: what a body reads is not described, so it cannot wait for it"
                )


def covered(before, one, k, ranges, others):
    """Whether each core's reads of the call before's other outputs are blocks its
    input's marks already tell it of (`k` or `ranges` of the call before's output)."""
    groups = before.group[1] // before.size
    widths = {id(tensor): width for tensor, width in before.outputs}
    for local in range(one.group[1]):
        rank, group = local % one.size, local // one.size
        known = {
            block
            for first, count in kc.tiles_of(k, ranges, rank, group, one.size)
            for block in kc.blocks_met(first, count, before.width, groups)
        }
        for key, read_k, read_ranges in others:
            for first, count in kc.tiles_of(read_k, read_ranges, rank, group, one.size):
                if set(kc.blocks_met(first, count, widths[key], groups)) - known:
                    return False
    return True


def stamp_row(one, index):
    """A record's stamp row: its index in the chain (0 to 7, the rows km_chain.h
    keeps; a later record's stamps go to no row) when its program has the stamps."""
    if not getattr(one.program, "chain_stamps", False):
        return 0
    words = kc.words()
    return index if index < words["kChainStampRecords"] else words["kChainRowMask"]


def direct_share(one, k, ranges):
    """(k, ranges) of a direct input: a group of one core reading the whole range
    base .. base + width of the call before's output is the K share (base, width),
    the same marks; the send writes a K range (km_chain.hpp chain_send)."""
    if k is None and one.size == 1 and len(ranges) == 1:
        (only,) = ranges
        if only.step == 0 and not only.root_only:
            return (only.base, only.width), ()
    return k, ranges


def direct_area(one):
    """A direct input's landing area: the program's (its tiles and their bytes),
    refused when the class has none."""
    area = getattr(one.program, "direct_area", None)
    if area is None:
        raise kc.ChainRefused(
            f"{one.name}: a direct input, and its program has no landing area (a "
            "class with the chain's walk has one when a handler declares "
            f"{spelling('direct')}=(...) and the make gives that tensor); send it "
            "through DRAM (drop km.direct)"
        )
    return kc.Direct(area[0], area[1])


# The in-class form: a km.chain method ------------------------------------------------


def checked(cls, method):
    """A km.chain method's lines, read when the class is made: calls of the
    class's handlers, loops of a constant count; anything else refused at its line."""
    definition, _, _ = compiler.source_of(method.function)
    ChainLines(cls, method, definition).check(definition.body)


class ChainLines:
    """A km.chain method's lines: checked at the make, expanded at a call."""

    def __init__(self, cls, method, definition, program=None):
        self.cls, self.method, self.definition = cls, method, definition
        self.program = program
        self.handlers = {m.name for m in compiler.handler_methods(cls)}
        self.parameters = list(inspect.signature(method.function).parameters)[1:]

    def refuse(self, node, why, fix):
        raise ProgramError(
            f"{self.cls.__name__}.{self.method.name}, line {node.lineno}: "
            f"`{ast.unparse(node).splitlines()[0]}`: {why}; {fix}"
        )

    def check(self, statements):
        for node in statements:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self.handler_call(node, node.value)
            elif isinstance(node, ast.For):
                self.count(node)
                self.check(node.body)
            elif isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) for t in node.targets
            ):
                continue
            elif isinstance(node, ast.If | ast.While):
                self.refuse(
                    node,
                    "a branch or a loop on data: a chain's calls are fixed when it is "
                    "written",
                    "write the calls in order; a loop of a constant count is unrolled",
                )
            elif isinstance(node, ast.Pass) or (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            ):
                continue
            else:
                self.refuse(
                    node,
                    "a chain's lines are calls of the class's handlers",
                    "write self.handler(args...)",
                )

    def handler_call(self, node, call):
        func = call.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
            and func.attr in self.handlers
        ):
            self.refuse(
                node,
                "a call of something other than a handler of the class",
                f"call one of {', '.join(sorted(self.handlers))} as self.name(...)",
            )

    def count(self, node):
        iterator = node.iter
        if not (
            isinstance(iterator, ast.Call)
            and getattr(iterator.func, "id", None) == "range"
            and isinstance(node.target, ast.Name)
        ):
            self.refuse(
                node,
                "a loop the build cannot count",
                "loop over range(n), n an int or a constant of the class",
            )
        for bound in iterator.args:
            try:
                self.constant(bound, {})
            except Undescribed:
                self.refuse(
                    node,
                    "a loop the build cannot count: its count is not an int or a "
                    "constant of the class",
                    "loop over range(n), n an int or a constant of the class",
                )

    def constant(self, node, env):
        """An int of the chain's lines: a literal, a loop's variable, a constant of
        the class (self.X), a module's int; arithmetic of them."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env and isinstance(env[node.id], int):
                return env[node.id]
            value = self.method.function.__globals__.get(node.id)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == (
            "self"
        ):
            if self.program is not None:
                values = getattr(self.program, "constant_values", None) or {}
                if node.attr in values:
                    return values[node.attr]
            else:
                declared = front.constants_of(self.cls).get(node.attr)
                if declared is not None and not isinstance(declared, front.Symbol):
                    return 1  # its value is the program's: counted at the call
        if isinstance(node, ast.BinOp):
            left, right = self.constant(node.left, env), self.constant(node.right, env)
            operations = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.FloorDiv: lambda a, b: a // b,
            }
            for kind, operation in operations.items():
                if isinstance(node.op, kind):
                    return operation(left, right)
        raise Undescribed(ast.unparse(node))

    def expand(self, values, named):
        bound = inspect.signature(self.method.function).bind(None, *values, **named)
        bound.apply_defaults()
        env = dict(list(bound.arguments.items())[1:])
        calls = []
        self.unrolled(self.definition.body, env, calls)
        return calls

    def unrolled(self, statements, env, calls):
        for node in statements:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                calls.append(self.made_call(node.value, env))
            elif isinstance(node, ast.For):
                bounds = [self.constant(b, env) for b in node.iter.args]
                for value in range(*bounds):
                    self.unrolled(node.body, {**env, node.target.id: value}, calls)
            elif isinstance(node, ast.Assign):
                value = self.argument(node.value, env)
                for target in node.targets:
                    env[target.id] = value

    def made_call(self, call, env):
        handler = getattr(self.program, call.func.attr)
        values = [self.argument(arg, env) for arg in call.args]
        named = {k.arg: self.argument(k.value, env) for k in call.keywords}
        return handler(*values, **named)

    def argument(self, node, env):
        """A call's argument: a name of the method, a constant's int, a tuple, an
        element of a list argument, km.past(t) or km.direct(t)."""
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.Tuple):
            return tuple(self.argument(e, env) for e in node.elts)
        if isinstance(node, ast.Subscript):
            return self.argument(node.value, env)[self.constant(node.slice, env)]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in (spelling("past"), spelling("direct"))
        ):
            marked = front.past if node.func.attr == spelling("past") else front.direct
            return marked(self.argument(node.args[0], env))
        if isinstance(node, ast.Constant):
            return node.value
        return self.constant(node, env)


def expand(program, method, values, named):
    """A km.chain method called on `program`: its calls, as the host list's."""
    definition, _, _ = compiler.source_of(method.function)
    return ChainLines(type(program), method, definition, program).expand(values, named)
