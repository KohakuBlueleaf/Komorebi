"""The walk: a class's handlers, read with `ast` as the class compiler does, as one IR
module's text with each line's origin. It knows the corpus's lines (a word, a partition,
streams, a loop pushing a tile function's result or expression, a sum reduced and
written by the leader); any other is refused by name, with its class's line."""

import ast
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path

from komorebi.chain import calls as kc
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compile.source import MAKE_OPTIONS, Where, file_of, refuse, source_of
from komorebi.dsl.compiler import class_annotations, handler_methods
from komorebi.dsl.front import front
from komorebi.dsl.front.names import kind_of, spelling

# The tensor formats the walk writes, by the DSL's names.
FORMATS = {"bfloat16": "bf16", "float32": "f32"}
# A tile's formats, by the DSL's constructs (km.fp32).
TILE_FORMATS = {"fp32": "f32", "bf16": "bf16"}
# The make's keywords that are no part of the module: load facts and the report's log.
LOAD_OPTIONS = ("log", "l1", "depth")
# Names a value or a parameter cannot have in the text: the IR's keywords, and its
# built-in names a module's text reads.
RESERVED = frozenset(
    {
        # keywords
        *("let", "var", "if", "else", "for", "in", "while", "return", "store"),
        *("consume", "await", "true", "false"),
        # names the text reads
        *("worker", "group", "island", "stream", "sink", "partition", "zip", "min"),
        *("max", "range", "zeros", "gather", "reduce", "all_reduce", "broadcast"),
        *("sync", "past", "send", "forward", "offer", "invoke", "outcome", "where"),
        *("matmul", "tile", "format"),
        # types and formats
        *("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f16", "f32", "f64"),
        "bf16",
    }
)
INDENT = "    "


@dataclass
class Made:
    """A class as an IR module: its text, and its origin as `komorebi_ir.Origin` takes
    it: the source files, their lines the map covers, the map from ranges of the text's
    lines to places of the source, and the renames of values and parameters."""

    text: str
    files: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    map: list = field(default_factory=list)
    renames: list = field(default_factory=list)

    @property
    def origin(self):
        """(files, lines, map, renames): `komorebi_ir.Origin(*made.origin)`."""
        return (self.files, self.lines, self.map, self.renames)


@dataclass
class Value:
    """What a name holds in a handler: its kind ("tensor", "u32", "f32", "partition",
    "sink", "block", "tile", "reduced"), a partition's tensor and range's words, a
    block's tiles."""

    kind: str
    format: str = ""
    tensor: str = ""
    first: str = ""
    count: str = ""
    tiles: int = 1


@dataclass
class Inline:
    """A tile expression pushed as it is (`out.push(x * scale)`), which today's class
    compiles as the function `tile_line_<line>` of the names it reads, in order."""

    name: str
    node: ast.AST
    value: ast.AST
    names: list
    where: Where


class Text:
    """The module's lines as they are written, each range of them with its place."""

    def __init__(self):
        self.lines, self.files, self.map = [], [], []

    def line(self, depth, text):
        """Writes one line at `depth`; its number."""
        self.lines.append(INDENT * depth + text if text else "")
        return len(self.lines)

    def place(self, first, file, start, end):
        """The lines from `first` to the last written: the source's `start` to `end`."""
        if file not in self.files:
            self.files.append(file)
        index = self.files.index(file)
        self.map.append((first, len(self.lines), index, start, end))


def module(cls, **make):
    """The IR module of class `cls` made with `make` (a `km.Program`'s keywords): its
    text and origin. Refuses, naming the line, what the IR cannot say yet."""
    tensors = given_tensors(cls, make)
    methods = handler_methods(cls)
    class_refusals(cls, methods)
    text = Text()
    walks = [Handler(cls, method, tensors) for method in methods]
    functions = {}
    for walk in walks:
        walk.read()
        for name, (function, kinds) in walk.functions.items():
            if functions.setdefault(name, (function, kinds))[1] != kinds:
                raise Refused(
                    f"{cls.__name__}: {name} is called on {functions[name][1]} and on "
                    f"{kinds}: a tile function on a second format is not written yet"
                )
    formats = []
    for walk in walks:
        for value in [*walk.params.values(), *walk.scope.values()]:
            if value.kind in ("tensor", "tile") and value.format not in formats:
                formats.append(value.format)
    text.line(0, f"module {cls.__name__.lower()}")
    text.line(0, "")
    text.line(0, "tile [32, 32]")
    text.line(0, f"format {', '.join(formats)}")
    helpers = {}
    for function, kinds in functions.values():
        text.line(0, "")
        tile_function(text, function, kinds, helpers)
    written = set()
    while len(written) < len(helpers):  # a helper's own helpers join as it is written
        for name, (helper, kinds) in list(helpers.items()):
            if name not in written:
                written.add(name)
                text.line(0, "")
                tile_function(text, helper, kinds, helpers)
    text.line(0, "")
    text.line(0, f"program {cls.__name__} {{")
    for walk in walks:
        walk.write(text)
    text.line(0, "}")
    renames = [pair for walk in walks for pair in walk.renames]
    return Made(
        "\n".join(text.lines) + "\n",
        text.files,
        held_lines(text),
        text.map,
        renames,
    )


def given_tensors(cls, make):
    """The make's tensors by parameter name; a keyword that would change the module and
    that the walk does not take is refused by name."""
    for key in make:
        if key in MAKE_OPTIONS and key not in LOAD_OPTIONS:
            raise Refused(
                f"{cls.__name__}: the make's {key}= is not written in an IR module yet"
            )
    return {k: v for k, v in make.items() if k not in MAKE_OPTIONS}


def class_refusals(cls, methods):
    """What the class states beyond its handlers' lines, refused by name."""
    name = cls.__name__
    annotations = class_annotations(cls).values()
    stated = [
        ("fields", any(isinstance(v, front.Field) for v in annotations)),
        ("constants", bool(front.constants_of(cls))),
        (spelling("compute_options"), bool(getattr(cls, "compute_given", None))),
        (spelling("defines"), bool(getattr(cls, "defines_given", None))),
        (spelling("startup"), getattr(cls, "startup_given", None) is not None),
        (
            "chains",
            any(
                isinstance(v, kc.ChainMethod)
                for k in cls.__mro__
                for v in vars(k).values()
            ),
        ),
    ]
    for what, given in stated:
        if given:
            raise Refused(f"{name}: its {what} are not written in an IR module yet")
    for method in methods:
        options = {
            spelling("overlay"): method.overlay,
            spelling("stamps"): method.stamps,
            spelling("direct"): method.direct,
            "in_place": method.in_place,
        }
        for what, value in options.items():
            if value:
                raise Refused(
                    f"{name}.{method.name}: {what}= is not written in an IR module yet"
                )


class Handler:
    """One handler's walk: its parameters and lines read, then written."""

    def __init__(self, cls, method, tensors):
        self.cls, self.method, self.tensors = cls, method, tensors
        definition, lines, first = source_of(method.function)
        self.definition = definition
        self.file = file_of(method.function)
        self.where = Where(cls.__name__, method.name, lines, first, self.file)
        self.globals = method.function.__globals__
        self.params, self.scope = {}, {}
        self.functions = (
            {}
        )  # tile function name -> (TileFunction, its parameters' kinds)
        self.restated = {}  # (partition, tensor) -> the name of its restatement
        self.statements = []  # (node, [(depth offset, text)], children)
        self.renames, self.names = [], {}
        used = {n.id for n in ast.walk(definition) if isinstance(n, ast.Name)}
        self.used = used | {a.arg for a in definition.args.args}

    def refuse(self, node, why, instead):
        refuse(self.where, node, why, instead)

    def name(self, python):
        """The text's name of a Python name: itself, or a safe name with its rename."""
        if python not in self.names:
            safe = python
            while safe in RESERVED or (safe != python and safe in self.used):
                safe += "_"
            if safe != python:
                self.renames.append((safe, python))
            self.names[python] = safe
            self.used.add(safe)
        return self.names[python]

    def read(self):
        """The parameters, then each line, into `statements`."""
        arguments = self.definition.args
        if arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
            self.refuse(self.definition, "*args and keywords", "name each parameter")
        for parameter in arguments.args[1:]:
            self.params[parameter.arg] = self.parameter(parameter)
        self.scope = dict(self.params)
        self.statements = self.body(self.definition.body)

    def parameter(self, parameter):
        annotation = parameter.annotation
        if isinstance(annotation, ast.Name) and annotation.id == "int":
            return Value("u32")
        if isinstance(annotation, ast.Name) and annotation.id == "float":
            return Value("f32")
        if isinstance(annotation, ast.Attribute) and kind_of(
            annotation.attr, ("tensor",)
        ):
            return Value("tensor", format=self.tensor_format(parameter))
        self.refuse(
            self.definition,
            f"parameter {parameter.arg}: a kind the IR module does not take yet",
            f"a parameter is km.{spelling('tensor')}, int or float",
        )

    def tensor_format(self, parameter):
        """The IR format of a tensor parameter, from the make's tensor; its pages one
        tile each."""
        tensor = self.tensors.get(parameter.arg)
        if tensor is None:
            self.refuse(
                self.definition,
                f"the make gives no tensor {parameter.arg}",
                f"make the class with {parameter.arg}=<a tensor>",
            )
        dtype = getattr(tensor, "dtype", None)
        if dtype not in FORMATS:
            self.refuse(
                self.definition,
                f"tensor {parameter.arg} of format {dtype}: not written in an IR "
                "module yet",
                f"a tensor of {', '.join(FORMATS)}",
            )
        tiles = getattr(tensor, "page_tiles", 1)
        if getattr(tensor, "layout", None) != "tile" or tiles != 1:
            self.refuse(
                self.definition,
                f"tensor {parameter.arg}: pages of {tiles} tiles of the tile layout "
                "only, one tile a page (an IR view is in tiles)",
                "make it of one tile a page",
            )
        return FORMATS[dtype]

    def body(self, statements):
        """[(node, lines, children)] of a list of statements."""
        return [self.statement(node) for node in statements]

    def statement(self, node):
        if isinstance(node, ast.Assign):
            return self.assign(node)
        if isinstance(node, ast.AugAssign):
            return (node, [self.augment(node)], [])
        if isinstance(node, ast.For):
            return self.loop(node)
        if isinstance(node, ast.If):
            return self.leader(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == spelling("write"):
                return (node, [self.store(node, node.value)], [])
            return (node, [self.push(node, node.value)], [])
        self.refuse(
            node, "a line the IR module does not write yet", "see the IR's lines"
        )

    def assign(self, node):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self.refuse(node, "an assignment to more than one name", "assign one name")
        target, value = node.targets[0].id, node.value
        if target in self.scope:
            self.refuse(
                node, f"{target} assigned twice", "give each value its own name"
            )
        construct = construct_of(value)
        word = self.int_text(value) if construct is None else None
        if construct == "partition":
            lines = [self.partition(node, target, value)]
        elif construct == "stream" and write_stream(value):
            lines = self.stream(node, value, write=True)
            lines[-1] = f"let {self.name(target)} = {lines[-1]}"
            self.scope[target] = Value("sink")
        elif construct == "zeros":
            lines = [self.zeros(node, target, value)]
        elif construct == "reduce":
            lines = [self.reduce(node, target, value)]
        elif word is not None:
            lines = [f"let {self.name(target)} = {bare(word)}"]
            self.scope[target] = Value("u32")
        else:
            self.refuse(
                node,
                "an assignment the IR module does not write yet",
                "see the IR's lines",
            )
        return (node, lines, [])

    def partition(self, node, target, value):
        """`p = self.partition(t.pages(a, b))`: this rank's part of the pages."""
        if len(value.args) != 1 or value.keywords:
            self.refuse(
                node, "a partition of more than its pages", "partition one range"
            )
        pages = value.args[0]
        if not (
            isinstance(pages, ast.Call)
            and isinstance(pages.func, ast.Attribute)
            and pages.func.attr == spelling("pages")
            and isinstance(pages.func.value, ast.Name)
            and self.kind(pages.func.value.id) == "tensor"
            and len(pages.args) == 2
            and not pages.keywords
        ):
            self.refuse(
                node,
                "a partition of what is not t.pages(first, count)",
                "partition a tensor's pages",
            )
        tensor = pages.func.value.id
        first, count = (bare(self.word(node, arg)) for arg in pages.args)
        self.scope[target] = Value("partition", tensor=tensor, first=first, count=count)
        view = f"{self.name(tensor)}.tiles({first}, {count})"
        return f"let {self.name(target)} = {partition_text(view)}"

    def zeros(self, node, target, value):
        """`total = km.tile.zeros(km.fp32)`: a tile of zeros the handler adds to."""
        (arg,) = value.args if len(value.args) == 1 and not value.keywords else (None,)
        format_ = None
        if (
            isinstance(arg, ast.Attribute)
            and isinstance(arg.value, ast.Name)
            and arg.value.id == "km"
        ):
            format_ = TILE_FORMATS.get(kind_of(arg.attr, tuple(TILE_FORMATS)))
        if format_ is None:
            self.refuse(
                node,
                "a tile of zeros of what is not a format written",
                f"write km.{spelling('tile')}.{spelling('zeros')}(km.fp32)",
            )
        self.scope[target] = Value("tile", format=format_)
        return f"var {self.name(target)}: tile<{format_}> = zeros"

    def augment(self, node):
        """`total += tile`: the loop's tile added to a tile of the handler."""
        target, value = node.target, node.value
        if not (
            isinstance(node.op, ast.Add)
            and isinstance(target, ast.Name)
            and self.kind(target.id) == "tile"
            and isinstance(value, ast.Name)
            and self.kind(value.id) == "block"
            and self.scope[value.id].tiles == 1
        ):
            self.refuse(
                node,
                "an augmented assignment other than tile += the loop's tile",
                "add the loop's tile (block=1) to a tile of zeros",
            )
        return f"{self.name(target.id)} += {self.name(value.id)}"

    def reduce(self, node, target, value):
        """`whole = km.reduce(total, form="tree")`: the group's sum, with `sum`; how the
        backend groups it is its own (on Tensix a tree), so no form is written."""
        tile = value.args[0] if len(value.args) == 1 else None
        if not (isinstance(tile, ast.Name) and self.kind(tile.id) == "tile"):
            self.refuse(node, "a reduce of what is not a tile", "reduce one tile")
        for keyword in value.keywords:
            given = keyword.value
            if keyword.arg != spelling("form"):
                self.refuse(
                    node,
                    f"a reduce's {keyword.arg}= is not written in an IR module yet",
                    "leave it out",
                )
            if isinstance(given, ast.Constant) and given.value == "owner":
                self.refuse(
                    node,
                    'a reduce by owner (form="owner") is not written in an IR module '
                    "yet: no tensix.* hint names it",
                    'leave form= out or write form="tree"',
                )
            if not (isinstance(given, ast.Constant) and given.value == "tree"):
                self.refuse(
                    node, "a reduce's form that is not written", 'write form="tree"'
                )
        self.scope[target] = Value("reduced", format=self.scope[tile.id].format)
        return f"let {self.name(target)} = reduce({self.name(tile.id)}, with = sum)"

    def leader(self, node):
        """`if self.rank == self.leader:` over its lines: `if worker.is_leader`."""
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and self_attribute(test.left) == spelling("rank")
            and self_attribute(test.comparators[0]) == spelling("leader")
            and not node.orelse
        ):
            self.refuse(
                node,
                "an if other than `if self.rank == self.leader:` with no else",
                "test the leader so",
            )
        return (node, ["if worker.is_leader {"], self.body(node.body))

    def store(self, node, call):
        """`t.pages(a, b).write(whole)`: a reduce's result written to t's tiles."""
        pages = call.func.value
        whole = call.args[0] if len(call.args) == 1 and not call.keywords else None
        if not (isinstance(whole, ast.Name) and self.kind(whole.id) == "reduced"):
            self.refuse(
                node,
                "a page write of what is not a reduce's result",
                "write the tile km.reduce(...) gives",
            )
        if not (
            isinstance(pages, ast.Call)
            and isinstance(pages.func, ast.Attribute)
            and pages.func.attr == spelling("pages")
            and isinstance(pages.func.value, ast.Name)
            and self.kind(pages.func.value.id) == "tensor"
        ):
            self.refuse(
                node, "a write to what is not t.pages(first, count)", "write t's pages"
            )
        return f"store {self.view(node, pages, [])} <- {self.name(whole.id)}"

    def view(self, node, view, lines):
        """The text of a stream's view: `t.pages(a, b)` as `t.tiles(a, b)`; of `t[p]`,
        p's name or p's range of tensor t restated (`lines` gains the restatement)."""
        if (
            isinstance(view, ast.Call)
            and isinstance(view.func, ast.Attribute)
            and view.func.attr == spelling("pages")
            and isinstance(view.func.value, ast.Name)
            and self.kind(view.func.value.id) == "tensor"
            and len(view.args) == 2
            and not view.keywords
        ):
            first, count = (bare(self.word(node, arg)) for arg in view.args)
            return f"{self.name(view.func.value.id)}.tiles({first}, {count})"
        if not (
            isinstance(view, ast.Subscript)
            and isinstance(view.value, ast.Name)
            and isinstance(view.slice, ast.Name)
            and self.kind(view.value.id) == "tensor"
            and self.kind(view.slice.id) == "partition"
        ):
            self.refuse(
                node,
                "a stream of what is not t[part] or t.pages(first, count)",
                "stream a tensor's partition or pages",
            )
        tensor, part = view.value.id, view.slice.id
        held = self.scope[part]
        if held.tensor == tensor:
            return self.name(part)
        key = (part, tensor)
        if key not in self.restated:
            # One partition applied to a second tensor: the IR states the same range of
            # the other tensor, partitioned alike.
            name = self.name(f"{part}_{tensor}")
            self.restated[key] = name
            text = f"{self.name(tensor)}.tiles({held.first}, {held.count})"
            lines.append(f"let {name} = {partition_text(text)}")
        return self.restated[key]

    def stream(self, node, call, write):
        """The lines of `km.Stream(t[p], block=n)` (n 1 when not given), the last one
        the stream itself."""
        keywords = {k.arg: k.value for k in call.keywords}
        allowed = {spelling("block")} | ({spelling("write_stream")} if write else set())
        for key in keywords:
            if key not in allowed:
                self.refuse(
                    node,
                    f"a stream's {key}= is not written in an IR module yet",
                    "leave it out",
                )
        # no block= is a block of one tile, as today's (`stream_options`, row=1)
        block = keywords.get(spelling("block"), ast.Constant(1))
        if not (isinstance(block, ast.Constant) and type(block.value) is int):
            self.refuse(
                node, "a stream's block that is not a number written", "write block=<n>"
            )
        if len(call.args) != 1:
            self.refuse(node, "a stream of more than one view", "stream one view")
        lines = []
        view = self.view(node, call.args[0], lines)
        word = "sink" if write else "stream"
        lines.append(f"{word}({view}, block = {block.value})")
        return lines

    def loop(self, node):
        """`for b in km.Stream(t[p], block=n):`, or `for x, y in zip(km.Stream(...),
        km.Stream(...)):`, over its lines."""
        zipped = (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "zip"
            and not node.iter.keywords
        )
        calls = node.iter.args if zipped else [node.iter]
        targets = node.target.elts if isinstance(node.target, ast.Tuple) else []
        names = targets if zipped else [node.target]
        if node.orelse or len(names) != len(calls) or zipped and len(calls) != 2:
            self.refuse(
                node,
                "a loop with an else, or names other than one a stream",
                "loop one name over a stream, or two over zip(a, b)",
            )
        if not all(isinstance(n, ast.Name) for n in names):
            self.refuse(node, "a loop's name that is not a name", "loop over names")
        for call in calls:
            if construct_of(call) != "stream" or write_stream(call):
                self.refuse(
                    node,
                    "a loop over what is not a read stream",
                    "loop over km.Stream(t[p], block=n)",
                )
        lines, streams = [], []
        for call in calls:
            made = self.stream(node, call, write=False)
            lines += made[:-1]
            streams.append(made[-1])
        held = ", ".join(self.name(n.id) for n in names)
        walked = f"zip({', '.join(streams)})" if zipped else streams[0]
        lines.append(f"for {held} in {walked} {{")
        for name, call in zip(names, calls):
            keywords = {k.arg: k.value for k in call.keywords}
            block = keywords.get(spelling("block"), ast.Constant(1)).value
            self.scope[name.id] = Value(
                "block", format=self.scope_format(call), tiles=block
            )
        children = self.body(node.body)
        return (node, lines, children)

    def scope_format(self, call):
        """The format of the tensor a stream's view reads."""
        view = call.args[0]
        tensor = view.func.value if isinstance(view, ast.Call) else view.value
        return self.scope[tensor.id].format

    def push(self, node, call):
        """`out.push(f(b, k))` or `out.push(<expression>)`: a tile function's result,
        or a tile expression's as the function `tile_line_<line>`, into a sink."""
        func = call.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("push")
            and isinstance(func.value, ast.Name)
            and self.kind(func.value.id) == "sink"
            and len(call.args) == 1
            and not call.keywords
        ):
            self.refuse(
                node, "a call the IR module does not write yet", "see the IR's lines"
            )
        pushed = call.args[0]
        if isinstance(pushed, ast.BinOp):
            called = self.tile_expression(node, pushed)
        else:
            called = self.tile_call(node, pushed)
        return f"{self.name(func.value.id)}.push({called})"

    def tile_expression(self, node, value):
        """A tile expression of the loop's block and floats as today's class compiles
        it, the function `tile_line_<line>` of the names it reads, in order."""
        names = []
        for leaf in ast.walk(value):
            if isinstance(leaf, ast.Name) and leaf.id not in names:
                names.append(leaf.id)
        name = f"tile_line_{node.lineno}"
        function = Inline(name, node, value, names, self.where)
        return self.tile_operands(node, function, names, "a tile expression")

    def tile_call(self, node, call):
        """A tile function's call on a block and scalar parameters."""
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and not call.keywords
        ):
            self.refuse(
                node,
                "a push of what is not a tile function's call",
                "push f(block, ...)",
            )
        function = self.globals.get(call.func.id)
        if not isinstance(function, front.TileFunction):
            self.refuse(
                node,
                f"{call.func.id} is not a tile function",
                "decorate it with km.tile_function",
            )
        if function.tolerance is not None or function.fidelity is not None:
            self.refuse(
                node,
                f"{function.name}'s options are not written in an IR module yet",
                "leave them out",
            )
        if not all(isinstance(arg, ast.Name) for arg in call.args):
            self.refuse(
                node,
                "a tile function's operand that is not a block or a float",
                "pass the loop's block or a float",
            )
        names = [arg.id for arg in call.args]
        return self.tile_operands(node, function, names, "a tile function")

    def tile_operands(self, node, function, names, what):
        """The call of `function` on the handler's `names`: a block first, then floats;
        a block of n tiles typed `tile<F>[n]` (the function is declared on blocks)."""
        kinds = []
        for name in names:
            if self.kind(name) not in ("block", "f32"):
                self.refuse(
                    node,
                    f"{what}'s operand that is not a block or a float",
                    "pass the loop's block or a float",
                )
            value = self.scope[name]
            tiles = f"[{value.tiles}]" if value.tiles > 1 else ""
            kinds.append(
                f"tile<{value.format}>{tiles}" if value.kind == "block" else "f32"
            )
        if "tile<" not in "".join(kinds[:1]):
            self.refuse(
                node,
                f"{what} whose first operand is not a block",
                "pass the block first",
            )
        self.functions[function.name] = (function, tuple(kinds))
        return f"{function.name}({', '.join(self.name(n) for n in names)})"

    def kind(self, name):
        value = self.scope.get(name)
        return value.kind if value else None

    def word(self, node, expression):
        """An int expression of the handler's words (`int_text`), else refused."""
        text = self.int_text(expression)
        nodes = ast.walk(expression)
        leader = any(self_attribute(n) == spelling("leader") for n in nodes)
        if text is None and leader:
            self.refuse(
                node,
                "self.leader as a number is not written in an IR module yet",
                "test it: `if self.rank == self.leader:`",
            )
        if text is None:
            self.refuse(
                node,
                "a word that is not an int parameter or value, a number, the rank, "
                "+ - or *",
                "write it of those",
            )
        return text

    def int_text(self, expression):
        """The text of an int expression of int parameters and values, numbers (a
        module's int constant as its number), `self.rank`, + - *; else None."""
        if isinstance(expression, ast.Name) and self.kind(expression.id) == "u32":
            return self.name(expression.id)
        if isinstance(expression, ast.Name) and expression.id not in self.scope:
            known = self.globals.get(expression.id)
            if type(known) is int and known >= 0:
                return str(known)
        if isinstance(expression, ast.Constant) and type(expression.value) is int:
            return str(expression.value)
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
            and expression.attr == spelling("rank")
        ):
            return "worker.rank"
        operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
        if isinstance(expression, ast.BinOp) and type(expression.op) in operators:
            left = self.int_text(expression.left)
            right = self.int_text(expression.right)
            if left is None or right is None:
                return None
            return f"({left} {operators[type(expression.op)]} {right})"
        return None

    def write(self, text):
        """The handler's lines into `text`, each range of them with its place."""
        start = text.line(1, f"handler {self.method.name}(")
        for name, value in self.params.items():
            kind = f"tensor<{value.format}>" if value.kind == "tensor" else value.kind
            text.line(2, f"{self.name(name)}: {kind},")
        text.line(1, ") @on(group) {")
        self.write_body(text, self.statements, 2)
        text.line(1, "}")
        line = self.definition.lineno
        text.place(start, self.file, line, line)

    def write_body(self, text, statements, depth):
        for node, lines, children in statements:
            first = len(text.lines) + 1
            for line in lines:
                text.line(depth, line)
            if lines[-1].endswith("{"):
                self.write_body(text, children, depth + 1)
                text.line(depth, "}")
            text.place(first, self.file, node.lineno, node.end_lineno)


def construct_of(node):
    """The construct a call is: "partition" (self.partition), "stream" (km.Stream),
    "reduce" (km.reduce), "zeros" (km.tile.zeros)."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    func = node.func
    if isinstance(func.value, ast.Name) and func.value.id == "self":
        return kind_of(func.attr, ("partition",))
    if isinstance(func.value, ast.Name) and func.value.id == "km":
        return kind_of(func.attr, ("stream", "reduce"))
    tile = func.value
    if (
        isinstance(tile, ast.Attribute)
        and tile.attr == spelling("tile")
        and isinstance(tile.value, ast.Name)
        and tile.value.id == "km"
    ):
        return kind_of(func.attr, ("zeros",))
    return None


def self_attribute(node):
    """The attribute's name of `self.<name>`, else None."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def write_stream(call):
    """True for km.Stream(..., write=True)."""
    for keyword in call.keywords:
        if keyword.arg == spelling("write_stream"):
            return (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            )
    return False


def partition_text(view):
    return f"partition({view}, worker.rank, group.size)"


@dataclass
class Helper:
    """A plain function a tile function calls (`def grown(x): return km.exp(x) * 2`),
    written as a `func` of its own, its lines its own."""

    name: str
    function: object


# the tile compiler's functions of one tile the IR writes by the same name (`exp(x)`)
MATH = tuple(
    f"math_{name}"
    for name in ("exp", "exp2", "sigmoid", "silu", "tanh", "erf", "log", "sqrt")
    + ("rsqrt", "reciprocal", "relu", "square", "floor")
)
# a call's own lines in the text: `(`, the call on lines of its own placed at the
# call's source lines (a node's line today is its call's, `caller_line`), then `)`
CALL_OPEN, CALL_CLOSE = "\x00", "\x01"


@dataclass
class Body:
    """What a tile function's expression is read with: its place, its parameters'
    kinds, its statement's first line, its module's names, the helpers it calls."""

    where: Where
    node: ast.AST
    kinds: dict
    first: int
    names: dict
    helpers: dict


def tile_function(text, function, kinds, helpers):
    """A tile function as a `func`: its one `return` of + - * / over its parameters and
    numbers, the tile compiler's functions of one tile and calls of helpers, its
    parameters typed by the call; a tile expression (`Inline`) alike. A call on a line
    after the statement's first is written on a line of its own, placed at the call."""
    if isinstance(function, Inline):
        where, parameters = function.where, function.names
        definition = returned = function.node
        value = function.value
        names = {}
    else:
        definition, lines, first = source_of(function.function)
        what = "helper" if isinstance(function, Helper) else "tile function"
        where = Where(function.name, what, lines, first, file_of(function.function))
        parameters = [a.arg for a in definition.args.args]
        statements = definition.body[1:] if docstring(definition) else definition.body
        if len(parameters) != len(kinds) or len(statements) != 1:
            refuse(
                where,
                definition,
                "not one return over its parameters",
                "write `return <expression>`",
            )
        returned = statements[0]
        if not isinstance(returned, ast.Return) or returned.value is None:
            refuse(
                where,
                returned,
                "a line that is not its return",
                "write `return <expression>`",
            )
        value = returned.value
        names = function.function.__globals__
    if (set(parameters) | {function.name}) & RESERVED:
        refuse(where, definition, "a name that is an IR word", "rename it")
    body = Body(
        where, returned, dict(zip(parameters, kinds)), returned.lineno, names, helpers
    )
    written = bare(expression(body, value))
    start = text.line(0, f"func {function.name}(")
    for name, kind in zip(parameters, kinds):
        text.line(1, f"{name}: {kind},")
    text.line(0, f") -> {kinds[0]} {{")
    line = len(text.lines) + 1
    depth, current, opened = 1, "return ", []
    marks = f"({CALL_OPEN}[0-9]+,[0-9]+{CALL_OPEN}|{CALL_CLOSE})"
    for token in re.split(marks, written):
        if token.startswith(CALL_OPEN):
            text.line(depth, current.rstrip())
            lo, hi = (int(n) for n in token.strip(CALL_OPEN).split(","))
            opened.append((len(text.lines) + 1, lo, hi))
            depth, current = 2, ""
        elif token == CALL_CLOSE:
            text.line(2, current.rstrip())
            first, lo, hi = opened.pop()
            text.place(first, where.file, lo, hi)
            current = ""
        else:
            current += token
    text.line(depth, current.rstrip())
    text.place(line, where.file, returned.lineno, returned.end_lineno)
    text.line(0, "}")
    text.place(start, where.file, definition.lineno, definition.lineno)


def docstring(definition):
    """Whether a function's first statement is its docstring (which writes nothing)."""
    first = definition.body[0]
    text = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
    return text and isinstance(first.value.value, str) and len(definition.body) > 1


def expression(body, value):
    """A tile function's expression: + - * / of its parameters and numbers, the tile
    compiler's functions of one tile (`km.exp(x)`) and calls of helpers on its
    parameters."""
    operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    if isinstance(value, ast.Name) and value.id in body.kinds:
        return value.id
    if isinstance(value, ast.Constant) and type(value.value) in (int, float):
        return repr(value.value)
    if isinstance(value, ast.BinOp) and type(value.op) in operators:
        left = expression(body, value.left)
        right = expression(body, value.right)
        return f"({left} {operators[type(value.op)]} {right})"
    if isinstance(value, ast.Call):
        return own_line(body, value, call(body, value))
    refuse(
        body.where,
        body.node,
        "an expression the IR module does not write yet",
        "write + - * / of its parameters",
    )


def own_line(body, value, written):
    """A call as written, on lines of its own in parentheses when it starts after the
    statement's first line (the call's line, as today's trace names it)."""
    if value.lineno == body.first:
        return written
    at = f"{CALL_OPEN}{value.lineno},{value.end_lineno}{CALL_OPEN}"
    return f"({at}{written}{CALL_CLOSE})"


def call(body, value):
    """`km.exp(x)`, one of the tile compiler's functions of one tile on one operand;
    or `helper(a, b)`, a plain function of the module on parameters."""
    func = value.func
    if value.keywords:
        refuse(
            body.where,
            body.node,
            f"a call's {value.keywords[0].arg}= is not written in an IR module yet",
            "leave it out",
        )
    if isinstance(func, ast.Attribute) and kind_of(func.attr, MATH):
        if len(value.args) != 1:
            refuse(body.where, body.node, f"{func.attr} of other than one operand", "")
        return f"{func.attr}({bare(expression(body, value.args[0]))})"
    helper = body.names.get(func.id) if isinstance(func, ast.Name) else None
    if not inspect.isfunction(helper):
        refuse(
            body.where,
            body.node,
            "a call the IR module does not write yet",
            "call km's functions of a tile, or a helper on its parameters",
        )
    if not all(isinstance(a, ast.Name) and a.id in body.kinds for a in value.args):
        refuse(
            body.where,
            body.node,
            f"a call of {func.id} on other than its parameters",
            "pass the function's parameters",
        )
    kinds = tuple(body.kinds[a.id] for a in value.args)
    if body.helpers.setdefault(func.id, (Helper(func.id, helper), kinds))[1] != kinds:
        refuse(
            body.where,
            body.node,
            f"{func.id} called on {body.helpers[func.id][1]} and on {kinds}",
            "call a helper on one kind",
        )
    return f"{func.id}({', '.join(a.id for a in value.args)})"


def bare(text):
    """An expression without the parentheses around the whole of it."""
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for i, char in enumerate(text):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if depth == 0 and i < len(text) - 1:
            return text
    return text[1:-1]


def held_lines(text):
    """(file, number, line as written) of every source line the map covers, in (file,
    number) order, each once."""
    wanted = sorted({(f, n) for (_, _, f, a, b) in text.map for n in range(a, b + 1)})
    sources = [Path(name).read_text().splitlines() for name in text.files]
    return [(f, n, sources[f][n - 1]) for f, n in wanted]
