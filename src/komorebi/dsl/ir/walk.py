"""The walk: a class's handlers, read with `ast` as the class compiler reads them,
written as one IR module's text with the origin of each of its lines. It knows the
first lesson's lines (a partition, a write stream, a read loop pushing a tile
function's result); any other line is refused by name, with its class's line."""

import ast
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
    "sink", "block") and, for a partition, its tensor and its range's words."""

    kind: str
    format: str = ""
    tensor: str = ""
    first: str = ""
    count: str = ""


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
        for value in walk.params.values():
            if value.kind == "tensor" and value.format not in formats:
                formats.append(value.format)
    text.line(0, f"module {cls.__name__.lower()}")
    text.line(0, "")
    text.line(0, "tile [32, 32]")
    text.line(0, f"format {', '.join(formats)}")
    for function, kinds in functions.values():
        text.line(0, "")
        tile_function(text, function, kinds)
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
        if isinstance(node, ast.For):
            return self.loop(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
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
        if construct == "partition":
            lines = [self.partition(node, target, value)]
        elif construct == "stream" and write_stream(value):
            lines = self.stream(node, value, write=True)
            lines[-1] = f"let {self.name(target)} = {lines[-1]}"
            self.scope[target] = Value("sink")
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

    def view(self, node, view, lines):
        """The text name of a stream's view `t[p]`: p's name, or p's range of tensor t
        restated (`lines` gains the restatement)."""
        if not (
            isinstance(view, ast.Subscript)
            and isinstance(view.value, ast.Name)
            and isinstance(view.slice, ast.Name)
            and self.kind(view.value.id) == "tensor"
            and self.kind(view.slice.id) == "partition"
        ):
            self.refuse(
                node, "a stream of what is not t[part]", "stream a tensor's partition"
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
        """The lines of `km.Stream(t[p], block=n)`, the last one the stream itself."""
        keywords = {k.arg: k.value for k in call.keywords}
        allowed = {spelling("block")} | ({spelling("write_stream")} if write else set())
        for key in keywords:
            if key not in allowed:
                self.refuse(
                    node,
                    f"a stream's {key}= is not written in an IR module yet",
                    "leave it out",
                )
        block = keywords.get(spelling("block"))
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
        """`for b in km.Stream(t[p], block=n):` over its lines."""
        if node.orelse or not isinstance(node.target, ast.Name):
            self.refuse(
                node, "a loop with an else or several names", "loop with one name"
            )
        if construct_of(node.iter) != "stream" or write_stream(node.iter):
            self.refuse(
                node,
                "a loop over what is not a read stream",
                "loop over km.Stream(t[p], block=n)",
            )
        lines = self.stream(node, node.iter, write=False)
        target = node.target.id
        lines[-1] = f"for {self.name(target)} in {lines[-1]} {{"
        self.scope[target] = Value("block", format=self.scope_format(node.iter))
        children = self.body(node.body)
        return (node, lines, children)

    def scope_format(self, call):
        return self.scope[call.args[0].value.id].format

    def push(self, node, call):
        """`out.push(f(b, k))`: a tile function's result into a write stream."""
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
        return f"{self.name(func.value.id)}.push({self.tile_call(node, call.args[0])})"

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
        kinds, names = [], []
        for arg in call.args:
            if not isinstance(arg, ast.Name) or self.kind(arg.id) not in (
                "block",
                "f32",
            ):
                self.refuse(
                    node,
                    "a tile function's operand that is not a block or a float",
                    "pass the loop's block or a float",
                )
            value = self.scope[arg.id]
            kinds.append(f"tile<{value.format}>" if value.kind == "block" else "f32")
            names.append(self.name(arg.id))
        if "tile<" not in "".join(kinds[:1]):
            self.refuse(
                node,
                "a tile function whose first operand is not a block",
                "pass the block first",
            )
        self.functions[function.name] = (function, tuple(kinds))
        return f"{function.name}({', '.join(names)})"

    def kind(self, name):
        value = self.scope.get(name)
        return value.kind if value else None

    def word(self, node, expression):
        """An int expression of the handler's words: names and numbers, + - *."""
        if isinstance(expression, ast.Name) and self.kind(expression.id) == "u32":
            return self.name(expression.id)
        if isinstance(expression, ast.Constant) and type(expression.value) is int:
            return str(expression.value)
        operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
        if isinstance(expression, ast.BinOp) and type(expression.op) in operators:
            left, right = self.word(node, expression.left), self.word(
                node, expression.right
            )
            return f"({left} {operators[type(expression.op)]} {right})"
        self.refuse(
            node,
            "a word that is not an int parameter, a number, + - or *",
            "write it of those",
        )

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
    """The construct a call is: "partition" (self.partition), "stream" (km.Stream)."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    func = node.func
    if isinstance(func.value, ast.Name) and func.value.id == "self":
        return kind_of(func.attr, ("partition",))
    if isinstance(func.value, ast.Name) and func.value.id == "km":
        return kind_of(func.attr, ("stream",))
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


def tile_function(text, function, kinds):
    """A tile function as a `func`: its one `return` of + - * / over its parameters and
    numbers, its parameters typed by the call."""
    definition, lines, first = source_of(function.function)
    where = Where(
        function.name, "tile function", lines, first, file_of(function.function)
    )
    parameters = [a.arg for a in definition.args.args]
    if len(parameters) != len(kinds) or len(definition.body) != 1:
        refuse(
            where,
            definition,
            "not one return over its parameters",
            "write `return <expression>`",
        )
    returned = definition.body[0]
    if not isinstance(returned, ast.Return) or returned.value is None:
        refuse(
            where,
            returned,
            "a line that is not its return",
            "write `return <expression>`",
        )
    names = set(parameters)
    if (names | {function.name}) & RESERVED:
        refuse(where, definition, "a name that is an IR word", "rename it")
    body = bare(expression(where, returned, returned.value, names))
    start = text.line(0, f"func {function.name}(")
    for name, kind in zip(parameters, kinds):
        text.line(1, f"{name}: {kind},")
    text.line(0, f") -> {kinds[0]} {{")
    line = text.line(1, f"return {body}")
    text.place(line, where.file, returned.lineno, returned.end_lineno)
    text.line(0, "}")
    text.place(start, where.file, definition.lineno, definition.lineno)


def expression(where, node, value, names):
    """A tile function's expression: + - * / of its parameters and numbers."""
    operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    if isinstance(value, ast.Name) and value.id in names:
        return value.id
    if isinstance(value, ast.Constant) and type(value.value) in (int, float):
        return repr(value.value)
    if isinstance(value, ast.BinOp) and type(value.op) in operators:
        left = expression(where, node, value.left, names)
        right = expression(where, node, value.right, names)
        return f"({left} {operators[type(value.op)]} {right})"
    refuse(
        where,
        node,
        "an expression the IR module does not write yet",
        "write + - * / of its parameters",
    )


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
