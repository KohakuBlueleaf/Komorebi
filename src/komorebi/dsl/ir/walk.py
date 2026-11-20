"""The walk: a class's handlers, read with `ast` as the class compiler does, as one IR
module's text with each line's origin. It knows the corpus's lines (a word, a partition,
streams, a loop pushing a tile function's result or expression, a sum reduced and
written by the leader); any other is refused by name, with its class's line."""

import ast
import inspect
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path

from komorebi.chain import calls as kc
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compile.source import MAKE_OPTIONS, Where, file_of, refuse, source_of
from komorebi.dsl.compiler import class_annotations, field_users, handler_methods
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
    "view", "sink", "block", "tile", "reduced"), a partition's or a view's tensor and
    range's words, a block's tiles."""

    kind: str
    format: str = ""
    tensor: str = ""
    first: str = ""
    count: str = ""
    tiles: int = 1
    block: str = (
        ""  # a block's count (or rows) as the text writes it, when not a number
    )
    rows: int = 1  # a block's rows of tiles: a stream of a column's in [n, 1]
    cols: str = ""  # a block's columns as the text writes them, when not a number
    field: bool = False  # a field of the class, the handler's var (R48)


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
    constants = constant_values(cls, make)
    fields = field_vars(cls, methods, constants)
    text = Text()
    walks = [Handler(cls, method, tensors, constants, fields) for method in methods]
    functions = {}
    for walk in walks:
        walk.read()
        for name, (function, kinds, result) in walk.functions.items():
            given = functions.setdefault(name, (function, kinds, result))
            if given[1:] != (kinds, result):
                raise Refused(
                    f"{cls.__name__}: {name} is called on {given[1]} and on "
                    f"{kinds}: a tile function on a second format is not written yet"
                )
    formats = []
    for walk in walks:
        for value in [*walk.params.values(), *walk.scope.values(), *walk.vars()]:
            if value.kind in ("tensor", "tile") and value.format not in formats:
                formats.append(value.format)
    text.line(0, f"module {cls.__name__.lower()}")
    text.line(0, "")
    text.line(0, "tile [32, 32]")
    text.line(0, f"format {', '.join(formats)}")
    # each constant, as today's thread header prints each (`class_thread`)
    if constants:
        text.line(0, "")
    for name in constants:
        # fixed when the module is built: the make's value (a compile input)
        text.line(0, f"param {name}: u32 = {constants[name]} @bind(build)")
    helpers = {}
    for function, kinds, result in functions.values():
        text.line(0, "")
        tile_function(text, function, kinds, helpers, result)
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
    declared = front.constants_of(cls)
    return {
        k: v for k, v in make.items() if k not in MAKE_OPTIONS and k not in declared
    }


def constant_values(cls, make):
    """The class's constants this make gives them, by name: each an int
    (`BLOCK = km.constant(8)`, a module parameter), its default or the make's keyword
    of its name; another kind of constant is refused by name."""
    values = {}
    for name, declared in front.constants_of(cls).items():
        if isinstance(declared, front.Derived):
            # a constant of others (`W_BLOCK = CHUNK * WIDTH`): its value at this make
            if name in make:
                raise Refused(
                    f"{cls.__name__}: {name}= sets a derived constant; set the "
                    "constants it is made of"
                )
            value = declared.value(values)
            if type(value) is not int or value < 0:
                raise Refused(f"{cls.__name__}: {name}={value!r} is not a u32")
            values[name] = value
            continue
        if not (isinstance(declared, front.Constant) and declared.kind is int):
            raise Refused(
                f"{cls.__name__}: its constant {name} is not an int; a constant other "
                "than an int is not written in an IR module yet"
            )
        value = make.get(name, declared.default)
        if type(value) is not int or value < 0:
            raise Refused(f"{cls.__name__}: {name}={value!r} is not a u32")
        values[name] = value
    return values


def class_refusals(cls, methods):
    """What the class states beyond its handlers' lines, refused by name."""
    name = cls.__name__
    stated = [
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


def field_vars(cls, methods, constants):
    """Each field's var type (R48), placed by today's `compiler.field_users`: a field
    one handler fills and another reads is kept, a state member, not written yet; a
    field of other than one row of tiles, or of words, is refused by name."""
    name = cls.__name__
    fields = {
        f: v for f, v in class_annotations(cls).items() if isinstance(v, front.Field)
    }
    names = {id(c): n for n, c in front.constants_of(cls).items()}
    out = {}
    for field_, (fillers, readers) in field_users(methods, fields).items():
        declared = fields[field_]
        if fillers and readers:
            raise Refused(
                f"{name}: self.{field_} is filled by {fillers[0]} and read by "
                f"{readers[0]}: a field kept between commands is a state member, not "
                "written in an IR module yet"
            )
        if inspect.getattr_static(cls, field_, None) is not None:
            raise Refused(
                f"{name}: self.{field_} has a value (km.own()): not written in an IR "
                "module yet"
            )
        format_ = TILE_FORMATS.get(kind_of(declared.format.dtype, ("fp32", "bf16")))
        format_ = format_ or FORMATS.get(declared.format.dtype)
        shape = declared.shape
        sizes = [
            s if type(s) is int else constants.get(names.get(id(s))) for s in shape
        ]
        if (
            format_ is None
            or len(shape) != 4
            or sizes[0] != 1
            or sizes[2:] != [32, 32]
            or sizes[1] is None
        ):
            raise Refused(
                f"{name}: self.{field_} is not one row of 32 x 32 tiles of "
                f"{', '.join(FORMATS)}: not written in an IR module yet"
            )
        width = names.get(id(shape[1]), str(shape[1]))
        out[field_] = Value("tile", format=format_, tiles=sizes[1], cols=width)
        out[field_].field = True
    return out


class Handler:
    """One handler's walk: its parameters and lines read, then written."""

    def __init__(self, cls, method, tensors, constants, fields=None):
        self.cls, self.method, self.tensors = cls, method, tensors
        self.constants = constants
        self.fields = fields or {}
        self.declared = []  # the fields this handler uses, in order of first use
        definition, lines, first = source_of(method.function)
        self.definition = definition
        self.file = file_of(method.function)
        self.where = Where(cls.__name__, method.name, lines, first, self.file)
        self.globals = method.function.__globals__
        self.params, self.scope = {}, {}
        # tile function name -> (TileFunction, its parameters' kinds, its result's)
        self.functions = {}
        self.restated = {}  # (partition, tensor) -> the name of its restatement
        self.statements = []  # (node, [(depth offset, text)], children)
        self.renames, self.names = [], {}
        used = {n.id for n in ast.walk(definition) if isinstance(n, ast.Name)}
        self.used = used | {a.arg for a in definition.args.args}

    def refuse(self, node, why, instead):
        refuse(self.where, node, why, instead)

    def vars(self):
        """The fields this handler declares as its vars."""
        return [self.fields[f] for f in self.declared]

    def field_of(self, node, expression):
        """The field `self.<name>` names, declared as the handler's var at its first
        use; None for another expression."""
        name = self_attribute(expression)
        if name not in self.fields:
            return None
        if name not in self.declared:
            if name in self.used or name in self.params:
                self.refuse(
                    node,
                    f"self.{name} and a name {name} of the handler: the field is the "
                    "handler's var of its name",
                    "rename one",
                )
            self.declared.append(name)
            self.renames.append((name, f"self.{name}"))  # the source's name of the var
        return name

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
        if isinstance(annotation, ast.Attribute) and kind_of(
            annotation.attr, ("pattern",)
        ):
            # a word the program does not read: a stream's hint names it
            return Value("pattern")
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
        """[(node, lines, children)] of its statements; a `pass` writes nothing."""
        return [self.statement(n) for n in statements if not isinstance(n, ast.Pass)]

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
            hint = self.pattern_of(node, [value])
            lines[-1] = f"let {self.name(target)} = {lines[-1]}{hint}"
            self.scope[target] = Value("sink")
        elif construct == "zeros":
            lines = [self.zeros(node, target, value)]
        elif construct == "reduce":
            lines = [self.reduce(node, target, value)]
        elif self.pages_of(value):
            # a view named (`pages = t.pages(a, b)`), streamed by its name
            tensor = value.func.value.id
            first, count = (bare(self.word(node, arg)) for arg in value.args)
            lines = [
                f"let {self.name(target)} = {self.name(tensor)}.tiles({first}, {count})"
            ]
            self.scope[target] = Value("view", tensor=tensor, first=first, count=count)
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
        """`total = km.tile.zeros(km.fp32)` (or `(km.fp32, 1)`, one tile): a tile of
        zeros the handler adds to."""
        given = list(value.args)
        one = len(given) == 2 and isinstance(given[1], ast.Constant)
        if one and given[1].value == 1 and type(given[1].value) is int:
            given = given[:1]
        (arg,) = given if len(given) == 1 and not value.keywords else (None,)
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
        """`total += tile`: the loop's tile added to a tile of the handler; or
        `acc += self.f(xb, wb)`, a tile function's result (a product of blocks)."""
        target, value = node.target, node.value
        field_ = self.field_of(node, target)
        if field_ is not None and isinstance(node.op, ast.Add):
            if not isinstance(value, ast.Call):
                self.refuse(
                    node,
                    "an add into a field of what is not a tile function's result",
                    f"add self.f(...) into self.{field_}",
                )
            result = type_text(self.fields[field_])
            return f"{self.name(field_)} += {self.tile_call(node, value, result)}"
        if (
            isinstance(node.op, ast.Add)
            and isinstance(target, ast.Name)
            and self.kind(target.id) == "tile"
            and isinstance(value, ast.Call)
        ):
            result = f"tile<{self.scope[target.id].format}>"
            return f"{self.name(target.id)} += {self.tile_call(node, value, result)}"
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
        """`t.pages(a, b).write(whole)`: a reduce's result, or a tile of the handler,
        written to t's tiles."""
        pages = call.func.value
        whole = call.args[0] if len(call.args) == 1 and not call.keywords else None
        field_ = self.field_of(node, whole)
        if field_ is not None:
            if self.rectangle_of(pages) is None:
                self.refuse(
                    node,
                    f"a write of self.{field_} to what is not a rectangle",
                    "write it to t.tiles[r0:r1, c0:c1]",
                )
            return f"store {self.view(node, pages, [])} <- {self.name(field_)}"
        if not (
            isinstance(whole, ast.Name) and self.kind(whole.id) in ("reduced", "tile")
        ):
            self.refuse(
                node,
                "a page write of what is not a reduce's result or a tile",
                "write the tile km.reduce(...) gives, or a tile of the handler",
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

    def pages_of(self, view):
        """Whether `view` is `t.pages(a, b)` of a tensor parameter t."""
        return (
            isinstance(view, ast.Call)
            and isinstance(view.func, ast.Attribute)
            and view.func.attr == spelling("pages")
            and isinstance(view.func.value, ast.Name)
            and self.kind(view.func.value.id) == "tensor"
            and len(view.args) == 2
            and not view.keywords
        )

    def view(self, node, view, lines):
        """The text of a stream's view: `t.pages(a, b)` as `t.tiles(a, b)`; a view's
        name; of `t[p]`, p's name or p's range of tensor t restated (`lines` gains the
        restatement)."""
        if isinstance(view, ast.Name) and self.kind(view.id) == "view":
            return self.name(view.id)
        if self.pages_of(view):
            first, count = (bare(self.word(node, arg)) for arg in view.args)
            return f"{self.name(view.func.value.id)}.tiles({first}, {count})"
        rectangle = self.rectangle_of(view)
        if rectangle is not None:
            ends = [bare(self.word(node, end)) for part in rectangle for end in part]
            tensor = self.name(view.value.value.id)
            return f"{tensor}.tiles[{ends[0]}:{ends[1]}, {ends[2]}:{ends[3]}]"
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

    def rectangle_of(self, view):
        """`t.tiles[r0:r1, c0:c1]` of a tensor parameter t: its (r0, r1) and (c0, c1)
        nodes; None for another view. A step is refused."""
        if not (
            isinstance(view, ast.Subscript)
            and isinstance(view.value, ast.Attribute)
            and view.value.attr == spelling("tiles")
            and isinstance(view.value.value, ast.Name)
            and self.kind(view.value.value.id) == "tensor"
            and isinstance(view.slice, ast.Tuple)
            and len(view.slice.elts) == 2
            and all(isinstance(s, ast.Slice) for s in view.slice.elts)
        ):
            return None
        parts = view.slice.elts
        if any(s.step is not None or s.lower is None or s.upper is None for s in parts):
            self.refuse(
                view,
                "a rectangle with a step or an end not written",
                "write t.tiles[r0:r1, c0:c1]",
            )
        return [(s.lower, s.upper) for s in parts]

    def stream(self, node, call, write):
        """The lines of `km.Stream(t[p], block=n)` (n 1 when not given), the last one
        the stream itself."""
        keywords = {k.arg: k.value for k in call.keywords}
        # `pattern=` and `repeat=` by the keyword, as page_streams reads them
        allowed = {spelling("block"), "pattern"}
        allowed |= {spelling("write_stream")} if write else {"repeat"}
        for key in keywords:
            if key not in allowed:
                self.refuse(
                    node,
                    f"a stream's {key}= is not written in an IR module yet",
                    "leave it out",
                )
        block, _ = self.block_of(node, call)
        if len(call.args) != 1:
            self.refuse(node, "a stream of more than one view", "stream one view")
        lines = []
        view = self.view(node, call.args[0], lines)
        word = "sink" if write else "stream"
        lines.append(f"{word}({view}, block = {self.block_text(node, call)})")
        return lines

    def block_text(self, node, call):
        """A stream's block as the text writes it (`block_of`); of a rectangle of one
        column, `[n, 1]`: n rows of one tile (a block is a matrix of tiles, R45); of a
        rectangle of C columns, more than one row, a block `A * C` declared (`W_BLOCK
        = CHUNK * WIDTH`) is `[A, C]`."""
        text, _ = self.block_of(node, call)
        view = call.args[0]
        if self.rectangle_of(view) is None or self.one_row(view):
            return text
        if self.column(view):
            return f"[{text}, 1]"
        rows = self.rows_of(node, call)
        return f"[{rows[0]}, {rows[1]}]"

    def one_row(self, view):
        """Whether a rectangle is one row of tiles: `r : r + 1`, or two numbers one
        apart."""
        return one_apart(*self.rectangle_of(view)[0])

    def rows_of(self, node, call):
        """A block over a rectangle of C columns (not one row, not one column): its
        declaration `A * C`, C the columns' constant (`c : c + self.C`), as (A, C)
        texts; any other is refused by name."""
        lower, upper = self.rectangle_of(call.args[0])[1]
        keywords = {k.arg: k.value for k in call.keywords}
        block = self_attribute(keywords.get(spelling("block")))
        declared = front.constants_of(self.cls).get(block)
        columns = (
            self_attribute(upper.right)
            if isinstance(upper, ast.BinOp)
            and isinstance(upper.op, ast.Add)
            and ast.dump(upper.left) == ast.dump(lower)
            else None
        )
        named = {id(c): n for n, c in front.constants_of(self.cls).items()}
        if (
            isinstance(declared, front.Derived)
            and declared.op is operator.mul
            and columns is not None
            and named.get(id(declared.right)) == columns
            and named.get(id(declared.left)) in self.constants
        ):
            return named[id(declared.left)], columns
        self.refuse(
            node,
            "a stream's block over a rectangle of several rows and columns that is not "
            "a constant declared A * C, C the rectangle's columns (`c : c + self.C`)",
            "declare the block W_BLOCK = ROWS * C and stream t.tiles[r0:r1, c:c + "
            "self.C]",
        )

    def column(self, view):
        """Whether a rectangle is one column of tiles: `c : c + 1`, or two numbers one
        apart."""
        return one_apart(*self.rectangle_of(view)[1])

    def block_of(self, node, call):
        """A stream's block as the text writes it and its tiles: a number (1 when not
        given, as today's `stream_options`, row=1), or `self.<constant>` of the class,
        the module parameter of its name."""
        keywords = {k.arg: k.value for k in call.keywords}
        block = keywords.get(spelling("block"), ast.Constant(1))
        if isinstance(block, ast.Constant) and type(block.value) is int:
            return str(block.value), block.value
        if (
            isinstance(block, ast.Attribute)
            and isinstance(block.value, ast.Name)
            and block.value.id == "self"
            and block.attr in self.constants
        ):
            return block.attr, self.constants[block.attr]
        self.refuse(
            node,
            "a stream's block that is not a number or a constant of the class",
            "write block=<n> or block=self.<constant>",
        )

    def pattern_of(self, node, calls):
        """The hint of the streams' `pattern=`: `@tensix.pattern(p)` of a km.Pattern
        parameter p, which only chooses which mover moves each block; none when no
        stream gives one. The streams of one loop give the same."""
        given = []
        for call in calls:
            keywords = {k.arg: k.value for k in call.keywords}
            given.append(keywords.get("pattern"))
        names = [g.id if isinstance(g, ast.Name) else None for g in given]
        if all(g is None for g in given):
            return ""
        if any(n is None or self.kind(n) != "pattern" for n in names):
            self.refuse(
                node,
                "a stream's pattern= that is not a km.Pattern parameter",
                "give the handler's km.Pattern parameter, or leave it out",
            )
        if len(set(names)) != 1:
            self.refuse(
                node,
                "the streams of one loop in different patterns",
                "give each stream of the loop the same pattern",
            )
        return f" @tensix.pattern({self.name(names[0])})"

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
        repeat = self.repeat_of(node, calls)
        if repeat is not None:
            # `repeat=r` is a counted loop around the stream's loop (reading R46)
            counter = "i"
            while counter in self.used or counter in RESERVED:
                counter += "_"
            self.used.add(counter)
            lines.append(f"for {counter} in range({repeat}) {{")
        lines.append(f"for {held} in {walked}{self.pattern_of(node, calls)} {{")
        for name, call in zip(names, calls):
            text, block = self.block_of(node, call)
            shaped = self.block_text(node, call)
            value = Value(
                "block",
                format=self.scope_format(call),
                tiles=block,
                block="" if text == str(block) else text,
            )
            if shaped.endswith(", 1]"):  # a column's: n rows of one tile
                value.tiles, value.rows = 1, block
            elif shaped.startswith("["):  # [A, C]: A rows of C tiles
                rows, cols = shaped[1:-1].split(", ")
                value.cols, value.block = cols, rows
                value.tiles = self.constants[cols]
                value.rows = self.constants[rows]
            self.scope[name.id] = value
        children = self.body(node.body)
        return (node, lines, children)

    def repeat_of(self, node, calls):
        """The count of a read stream's `repeat=`, an int of the handler's words; none
        when not given. Only a loop over one stream that drops its blocks repeats."""
        given = [k.value for c in calls for k in c.keywords if k.arg == "repeat"]
        if not given:
            return None
        if len(calls) != 1:
            self.refuse(
                node, "a repeat of a loop over two streams", "repeat one stream's loop"
            )
        if not all(isinstance(n, ast.Pass) for n in node.body):
            self.refuse(
                node,
                "a repeat of a loop with a body",
                "repeat a loop that drops its blocks (`pass`)",
            )
        return bare(self.word(node, given[0]))

    def scope_format(self, call):
        """The format of the tensor a stream's view reads."""
        view = call.args[0]
        if isinstance(view, ast.Name):
            return self.scope[self.scope[view.id].tensor].format
        if self.rectangle_of(view) is not None:
            return self.scope[view.value.value.id].format
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
        if isinstance(pushed, ast.Name) and self.kind(pushed.id) == "block":
            # the loop's block itself, copied
            return f"{self.name(func.value.id)}.push({self.name(pushed.id)})"
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

    def tile_call(self, node, call, result=None):
        """A tile function's call on a block and scalar parameters: `f(...)` of the
        module, or `self.f(...)` of the class; `result` the type of its result, when
        not its first operand's (a product's, the tile it is added to)."""
        func = call.func if isinstance(call, ast.Call) else None
        method = self_attribute(func)
        if not (
            (isinstance(func, ast.Name) or method is not None) and not call.keywords
        ):
            self.refuse(
                node,
                "a push of what is not a tile function's call",
                "push f(block, ...)",
            )
        name = method if method is not None else func.id
        if method is not None:
            found = [vars(k)[name] for k in self.cls.__mro__ if name in vars(k)]
            function = found[0] if found else None
        else:
            function = self.globals.get(name)
        if not isinstance(function, front.TileFunction):
            self.refuse(
                node,
                f"{name} is not a tile function",
                "decorate it with km.tile_function",
            )
        # tolerance= writes nothing (the IR states no precision; today's files are the
        # same without it); fidelity= is the hint `@tensix.fidelity(...)`
        if not all(isinstance(arg, ast.Name) for arg in call.args):
            self.refuse(
                node,
                "a tile function's operand that is not a block or a float",
                "pass the loop's block or a float",
            )
        names = [arg.id for arg in call.args]
        return self.tile_operands(node, function, names, "a tile function", result)

    def tile_operands(self, node, function, names, what, result=None):
        """The call of `function` on the handler's `names`: a block first, then floats;
        a block of n tiles typed `tile<F>[n]`, of r rows `tile<F>[r, n]` (the function
        is declared on blocks)."""
        kinds = []
        for name in names:
            if self.kind(name) not in ("block", "f32"):
                self.refuse(
                    node,
                    f"{what}'s operand that is not a block or a float",
                    "pass the loop's block or a float",
                )
            value = self.scope[name]
            tiles = f"[{value.block or value.tiles}]" if value.tiles > 1 else ""
            if value.rows > 1:
                tiles = f"[{value.block or value.rows}, {value.cols or value.tiles}]"
            kinds.append(
                f"tile<{value.format}>{tiles}" if value.kind == "block" else "f32"
            )
        if "tile<" not in "".join(kinds[:1]):
            self.refuse(
                node,
                f"{what} whose first operand is not a block",
                "pass the block first",
            )
        self.functions[function.name] = (function, tuple(kinds), result or kinds[0])
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
                "a constant of the class, + - * // or %",
                "write it of those",
            )
        return text

    def int_text(self, expression):
        """The text of an int expression of int parameters and values, numbers (a
        module's int constant as its number), `self.rank`, a class's constant (its
        module parameter), + - * // %; else None."""
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
        if self_attribute(expression) in self.constants:
            return expression.attr
        operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
        operators |= {ast.FloorDiv: "/", ast.Mod: "%"}
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
            kind = "u32" if kind == "pattern" else kind
            text.line(2, f"{self.name(name)}: {kind},")
        text.line(1, ") @on(group) {")
        for field_ in self.declared:  # a field: the handler's var from zeros (R48)
            value = self.fields[field_]
            text.line(2, f"var {self.name(field_)}: {type_text(value)} = zeros")
        self.write_body(text, self.statements, 2)
        text.line(1, "}")
        line = self.definition.lineno
        text.place(start, self.file, line, line)

    def write_body(self, text, statements, depth):
        for node, lines, children in statements:
            first = len(text.lines) + 1
            opened = 0  # the blocks the statement's lines open, one inside the other
            for line in lines:
                text.line(depth + opened, line)
                opened += line.endswith("{")
            if opened:
                self.write_body(text, children, depth + opened)
            for inner in reversed(range(opened)):
                text.line(depth + inner, "}")
            text.place(first, self.file, node.lineno, node.end_lineno)


def one_apart(lower, upper):
    """Whether a side `lower : upper` is one tile: `a : a + 1`, or two numbers one
    apart."""
    if isinstance(lower, ast.Constant) and isinstance(upper, ast.Constant):
        return upper.value == lower.value + 1
    return (
        isinstance(upper, ast.BinOp)
        and isinstance(upper.op, ast.Add)
        and ast.dump(upper.left) == ast.dump(lower)
        and isinstance(upper.right, ast.Constant)
        and upper.right.value == 1
    )


def type_text(value):
    """The type of a tile of the handler: `tile<F>`, or a field's var `tile<F>[n]`
    (one row of its tiles, even of one)."""
    if value.field:
        return f"tile<{value.format}>[{value.cols or value.tiles}]"
    return f"tile<{value.format}>"


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


def tile_function(text, function, kinds, helpers, result=None):
    """A tile function as a `func`: its one `return` of + - * / @ over its parameters
    and numbers, the tile compiler's functions of one tile and calls of helpers, typed
    by the call (`result`, else its first parameter's); a tile expression alike. A call
    on a line after the statement's first is written on a line of its own."""
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
    fidelity = getattr(function, "fidelity", None)
    hint = f" @tensix.fidelity({fidelity})" if fidelity else ""
    text.line(0, f") -> {result or kinds[0]}{hint} {{")
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
    """A tile function's expression: + - * / of its parameters and numbers, `a @ b`
    (`matmul(a, b)`), the tile compiler's functions of one tile (`km.exp(x)`) and calls
    of helpers on its parameters."""
    operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    if isinstance(value, ast.Name) and value.id in body.kinds:
        return value.id
    if isinstance(value, ast.Constant) and type(value.value) in (int, float):
        return repr(value.value)
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.MatMult):
        left = bare(expression(body, value.left))
        right = bare(expression(body, value.right))
        return f"matmul({left}, {right})"
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
