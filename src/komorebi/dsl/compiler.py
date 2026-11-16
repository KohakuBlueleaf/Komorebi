"""The compiler of a program written as a class, level 1: each handler's body, read
with `ast`, becomes the nodes of the movement source (both movers) and of the compute
kernel; streams become buffer pairs, parameters command words, a refusal names the
line. `komorebi.dsl.program` writes the files and makes the Program.
"""

import ast
import copy
import inspect
import math
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from komorebi import abi
from komorebi.dsl import front, island_place
from komorebi.dsl.compile import (
    dst_tiles,
    field_fills,
    layout,
    messages,
    scope,
    stamp_pages,
    statements,
)
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.dst_code import (
    DST_HALF,
)
from komorebi.dsl.compile.frame import Frame
from komorebi.dsl.compile.messages import arrive_of
from komorebi.dsl.compile.records import (
    Buffer,
    Compiled,
    Param,
    Refused,
    Value,
)
from komorebi.dsl.compile.source import (
    Where,
    check_name,
    refuse,
    source_of,
)
from komorebi.dsl.names import kind_of, spelling
from komorebi.dsl.nodes import Block, Glue, Group, Line
from komorebi.program.several import COMPUTE_FLAGS

# Words of a command a handler's arguments may take (a0 to a9).
COMMAND_WORDS = 10
TILE_ELEMENTS = 32 * 32
TIMED_WAITS = (  # (the call's start, the counter, an expression or a statement)
    ("compute.wait_front(", "km_waited_in", True),
    ("km::broadcast_receive(compute, ", "km_waited_in", True),
    ("compute.reserve_back(", "km_waited_out", True),
    ("tile_regs_acquire();", "km_waited_out", False),
    ("tile_regs_wait();", "km_waited_in", False),
)


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
        Line(f"using {camel(tensor)} [[maybe_unused]] = {camel(alias)};")
        for tensor, alias in aliases.items()
    ]
    compiled.movement[handler] = [
        Block("{", "}", [*lines, *compiled.movement[handler]])
    ]


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
        if key not in COMPUTE_FLAGS:
            raise Refused(
                f"{where} {key!r}: the host API has no such option; give "
                f"math_fidelity or one of {', '.join(COMPUTE_FLAGS)}"
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
            layout.next_index(compiled),
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
        layout.added(compiled, buffer)
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


class HandlerCompiler(Frame):
    """One handler: its parameters, then each statement into the nodes of both
    kernels. Its state is the Frame's; its methods are the features not yet moved
    out as functions of the frame."""

    def __init__(self, compiled, method, names, plan, tensors):
        super().__init__(compiled, method, names, plan, tensors)
        method.widths = {}  # tensor parameter -> its rectangles' tile width
        self.definition, lines, first = source_of(method.function)
        self.where = Where(compiled.name, method.name, lines, first)
        self.assigned = scope.assigned_names(self)
        self.dynamic = messages.arrives(self)
        self.statements = partial(statements.statements, self)

    # The source -------------------------------------------------------------------

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
            messages.dynamic_start(self, compute)
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
            spans = stamp_pages.tile_stamps(self)
        elif self.method.stamps is not None:
            stamp_pages.stamps(self)
        if not self.dynamic:
            compute.append(Glue("bool km_ok = true;", {"km_ok"}))
        if spans is not None:
            compute.append(spans[0])
        self.block_words = {}
        marks = (len(movement), len(compute))
        self.rect_read = False
        body_m, body_c = self.statements(self.definition.body)
        if self.rect_param is not None and not (self.rect_read or self.movers):
            refuse(
                self.where,
                self.definition,
                "its rectangle is given at each submit, and it reads no self.x, "
                "self.y or broadcast",
                f"leave rect out of km.{spelling('up_to')}",
            )
        compute[marks[1] : marks[1]] = self.block_lines()
        movement[marks[0] : marks[0]] = self.block_lines()
        movement += self.opening_b + body_m + self.put_waits()
        compute += self.opening + body_c + dst_tiles.carried_end(self)
        if spans is not None:
            if not any(call[0] == name for call in self.compiled.tile_calls):
                refuse(
                    self.where,
                    self.definition.decorator_list[0],
                    f"stamps={self.method.stamps!r} of a resident handler are the "
                    "cycles of its tile function calls, and it calls none",
                    "call a tile function, or write @km.handler(overlay=True, "
                    'stamps="t")',
                )
            if stamp_pages.calls_steps(self):
                timed_waits(body_c)
            compute.append(spans[1])
            movement.append(spans[2])
        for kept, filled in self.filled.items():
            if filled is not None and not (
                kept in self.waited_fields or field_fills.kept_field(self, kept)
            ):
                refuse(
                    self.where,
                    filled[0],
                    f"self.{kept} is filled and no km.cxx body takes it",
                    f"take it: km.cxx(..., takes=[self.{kept}])",
                )
        for value in self.reduced:
            if value not in self.leader_writes:
                refuse(
                    self.where,
                    self.definition,
                    f"the reduce's {value} is not written by the leader",
                    f"write it under `if self.{spelling('rank')} == "
                    f"self.{spelling('leader')}:`",
                )
        self.compiled.movement[name] = movement
        self.compiled.compute[name] = compute + self.closing

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
            refuse(
                self.where,
                self.definition,
                "a handler's parameters are plain, one a command word",
                "write `def name(self, a: km.Tensor, b: int, ...)`",
            )
        parameters = [p for p in args.args[1:] if p.arg not in constants]
        words = []
        for parameter in parameters:
            word = len(words)  # a rectangle takes four
            kind = self.parameter_kind(parameter)
            check_name(self.where, parameter, parameter.arg)
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
            refuse(
                self.where,
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
                refuse(
                    self.where,
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
            refuse(
                self.where,
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
        refuse(
            self.where,
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
            refuse(
                self.where,
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
                    "const km::RectPlace km_place = km::rect_place("
                    f"{island_place.card_core(me, self.compiled.islands)}, "
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
            refuse(
                self.where,
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

    # Statements --------------------------------------------------------------------

    # Integers ----------------------------------------------------------------------

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

    # Ranges, partitions ------------------------------------------------------------

    # Tiles -------------------------------------------------------------------------

    # Streams -----------------------------------------------------------------------

    # Tile functions ----------------------------------------------------------------

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

    # Reduce ------------------------------------------------------------------------

    # Broadcasts --------------------------------------------------------------------

    # Inline bodies and results -----------------------------------------------------

    # Entries: a page's words as a table, on every CPU -------------------------------
