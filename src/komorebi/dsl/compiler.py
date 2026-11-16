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
from pathlib import Path

from komorebi import abi
from komorebi.dsl import front, island_place
from komorebi.dsl.compile import (
    bodies,
    broadcasts,
    drained,
    dst_tiles,
    entry_tables,
    field_fills,
    handler_buffers,
    layout,
    messages,
    page_streams,
    page_words,
    puts,
    ranges,
    scope,
    stamp_pages,
    tile_functions,
    tile_lookup,
    tile_products,
    tile_reduce,
    tile_steps,
)
from komorebi.dsl.compile.cxx import camel, pattern_uses, range_uses
from komorebi.dsl.compile.dst_code import (
    DST_HALF,
    pack,
    sums_room,
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
    WAIT_DATA,
    WAIT_INPUT,
    WAIT_OUTPUT,
    Where,
    check_name,
    line_text,
    refuse,
    self_call,
    source_of,
    wait,
)
from komorebi.dsl.compile.tile_functions import tile_format_refusal
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

    def message_word(self, node):
        """The C++ of `message.word` (a word of a message), or None."""
        base = node.value
        if not (isinstance(base, ast.Name) and base.id in self.scope):
            return None
        value = self.scope[base.id]
        if value.kind != "message":
            return None
        if node.attr not in value.words:
            refuse(
                self.where,
                node,
                f"{base.id} has no word {node.attr}",
                f"use one of {', '.join(value.words)}",
            )
        cls = next(c for c, w in self.compiled.messages.items() if w is value.words)
        return f"{base.id}[k{cls}{camel(node.attr)}]"

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

    def statements(self, body):
        """The movement nodes and the compute nodes of a list of statements."""
        movement, compute = [], []
        self.blocks.append(id(body))
        for statement in body:
            m, c = self.statement(statement)
            comment = f"line {statement.lineno}: {line_text(self.where, statement)}"
            if statement.end_lineno - statement.lineno > 2 and not hasattr(
                statement, "body"
            ):  # a long call (an inline body): its first line
                first = self.where.lines[statement.lineno - self.where.first].strip()
                comment = f"line {statement.lineno}: {first} ..."
            if m:
                movement.append(Group(comment, m))
            if c:
                compute.append(Group(comment, c))
        compute += self.held_ends(body)
        compute += self.held_block_ends(body)
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
        if isinstance(node, ast.With):
            return drained.drained_with(self, node)
        if isinstance(node, ast.If):
            return self.branch(node)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return self.call_statement(node, node.value)
        if isinstance(node, ast.Return):
            return self.returned(node)
        refuse(
            self.where,
            node,
            "not a line of a handler at level 1",
            "write a move, an int, a loop over a stream or a range, an if, or "
            f"km.{spelling('cxx')}(...)",
        )

    def assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            if tile_lookup.tile_function_of(self, node.value) is not None:
                return tile_functions.tile_results(
                    self, node, node.targets[0], node.value
                )
            return page_words.read_words(self, node, node.targets[0], node.value)
        if len(node.targets) == 1 and self.word_name(node.targets[0]):
            return self.set_field_word(node, node.targets[0])
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
            return self.set_word(node, node.targets[0], self.integer(node.value))
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            refuse(self.where, node, "one name a line", "assign one name")
        name = node.targets[0].id
        check_name(self.where, node, name)
        scope.not_parameter(self, node, name)
        value = node.value
        made = messages.message_class(self, node, value)
        if made is not None:
            return messages.made_message(self, node, name, made, value)
        value = tile_functions.tile_expression(self, node, value) or value
        if tile_steps.step_of_field(self, value):
            return tile_steps.started_from_field(self, node, name, value)
        if tile_lookup.tile_function_of(self, value) is not None:
            return tile_functions.tile_call(self, node, value, name)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == spelling("read")
            and not value.args
            and not value.keywords
            and ranges.is_range(self, value.func.value)
        ):
            return page_words.read_word(
                self, node, name, ranges.range_of(self, value.func.value)
            )
        if entry_tables.entries_call(self, value) is not None:
            return entry_tables.entries_made(self, node, name, value)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == spelling("bump")
            and scope.field_name(self, value.func.value)
        ):
            return puts.bump_line(self, node, name, value)
        if self_call(value, "direct_message"):
            return messages.direct_call(self, node, name, value)
        call = self.construct(value)
        if call == "partition":
            return ranges.partition(self, node, name, value)
        if call == "zeros":
            return dst_tiles.zeros(self, node, name, value)
        if call == "stream" and scope.field_name(
            self, value.args[0] if value.args else None
        ):
            return drained.drained_stream(self, node, name, value)
        if call == "stream" and not self.write_keyword(value):
            return page_streams.held_stream(self, node, name, value)
        if call == "stream":
            return page_streams.write_stream(self, node, name, value)
        if call == "reduce":
            return tile_reduce.reduce(self, node, name, value)
        if call == "all_reduce":
            return tile_reduce.reduce(self, node, name, value, everywhere=True)
        if call == "broadcast":
            return broadcasts.broadcast(self, node, name, value)
        if call == "cxx":
            return bodies.inline(self, node, value, name)
        if call:
            refuse(
                self.where,
                node,
                f"km.{spelling(call)} gives no value to name",
                "write it on a line of its own",
            )
        if ranges.is_range(self, value):
            self.scope[name] = ranges.range_of(self, value)
            return [], []
        text = self.integer(value)
        declared = "uint32_t" if name in self.assigned else "const uint32_t"
        if name in self.scope:
            if self.scope[name].kind != "int" or name not in self.assigned:
                refuse(self.where, node, f"{name} is already defined", "use a new name")
            line = f"{name} = {text};"
            uses = scope.uses(self, value) | {name}
            return [Line(line, uses)], [Line(line, uses)]
        self.scope[name] = Value("int", name)
        glue = f"{declared} {name} = {text};"
        uses = scope.uses(self, value)
        return [Glue(glue, {name}, uses)], [Glue(glue, {name}, uses)]

    def augmented(self, node):
        target = node.target
        if self.word_name(target):
            op = OPERATORS.get(type(node.op))
            if op is None:
                refuse(
                    self.where, node, "an operator level 1 has no C++ for", "use + - *"
                )
            return self.set_field_word(node, target, op)
        field_ = scope.field_name(self, target)
        if (
            field_ is not None
            and tile_lookup.tile_function_of(self, node.value) is not None
        ):
            return tile_products.field_product(self, node, field_, node.value)
        if isinstance(target, ast.Attribute):
            op = OPERATORS.get(type(node.op))
            if op is None:
                refuse(
                    self.where, node, "an operator level 1 has no C++ for", "use + - *"
                )
            return self.set_word(node, target, self.integer(node.value), op)
        if isinstance(target, ast.Name):
            scope.not_parameter(self, node, target.id)
            scope.not_ended(self, node, target.id)
        if not isinstance(target, ast.Name) or target.id not in self.scope:
            refuse(self.where, node, "an unknown name", "assign it first")
        value = self.scope[target.id]
        if (
            value.kind == "tile"
            and tile_lookup.tile_function_of(self, node.value) is not None
        ):
            return tile_products.product(self, node, value, node.value)
        if value.kind == "tile":
            return dst_tiles.accumulate(self, node, value)
        if value.kind == "word" and value.mover in ("A", "B"):
            return self.word_sum(node, value)
        if value.kind != "int":
            refuse(self.where, node, f"{target.id} is a {value.kind}", "add to an int")
        op = OPERATORS.get(type(node.op))
        if op is None:
            refuse(
                self.where, node, "an operator level 1 has no C++ for", "use + - * etc."
            )
        text = f"{target.id} {op}= {self.integer(node.value)};"
        uses = scope.uses(self, node.value) | {target.id}
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
            refuse(
                self.where,
                node,
                f"self.{name} has {count} words, 0 to {count - 1}",
                "index one of them",
            )
        return f"km_field_{name}[{index}]", scope.uses(self, node.slice) | {
            f"km_field_{name}"
        }

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
        uses |= scope.uses(self, node.value)
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
            refuse(self.where, node, "an attribute set is a message's word", "set one")
        uses = scope.uses(self, node.value) | {target.value.id}
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
                return kind_of(
                    func.attr, ("stream", "reduce", "all_reduce", "broadcast", "cxx")
                )
            value = isinstance(func.value, ast.Name) and self.scope.get(func.value.id)
            if value and value.kind == "broadcast":
                return kind_of(func.attr, ("partition",))
        return None

    # Integers ----------------------------------------------------------------------

    def integer(self, node):
        """The C++ of an integer expression (glue)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            if isinstance(node.value, bool) or node.value < 0:
                refuse(self.where, node, "a word is a non-negative int", "write one")
            return f"{node.value}u" if node.value > 0x7FFFFFFF else str(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.scope:
                value = self.scope[node.id]
                if value.kind == "word" and self.allow_words:
                    return node.id
                if value.kind in ("streamed", "tile"):
                    refuse(
                        self.where,
                        node,
                        f"{node.id} is a tile: tile arithmetic is a tile function's",
                        "write it in a @km.tile_function and call that on the "
                        "stream's tiles",
                    )
                if value.kind == "float":
                    refuse(
                        self.where,
                        node,
                        f"{node.id} is a float word (a float32's bits), and this is "
                        "the handler's integer arithmetic (a slice bound, a range, a "
                        "count)",
                        f"pass an int word for it, or use {node.id} in a tile "
                        "expression, a tile function's call or a body",
                    )
                if value.kind != "int":
                    refuse(
                        self.where,
                        node,
                        f"{node.id} is a {value.kind}, not an int",
                        "use an int",
                    )
                return node.id
            scope.not_ended(self, node, node.id)
            return str(self.constant(node, node.id))
        if isinstance(node, ast.Attribute):
            return self.message_word(node) or self.attribute(node)
        if self.word_name(node):
            if not self.field_line:
                refuse(
                    self.where,
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
                refuse(
                    self.where,
                    node,
                    "an operator level 1 has no C++ for",
                    "use + - * //",
                )
            return f"({self.operand(node.left)} {op} {self.operand(node.right)})"
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = COMPARISONS.get(type(node.ops[0]))
            if op is None:
                refuse(
                    self.where,
                    node,
                    "a comparison level 1 has no C++ for",
                    "use == < >",
                )
            left, right = node.left, node.comparators[0]
            mirror = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
            for one, other, said in ((left, right, op), (right, left, mirror.get(op))):
                if (
                    isinstance(one, ast.BinOp)
                    and isinstance(one.op, ast.Sub)
                    and isinstance(other, ast.Constant)
                    and other.value == 0
                ):
                    refuse(
                        self.where,
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
                    refuse(
                        self.where,
                        node,
                        f"`{word}` of words is a word in Python ({ast.unparse(value)} "
                        "itself), and a truth in C++",
                        f"write {spelling('max')}(a, 1), or an if",
                    )
            op = " && " if isinstance(node.op, ast.And) else " || "
            return "(" + op.join(self.integer(v) for v in node.values) + ")"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return f"!({self.integer(node.operand)})"
        if self_call(node, "direct_message"):
            messages.direct_unnamed(self, node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in (spelling("min"), spelling("max"))
            and node.func.id not in self.scope
        ):  # the smaller (larger) of ints, as words
            if len(node.args) < 2 or node.keywords:
                refuse(
                    self.where,
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
        refuse(
            self.where, node, "not an integer of level 1", "write ints, names, + - * //"
        )

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
            refuse(
                self.where, node, f"{name} is not known here", "define it, or pass it"
            )
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            refuse(
                self.where, node, f"{name} = {value!r}: not a word", "make it an int"
            )
        self.compiled.constants[name] = value
        return value

    def constant_int(self, node):
        """An int known at build: a literal, a module constant or a plan value."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if self.is_class_constant(node):
            return self.class_constant(node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self.constant(node, node.value)  # a symbol: the plan's
        if isinstance(node, ast.Name) and node.id not in self.scope:
            return self.constant(node, node.id)
        self.word_given(node, "a size the build needs")
        refuse(
            self.where, node, "a size the build needs", "give an int, or a plan name"
        )

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

    def attribute(self, node):
        base = node.value
        if self.is_class_constant(node):
            return str(self.class_constant(node))
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
                where = f"line {node.lineno}"
                self.compiled.rect_lines.setdefault(self.method.name, where)
                self.compiled.rectangle = True
                self.compiled.rect_plain |= self.rect_param is None
                self.rect_read = True
                return f"km_self_{kind}"
            self.field_use(node)
            refuse(self.where, node, f"self.{node.attr} is not known", "use self.rank")
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
        refuse(
            self.where, node, "an attribute level 1 does not know", "use .start, .count"
        )

    def is_class_constant(self, node):
        """Whether `node` is self.X of a constant the class declares."""
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in self.compiled.declared
        )

    def class_constant(self, node):
        """self.X of a constant of the class: this program's value, lowered as a
        module constant is (a non-negative int)."""
        name = node.attr
        if name not in self.compiled.class_constants:
            refuse(
                self.where,
                node,
                f"self.{name} is a symbol the build solves after the handlers' lines",
                f"give {name} in the {spelling('plan')}, or as {name}= at the make",
            )
        value = self.compiled.class_constants[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            refuse(
                self.where,
                node,
                f"self.{name} = {value!r}: not a word",
                "make it an int",
            )
        return value

    def word_given(self, node, what):
        """Refuses a command word where `what` needs a constant: the tier's error."""
        if isinstance(node, ast.Name) and any(p.name == node.id for p in self.params):
            refuse(
                self.where,
                node,
                f"{node.id} is a word of the command, known only when the command "
                f"runs, and {what} is fixed when the program is built: it needs a "
                "constant of the class",
                f"declare {node.id.upper()} = km.{spelling('constant')}(n) in the "
                f"class and write self.{node.id.upper()}",
            )

    def stated_words(self, node, what):
        """word_given of each `n` a rectangle's side states as its most (r:r + n,
        r:min(r + n, e))."""
        if not isinstance(node, ast.Subscript):
            return
        sides = node.slice.elts if isinstance(node.slice, ast.Tuple) else []
        for side in sides:
            if not isinstance(side, ast.Slice) or side.upper is None:
                continue
            upper = side.upper
            ends = upper.args if isinstance(upper, ast.Call) else [upper]
            for end in ends:
                if isinstance(end, ast.BinOp) and isinstance(end.op, ast.Add):
                    self.word_given(end.right, what)

    # Ranges, partitions ------------------------------------------------------------

    def known_int(self, node):
        """An int the build knows (a literal, a module constant, a plan value), or
        None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if self.is_class_constant(node):
            return self.class_constant(node)
        if isinstance(node, ast.Name) and node.id not in self.scope:
            value = self.plan.get(node.id, self.names.get(node.id))
            return value if isinstance(value, int) else None
        return None

    def is_word(self, node):
        """Whether `node` is a name or attribute the handler holds as an int word."""
        if isinstance(node, ast.Name):
            value = self.scope.get(node.id)
            return value is not None and value.kind in ("int", "word")
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (
                getattr(self.scope.get(node.value.id), "kind", None)
                in ("entry", "entries")
                or node.value.id == "self"
            )
        )

    # Tiles -------------------------------------------------------------------------

    def field_sides(self, node, field_):
        """A field of tiles' rows and columns, its symbols the plan's."""
        sides = []
        for side in self.compiled.fields[field_].shape[:2]:
            if isinstance(side, str):
                if not isinstance(self.plan.get(side), int):
                    refuse(
                        self.where,
                        node,
                        f"self.{field_}'s {side} is the plan's, and the plan has none",
                        f"make the program with {spelling('plan')}={{{side!r}: n}}",
                    )
                side = self.plan[side]
            sides.append(side)
        return tuple(sides)

    # Streams -----------------------------------------------------------------------

    def loop(self, node):
        if node.orelse:
            refuse(self.where, node, "a loop with an else", "write the else after it")
        iterator = node.iter
        self.held_not_iterated(node, iterator)
        if self_call(iterator, "arrive_messages"):
            return messages.arrive_loop(self, node)
        if (
            isinstance(iterator, ast.Name)
            and self.scope.get(iterator.id, Value("")).kind == "arrived"
        ):
            return broadcasts.block_loop(self, node, self.scope[iterator.id])
        if (
            isinstance(iterator, ast.Name)
            and self.scope.get(iterator.id, Value("")).kind == "drained"
        ):
            return drained.drain_loop(self, node, iterator.id)
        if self.construct(iterator) == "stream":
            return page_streams.read_loop(self, node, iterator)
        streams = self.zipped(node)
        if streams is not None:
            return page_streams.read_loop(self, node, None, streams)
        if entry_tables.entries_call(
            self, iterator
        ) is not None or entry_tables.entries_named(self, iterator):
            return entry_tables.entries_loop(self, node)
        if (
            isinstance(iterator, ast.Call)
            and isinstance(iterator.func, ast.Name)
            and iterator.func.id == "range"
        ):
            return self.range_loop(node, iterator)
        refuse(
            self.where,
            node,
            "a loop of level 1 is over a stream or a range",
            f"write `for tile in km.{spelling('stream')}(...)` or "
            "`for i in range(n)`",
        )

    def range_loop(self, node, call):
        if not isinstance(node.target, ast.Name):
            refuse(self.where, node, "one loop name", "write for i in range(n)")
        name = node.target.id
        check_name(self.where, node, name)
        if not 1 <= len(call.args) <= 2:
            refuse(self.where, node, "range(n) or range(first, n)", "give one or two")
        first = self.integer(call.args[0]) if len(call.args) == 2 else "0"
        end = self.integer(call.args[-1])
        uses = set().union(*(scope.uses(self, arg) for arg in call.args))
        self.scope[name] = Value("int", name)
        loop = {"name": name, "first": first, "end": end, "casts": []}
        loop["body"] = node.body  # a running total's line stands in it (carries)
        self.loops.append(loop)
        before = set(self.scope)
        body_m, body_c = self.statements(node.body)
        body_c += loop.pop("frees", [])
        body_c += dst_tiles.carried_back(self, loop)
        scope.scoped(self, "loop", node, before)
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
        compute += loop.pop("after", [])  # a sum held across the loop, freed
        for broadcast in loop["casts"]:
            m, c = broadcasts.broadcast_end(self, broadcast, end, uses)
            movement += m
            compute += c
        m, c = broadcasts.counted_ends(self, loop)
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
            refuse(
                self.where,
                node,
                "streams read in step name one tile each",
                f"write `for a, b in {spelling('zip')}(km.{spelling('stream')}(...), "
                f"km.{spelling('stream')}(...))`",
            )
        return [(name.id, call) for name, call in zip(target.elts, iterator.args)]

    def call_statement(self, node, call):
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == spelling("push"):
            return tile_functions.push(self, node, call)
        if isinstance(func, ast.Attribute) and func.attr == spelling("write"):
            return self.write(node, call)
        if isinstance(func, ast.Attribute) and func.attr == spelling("put"):
            return puts.put_line(self, node, call)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("stream_end")
            and isinstance(func.value, ast.Name)
            and getattr(self.scope.get(func.value.id), "kind", None) == "drained"
        ):
            return drained.drained_end(self, node, func.value.id)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("fill")
            and scope.field_name(self, func.value)
        ):
            return field_fills.fill(
                self, node, call, scope.field_name(self, func.value)
            )
        if (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("zero")
            and scope.field_name(self, func.value)
        ):
            return field_fills.zero(
                self, node, call, scope.field_name(self, func.value)
            )
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
            self.field_use(func.value)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == spelling("drain")
            and self.construct(func.value) == "stream"
        ):
            return page_streams.drain(self, node, call)
        for kind in ("send", "forward", "consume"):
            if self_call(call, kind):
                return messages.message_call(self, node, call, kind)
        if self_call(call, "direct_message"):
            messages.direct_unnamed(self, node)
        construct = self.construct(call)
        if construct == "cxx":
            return bodies.inline(self, node, call)
        if construct == "reduce":
            return tile_reduce.reduce(self, node, None, call)
        if construct == "all_reduce":
            return tile_reduce.reduce(self, node, None, call, everywhere=True)
        if construct == "result":
            refuse(
                self.where,
                node,
                "a command's result word is the handler's return",
                "write `return word` last under `if self.rank == self.leader:`",
            )
        refuse(
            self.where,
            node,
            "a call level 1 does not know",
            "write a move, or an inline body",
        )

    # Tile functions ----------------------------------------------------------------

    def write(self, node, call):
        target = call.func.value
        if scope.field_name(self, target) is not None:
            return self.write_field(node, call, scope.field_name(self, target))
        if isinstance(target, ast.Subscript) and scope.field_name(self, target.value):
            return self.write_row(
                node, call, scope.field_name(self, target.value), target.slice
            )
        if not ranges.is_range(self, target):
            refuse(
                self.where,
                node,
                "write to a page range",
                "write tensor.pages(first, 1).write(tile)",
            )
        pages = ranges.range_of(self, target)
        handler_buffers.pages_only(self, node, pages)
        summed = call.args[0] if len(call.args) == 1 else None
        if scope.field_name(self, summed) in self.summed:
            return self.write_sums(node, pages, scope.field_name(self, summed))
        if (
            tile_lookup.tile_function_of(self, summed) is not None
            and summed.args
            and all(scope.field_name(self, arg) in self.summed for arg in summed.args)
        ):
            return self.write_function_of_sums(node, pages, summed)
        if tile_lookup.tile_function_of(self, summed) is not None and self.dst_pair_of(
            summed
        ):
            return self.write_function_of_pair(node, call, summed)
        if tile_lookup.tile_function_of(self, summed) is not None and self.dst_sums_of(
            summed
        ):
            return self.write_function_of_dst(node, call, summed)
        if tile_lookup.tile_function_of(
            self, summed
        ) is not None and tile_steps.step_of_dst(self, summed):
            return tile_steps.write_step_of_dst(self, node, call, summed)
        if scope.field_name(self, summed) in self.stepped:
            return tile_steps.write_state(
                self, node, pages, scope.field_name(self, summed)
            )
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            refuse(self.where, node, "write one tile", "write pages.write(total)")
        name = call.args[0].id
        value = self.scope.get(name)
        if value is not None and value.kind == "word":
            return page_words.write_word(self, node, pages, value)
        dst_tiles.not_paired(self, node, name)
        carried = dst_tiles.carries(self, node, value)
        held_sum = dst_tiles.sum_held(self, value)
        if (
            value is not None
            and value.kind in ("tile", "reduced")
            and not value.popped
            and not carried
            and not (held_sum and not value.summed)
        ):
            dst_tiles.same_loops(
                self, node, name, value, "this write", *dst_tiles.once_text(self, value)
            )
        if value is not None and value.kind == "reduced" and value.summed:
            if not self.in_leader or value.popped:
                refuse(
                    self.where,
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
            refuse(
                self.where,
                node,
                f"{name} is not a tile of compute (or is written already)",
                "write a tile made by km.tile.zeros and added to, once",
            )
        width = value.width
        if pages.rect is not None:
            refuse(
                self.where,
                node,
                "a page write of a rectangle is one row of it",
                f"write {pages.tensor}.{spelling('tiles')}[r:r + 1, c0:c1].write(...)",
            )
        if pages.count != str(width):
            refuse(
                self.where,
                node,
                f"{name} is {width} tile(s), {width} page(s)",
                f"write pages(first, {width})",
            )
        self.compiled.includes.add("pack")
        compute = []
        converted = False
        if value.kind == "tile":
            dtype, page_bytes = handler_buffers.tensor_format(self, node, pages.tensor)
            buffer = handler_buffers.out_buffer(
                self,
                dtype,
                page_bytes,
                f"line {node.lineno}: {line_text(self.where, node)}",
                width,
            )
            if carried:
                compute.append(
                    Line(dst_tiles.carried_write(self, node, value, buffer), {"km_ok"})
                )
                value.buffer = buffer
                wait(self.where, self.compiled.waits, node, "compute", WAIT_OUTPUT)
                wait(self.where, self.compiled.waits, node, "compute", WAIT_INPUT)
                movement, _ = self.write_moved(node, pages, value, buffer, width)
                return movement, compute
            compute.append(Line(pack(value.dst, buffer.name, width), {"km_ok"}))
            if getattr(value, "state", None) is not None:  # started from a field
                tile_steps.state_packed(self, compute, *value.state)
            tile_steps.row_given_back(self, compute, value)
            self.dst_live = None
            value.kind = "written"
            value.buffer = buffer
            wait(self.where, self.compiled.waits, node, "compute", WAIT_OUTPUT)
        else:
            if not self.in_leader:
                refuse(
                    self.where,
                    node,
                    f"{name} is the reduce's sum, which only the leader holds",
                    f"write it under `if self.{spelling('rank')} == "
                    f"self.{spelling('leader')}:`",
                )
            buffer = value.buffer
            self.leader_writes.add(name)
            dtype, page_bytes = handler_buffers.tensor_format(self, node, pages.tensor)
            if held_sum:
                buffer = handler_buffers.out_buffer(
                    self,
                    dtype,
                    page_bytes,
                    f"line {node.lineno}: {line_text(self.where, node)}",
                    width,
                )
                compute.append(dst_tiles.held_sum_copy(self, value, buffer, width))
                converted = True
            elif dtype != buffer.dtype:  # the sum packed again in the tensor's format
                source = buffer
                buffer = handler_buffers.out_buffer(
                    self,
                    dtype,
                    page_bytes,
                    f"line {node.lineno}: {line_text(self.where, node)}",
                    width,
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
        return self.write_moved(node, pages, value, buffer, width, converted, compute)

    def write_moved(
        self, node, pages, value, buffer, width, converted=False, compute=()
    ):
        """Mover B's page write of `value`'s `width` pages from `buffer`, and
        `compute`'s lines with it."""
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
            opening = (
                handler_buffers.page_wait(self, value, width)[:-3]
                + f" ||\n    !{write}) {{"
            )
        text = (
            f"{opening}\n"
            "    return false;\n"
            "}\n"
            "noc_async_writes_flushed();\n"
            f"cb_pop_front({buffer.name}, {width});"
        )
        if value.cxx not in self.carried:  # a running total stays live
            value.waited = value.popped = True
        handler_buffers.moved(self, node, pages.tensor, "page write")
        movement = [Line(text, range_uses(pages, self), mover="B")]
        self.compiled.includes.add("ask")
        wait(self.where, self.compiled.waits, node, "mover B", WAIT_DATA)
        return movement, list(compute)

    def write_field(self, node, call, field_):
        """`self.f.write(acc)`, acc DST sums (or `f(acc)`, `f(acc, self.s[r])` a
        function of them): acc's tiles packed into the field's pages for products
        that take it as their left side. A write frees the last one's pages (their
        readers are done), the handler's end frees the last."""
        written = call.args[0] if len(call.args) == 1 else None
        if tile_lookup.tile_function_of(self, written) is not None and self.row_written(
            written
        ):
            return self.write_row_value(node, field_, written)
        if tile_lookup.tile_function_of(self, written) is not None and self.dst_sums_of(
            written
        ):
            return self.write_function_of_dst(node, call, written)
        if tile_lookup.tile_function_of(
            self, written
        ) is not None and tile_steps.step_of_dst(self, written):
            return tile_steps.write_step_of_dst(self, node, call, written)
        value = isinstance(written, ast.Name) and self.scope.get(written.id)
        if not value or value.kind != "tile" or value.popped:
            refuse(
                self.where,
                node,
                f"self.{field_} takes a tile of compute's DST sums (or a tile function "
                "of them), once",
                f"write self.{field_}.write(acc), acc made by km.tile.zeros and added "
                "to",
            )
        dst_tiles.same_loops(
            self,
            node,
            written.id,
            value,
            "this write",
            *dst_tiles.once_text(self, value),
        )
        stored = self.compiled.fields[field_]
        pages = math.prod(stored.shape[:-2])
        fillers = self.compiled.field_users.get(field_, ([], []))[0]
        if (
            fillers
            or field_ in self.compiled.own
            or field_ in self.summed
            or field_ in self.stepped
            or len(stored.shape) < 3
        ):
            refuse(
                self.where,
                node,
                f"self.{field_} is filled, summed into or a body's: compute writes "
                "a field of its own",
                f"declare a field for the write: km.L1[km.bf16, {value.width}, 32, 32]",
            )
        if pages != value.width:
            refuse(
                self.where,
                node,
                f"self.{field_} holds {pages} tile(s) and {written.id} is "
                f"{value.width}",
                f"declare self.{field_}: km.L1[km.bf16, {value.width}, 32, 32]",
            )
        field_fills.producer(self, node, field_, "product")
        if getattr(value, "row_of", None) is not None:  # sums started from a mask
            self.masked.add(field_)
        else:
            self.masked.discard(field_)
        buffer, held = f"kField{camel(field_)}", f"km_packed_{field_}"
        self.compiled.summed_fields.add(buffer)  # pack writes it: one of its formats
        if field_ not in self.packed:
            self.packed[field_] = held
            self.opening.append(
                Line(
                    f"bool {held} = false;  // self.{field_}'s pages pushed, not freed"
                )
            )
            self.closing.append(
                Line(f"if ({held}) {{\n    cb_pop_front({buffer}, {buffer}Pages);\n}}")
            )
        self.compiled.includes.add("pack")
        wait(self.where, self.compiled.waits, node, "compute", WAIT_OUTPUT)
        value.kind, value.popped = "written", True
        self.dst_live = None
        state = getattr(value, "state", None)
        text = (
            f"if ({held}) {{  // the last write's pages: their products are done\n"
            f"    cb_pop_front({buffer}, {buffer}Pages);\n"
            "}\n"
            f"{pack(value.dst, buffer, value.width)}\n"
            f"{held} = km_ok;"
        )
        compute = [Line(text, {"km_ok"})]
        if state is not None:
            tile_steps.state_packed(self, compute, *state)
        tile_steps.row_given_back(self, compute, value)
        return [], compute

    def row_written(self, call):
        """Whether `call` is a tile function's call on a row of more than 2 tiles."""
        first = call.args[0] if call.args else None
        value = isinstance(first, ast.Name) and self.scope.get(first.id)
        return bool(value) and value.kind == "streamed" and (value.width or 1) > 2

    def write_row_value(self, node, field_, written):
        """`self.part.write(f(row))`, f's result a per-row value (`s.col()`): the
        row's sum passes, then the value's tile packed into the field, one tile of
        compute's own, for a later loop's `.col()` (a write frees the last one)."""
        stored = self.compiled.fields[field_]
        if (
            self.compiled.field_users.get(field_, ([], []))[0]
            or field_ in self.compiled.own
            or math.prod(stored.shape[:-2]) != 1
            or stored.format.dtype != "float32"
        ):
            refuse(
                self.where,
                node,
                f"self.{field_} takes a row's per-row value: one float32 tile of "
                "compute's own",
                f"declare self.{field_}: km.L1[km.fp32, 1, 32, 32], not filled",
            )
        field_fills.producer(self, node, field_, "product")
        buffer, held = f"kField{camel(field_)}", f"km_packed_{field_}"
        self.compiled.summed_fields.add(buffer)
        if field_ not in self.packed:
            self.packed[field_] = held
            self.opening.append(Line(f"bool {held} = false;  // self.{field_} pushed"))
            self.closing.append(
                Line(f"if ({held}) {{\n    cb_pop_front({buffer}, {buffer}Pages);\n}}")
            )
        self.compiled.includes.add("pack")
        wait(self.where, self.compiled.waits, node, "compute", WAIT_OUTPUT)
        self.loop_states[-1]["value"] = (buffer, held)
        _, made = tile_functions.tile_call(self, node, written, None)
        self.scope.pop("<tile function>", None)
        return [], made

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
            refuse(
                self.where,
                node,
                f"self.{field_} is {rows} x {columns} tiles; write a rectangle of them "
                "(a side a word: the build must know it is at most the field's)",
                f"write {pages.tensor}.{spelling('tiles')}[r:r + {rows}, c:c + "
                f"{columns}].write(self.{field_}), or c:{spelling('min')}(c + "
                f"{columns}, e)",
            )
        written_rows, written_columns = sides
        dtype, page_bytes = handler_buffers.tensor_format(self, node, pages.tensor)
        name = f"kOutRows{columns}"  # its rows' pages stay whole: one width a buffer
        held = [b.pages[0] for b in self.compiled.buffers if b.name == name]
        out = handler_buffers.out_buffer(
            self,
            dtype,
            page_bytes,
            f"line {node.lineno}: {line_text(self.where, node)}",
            columns,
            name,
        )
        # a row packed while the one before is written (one row: the row alone)
        out.pages = (max(held + [columns * min(rows, 2)]),)
        buffer, sums = f"kField{camel(field_)}", self.summed[field_]
        room, out_room = sums_room(field_), f"km_room{node.lineno}"
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
            f"    if ({room}) {{  // pack's: its pages held\n"
            f"        cb_push_back({buffer}, {buffer}Pages);\n"
            "    }\n"
            f"    km_ok = compute.wait_front({buffer}, {buffer}Pages);\n"
            "}\n"
            f"if (km_ok && {sums} > 0) {{\n"
        )
        compute = (
            f"// self.{field_}'s sums, packed again in {dtype}\n"
            f"{taken}"
            f"    KM_UNPACK_FORMAT_A({buffer});\n"
            f"    copy_init({buffer});\n"
            f"    bool {out_room} = true;  // pack's alone: it guards the packs only\n"
            f"    for (uint32_t {tile} = 0; km_ok && "
            f"{tile} < {rows * columns}; {tile} += "
            f"{step}) {{\n"
            "        tile_regs_acquire();\n"
            f"{copies}\n"
            "        tile_regs_commit();\n"
            "        tile_regs_wait();\n"
            f"        {out_room} = {out_room} && compute.reserve_back({out.name}, "
            f"{step});\n"
            f"        if ({out_room}) {{\n"
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
        handler_buffers.moved(self, node, pages.tensor, "page write")
        wait(self.where, self.compiled.waits, node, "compute", WAIT_OUTPUT)
        wait(self.where, self.compiled.waits, node, "mover B", WAIT_DATA)
        return [Line(movement, range_uses(pages, self), mover="B")], [
            Line(compute, {"km_ok"})
        ]

    def dst_sums_of(self, call):
        """The name of the DST sums a tile function's call takes (`f(acc)`, acc a
        km.tile.zeros sum compute holds), or None."""
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            return None
        value = self.scope.get(call.args[0].id)
        if value is None or value.kind != "tile" or value.batch is not None:
            return None
        return call.args[0].id

    def write_function_of_dst(self, node, write, call):
        """`pages.write(f(acc))`, acc DST sums (a product's over K, added in the
        loop): f's body run on each of acc's DST tiles in place on the math CPU, then
        acc written as `pages.write(acc)` writes it."""
        function = tile_lookup.tile_function_of(self, call)
        name = self.dst_sums_of(call)
        value = self.scope[name]
        unit, cxx = tile_functions.tile_unit(self, node, function, ["float32"])
        if unit.products:
            refuse(
                self.where,
                node,
                f"{function.name} has a matrix product (`@`), and its operand here is "
                f"{name}, sums in DST that the matrix unit does not read",
                f"add the product in the loop and write {function.name}({name}) of "
                "its sums",
            )
        calls = "\n".join(
            f"MATH((km_tile::{cxx}({value.dst + k})));" for k in range(value.width)
        )
        body = f"MATH((km_tile::{cxx}_init()));  // {function.name}'s setup\n{calls}"
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, line_text(self.where, node), function, cxx)
        )
        write.args[0] = ast.Name(id=name, ctx=ast.Load())
        movement, compute = self.write(node, write)
        return movement, [Line(body), *compute]

    def dst_pair_of(self, call):
        """The two DST sums held together that a tile function's call takes (`f(a,
        b)`, a and b the pair of a second zeros, each once), or None."""
        if (
            self.dst_pair is None
            or len(call.args) != 2
            or call.keywords
            or not all(isinstance(arg, ast.Name) for arg in call.args)
        ):
            return None
        names = tuple(arg.id for arg in call.args)
        return names if set(names) == set(self.dst_pair) else None

    def write_function_of_pair(self, node, write, call):
        """`pages.write(f(a, b))`, a and b DST sums held together (a second zeros):
        f's body run on DST tile t of the first of them with the other operand read
        its distance apart, for each of their tiles, then the first's tiles written as
        `pages.write(acc)` writes them (DST released)."""
        function = tile_lookup.tile_function_of(self, call)
        values = [self.scope[arg.id] for arg in call.args]
        if values[0].width != values[1].width:
            refuse(
                self.where,
                node,
                f"{function.name} takes {call.args[0].id} ({values[0].width} tiles) "
                f"and {call.args[1].id} ({values[1].width} tiles): tile t of each",
                "make the two sums one width",
            )
        for value in values:
            dst_tiles.same_loops(
                self,
                node,
                value.cxx,
                value,
                "this write",
                *dst_tiles.once_text(self, value),
            )
        base = min(values, key=lambda v: v.dst)
        parameters = list(inspect.signature(function.function).parameters)
        tiles = {p: v.dst - base.dst for p, v in zip(parameters, values, strict=False)}
        unit, cxx = tile_functions.tile_unit(
            self, node, function, ["float32", "float32"], tiles=tiles
        )
        if unit.products:
            refuse(
                self.where,
                node,
                f"{function.name} has a matrix product (`@`), and its operands here "
                "are sums in DST that the matrix unit does not read",
                f"add the products in the loops and write {function.name} of the sums",
            )
        calls = "\n".join(
            f"MATH((km_tile::{cxx}({base.dst + k})));" for k in range(base.width)
        )
        body = f"MATH((km_tile::{cxx}_init()));  // {function.name}'s setup\n{calls}"
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, line_text(self.where, node), function, cxx)
        )
        other = values[1] if base is values[0] else values[0]
        other.kind = "written"  # its sums end with the write of the first's
        self.dst_pair = None
        self.dst_live = base.cxx
        write.args[0] = ast.Name(id=base.cxx, ctx=ast.Load())
        movement, compute = self.write(node, write)
        return movement, [Line(body), *compute]

    def write_function_of_sums(self, node, pages, call):
        """`t.tiles[r0:r1, c0:c1].write(f(self.h, self.g))`, f a tile function of
        fields of sums of one shape: each tile's fields copied into DST k (the unit's
        operand order), f's body, DST 0 packed in t's format and written a row at a
        time by mover B as `write(self.h)` is; the fields freed, the next sums start."""
        function = tile_lookup.tile_function_of(self, call)
        fields = [scope.field_name(self, arg) for arg in call.args]
        sides = {self.field_sides(node, f) for f in fields}
        if len(sides) > 1 or len(set(fields)) != len(fields):
            refuse(
                self.where,
                node,
                f"{function.name} takes fields of sums of one shape, each once, and "
                f"is given {', '.join('self.' + f for f in fields)}",
                "give it fields of one shape",
            )
        ((rows, columns),) = sides
        formats = [self.compiled.fields[f].format.dtype for f in fields]
        why = tile_format_refusal(formats)
        if why is not None:
            refuse(
                self.where, node, f"{function.name}: {why}", "give it fields of those"
            )
        unit, cxx = tile_functions.tile_unit(self, node, function, formats)
        if unit.products:
            refuse(
                self.where,
                node,
                f"{function.name} has a matrix product (`@`), and its operands here "
                "are fields of sums in L1, which the unit reads from DST",
                f"add the product into a field (self.f += self.p(a, b)) and write "
                f"{function.name} of the fields",
            )
        rect = pages.rect
        bounds = pages.bounds or (None, None)
        written = rect[:2] if rect is not None else ("1", pages.count)
        fits = all(
            side_fits(side, bound, most)
            for side, bound, most in zip(written, bounds, (rows, columns), strict=True)
        )
        if not fits:
            refuse(
                self.where,
                node,
                f"the fields are {rows} x {columns} tiles; write a rectangle of them",
                f"write {pages.tensor}.{spelling('tiles')}[r:r + {rows}, c:c + "
                f"{columns}].write({function.name}(...))",
            )
        dtype, page_bytes = handler_buffers.tensor_format(self, node, pages.tensor)
        name = f"kOutRows{columns}"
        held = [b.pages[0] for b in self.compiled.buffers if b.name == name]
        out = handler_buffers.out_buffer(
            self,
            dtype,
            page_bytes,
            f"line {node.lineno}: {line_text(self.where, node)}",
            columns,
            name,
        )
        out.pages = (max(held + [columns * min(rows, 2)]),)
        parameters = list(inspect.signature(function.function).parameters)
        buffers = {f: f"kField{camel(f)}" for f in fields}
        counts = [self.summed[f] for f in fields]
        made = " && ".join(f"{sums} > 0" for sums in counts)
        tile, out_room = f"km_tile{node.lineno}", f"km_room{node.lineno}"
        lines = [
            f"// {line_text(self.where, node)}: {function.name} of the fields' sums"
        ]
        for f in fields:
            buffer, sums = buffers[f], self.summed[f]
            lines += [
                f"if (km_ok && {sums} > 0) {{",
                f"    if ({sums_room(f)}) {{  // pack's: its pages held",
                f"        cb_push_back({buffer}, {buffer}Pages);",
                "    }",
                f"    km_ok = compute.wait_front({buffer}, {buffer}Pages);",
                "}",
            ]
        first = buffers[fields[0]]
        lines += [
            f"if (km_ok && {made}) {{",
            f"    KM_UNPACK_FORMAT_A({first});",
            f"    copy_init({first});",
            f"    MATH((km_tile::{cxx}_init()));  // {function.name}'s setup",
            f"    bool {out_room} = true;  // pack's alone: it guards the packs only",
            f"    for (uint32_t {tile} = 0; km_ok && {tile} < {rows * columns}; "
            f"{tile}++) {{",
            "        tile_regs_acquire();",
        ]
        for dst, operand in enumerate(unit.operands):
            buffer = buffers[fields[parameters.index(operand)]]
            lines.append(f"        copy_tile({buffer}, {tile}, {dst});")
        lines += [
            f"        MATH((km_tile::{cxx}(0)));",
            "        tile_regs_commit();",
            "        tile_regs_wait();",
            f"        {out_room} = {out_room} && compute.reserve_back({out.name}, 1);",
            f"        if ({out_room}) {{",
            f"            KM_PACK_FORMAT({out.name});",
            f"            pack_tile(0, {out.name});",
            f"            cb_push_back({out.name}, 1);",
            "        }",
            "        tile_regs_release();",
            "    }",
            *[f"    cb_pop_front({b}, {b}Pages);" for b in buffers.values()],
            "}",
            *[f"{sums} = 0;" for sums in counts],
        ]
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        row = f"km_row{node.lineno}"
        row_step = rect[2] if rect is not None else columns
        whole = str(written[0]) == str(rows)
        part = "" if whole else f"{row} < ({written[0]}) &&\n    "
        movement = [
            f"if ({made}) {{",
            f"    for (uint32_t {row} = 0; {row} < {rows}; {row}++) {{",
            f"        if (!movement.wait_front({out.name}, {columns}) ||",
            f"            {'(' if part else ''}{part}!km::write_pages(movement, "
            f"command, {tensor}, {pages.first} + {row} * {row_step}, {written[1]}, "
            f"get_read_ptr({out.name})){')' if part else ''}) {{",
            "            return false;",
            "        }",
            "        noc_async_writes_flushed();",
            f"        cb_pop_front({out.name}, {columns});",
            "    }",
            "}",
            *[f"{sums} = 0;" for sums in counts],
        ]
        self.compiled.tile_calls.append(
            (self.method.name, node.lineno, line_text(self.where, node), function, cxx)
        )
        self.compiled.includes |= {"pack", "ask"}
        handler_buffers.moved(self, node, pages.tensor, "page write")
        wait(self.where, self.compiled.waits, node, "compute", WAIT_OUTPUT)
        wait(self.where, self.compiled.waits, node, "mover B", WAIT_DATA)
        return [Line("\n".join(movement), range_uses(pages, self), mover="B")], [
            Line("\n".join(lines), {"km_ok"})
        ]

    def word_sum(self, node, value):
        """`word += x` (or another operator) on a mover's word: its mover's line; x
        reads ints and that mover's words."""
        op = OPERATORS.get(type(node.op))
        if op is None:
            refuse(
                self.where, node, "an operator level 1 has no C++ for", "use + - * etc."
            )
        for item in ast.walk(node.value):
            other = isinstance(item, ast.Name) and self.scope.get(item.id)
            if other and other.kind == "word" and other.mover != value.mover:
                refuse(
                    self.where,
                    node,
                    f"{item.id} is another CPU's word",
                    "add words of one mover",
                )
        self.allow_words = True
        text = f"{value.cxx} {op}= {self.integer(node.value)};"
        self.allow_words = False
        uses = scope.uses(self, node.value) | {value.cxx}
        return [Line(text, uses, mover=value.mover)], []

    def branch(self, node):
        if self_call(node.test, "seeding"):
            return messages.seeding_branch(self, node)
        broadcast = broadcasts.broadcast_branch(self, node)
        if broadcast is not None:
            return broadcast
        condition = self.integer(node.test)
        uses = scope.uses(self, node.test)
        live = self.dst_live
        leader, self.in_leader = self.in_leader, self.in_leader or self.leader_test(
            node
        )
        if self.leader_test(node):
            self.leader_condition = (condition, uses)  # a held sum's free (sum_held)
        before = set(self.scope)
        then_m, then_c = self.statements(node.body)
        scope.scoped(self, "if", node, before)
        self.in_leader = leader
        else_m, else_c = self.statements(node.orelse)
        scope.scoped(self, "if", node, before)
        if self.dst_live != live:
            refuse(
                self.where,
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

    def mover_integer(self, node, value, mover):
        """The C++ of an int of a line on `mover`: a word of that mover by its name,
        else an int of every CPU; a word of another CPU refused."""
        self.mover_word(node, value, mover)
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            page = self.scope.get(value.value.id)
            if page is not None and page.kind == "page":
                return f"{value.value.id}_words[{self.integer(value.slice)}]"
        if (
            isinstance(value, ast.Name)
            and getattr(self.scope.get(value.id), "kind", None) == "word"
        ):
            return value.id
        return self.integer(value)

    def mover_word(self, node, value, mover):
        """Refuses a word of another CPU where a line runs on `mover`."""
        for item in ast.walk(value):
            if not isinstance(item, ast.Name) or item.id not in self.scope:
                continue
            known = self.scope[item.id]
            if known.kind in ("word", "page") and known.mover != mover:
                where = {"A": "mover A", "B": "mover B"}.get(known.mover, known.mover)
                refuse(
                    self.where,
                    node,
                    f"{item.id} is a word of {where}, and this line runs on mover "
                    f"{mover}",
                    "give it a word of that mover, or an int of every CPU",
                )

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

    def field_use(self, node):
        """Refuses any other use of a field: `@`, slices and `.columns` of a field
        are tile expressions of level 2."""
        if scope.field_name(self, node):
            refuse(
                self.where,
                node,
                f"self.{node.attr} is a field: a line fills it from a stream and a "
                "km.cxx body takes it; `@`, slices and `.columns` of a field are "
                "level 2",
                f"write self.{node.attr}.fill(km.Stream(...)) and "
                f"km.cxx(..., takes=[self.{node.attr}])",
            )

    # Reduce ------------------------------------------------------------------------

    # Broadcasts --------------------------------------------------------------------

    # Inline bodies and results -----------------------------------------------------

    def held_block_ends(self, body):
        """compute's frees of the blocks products took (taken_by_field_products),
        cast in `body`, at its end."""
        ends = self.held_blocks.pop(id(body), [])
        nodes = [
            Line(f"if (km_ok) {{\n    km::broadcast_done({w}, {w}Pages);\n}}")
            for _, w in ends
        ]
        if not nodes:
            return []
        return [Group("the end of the cast's lines: the products' blocks freed", nodes)]

    def write_row(self, node, call, field_, row):
        """`self.o[j].write(acc)`, acc started from a row of self.o: its tiles packed
        into row j in place, the token pushed, then the state packed (acc's)."""
        written = call.args[0] if len(call.args) == 1 else None
        value = isinstance(written, ast.Name) and self.scope.get(written.id)
        if not value or value.kind != "tile" or (value.row_of or (None,))[0] != field_:
            refuse(
                self.where,
                node,
                f"a row of self.{field_} takes sums started from one of its rows "
                "(their token orders its reads and writes)",
                f"write acc = f(self.{field_}[i], self.s[r]) above, then "
                f"self.{field_}[j].write(acc)",
            )
        dst_tiles.same_loops(
            self,
            node,
            written.id,
            value,
            "this write",
            *dst_tiles.once_text(self, value),
        )
        _, source, token_o = value.row_of
        line = self.integer(row)
        items = self.compiled.fields[field_].shape[0]
        width = value.width
        bound = (
            ""
            if line.rstrip("u").isdigit()
            else (
                f"if (!(({line}) < {items})) {{  // self.{field_}'s rows\n"
                "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
                "    return;\n}\n"
            )
        )
        if line.rstrip("u").isdigit() and not int(line.rstrip("u")) < items:
            refuse(
                self.where,
                node,
                f"self.{field_} holds {items} rows, 0 .. {items - 1}",
                f"index it below {items}",
            )
        packs = "".join(
            f"    pack_tile<true>({value.dst + t}, {source}, ({line}) * {width} + "
            f"{t});\n"
            for t in range(width)
        )
        text = (
            f"{bound}"
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"if (km_ok && compute.reserve_back({token_o}, 1)) {{  // pack's alone\n"
            f"    KM_PACK_FORMAT({source});\n"
            f"{packs}"
            f"    cb_push_back({token_o}, 1);\n"
            "}\n"
            "tile_regs_release();"
        )
        self.compiled.includes.add("pack")
        value.kind, value.popped = "written", True
        self.dst_live = None
        compute = [Line(text, {"km_ok"} | scope.uses(self, row))]
        tile_steps.state_packed(self, compute, *value.state)
        return [], compute

    def take_write(self, node, name, value, bound, before, after):
        """A write stream's page for a body on pack: in a stream loop the loop's
        page (reserved with the block, pushed after it); outside one, the stream's
        one block, reserved, pushed and moved at this line. The page's L1 address,
        0 when compute gave up."""
        if value.pushed:
            refuse(
                self.where,
                node,
                f"{name} is fed above (a push or another body)",
                "open a write stream for each",
            )
        value.pushed = True
        buffer = value.buffer.name
        if self.stream_loops:
            state = self.loop_states[-1]
            if value.buffer.block != state["block"]:
                refuse(
                    self.where,
                    node,
                    f"{name}'s block is {value.buffer.block} page(s), the loop's "
                    f"{state['block']}",
                    "give the write stream the loop's block=",
                )
            page_streams.loop_fed(self, node, name, value)
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
            refuse(
                self.where,
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
        wait(self.where, self.compiled.waits, node, "compute", WAIT_OUTPUT)
        wait(self.where, self.compiled.waits, node, "mover B", WAIT_DATA)
        return set()

    def take_fed(self, node, name, value, bound, before, after):
        """A write stream made outside the loops around a body on compute or pack:
        one block of it each time the body runs (compute counts them), reserved
        before the body and pushed after it; the movers move the whole stream after
        the outermost of those loops (fed_moves)."""
        if value.move is not None and not value.table:
            refuse(
                self.where,
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
        wait(self.where, self.compiled.waits, node, "compute", WAIT_OUTPUT)
        wait(self.where, self.compiled.waits, node, "mover B", WAIT_DATA)
        return set()

    def fed_moves(self, node, loop, body_m):
        """The moves of the write streams a loop's bodies feed (take_fed), after the
        loop; refused when the loop has movers' lines too (each would wait on the
        other)."""
        moves = loop.get("moves", [])
        if moves and body_m:
            refuse(
                self.where,
                node,
                "a write stream fed a block a turn is moved after the loop, and this "
                "loop has movers' lines too: the movers would wait on each other",
                "feed it in a loop of compute's lines alone",
            )
        return list(moves)

    def write_keyword(self, call):
        """True when a stream's call has write= (a write stream's, or refused there)."""
        return any(k.arg == spelling("write_stream") for k in call.keywords)

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
            refuse(
                self.where,
                node,
                f"{iterator.id} is a read stream that bodies take, one block: it is "
                "not iterated too",
                f"iterate a stream of its own, `for tile in km.{spelling('stream')}"
                "(...)`, or take this one in a body",
            )

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
            refuse(
                self.where,
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
        return [
            Line(f"movement.result({text});", scope.uses(self, node.value), mover="B")
        ], []

    # Entries: a page's words as a table, on every CPU -------------------------------

    def entries_word(self, node, value):
        """The C++ of `e.word` (a word of an entry) or `table.count`."""
        cls = value.table["class"]
        if value.kind == "entries":
            if node.attr != spelling("entries_count"):
                refuse(
                    self.where,
                    node,
                    f"a table has its {spelling('entries_count')} and its entries",
                    f"write {node.value.id}.{spelling('entries_count')}, or loop "
                    f"over {node.value.id}",
                )
            return value.table["count"]
        if node.attr not in cls.words:
            refuse(
                self.where,
                node,
                f"{node.value.id}, a {cls.__name__}, has no word {node.attr}",
                f"use one of {', '.join(cls.words)}",
            )
        return f"km_{node.value.id}_{node.attr}"


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
