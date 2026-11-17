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
from komorebi.dsl.compile import layout
from komorebi.dsl.compile.casts.messages import arrive_of
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.dst.dst_code import (
    DST_HALF,
)
from komorebi.dsl.compile.handler import handler_compile
from komorebi.dsl.compile.records import (
    Buffer,
    Compiled,
    Refused,
)
from komorebi.dsl.compile.source import (
    source_of,
)
from komorebi.dsl.front import front
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Block, Line
from komorebi.program.several import COMPUTE_FLAGS

TILE_ELEMENTS = 32 * 32


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
    share_pairs=False,
):
    """Compiles each handler of `cls`; `plan` sizes, `tensors` an example of each tensor
    parameter, `l1` the bytes for buffers, `constants` their values, `handlers` the sets
    (handler_variants), `reserved` indices left to others, `islands` the session's (a
    tree's slots); `share_pairs`: handlers of other blocks share read pairs (layout)."""
    methods = handler_methods(cls)
    # resolved: the #line text is the same however the script was started
    source = str(Path(inspect.getsourcefile(methods[0].function)).resolve())
    compiled = Compiled(cls.__name__, source)
    compiled.cls = cls
    compiled.reserved, compiled.share_pairs = set(reserved), share_pairs
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
        compile_handler(compiled, method, names, plan, tensors)
    listed_words(compiled, methods)
    bind_fields(compiled, plan, l1)
    compiled.compute_options = class_compute(cls, compiled)
    compiled.startup = getattr(cls, "startup_given", None)
    return compiled


def compile_handler(compiled, method, names, plan, tensors):
    """One handler compiled into `compiled`; the values its compile returns set on its
    front.HandlerMethod: its command words, its rectangles' tile widths, and whether
    it returns a word (as far as the compile got when it refuses)."""
    frame = None
    try:
        frame = handler_compile.frame_of(compiled, method, names, plan, tensors)
        handler_compile.run(frame)
    finally:
        method.widths = frame.widths if frame is not None else {}
        if frame is not None and frame.words is not None:
            method.words = frame.words
        if frame is not None and frame.returns:
            method.returns = True


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
    """One class of a merged program (komorebi.dsl.merged): the class, its plan,
    its example tensors, its constants' values, its handlers [(merged number,
    method)], the prefix of its tensors that take an alias of their own."""

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
    # resolved: the #line text is the same however the script was started
    source = str(Path(inspect.getsourcefile(methods[0].function)).resolve())
    compiled = Compiled(name, source)
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
            compile_handler(
                compiled, method, handler_names(method), part.plan, part.tensors
            )
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
