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
from komorebi.dsl.compile import layout, shared
from komorebi.dsl.compile.casts.messages import arrive_of
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.dst.dst_code import (
    DST_HALF,
)
from komorebi.dsl.compile.handler import handler_compile, kernel_handler
from komorebi.dsl.compile.records import (
    Buffer,
    Compiled,
    Refused,
    Unfit,
)
from komorebi.dsl.compile.source import (
    file_of,
    source_of,
)
from komorebi.dsl.compile.tiles import wholes
from komorebi.dsl.front import front
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Block, Line
from komorebi.program.several import COMPUTE_FLAGS, IF_PRESENT, RVV

TILE_ELEMENTS = 32 * 32
RAW_ALIGN = 16  # a raw field's page: whole 16 B (the NoC's and L1's alignment)


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
    indices=64,
    step=1,
):
    """Compiles each handler of `cls`; `plan` sizes, `tensors` an example of each tensor
    parameter, `l1` the bytes for buffers, `constants` their values, `handlers` the sets
    (handler_variants), `reserved` indices left to others, `islands` the session's,
    `share_pairs` (layout), `indices` the chip's indices, `step` (compile.shared)."""
    methods = handler_methods(cls)
    # resolved: the #line text is the same however the script was started
    source = str(Path(inspect.getsourcefile(methods[0].function)).resolve())
    compiled = Compiled(cls.__name__, source)
    compiled.cls = cls
    compiled.reserved, compiled.share_pairs = set(reserved), share_pairs
    compiled.islands, compiled.indices = tuple(islands), indices
    compiled.step = step
    try:
        return compiled_class(compiled, cls, methods, plan, tensors, l1, constants,
                              handlers)  # fmt: skip
    except Unfit as unfit:
        unfit.compiled = compiled
        raise


def compiled_class(compiled, cls, methods, plan, tensors, l1, constants, handlers):
    """compile_class's work on `compiled`."""
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
    compiled.raw = {
        name for name in compiled.own if inspect.getattr_static(cls, name).raw
    }
    for name, value in list(compiled.fields.items()):
        if name in compiled.raw:
            raw_checked(compiled, name, value)
            continue
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
    whole_fields(compiled, methods)
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
    declared_pairs(compiled, class_pairs(compiled, cls, {}))
    kernel_handler.constants_kept(compiled, [(cls.__name__, compiled.class_constants)])
    bind_fields(compiled, plan, l1)
    compiled.compute_options = class_compute(cls, compiled)
    compiled.startup = getattr(cls, "startup_given", None)
    return compiled


def compile_handler(compiled, method, names, plan, tensors):
    """One handler compiled into `compiled`; the values its compile returns set on its
    front.HandlerMethod: its command words, its rectangles' tile widths, and whether
    it returns a word (as far as the compile got when it refuses)."""
    frame = None
    compiled.current = method.name  # the buffers of one call it names (their users)
    try:
        frame = handler_compile.frame_of(compiled, method, names, plan, tensors)
        handler_compile.run(frame)
    finally:
        compiled.current = None
        method.widths = frame.widths if frame is not None else {}
        method.bounds = frame.bounds if frame is not None else []
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
        if all(
            v.words == first.words
            and v.widths == first.widths
            and v.bounds == first.bounds
            for v in built
        ):
            method.words, method.widths = first.words, first.widths
            method.bounds = first.bounds
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


def compile_merged(name, parts, l1=None, indices=64, step=1):
    """Compiles the handlers of several classes as ONE program (`indices` the chip's
    buffer indices, `step` the way they share buffers): each handler reads its own
    class's constants, plan and tensors; a field of one name in two classes is one field
    when declared alike; returns the Compiled and the example tensors by their names."""
    methods = [method for part in parts for _, method in part.methods]
    # resolved: the #line text is the same however the script was started
    source = str(Path(inspect.getsourcefile(methods[0].function)).resolve())
    compiled = Compiled(name, source)
    compiled.indices, compiled.step = indices, step
    try:
        return merged_compiled(compiled, name, parts, l1, methods)
    except Unfit as unfit:
        unfit.compiled = compiled
        raise


def merged_compiled(compiled, name, parts, l1, methods):
    """compile_merged's work on `compiled`."""
    merged_fields(compiled, parts)
    compiled.field_users = field_users(methods, compiled.fields)
    for method in methods:
        if arrive_of(source_of(method.function)[0]):
            raise Refused(
                f"{name}: {method.owner.__name__}.{method.name} is a dynamic handler, "
                "whose message loop is its program's own kernels; not merged"
            )
    plan, tensors, order = {}, {}, []
    pairs, constants = {}, []  # the classes' km.Pair and constants' values
    for part in parts:
        for key, value in part.plan.items():
            if plan.setdefault(key, value) != value:
                raise Refused(
                    f"{name}: the plan's {key} is {plan[key]!r} in one class and "
                    f"{value!r} in {part.cls.__name__}; one value a merged program"
                )
        declared_constants(compiled, part.cls, part.constants)
        pairs = class_pairs(compiled, part.cls, pairs)
        if any(method.kernel is not None for _, method in part.methods):
            constants.append((part.cls.__name__, compiled.class_constants))
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
    declared_pairs(compiled, pairs)
    kernel_handler.constants_kept(compiled, constants)
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
            raw = isinstance(own, front.Own) and own.raw
            if field_name in owners and owners[field_name][1:] != (made, push, raw):
                raise Refused(
                    f"{compiled.name}: field {field_name} is {owners[field_name][1]} "
                    f"in {owners[field_name][0]} and {made} in {part.cls.__name__}; "
                    "a field of one name is one field: declare it alike, or rename "
                    "one"
                )
            owners[field_name] = (part.cls.__name__, made, push, raw)
            compiled.fields[field_name] = made
            compiled.field_files.setdefault(field_name, file_of(part.cls))
            if isinstance(own, front.Own):
                compiled.own[field_name] = own.push
            if raw:
                compiled.raw.add(field_name)
    for field_name, value in list(compiled.fields.items()):
        if field_name in compiled.raw:
            raw_checked(compiled, field_name, value)
            continue
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


def class_pairs(compiled, cls, found):
    """`found` with the class's km.Pair attributes, parents first, each (its class,
    format, this program's pages of each half, write); one name declared two ways in
    two classes of a merged program refused naming both."""
    own = {
        name: value
        for klass in reversed(cls.__mro__)
        for name, value in vars(klass).items()
        if isinstance(value, front.Pair)
    }
    found = dict(found)
    for name, pair in own.items():
        pages = pair.pages
        if isinstance(pages, front.Constant):
            try:
                pages = pages.value(compiled.class_constants)
            except KeyError:  # a symbol the build solves after the pairs
                pages = repr(pair.pages)
        if isinstance(pages, bool) or not isinstance(pages, int) or pages < 1:
            raise Refused(
                f"{cls.__name__}.{name}: km.{spelling('pair')}'s pages are "
                f"{pages!r}; give an int of 1 or more"
            )
        made = (cls.__name__, pair.format, pages, pair.write)
        if name in found and found[name][1:] != made[1:]:
            raise Refused(
                f"{compiled.name}: km.{spelling('pair')} {name} is "
                f"{pair_text(found[name])} in {found[name][0]} and {pair_text(made)} "
                f"in {cls.__name__}; a pair of one name is one pair: declare it "
                "alike, or rename one"
            )
        found.setdefault(name, made)
    return found


def pair_text(made):
    """A km.Pair as its declaration reads: format, pages, write."""
    _, format_, pages, write = made
    return f"({format_.dtype}, pages={pages}, write={write})"


def declared_pairs(compiled, pairs):
    """Each km.Pair a buffer pair of role read (write with write=True), indices by
    the stream pairs' rule, after every handler's streams took theirs."""
    for name, (_, format_, pages, write) in pairs.items():
        role = "write" if write else "read"
        what = (
            f"pair {name}: {format_.dtype}, {pages} pages a half; "
            f"km.{spelling('pair')}: streamed by hand kernels"
        )
        layout.declared_pair(
            compiled, name, role, format_.dtype, format_.tile_bytes, pages, what
        )


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
        if key == RVV and value == IF_PRESENT:  # as a hand program's: where present
            continue
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


def whole_fields(compiled, methods):
    """The hidden field of each reduce a later stream loop reads (wholes.lower): one
    fp32 tile, read by its handler."""
    for method in methods:
        for name in wholes.found(source_of(method.function)[0]):
            compiled.fields[wholes.hidden(name)] = front.Field(
                front.FORMATS["fp32"], (1, 32, 32)
            )


def field_users(methods, fields):
    """field -> (the handlers whose lines fill it, the handlers that read it and do
    not fill it), by `self.f.fill(...)` and the other `self.f` of each handler."""
    users = {name: ([], []) for name in fields}
    for method in methods:
        definition = source_of(method.function)[0]
        wholes.lower(definition)
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
    """A refusal of field `name`: the class, the field, its line in its class's file."""
    lines = (
        Path(compiled.field_files.get(name, compiled.source)).read_text().split("\n")
    )
    pattern = re.compile(rf"^\s*{name}\s*:")
    found = [n for n, line in enumerate(lines, start=1) if pattern.match(line)]
    where = f", line {found[0]}, `{lines[found[0] - 1].strip()}`" if found else ""
    return Refused(f"{compiled.name}.{name}{where}: {why}; {instead}")


def raw_checked(compiled, name, value):
    """A raw field's declaration: pages x items, of a format with item bytes."""
    if value.format.itemsize == 0:
        raise field_refused(
            compiled,
            name,
            f"a raw field of {value.format.dtype}: a block format has a tile's bytes "
            "only",
            "give km.u32, km.fp32, km.bf16 or km.fp16",
        )
    if len(value.shape) < 2:
        raise field_refused(
            compiled,
            name,
            "a raw field is pages x items: give both",
            f"declare it km.L1[format, pages, items] = km.{spelling('own')}(raw=True)",
        )


def raw_pages(compiled, name, value, solved):
    """A raw field's (page bytes, pages): its first size its pages, the product of
    the others times the format's item bytes a page's bytes (16 B or more, whole
    16 B)."""
    items = math.prod(solved[n] if isinstance(n, str) else n for n in value.shape[1:])
    page_bytes = items * value.format.itemsize
    if page_bytes < RAW_ALIGN or page_bytes % RAW_ALIGN:
        raise field_refused(
            compiled,
            name,
            f"a raw field's page of {page_bytes} B: a page is whole {RAW_ALIGN} B, "
            f"{RAW_ALIGN} B or more",
            "give items a page whose bytes are a multiple of 16",
        )
    first = value.shape[0]
    return page_bytes, solved[first] if isinstance(first, str) else first


def bind_fields(compiled, plan, l1):
    """Each field a buffer of whole tiles (a raw field's: pages of its own bytes);
    its shape's symbols from the plan, and one symbol left unknown solved as the
    largest that fits the L1 the others leave."""
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
        items = [n for n in symbols if name in compiled.raw and n in value.shape[1:]]
        if items:
            raise field_refused(
                compiled,
                name,
                f"{items[0]} is solved against the L1, and a raw field's solved "
                "size is its first, its pages",
                f"give {items[0]} in the {spelling('plan')}, or make it the pages",
            )
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
        raw = name in compiled.raw
        if raw:
            page_bytes, pages = raw_pages(compiled, name, value, solved)
        else:
            elements = field_elements(value, solved)
            if elements % TILE_ELEMENTS:
                raise field_refused(
                    compiled,
                    name,
                    f"{elements} items are not whole tiles ({TILE_ELEMENTS} a tile)",
                    "give sizes whose product is a multiple of 1,024",
                )
            page_bytes, pages = value.format.tile_bytes, elements // TILE_ELEMENTS
        shape = " x ".join(
            f"{n} = {solved[n]}" if isinstance(n, str) else str(n) for n in value.shape
        )
        buffer = Buffer(
            f"kField{camel(name)}",
            "field",
            layout.next_index(compiled, upper=name in compiled.own),
            None,
            value.format.dtype,
            page_bytes,
            (pages,),
            1,
            f"field {name}: {value.format.dtype}, {shape}"
            + (
                "; own: used by bodies, not filled by a line"
                if name in compiled.own
                else ""
            )
            + ("; raw pages" if raw else ""),
        )
        buffer.own, buffer.raw = name in compiled.own, raw
        layout.added(compiled, buffer)
        if name in compiled.own and value.format.dtype == "float32" and not raw:
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
            lines = Path(compiled.handler_files[handler]).read_text().split("\n")
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
            lines = Path(compiled.handler_files[handler]).read_text().split("\n")
            raise Refused(
                f"{compiled.name}.{handler}, line {line}, `{lines[line - 1].strip()}`: "
                f"a fill reads the field's {tiles} tiles, and its count is not known "
                f"to be {tiles}; write the count as {tiles}, as ints the build knows "
                "(+ - * //), or as the field's symbol"
            )


def field_elements(value, solved):
    return math.prod(solved[n] if isinstance(n, str) else n for n in value.shape)


def solve(compiled, symbol, plan, l1, name):
    """The largest value of `symbol` for which every field is whole tiles and the
    fields and buffers fit `l1` bytes."""
    used = shared.used_bytes(compiled)  # at shared indices, each index's largest
    fixed, per = 0, 0  # without the symbol, and per unit of it: 1/1,024ths of bytes
    step = 1
    for name, value in compiled.fields.items():
        items = field_elements(value, {**plan, symbol: 1})
        rest = items * value.format.tile_bytes
        if name in compiled.raw:  # pages x a page's bytes, no whole tiles
            rest = items * value.format.itemsize * TILE_ELEMENTS
            if symbol in value.shape:
                per += rest
            else:
                fixed += rest
            continue
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
