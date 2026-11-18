"""A handler's compile: its parameters bound (tensors, rectangles, regions,
constants), its body's statements, and the movers' and compute's lines put together
with the waits, blocks and puts its lines leave open."""

import ast
from functools import partial

from komorebi.dsl.compile import handler_buffers, scope
from komorebi.dsl.compile.casts import messages
from komorebi.dsl.compile.dst import dst_tiles
from komorebi.dsl.compile.frame import Frame
from komorebi.dsl.compile.handler import statements
from komorebi.dsl.compile.pages import field_fills, page_streams, stamp_pages
from komorebi.dsl.compile.pages.stamp_pages import timed_waits
from komorebi.dsl.compile.records import Param, Value
from komorebi.dsl.compile.source import (
    MAKE_OPTIONS,
    Where,
    check_name,
    file_of,
    refuse,
    source_of,
)
from komorebi.dsl.compile.tiles import wholes
from komorebi.dsl.front import front
from komorebi.dsl.front.names import kind_of, spelling
from komorebi.dsl.front.nodes import Glue, Line
from komorebi.dsl.islands import island_place

# Words of a command a handler's arguments may take (a0 to a9).
COMMAND_WORDS = 10


def frame_of(compiled, method, names, plan, tensors):
    """The frame of one handler's compile: its def parsed, its lines, the names it
    assigns twice, whether it is a message loop, and its statements' lowering."""
    frame = Frame(compiled, method, names, plan, tensors)
    frame.definition, lines, first = source_of(method.function)
    wholes.lower(frame.definition)  # a reduce a later loop reads: a field
    file = file_of(method.function)
    frame.where = Where(compiled.name, method.name, lines, first, file)
    compiled.handler_files[method.name] = file
    frame.assigned = scope.assigned_names(frame)
    frame.dynamic = messages.arrives(frame)
    frame.statements = partial(statements.statements, frame)
    return frame


def run(frame):
    name = frame.method.name
    frame.compiled.lines[name] = frame.definition.lineno
    if name in frame.compiled.folded:  # its loop is a case of the dynamic one's
        frame.compiled.words[name] = []
        frame.compiled.movement[name] = []
        frame.compiled.compute[name] = []
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
    if frame.dynamic:  # the ranks of the message loop: no counter tiles
        messages.dynamic_start(frame, compute)
    for nodes, me in ((movement, "movement"), (compute, "compute")):
        if not (frame.dynamic and nodes is compute):
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
    for index, field_ in enumerate(frame.compiled.word_fields):
        where = f"km::kMessagesCArgs + {index}" if frame.dynamic else str(index)
        compute.append(
            Glue(
                f"volatile tt_l1_ptr uint32_t* km_field_{field_} = "
                f"km::l1(compute.arg({where}));",
                {f"km_field_{field_}"},
            )
        )
    frame.rect_param = None
    bind_parameters(frame, movement, compute)
    rect_glue(frame, movement, compute)
    spans = None
    if frame.method.stamps is not None and not frame.method.overlay:
        spans = stamp_pages.tile_stamps(frame)
    elif frame.method.stamps is not None:
        stamp_pages.stamps(frame)
    if not frame.dynamic:
        compute.append(Glue("bool km_ok = true;", {"km_ok"}))
    if spans is not None:
        compute.append(spans[0])
    frame.block_words, frame.block_shadows = {}, {}
    marks = (len(movement), len(compute))
    frame.rect_read = False
    body_m, body_c = frame.statements(frame.definition.body)
    if frame.rect_param is not None and not (frame.rect_read or frame.movers):
        refuse(
            frame.where,
            frame.definition,
            "its rectangle is given at each submit, and it reads no self.x, "
            "self.y or broadcast",
            f"leave rect out of km.{spelling('up_to')}",
        )
    compute[marks[1] : marks[1]] = block_lines(frame)
    movement[marks[0] : marks[0]] = block_lines(frame)
    movement += frame.opening_b + body_m + put_waits(frame)
    compute += frame.opening + body_c + dst_tiles.carried_end(frame)
    if spans is not None:
        if not any(call[0] == name for call in frame.compiled.tile_calls):
            refuse(
                frame.where,
                frame.definition.decorator_list[0],
                f"stamps={frame.method.stamps!r} of a resident handler are the "
                "cycles of its tile function calls, and it calls none",
                "call a tile function, or write @km.handler(overlay=True, "
                'stamps="t")',
            )
        if stamp_pages.calls_steps(frame):
            timed_waits(body_c)
        compute.append(spans[1])
        movement.append(spans[2])
    for kept, filled in frame.filled.items():
        if filled is not None and not (
            kept in frame.waited_fields or field_fills.kept_field(frame, kept)
        ):
            refuse(
                frame.where,
                filled[0],
                f"self.{kept} is filled and no km.cxx body takes it",
                f"take it: km.cxx(..., takes=[self.{kept}])",
            )
    for value in frame.reduced:
        if value not in frame.leader_writes:
            refuse(
                frame.where,
                frame.definition,
                f"the reduce's {value} is not written by the leader",
                f"write it under `if self.{spelling('rank')} == "
                f"self.{spelling('leader')}:`",
            )
    page_streams.one_move(frame, movement)
    frame.compiled.movement[name] = handler_buffers.deferred_finishes(frame, movement)
    frame.compiled.compute[name] = compute + frame.closing


def bind_parameters(frame, movement, compute):
    """Each parameter after self is a command word: a tensor's reference word, an
    int, a pattern's word; a tensor whose pages are counted gets a word more."""
    args = frame.definition.args
    constants = constant_parameters(frame, args)
    if (
        args.vararg
        or args.kwarg
        or args.kwonlyargs
        or (
            len(args.defaults)
            > len(constants) + len(front.sized_parameters(frame.method.function))
        )
    ):
        refuse(
            frame.where,
            frame.definition,
            "a handler's parameters are plain, one a command word",
            "write `def name(self, a: km.Tensor, b: int, ...)`",
        )
    parameters = [p for p in args.args[1:] if p.arg not in constants]
    words = []
    for parameter in parameters:
        word = len(words)  # a rectangle takes four
        kind = parameter_kind(frame, parameter)
        check_name(frame.where, parameter, parameter.arg)
        param = Param(parameter.arg, kind, word)
        frame.params.append(param)
        words.append((parameter.arg, kind))
        if kind == "tensor" and param.name in MAKE_OPTIONS:
            refuse(
                frame.where,
                parameter,
                f"`{param.name}` is one of the class's make options "
                f"({', '.join(MAKE_OPTIONS)}): a tensor of that name cannot be "
                f"given to {frame.where.cls}(...)",
                "rename the parameter",
            )
        if kind == "tensor":
            frame.scope[param.name] = Value("tensor", param.name)
            if param.name not in frame.compiled.tensors:
                frame.compiled.tensors.append(param.name)
            text = f"const uint32_t {param.name} = command.a{word};"
        elif kind == "region":
            text = bind_region(frame, parameter, word)
        elif kind == "pattern":
            frame.scope[param.name] = Value("pattern", param.name)
            text = (
                f"const km::Pattern {param.name} = "
                f"km::Pattern::of(command.a{word});"
            )
        elif kind == "rect":
            text = bind_rect(frame, param.name, word, words)
        elif kind == "float":
            frame.scope[param.name] = Value("float", param.name)
            text = (
                f"const float {param.name} = "
                f"__builtin_bit_cast(float, command.a{word});"
            )
        else:
            frame.scope[param.name] = Value("int", param.name)
            text = f"const uint32_t {param.name} = command.a{word};"
        for nodes in (movement, compute):
            nodes.append(Glue(text, {param.name}))
    if front.rect_given(frame.method.function) is not None:  # km.UPTO(rect=)
        word = len(words)
        words.append(("rect", "rect"))
        text = bind_rect(frame, "rect", word, words, "km_rect_given")
        for nodes in (movement, compute):
            nodes.append(Glue(text, {"km_rect_given"}))
    counted = counted_tensors(frame)
    for tensor in counted:
        word = len(words)
        words.append((tensor, "pages"))
        frame.counts[tensor] = f"{tensor}_pages"
        for nodes in (movement, compute):
            nodes.append(
                Glue(
                    f"const uint32_t {tensor}_pages = command.a{word};",
                    {f"{tensor}_pages"},
                )
            )
    frame.column_words = {}
    sized = front.sized_parameters(frame.method.function)
    for tensor in [n for n, kind in words if kind == "tensor" and n in sized]:
        word = len(words)  # its tile columns, the tensor's at each submit
        words.append((tensor, "columns"))
        frame.column_words[tensor] = f"{tensor}_columns"
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
            frame.where,
            frame.definition,
            f"{len(words)} command words ({', '.join(n for n, _ in words)}); a "
            f"command has {COMMAND_WORDS}",
            "pass fewer arguments, or a tensor of them",
        )
    frame.words = words
    frame.compiled.words[frame.method.name] = words


def constant_parameters(frame, args):
    """The handler's constant parameters (km.constant[int] = n): each with its
    default, after the command words."""
    constants = front.constant_parameters(frame.method.function)
    defaulted = [p.arg for p in args.args[len(args.args) - len(args.defaults) :]]
    for parameter in args.args[1:]:
        if parameter.arg in constants and parameter.arg not in defaulted:
            refuse(
                frame.where,
                parameter,
                f"{parameter.arg} is a constant parameter: the handler is built "
                "for its default when no set of it is listed",
                f"give it a default, {parameter.arg}: "
                f"km.{spelling('constant')}[int] = n",
            )
    sized = front.sized_parameters(frame.method.function)
    if any(name not in constants and name not in sized for name in defaulted):
        return {}  # a default of a command word: refused by bind_parameters
    return constants


def parameter_kind(frame, parameter):
    annotation = parameter.annotation
    if annotation is None:
        refuse(
            frame.where,
            frame.definition,
            f"parameter {parameter.arg} has no type",
            f"write `{parameter.arg}: km.{spelling('tensor')}`, `: int` or "
            f"`: km.{spelling('pattern')}`",
        )
    if isinstance(annotation, ast.Name) and annotation.id in ("int", "float"):
        return annotation.id
    most = front.sized_parameters(frame.method.function).get(parameter.arg)
    if isinstance(annotation, ast.Name) and isinstance(most, tuple):
        return "rect"  # a rectangle given at each submit: four words
    if isinstance(annotation, ast.Subscript) and isinstance(most, int):
        return "int"  # a constant int km.UPTO gives at each submit: a word
    if isinstance(annotation, ast.Attribute):
        kind = kind_of(annotation.attr, ("tensor", "pattern", "region"))
        if kind:
            return kind
    refuse(
        frame.where,
        frame.definition,
        f"parameter {parameter.arg}: a type the command words cannot carry",
        f"a parameter is km.{spelling('tensor')}, int, float or "
        f"km.{spelling('pattern')}",
    )


def bind_rect(frame, name, word, words, text=None):
    """A rectangle given at each submit: its four words (x0, y0, columns, rows)
    as an array (named `text` in C++, else `name`), three more words after its
    place (`words` the handler's)."""
    text = text or name
    if frame.rect_param is not None:
        refuse(
            frame.where,
            frame.definition,
            f"two rectangles given at each submit ({frame.rect_param}, {name})",
            "give one",
        )
    frame.rect_param = text
    frame.compiled.rect_sized = True
    words += [(name, "rect words")] * 3
    places = ", ".join(f"command.a{word + i}" for i in range(4))
    return f"const uint32_t {text}[4] = {{{places}}};"


def rect_glue(frame, movement, compute):
    """A handler with a rectangle given at each submit: self.x, self.y, the
    rectangle's sides and whether the core is in it computed on the core
    (km_rect.hpp) from its words and the grid's (program arguments)."""
    if frame.rect_param is None:
        return
    for nodes, me in ((movement, "movement"), (compute, "compute")):
        nodes.append(Glue(island_place.grid(me, frame.compiled.islands), {"km_grid"}))
        nodes.append(
            Glue(
                "const km::RectPlace km_place = km::rect_place("
                f"{island_place.place(me, frame.compiled.islands)}, "
                f"km_grid.width, {frame.rect_param});",
                {"km_place"},
                {"km_grid", frame.rect_param},
            )
        )
        fields = {"km_self_x": "x", "km_self_y": "y"}
        fields |= {"km_rect_columns": "columns", "km_rect_rows": "rows"}
        nodes[:] = [  # the program arguments' place words: here the command's
            n for n in nodes if not (isinstance(n, Glue) and n.defines <= set(fields))
        ]
        for value, field_ in fields.items():
            nodes.append(
                Glue(
                    f"const uint32_t {value} = km_place.{field_};",
                    {value},
                    {"km_place"},
                )
            )


def bind_region(frame, parameter, word):
    """A region parameter: a tensor of the handler's lines in pages of the thread's
    granularity, its bank address this word and its bank a word after the rest."""
    name = parameter.arg
    example = frame.tensors.get(name)
    islands = {
        len(t.placement.parts)
        for t in frame.tensors.values()
        if hasattr(t, "placement") and hasattr(t.placement, "parts")
    }
    if islands and max(islands) > 1:
        island = getattr(example, "island", 0)
        refuse(
            frame.where,
            parameter,
            f"{name} is a region of island {island}'s DRAM, and the program is "
            f"made with island tensors: another island's core would read it",
            "a region is read by its own island's cores only; make the program "
            "with this island's tensors",
        )
    frame.scope[name] = Value("tensor", name)
    if name not in frame.compiled.regions:
        frame.compiled.regions.append(name)
    return f"km::Region {name}{{command.a{word}, 0}};"


def counted_tensors(frame):
    """Tensor parameters whose `pages` the handler uses whole (not called)."""
    tensors = {p.name for p in frame.params if p.kind == "tensor"}
    called = set()
    for node in ast.walk(frame.definition):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called.add(id(node.func))
    found = []
    for node in ast.walk(frame.definition):
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


def block_lines(frame):
    """Each pair's block given at each submit, as `<pair>Block` over the
    constant (the most), in the handler's movement and compute."""
    given = [
        Line(
            f"[[maybe_unused]] const uint32_t {name}Block = {word};  // "
            "given at each submit: a divisor of the constant's",
            {word},
        )
        for name, word in frame.block_words.items()
        if word is not None
    ]
    return given + [
        Line(
            f"[[maybe_unused]] constexpr uint32_t {name}Block = {block};  // this "
            "handler's block of a pair shared with other blocks",
            set(),
        )
        for name, block in frame.block_shadows.items()
    ]


def put_waits(frame):
    """The handler's end on each mover that put items: its puts waited for."""
    return [
        Line(
            "if (!km::put_wait(movement, command)) {\n    return false;\n}",
            set(),
            mover=mover,
        )
        for mover in frame.putters
    ]
