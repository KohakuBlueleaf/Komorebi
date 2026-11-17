"""A class deriving km.Program made into an ordinary program: the class is compiled
(komorebi.dsl.compiler), its files written under build/ beside its source and kept,
the sizes printed, and the made object is a Program of those files.
"""

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from komorebi import _env, abi, formats, layout
from komorebi.chain import calls as kc
from komorebi.dsl import compiler, first_use
from komorebi.dsl.chain import chain_calls
from komorebi.dsl.chain.chain_stamps import STAMP_BODY, STAMP_LOADED, STAMPS_PAGE
from komorebi.dsl.compile.layout import REGION_PAGE
from komorebi.dsl.compile.records import Buffer
from komorebi.dsl.files import (
    camel,
    put,
    text_key,
    thread_header,
    thread_header_name,
)
from komorebi.dsl.first_use import (
    constant_checked,
    reserve_of,
)
from komorebi.dsl.front import front
from komorebi.dsl.front.front import FIDELITIES, MESSAGE_WORDS, STARTUP_ORDERS, Startup
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import prune, render
from komorebi.dsl.thread import class_thread
from komorebi.dsl.tile import emit
from komorebi.dsl.tile.forms.rows import chunk_cost
from komorebi.dsl.tile.unit import matrix_cost
from komorebi.errors import L1BudgetError, ProgramError
from komorebi.program import description as km_program
from komorebi.program import overlay

BUILD = "build"
# A core's kernels as a CommandTimeout names them, and the CPU a class's wait is on.
CPUS = {"reader": "mover A", "writer": "mover B", "compute": "compute"}
INDENT = "    "
# The program arguments of every kernel: the core's column and row in the rectangle,
# whether a broadcast takes it, then (on a mover) its broadcast's arguments.
ARG_X, ARG_Y, ARG_INCLUDED, ARG_COLUMNS, ARG_ROWS, ARG_BROADCAST = 0, 1, 2, 3, 4, 5


@dataclass(frozen=True)
class Built:
    """What the build of a class wrote and chose: its folder, the files, the buffers'
    lines, and the waits of its code by handler and CPU; `make` the key of the make
    that wrote it (sources_key), `digest` the files' (text_key)."""

    folder: Path
    files: dict
    report: list
    waits: list
    make: str = ""
    digest: str = ""

    def lines(self, cores):
        """For each of `cores` (a CommandTimeout's) whose CPU gave up waiting, the
        handlers' Python lines that wait there."""
        named = []
        for core in cores:
            for kernel, cpu in CPUS.items():
                gave = core.get(kernel)
                where = gave.get("where") if isinstance(gave, dict) else None
                if not where:
                    continue
                named.extend(
                    f"core {core['core']} {cpu}: {wait.handler}, line {wait.line}: "
                    f"waiting in {wait.text}"
                    for wait in self.waits
                    if wait.cpu == cpu and wait.where == where
                )
        return named


def prepare(cls, compute=None, defines=None, startup=None):
    """Makes `cls`, a class deriving Program, a program written as a class if it has
    handler methods: its handlers numbered, its constructor the compiler; `compute`
    the class's compute options (checked when it is compiled), `defines` its
    kernels' defines, `startup` compute's start-up (a km.Startup)."""
    cls.constants_declared = front.constants_of(cls)
    methods = compiler.handler_methods(cls)
    if not methods:
        for given, kind in ((compute, "compute_options"), (defines, "defines")):
            if given is not None:
                raise ProgramError(
                    f"{cls.__name__}: {spelling(kind)}= is a class's with handlers "
                    f"(an ordinary program takes Program({spelling(kind)}=...))"
                )
        if startup is not None:
            raise ProgramError(f"{cls.__name__}: startup= is a class's with handlers")
        return
    if startup is not None and not isinstance(startup, Startup):
        raise ProgramError(
            f"{cls.__name__}: startup= is a km.{spelling('startup')}(...), got "
            f"{startup!r}"
        )
    # a subclass not giving them: its parent's (class_inherited)
    cls.compute_given = dict(class_inherited(cls, "compute_given", compute, {}))
    cls.defines_given = class_inherited(
        cls, "defines_given", defines_checked(cls.__name__, defines), {}
    )
    cls.startup_given = class_inherited(cls, "startup_given", startup, None)
    user_max = abi.headers()["KM_HANDLER_USER_MAX"]
    if len(methods) > user_max:
        raise ProgramError(f"{cls.__name__}: {len(methods)} handlers, {user_max} most")
    for number, method in enumerate(methods, start=1):
        method.number = number
    cls.__init__ = make
    class_thread.prepared(cls)  # km.x280_thread, km.region; program.thread
    cls.built_handlers = [method.name for method in methods]
    cls.overlays_built = property(overlays_built, set_overlays_built)


def class_inherited(cls, name, given, empty):
    """What the class gives (`given`), or when it gives nothing, its parent's."""
    if given is not None and given != {}:
        return given
    return getattr(cls, name, empty)


def make(
    self,
    plan=None,
    l1=None,
    depth=4,
    log=print,
    rect=None,
    nocs=None,
    session=None,
    trace=0,
    defines=None,
    handlers=None,
    broadcast_wait=False,
    guard=True,
    chain=False,
    chain_stamps=False,
    form="kernel",
    beside=(),
    **tensors,
):
    """Compiles the class: `tensors` an example of each tensor parameter, `plan` sizes
    by name; `session` (`trace` the load's) or `l1`; `rect`, `nocs` a broadcast's;
    `defines`; `handlers` constants' sets; `broadcast_wait`, `guard` a Program's;
    `chain` the walk; `chain_stamps` its stamps; `form`; `beside` programs beside it.
    """
    cls = type(self)
    thread_words = class_thread.words_given(tensors)
    thread_tensors = class_thread.tensors_given(tensors)
    if form not in FORMS:
        raise ProgramError(
            f"{cls.__name__}: {spelling('program_form')}={form!r}: one of "
            f"{', '.join(FORMS)}"
        )
    defines = {**cls.defines_given, **defines_checked(cls.__name__, defines)}
    plan = dict(plan or {})
    values, tensors = constants_given(cls, plan, tensors)
    handlers, self.up_to_given = up_to_made(cls, handlers, rect)  # km.UPTO
    handlers, found = first_use.opted(cls, handlers)  # km.ANY: first use allowed
    sets = handler_sets(cls, handlers)
    if session is not None:
        l1 = reserve_of(session)[0].l1()[1] if l1 is None else l1
        room = l1 - runtime_bytes(session, trace)
    else:
        room = l1
    reserved = beside_indices(cls.__name__, beside)
    given_room = room  # the class's buffers get it less a direct input's landing area
    room = direct_room(cls, compiler.handler_methods(cls), tensors, chain, room)
    islands = islands_of(session)
    compiled = compiler.compile_class(
        cls, plan, tensors, room, values, sets, reserved, islands
    )
    if found:
        first_use.reserve(
            cls, compiled, found, plan, tensors, room, values, sets, reserved
        )
    walked = chain_made(self, cls, compiled, chain, tensors)
    defines = stamps_made(self, cls, compiled, walked, chain_stamps, defines)
    if compiled.chain_gather:  # km.all_gather's marks in the walk (km_chain.hpp)
        defines = {**defines, "KM_CHAIN_GATHER": "1"}
    # what km.merge builds this program's handlers again with
    # (komorebi.dsl.merged)
    self.make_given = {
        "plan": plan, "tensors": tensors, "constants": values, "sets": sets,
        "session": session, "l1": l1, "trace": trace, "depth": depth, "rect": rect,
        "nocs": nocs, "defines": defines, "broadcast_wait": broadcast_wait,
        "guard": guard,
    }  # fmt: skip
    constants_made(self, cls, values, compiled)
    self.constant_handlers = compiled.variants
    refused = runtime_refused(compiled)
    if form == "runtime" and refused:
        raise ProgramError(refused)
    program_args = None
    # The rectangle's state, shared by the copies a session of islands runs.
    self.rectangle = {"rect": rect, "nocs": nocs, "args": None, "group": None}
    self.broadcast_lines = compiled.broadcast_lines
    self.reduce_groups = compiled.reduce_groups
    self.port_handlers = set(compiled.port_handlers)
    if compiled.rectangle:
        program_args = rectangle_of(self, compiled)
    self.word_fields = dict(compiled.word_fields)
    if compiled.word_fields:
        program_args = words_of(compiled, tensors, program_args)
    keyed = constants_keyed(cls, values)
    if sets:  # the listed sets of the handlers' constant parameters
        keyed[spelling("handlers")] = sorted(
            (name, [sorted(one.items()) for one in listed])
            for name, listed in sets.items()
        )
    if walked:
        keyed[spelling("chain")] = True
    if compiled.chain_stamps:
        keyed["chain_stamps"] = True
    if found:
        keyed[spelling("first_use")] = first_use.keyed(found)
    solved = {n: v for n, v in compiled.constants.items() if n not in plan}
    key = sources_key(
        plan, tensors, solved, depth, rect, nocs, broadcast_wait, keyed, guard
    )
    key = form_key(key, form, reserved)
    base = class_thread.header_text(
        class_thread.class_parts(self, compiled, tensors, found)
    )
    built = write(cls, compiled, given_room, key, form, thread=base)
    for line in built.report:
        log(line)
    if defines:
        log("  defines: " + ", ".join(f"{k}={v}" for k, v in defines.items()))
    report = may_cross(compiled, tensors)
    setattr(self, spelling("may_cross"), report)
    for line in report:
        log(line)
    self.placements = placements_of(compiled, tensors)
    place_handlers(compiled, {m.name: [m] for m in compiler.handler_methods(cls)})
    for line in self.placements:
        log(line)
    buffers = []
    for buffer in compiled.buffers:
        if buffer.index_b is None:
            buffers.append(
                km_program.Buffer(
                    buffer.index_a, buffer.dtype, buffer.page_bytes, buffer.pages[0]
                )
            )
        else:
            buffers.append(
                km_program.BufferPair(
                    buffer.index_a,
                    buffer.index_b,
                    buffer.dtype,
                    buffer.page_bytes,
                    buffer.pages,
                )
            )
    movement_args = []
    for name in compiled.tensors:
        movement_args += tensors[name].tensor_args()
    kernels = built.folder
    if form == "runtime":  # the parts of a program under the runtime's kernels
        kernels = {
            "reader": built.folder / "movement.cpp",
            "compute": built.folder / "compute.cpp",
            "writer": built.folder / "movement.cpp",
        }
    if compiled.dynamic:
        reader, writer = runtime_kernels()
        kernels = {
            "reader": reader,
            "compute": built.folder / "compute.cpp",
            "writer": writer,
        }
    km_program.Program.__init__(
        self,
        name=compiled.name.lower(),
        kernels=kernels,
        buffers=buffers,
        compile_args=form_compile_args(movement_args, form),
        compute=beside_compute(compute_options(compiled), beside),
        args=program_args,
        defines=defines,
        depth=depth,
        handlers=built.folder / header_name(compiled),
        broadcast_wait=broadcast_wait,
        guard=guard,
    )
    self.built = built
    self.log = log
    self.runtime_refused = refused  # what form="runtime" would refuse ("": nothing)
    self.load_start = form == "runtime" and load_started(compiled)  # programs.plan
    class_thread.made(self, cls, compiled, tensors, thread_words, found, thread_tensors)
    for name in compiled.overlays:
        sources = {"reader": built.folder / name / "movement.cpp"}
        sources["writer"] = sources["reader"]
        if f"{name}/compute.cpp" in built.files:
            sources["compute"] = built.folder / name / "compute.cpp"
        self.overlay(name, sources, defines=self.defines or None)
    if compiled.overlays:
        self.slots()
    if found:
        made_with = (plan, tensors, room, values, sets, reserved)
        first_use.kept(self, compiled, found, made_with, built, session, l1, trace)
    if session is not None:
        l1_checked(self, session, l1, trace, log)


def constants_given(cls, plan, given):
    """(this program's values of the class's constants by name, the make's other
    keywords): each declared constant's default, or its override by a keyword of its
    name or by the plan (not both); a derived one from those values, a subclass's
    declarations over its parents'. A symbol given by keyword joins the plan."""
    declared = getattr(cls, "constants_declared", {})
    tensors, overrides = {}, {}
    for name, value in given.items():
        if name in declared:
            overrides[name] = value
        elif isinstance(value, int | float):
            names = ", ".join(declared) or "none"
            raise ProgramError(
                f"{cls.__name__}: {name}={value!r} is not a constant of the class "
                f"(its constants: {names}); declare {name} = "
                f"km.{spelling('constant')}(default) in its body"
            )
        else:
            tensors[name] = value
    values = {}
    for name, declaration in declared.items():
        if name in overrides and name in plan:
            raise ProgramError(
                f"{cls.__name__}: {name} is given twice, as {name}= and in the "
                f"{spelling('plan')}; give it once"
            )
        given_value = overrides.get(name, plan.get(name))
        if isinstance(declaration, front.Symbol):
            if given_value is not None:
                plan[name] = values[name] = constant_checked(
                    cls, name, int, given_value
                )
            continue
        if isinstance(declaration, front.Derived):
            if name in overrides or name in plan:
                raise ProgramError(
                    f"{cls.__name__}: {name} is derived from the class's constants; "
                    "give the constants it is made of"
                )
            continue
        values[name] = (
            declaration.default
            if given_value is None
            else constant_checked(cls, name, declaration.kind, given_value)
        )
    for name, declaration in declared.items():
        if isinstance(declaration, front.Derived):
            values[name] = derived_value(cls, name, declaration, values)
    return values, tensors


def up_to_made(cls, handlers, rect):
    """(the make's `handlers` without km.UPTO, {handler: {name: most}}), the sizes
    set on each handler (front.up_to_set). Refused by name:
    km.UPTO twice for a handler, a listed set or km.ANY(p=n) naming what it gives, a
    rectangle given at each submit with rect= at the make."""
    word = f"km.{spelling('up_to')}"
    found, plain = {}, handlers
    if isinstance(handlers, dict):
        plain = {}
        for name, listed in handlers.items():
            if isinstance(listed, front.UpTo):
                listed = [listed]
            if isinstance(listed, list | tuple):
                ups = [one for one in listed if isinstance(one, front.UpTo)]
                if len(ups) > 1:
                    raise ProgramError(
                        f"{cls.__name__}.{name}: {spelling('handlers')}= gives {word} "
                        f"{len(ups)} times; give it once, naming each size"
                    )
                if ups:
                    if not ups[0].mosts:
                        raise ProgramError(
                            f"{cls.__name__}.{name}: {word} names no size; write "
                            f"{word}(block=8)"
                        )
                    found[name] = ups[0]
                    listed = [one for one in listed if not isinstance(one, front.UpTo)]
                    for one in listed:
                        named = sorted(set(found[name].mosts) & set(
                            one.reserve if isinstance(one, front.FirstUse) else one
                        ))  # fmt: skip
                        if named:
                            raise ProgramError(
                                f"{cls.__name__}.{name}: {', '.join(named)} is given "
                                f"at each submit ({word}); a set does not name it"
                            )
            plain[name] = listed
    methods = {m.name: m for m in compiler.handler_methods(cls)}
    unknown = sorted(set(found) - set(methods))
    if unknown:
        raise ProgramError(
            f"{cls.__name__}: {spelling('handlers')}= names {unknown[0]!r}, not a "
            f"handler of the class ({', '.join(methods)})"
        )
    given = {}
    for name, method in methods.items():
        given[name] = front.up_to_set(cls, method, found.get(name))
        if rect is not None and front.rect_given(method.function) is not None:
            raise ProgramError(
                f"{cls.__name__}.{name}: its rectangle is given at each submit "
                f"({word}(rect=...)); the make takes no rect="
            )
    if not found:
        return plain, given
    return {n: one for n, one in plain.items() if one or n not in found}, given


def handler_sets(cls, handlers):
    """The make's `handlers`, {handler: [{parameter: value}, ...]}: each a handler
    of the class with constant parameters, each set of them (a parameter not named
    takes its default), each value of its parameter's type, no set twice."""
    if handlers is None:
        return {}
    word = spelling("handlers")
    if not isinstance(handlers, dict):
        raise ProgramError(
            f"{cls.__name__}: {word}= is a dict, {{handler: [{{parameter: value}}, "
            "...]}"
        )
    methods = {method.name: method for method in compiler.handler_methods(cls)}
    sets = {}
    for name, listed in handlers.items():
        method = methods.get(name)
        if method is None:
            raise ProgramError(
                f"{cls.__name__}: {word}= names {name!r}, not a handler of the class "
                f"({', '.join(methods)})"
            )
        parameters = front.constant_parameters(method.function)
        if not parameters:
            raise ProgramError(
                f"{cls.__name__}.{name}: {word}= lists sets of a handler's constant "
                f"parameters (p: km.{spelling('constant')}[int] = n), and it has none"
            )
        if not isinstance(listed, list | tuple) or not all(
            isinstance(one, dict) for one in listed
        ):
            raise ProgramError(
                f"{cls.__name__}.{name}: {word}= gives a list of sets, each a dict "
                "of its constant parameters' values"
            )
        sets[name] = []
        for one in listed:
            unknown = sorted(set(one) - set(parameters))
            if unknown:
                raise ProgramError(
                    f"{cls.__name__}.{name}: {', '.join(unknown)} is not a constant "
                    f"parameter of it ({', '.join(parameters)})"
                )
            full = {
                parameter: constant_checked(
                    cls, f"{name}'s {parameter}", kind, one.get(parameter, default)
                )
                for parameter, (kind, default) in parameters.items()
            }
            if full in sets[name]:
                raise ProgramError(
                    f"{cls.__name__}.{name}: the set {full} is listed twice"
                )
            sets[name].append(full)
    return sets


def derived_value(cls, name, declaration, values):
    """A derived constant's value from this program's `values` by name."""
    try:
        return declaration.value(values)
    except KeyError as missing:
        raise ProgramError(
            f"{cls.__name__}: {name} is derived from {missing.args[0]}, which has no "
            f"value at the make (a symbol the build solves); give it in the "
            f"{spelling('plan')}"
        ) from None


def constants_made(program, cls, values, compiled):
    """Stores this program's constants on it, frozen: its values, each symbol's as
    the plan gave it or the build solved it."""
    values = dict(values)
    for name, declaration in getattr(cls, "constants_declared", {}).items():
        if isinstance(declaration, front.Symbol) and name in compiled.constants:
            values.setdefault(name, compiled.constants[name])
    program.__dict__["constant_values"] = MappingProxyType(values)


def constants_keyed(cls, values):
    """The constants a build's key holds: each declared constant but the symbols
    (the plan holds those given; the build solves the others)."""
    declared = getattr(cls, "constants_declared", {})
    return {
        name: value
        for name, value in values.items()
        if not isinstance(declared.get(name), front.Symbol)
    }


def defines_checked(name, defines):
    """The defines of a class or a make, by name: each a C++ name the runtime's do
    not take (they start KM_), each value an int or a string; as strings."""
    if defines is None:
        return {}
    if not isinstance(defines, dict):
        raise ProgramError(
            f"{name}: {spelling('defines')}= is a dict of names and values"
        )
    checked = {}
    for key, value in defines.items():
        if not (isinstance(key, str) and key.isidentifier()) or key.startswith("KM_"):
            raise ProgramError(
                f"{name}: define {key!r}: a C++ name, not one of the runtime's "
                "(KM_...)"
            )
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise ProgramError(f"{name}: define {key} = {value!r}: an int or a string")
        checked[key] = str(value)
    return checked


class Unbuilt:
    """A program of no buffers, threads or overlays: what the runtime takes of a core's
    L1 whatever the program's buffers (runtime_bytes)."""

    buffers, threads, overlays = (), {}, {}


def islands_of(session):
    """(islands, an island's most ranks) of a session: a tree reduce's slots over
    them (compile.tiles.reduce.tree_slots); (1, 0) on one card or with no session."""
    graph = getattr(session, "graph", None)
    if graph is None or len(graph.islands) <= 1:
        return (1, 0)
    count = len(graph.islands)
    return (count, max(graph.count(i) for i in range(count)))


def runtime_bytes(session, trace):
    """The L1 the load takes besides a class's buffers, at least: its table's rows for
    a program of none (Session.l1_rows) and the service overlay's slot at its most. A
    field's symbol is solved against what is left; l1_checked checks the class."""
    one, most = reserve_of(session)
    return sum(size for _, size in one.l1_rows(Unbuilt(), trace)) + most


def l1_checked(program, session, l1, trace, log):
    """The load's L1 table of a core for the built program (Session.l1_rows), the
    service overlay's slot at its most (the load logs a larger one), refused
    (L1BudgetError) as the load refuses it; a class's own overlays' slots are the
    load's."""
    session, most = reserve_of(session)
    rows = session.l1_rows(program, trace)
    rows[2:2] = [(f"slots: service (reader), 1 x {most:,} B (at most)", most)]
    table, fits = overlay.budget(l1, rows)
    log(f"L1 budget of a core (the build's, as the load's):\n{table}")
    if not fits:
        held = sum(size for _, size in session.l1_held())
        raise L1BudgetError(program.name, table, held)


def code_table(program, built):
    """Where each handler's code lives, once the session built the overlays: in the
    kernels, or an overlay (its parts' bytes, the slots of their kernels)."""
    compiled_overlays = {o.name: o.number for o in program.overlays.values()}
    lines = ["where the handlers' code lives:"]
    for name in program.built_handlers:
        if name not in compiled_overlays:
            lines.append(f"  {name}: resident, in the kernels")
            continue
        number = compiled_overlays[name]
        sizes = ", ".join(
            f"{part} {len(built.images[number, part]):,} B"
            for part in overlay.PARTS
            if (number, part) in built.images
        )
        slots = ", ".join(
            f"{part} {count} x {size:,} B"
            for part, (count, size) in built.slots.items()
        )
        lines.append(f"  {name}: overlay {number}: {sizes}; slots: {slots}")
    return lines


def overlays_built(self):
    return self.__dict__.get("_overlays_built")


def set_overlays_built(self, built):
    """The session's build of the overlays: kept, and the code table logged."""
    self.__dict__["_overlays_built"] = built
    if built is not None and self.overlays:
        for line in code_table(self, built):
            self.log(line)


# What a move lowers to: a tensor built for one island (its accessor), and one that
# may cross (km::Tensor with the tensor table: the far island's runs by its link).
LOCAL = {
    "read": "noc_async_read_page a page",
    "write": "noc_async_write_page a page",
}
CROSSING = {
    "read": "km::read_pages_at a row: this island's pages by noc_async_read_page, "
    "another island's run by an ask line to its link core (km::ask_line_at)",
    "write": "km::write_pages_at: this island's pages by noc_async_write_page, "
    "another island's run by a put line to its link core (km::ask_line_at)",
}


def may_cross(compiled, tensors):
    """The "may cross" report of a program made with island tensors (their
    placement): each move of each handler, whether a page of it can be another
    island's, and what it lowers to then; nothing on one island."""
    if not any(hasattr(tensors.get(name), "placement") for name in compiled.tensors):
        return []
    lines = ["may cross (each move: its tensor's placement; what it lowers to):"]
    for move in compiled.moves:
        placement = tensors[move.tensor].placement
        direction = (
            "read"
            if move.how in ("read stream", "fill", "broadcast's read", "word read")
            else "write"
        )
        holders = [i for i, (_, steps) in enumerate(placement.parts) if steps]
        if placement.copy:
            where = "a copy on every island: each island's own"
            crosses = False
        elif len(holders) == 1:
            where = f"held by island {holders[0]} alone"
            crosses = True
        else:
            parts = ", ".join(
                f"island {i} steps {first}..{first + steps - 1}"
                for i, (first, steps) in enumerate(placement.parts)
                if steps
            )
            where = f"split {placement.split} ({parts})"
            crosses = True
        lowered = CROSSING[direction] if crosses else LOCAL[direction]
        lines.append(
            f"  {move.handler}, line {move.line}, {move.how} of {move.tensor} "
            f"({where}): {'may cross' if crosses else 'never crosses'}; "
            f"{lowered}; one island: {LOCAL[direction]}"
        )
    return lines


def placements_of(compiled, tensors):
    """The build's report of each product's two placements and the significand bits
    (of 11) each operand keeps in each (formats.bits_used), the one built; refuses a
    turned product's W tensor not stored with transposed tiles, a plain one's so."""
    fidelity = compiled.fidelity or "HiFi4"
    lines = []
    for handler, line, text, left, right, placement, weight in compiled.placements:
        where = f"{compiled.name}.{handler}, line {line}: {text}"
        plain = (
            formats.bits_used(left, "srcb", fidelity),
            formats.bits_used(right, "srca", fidelity),
        )
        turned = (
            formats.bits_used(left, "srca", fidelity),
            formats.bits_used(right, "srcb", fidelity),
        )
        built = {"none": "plain, no marker", "plain": "plain, marked"}.get(
            placement, "turned, marked: W's tiles stored transposed, X's read so"
        )
        lines.append(
            f"  {where}: left {left}, right {right}, {fidelity}; plain (left in "
            f"SrcB): left keeps {plain[0]}, right {plain[1]}; turned (right in "
            f"SrcB): left {turned[0]}, right {turned[1]}; built {built}"
        )
        for _, name in weight_tensors(compiled, handler, weight):
            stored = getattr(tensors.get(name), "transposed_tiles", False)
            if stored != (placement == "turned"):
                raise ProgramError(
                    f"{where}: the product is {built.split(',')[0]} and tensor {name} "
                    f"is {'' if stored else 'not '}stored with its tiles transposed; "
                    f"write it with session.tensor(..., transposed_tiles={not stored})"
                )
    if lines:
        lines.insert(
            0,
            "placements (each product: the significand bits of 11 each operand keeps "
            "in the matrix unit, either way):",
        )
    return lines


def weight_tensors(compiled, handler, weight):
    """(handler, tensor parameter) of a product's W: its stream's or those a
    broadcast's block casts (the product's handler), or each fill of its field."""
    if weight is None:
        return []
    kind, name = weight
    if kind == "tensor":
        return [(handler, one) for one in name.split(",") if one]
    lines = {(h, line) for f, _, h, line in compiled.fills if f == name}
    return sorted(
        {
            (m.handler, m.tensor)
            for m in compiled.moves
            if m.how == "fill" and (m.handler, m.line) in lines
        }
    )


def placed_parameters(compiled):
    """{(handler, tensor parameter): (turned, the product's place)} of every product's
    W: what a command's tensor there is checked against (HandlerMethod.args)."""
    placed = {}
    for handler, line, text, _, _, placement, weight in compiled.placements:
        where = f"{compiled.name}.{handler}, line {line}: {text}"
        for pair in weight_tensors(compiled, handler, weight):
            placed[pair] = (placement == "turned", where)
    return placed


def place_handlers(compiled, methods):
    """Sets each handler method's `placed` (HandlerMethod.placed_check) and `paged`
    (its tensors' tiles a page as built): `methods` {the handler's name in
    `compiled`: the method objects a command reaches it by}."""
    placed = placed_parameters(compiled)
    for name, objects in methods.items():
        mine = {p: value for (h, p), value in placed.items() if h == name}
        paged = {p: k for (h, p), k in compiled.page_tiles.items() if h == name}
        for method in objects:
            method.placed = mine
            method.paged = paged


def rectangle_of(program, compiled):
    """The program arguments of a class with a rectangle, made at their first use
    (the load), once the session gave its NoCs."""
    state = program.rectangle
    if getattr(compiled, "rect_sized", False) and compiled.rect_plain:
        raise ProgramError(
            f"{compiled.name}: a handler reads self.x, self.y or a broadcast of the "
            "make's rectangle and another is given its rectangle at each submit: give "
            "every such handler its rectangle at each submit"
        )

    def args(core):
        if state["args"] is None and getattr(compiled, "rect_sized", False):
            nocs = state["nocs"] or nocs_needed()
            state["args"] = rect_grid_args(compiled, nocs, state.get("stack"))
        if state["args"] is None:
            nocs = state["nocs"]
            if nocs is None:
                raise ProgramError(
                    f"{compiled.name} uses self.x, self.y or a broadcast: load it with "
                    f"Session.load, or make it with {spelling('nocs')}=session.nocs()"
                )
            split = state.get("split")  # a split card's islands (island_session)
            stack = state.get("stack")  # the islands one below the other (the same)
            if stack is not None and spans(stack, state["rect"]):
                state["args"], state["group"] = stacked_rectangle(
                    compiled, stack, state["rect"], lambda: state.get("flat")
                )
                state["span"] = True
            elif stack is not None and state["rect"] is None:
                state["args"], state["group"], made = flat_rectangle(
                    compiled, stack, lambda: state.get("flat")
                )
                state["rect"], state["span"] = made, "flat"
            elif split is not None:
                state["args"], state["islands"] = island_rectangles(
                    compiled, nocs, split, state["rect"]
                )
                state["group"], _ = state["islands"][0]
            else:
                rect = state["rect"] or (0, 0, *nocs.grid)
                state["args"], state["group"] = rectangle_args(compiled, rect, nocs)
        flat = state.get("flat")  # this card's first flat rank (session_nocs)
        sized, span = getattr(compiled, "rect_sized", 0), state.get("span")
        if flat and compiled.broadcasts and not sized and not span:
            return flat_groups(state["args"](core), flat)
        return state["args"](core)

    return args


def flat_groups(movers, flat):
    """A core's program arguments with each broadcast's group moved to flat ranks
    from `flat` (the cores compare it with the command's group, in flat ranks): the
    card of a session over cards whose first flat rank is `flat`."""
    size = len(abi.struct_fields(abi.headers().device[-1], "BroadcastArgs"))
    out = []
    for words in movers[:2]:
        words = list(words)
        for at in range(5 + 1, len(words), size):  # after the place's five words
            words[at] += flat
        out.append(words)
    return (*out, movers[2])


def nocs_needed():
    raise ProgramError(
        "a rectangle given at each submit: load the class with Session.load, which "
        "gives the grid's NoC coordinates"
    )


def rect_header_lines(compiled):
    """A class whose rectangle is given at each submit: its program arguments are the
    grid's (komorebi.layout.rect_grid_words), the same on every core; each broadcast
    computed on the core takes its number, buffer, window and pages from here."""
    lines = [
        '#include "komorebi/device/movement/km_rect.hpp"',
        "// Program arguments: the grid's width and NoC coordinates (km::rect_grid);",
        "// each core's place and broadcasts are computed from the command's words.",
        "constexpr uint32_t kArgGrid = 0;",
        f"constexpr uint32_t kGridWords = {RECT_GRID_WORDS + grid_islands(compiled)};",
    ]
    if grid_islands(compiled):  # the island table after the grid (km::rect_islands)
        lines.append(f"constexpr uint32_t kGridIslands = {grid_islands(compiled)};")
    for number, (_, buffer, window, mover, _, index) in enumerate(compiled.broadcasts):
        place = f"{mover}{index}"
        if (mover, index) not in compiled.rect_broadcasts:
            continue
        lines += [
            f"constexpr uint32_t kRectId{place} = {number};",
            f"constexpr uint32_t kRectCb{place} = {buffer.index_a};",
            f"constexpr uint32_t kRectWindow{place} = {window};",
            f"constexpr uint32_t kRectPages{place} = {buffer.block};",
        ]
    return lines


RECT_GRID_WORDS = 11  # width, then 16 bytes of NoC x, 16 of NoC y, 8 of panels


def grid_islands(compiled):
    """The words of the island table after the grid's (a word an island) of a class
    whose rectangle is given at each submit, built for islands; 0 on one island."""
    count = compiled.islands[0]
    return count if count > 1 and getattr(compiled, "rect_sized", False) else 0


def rect_grid_args(compiled, nocs, stack=None):
    """The program arguments of a class whose rectangle is given at each submit: the
    grid's words for every core (km_rect.hpp's RectGrid), checked separable; built for
    islands, then a word an island of `stack` (island_session): its first flat rank |
    its first row in the stack << 16 | its first row of the card << 24."""
    columns, rows = nocs.grid
    xs = [nocs.noc_xy[(x, 0)][0] for x in range(columns)]
    ys = [nocs.noc_xy[(0, y)][1] for y in range(rows)]
    for x, y in ((x, y) for y in range(rows) for x in range(columns)):
        if tuple(nocs.noc_xy[(x, y)][:2]) != (xs[x], ys[y]):
            raise ProgramError(
                f"{compiled.name}: core ({x}, {y})'s NoC coordinates are not its "
                "column's and its row's; a rectangle given at each submit needs a grid "
                "whose coordinates are"
            )
    if columns > 16 or rows > 16 or len(nocs.panels) > 7:
        raise ProgramError(f"{compiled.name}: a grid past 16 x 16 or 7 panels")
    data = bytearray(40)
    data[0:columns] = bytes(xs)
    data[16 : 16 + rows] = bytes(ys)
    panels = [first for first, _ in nocs.panels] + [0xFF]
    data[32 : 32 + len(panels)] = bytes(panels)
    words = [columns]
    words += [int.from_bytes(data[i : i + 4], "little") for i in range(0, 40, 4)]
    if grid_islands(compiled):
        if stack is None or len(stack["islands"]) != grid_islands(compiled):
            raise ProgramError(
                f"{compiled.name} is built for {grid_islands(compiled)} islands: load "
                "it on the session of islands it was made with"
            )
        bands = stack["bands"] or [(0, 0)] * len(stack["tops"])
        for (_, first, _), top, (row, _) in zip(
            stack["islands"], stack["tops"], bands, strict=True
        ):
            if first > 0xFFFF or top > 0xFF:
                raise ProgramError(
                    f"{compiled.name}: an island's first rank {first} or first row "
                    f"{top} of the islands' stack past 65535 or 255"
                )
            words.append(first | top << 16 | row << 24)
    return lambda core: (words, words, words)


def words_of(compiled, tensors, rectangle):
    """The program arguments of a class with word fields: compute's, the L1 address
    of each field's tensor (made with a tensor of the field's name, one row in each
    core's L1, the host's to read after the run)."""
    if rectangle is not None:
        raise ProgramError(
            f"{compiled.name}: word fields with self.x, self.y or a broadcast are not "
            "in this version"
        )
    addresses = []
    for name, count in compiled.word_fields.items():
        tensor = tensors.get(name)
        if tensor is None or tensor.shard_grid is None or tensor.dtype != "uint32":
            raise ProgramError(
                f"{compiled.name}.{name} is {count} words in each core's L1: make "
                f"the program with {name}=session.empty((cores, {count}), "
                '"uint32", shard_grid=session.grid)'
            )
        if tensor.page_bytes < 4 * count:
            raise ProgramError(
                f"{compiled.name}.{name} is {count} words; a row of {name} is "
                f"{tensor.page_bytes} B"
            )
        addresses.append(tensor.address)
    return lambda core: ([], [], addresses)


def island_rectangles(compiled, nocs, islands, rect):
    """A class's `rect=` on a split card, each island its own: `islands` each one's
    (first row, rows, ranks, first flat rank); `rect` in the island's own rows, as on
    each card over cards. (args, {an island's first core: (its group in the card's
    cores, its rect)})."""
    columns = nocs.grid[0]
    parts, by_core = {}, []
    for island, (first_row, rows, ranks, flat) in enumerate(islands):
        base = first_row * columns
        x, y, width, height = rect
        if x + width > columns or (y + height - 1) * columns + x + width > ranks:
            raise ValueError(
                f"rect={tuple(rect)} is past island {island}'s {ranks} rank cores "
                f"({rows} rows of {columns}, its link cores last): give a rect= "
                "that every island's ranks hold"
            )
        at = flat + y * columns + x  # its first rank, flat
        args, group = rectangle_args(
            compiled, (x, y + first_row, width, height), nocs, at
        )
        parts[base] = (tuple(group), tuple(rect))
        by_core.append((base, base + rows * columns, args))

    def program_args(core):
        for low, high, args in by_core:
            if low <= core < high:
                return args(core)
        raise ValueError(f"core {core} is on no island")

    return program_args, parts


def spans(stack, rect):
    """Whether a make's `rect` is taller than island 0 (its rows run on to the next
    islands: stacked_rectangle); None, the islands' own rank cores, does not."""
    return rect is not None and rect[1] + rect[3] > stack["rows"][0]


def stacked_rectangle(compiled, stack, rect, flat):
    """A make's rect taller than island 0: one rectangle over the islands stacked from
    its first row (`stack`), a broadcast a part an island fed across, a core's place in
    the whole (layout.island_broadcast_args). (args, its group in flat ranks); `flat()`
    the loading card's first flat rank over cards (a split card's core says it)."""
    islands, bands, tops, rows = (
        stack[k] for k in ("islands", "bands", "tops", "rows")
    )
    columns = islands[0][0].grid[0]
    x0, y0, width, height = rect
    whole = (x0, y0 + tops[0], width, height)  # in the stack's rows (a split card's)

    corners = [(x0, whole[1]), (x0 + width - 1, whole[1] + height - 1)]
    ends = [
        layout.stack_rank(stack, columns, x0 + width - 1, y)
        for y in range(whole[1], whole[1] + height)
    ]
    if x0 + width > columns or None in ends:
        raise ValueError(
            f"rect={tuple(rect)} is taller than island 0: one rectangle from its first "
            "row on, and a cell of it is past the islands' rank cores (each island's "
            "link cores last): give a rect= on the islands' ranks"
        )
    first, last = (layout.stack_rank(stack, columns, *one) for one in corners)
    group = layout.Group(first, last + 1 - first)
    fields = abi.struct_fields(abi.headers().device[-1], "BroadcastArgs")
    by_mover = {}  # mover -> {its broadcasts' place: [their arguments by island core]}
    for number, (along, buffer, window, mover, length, index) in enumerate(
        compiled.broadcasts
    ):
        by_mover.setdefault(mover, {})[index] = layout.island_broadcast_args(
            islands,
            group,
            whole,
            along,
            0 if mover == "A" else 1,  # each mover runs on its own NoC
            (number, buffer.index_a, window, buffer.block),
            "below",
            bands,
            length,
        )

    def program_args(core):
        if bands is not None:  # a card's core: its island's band
            island = next(
                i for i, (top, n) in enumerate(bands) if core < (top + n) * columns
            )
            local = core - bands[island][0] * columns
        else:
            island = [one[1] for one in islands].index(flat() or 0)
            local = core
        x, y = local % columns, tops[island] + local // columns
        inside = local < islands[island][2] and (
            x0 <= x < x0 + width and whole[1] <= y < whole[1] + height
        )
        place = [x - x0, y - whole[1], 1] if inside else [0, 0, 0]
        common = [*place, width, height]
        movers = []
        for mover in ("A", "B"):  # each broadcast's words after the last's
            sets = by_mover.get(mover, {})
            words = [sets[index][island].get(local) for index in sorted(sets)]
            if not all(words):  # a core no broadcast of the mover takes
                words = []
            movers.append(common + [w[name] for w in words for name in fields])
        return movers[0], movers[1], common

    return program_args, (group.first, group.count)


def flat_rectangle(compiled, stack, flat):
    """A class's default rectangle on islands (`stack`): one island's, in the flat ranks
    wrapped at the grid's width (rank r at (r % width, r // width)), a broadcast's part
    an island or physical row fed across (island_broadcast_args flat). (args, group in
    flat ranks, rect: None the whole grid); `flat()` as stacked_rectangle's."""
    islands, bands = stack["islands"], stack["bands"]
    columns, rows = islands[0][0].grid
    ranks = sum(count for _, _, count in islands)
    if ranks >= columns * rows:  # as one island's: the whole grid
        rect, made, count = (0, 0, columns, rows), None, columns * rows
    elif compiled.broadcasts:  # its whole rows of ranks
        rect = made = (0, 0, columns, ranks // columns)
        count = columns * rect[3]
    else:  # every rank (self.x, self.y)
        rect, made, count = (0, 0, columns, rows), None, ranks
    group = layout.Group(0, count)
    fields = abi.struct_fields(abi.headers().device[-1], "BroadcastArgs")
    by_mover = {}  # mover -> {its broadcasts' place: [their arguments by island core]}
    for number, (along, buffer, window, mover, length, index) in enumerate(
        compiled.broadcasts
    ):
        by_mover.setdefault(mover, {})[index] = layout.island_broadcast_args(
            islands,
            group,
            rect,
            along,
            0 if mover == "A" else 1,  # each mover runs on its own NoC
            (number, buffer.index_a, window, buffer.block),
            "below",
            bands,
            length,
            flat=True,
        )

    def program_args(core):
        if bands is not None:  # a card's core: its island's band
            island = next(
                i for i, (top, n) in enumerate(bands) if core < (top + n) * columns
            )
            local = core - bands[island][0] * columns
        else:
            island = [one[1] for one in islands].index(flat() or 0)
            local = core
        rank = islands[island][1] + local
        inside = local < islands[island][2] and rank < count
        place = [rank % columns, rank // columns, 1] if inside else [0, 0, 0]
        common = [*place, columns, rect[3]]
        movers = []
        for mover in ("A", "B"):  # each broadcast's words after the last's
            sets = by_mover.get(mover, {})
            words = [sets[index][island].get(local) for index in sorted(sets)]
            if not all(words):  # a core no broadcast of the mover takes
                words = []
            movers.append(common + [w[name] for w in words for name in fields])
        return movers[0], movers[1], common

    return program_args, (group.first, group.count), made


def session_nocs(program, nocs, island=None):
    """Session.load's hand-over: the session's NoCs, unless the class was made with
    its own; on islands (`island`, the load's (graph, i)), its first flat rank."""
    state = getattr(program, "rectangle", None)
    if state is not None and state["nocs"] is None:
        state["nocs"] = nocs
    if state is not None:
        state["flat"] = island[0].first(island[1]) if island else None


def rectangle_args(compiled, rect, nocs, flat=None):
    """Each core's program arguments (a function of the core, as Program takes): its
    column and row in `rect`, whether it is there, its mover's broadcast arguments
    (their group the command's: `flat`, its first flat rank, on a split card); and
    the rectangle's group (first, count) in cores."""
    first_x, first_y, columns, rows = rect
    grid_width = nocs.grid[0]
    first = first_y * grid_width + first_x
    group = layout.Group(
        first, (first_y + rows - 1) * grid_width + first_x + columns - first
    )
    fields = abi.struct_fields(abi.headers().device[-1], "BroadcastArgs")
    by_mover = {}  # mover -> {its broadcasts' place: their arguments by core}
    for number, (along, buffer, window, mover, length, index) in enumerate(
        compiled.broadcasts
    ):
        noc = 0 if mover == "A" else 1  # each mover runs on its own NoC
        by_mover.setdefault(mover, {})[index] = layout.broadcast_args(
            nocs.grid,
            nocs.panels,
            group,
            rect,
            along,
            noc,
            nocs.noc0,
            nocs.noc_xy,
            (number, buffer.index_a, window, buffer.block),
            length,
        )
        if flat is not None:  # the cores compare it with the command's flat group
            sets = by_mover[mover][index]
            for core, words in sets.items():
                sets[core] = {**words, "group": flat | group.count << 16}
    cells = {
        y * grid_width + x: (x - first_x, y - first_y)
        for y in range(first_y, first_y + rows)
        for x in range(first_x, first_x + columns)
    }

    def program_args(core):
        x, y = cells.get(core, (0, 0))
        common = [x, y, int(core in cells), columns, rows]
        movers = []
        for mover in ("A", "B"):  # each broadcast's words after the last's
            sets = by_mover.get(mover, {})
            words = [sets[index].get(core) for index in sorted(sets)]
            if not all(words):  # a core no broadcast of the mover takes
                words = []
            movers.append(common + [w[name] for w in words for name in fields])
        return movers[0], movers[1], common

    return program_args, (group.first, group.count)


def dst_is_fp32(compiled):
    """Whether the class's compute has an fp32 Dst: yes, unless its buffers are of
    the A family, which the card reads right only into a 16-bit Dst; such a class
    has no float32 tile unpacked to Dst (a sum in a field, an own fp32 field)."""
    family = [b for b in compiled.buffers if formats.FAMILY.get(b.dtype) == "a"]
    if not family:
        return True
    fp32 = [b for b in compiled.buffers if b.name in compiled.unpack_fp32]
    if fp32:
        raise ProgramError(
            f"{compiled.name}: {family[0].what or family[0].name} is of the A family, "
            "whose compute has a 16-bit Dst, and "
            f"{fp32[0].what or fp32[0].name} is float32 tiles unpacked to Dst (fp32 "
            "only): give the class's tiles the B family, or no float32 sum"
        )
    return False


def compute_options(compiled):
    """The compute kernel's options: fp32 in DST; float32 buffers unpacked to it;
    DST in full-sync mode for a row-form tile function; then the class's own
    (compute=), over these."""
    options = {
        "math_fidelity": compiled.fidelity or "HiFi4",
        "fp32_dest_acc_en": dst_is_fp32(compiled),
    }
    if compiled.full_dst:  # a row-form tile function's DST tiles past the half
        options["dst_full_sync_en"] = True
    fp32 = [b for b in compiled.buffers if b.name in compiled.unpack_fp32]
    for buffer in fp32:
        options.setdefault("unpack_to_dest_mode", {})
        for index in (buffer.index_a, buffer.index_b):
            if index is not None:
                options["unpack_to_dest_mode"][index] = "UnpackToDestFp32"
    options.update(compiled.compute_options)
    return options


def header_name(compiled):
    return f"{compiled.name.lower()}.h"


FORMS = ("kernel", "runtime")


def form_compile_args(movement_args, form):
    """The movers' compile arguments: as "movement" in the kernel form, as the reader's
    and the writer's in the runtime's (its parts are built per kernel)."""
    if not movement_args:
        return {}
    if form == "runtime":
        return {"reader": list(movement_args), "writer": list(movement_args)}
    return {"movement": movement_args}


def form_key(key, form, reserved=()):
    """The build key of a form and of the indices left to other programs: the kernel
    form's without any as it was, the others apart."""
    if form == "kernel" and not reserved:
        return key
    made = f"{key} form={form} reserved={sorted(reserved)}"
    return hashlib.sha256(made.encode()).hexdigest()[:8]


def beside_indices(name, beside):
    """The buffer indices of the programs a class is made beside (a list load shares
    one L1 layout: its buffers take others)."""
    if isinstance(beside, km_program.Program):
        beside = (beside,)
    indices = set()
    for other in beside:
        if not isinstance(other, km_program.Program):
            raise ProgramError(f"{name}: beside= takes programs, got {other!r}")
        indices |= {buffer.index for buffer in other.buffers}
    return indices


def beside_compute(options, beside):
    """The compute options with the unpack modes of the programs a class is made
    beside, at their indices (which it leaves): one run's programs share one set."""
    if isinstance(beside, km_program.Program):
        beside = (beside,)
    for other in beside:
        modes = other.compute.get("unpack_to_dest_mode", {})
        if modes:
            options.setdefault("unpack_to_dest_mode", {}).update(modes)
    return options


def runtime_refused(compiled):
    """runtime_refusals' message, or "" when the runtime's form takes the class."""
    try:
        runtime_refusals(compiled)
    except ProgramError as error:
        return str(error)
    return ""


def runtime_refusals(compiled):
    """What the runtime's form has no place for, refused at its line: a dynamic
    handler, a body at a kernel's end. A body at a kernel's start and a km.Startup
    run at the load of the program (load_started)."""
    name, form = compiled.name, f'{spelling("program_form")}="runtime"'
    if compiled.dynamic:
        raise ProgramError(
            f"{name}.{compiled.dynamic} (line {compiled.lines[compiled.dynamic]}): a "
            f"dynamic handler; in {form} a program has none (the runtime's movers "
            "carry no messages)"
        )
    for at, line in compiled.kernel_body_lines:
        if at == "end":
            raise ProgramError(
                f"{name}: the body at the kernel's end (line {line}): in {form} the "
                "runtime owns the kernels' end (no program's part is called after "
                "the run stops); put it in a handler"
            )


def load_started(compiled):
    """Whether a program in the runtime's form has a start-up of its own, a body at
    a kernel's start or a km.Startup: run at a load of the program (KM_HANDLER_LOAD),
    so its run loads it first (komorebi.program.several.plan)."""
    starts = any(at == "start" for at, _ in compiled.kernel_body_lines)
    return starts or compiled.startup is not None


def runtime_start(compiled, kernel):
    """km_program_start of a program in the runtime's form with a start-up of its own
    (load_started): what the kernel form runs before the commands, run at a load of
    the program (km_overlay_entry.hpp). compute: the class's start-up (else the
    load's, on the program's first buffers), then its bodies."""
    bodies = compiled.kernel_bodies.get(("start", kernel))
    if not load_started(compiled) or (kernel == "movement" and not bodies):
        return []  # the movers do nothing at a load
    lines = [
        "",
        "// the program's start-up, at a load of the program",
        "#define KM_PROGRAM_START 1",
    ]
    if kernel == "movement":
        lines += [
            "static void km_program_start(km::MoverRuntime& km_core) {",
            *PART_MOVEMENT,
            f"{INDENT}(void)movement;",
            *(INDENT + line for line in tensor_aliases(compiled)),
        ]
    else:
        startup = (
            [INDENT + startup_line(compiled, "", "")]
            if compiled.startup is not None
            else [
                "#if defined(KM_PROGRAM_SETUP_IN)",
                f"{INDENT}compute_kernel_hw_startup(KM_PROGRAM_SETUP_IN, "
                "KM_PROGRAM_SETUP_OUT);",
                "#endif",
            ]
        )
        lines += [
            "static void km_program_start(km::Compute& compute) {",
            f"{INDENT}(void)compute;",
            *startup,
        ]
    return lines + [*kernel_bodies(compiled, "start", kernel), "}"]


def runtime_movement_source(compiled):
    """The movement source in the runtime's form: the handler switch as a function
    on the mover's runtime, called by each mover's km_program."""
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// its parts on the runtime's reader (RISCV_0) and writer (RISCV_1).",
        *movement_includes(compiled),
        '#include "komorebi/device/core/km_program.hpp"',
        f'#include "{header_name(compiled)}"',
        "",
        "static bool km_handlers(",
        f"{INDENT}km::MoverRuntime& km_core, const km_command_t& command",
        ") {",
        *PART_MOVEMENT,
        *(INDENT + line for line in tensor_aliases(compiled)),
        *cases(compiled, compiled.movement, "return true;", 1, "movement"),
        "}",
        *runtime_start(compiled, "movement"),
        "",
        "#if defined(COMPILE_FOR_BRISC)",
        "void km_program(km::Reader& reader, const km_command_t& command) {",
        f"{INDENT}km_handlers(reader, command);",
        "}",
        "#else",
        "bool km_program(km::Writer& writer, const km_command_t& command) {",
        f"{INDENT}return km_handlers(writer, command);",
        "}",
        "#endif",
    ]
    return "\n".join(lines) + "\n"


def runtime_compute_source(compiled):
    """The compute source in the runtime's form: the handler switch as compute's
    km_program (the start-up is the runtime's)."""
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// its part on compute, under the runtime's compute kernel.",
        *compute_includes(compiled),
        '#include "api/compute/reconfig_data_format.h"',
        '#include "komorebi/device/core/km_program.hpp"',
        f'#include "{header_name(compiled)}"',
        "",
        "void km_program(km::Compute& compute, const km_command_t& command) {",
        *start_formats(compiled),
        *cases(compiled, compiled.compute, "break;", 1, "compute"),
        "}",
        *runtime_start(compiled, "compute"),
    ]
    return "\n".join(lines) + "\n"


def start_formats(compiled):
    """A runtime-form command's first lines on compute: the formats of the runtime's
    start-up on this program's first buffers (programs.start_buffers), as the kernel
    form's code finds them after another program's command. None with a km.Startup:
    it set the formats at the load, on cores holding it alone."""
    if compiled.startup is not None:
        return []
    indices = sorted(
        index
        for buffer in compiled.buffers
        for index in (buffer.index_a, buffer.index_b)
        if index is not None
    )
    inputs = [index for index in indices if index < 16]
    outputs = [index for index in indices if index >= 16]
    lines = [INDENT + "// the start-up's formats on this program's buffers"]
    if inputs:
        lines.append(f"{INDENT}reconfig_data_format({inputs[0]}, {inputs[0]});")
    if outputs:
        lines.append(f"{INDENT}pack_reconfig_data_format({outputs[0]});")
    return lines if len(lines) > 1 else []


def sources_key(
    plan, tensors, solved, depth, rect, nocs, wait=False, constants=None, guard=True
):
    """A short hash of what changes a class's generated sources: the plan, the field
    sizes solved from the L1 left (`solved`), each tensor's format, page and shape, the
    make's options (a broadcast_wait of True or a guard of False too) and the class's
    constants (constants_keyed)."""
    made = [sorted(plan.items()), sorted((solved or {}).items()), depth, rect, nocs]
    if wait:
        made.append("broadcast_wait=True")
    if not guard:
        made.append("guard=False")
    if constants:  # none: the key of a class without constants as before them
        made.append(sorted(constants.items()))
    for name, tensor in sorted(tensors.items()):
        made.append((name, tensor.dtype, tensor.page_bytes, tuple(tensor.shape)))
        made.append(getattr(tensor, "layout", None))
    return hashlib.sha256(repr(made).encode()).hexdigest()[:8]


def write(cls, compiled, l1, make, form="kernel", key=None, thread=None):
    """Writes the class's files into build/<name>-<digest>/ beside its source, `digest`
    text_key of the files (equal text is one folder, whoever made it; text that differs
    never is); `make` is sources_key, reported; `key` names the folder instead (a first
    use's); `thread` is class_thread's base header; an equal file is not written."""
    files = {
        header_name(compiled): header(compiled),
        thread_header_name(compiled): thread_header(compiled),
        "movement.cpp": movement_source(compiled),
        "compute.cpp": compute_source(compiled),
        **parts(compiled),
    }
    for cxx, (_, unit) in compiled.units.items():  # the tile functions' bodies
        files[f"tile_{cxx}.hpp"] = emit(unit)
        files[f"tile_{cxx}.txt"] = unit.dump() + "\n"
    if form == "runtime":
        files["movement.cpp"] = runtime_movement_source(compiled)
        files["compute.cpp"] = runtime_compute_source(compiled)
    if compiled.dynamic:  # compute alone: the runtime's movers carry the messages
        del files["movement.cpp"]
        files["compute.cpp"] = dynamic_source(compiled)
    if thread is not None:
        files[thread_header_name(compiled)] = thread
    for name, text in files.items():
        files[name] = resume_lines(text, name)
    digest = text_key(files)
    folder = Path(compiled.source).resolve().parent / BUILD
    folder = folder / f"{compiled.name.lower()}-{key or digest}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        put(folder / name, text)
    report = summary(compiled, folder, files, l1)
    report.insert(1, f"  make {make}, files {digest}")
    put(folder / "program.txt", "\n".join(report) + "\n")
    return Built(folder, files, report, compiled.waits, make, digest)


def resume_lines(text, name):
    """Each `#line RESUME` after an inline body: the line it stands on, in `name`."""
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.strip() == "#line RESUME":
            lines[index] = f'#line {index + 2} "{name}"'
    return "\n".join(lines)


def header(compiled):
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// the handlers, buffers and command words its kernels share.",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        '#include "komorebi/device/movement/km_movement.hpp"',
    ]
    if "port" in compiled.includes:
        lines.append('#include "komorebi/device/movement/km_port.hpp"')
    if "tree" in compiled.includes:
        lines.append('#include "komorebi/device/movement/km_tree.hpp"')
    if "broadcast" in compiled.includes:
        lines.append('#include "komorebi/device/movement/km_broadcast.hpp"')
    lines.append("")
    if getattr(compiled, "rect_sized", False):
        lines += rect_header_lines(compiled)
    elif compiled.rectangle:
        lines += [
            "// Program arguments: the core's column and row in the rectangle, 1 when",
            "// a broadcast takes it, the rectangle's columns and rows, then its",
            "// mover's broadcast's arguments.",
            f"constexpr uint32_t kArgX = {ARG_X};",
            f"constexpr uint32_t kArgY = {ARG_Y};",
            f"constexpr uint32_t kArgIncluded = {ARG_INCLUDED};",
            f"constexpr uint32_t kArgColumns = {ARG_COLUMNS};",
            f"constexpr uint32_t kArgRows = {ARG_ROWS};",
            f"constexpr uint32_t kArgBroadcast = {ARG_BROADCAST};",
        ]
    for name, number in compiled.handlers.items():
        words = ", ".join(
            f"a{i} {word}{' pages' if kind == 'pages' else ''}"
            for i, (word, kind) in enumerate(compiled.words[name])
        )
        lines.append(
            f"// {name.upper()}, line {compiled.lines[name]}: {words or 'no words'}."
        )
        lines.append(f"constexpr uint32_t kHandler{camel(name)} = {number};")
        if name in compiled.overlays:
            lines.append(
                f"constexpr uint32_t kOverlay{camel(name)} = "
                f"{compiled.overlays[name]};  // its code: an overlay"
            )
    lines += chain_header(compiled)
    if compiled.messages:
        lines += ["", f"constexpr uint32_t kMessageWords = {MESSAGE_WORDS};"]
    for cls, words in compiled.messages.items():
        lines += [f"// {cls}: a message's words (word 0 the runtime's mark)."]
        for word, index in words.items():
            lines.append(f"constexpr uint32_t k{cls}{camel(word)} = {index};")
    for cls, words in compiled.entries.items():
        lines += ["", f"// {cls}: an entry's words in a table of entries."]
        for word, index in words.items():
            lines.append(f"constexpr uint32_t k{cls}{camel(word)} = {index};")
    for buffer in compiled.buffers:
        lines.append("")
        lines.append(f"// {buffer.what}")
        if buffer.index_b is None:
            if buffer.role == "arrival":
                slots = buffer.pages[0]
                lines.append(
                    f"constexpr km::Arrival {buffer.name} = "
                    f"{{{buffer.index_a}, {slots}, {buffer.block}}};"
                )
                lines += [  # a reduce of another part's pages: slot k at k x per
                    f"constexpr km::Arrival {buffer.name}Per{per} = "
                    f"{{{buffer.index_a}, {slots}, {per}}};"
                    for per in buffer.pers
                    if per != buffer.block
                ]
            elif buffer.role == "field":
                lines += [
                    f"constexpr uint32_t {buffer.name} = {buffer.index_a};",
                    f"constexpr uint32_t {buffer.name}Pages = {buffer.pages[0]};",
                ]
            elif buffer.role == "broadcast":
                window = buffer.pages[0] // buffer.block
                lines += [
                    f"constexpr uint32_t {buffer.name} = {buffer.index_a};",
                    f"constexpr uint32_t {buffer.name}Pages = {buffer.block};",
                    f"constexpr uint32_t {buffer.name}Window = {window};",
                ]
            elif buffer.role == "word":
                lines += [
                    f"constexpr uint32_t {buffer.name} = {buffer.index_a};",
                    f"constexpr uint32_t {buffer.name}Bytes = {buffer.page_bytes};",
                ]
            else:
                lines.append(f"constexpr uint32_t {buffer.name} = {buffer.index_a};")
            continue
        lines.append(
            f"constexpr km::BufferPair {buffer.name} = "
            f"{{{buffer.index_a}, {buffer.index_b}}};"
        )
        lines.append(f"constexpr uint32_t {buffer.name}Block = {buffer.block};")
        for constant, letters in buffer.patterns.items():
            lines.append(
                f'constexpr km::Pattern {constant} = km::pattern("{letters}");'
            )
    lines += format_macros(compiled)
    return "\n".join(lines) + "\n"


def mixed_formats(compiled):
    """(the formats compute unpacks are more than one, those it packs are): a
    kernel of one of each sets them once at its start."""
    unpacked, packed = set(), set()
    for buffer in compiled.buffers:
        if buffer.role in ("read", "field", "arrival", "broadcast"):
            unpacked.add(buffer.dtype)
            if buffer.name in compiled.summed_fields:  # sums packed into the field
                packed.add(buffer.dtype)
        elif buffer.role in ("write", "out"):
            packed.add(buffer.dtype)
            if buffer.name in compiled.unpack_fp32:  # a sum packed again
                unpacked.add(buffer.dtype)
    return len(unpacked) > 1, len(packed) > 1


def format_macros(compiled):
    """Compute's format changes before an unpack or a pack: the library's reconfig
    calls when the kernel's buffers have more than one format, else nothing."""
    unpack, pack = mixed_formats(compiled)
    return [
        "",
        "// Compute's formats: set again before each unpack and pack when its buffers",
        "// have more than one.",
        "#define KM_UNPACK_FORMAT_A(cb) "
        + ("reconfig_data_format_srca(cb)" if unpack else "((void)0)"),
        "#define KM_UNPACK_FORMATS(a, b) "
        + ("reconfig_data_format((a), (b))" if unpack else "((void)0)"),
        "#define KM_PACK_FORMAT(cb) "
        + ("pack_reconfig_data_format(cb)" if pack else "((void)0)"),
    ]


def tensor_aliases(compiled):
    lines, previous = [], None
    for name in compiled.tensors:
        offset = (
            "0" if previous is None else f"{previous}::next_compile_time_args_offset()"
        )
        lines.append(
            f"using {camel(name)} [[maybe_unused]] = km::Tensor<{offset}>;  // {name}"
        )
        previous = camel(name)
    return lines + region_aliases(compiled)


def region_aliases(compiled):
    """A region parameter's type: a thread's region in its pages (km_region.hpp)."""
    pages = f"km::RegionPages<{REGION_PAGE}>"
    return [
        f"using {camel(name)} [[maybe_unused]] = {pages};" for name in compiled.regions
    ]


def cases(compiled, nodes_of, ending, depth, kernel):
    """The switch on the handler: a case for each handler, its pruned nodes; an
    overlay handler's case loads its overlay and calls this kernel's part."""
    pad = INDENT * depth
    lines = [pad + "switch (km::handler(command)) {"]
    for name in compiled.handlers:
        line = compiled.lines[name]
        lines.append(f"{pad}{INDENT}case kHandler{camel(name)}: {{  // line {line}")
        if name not in compiled.overlays:
            lines += render(prune(nodes_of[name]), depth + 2)
        elif kernel == "movement" or has_part(nodes_of[name]):  # the reader loads
            inner = pad + INDENT * 2
            gave_up = "return false;" if kernel == "movement" else "break;"
            lines += [
                f"{inner}km::Overlay km_part({kernel}, kOverlay{camel(name)});",
                f"{inner}if (!km_part.load()) {{",
                f"{inner}{INDENT}{gave_up}",
                f"{inner}}}",
            ]
            if kernel == "movement" and name in compiled.stamps:
                lines[-4:] = stamped(compiled, name, inner)
                lines.append(f"{pad}{INDENT}}}")
                continue
            if kernel == "movement":
                lines.append(f"{inner}return km_part.call(command) != 0;")
                lines.append(f"{pad}{INDENT}}}")
                continue
            lines.append(f"{inner}km_part.call(command);")
        lines.append(f"{pad}{INDENT * 2}{ending}")
        lines.append(f"{pad}{INDENT}}}")
    lines.append(f"{pad}{INDENT}default:")
    lines.append(f"{pad}{INDENT * 2}{ending}")
    lines.append(pad + "}")
    return lines


def stamped(compiled, name, inner):
    """An overlay's resident case with its stamps: the wall clock around load and
    call, then the mover's page of the stamps tensor, written and settled."""
    tensor, word = compiled.stamps[name]
    overlay = f"kOverlay{camel(name)}"
    body = [
        f"km::Overlay km_part(movement, {overlay});",
        "const bool km_held = km_part.held();",
        "const uint64_t km_started = km::wall64();",
        "if (!km_part.load()) {",
        f"{INDENT}return false;",
        "}",
        "const uint64_t km_loaded = km::wall64();",
        "const bool km_called_ok = km_part.call(command) != 0;",
        "const uint64_t km_called = km::wall64();",
        "#if defined(COMPILE_FOR_BRISC)",
        "const uint32_t km_mover = 0;",
        "#else",
        "const uint32_t km_mover = 1;",
        "#endif",
        "const uint32_t km_stamp = get_write_ptr(kStamps) + km_mover * kStampsBytes;",
        "volatile tt_l1_ptr uint32_t* km_stamped = km::l1(km_stamp);",
        "km_stamped[0] = command.sequence;",
        f"km_stamped[1] = {overlay};",
        "km_stamped[2] = static_cast<uint32_t>(km_loaded - km_started);",
        "km_stamped[3] = static_cast<uint32_t>(km_called - km_loaded);",
        "km_stamped[4] = km_held;",
        f"const {camel(tensor)} km_stamps(command.a{word});",
        "if (!km::write_pages(movement, command, km_stamps,",
        "                     2 * movement.me() + km_mover, 1, km_stamp)) {",
        f"{INDENT}return false;",
        "}",
        "noc_async_write_barrier();",
        "return km_called_ok;",
    ]
    return [line if line.startswith("#") else inner + line for line in body]


def has_part(nodes):
    """Whether a kernel has code of a handler: lines other than comments."""
    return any(not line.strip().startswith("//") for line in render(prune(nodes), 0))


def movement_includes(compiled):
    lines = [
        "#include <stdint.h>",
        "",
        '#include "api/dataflow/dataflow_api.h"',
        '#include "komorebi/device/movement/km_movement.hpp"',
    ]
    if "port" in compiled.includes:
        lines.append('#include "komorebi/device/movement/km_port.hpp"')
    if "tree" in compiled.includes:
        lines.append('#include "komorebi/device/movement/km_tree.hpp"')
    if compiled.includes & {"broadcast", "ask"}:
        lines.append('#include "komorebi/device/tensors/km_ask.hpp"')
    if "broadcast" in compiled.includes:
        lines.append('#include "komorebi/device/movement/km_broadcast.hpp"')
    if compiled.overlays:
        lines.append('#include "komorebi/device/overlays/km_overlay.hpp"')
    if compiled.regions:
        lines.append('#include "komorebi/device/tensors/km_region.hpp"')
    return lines + compiled.body_includes["movement"]


def compute_includes(compiled):
    lines = ["#include <stdint.h>", ""]
    for path, needs in COMPUTE_INCLUDES:
        if needs is None or needs in compiled.includes:
            lines.append(f'#include "{path}"')
    if any(mixed_formats(compiled)):
        lines.append('#include "api/compute/reconfig_data_format.h"')
    if compiled.overlays:
        lines.append('#include "komorebi/device/overlays/km_overlay.hpp"')
    return lines + compiled.body_includes["compute"]


# A movement part's first lines: the resident case's km::Movement, as its runtime.
PART_MOVEMENT = [
    INDENT + "// the resident case's km::Movement, as its mover's runtime",
    INDENT + "km::Movement& movement = static_cast<km::Movement&>(km_core);",
]


def parts(compiled):
    """{path: text} of each overlay handler's parts: one movement source (built for
    each mover) and, if compute has code, a compute source."""
    files = {}
    for name in compiled.overlays:
        made = f"// {compiled.name}.{name}, made by komorebi.dsl from " + (
            f"{Path(compiled.source).name}: overlay kOverlay{camel(name)},"
        )
        files[f"{name}/movement.cpp"] = chain_part(
            compiled,
            "\n".join(
                [
                    made,
                    "// its part on each mover (mover A: RISCV_0, mover B: RISCV_1).",
                    *movement_includes(compiled),
                    f'#include "../{header_name(compiled)}"',
                    "",
                    "uint32_t km_overlay("
                    "km::MoverRuntime& km_core, const km_command_t& command) {",
                    *PART_MOVEMENT,
                    *(INDENT + line for line in tensor_aliases(compiled)),
                    *render(prune(compiled.movement[name]), 1),
                    f"{INDENT}return true;",
                    "}",
                ]
            )
            + "\n",
        )
        if has_part(compiled.compute[name]):
            files[f"{name}/compute.cpp"] = (
                "\n".join(
                    [
                        made,
                        "// its part on compute.",
                        *part_includes(compute_includes(compiled)),
                        f'#include "../{header_name(compiled)}"',
                        "",
                        "void km_overlay(",
                        "    km::Compute& compute, const km_command_t& command",
                        ") {",
                        *render(prune(compiled.compute[name]), 1),
                        "}",
                    ]
                )
                + "\n"
            )
    return files


def chain_part(compiled, text):
    """A movement part of a chain build: a tree's emit after its credit, as in the
    walk's cases (km::chain_credit reads only the chain's words, so the part may
    hold it); a part of a build without the walk as it is."""
    if not compiled.chain:
        return text
    head = f'#include "../{header_name(compiled)}"'
    text = text.replace(
        head, '#include "komorebi/device/movement/km_chain.hpp"\n' + head, 1
    )
    emit = "!km::reduce_emit(movement,"
    return text.replace(emit, "!km::chain_credit(movement) || " + emit)


def part_includes(lines):
    """A part's include lines: the tile functions' headers from the build folder, one
    up (a part built against the runtime's kernels has no class folder to search)."""
    return [line.replace('#include "tile_', '#include "../tile_') for line in lines]


def dynamic_source(compiled):
    """The compute kernel of a dynamic class: its handler's message loop (the reader
    and writer are the runtime's, messages/reader.cpp and messages/writer.cpp)."""
    name = compiled.dynamic
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// the compute kernel: the message loop of its dynamic handler.",
        "#include <stdint.h>",
        "",
        '#include "api/compute/common.h"',
        '#include "komorebi/device/messages/km_messages.hpp"',
        '#include "komorebi/device/km_worker.hpp"',
        *compiled.body_includes["compute"],
        f'#include "{header_name(compiled)}"',
        "",
        "void kernel_main() {",
        f"{INDENT}km::Compute compute;",
        *kernel_bodies(compiled, "start", "compute"),
        f"{INDENT}compute.run([&](const km_command_t& command) {{",
        f"{INDENT * 2}if (km::handler(command) != kHandler{camel(name)}) {{  "
        f"// line {compiled.lines[name]}",
        f"{INDENT * 3}return;",
        f"{INDENT * 2}}}",
        *render(prune(compiled.compute[name]), 2),
        f"{INDENT}}});",
        *kernel_bodies(compiled, "end", "compute"),
        "}",
    ]
    return "\n".join(lines) + "\n"


def runtime_kernels():
    """The runtime's message movers, the reader and writer of a dynamic class."""
    folder = _env.include_directory().parent / "kernels"
    return folder / "messages/reader.cpp", folder / "messages/writer.cpp"


def movement_source(compiled):
    if compiled.chain:
        return chain_movement_source(compiled)
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// the movement source of both movers (mover A: RISCV_0, mover B: RISCV_1).",
        *movement_includes(compiled),
    ]
    lines += [f'#include "{header_name(compiled)}"', "", "void kernel_main() {"]
    lines += [INDENT + line for line in tensor_aliases(compiled)]
    lines += [f"{INDENT}km::Movement movement;"]
    lines += kernel_bodies(compiled, "start", "movement")
    lines += [f"{INDENT}movement.run([&](const km_command_t& command) {{"]
    lines += cases(compiled, compiled.movement, "return true;", 2, "movement")
    lines += [f"{INDENT}}});", *kernel_bodies(compiled, "end", "movement"), "}"]
    return "\n".join(lines) + "\n"


def kernel_bodies(compiled, at, kernel):
    """The lines of the inline bodies at `kernel`'s start or end (km.cxx at=)."""
    lines = []
    for body in compiled.kernel_bodies.get((at, kernel), []):
        lines.append(f"{INDENT}// the class's body at the kernel's {at}")
        lines += [INDENT + line if line else "" for line in body.split("\n")]
    return lines


def startup_line(compiled, first_in, first_out):
    """compute's start-up: the class's (km.Startup, by its fields' buffers) or the
    default, by the first buffers it takes and gives."""
    startup = compiled.startup
    if startup is None:
        return f"compute_kernel_hw_startup({first_in}, {first_out});"
    names = []
    for name in startup.fields:
        buffer = next(
            (b for b in compiled.buffers if b.name == f"kField{camel(name)}"), None
        )
        if buffer is None:
            raise ProgramError(
                f"{compiled.name}: km.{spelling('startup')} names {name!r}, not a "
                f"field of the class ({', '.join(compiled.fields) or 'none'})"
            )
        names.append(buffer.name)
    order = "" if startup.order == "regular" else f"<{STARTUP_ORDERS[startup.order]}>"
    return f"compute_kernel_hw_startup{order}({', '.join(names)});"


COMPUTE_INCLUDES = [
    ("api/compute/bcast.h", "bcast"),
    ("api/compute/cb_api.h", None),
    ("api/compute/compute_kernel_api.h", None),
    ("api/compute/compute_kernel_hw_startup.h", None),
    ("api/compute/eltwise_binary_sfpu.h", "add"),
    ("api/compute/eltwise_unary/fill.h", "fill"),
    ("api/compute/experimental/matmul_custom.h", "matmul_custom"),
    ("api/compute/matmul.h", "matmul"),
    ("api/compute/pack.h", None),
    ("api/compute/reg_api.h", None),
    ("api/compute/tile_move_copy.h", None),
    ("api/compute/transpose_dest.h", "transpose_dest"),
    ("komorebi/device/movement/km_broadcast.hpp", "broadcast"),
    ("komorebi/device/movement/km_movement.hpp", None),
    ("komorebi/device/movement/km_port.hpp", "port"),
    ("komorebi/device/movement/km_tree.hpp", "tree"),
    ("komorebi/device/km_worker.hpp", None),
]


def compute_source(compiled, walk=True):
    if compiled.chain and walk:
        return chain_compute_source(compiled)
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// the compute kernel: the tiles of its streams, in block order.",
        *compute_includes(compiled),
    ]
    lines += [f'#include "{header_name(compiled)}"', "", "void kernel_main() {"]
    # The unit starts with buffers the class declares: its first read (or another
    # buffer compute takes) and its first write (or that one); none: 0 and 16.
    reads = [b for b in compiled.buffers if b.role == "read"]
    writes = [b for b in compiled.buffers if b.role in ("write", "out")]
    others = [b for b in compiled.buffers if b.role == "broadcast"] + [
        b
        for b in compiled.buffers
        if b.role not in ("read", "write", "out", "broadcast", "arrival")
    ]
    first_in = (
        f"{reads[0].name}.mover_a" if reads else others[0].name if others else "0"
    )
    first_out = "16" if first_in == "0" else first_in
    if writes:
        first_out = writes[0].name
        if writes[0].index_b is not None:
            first_out += ".mover_a"
    lines += [
        f"{INDENT}km::Compute compute;",
        INDENT + startup_line(compiled, first_in, first_out),
        *kernel_bodies(compiled, "start", "compute"),
        f"{INDENT}compute.run([&](const km_command_t& command) {{",
    ]
    lines += cases(compiled, compiled.compute, "break;", 2, "compute")
    lines += [f"{INDENT}}});", *kernel_bodies(compiled, "end", "compute"), "}"]
    return "\n".join(lines) + "\n"


def summary(compiled, folder, files, l1):
    """The build's report: the files, each buffer (its line, indices, sizes), the
    plan's values, and the L1 the buffers take."""
    report = [f"{compiled.name}: built into {folder}"]
    for name, text in files.items():
        report.append(f"  {name}: {text.count(chr(10))} lines")
    total = 0
    for buffer in compiled.buffers:
        indices = (
            f"{buffer.index_a}"
            if buffer.index_b is None
            else f"{buffer.index_a}, {buffer.index_b}"
        )
        pages = " + ".join(str(count) for count in buffer.pages)
        size = sum(buffer.pages) * buffer.page_bytes
        total += size
        report.append(
            f"  {buffer.name} ({buffer.role}, buffers {indices}): {buffer.dtype}, "
            f"{pages} pages of {buffer.page_bytes} B, {size} B; {buffer.what}"
        )
    for name, value in sorted(compiled.constants.items()):
        report.append(f"  {spelling('plan')}: {name} = {value}")
    report += tile_costs(compiled)
    report += product_costs(compiled)
    for name, (fillers, readers) in compiled.field_users.items():
        if fillers and readers:
            report.append(
                f"  field {name}: filled by {', '.join(fillers)}, read by "
                f"{', '.join(readers)} (it stays in L1 between commands)"
            )
    for handover in compiled.handovers:
        report.append(f"  hand-over: {handover}")
    room = "" if l1 is None else f" of {l1} B left for buffers"
    report.append(f"  L1: the buffers take {total} B{room}")
    if l1 is not None and total > l1:
        sizes = sorted(
            ((sum(b.pages) * b.page_bytes, b) for b in compiled.buffers),
            key=lambda pair: -pair[0],
        )
        largest = "; ".join(  # by name, so the line that holds it is plain
            f"{b.name} {size} B ({b.dtype}, {b.what})" for size, b in sizes[:4]
        )
        raise ProgramError(
            f"{compiled.name}: its buffers take {total} B, {l1} B left: "
            f"give smaller blocks in the plan, or smaller fields; the largest: "
            f"{largest}"
        )
    return report


def tile_costs(compiled):
    """Per call of a tile function: its unit program's cost on the math CPU (the
    tile compiler's model, the vector unit only), the rules that fired, the stated
    tolerance."""
    if not compiled.tile_calls:
        return []
    lines = [
        "tile functions (the math CPU's vector-unit cycles a tile, or a row of the "
        "row form, by the tile compiler's cost model; the copies into DST and the "
        "pack not priced):"
    ]
    for handler, line, text, function, cxx in compiled.tile_calls:
        key, unit = compiled.units[cxx]
        width = key[2]  # (function, formats, width[, the operands' DST tiles])
        count, cycles, _ = unit.cost("math")
        if width > 2:  # a wide row: the chunk parts once a chunk
            count, cycles = chunk_cost(unit, width)
        rules = [
            f"{p.name}: {'; '.join(counted(p.done))}"
            for p in unit.passes
            if p.done and p.name in ("load_macro", "replay", "fuse_mad")
        ]
        code = function.function.__code__
        products = {p.dst: p for p in unit.products}  # a product's Dst tile: a @ w
        operands = [
            f"{products[k].a} @ {products[k].w}" if name is None else name
            for k, name in enumerate(unit.operands)
        ]
        taken = set(unit.operands) | {n for p in unit.products for n in (p.a, p.w)}
        operands += [
            f"{name} (resident)"
            for name in code.co_varnames[: code.co_argcount]
            if name not in taken
        ]
        each = "a tile" if width == 1 else f"a row of {width} tiles"
        tolerance = (
            "not stated"
            if function.tolerance is None
            else f"{function.tolerance:g} relative against float64"
        )
        lines.append(
            f"  {handler}, line {line}, `{text}`: {function.name} on "
            f"{', '.join(operands)} (tile_{cxx}.hpp): {count} instructions, "
            f"{cycles} cycles {each}; {'; '.join(rules) or 'no rule fired'}; "
            f"tolerance {tolerance}"
        )
    return lines


def product_costs(compiled):
    """Per product `acc += f(block, self.w)`: the library's call, its matrix-unit
    words (16 a tile product a fidelity phase, HiFi4: 4) and their price by the tile
    compiler's model, between 6 or more Dst blocks rotating and none: the library's
    own word order is not known."""
    if not compiled.products:
        return []
    lines = [
        "products (the library's matmul_block, priced by the tile compiler's "
        "matrix_cost; the library's order, rotation not known):"
    ]
    for handler, line, text, function, columns, block, fidelity in compiled.products:
        phases = FIDELITIES[fidelity]
        if isinstance(block, int):  # a field of sums: a pair of broadcast blocks
            products = columns * block
            words = 16 * phases * products
            rotating = matrix_cost([("MVMUL", i % 8) for i in range(words)])
            alone = matrix_cost([("MVMUL", 0)] * words)
            lines.append(
                f"  {handler}, line {line}, `{text}`: {function.name}, a pair of "
                f"blocks into the field's {columns} tiles over k = {block}: "
                f"{products} tile products, {words} MVMUL words ({fidelity}), "
                f"{rotating:.1f} to {alone:.1f} cycles a pair; the unpack, the pack "
                "and its L1 accumulation not priced"
            )
            continue
        words = 16 * phases * columns
        rotating = matrix_cost([("MVMUL", i % 8) for i in range(words)])
        alone = matrix_cost([("MVMUL", 0)] * words)
        lines.append(
            f"  {handler}, line {line}, `{text}`: {function.name}, a tile of the "
            f"stream by {columns} tile(s) of the field: {columns} tile product(s), "
            f"{words} MVMUL words ({fidelity}), {rotating:.1f} to {alone:.1f} "
            "cycles a stream tile; the unpack of the operands and the pack not priced"
        )
    return lines


def counted(done):
    """A rule's entries in order, each once, a repeated one with its count."""
    seen = {}
    for entry in done:
        seen[entry] = seen.get(entry, 0) + 1
    return [entry if n == 1 else f"{entry} ({n} times)" for entry, n in seen.items()]


def lines_of(program, error):
    """The handler and Python line of each wait a not-reported core of `error` (a
    CommandTimeout) gave up in, by its CPU and where: one text a core."""
    built = getattr(program, "built", None)
    return built.lines(error.cores) if built is not None else []


# The chain's walk (km_chain.hpp) in a class's kernels: built when the class has a
# km.chain method or is made with chain=True; a class without it pays nothing.
CHAIN_PAGE = 2816  # the chain's words on each core (km_chain.hpp's kChainAreaBytes)
CHAIN_HANDLER = "km_chain"  # its handler's name: kHandlerKmChain
READ_MOVES = ("read stream", "word read", "fill", "broadcast's read", "table read")


def chain_methods(cls):
    """The class's km.chain methods (its parents' too), by name."""
    return {
        name: value
        for klass in reversed(cls.__mro__)
        for name, value in vars(klass).items()
        if isinstance(value, kc.ChainMethod)
    }


def chain_made(program, cls, compiled, chain, tensors=None, handlers=None):
    """The chain's walk when `chain` or the class has a km.chain method (its lines
    checked here, refused at their line); on `program`: its handler (0: none) and
    its direct input's landing area (None: none), whether a km.chain method gathers
    (km.all_gather); `handlers` its methods by name (a merged class's copies)."""
    methods = chain_methods(cls)
    gathers = [chain_calls.checked(cls, method) for method in methods.values()]
    compiled.chain_gather = any(gathers)
    if chain or methods:
        chain_prepared(cls, compiled)
        if handlers is None:
            handlers = {m.name: m for m in compiler.handler_methods(cls)}
        direct_prepared(cls, compiled, tensors or {}, handlers)
    program.chain_handler = compiled.chain
    program.chain_gather = compiled.chain_gather
    program.direct_area = getattr(compiled, "direct_area", None)
    program.direct_in = getattr(compiled, "direct_in", None)  # in place: (handler, ...)
    return bool(compiled.chain)


DIRECT_FIRST = 33  # the landing area's buffer: the first index free from here
IN_PLACE_LAST = 32  # an in-place stream's pair: two indices free under this (move())


def direct_declared(methods, tensors):
    """{(handler, parameter): its example tensor} of each direct-declared parameter
    the make gives a tensor for."""
    return {
        (method.name, place): tensors[place]
        for method in methods
        for place in getattr(method, "direct", ())
        if place in tensors
    }


def direct_room(cls, methods, tensors, chain, room):
    """The L1 left for the class's buffers once a chain build's landing area is
    taken (`room` unchanged without one, or without a room given); refused with the
    numbers when the area alone does not fit."""
    if room is None or not (chain or chain_methods(cls)):
        return room
    declared = direct_declared(methods, tensors)
    area = max((t.pages * t.page_bytes for t in declared.values()), default=0)
    if area > room:
        raise ProgramError(
            f"{cls.__name__}: the direct inputs' landing area takes {area:,} B "
            f"(for {', '.join(f'{h}.{p}' for h, p in sorted(declared))}) and "
            f"{room:,} B are left for buffers; give a smaller tensor at the make or "
            "drop direct="
        )
    return room - area


def direct_prepared(cls, compiled, tensors, handlers):
    """A chain build of a class whose handlers declare direct inputs: one landing
    area (buffer kDirectArea) the size of the largest such tensor given at the make
    (its bytes taken out of the room before the compile, direct_room), its tiles'
    bytes the same for each; an overlay handler's direct input refused."""
    declared = {}
    for name in compiled.handlers:
        method = handlers.get(name)
        for place in getattr(method, "direct", ()):
            if name in compiled.overlays:
                raise ProgramError(
                    f"{cls.__name__}.{name}: a direct input ({place}) to an overlay "
                    "handler is not built (its part cannot reach the chain's record); "
                    "make it resident or drop direct="
                )
            if place in tensors:
                declared[(name, place)] = tensors[place]
    if not declared:
        return
    sizes = {tensor.page_bytes for tensor in declared.values()}
    if len(sizes) != 1:
        raise ProgramError(
            f"{cls.__name__}: direct inputs of tiles of {sorted(sizes)} bytes; one "
            "landing area takes one tile size"
        )
    used = {b.index_a for b in compiled.buffers} | {
        b.index_b for b in compiled.buffers if b.index_b is not None
    }
    index = next(i for i in range(DIRECT_FIRST, 64) if i not in used)
    tiles = max(tensor.pages for tensor in declared.values())
    (page,) = sizes
    compiled.direct_area = (tiles, page, sorted(declared))
    placed = [key for key in declared if getattr(handlers.get(key[0]), "in_place", 0)]
    if placed:
        return in_place_prepared(cls, compiled, sorted(declared), placed[0], used)
    compiled.buffers.append(
        Buffer(
            "kDirectArea",
            "direct",
            index,
            None,
            "uint32",
            page,
            (tiles,),
            what="a direct input's landing area (km_chain.hpp chain_send, DirectOr)",
        )
    )


def chain_prepared(cls, compiled):
    """A build with the chain's walk: its handler number after the class's, and the
    chain's buffer (km_chain.h's kChainBuffer); a dynamic class refused."""
    words = abi.constants(abi.headers().chain)
    if compiled.dynamic:
        raise ProgramError(
            f"{cls.__name__}: a chain's walk in a class with a dynamic handler is not "
            "built; make it without one"
        )
    number = max(compiled.handlers.values(), default=0) + 1
    if number > abi.headers()["KM_HANDLER_USER_MAX"]:
        raise ProgramError(f"{cls.__name__}: no handler number left for the chain")
    compiled.chain = number
    compiled.buffers.append(
        Buffer(
            "kChainArea",
            "chain",
            words["kChainBuffer"],
            None,
            "uint32",
            CHAIN_PAGE,
            (1,),
            what="the chain's words (km_chain.hpp): marks, credits, the records' ring",
        )
    )


def chain_header(compiled):
    """The chain's handler in the header (none without the walk)."""
    if not compiled.chain:
        return []
    return [
        "// The chain's walk: a chain command's records, each its handler's case.",
        f"constexpr uint32_t kHandler{camel(CHAIN_HANDLER)} = {compiled.chain};",
        *direct_header(compiled),
    ]


def direct_header(compiled):
    """The landing area's tiles and their bytes (none without a direct input)."""
    area = getattr(compiled, "direct_area", None)
    if area is None:
        return []
    return [
        "// A direct input's landing area: its tiles, each of these bytes.",
        f"constexpr uint32_t kDirectAreaTiles = {area[0]};",
        f"constexpr uint32_t kDirectAreaBytes = {area[1]};",
    ]


def statement_starts(source):
    """Each line of the class's file that a statement's head covers -> that
    statement's first line (a kernel's `// line N:` comment)."""
    tree = ast.parse(Path(source).read_text())
    starts = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        body = getattr(node, "body", None)
        end = body[0].lineno - 1 if isinstance(body, list) and body else node.end_lineno
        for line in range(node.lineno, max(end, node.lineno) + 1):
            if line not in starts or starts[line] < node.lineno:
                starts[line] = node.lineno
    return starts


def chain_reads(compiled):
    """handler -> {statement line: (the bits of the tensors it reads, prefetch=)}: a
    tensor's bit its place among the handler's tensor parameters."""
    files = handler_files(compiled)
    reads = body_reads(compiled, files)
    for move in compiled.moves:
        if move.how not in READ_MOVES:
            continue
        for name in compiled.handlers:
            words = compiled.words[name]
            tensors = [word for word, kind in words if kind == "tensor"]
            if name != move.handler or move.tensor not in tensors:
                continue
            starts, prefetch = files[name]
            line = starts.get(move.line, move.line)
            bits, stated = reads.setdefault(name, {}).get(line, (0, None))
            bits |= 1 << tensors.index(move.tensor)
            if stated is not False:
                stated = prefetch.get(line, stated)
            reads[name][line] = (bits, stated)
    return reads


def handler_files(compiled):
    """handler -> (statement_starts, chain_prefetch) of the file its lines are in: the
    class's, or in a merged build each class's own (compiled.handler_sources)."""
    sources = getattr(compiled, "handler_sources", {})
    parsed = {}
    for name in compiled.handlers:
        source = sources.get(name, compiled.source)
        if source not in parsed:
            starts = statement_starts(source)
            parsed[source] = (starts, chain_prefetch(source, starts))
    return {n: parsed[sources.get(n, compiled.source)] for n in compiled.handlers}


def body_reads(compiled, files):
    """handler -> {line: (all bits, False)} of each inline body that takes a tensor:
    what it reads is its own C++, so its record's input is waited for before it."""
    sources = getattr(compiled, "handler_sources", {})
    trees = {}
    reads = {}
    for name in compiled.handlers:
        starts = files[name][0]
        source = sources.get(name, compiled.source)
        if source not in trees:
            trees[source] = {
                node.lineno: node
                for node in ast.walk(ast.parse(Path(source).read_text()))
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            }
        functions = trees[source]
        tensors = {word for word, kind in compiled.words[name] if kind == "tensor"}
        function = functions.get(compiled.lines[name])
        for node in ast.walk(function) if function is not None else ():
            if not (isinstance(node, ast.Call) and node.keywords):
                continue
            taken = [
                taken.id
                for keyword in node.keywords
                if keyword.arg == "takes" and isinstance(keyword.value, ast.List)
                for taken in keyword.value.elts
                if isinstance(taken, ast.Name)
            ]
            if tensors & set(taken):
                line = starts.get(node.lineno, node.lineno)
                reads.setdefault(name, {})[line] = (0xFFFFFFFF, False)
    return reads


def chain_prefetch(source, starts):
    """Each read stream's statement (its first line, `starts`) in the class's file ->
    its prefetch= (True, False); False when any stream of the statement says False."""
    found = {}
    for node in ast.walk(ast.parse(Path(source).read_text())):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == spelling("prefetch") and isinstance(
                    keyword.value, ast.Constant
                ):
                    line = starts.get(node.lineno, node.lineno)
                    if found.get(line) is not False:
                        found[line] = keyword.value.value
    return found


def chain_mask_word(compiled, name):
    """The record word that holds a handler's mask of chain-written tensors: the
    first word after its own (None: it has ten, so it waits before any read)."""
    count = len(compiled.words[name])
    return count if count < 10 else None


def chain_waits(compiled, name, lines, reads):
    """A handler's case lines with the input's wait before each line that reads a
    tensor a chain call writes (the record's mask), at the case's start when a read's
    line is not found; a loop over streams in step holds only the streams of those
    tensors instead (chain_gated)."""
    out, found = [], set()
    word = chain_mask_word(compiled, name)
    lines = list(lines)
    for at, text in enumerate(lines):
        out.append(text)
        stripped = text.lstrip()
        if not stripped.startswith("// line "):
            continue
        line = int(stripped[len("// line ") :].split(":")[0])
        if line not in reads:
            continue
        bits, stated = reads[line]
        mask = "~0u" if word is None or stated is False else f"command.a{word}"
        pad = text[: len(text) - len(stripped)]
        if word is not None and stated is not False:
            gate = chain_gated(compiled, name, line, lines, at, mask, bits)
            if gate is not None:
                out.append(pad + gate)
                found.add(line)
                continue
        out.append(f"{pad}if (!km_chain_read(movement, {mask}, {bits}u)) {{")
        out.append(f"{pad}{INDENT}return false;")
        out.append(f"{pad}}}")
        found.add(line)
    if set(reads) - found:  # a read's line not seen: the wait first
        pad = INDENT * 4
        out[:0] = [
            f"{pad}if (!km_chain_read(movement, ~0u, ~0u)) {{",
            f"{pad}{INDENT}return false;",
            f"{pad}}}",
        ]
    return out


def chain_gated(compiled, name, line, lines, at, mask, bits):
    """A loop over streams in step at `line` (case lines from `at`): its move's read
    streams held by one gate (km::Gated, km::ChainGate), so the streams of tensors
    the chain does not write go on while the input is awaited. The gate's line, the
    move's line rewritten in `lines`; None when the line is not such a loop."""
    zipped = compiled.zips.get((name, line))
    if zipped is None or any(stated is False for *_, stated in zipped):
        return None  # a prefetch=False stream: the record's whole wait
    tensors = [word for word, kind in compiled.words[name] if kind == "tensor"]
    gate = f"km_gate{zipped[0][0].removeprefix('km_read')}"
    for index in range(at + 1, len(lines)):
        text = lines[index]
        if text.lstrip().startswith("// line "):
            return None
        if "movement.move(" not in text:
            continue
        for variable, tensor, _ in zipped:
            held = f"km::Gated({variable}, {gate}, {1 << tensors.index(tensor)}u)"
            text, count = re.subn(rf"\b{variable}\b", held, text)
            if count != 1:
                return None
        lines[index] = text
        return (
            f"km::ChainGate {gate} = km::chain_gate(km::ChainShape{{}}, movement, "
            f"{mask}, {bits}u, km_chain_waited);"
        )
    return None


CHAIN_HELPERS = """namespace {

    bool km_chain_waited = false;  // this record's input waited for

    // Before a line reading tensors `bits` of a chain's record: its input's marks
    // waited for once, unless `mask` (the chain's written tensors) has none of them.
    __attribute__((noinline)) bool km_chain_read(
        km::Movement& movement, uint32_t mask, uint32_t bits
    ) {
        if (km_chain_waited || !km::chain_now.on || (mask & bits) == 0) {
            return true;
        }
        km_chain_waited = true;
        return km::chain_input(km::ChainShape{}, movement);
    }

    // A record's end: on mover B each group's root marks the next record's cores once
    // its block is written.
    __attribute__((noinline)) bool km_chain_end(
        km::Movement& movement, const km_command_t& command
    ) {
        if (!km::chain_now.on) {
            return true;
        }
#if defined(COMPILE_FOR_BRISC)  // mover A
        return true;
#else  // mover B
        const uint32_t size = km::ChainShape{}.size(movement, km::chain_now.now);
        if ((movement.me() - km::first(command)) % size != 0) {
            return true;
        }
        const km::ChainDirect none = {0, 0, 0, 0};
        return km::chain_mark(km::ChainShape{}, movement, command, none, false);
#endif
    }

}  // namespace
"""


# A class with a direct input: the root's blocks sent as written, read DRAM no more.
DIRECT_HELPERS = """
    // The tiles of the root's block sent direct in this record (in pieces as written).
    uint32_t km_direct_sent = 0;

    // The record km_direct_next answered for (its tag), the answer and that record's
    // words, until its end.
    uint32_t km_direct_tag = 0;
    bool km_direct_known = false, km_direct_is = false;
    uint32_t km_direct_words[kChainWords];

    // Whether the next record takes this record's output direct (false outside a
    // chain), read once a record. False: gave up (recorded).
    __attribute__((noinline)) bool km_direct_next(
        km::Movement& movement, bool& direct
    ) {
        direct = false;
        if (!km::chain_now.on || km::chain_next_address(km::chain_now.now) == 0) {
            return true;
        }
        const uint32_t tag = km::chain_now.tag_base + km::chain_now.record + 1;
        if (km_direct_known && km_direct_tag == tag) {
            direct = km_direct_is;
            return true;
        }
        if (!km::chain_reached(movement, km::kChainReady, tag, km::kStWriter)) {
            return false;
        }
        km::chain_copy(tag, km_direct_words);
        direct = (km_direct_words[kChainInfo] & kChainDataMark) != 0;
        km_direct_tag = tag;
        km_direct_known = true;
        km_direct_is = direct;
        return true;
    }

    // A write of pages from a buffer: when the next record takes it direct, its tiles
    // sent to the next cores' landing areas now (acks waited before the marks), else
    // written to the tensor.
    template <typename T>
    bool km_direct_write(
        km::Movement& movement, const km_command_t& command, const T& tensor,
        uint32_t page, uint32_t pages, uint32_t address
    ) {
        bool direct = false;
        if (!km_direct_next(movement, direct)) {
            return false;
        }
        if (!direct) {
            return km::write_pages(movement, command, tensor, page, pages, address);
        }
        const km::ChainShape shape{};
        const uint32_t* now = km::chain_now.now;
        const uint32_t width = shape.width(movement, now);
        const uint32_t low = (
            (movement.me() - km::first(command)) / shape.size(movement, now) * width
        );
        if (page < low || page + pages > low + width) {  // not of its block: no send
            movement.gave_up(km::kStWriter, km::kGaveBadItem, km::kWhereData, 0xC4u, 0);
            return false;
        }
        const km::ChainDirect area = {
            address, km::chain_area_base(kDirectArea), kDirectAreaTiles,
            kDirectAreaBytes
        };
        km_direct_sent += pages;
        return km::chain_send_writes(
            shape, movement, km_direct_words, page, page + pages, area,
            KM_DIRECT_LANDING
        );
    }

    inline void km_direct_release() {
        km_direct_sent = 0;
        km_direct_known = false;  // the record's end: the next one asks again
    }

    // This core's landing area in a direct record (base 0: read DRAM).
    __attribute__((noinline)) km::DirectArea km_direct_area(km::Movement& movement) {
        const uint32_t* now = km::chain_now.now;
        if (!km::chain_now.on || (now[kChainInfo] & kChainDataMark) == 0) {
            return {0, 0, 0};
        }
        const uint32_t size = km::ChainShape{}.size(movement, now);
        const uint32_t index = movement.me() - (now[kChainGroup] & 0xFFFFu);
        const km::ChainShape shape{};
        const km::ChainRange range = (
            km::chain_need(shape, movement, now, 0, index % size, index / size)
        );
        return {km::chain_area_base(kDirectArea), range.first, kDirectAreaBytes};
    }
"""

DIRECT_END = """#else  // mover B
        const uint32_t size = km::ChainShape{}.size(movement, km::chain_now.now);
        if ((movement.me() - km::first(command)) % size != 0) {
            km_direct_release();
            return true;
        }
        bool direct = false;
        if (!km_direct_next(movement, direct)) {
            km_direct_release();
            return false;
        }
        const uint32_t width = km::ChainShape{}.width(movement, km::chain_now.now);
        if (direct && km_direct_sent != width) {
            km_direct_release();  // its block not sent whole: no marks
            movement.gave_up(km::kStWriter, km::kGaveBadItem, km::kWhereData, 0xC4u, 0);
            return false;
        }
        const km::ChainDirect area = {0, 0, 0, 0};  // sent as written: acks waited
        const bool marked = (
            km::chain_mark<true>(km::ChainShape{}, movement, command, area, false)
        );
        km_direct_release();
        return marked;
#endif"""


def chain_helpers(compiled):
    """The walk's helpers in the movement source; with a direct input's landing area,
    the root's kept block, its send before the marks and the reader's area; with the
    stamps, the record's body end stamped."""
    text = direct_helpers(compiled)
    if getattr(compiled, "chain_stamps", False):
        head = (
            "    ) {\n        if (!km::chain_now.on) {\n"  # km_chain_end's first lines
        )
        at = text.index(head, text.index("bool km_chain_end("))
        text = (
            text[: at + len("    ) {\n")]
            + f"        km::chain_stamp({STAMP_BODY});  // the record's body done\n"
            + text[at + len("    ) {\n") :]
        )
    return text


def stamps_made(program, cls, compiled, walked, chain_stamps, defines):
    """The make's defines with KM_CHAIN_STAMPS for chain_stamps=True (refused without
    the walk); on `compiled` and `program` whether the stamps are built."""
    compiled.chain_stamps = bool(chain_stamps)
    program.chain_stamps = compiled.chain_stamps
    if not chain_stamps:
        return defines
    if not walked:
        raise ProgramError(
            f"{cls.__name__}: chain_stamps=True stamps the chain's walk; make it with "
            f"{spelling('chain')}=True or give the class a km.{spelling('chain')} "
            "method"
        )
    return {**defines, "KM_CHAIN_STAMPS": "1"}


def stamps_readout(compiled):
    """The movers' lines for the stamps' read-out (a chain command with no records,
    a7 0): mover B writes this core's stamps to page `me` of the tensor at a0."""
    if not getattr(compiled, "chain_stamps", False):
        return []
    return [
        f"{INDENT * 2}if (command.a7 == 0) {{  // the stamps' read-out",
        "#if defined(COMPILE_FOR_BRISC)",
        f"{INDENT * 3}return true;",
        "#else",
        f"{INDENT * 3}return km::chain_stamps_out(movement, command.a0, "
        f"{STAMPS_PAGE}u);",
        "#endif",
        f"{INDENT * 2}}}",
    ]


LINEAR_LANDING = "km::LinearLanding{km::chain_area_base(kDirectArea), kDirectAreaBytes}"
IN_PLACE_LANDING = (  # the reading stream's halves (in_place)
    "km::InPlaceLanding{\n"
    "                km::chain_area_base(kDirectIn.mover_a),\n"
    "                km::chain_area_base(kDirectIn.mover_b), kDirectAreaBytes,\n"
    "                kDirectInBlock, kDirectInPattern\n"
    "            }"
)


def direct_helpers(compiled):
    """The walk's helpers with a direct input's landing area (CHAIN_HELPERS without);
    in place, the area the reading stream's halves."""
    if getattr(compiled, "direct_area", None) is None:
        return CHAIN_HELPERS
    helpers = DIRECT_HELPERS.replace("KM_DIRECT_LANDING", LINEAR_LANDING)
    if getattr(compiled, "direct_in", None) is not None:
        helpers = DIRECT_HELPERS.replace("KM_DIRECT_LANDING", IN_PLACE_LANDING)
        helpers = helpers.replace(
            "km::chain_area_base(kDirectArea)", "km::chain_area_base(kDirectIn.mover_a)"
        )
    start = CHAIN_HELPERS.index("#else  // mover B")
    end = CHAIN_HELPERS.index("#endif", start) + len("#endif")
    text = CHAIN_HELPERS[:start] + DIRECT_END + CHAIN_HELPERS[end:]
    text = text.replace(
        "        if (!km::chain_now.on) {\n            return true;\n        }\n#if",
        "        if (!km::chain_now.on) {\n            km_direct_release();\n"
        "            return true;\n        }\n#if",
        1,
    )
    head = "namespace {\n"
    return text.replace(head, head + helpers, 1)


# A direct-declared parameter's read stream, as the class compiler renders it: the
# plain form (first page, tiles, block, repetitions, its pair, its pattern).
IN_PLACE_STREAM = (
    r"km::kMoveRead, {tensor}\({place}\), (.+), (\w+), (kRead\d+)Block, (\w+), "
    r"\3, \3Pattern$"
)


def in_place_prepared(cls, compiled, declared, key, used):
    """A direct input that lands in the halves of the stream reading it (in_place):
    the stream's own pair (kDirectIn, each half the share's blocks it holds, so its
    ring is back at its base at each record's end) in place of a landing area;
    refused by name where that cannot hold."""
    name, place = key
    label = f"{cls.__name__}.{name}: in_place for {place}"
    if len(declared) > 1:
        raise ProgramError(
            f"{label}: the class declares {len(declared)} direct inputs "
            f"({', '.join(f'{h}.{p}' for h, p in declared)}); in place takes one"
        )
    pattern = re.compile(IN_PLACE_STREAM.format(tensor=camel(place), place=place))
    lines = render(prune(compiled.movement[name]), 4)
    found = [m for m in map(pattern.search, lines) if m]
    reads = sum(f"km::kMoveRead, {camel(place)}({place})," in line for line in lines)
    if reads != 1 or len(found) != 1:
        raise ProgramError(
            f"{label}: {place} is read by {reads} streams"
            + ("" if reads != 1 else ", not in pages in order (a rectangle's rows)")
            + "; in place takes one stream of its pages in order"
        )
    tiles, repetitions, pair = found[0].group(2, 4, 3)
    if not tiles.isdigit() or repetitions != "1":
        raise ProgramError(
            f"{label}: its stream's tiles ({tiles}) and repetitions ({repetitions}) "
            "must be constants of the class, one repetition: the halves hold each "
            "record's tiles whole"
        )
    buffer = next(b for b in compiled.buffers if b.name == pair)
    block, tiles = buffer.block, int(tiles)
    if tiles % block != 0 or buffer.page_bytes != compiled.direct_area[1]:
        raise ProgramError(
            f"{label}: {tiles} tiles in blocks of {block}, pages of "
            f"{buffer.page_bytes} B against the tensor's {compiled.direct_area[1]}; "
            "in place takes whole blocks of the tensor's pages"
        )
    (letters,) = buffer.patterns.values()
    blocks = [letters[b % len(letters)] for b in range(tiles // block)]
    pages = tuple(max(blocks.count(half), 1) * block for half in "AB")
    free = [i for i in range(IN_PLACE_LAST) if i not in used]
    if len(free) < 2:
        raise ProgramError(
            f"{label}: the stream's own pair needs two buffers under "
            f"{IN_PLACE_LAST} (the movers' streams), and the class takes "
            f"{IN_PLACE_LAST - len(free)} of them"
        )
    compiled.direct_area = (tiles, buffer.page_bytes, declared)
    compiled.direct_in = (name, place, pair)
    compiled.buffers.append(
        Buffer(
            "kDirectIn",
            "read",
            free[0],
            free[1],
            buffer.dtype,
            buffer.page_bytes,
            pages,
            block=block,
            what=f"{name}'s stream of {place}: a direct input lands in its halves",
            patterns={"kDirectInPattern": letters},
        )
    )


def in_place_case(compiled, name, text):
    """A handler's case text with its in-place input's stream on kDirectIn (the
    stream's pair renamed there; other handlers keep it)."""
    held = getattr(compiled, "direct_in", None)
    if held is None or held[0] != name:
        return text
    return re.sub(rf"\b{held[2]}(Block|Pattern)?\b", r"kDirectIn\1", text)


def direct_reads(compiled, name, lines):
    """A handler's case lines with each read stream of its direct-declared tensors
    through the core's landing area in a direct record (km::direct_or), or its own
    halves (km::in_place)."""
    area = getattr(compiled, "direct_area", None)
    places = [place for handler, place in (area[2] if area else ()) if handler == name]
    for place in places:
        paged = "km::pages_of<"
        if any(paged in line and f"{camel(place)}({place})" in line for line in lines):
            raise ProgramError(
                f"{compiled.name}.{name}: {place} is in pages of several tiles "
                "(page_tiles), and a direct input lands a page a tile: give the "
                "chain's direct input a tensor of page_tiles=1"
            )
    if getattr(compiled, "direct_in", None) is not None:
        for place in places:
            given = f"km::kMoveRead, {camel(place)}({place}),"
            wrapped = (
                f"km::kMoveRead, km::in_place({camel(place)}({place}), "
                "km_direct_area(movement).base != 0),"
            )
            lines = [line.replace(given, wrapped) for line in lines]
        return in_place_case(compiled, name, "\n".join(lines)).split("\n")
    for place in places:
        given = f"km::kMoveRead, {camel(place)}({place}),"
        wrapped = (
            f"km::kMoveRead, km::direct_or({camel(place)}({place}), "
            "km_direct_area(movement)),"
        )
        lines = [line.replace(given, wrapped) for line in lines]
    return lines


def direct_writes(compiled, text):
    """A chain build's cases with a direct input: each write of pages from a buffer
    sent to the next record's cores when it takes it direct (km_direct_write; the
    pages popped once flushed, as a write's)."""
    if getattr(compiled, "direct_area", None) is None:
        return text
    return text.replace(
        "!km::write_pages(movement, command, ", "!km_direct_write(movement, command, "
    )


def chain_cases(compiled):
    """The switch of a chain build's movers: each handler's case with the input's
    waits, its end's marks, and a tree's credit before each emit."""
    reads = chain_reads(compiled)
    pad = INDENT * 2
    lines = [pad + "switch (km::handler(command)) {"]
    for name in compiled.handlers:
        line = compiled.lines[name]
        lines.append(f"{pad}{INDENT}case kHandler{camel(name)}: {{  // line {line}")
        lines.append(f"{pad}{INDENT * 2}km_chain_waited = false;")
        if name in compiled.overlays:
            lines += chain_overlay_case(compiled, name, reads.get(name, {}))
            continue
        body = direct_reads(compiled, name, render(prune(compiled.movement[name]), 4))
        lines += chain_waits(compiled, name, body, reads.get(name, {}))
        lines += [
            f"{pad}{INDENT * 2}return km_chain_end(movement, command);",
            f"{pad}{INDENT}}}",
        ]
    if getattr(compiled, "chain_gather", False):
        lines += gather_case(pad + INDENT)
    lines += [f"{pad}{INDENT}default:", f"{pad}{INDENT * 2}return true;", pad + "}"]
    text = "\n".join(lines)
    emit = "!km::reduce_emit(movement,"
    text = text.replace(emit, "!km::chain_credit(movement) || " + emit)
    return direct_writes(compiled, text).split("\n")


def gather_case(pad):
    """km.all_gather's case (a record naming the walk's handler): on mover B its
    leaders' input, shares and marks (km::chain_gather), nothing on mover A."""
    return [
        f"{pad}case kHandler{camel(CHAIN_HANDLER)}: {{  // km.all_gather's record",
        "#if defined(COMPILE_FOR_BRISC)  // mover A",
        f"{pad}{INDENT}return true;",
        "#else  // mover B",
        f"{pad}{INDENT}return km::chain_gather(km::ChainShape{{}}, movement, command);",
        "#endif",
        f"{pad}}}",
    ]


def chain_overlay_case(compiled, name, reads):
    """An overlay handler's case in a chain build: its overlay loaded (the reader's
    load, the followers' meet, as in a plain build), then the record's input waited
    for whole when the handler reads a tensor a chain call writes (its part cannot
    reach the walk's state), then the part called, then the record's marks."""
    pad = INDENT * 4
    overlay = f"kOverlay{camel(name)}"
    bits = 0
    whole = chain_mask_word(compiled, name) is None
    for line_bits, stated in reads.values():
        bits |= line_bits
        whole = whole or stated is False
    mask = "~0u" if whole else f"command.a{chain_mask_word(compiled, name)}"
    wait = (
        [
            f"if (!km_chain_read(movement, {mask}, {bits}u)) {{",
            f"{INDENT}return false;",
            "}",
        ]
        if reads
        else []
    )
    end = "return km_chain_end(movement, command);"
    if name in compiled.stamps:
        body = [line[len(pad) :] for line in stamped(compiled, name, pad)]
        at = body.index("const uint64_t km_loaded = km::wall64();")
        body[at + 1 : at + 1] = [*wait, "const uint64_t km_waited = km::wall64();"]
        body = [  # the call's cycles from the input's arrival
            line.replace("km_called - km_loaded", "km_called - km_waited")
            for line in body
        ]
        body[-1:] = ["if (!km_called_ok) {", f"{INDENT}return false;", "}", end]
    else:
        body = [
            f"km::Overlay km_part(movement, {overlay});",
            "if (!km_part.load()) {",
            f"{INDENT}return false;",
            "}",
            *(
                [f"km::chain_stamp({STAMP_LOADED});  // the overlay loaded"]
                if getattr(compiled, "chain_stamps", False)
                else []
            ),
            *wait,
            "if (km_part.call(command) == 0) {",
            f"{INDENT}return false;",
            "}",
            end,
        ]
    body = [line if line.startswith("#") else pad + line for line in body]
    return [*body, f"{INDENT * 3}}}"]


def chain_movement_source(compiled):
    """The movement source of a chain build: a chain command's records walked
    (km::chain_reader on mover A, km::chain_writer on mover B), each its handler's
    case; any other command its case as in a plain build."""
    chain = f"kHandler{camel(CHAIN_HANDLER)}"
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// the movement source of both movers (mover A: RISCV_0, mover B: RISCV_1),",
        "// with the chain's walk (km_chain.hpp).",
        *movement_includes(compiled),
        '#include "komorebi/device/movement/km_chain.hpp"',
        f'#include "{header_name(compiled)}"',
        "",
        chain_helpers(compiled),
        "void kernel_main() {",
    ]
    lines += [INDENT + line for line in tensor_aliases(compiled)]
    lines += [
        f"{INDENT}km::Movement movement;",
        "#if defined(COMPILE_FOR_BRISC)  // mover A: the chain's words zero",
        f"{INDENT}km::chain_start();",
        "#endif",
    ]
    lines += kernel_bodies(compiled, "start", "movement")
    lines += [  # one copy of the cases, for a plain command and a chain's records
        f"{INDENT}const auto km_case = [&](const km_command_t& command)",
        f"{INDENT * 3}__attribute__((noinline)) -> bool {{",
    ]
    lines += [
        line[len(INDENT) :] if line.startswith(INDENT) else line
        for line in chain_cases(compiled)
    ]
    lines += [
        f"{INDENT}}};",
        f"{INDENT}movement.run([&](const km_command_t& command) {{",
        f"{INDENT * 2}if (km::handler(command) != {chain}) {{",
        f"{INDENT * 3}return km_case(command);",
        f"{INDENT * 2}}}",
        *stamps_readout(compiled),
        "#if defined(COMPILE_FOR_BRISC)  // mover A",
        f"{INDENT * 2}km::chain_reader(movement, command, km_case);",
        f"{INDENT * 2}return true;",
        "#else  // mover B",
        f"{INDENT * 2}return km::chain_writer(km::ChainShape{{}}, movement, command, "
        "km_case);",
        "#endif",
        f"{INDENT}}});",
        *kernel_bodies(compiled, "end", "movement"),
        "}",
    ]
    return "\n".join(lines) + "\n"


def chain_compute_source(compiled):
    """The compute kernel of a chain build: a chain command's records taken in turn
    (km::chain_compute), each its handler's case."""
    text = compute_source(compiled, walk=False)
    chain = f"kHandler{camel(CHAIN_HANDLER)}"
    head = f"{INDENT}compute.run([&](const km_command_t& command) {{"
    before, after = text.split(head + "\n", 1)
    cases_text, tail = after.rsplit(f"{INDENT}}});\n", 1)
    held = getattr(compiled, "direct_in", None)
    if held is not None:  # its case's stream on kDirectIn, up to the next case
        head_at = cases_text.index(f"case kHandler{camel(held[0])}: {{")
        end_at = cases_text.find("            case kHandler", head_at + 1)
        end_at = len(cases_text) if end_at < 0 else end_at
        cases_text = (
            cases_text[:head_at]
            + in_place_case(compiled, held[0], cases_text[head_at:end_at])
            + cases_text[end_at:]
        )
    before = before.replace(
        f'#include "{header_name(compiled)}"',
        '#include "komorebi/device/movement/km_chain.hpp"\n'
        f'#include "{header_name(compiled)}"',
        1,
    )
    return (
        before
        + f"{INDENT}const auto km_case = [&](const km_command_t& command)\n"
        + f"{INDENT * 3}__attribute__((noinline)) {{\n"
        + cases_text
        + f"{INDENT}}};\n"
        + head
        + "\n"
        + f"{INDENT * 2}if (km::handler(command) == {chain}) {{\n"
        + (
            f"{INDENT * 3}if (command.a7 != 0)  // 0: the stamps' read-out\n"
            if getattr(compiled, "chain_stamps", False)
            else ""
        )
        + f"{INDENT * 3}km::chain_compute(compute, command, km_case);\n"
        + f"{INDENT * 2}}} else {{\n"
        + f"{INDENT * 3}km_case(command);\n"
        + f"{INDENT * 2}}}\n"
        + f"{INDENT}}});\n"
        + tail
    )
