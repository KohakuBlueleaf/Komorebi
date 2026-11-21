"""A class deriving km.Program made into an ordinary program: the class is compiled
(komorebi.dsl.compiler), its files written under build/ beside its source and kept,
the sizes printed, and the made object is a Program of those files.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from komorebi import _env, abi, formats
from komorebi.chain import calls as kc
from komorebi.dsl import compiler, first_use, padding
from komorebi.dsl.chain import chain_calls
from komorebi.dsl.compile.layout import padded_rings
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
from komorebi.dsl.program_sources import (  # noqa: F401 (kept importable from here)
    CHAIN_HANDLER,
    CHAIN_HELPERS,
    COMPUTE_INCLUDES,
    DIRECT_END,
    DIRECT_HELPERS,
    IN_PLACE_LANDING,
    INDENT,
    LINEAR_LANDING,
    PART_MOVEMENT,
    READ_MOVES,
    body_reads,
    cases,
    chain_cases,
    chain_gated,
    chain_helpers,
    chain_mask_word,
    chain_movement_source,
    chain_overlay_case,
    chain_part,
    chain_prefetch,
    chain_reads,
    chain_waits,
    compute_includes,
    direct_helpers,
    direct_reads,
    direct_writes,
    flat_groups,
    flat_rectangle,
    gather_case,
    grid_islands,
    handler_files,
    has_part,
    header_name,
    in_place_case,
    island_rectangles,
    kernel_bodies,
    mixed_formats,
    movement_includes,
    movement_source,
    nocs_needed,
    part_includes,
    parts,
    ranks_rectangle,
    rect_grid_args,
    rectangle_args,
    rectangle_of,
    region_aliases,
    spans,
    stacked_rectangle,
    stamped,
    stamps_readout,
    statement_starts,
    tensor_aliases,
)
from komorebi.dsl.thread import class_thread
from komorebi.dsl.tile import emit
from komorebi.dsl.tile.forms.rows import chunk_cost
from komorebi.dsl.tile.hazards import wormhole as wormhole_forms
from komorebi.dsl.tile.unit import AICLK_MHZ, chip_words, matrix_cost
from komorebi.errors import L1BudgetError, ProgramError
from komorebi.islands import local as island_local
from komorebi.program import description as km_program
from komorebi.program import overlay

BUILD = "build"
# A core's kernels as a CommandTimeout names them, and the CPU a class's wait is on.
CPUS = {"reader": "mover A", "writer": "mover B", "compute": "compute"}
# A give-up of compute at a check, not a wait: its line (kLvLine) is one of compute's
# roles, the code none of a wait's; compute.cpp's comments above it name the check.
WAIT_CODES = ("stop", "limit", "deadline")
COMPUTE_ROLES = ("unpack", "math", "pack")
PYTHON_LINE = re.compile(r"\s*// line (\d+): (.*)$")
HANDLER_CASE = re.compile(r"\s*case kHandler(\w+): \{")
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
        """For each of `cores` (a CommandTimeout's) whose compute gave up at a check,
        first, the check's Python line; for each whose CPU gave up waiting, the
        handlers' Python lines that wait there."""
        checks, named = [], []
        for core in cores:
            for kernel, cpu in CPUS.items():
                gave = core.get(kernel)
                where = gave.get("where") if isinstance(gave, dict) else None
                if not where:
                    continue
                check = self.check(core, gave) if kernel == "compute" else None
                if check:
                    checks.append(f"core {core['core']} {cpu}: {check}")
                    continue
                named.extend(
                    f"core {core['core']} {cpu}: {wait.handler}, line {wait.line}: "
                    f"waiting in {wait.text}"
                    for wait in self.waits
                    if wait.cpu == cpu and wait.where == where
                )
        return checks + named

    def check(self, core, gave):
        """`handler, line N: gave up (code, where) in text` of compute's give-up at a
        check of compute.cpp (its line in the live words, kLvLine), by the `// line N:`
        comment above it; None for a wait's give-up or a line not in compute.cpp."""
        line = (core.get("live") or {}).get("line")
        text = self.files.get("compute.cpp")
        if gave["code"] in WAIT_CODES or not line or text is None:
            return None
        if line["kernel"] not in COMPUTE_ROLES:
            return None
        above = text.split("\n")[: line["line"]]
        said = next((m for s in reversed(above) if (m := PYTHON_LINE.match(s))), None)
        case = next((m for s in reversed(above) if (m := HANDLER_CASE.match(s))), None)
        if said is None or case is None:
            return None
        handler = next(
            (w.handler for w in self.waits if camel(w.handler) == case[1]), case[1]
        )
        return (
            f"{handler}, line {said[1]}: gave up ({gave['code']}, {gave['where']}) "
            f"in {said[2]}"
        )


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
        cls, plan, tensors, room, values, sets, reserved, islands, not found
    )
    if found:
        first_use.reserve(
            cls, compiled, found, plan, tensors, room, values, sets, reserved
        )
    walked = chain_made(self, cls, compiled, chain, tensors)
    defines = stamps_made(self, cls, compiled, walked, chain_stamps, defines)
    if chain_stamps and form == "runtime":
        raise ProgramError(
            f"{cls.__name__}: chain_stamps=True with {spelling('program_form')}="
            "\"runtime\": the chain's stamps are not compiled into a program's parts; "
            'stamp the chain in the kernel form (form="kernel")'
        )
    if compiled.chain_gather:  # km.all_gather's marks in the walk (km_chain.hpp)
        defines = {**defines, "KM_CHAIN_GATHER": "1"}
    if walked and islands[0] > 1:  # a share of K by island, a write by island
        defines = {**defines, "KM_CHAIN_K_ISLANDS": "1", "KM_CHAIN_BY_ISLAND": "1"}
    # what km.merge builds this program's handlers again with
    # (komorebi.dsl.merged)
    self.make_given = {
        "plan": plan, "tensors": tensors, "constants": values, "sets": sets,
        "session": session, "l1": l1, "trace": trace, "depth": depth, "rect": rect,
        "nocs": nocs, "defines": defines, "broadcast_wait": broadcast_wait,
        "guard": guard,
    }  # fmt: skip
    padded = padded_rings(compiled)  # not km.merge's: its compile shares no pair
    if padded:  # read pairs handlers of other blocks share (km_core.hpp ring_pad)
        defines = {**defines, "KM_RING_PADDED": f"{padded:#x}"}
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
    self.reduce_local = compiled.reduce_local
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
    self.writes_every_call = writes_every_call(compiled)
    self.moved_tensors = moved_tensors(compiled)
    for line in report:
        log(line)
    self.placements = placements_of(compiled, tensors)
    self.tile_words = tile_words(compiled)
    place_handlers(compiled, {m.name: [m] for m in compiler.handler_methods(cls)})
    for method in compiler.handler_methods(cls):  # a padded axis read (dsl.padding)
        padding.checked(method.name, method.padding, tensors, built=True)
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
        port_handlers=self.port_handlers,  # the init would set it back to none
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
        number = compiled_overlays[name] + getattr(program, "overlay_shift", 0)
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


def move_direction(move):
    """A move's direction: "read" or "write"."""
    reads = ("read stream", "fill", "broadcast's read", "word read")
    return "read" if move.how in reads else "write"


def writes_every_call(compiled):
    """{handler: ((tensor, line, text), ...)}: the writes of each handler that run at
    every call of it (a far write refused on the host: islands.local)."""
    out = {}
    for move in compiled.moves:
        if move.every and move_direction(move) == "write":
            out.setdefault(move.handler, []).append((move.tensor, move.line, move.text))
    return {handler: tuple(writes) for handler, writes in out.items()}


def moved_tensors(compiled):
    """{handler: (tensors it reads, tensors it writes)}: the names its lines move pages
    of, at any call (a command's order behind another at submit: session.run)."""
    out = {}
    for move in compiled.moves:
        reads, writes = out.setdefault(move.handler, (set(), set()))
        (reads if move_direction(move) == "read" else writes).add(move.tensor)
    return {name: (frozenset(r), frozenset(w)) for name, (r, w) in out.items()}


def may_cross(compiled, tensors):
    """The "may cross" report of a program made with island tensors (their
    placement): each move of each handler, whether a page of it can be another
    island's, and what it lowers to then; nothing on one island."""
    if not any(hasattr(tensors.get(name), "placement") for name in compiled.tensors):
        return []
    lines = ["may cross (each move: its tensor's placement; what it lowers to):"]
    for move in compiled.moves:
        placement = tensors[move.tensor].placement
        direction = move_direction(move)
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
    facts = padding.facts_of(compiled, methods)
    for name, objects in methods.items():
        mine = {p: value for (h, p), value in placed.items() if h == name}
        paged = {p: k for (h, p), k in compiled.page_tiles.items() if h == name}
        for method in objects:
            method.placed = mine
            method.paged = paged
            method.padding = facts[name]


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


def session_nocs(program, nocs, island=None, ranks=None):
    """Session.load's hand-over: the session's NoCs, unless the class was made with
    its own; on islands (`island`, the load's (graph, i)), its first flat rank; the
    load's ranks when fewer than the grid's cores (ranks_rectangle)."""
    state = getattr(program, "rectangle", None)
    if state is not None and state["nocs"] is None:
        state["nocs"] = nocs
    if state is not None:
        state["flat"] = island[0].first(island[1]) if island else None
    if state is not None and ranks is not None:
        state["ranks"] = ranks if ranks < nocs.grid[0] * nocs.grid[1] else None


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
    on the mover's runtime, called by each mover's km_program; a chain build's walk
    of a chain command's records inside it, as the kernel form's."""
    walk = runtime_movement_walk(compiled) if compiled.chain else None
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// its parts on the runtime's reader (RISCV_0) and writer (RISCV_1).",
        *movement_includes(compiled),
        '#include "komorebi/device/core/km_program.hpp"',
        *(['#include "komorebi/device/movement/km_chain.hpp"'] if walk else []),
        f'#include "{header_name(compiled)}"',
        "",
        *([chain_helpers(compiled)] if walk else []),
        "static bool km_handlers(",
        f"{INDENT}km::MoverRuntime& km_core, const km_command_t& command",
        ") {",
        *PART_MOVEMENT,
        *(INDENT + line for line in tensor_aliases(compiled)),
        *(walk or cases(compiled, compiled.movement, "return true;", 1, "movement")),
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


CHAIN_CASE = re.compile(r"^( *)case kHandler\w+: \{")
CASE_CALL = "return [&]() __attribute__((noinline)) -> bool {"


def split_cases(lines):
    """Each case of a chain switch as its own function (a lambda not inlined), so the
    walk's frame on the writer's small stack holds one case's locals, not them all."""
    out, closing = [], None
    for line in lines:
        found = CHAIN_CASE.match(line)
        if found:
            pad = found.group(1)
            closing = pad + "}"
            out += [line, f"{pad}{INDENT}{CASE_CALL}"]
        elif closing is not None and line == closing:
            out += [f"{closing[:-1]}{INDENT}}}();", line]
            closing = None
        else:
            out.append(line)
    return out


def runtime_movement_walk(compiled):
    """km_handlers' body in a chain build of the runtime's form: the cases (with the
    input's waits and the marks) as one function, a chain command's records walked
    through it (km::chain_reader on the reader, km::chain_writer on the writer)."""
    chain = f"kHandler{camel(CHAIN_HANDLER)}"
    return [
        f"{INDENT}const auto km_case = [&](const km_command_t& command)",
        f"{INDENT * 3}__attribute__((noinline)) -> bool {{",
        *(
            line[len(INDENT) :] if line.startswith(INDENT) else line
            for line in split_cases(chain_cases(compiled))
        ),
        f"{INDENT}}};",
        f"{INDENT}if (km::handler(command) != {chain}) {{",
        f"{INDENT * 2}return km_case(command);",
        f"{INDENT}}}",
        "#if defined(COMPILE_FOR_BRISC)  // the reader",
        f"{INDENT}km::chain_reader(movement, command, km_case);",
        f"{INDENT}return true;",
        "#else  // the writer",
        f"{INDENT}return km::chain_writer(km::ChainShape{{}}, movement, command, "
        "km_case);",
        "#endif",
    ]


def runtime_compute_source(compiled):
    """The compute source in the runtime's form: the handler switch as compute's
    km_program (the start-up is the runtime's); a chain build's records taken in
    turn through it (km::chain_compute), as the kernel form's."""
    depth = 2 if compiled.chain else 1  # the chain's: inside km_case
    switch = cases(compiled, compiled.compute, "break;", depth, "compute")
    if compiled.chain:
        chain = f"kHandler{camel(CHAIN_HANDLER)}"
        text = in_place_compute("\n".join(switch) + "\n", compiled)
        switch = [
            f"{INDENT}const auto km_case = [&](const km_command_t& command)",
            f"{INDENT * 3}__attribute__((noinline)) {{",
            *text.rstrip("\n").split("\n"),
            f"{INDENT}}};",
            f"{INDENT}if (km::handler(command) == {chain}) {{",
            f"{INDENT * 2}km::chain_compute(compute, command, km_case);",
            f"{INDENT}}} else {{",
            f"{INDENT * 2}km_case(command);",
            f"{INDENT}}}",
        ]
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// its part on compute, under the runtime's compute kernel.",
        *compute_includes(compiled),
        '#include "api/compute/reconfig_data_format.h"',
        '#include "komorebi/device/core/km_program.hpp"',
        *(
            ['#include "komorebi/device/movement/km_chain.hpp"']
            if compiled.chain
            else []
        ),
        f'#include "{header_name(compiled)}"',
        "",
        "void km_program(km::Compute& compute, const km_command_t& command) {",
        *start_formats(compiled),
        *switch,
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
    walk_refused(compiled, files, form)
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
    if not dst_is_fp32(compiled):  # what km.fp32 in the class's sums means here
        a = [b.dtype for b in compiled.buffers if formats.FAMILY.get(b.dtype) == "a"]
        family = sorted(set(a))
        report.append(
            f"  Dst is 16-bit (buffers of {', '.join(family)}, the A family): a sum "
            f"held in Dst (`km.tile.zeros(km.fp32)` and `+=`, a product's) is float16, "
            "11 significant bits and at most 65504, whatever km.fp32 says"
        )
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
        "row form, by the tile compiler's cost model, and us at the running "
        f"{AICLK_MHZ:,} MHz; a wide row's float32 copies into DST priced, a tile's "
        "copies and the pack not, nor a step's turn: 87 cycles after a step of "
        "another state row, 169 after one of the same row):"
    ]
    for handler, line, text, function, cxx in compiled.tile_calls:
        key, unit = compiled.units[cxx]
        width = key[2]  # (function, formats, width[, the operands' DST tiles])
        count, cycles, _ = unit.cost("math")
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
        if width > 2:  # a wide row: the chunk parts once a chunk, its copies a pass
            residents = len(operands) - len(unit.operands)
            count, cycles, copies = chunk_cost(unit, width, residents)
            each += f" (its float32 copies into DST {copies} of them)"
        each += f", {cycles / AICLK_MHZ:.2f} us"
        held = compiled.held_calls.get((handler, line))
        if held:  # broadcast fields kept in DST: loaded twice a stream
            each += f"; its {held} field broadcast(s) held in DST, 2 loads a stream"
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


def chain_methods(cls):
    """The class's km.chain methods (its parents' too), by name."""
    return {
        name: value
        for klass in reversed(cls.__mro__)
        for name, value in vars(klass).items()
        if isinstance(value, kc.ChainMethod)
    }


def tile_words(compiled):
    """Each tile function's math words for the load's check of its chip: [(the function
    and where it is called, its straight-line runs of words (unit.chip_words))]."""
    called = {}
    for handler, line, _, function, cxx in compiled.tile_calls:
        name = getattr(function.function, "__name__", cxx)
        called.setdefault(cxx, f"{name} (in {handler}, line {line})")
    for _, unit in compiled.units.values():
        wormhole_forms.forms(unit)  # the words a Wormhole core runs
    return [
        (
            called.get(cxx, cxx),
            [
                run
                for function in (
                    unit.init["math"],
                    unit.tile["math"],
                    *unit.parts.values(),
                )
                for run in chip_words(function)
            ],
        )
        for cxx, (_, unit) in compiled.units.items()
    ]


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


DIRECT_LAST = 27  # the landing area's buffer: the first index free from here down
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
                if island_local.one_island(tensors[place]) == -1:
                    raise ProgramError(
                        f"{cls.__name__}.{name}: its direct input {place} is split "
                        "over islands: a direct input is read on one island; place "
                        "it on one island (a group inside it) or as a copy, or leave "
                        "km.direct out on two cards"
                    )
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
    # built for islands: the link cores' own buffer index is theirs (KM_LINK_CB)
    links = {abi.headers()["KM_LINK_CB"]} if compiled.islands[0] > 1 else set()
    index = next((i for i in range(DIRECT_LAST, -1, -1) if i not in used | links), None)
    if index is None:
        raise ProgramError(
            f"{cls.__name__}: no buffer index from {DIRECT_LAST} down is free for the "
            "direct inputs' landing area; drop direct= or a buffer"
        )
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


# What each kernel of a chain build calls to walk a chain command's records.
WALK_CALLS = {
    "movement.cpp": ("km::chain_reader(", "km::chain_writer("),
    "compute.cpp": ("km::chain_compute(",),
}


def walk_refused(compiled, files, form):
    """Refuses a chain build whose kernels have no walk: its chain command would
    fall to the switch's default and be done with nothing run."""
    if not compiled.chain:
        return
    for name, calls in WALK_CALLS.items():
        lacking = [call for call in calls if call not in files.get(name, call)]
        if lacking:
            raise ProgramError(
                f"{compiled.name}: a chain build (chain=True or a km.chain method) "
                f'whose {name} in {spelling("program_form")}="{form}" has no '
                f"chain walk ({', '.join(c + ')' for c in lacking)}): a chain "
                "command would be done with no record run; submit the calls as "
                "commands"
            )


def in_place_compute(cases_text, compiled):
    """Compute's cases with an in-place direct input's case on kDirectIn (its case
    up to the next one); the cases unchanged without one."""
    held = getattr(compiled, "direct_in", None)
    if held is None:
        return cases_text
    head_at = cases_text.index(f"case kHandler{camel(held[0])}: {{")
    end_at = cases_text.find("            case kHandler", head_at + 1)
    end_at = len(cases_text) if end_at < 0 else end_at
    return (
        cases_text[:head_at]
        + in_place_case(compiled, held[0], cases_text[head_at:end_at])
        + cases_text[end_at:]
    )


def chain_compute_source(compiled):
    """The compute kernel of a chain build: a chain command's records taken in turn
    (km::chain_compute), each its handler's case."""
    text = compute_source(compiled, walk=False)
    chain = f"kHandler{camel(CHAIN_HANDLER)}"
    head = f"{INDENT}compute.run([&](const km_command_t& command) {{"
    before, after = text.split(head + "\n", 1)
    cases_text, tail = after.rsplit(f"{INDENT}}});\n", 1)
    cases_text = in_place_compute(cases_text, compiled)
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
