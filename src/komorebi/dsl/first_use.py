"""A handler's constant set compiled on its first use (km.ANY at the make): at
run.submit, the class compiler makes it the handler's overlay parts on the load's
buffers, run.addon adds them as a handler; a later submit finds it added, a later
process its files under build/ and its parts in the parts' cache."""

import hashlib
import importlib
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path

from komorebi import overlay
from komorebi.dsl import compiler, front
from komorebi.dsl.names import spelling
from komorebi.errors import L1BudgetError, ProgramError

# Bytes of each kernel part's slot the opt-in reserves for the added handlers' parts
# (Program.slots(bytes=) after the make gives other bytes; the load checks them).
SLOT_BYTES = 16 * 1024
# The file of a set's build folder that holds what a later process needs of it.
KEPT = "first_use.json"


@dataclass
class Kept:
    """What a made program keeps for its handlers' first uses: the opted handlers'
    km.ANY, the make's inputs (plan, tensors, room, constants, listed sets), the load's
    buffers {index: Buffer} and fields {name: Field}, its build's key, the sets compiled
    so far {(handler, set): (folder, variant, seconds, cached)}, the indices it left."""

    opted: dict
    plan: dict
    tensors: dict
    room: int
    values: dict
    sets: dict
    buffers: dict
    fields: dict
    key: str
    made: dict
    reserved: frozenset = frozenset()


def opted(cls, handlers):
    """(the make's `handlers` without km.ANY, {handler: km.ANY}): a handler given
    km.ANY alone lists no set (its default is built), one with sets too lists them."""
    if not isinstance(handlers, dict):
        return handlers, {}
    plain, found = {}, {}
    for name, listed in handlers.items():
        if isinstance(listed, front.FirstUse):
            found[name], listed = listed, []
        elif isinstance(listed, list | tuple):
            anys = [one for one in listed if isinstance(one, front.FirstUse)]
            if len(anys) > 1:
                raise ProgramError(
                    f"{cls.__name__}.{name}: {spelling('handlers')}= gives "
                    f"km.{spelling('first_use')} {len(anys)} times; give it once"
                )
            if anys:
                found[name] = anys[0]
                listed = [one for one in listed if not isinstance(one, front.FirstUse)]
        plain[name] = listed
    return plain, found


def template(cls, name, sets):
    """The set whose buffers a first use of handler `name` runs on: its first listed
    set, or its defaults."""
    if sets.get(name):
        return dict(sets[name][0])
    method = {m.name: m for m in compiler.handler_methods(cls)}[name]
    parameters = front.constant_parameters(method.function)
    return {n: default for n, (_, default) in parameters.items()}


def reserve(cls, compiled, found, plan, tensors, room, values, sets, reserved=()):
    """The load's buffers of each opted handler's template also sized for the set
    km.ANY(p=n) names (its template with those values): each buffer's pages the
    larger of the two compiles'; refused when the buffers no longer fit `room`."""
    made = importlib.import_module("komorebi.dsl.program")  # it imports this
    for name, first_use in found.items():
        if not first_use.reserve:
            continue
        method = {m.name: m for m in compiler.handler_methods(cls)}[name]
        parameters = front.constant_parameters(method.function)
        unknown = sorted(set(first_use.reserve) - set(parameters))
        if unknown:
            raise ProgramError(
                f"{cls.__name__}.{name}: km.{spelling('first_use')}("
                f"{', '.join(unknown)}=...) names no constant parameter of it "
                f"({', '.join(parameters)})"
            )
        wanted = template(cls, name, sets)
        for parameter, value in first_use.reserve.items():
            kind = parameters[parameter][0]
            wanted[parameter] = made.constant_checked(
                cls, f"{name}'s {parameter}", kind, value
            )
        other = compiler.compile_class(
            cls, plan, tensors, room, values, replaced(sets, name, wanted), reserved
        )
        sized = {buffer.index_a: buffer for buffer in other.buffers}
        for buffer in compiled.buffers:
            larger = sized.get(buffer.index_a)
            if larger is not None and larger.pages > buffer.pages:
                buffer.pages = tuple(
                    max(a, b) for a, b in zip(buffer.pages, larger.pages)
                )
    if room is not None:
        used = sum(b.page_bytes * sum(b.pages) for b in compiled.buffers)
        if used > room:
            raise ProgramError(
                f"{cls.__name__}: the buffers sized for km.{spelling('first_use')}'s "
                f"sets take {used:,} B, and {room:,} B are left for buffers"
            )


def replaced(sets, name, wanted):
    """The listed sets with handler `name`'s first (its template) replaced by
    `wanted`, so `wanted` is compiled at the template's number and buffers."""
    listed = list(sets.get(name, []))
    listed[:1] = [wanted]
    return {**sets, name: listed}


def keyed(found):
    """What a build's key holds of the opt-in: each opted handler and its reserve."""
    return sorted((name, sorted(one.reserve.items())) for name, one in found.items())


def kept(program, compiled, found, made_with, built, session, l1, trace):
    """Keeps on `program` what its first uses need (program.first_use, a Kept),
    names each kernel part's slot bytes (KM_ADDED_HANDLERS: the kernels' path of an
    added handler) and, with a session, refuses at the make a program whose L1 does
    not hold those slots beside its buffers (L1BudgetError, with the bytes)."""
    plan, tensors, room, values, sets, reserved = made_with
    program.first_use = Kept(
        found,
        dict(plan),
        dict(tensors),
        room,
        dict(values),
        {name: [dict(one) for one in listed] for name, listed in sets.items()},
        {buffer.index_a: buffer for buffer in compiled.buffers},
        dict(compiled.fields),
        built.folder.name,
        {},
        frozenset(reserved),
    )
    program.slots(bytes=SLOT_BYTES)
    if session is not None:
        afforded(program, session, l1, trace)


def afforded(program, session, l1, trace):
    """The load's L1 table with the added handlers' slots: each kernel part's
    SLOT_BYTES (the reader's beside the service part's, the larger); refused as the
    load refuses it."""
    made = importlib.import_module("komorebi.dsl.program")  # it imports this
    one, most = made.reserve_of(session)
    l1 = one.l1()[1] if l1 is None else l1
    rows = one.l1_rows(program, trace)
    rows.append(
        (f"slots: reader, 1 x {max(SLOT_BYTES, most):,} B", max(SLOT_BYTES, most))
    )
    for part in overlay.PARTS[1:]:
        rows.append((f"slots: {part}, 1 x {SLOT_BYTES:,} B (added)", SLOT_BYTES))
    table, fits = overlay.budget(l1, rows)
    if not fits:
        held = sum(size for _, size in one.l1_held())
        raise L1BudgetError(
            f"{program.name} (km.{spelling('first_use')}: the added handlers' slots)",
            table,
            held,
        )


def handler(run, program, method, wanted):
    """(number, built method) of `method` at the set `wanted` for a submit to `run`:
    added to the run before, or compiled (or found built by an earlier process) and
    added now by run.addon as a handler. Refused without a run (a submit path that
    does not reach the first use) and for a handler not opted in."""
    kept_ = program.first_use
    asked = ", ".join(f"{k}={v}" for k, v in wanted.items())
    where = f"{type(program).__name__}.{method.name} with {asked}"
    if run is None:
        raise ProgramError(
            f"{where}: no handler was built for it, and a first use "
            f"(km.{spelling('first_use')}) is reached from run.submit only, not from "
            "this submit; submit it once with run.submit first, or list the set"
        )
    folder, variant = compiled(program, method, wanted)
    name = f"{kept_.key}.{variant.name}"
    added = getattr(run, "added", {}).get(name)
    if added is not None:
        return added.number, variant
    sources = {"reader": folder / variant.name / "movement.cpp"}
    sources["writer"] = sources["reader"]
    if (folder / variant.name / "compute.cpp").is_file():
        sources["compute"] = folder / variant.name / "compute.cpp"
    number = run.addon(name, sources, handler=True, **added_with(program))
    return number, variant


def added_with(program):
    """run.addon's keywords for a set of `program`: its defines; in the runtime's
    form (form="runtime") also its compile arguments and its args' place, as its own
    parts are built (the kernel form's added parts take the loaded kernels')."""
    defines = dict(program.defines)
    if program.form() != "program":
        return {"defines": defines or None}
    if program.args_at:
        defines["KM_PROGRAM_ARGS_AT"] = str(program.args_at)
    compile_args = {k: program.kernel_compile_args(k) for k in overlay.KERNELS}
    if not any(compile_args.values()):
        return {"defines": defines or None}
    return {"defines": defines or None, "compile_args": compile_args}


def compiled(program, method, wanted):
    """(folder, variant) of `method` at `wanted` on `program`'s load: compiled once a
    program, its files kept in build/<class>-<key>/ (the key: the load's build, the
    handler, the set, the class's source); a folder an earlier process wrote is
    taken as it is."""
    kept_ = program.first_use
    cls = type(program)
    marker = (method.name, tuple(sorted(wanted.items())))
    if marker in kept_.made:
        return kept_.made[marker][:2]
    started = time.perf_counter()
    folder = folder_of(program, method, wanted)
    variant = found_built(cls, method, wanted, folder)
    cached = variant is not None
    if not cached:
        variant = compile_set(program, method, wanted, folder.name.split("-")[-1])
    seconds = time.perf_counter() - started
    kept_.made[marker] = (folder, variant, seconds, cached)
    log = getattr(program, "log", print)
    log(
        f"first use of {cls.__name__}.{variant.name}: "
        + ("found built" if cached else "compiled")
        + f" in {seconds * 1e3:.1f} ms ({folder.name})"
    )
    return folder, variant


def folder_of(program, method, wanted):
    """The build folder of `method` at `wanted` on `program`'s load, where the class's
    build writes (beside its first handler's source): build/<class>-<folder_key>."""
    cls = type(program)
    methods = compiler.handler_methods(cls)
    files = sorted(
        {inspect.getsourcefile(cls)}
        | {inspect.getsourcefile(m.function) for m in methods}
    )
    sources = [Path(name).read_bytes() for name in files]
    key = folder_key(program.first_use.key, method.name, wanted, sources)
    folder = Path(inspect.getsourcefile(methods[0].function)).resolve().parent
    return folder / "build" / f"{cls.__name__.lower()}-{key}"


def folder_key(load, name, wanted, sources):
    """A short hash of what a first use's files depend on: the load's build key, the
    handler, its set of constants, the class's source files."""
    digests = [hashlib.sha256(source).hexdigest() for source in sources]
    made = [load, name, sorted(wanted.items()), digests]
    return "use" + hashlib.sha256(repr(made).encode()).hexdigest()[:8]


def found_built(cls, method, wanted, folder):
    """The variant of a set an earlier process compiled into `folder`, or None."""
    path = folder / KEPT
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if not (folder / data["name"] / "movement.cpp").is_file():
        return None
    return variant_of(cls, method, wanted, data)


def variant_of(cls, method, wanted, data):
    """The built method a set's kept file describes: its name, words, widths and
    whether it returns a word."""
    original = {m.name: m for m in compiler.handler_methods(cls)}[method.name]
    variant = front.HandlerMethod(original.function, True, original.stamps)
    variant.owner = cls
    variant.name = data["name"]
    variant.constant_set = dict(wanted)
    variant.words = [tuple(word) for word in data["words"]]
    variant.widths = dict(data["widths"])
    variant.returns = data["returns"]
    return variant


def compile_set(program, method, wanted, key):
    """Compiles `program`'s class with `wanted` in place of the handler's template
    set (so at its number, its buffers' indices), checks its buffers and fields are
    the load's, writes its files (the handler's overlay parts among them) into
    build/<class>-<key>/; returns its variant."""
    made = importlib.import_module("komorebi.dsl.program")  # it imports this
    kept_ = program.first_use
    cls = type(program)
    sets = replaced(kept_.sets, method.name, wanted)
    result = compiler.compile_class(
        cls, kept_.plan, kept_.tensors, kept_.room, kept_.values, sets, kept_.reserved
    )
    variant = result.variants[method.name][0][1]
    where = f"{cls.__name__}.{variant.name}"
    if variant.stamps is not None or variant.name in result.broadcast_lines:
        raise ProgramError(
            f"{where}: a handler with {spelling('stamps')}= or a broadcast is "
            "resident only; list its sets at the make"
        )
    load_checked(kept_, result, variant, where)
    result.overlays[variant.name] = len(result.overlays) + 1
    built = made.write(cls, result, kept_.room, key)
    data = {
        "name": variant.name,
        "words": variant.words,
        "widths": variant.widths,
        "returns": variant.returns,
    }
    (built.folder / KEPT).write_text(json.dumps(data, indent=1) + "\n")
    return variant_of(cls, method, wanted, data)


def load_checked(kept_, result, variant, where):
    """Refuses a set whose handler would need a buffer the load does not have, or one
    of other format, pages or more pages than the load's (or a count its pages are
    not a multiple of: a block must not wrap), or a field of another shape."""
    for name, field_ in result.fields.items():
        loaded = kept_.fields.get(name)
        if loaded is not None and loaded != field_:
            raise ProgramError(
                f"{where}: field {name} would be {field_.shape}, and the load's is "
                f"{loaded.shape}: the L1 layout is the load's; list the set at the "
                "make"
            )
    for buffer in result.buffers:
        loaded = kept_.buffers.get(buffer.index_a)
        why = None
        if loaded is None or loaded.name != buffer.name:
            why = "the load has no such buffer"
        elif (loaded.dtype, loaded.page_bytes, loaded.index_b) != (
            buffer.dtype,
            buffer.page_bytes,
            buffer.index_b,
        ):
            why = (
                f"its pages would be {buffer.dtype} of {buffer.page_bytes} B, the "
                f"load's are {loaded.dtype} of {loaded.page_bytes} B"
            )
        elif any(
            new > old or old % new
            for new, old in zip(buffer.pages, loaded.pages)
            if new
        ):
            why = (
                f"it would hold {' + '.join(map(str, buffer.pages))} pages, the "
                f"load's holds {' + '.join(map(str, loaded.pages))}"
            )
        if why is not None:
            raise ProgramError(
                f"{where}: buffer {buffer.name} (index {buffer.index_a}): {why}; the "
                "L1 layout is the load's: list the set at the make, or make it with "
                f"km.{spelling('first_use')}(...) naming the largest set"
            )
