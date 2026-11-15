"""A class added to a live run (run.addon(program)): one more program of a run loaded
as a dict, its parts built against the loaded kernels on its own format tables, its
record in DRAM; run.load gives a group its buffers, in the switch area the load keeps
(by default what its L1 table leaves, less KEEP; session.load(buffers=N) states it)."""

import dataclasses
import hashlib
import importlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from komorebi import abi, added, formats, native, overlay, programs
from komorebi.errors import ProgramError
from komorebi.program import Program, check_families

# Bytes of the L1 left that the switch area does not take: L1 tensors the session
# makes while the run is live (placed from the top of L1 down).
KEEP = 64 * 1024
# The build's tables a buffer's format sets (tt-metal's chlkc_descriptors.h), one entry
# a buffer index: the four format tables, then the two tile sizes.
TABLES = (
    "unpack_src_format",
    "unpack_dst_format",
    "pack_src_format",
    "pack_dst_format",
)
SIZES = ("unpack_tile_size", "pack_tile_size")
DESCRIPTORS = "chlkc_descriptors.h"


def spare_fields(placed):
    """The program fields an added class may take in a run loaded as a dict: every
    field after the loaded ones whose overlay number stays above the programs' own
    overlays (field p is overlay kOverlayService - 1 - p), each a record row."""
    service = overlay.constants()["kOverlayService"]
    return range(len(placed.programs), service - 1 - _own_end(placed))


def _own_end(placed):
    """The highest overlay number the load's programs' own overlays take, and those of
    the classes named ahead (numbered after them, in order: ahead_programs)."""
    last = placed.programs[-1]
    shift = last.overlay_shift + max(
        (each.number for each in last.overlays.values()), default=0
    )
    for other in _ahead_of(placed):
        shift += max((each.number for each in other.overlays.values()), default=0)
    return shift


def buffers_given(buffers):
    """session.load's `buffers=`: None (the area grows into the L1 left) or bytes."""
    if buffers is not None and (not isinstance(buffers, int) or buffers < 0):
        raise ValueError(f"buffers={buffers!r}: bytes, 0 or more, or None")
    return buffers


def ahead_programs(placed):
    """The programs a load's programs named ahead (Program.ahead(program)), each given
    the program field it takes when added, after the loaded ones."""
    out = _ahead_of(placed)
    last = placed.programs[-1]
    shift = last.overlay_shift + max(
        (each.number for each in last.overlays.values()), default=0
    )
    for field, other in enumerate(out, start=len(placed.programs)):
        other.program_field, other.overlay_shift, other.args_at = field, shift, 0
        other.added_own_shift = shift  # its own overlays' numbers, kept by the add
        shift += max((each.number for each in other.overlays.values()), default=0)
    return out


def ahead_own_kernels(placed):
    """The kernels where a class named ahead has a part of an own overlay (the load
    gives them a slot more: pay for use)."""
    return {
        kernel
        for other in _ahead_of(placed)
        for each in other.overlays.values()
        for kernel in each.sources
    }


def _ahead_of(placed):
    out = []
    for program in placed.programs:
        for other in getattr(program, "added_ahead_programs", ()):
            if other not in out and other not in placed.programs:
                out.append(other)
    return out


def own_overlays(program):
    """An added class's own overlays as the run numbers them (programs.own_overlays),
    with the defines its parts read their args from L1 by when it has args."""
    given = {
        name: program.defines[name]
        for name in ("KM_ARGS_L1", "KM_ARGS_WRITER", "KM_ARGS_COMPUTE")
        if name in program.defines
    }
    return [
        dataclasses.replace(each, defines={**each.defines, **given})
        for each in programs.own_overlays(program)
    ]


def args_defines(program, cores):
    """Gives an added class with per-core args (a rectangle, a broadcast, self.x) the
    defines its parts read them from L1 by (KM_ARGS_L1: the record word that holds the
    row's address, from the control block; the writer's and compute's offsets in the
    row) and returns ([row words a core], row bytes); ([], 0) without args."""
    if program.args is None:
        return [], 0
    rows = [program.core_args(core) for core in range(cores)]
    widths = [max((len(row[k]) for row in rows), default=0) for k in range(3)]
    c = overlay.constants()
    program.defines = {
        **program.defines,
        "KM_ARGS_L1": str(
            c["kOverlayArea"] + c["kOverlayRecord"] + 4 * c["kRecordArgsL1"]
        ),
        "KM_ARGS_WRITER": str(widths[0]),
        "KM_ARGS_COMPUTE": str(widths[0] + widths[1]),
    }
    words = [
        [*r, *[0] * (widths[0] - len(r)), *w, *[0] * (widths[1] - len(w)), *k]
        for r, w, k in rows
    ]
    row_bytes = -(-4 * max(sum(widths), 1) // programs.CB_ALIGN) * programs.CB_ALIGN
    return words, row_bytes


def layout_of(backend, program):
    """({index: four words}, end) of `program`'s buffers in the switch area."""
    session = importlib.import_module("komorebi.session")
    ordered = sorted(program.buffers, key=lambda buffer: buffer.index)
    start = backend.switch_start + backend.switch_shift
    return programs.buffer_layout(session._program_cbs(ordered), start)


def pad_reserve(backend, placed, reserve):
    """The end of a switch run's buffers: the largest program's, or one named ahead's,
    then one alignment more, so every group has the area's buffer (switch_pad, one
    index on every group) that grow_area resizes."""
    ahead = ahead_programs(placed)
    used = {b.index for p in [*placed.programs, *ahead] for b in p.buffers}
    used |= {cb[0] for cb in backend._runtime_cbs()}
    # the highest free index: a class made later takes the low ones first
    backend.switch_pad = max(set(range(programs.SWITCH_BUFFERS)) - used)
    backend.switch_loaded = len(placed.programs)
    backend.added_own = set()  # the overlay numbers added classes' own overlays took
    # the entry command's code in the service overlay, for a class ahead with its own
    backend.added_entry = bool(ahead_own_kernels(placed))
    # the load command's row read is built only for a class named ahead with args
    backend.switch_args = any(other.args is not None for other in ahead)
    for other in ahead:
        backend.session._hand_nocs(other)
        _, row_bytes = args_defines(other, placed.cores)
        reserve = max(reserve, _args_at(layout_of(backend, other)[1]) + row_bytes)
    align = programs.CB_ALIGN
    return -(-reserve // align) * align + align


def _args_at(end):
    """Where an added class's per-core row goes: after its buffers, aligned."""
    return -(-end // programs.CB_ALIGN) * programs.CB_ALIGN


def ahead_overlays(backend, placed, number, recipes):
    """The parts of the classes named ahead, built with group 0's kernels and each
    class's own format tables (own_recipes), so the slots are sized by them too."""
    if not placed.switch or number != 0:
        return []
    run = SimpleNamespace(backend=backend, programs=list(placed.programs))
    by_part = {overlay.TARGETS[target]: recipe for target, recipe in recipes.items()}
    out = []
    for other in ahead_programs(placed):
        mine = own_recipes(run, other, by_part)
        made = [programs.program_overlay(other, True), *own_overlays(other)]
        out += [_Own(each, mine) for each in made]
    return out


class _Own:
    """A class's overlay whose parts are built with its own recipes (build_overlays
    calls build_part when an overlay has one)."""

    def __init__(self, made, recipes):
        self.made, self.recipes = made, recipes

    def __getattr__(self, name):
        return getattr(self.made, name)

    def build_part(self, recipe, kernel, part, cache, size):
        own = self.recipes.get(part, recipe)
        return overlay.build_part_cached(own, self.made, kernel, part, cache, size)[0]


def build_ahead(session, backend, room, placed):
    """At a load: the parts of the classes named ahead built against group 0's kernels
    and fitted slots into the parts' cache, so the add finds them built."""
    run = SimpleNamespace(backend=backend, programs=list(placed.programs))
    for other in ahead_programs(placed) if room is not None else ():
        started = time.perf_counter()
        group = _own_group(run, other, room.groups[0])
        hits = [
            added._part(session, group, made, kernel, part)[1]
            for made in [programs.program_overlay(other, True), *own_overlays(other)]
            for kernel, part in made.parts()
        ]
        session.log(
            f"ahead {other.name}: {len(hits)} parts in "
            f"{time.perf_counter() - started:.2f} s ({sum(hits)} built before)"
        )


def _own_group(run, program, group):
    """Group `group` with the added class's own recipes (own_recipes)."""
    return dataclasses.replace(group, recipes=own_recipes(run, program, group.recipes))


def _table(text, name):
    """The values of every table `name` in a descriptors file (each copy alike)."""
    found = re.findall(rf"\b{name}\[(\d+)\] = \{{\s*([\d,\s]*?)\s*\}};", text)
    return [int(value) for value in found[0][1].split(",")] if found else None


def _with_table(text, name, values):
    pattern = rf"(\b{name}\[\d+\] = \{{\s*)([\d,\s]*?)(\s*\}};)"
    joined = ",".join(str(value) for value in values)
    return re.sub(pattern, lambda m: m.group(1) + joined + m.group(3), text)


def _descriptors(recipe):
    """The folder of the kernel build's descriptors file among `recipe`'s includes."""
    words = list(recipe.includes)
    for index, word in enumerate(words):
        folder = word[2:] if word.startswith("-I") and len(word) > 2 else None
        if word == "-I" and index + 1 < len(words):
            folder = words[index + 1]
        if folder and not Path(folder).is_absolute():  # as compile_command reads it
            folder = (recipe.elf.parent / folder).resolve()
        if folder and (Path(folder) / DESCRIPTORS).is_file():
            return Path(folder)
    return None


def _loaded_formats(run):
    """([format or "" a buffer index], {index: unpack mode}) of the load's buffers: the
    runtime's, the slots', the area's and the loaded programs' (an added one's are in
    no kernel build)."""
    backend = run.backend
    names = [""] * len(_FORMAT_INDICES)
    for index, fmt, *_ in backend._runtime_cbs():
        names[index] = fmt
    names[backend.slot_index] = names[backend.switch_pad] = "uint32"
    for program in run.programs[: backend.switch_loaded]:
        for buffer in program.buffers:
            names[buffer.index] = buffer.format
    modes = dict(run.programs[0].compute.get("unpack_to_dest_mode", {}))
    return names, modes


_FORMAT_INDICES = range(64)  # tt-metal's buffer indices on Blackhole (max_dfbs)


def own_recipes(run, program, recipes):
    """{part: Recipe}: `recipes` compiling with a copy of the kernel's descriptors file
    holding the class's buffer formats (first among the includes). Refused when the
    load's tables are not reproduced, an index holds another format there, or the
    class's formats change a loaded index's entry."""
    module = run.backend.session._module
    compute = run.programs[0].compute
    flags = (
        bool(compute.get("fp32_dest_acc_en")),
        bool(compute.get("bfp8_pack_precise")),
    )
    names, modes = _loaded_formats(run)
    mine, own = list(names), dict(modes)
    for buffer in program.buffers:
        if names[buffer.index] not in ("", buffer.format):
            raise added.AddError(
                f"{program.name}: buffer {buffer.index} is {buffer.format} and the "
                f"loaded kernels' buffer {buffer.index} is {names[buffer.index]}"
            )
        mine[buffer.index] = buffer.format
    for index, mode in program.compute.get("unpack_to_dest_mode", {}).items():
        if own.setdefault(index, mode) != mode:
            raise added.AddError(
                f"{program.name}: unpack_to_dest_mode {mode} at buffer {index} and the "
                f"load's {modes[index]}"
            )

    def tables(buffers, unpack):
        listed = [unpack.get(index, "Default") for index in _FORMAT_INDICES]
        made = module.format_tables(buffers, flags[0], listed, flags[1])
        return dict(zip(TABLES, made, strict=True))

    base, wanted = tables(names, modes), tables(mine, own)
    indices = {buffer.index for buffer in program.buffers}
    out, cache = dict(recipes), native.cache_directory() / "descriptors"
    for part, recipe in recipes.items():
        folder = _descriptors(recipe)
        if folder is None:
            continue
        text = (folder / DESCRIPTORS).read_text()
        for name in TABLES:
            have = _table(text, name)
            if have is None:
                continue
            if have != list(base[name]):
                raise added.AddError(
                    f"{program.name}: the loaded kernels' {name} ({part}) is not the "
                    "one the load's buffers give; the added class's tables cannot be "
                    "made beside it"
                )
            if any(
                wanted[name][i] != have[i] for i in _FORMAT_INDICES if i not in indices
            ):
                raise added.AddError(
                    f"{program.name}: its formats change the loaded kernels' {name} "
                    "(the shared Dst format): load it at the start"
                )
            text = _with_table(text, name, wanted[name])
        for name in SIZES:
            have = _table(text, name)
            if have is not None:
                for buffer in program.buffers:
                    have[buffer.index] = formats.tile_bytes(buffer.format)
                text = _with_table(text, name, have)
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        made = cache / digest
        made.mkdir(parents=True, exist_ok=True)
        if not (made / DESCRIPTORS).is_file():
            (made / DESCRIPTORS).write_text(text)
        out[part] = dataclasses.replace(
            recipe, includes=(f"-I{made}", *recipe.includes)
        )
    return out


def _check_included(recipe, made, kernel, part):
    """The part was compiled with the copy of the descriptors file given it."""
    folder = _descriptors(recipe)
    if folder is None:
        return
    index = native.cache_directory() / "overlays" / "built"
    index = index / f"{overlay.built_key(recipe, made, kernel)}-{part}.json"
    files = json.loads(index.read_text())["files"]
    if str((folder / DESCRIPTORS).resolve()) not in files:
        raise added.AddError(
            f"{made.name}: its {part} part was not compiled with its own format tables"
        )


def grow_area(backend, descriptor, rows, placed):
    """The switch area grown to `buffers=` bytes from its start, or by default into
    the L1 the table `rows` leaves, less KEEP; rows' first row says the new size."""
    given = getattr(backend, "added_buffers", None)
    if given == 0:
        return
    align = programs.CB_ALIGN
    free = backend.session.l1()[1]
    left = free - sum(size for _, size in rows)
    have = backend.switch_base + backend.switch_reserve - backend.switch_start
    if given is None:
        extra = max(0, left - KEEP) // align * align
    else:
        extra = -(-max(0, given - have) // align) * align
        if extra > left:
            raise ProgramError(
                f"buffers={given:,}: the switch area for added programs would take "
                f"{extra:,} B more and the load's L1 leaves {left:,} B"
            )
    if extra:
        descriptor.grow_cb(backend.switch_pad, extra)
        backend.switch_reserve += extra
    rows[0] = ("circular buffers (programs, added ones' area)", backend.switch_reserve)
    backend.session.log(
        f"switch area for added programs: {have + extra:,} B from the runtime's "
        f"buffers ({'buffers=' if given is not None else 'the L1 left'}"
        f"{given if given is not None else f' less {KEEP:,} B'})"
    )


def _kernel_wide(compute, session):
    """{option: value} of the compute options that are one a kernel (all but the
    unpack modes, which are per buffer index: own_recipes), unset ones left out."""
    config = session.compute_config(compute)
    out = dict(config["flags"])
    if config["math_fidelity"] is not None:
        out["math_fidelity"] = config["math_fidelity"]
    return {name: value for name, value in out.items() if value}


def refused(run, program):
    """Why `program` cannot be added to `run`, by name; "" when it can."""
    loaded = run.programs[0]
    session = importlib.import_module("komorebi.session")
    if not isinstance(program, Program) or program.form() != "program":
        return (
            f"{getattr(program, 'name', program)}: a class added to a live run is "
            'made in the runtime\'s form (form="runtime")'
        )
    if program in run.programs:
        return f"{program.name}: a program of this run already"
    try:
        programs.check(program, run.cores)
    except ProgramError as error:
        return str(error)
    if run._switch is None or not hasattr(run.backend, "switch_start"):
        return (
            f"{program.name}: the run was not loaded as a dict ({{program: group}}): "
            "only a run whose groups switch program has the load command and the "
            "switch area an added program takes"
        )
    if run.island_ranks != run.cores or getattr(run.backend, "island", None):
        return f"{program.name}: an island run (its parts are built per island)"
    if getattr(run.backend, "trace_cap", 0):
        return f"{program.name}: a traced run (the trace buffer is placed at launch)"
    if getattr(program, "merged_sources", None) is not None:
        return f"{program.name}: a merged program (merge its sources with the load's)"
    kernel_wide = _kernel_wide(program.compute, session)
    for name, value in _kernel_wide(loaded.compute, session).items():
        if kernel_wide.get(name) != value:
            return (
                f"{program.name}: compute option {name} is {kernel_wide.get(name)} "
                f"and the loaded kernels' {value}: one a compute kernel, built at "
                "the load"
            )
    if set(kernel_wide) - set(_kernel_wide(loaded.compute, session)):
        name = sorted(set(kernel_wide) - set(_kernel_wide(loaded.compute, session)))[0]
        return (
            f"{program.name}: compute option {name} is {kernel_wide[name]} and the "
            "loaded kernels' None: one a compute kernel, built at the load"
        )
    if program.opt != loaded.opt:
        return f"{program.name}: opt levels {program.opt} and the load's {loaded.opt}"
    if (program.kernel_noc("reader"), program.kernel_noc("writer")) != (
        loaded.kernel_noc("reader"),
        loaded.kernel_noc("writer"),
    ):
        return f"{program.name}: other NoCs than the loaded kernels'"
    if program.args is not None and not getattr(run.backend, "switch_args", False):
        return (
            f"{program.name}: per-core args (a rectangle, a broadcast, self.x) need "
            "the load command's row read, built only when a class with them is named "
            "with Program.ahead(...) before the load"
        )
    room = getattr(run.backend, "room", None)
    entered = getattr(run.backend, "added_entry", False) or (room and room.entered)
    if program.overlays and not entered:
        return (
            f"{program.name}: overlays of its own need the entry command's code, built "
            "only when a class with them is named with Program.ahead(...) before the "
            "load (or the program names slot bytes)"
        )
    if program.messages is not None:
        return (
            f"{program.name}: a dynamic handler (its queues and tables are set at the "
            "launch)"
        )
    runtime = {cb[0] for cb in run.backend._runtime_cbs()} | {run.backend.slot_index}
    clash = sorted({b.index for b in program.buffers} & runtime)
    if clash or any(b.index >= programs.SWITCH_BUFFERS for b in program.buffers):
        return (
            f"{program.name}: buffer indices {clash or 'from 32'}: the runtime's own "
            f"(in a switch run a program's buffers are 0 to 31, not {sorted(runtime)})"
        )
    try:
        check_families(
            program.name,
            [*program.buffers, *(b for p in run.programs for b in p.buffers)],
            program.compute,
        )
    except ProgramError as error:
        return f"{error} (the loaded programs' buffers and the added one's)"
    return ""


def add(run, program):
    """run.addon(program): `program` one more program of the live run; returns its
    program field. Nothing is written to the card when it is refused."""
    why = refused(run, program)
    if why:
        raise added.AddError(why)
    run._check_alive()
    field = len(run.programs)
    if field not in run._switch.get("spare", {}):
        raise added.AddError(
            f"{program.name}: field {field} is past the record rows the load reserved "
            f"(fields to {len(run.programs) + len(run._switch.get('spare', {})) - 1}; "
            "a program's own overlays take the numbers above them)"
        )
    backend, started = run.backend, time.perf_counter()
    session = backend.session
    shift = getattr(program, "added_own_shift", None)
    if shift is None or program.program_field != field:
        shift = _own_shift(run, program, field)
    service = overlay.constants()["kOverlayService"]
    if service - 1 - field in backend.added_own:
        raise added.AddError(
            f"{program.name}: field {field}'s overlay number {service - 1 - field} is "
            "an added class's own overlay; fewer or lower own overlays"
        )
    program.program_field, program.overlay_shift, program.args_at = field, shift, 0
    session._hand_nocs(program)
    rows, row_bytes = args_defines(program, run.cores)
    layout, end = layout_of(backend, program)
    at, end = _args_at(end), _args_at(end) + row_bytes
    start = backend.switch_start + backend.switch_shift
    top = backend.switch_base + backend.switch_reserve + backend.switch_shift
    if end > top:
        raise added.AddError(
            f"{program.name}: its buffers take {end - start:,} B and the switch area "
            f"holds {top - start:,} B (session.load(..., buffers=N), or "
            "Program.ahead(program) at the load)"
        )
    made = programs.program_overlay(program, True)
    room = backend.room
    if room is None:
        raise added.AddError(
            f"{program.name}: the load reserved no room for added code (reserve=0)"
        )
    images, cached = {}, True
    group = _own_group(run, program, room.groups[0])
    for kernel, part in made.parts():
        linked, hit = added._part(session, group, made, kernel, part)
        _check_included(group.recipes[part], made, kernel, part)
        images[part], cached = linked.image, cached and hit
    own = _own_parts(run, program, group)
    built_s = time.perf_counter() - started
    placed = added._place(run, made, [images])[0]
    table = [placed.get(part, (0, 0)) for part in overlay.PARTS]
    words = programs.record_words(field, layout, table)
    if rows:  # the per-core rows: a DRAM page a core, copied by the load (kRecordArgs)
        c = overlay.constants()
        table = np.zeros((run.cores, row_bytes // 4), np.uint32)
        for core, row in enumerate(rows):
            table[core, : len(row)] = row
        rows_tensor = importlib.import_module("komorebi.session")._words(session, table)
        backend.io.append(rows_tensor)
        if rows_tensor.aligned_page_bytes != row_bytes:
            raise added.AddError(f"{program.name}: rows of {row_bytes} B are not pages")
        words[c["kRecordArgs"]] = rows_tensor.address
        words[c["kRecordArgsBytes"]] = row_bytes
        words[c["kRecordArgsL1"]] = at
    address, row = run._switch["spare"].pop(field)
    banks = overlay.constants()["kOverlayBanks"]
    tensor = run._switch["tensor"]
    page = np.zeros((banks, tensor.page_bytes), np.uint8)
    page[:, : words.nbytes] = np.frombuffer(words.tobytes(), np.uint8)
    session.write(tensor, page, first=banks * row)  # a copy in each bank, as loaded
    backend.switch_layouts[field] = layout
    run._switch["records"][field] = address
    entries_s = _own_entries(run, own)
    run.programs.append(program)
    if run._holds is not None:
        run._holds.append(np.zeros(run.cores, bool))
    for thread in run.threads.values():  # kmx_load reaches it
        thread.write_records(run._switch)
    session.log(
        f"added program {program.name} as field {field}: parts "
        f"{ {p: len(i) for p, i in images.items()} } built in {built_s * 1e3:.1f} ms"
        f"{' (cached)' if cached else ''}; buffers {end - start:,} B in the switch "
        f"area; {time.perf_counter() - started:.2f} s"
        + (
            f"; own overlays {sorted(number for number, _, _ in own)}, their entries "
            f"sent to every core in {entries_s * 1e3:.1f} ms"
            if own
            else ""
        )
    )
    return field


def _own_shift(run, program, field):
    """The shift of an added class's own overlays (each declared number + shift): the
    lowest that leaves every number free (the run's overlays, added code, earlier added
    classes' own) and below every program field's overlay up to `field`."""
    if not program.overlays:
        return 0
    backend, service = run.backend, overlay.constants()["kOverlayService"]
    room = backend.room
    taken = set(backend.added_own) | {
        n for each in (room.groups if room else ()) for n in each.numbers
    }
    taken |= {each.number for each in (room.added.values() if room else ())}
    lowest = service - 1 - field  # field p is overlay service - 1 - p
    numbers = [each.number for each in program.overlays.values()]
    for shift in range(lowest):
        mine = {number + shift for number in numbers}
        if max(mine) < lowest and not mine & taken:
            return shift
    raise added.AddError(
        f"{program.name}: no overlay numbers left for its own overlays "
        f"{sorted(numbers)} below its field's {lowest}"
    )


def _own_parts(run, program, group):
    """[(number, name, {part: image})] of an added class's own overlays, built against
    `group` (its own recipes); refused by name where a kernel has one slot (a part of
    the class runs from one while it loads its own overlay into another)."""
    out = []
    for made in own_overlays(program):
        images = {}
        for kernel, part in made.parts():
            count = group.slots.get(part, (0, 0))[0]
            if count == 1:
                raise added.AddError(
                    f"{program.name}: its own overlay {made.name} has a {kernel} part "
                    f"and the load gave the {kernel} one slot; name the class with "
                    "Program.ahead(...) before the load, or Program.slots(count=2)"
                )
            linked, _ = added._part(run.backend.session, group, made, kernel, part)
            _check_included(group.recipes[part], made, kernel, part)
            images[part] = linked.image
        out.append((made.number, made.name, images))
    return out


def _own_entries(run, own):
    """Places an added class's own overlays' parts into the room and sends each group
    the entry command of each (the service overlay writes the table entry, each part's
    bank address and bytes, in every core's overlay area); returns its seconds."""
    if not own:
        return 0.0
    backend, started = run.backend, time.perf_counter()
    entry = abi.headers()["KM_HANDLER_ENTRY"]
    for number, name, images in own:
        made = SimpleNamespace(name=name, number=number)
        placed = added._place(run, made, [images])[0]
        words = added.entry_words(number, False, placed)
        handles = [
            run._queue_command(entry, (each.first, each.count), words)
            for each in backend.room.groups
        ]
        for handle in handles:
            event = run.wait(handle, 30.0)
            if event.kind != "done":
                raise added.AddError(
                    f"{name}: the entry command of its own overlay {number} ended "
                    f"{event.kind}: {event}"
                )
        for each in backend.room.groups:
            added._mirror(run, each, number, words)
            each.numbers.add(number)
        backend.added_own.add(number)
    return time.perf_counter() - started
