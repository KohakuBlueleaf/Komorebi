"""Several programs in one run: which cores hold which programs, their joint L1 layout,
and each program as an overlay under the runtime's kernels. A plain description;
nothing here needs a device.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from komorebi import _env, overlay
from komorebi.errors import L1LayoutError, ProgramError
from komorebi.program import Program

# Below these buffer indices a buffer is an input, from them on an output (tt-metal's
# convention): the compute kernel's start-up takes the first of each.
OUTPUT_BUFFERS = 16
CB_ALIGN = 64  # tt-metal starts each core's circular buffer on this boundary
SWITCH_BUFFERS = 32  # a load sets up buffers 0 to 31 (a record's mask is a word)


@dataclass(frozen=True)
class Group:
    """Cores [first, first + count) and the programs (by program field) they hold."""

    first: int
    count: int
    programs: tuple


@dataclass(frozen=True)
class Plan:
    """A load of programs in the runtime's form: `programs` by program field, `groups`
    disjoint, covering every core (a group with no program holds none)."""

    programs: tuple
    groups: tuple
    cores: int
    owner: object = None  # the program that declares the X280 threads and ports
    switch: bool = False  # a dict: a group can change program on the live run

    def threads_owner(self):
        """The program whose X280 threads and ports the run starts: the one that
        declares them, else the first."""
        return self.owner or self.programs[0]

    def holds(self):
        """[field] -> bool array over the cores: the cores that hold that program."""
        out = [np.zeros(self.cores, bool) for _ in self.programs]
        for group in self.groups:
            for program_field in group.programs:
                out[program_field][group.first : group.first + group.count] = True
        return out


def runtime_kernels():
    """{"reader", "writer", "compute"}: the runtime's kernels of a run of programs."""
    include = _env.include_directory()
    if include is None:
        raise FileNotFoundError("the native kernels are not beside the package")
    folder = include.parent / "kernels"
    return {kernel: folder / f"program_{kernel}.cpp" for kernel in overlay.KERNELS}


def plan(spec, cores):
    """The Plan of `spec`: a Program in the runtime's form, a list of them (each core
    holds every one) or {Program: (first, count)}; sets each program's field."""
    if isinstance(spec, Program):
        spec = [spec]
    if isinstance(spec, dict):
        programs = list(spec)
        ranges = [_range(program, spec[program], cores) for program in programs]
    elif isinstance(spec, list | tuple):
        programs = list(spec)
        ranges = None
    else:
        raise TypeError(f"load takes a Program, a list or a dict of them: {spec!r}")
    if not programs or not all(isinstance(p, Program) for p in programs):
        raise TypeError("load takes Programs")
    if len({id(program) for program in programs}) != len(programs):
        raise ValueError("a program is given twice")
    most = overlay.constants()["kOverlayService"] - 1
    if len(programs) > most:
        raise ValueError(f"{len(programs)} programs; a run holds at most {most}")
    for program in programs:
        check(program, len(programs))
    owners = [p for p in programs if p.threads or p.x280_ports is not None]
    if len(owners) > 1:
        raise ProgramError(
            f"{', '.join(p.name for p in owners)}: X280 threads and ports belong to "
            "one program of a run; declare them in one"
        )
    if ranges is None:
        groups = [Group(0, cores, tuple(range(len(programs))))]
    else:
        groups = []
        order = sorted(range(len(programs)), key=lambda index: ranges[index][0])
        at = 0
        for index in order:
            first, count = ranges[index]
            if first < at:
                raise ValueError(
                    f"{programs[index].name}: group ({first}, {count}) overlaps "
                    "another program's group"
                )
            if first > at:
                groups.append(Group(at, first - at, ()))
            groups.append(Group(first, count, (index,)))
            at = first + count
        if at < cores:
            groups.append(Group(at, cores - at, ()))
    for group in groups:
        held = [programs[index] for index in group.programs]
        _same(held, "compute", "compute options")
        _same(held, "opt", "opt levels")
        _same_nocs(held, "programs that share cores")
        if len(held) > 1 and any(program.args is not None for program in held):
            raise ProgramError(
                f"{', '.join(p.name for p in held)}: programs that share cores take no "
                "per-core args"
            )
    if ranges is not None:  # one build of the runtime's kernels serves every group
        _same_nocs(programs, "programs whose groups switch")
    for program in programs if ranges is not None else ():
        if any(buffer.index >= SWITCH_BUFFERS for buffer in program.buffers):
            raise ProgramError(
                f"{program.name}: in a run whose groups switch program, buffer "
                f"indices are below {SWITCH_BUFFERS}"
            )
    for index, program in enumerate(programs):
        program.program_field = index
    shift_overlays(programs)
    return Plan(
        tuple(programs),
        tuple(groups),
        cores,
        owners[0] if owners else None,
        ranges is not None,
    )


def check(program, count):
    """Refuses what a program in the runtime's form does not have (yet)."""
    if program.form() != "program":
        if count > 1:
            raise ProgramError(
                f"{program.name}: several programs in one run are each in the "
                "runtime's form (km_program); its sources define kernel_main"
            )
        raise ProgramError(f"{program.name}: its sources define kernel_main")
    if program.messages is not None:
        raise ProgramError(f"{program.name}: a dynamic handler needs the kernel form")
    if program.compile_args.get("movement") or any(
        not isinstance(value, list | tuple) for value in program.compile_args.values()
    ):
        raise ProgramError(f"{program.name}: compile_args are lists, per kernel")


def shift_overlays(programs):
    """Gives each program's own overlays their numbers in the run: its own numbers
    plus the largest of each program before it (Program.overlay_shift). The programs
    take 62 down; ProgramError when the overlays reach them."""
    service = overlay.constants()["kOverlayService"]
    shift = 0
    for program in programs:
        program.overlay_shift = shift
        shift += max((each.number for each in program.overlays.values()), default=0)
    declared = programs[0].overlay_slots  # the first program's slots are the run's
    for kernel in overlay.KERNELS:
        count = declared.count
        count = count.get(kernel) if isinstance(count, dict) else count
        if count == 1 and any(
            kernel in each.sources for p in programs for each in p.overlays.values()
        ):
            raise ProgramError(
                f"{programs[0].name}: one {kernel} slot, and a program's {kernel} part "
                "loads its own overlay into another slot while it runs; give it two "
                "(Program.slots)"
            )
    lowest = service - len(programs)  # program field p is overlay service - 1 - p
    if shift >= lowest:
        raise ProgramError(
            f"{', '.join(p.name for p in programs)}: their own overlays take numbers 1 "
            f"to {shift} in the run (each program's after the largest of those before "
            f"it) and the {len(programs)} programs {lowest} to {service - 1}; fewer "
            "overlays or programs, or smaller numbers"
        )


def own_overlays(program):
    """A program's own overlays as the run numbers them (its shift added, also to the
    numbers its parts name: KM_OVERLAY_SHIFT)."""
    shift = program.overlay_shift
    return [
        overlay.Overlay(
            each.name,
            each.number + shift,
            each.sources,
            {**each.defines, "KM_OVERLAY_SHIFT": str(shift)},
        )
        for each in program.overlays.values()
    ]


def _range(program, group, cores):
    first, count = int(group[0]), int(group[1])
    if count < 1 or first < 0 or first + count > cores:
        raise ValueError(
            f"{program.name}: group ({first}, {count}) is not inside {cores} cores"
        )
    return first, count


def _same(programs, attribute, what):
    values = [getattr(program, attribute) for program in programs]
    if any(value != values[0] for value in values[1:]):
        raise ProgramError(
            f"{', '.join(p.name for p in programs)}: programs that share cores share "
            f"one set of {what}"
        )


def _same_nocs(programs, who):
    nocs = {(p.kernel_noc("reader"), p.kernel_noc("writer")) for p in programs}
    if len(nocs) > 1:
        raise ProgramError(
            f"{', '.join(p.name for p in programs)}: {who} put their reader and "
            "writer on the same NoCs (Program noc)"
        )


def joint_buffers(programs, free):
    """The buffers of programs that share cores: one L1 layout, where an index two
    programs declare is declared identically (one shared buffer); else L1LayoutError
    with the L1 budget table (`free`: the bytes a core leaves for buffers)."""
    by_index, owner = {}, {}
    for program in programs:
        for buffer in program.buffers:
            seen = by_index.get(buffer.index)
            if seen is None:
                by_index[buffer.index], owner[buffer.index] = buffer, program.name
            elif seen != buffer:
                raise L1LayoutError(
                    f"buffer {buffer.index} is declared by {owner[buffer.index]} as "
                    f"{_text(seen)} and by {program.name} as {_text(buffer)}",
                    overlay.budget(free, budget_rows(programs))[0],
                )
    return [by_index[index] for index in sorted(by_index)]


def budget_rows(programs):
    """[(name, bytes)]: each program's buffers, for the L1 budget table."""
    return [
        (f"{program.name}: buffer {buffer.index}", buffer.page_bytes * buffer.pages)
        for program in programs
        for buffer in program.buffers
    ]


def _text(buffer):
    return f"{buffer.format}, {buffer.pages} x {buffer.page_bytes} B"


def start_buffers(buffers):
    """(input, output) buffer indices the compute kernel's start-up configures: the
    first below OUTPUT_BUFFERS and the first from it (0 and 16 when there is none)."""
    indices = sorted(buffer.index for buffer in buffers)
    inputs = [index for index in indices if index < OUTPUT_BUFFERS]
    outputs = [index for index in indices if index >= OUTPUT_BUFFERS]
    return (inputs[0] if inputs else 0), (outputs[0] if outputs else OUTPUT_BUFFERS)


def program_overlay(program, switch=False):
    """The overlay that holds `program` (its number from its program field); with
    `switch`, its compute parts run the hardware start-up on its buffers at a load."""
    sources = {kernel: Path(path) for kernel, path in program.sources().items()}
    defines = dict(program.defines)
    if program.overlays:  # the numbers its parts name, as the run numbers them
        defines["KM_OVERLAY_SHIFT"] = str(program.overlay_shift)
    if switch:
        cb_in, cb_out = start_buffers(program.buffers)
        defines |= {
            "KM_PROGRAM_SETUP_IN": str(cb_in),
            "KM_PROGRAM_SETUP_OUT": str(cb_out),
        }
    return overlay.Overlay(
        program.name,
        overlay.constants()["kOverlayService"] - 1 - program.program_field,
        sources,
        defines,
        program=True,
        compile_args={
            kernel: program.kernel_compile_args(kernel) for kernel in overlay.KERNELS
        },
    )


def program_overlays(program, switch=False, alone=True):
    """The overlays that hold `program`: its own (program_overlay), or a merged
    program's part and the sources it places as overlays (komorebi.merge)."""
    parts = getattr(program, "part_overlays", None)
    if parts is not None:
        return parts(switch, alone)
    return [program_overlay(program, switch)]


def rectangles(first, count, row):
    """Cores [first, first + count) in row-major order as at most three rectangles
    (x0, y0, x1, y1) of a grid `row` cores wide."""
    last = first + count - 1
    (y0, x0), (y1, x1) = divmod(first, row), divmod(last, row)
    if y0 == y1:
        return [(x0, y0, x1, y1)]
    out = []
    if x0 != 0:
        out.append((x0, y0, row - 1, y0))
        y0 += 1
    if x1 != row - 1:
        if y0 <= y1 - 1:
            out.append((0, y0, row - 1, y1 - 1))
        out.append((0, y1, x1, y1))
    else:
        out.append((0, y0, row - 1, y1))
    return out


def buffer_layout(buffers, base):
    """{index: (address, bytes, pages, page bytes)} of circular buffers `buffers`
    ((index, format, page bytes, pages), in creation order) as tt-metal places them on
    a core from `base`, each on a CB_ALIGN boundary; and the end of the last."""
    out, at = {}, base
    for index, _, page, pages in buffers:
        at = -(-at // CB_ALIGN) * CB_ALIGN
        out[index] = (at, page * pages, pages, page)
        at += page * pages
    return out, at


def record_words(program_field, layout, table):
    """A program's record (km_overlay.hpp): its buffers' mask, its field, its table
    entries [(address, bytes)] per part, then each buffer's four words at its index."""
    c = overlay.constants()
    words = np.zeros(c["kRecordBytes"] // 4, np.uint32)
    words[c["kRecordMask"]] = sum(1 << index for index in layout)
    words[c["kRecordField"]] = program_field
    for part, (address, size) in enumerate(table):
        words[c["kRecordTable"] + 2 * part : c["kRecordTable"] + 2 * part + 2] = (
            address,
            size,
        )
    for index, four in layout.items():
        at = c["kRecordBuffers"] + 4 * index
        words[at : at + 4] = four
    return words
