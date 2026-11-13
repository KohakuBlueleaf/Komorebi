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
    most = overlay.constants()["kOverlaysMax"] - 1
    if len(programs) > most:
        raise ValueError(f"{len(programs)} programs; a run holds at most {most}")
    for index, program in enumerate(programs):
        check(program, len(programs))
        if index and (program.threads or program.x280_ports is not None):
            raise ProgramError(
                f"{program.name}: X280 threads and ports belong to the run's first "
                "program"
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
        if len(held) > 1 and any(program.args is not None for program in held):
            raise ProgramError(
                f"{', '.join(p.name for p in held)}: programs that share cores take no "
                "per-core args"
            )
    for index, program in enumerate(programs):
        program.program_field = index
    return Plan(tuple(programs), tuple(groups), cores)


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
        raise ProgramError(f"{program.name}: a dynamic handler needs today's form")
    if program.overlays:
        raise ProgramError(
            f"{program.name}: overlays of a program in the runtime's form are not "
            "supported yet"
        )
    if program.compile_args.get("movement") or any(
        not isinstance(value, list | tuple) for value in program.compile_args.values()
    ):
        raise ProgramError(f"{program.name}: compile_args are lists, per kernel")


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


def program_overlay(program):
    """The overlay that holds `program` (its number from its program field)."""
    sources = {kernel: Path(path) for kernel, path in program.sources().items()}
    return overlay.Overlay(
        program.name,
        overlay.constants()["kOverlaysMax"] - 1 - program.program_field,
        sources,
        dict(program.defines),
        program=True,
        compile_args={
            kernel: program.kernel_compile_args(kernel) for kernel in overlay.KERNELS
        },
    )


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
