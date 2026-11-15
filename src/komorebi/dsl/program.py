"""A class deriving km.Program made into an ordinary program: the class is compiled
(komorebi.dsl.compiler), its files written under build/ beside its source and kept,
the sizes printed, and the made object is a Program of those files.
"""

from dataclasses import dataclass
from pathlib import Path

from komorebi import abi, layout
from komorebi import program as km_program
from komorebi.dsl import compiler
from komorebi.dsl.names import spelling
from komorebi.dsl.nodes import prune, render
from komorebi.errors import ProgramError

BUILD = "build"
INDENT = "    "
# The program arguments of every kernel: the core's column and row in the rectangle,
# whether a broadcast takes it, then (on a mover) its broadcast's arguments.
ARG_X, ARG_Y, ARG_INCLUDED, ARG_BROADCAST = 0, 1, 2, 3


@dataclass(frozen=True)
class Built:
    """What the build of a class wrote and chose: its folder, the files, the buffers'
    lines, and the waits of its code by handler and CPU."""

    folder: Path
    files: dict
    report: list
    waits: list


def prepare(cls):
    """Makes `cls`, a class deriving Program, a program written as a class if it has
    handler methods: its handlers numbered, its constructor the compiler."""
    methods = compiler.handler_methods(cls)
    if not methods:
        return
    user_max = abi.headers()["KM_HANDLER_USER_MAX"]
    if len(methods) > user_max:
        raise ProgramError(f"{cls.__name__}: {len(methods)} handlers, {user_max} most")
    for number, method in enumerate(methods, start=1):
        method.number = number
    cls.__init__ = make


def make(self, plan=None, l1=None, depth=4, log=print, rect=None, nocs=None, **tensors):
    """Compiles the class: `tensors` an example tensor of each tensor parameter (by
    name), `plan` sizes by name, `l1` the L1 left for buffers (Session.l1()[1]); with
    broadcasts, `rect` (x, y, columns, rows) and `nocs` (Session.nocs())."""
    cls = type(self)
    plan = dict(plan or {})
    compiled = compiler.compile_class(cls, plan, tensors, l1)
    program_args = None
    if compiled.rectangle:
        program_args = rectangle_args(compiled, rect, nocs)
    built = write(cls, compiled, l1)
    for line in built.report:
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
    km_program.Program.__init__(
        self,
        name=compiled.name.lower(),
        kernels=built.folder,
        buffers=buffers,
        compile_args={"movement": movement_args} if movement_args else {},
        compute=compute_options(compiled),
        args=program_args,
        depth=depth,
        handlers=built.folder / header_name(compiled),
    )
    self.built = built


def rectangle_args(compiled, rect, nocs):
    """Each core's program arguments (a function of the core, as Program takes): its
    column and row in `rect`, whether it is there, its mover's broadcast arguments."""
    if rect is None or nocs is None:
        raise ProgramError(
            f"{compiled.name} uses self.x, self.y or a broadcast: make it with "
            f"{spelling('rect')}=(x, y, columns, rows) and {spelling('nocs')}="
            "session.nocs()"
        )
    first_x, first_y, columns, rows = rect
    grid_width = nocs.grid[0]
    first = first_y * grid_width + first_x
    group = layout.Group(
        first, (first_y + rows - 1) * grid_width + first_x + columns - first
    )
    fields = abi.struct_fields(abi.headers().device[-1], "BroadcastArgs")
    by_mover = {}
    for number, (along, buffer, window) in enumerate(compiled.broadcasts):
        noc = 0 if along == "row" else 1
        by_mover[compiler.ALONG[along]] = layout.broadcast_args(
            nocs.grid,
            nocs.panels,
            group,
            rect,
            along,
            noc,
            nocs.noc0,
            nocs.noc_xy,
            (number, buffer.index_a, window, buffer.block),
        )
    cells = {
        y * grid_width + x: (x - first_x, y - first_y)
        for y in range(first_y, first_y + rows)
        for x in range(first_x, first_x + columns)
    }

    def program_args(core):
        x, y = cells.get(core, (0, 0))
        common = [x, y, int(core in cells)]
        movers = []
        for mover in ("A", "B"):
            words = by_mover.get(mover, {}).get(core)
            movers.append(common + ([words[name] for name in fields] if words else []))
        return movers[0], movers[1], common

    return program_args


def compute_options(compiled):
    """The compute kernel's options: fp32 in DST; float32 buffers unpacked to it."""
    options = {"math_fidelity": "HiFi4", "fp32_dest_acc_en": True}
    fp32 = [b for b in compiled.buffers if b.name in compiled.unpack_fp32]
    for buffer in fp32:
        options.setdefault("unpack_to_dest_mode", {})
        for index in (buffer.index_a, buffer.index_b):
            if index is not None:
                options["unpack_to_dest_mode"][index] = "UnpackToDestFp32"
    return options


def header_name(compiled):
    return f"{compiled.name.lower()}.h"


def camel(name):
    return "".join(part.capitalize() for part in name.split("_"))


def write(cls, compiled, l1):
    """Writes the header, the two kernels and program.py into build/<name>/ beside
    the class's source; returns what was written and the report."""
    folder = Path(compiled.source).resolve().parent / BUILD / compiled.name.lower()
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        header_name(compiled): header(compiled),
        "movement.cpp": movement_source(compiled),
        "compute.cpp": compute_source(compiled),
    }
    for name, text in files.items():
        text = resume_lines(text, name)
        files[name] = text
        (folder / name).write_text(text)
    report = summary(compiled, folder, files, l1)
    (folder / "program.txt").write_text("\n".join(report) + "\n")
    return Built(folder, files, report, compiled.waits)


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
        '#include "komorebi/device/km_movement.hpp"',
    ]
    if "port" in compiled.includes:
        lines.append('#include "komorebi/device/km_port.hpp"')
    if "broadcast" in compiled.includes:
        lines.append('#include "komorebi/device/km_broadcast.hpp"')
    lines.append("")
    if compiled.rectangle:
        lines += [
            "// Program arguments: the core's column and row in the rectangle, 1",
            "// when a broadcast takes it, then its mover's broadcast's arguments.",
            f"constexpr uint32_t kArgX = {ARG_X};",
            f"constexpr uint32_t kArgY = {ARG_Y};",
            f"constexpr uint32_t kArgIncluded = {ARG_INCLUDED};",
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
    for buffer in compiled.buffers:
        lines.append("")
        lines.append(f"// {buffer.what}")
        if buffer.index_b is None:
            if buffer.role == "arrival":
                slots = buffer.pages[0]
                lines.append(
                    f"constexpr km::Arrival {buffer.name} = "
                    f"{{{buffer.index_a}, {slots}, 1}};"
                )
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
    return "\n".join(lines) + "\n"


def tensor_aliases(compiled):
    lines, previous = [], None
    for name in compiled.tensors:
        offset = (
            "0" if previous is None else f"{previous}::next_compile_time_args_offset()"
        )
        lines.append(f"using {camel(name)} = km::Tensor<{offset}>;  // {name}")
        previous = camel(name)
    return lines


def cases(compiled, nodes_of, ending, depth):
    """The switch on the handler: a case for each handler, its pruned nodes."""
    pad = INDENT * depth
    lines = [pad + "switch (km::handler(command)) {"]
    for name in compiled.handlers:
        line = compiled.lines[name]
        lines.append(f"{pad}{INDENT}case kHandler{camel(name)}: {{  // line {line}")
        lines += render(prune(nodes_of[name]), depth + 2)
        lines.append(f"{pad}{INDENT * 2}{ending}")
        lines.append(f"{pad}{INDENT}}}")
    lines.append(f"{pad}{INDENT}default:")
    lines.append(f"{pad}{INDENT * 2}{ending}")
    lines.append(pad + "}")
    return lines


def movement_source(compiled):
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// the movement source of both movers (mover A: RISCV_0, mover B: RISCV_1).",
        "#include <stdint.h>",
        "",
        '#include "api/dataflow/dataflow_api.h"',
        '#include "komorebi/device/km_movement.hpp"',
    ]
    if "port" in compiled.includes:
        lines.append('#include "komorebi/device/km_port.hpp"')
    if "broadcast" in compiled.includes:
        lines.append('#include "komorebi/device/km_ask.hpp"')
        lines.append('#include "komorebi/device/km_broadcast.hpp"')
    lines += [f'#include "{header_name(compiled)}"', "", "void kernel_main() {"]
    lines += [INDENT + line for line in tensor_aliases(compiled)]
    lines += [
        f"{INDENT}km::Movement movement;",
        f"{INDENT}movement.run([&](const km_command_t& command) {{",
    ]
    lines += cases(compiled, compiled.movement, "return true;", 2)
    lines += [f"{INDENT}}});", "}"]
    return "\n".join(lines) + "\n"


COMPUTE_INCLUDES = [
    ("api/compute/cb_api.h", None),
    ("api/compute/compute_kernel_api.h", None),
    ("api/compute/compute_kernel_hw_startup.h", None),
    ("api/compute/eltwise_binary_sfpu.h", "add"),
    ("api/compute/eltwise_unary/fill.h", "fill"),
    ("api/compute/pack.h", None),
    ("api/compute/reg_api.h", None),
    ("api/compute/tile_move_copy.h", None),
    ("komorebi/device/km_broadcast.hpp", "broadcast"),
    ("komorebi/device/km_movement.hpp", None),
    ("komorebi/device/km_port.hpp", "port"),
    ("komorebi/device/km_worker.hpp", None),
]


def compute_source(compiled):
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// the compute kernel: the tiles of its streams, in block order.",
        "#include <stdint.h>",
        "",
    ]
    for path, needs in COMPUTE_INCLUDES:
        if needs is None or needs in compiled.includes:
            lines.append(f'#include "{path}"')
    lines += [f'#include "{header_name(compiled)}"', "", "void kernel_main() {"]
    reads = [b for b in compiled.buffers if b.role == "read"]
    writes = [b for b in compiled.buffers if b.role == "write"]
    others = [b for b in compiled.buffers if b.role == "broadcast"]
    first_in = (
        f"{reads[0].name}.mover_a" if reads else others[0].name if others else "0"
    )
    first_out = f"{writes[0].name}.mover_a" if writes else "16"
    lines += [
        f"{INDENT}km::Compute compute;",
        f"{INDENT}compute_kernel_hw_startup({first_in}, {first_out});",
        f"{INDENT}compute.run([&](const km_command_t& command) {{",
    ]
    lines += cases(compiled, compiled.compute, "break;", 2)
    lines += [f"{INDENT}}});", "}"]
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
    room = "" if l1 is None else f" of {l1} B left for buffers"
    report.append(f"  L1: the buffers take {total} B{room}")
    if l1 is not None and total > l1:
        raise ProgramError(
            f"{compiled.name}: its buffers take {total} B, {l1} B left: "
            "give smaller blocks in the plan"
        )
    return report


def lines_of(program, error):
    """The handler and Python line of each wait a not-reported core of `error` (a
    CommandTimeout) gave up in, by its CPU and where: one text a core."""
    built = getattr(program, "built", None)
    if built is None:
        return []
    cpus = {"reader": "mover A", "writer": "mover B", "compute": "compute"}
    named = []
    for core in error.cores:
        for kernel, cpu in cpus.items():
            gave = core.get(kernel)
            where = gave.get("where") if isinstance(gave, dict) else None
            if not where:
                continue
            named.extend(
                f"core {core['core']} {cpu}: {wait.handler}, line {wait.line}: "
                f"waiting in {wait.text}"
                for wait in built.waits
                if wait.cpu == cpu and wait.where == where
            )
    return named
