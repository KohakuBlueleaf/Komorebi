"""A class deriving km.Program made into an ordinary program: the class is compiled
(komorebi.dsl.compiler), its files written under build/ beside its source and kept,
the sizes printed, and the made object is a Program of those files.
"""

from dataclasses import dataclass
from pathlib import Path

from komorebi import _env, abi, layout, overlay
from komorebi import program as km_program
from komorebi.dsl import compiler
from komorebi.dsl.front import MESSAGE_WORDS
from komorebi.dsl.names import spelling
from komorebi.dsl.nodes import prune, render
from komorebi.errors import ProgramError

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
    lines, and the waits of its code by handler and CPU."""

    folder: Path
    files: dict
    report: list
    waits: list

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
    cls.built_handlers = [method.name for method in methods]
    cls.overlays_built = property(overlays_built, set_overlays_built)


def make(self, plan=None, l1=None, depth=4, log=print, rect=None, nocs=None, **tensors):
    """Compiles the class: `tensors` an example tensor of each tensor parameter (by
    name), `plan` sizes by name, `l1` the L1 left for buffers (Session.l1()[1]); with
    broadcasts, `rect` (x, y, columns, rows; the whole grid when not given) and `nocs`
    (Session.load gives the session's when not given)."""
    cls = type(self)
    plan = dict(plan or {})
    compiled = compiler.compile_class(cls, plan, tensors, l1)
    program_args = None
    # The rectangle's state, shared by the copies a session of islands runs.
    self.rectangle = {"rect": rect, "nocs": nocs, "args": None, "group": None}
    self.broadcast_lines = compiled.broadcast_lines
    if compiled.rectangle:
        program_args = rectangle_of(self, compiled)
    built = write(cls, compiled, l1)
    for line in built.report:
        log(line)
    report = may_cross(compiled, tensors)
    setattr(self, spelling("may_cross"), report)
    for line in report:
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
        compile_args={"movement": movement_args} if movement_args else {},
        compute=compute_options(compiled),
        args=program_args,
        depth=depth,
        handlers=built.folder / header_name(compiled),
    )
    self.built = built
    self.log = log
    for name in compiled.overlays:
        sources = {"reader": built.folder / name / "movement.cpp"}
        sources["writer"] = sources["reader"]
        if f"{name}/compute.cpp" in built.files:
            sources["compute"] = built.folder / name / "compute.cpp"
        self.overlay(name, sources)
    if compiled.overlays:
        self.slots()


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


def rectangle_of(program, compiled):
    """The program arguments of a class with a rectangle, made at their first use
    (the load), once the session gave its NoCs."""
    state = program.rectangle

    def args(core):
        if state["args"] is None:
            nocs = state["nocs"]
            if nocs is None:
                raise ProgramError(
                    f"{compiled.name} uses self.x, self.y or a broadcast: load it with "
                    f"Session.load, or make it with {spelling('nocs')}=session.nocs()"
                )
            rect = state["rect"] or (0, 0, *nocs.grid)
            state["args"], state["group"] = rectangle_args(compiled, rect, nocs)
        return state["args"](core)

    return args


def session_nocs(program, nocs):
    """Session.load's hand-over: the session's NoCs, unless the class was made with
    its own."""
    state = getattr(program, "rectangle", None)
    if state is not None and state["nocs"] is None:
        state["nocs"] = nocs


def rectangle_args(compiled, rect, nocs):
    """Each core's program arguments (a function of the core, as Program takes): its
    column and row in `rect`, whether it is there, its mover's broadcast arguments;
    and the rectangle's group (first, count)."""
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
        common = [x, y, int(core in cells), columns, rows]
        movers = []
        for mover in ("A", "B"):
            words = by_mover.get(mover, {}).get(core)
            movers.append(common + ([words[name] for name in fields] if words else []))
        return movers[0], movers[1], common

    return program_args, (group.first, group.count)


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
        thread_header_name(compiled): thread_header(compiled),
        "movement.cpp": movement_source(compiled),
        "compute.cpp": compute_source(compiled),
        **parts(compiled),
    }
    if compiled.dynamic:  # compute alone: the runtime's movers carry the messages
        del files["movement.cpp"]
        files["compute.cpp"] = dynamic_source(compiled)
    for name, text in files.items():
        text = resume_lines(text, name)
        files[name] = text
        (folder / name).parent.mkdir(exist_ok=True)
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


def thread_header_name(compiled):
    return f"{compiled.name.lower()}_thread.h"


def thread_header(compiled):
    """The C header of an X280 thread that submits the class's commands: each
    handler's number, the place of each of its words, their count, and whether its
    command asks for the leader's result."""
    prefix = compiled.name.upper()
    lines = [
        f"/* {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        " * its handlers' numbers and each command word's place (a0 is 0), for a",
        " * thread's commands (kmx_submit). */",
        "#pragma once",
    ]
    for name, number in compiled.handlers.items():
        handler = f"{prefix}_{name.upper()}"
        words = compiled.words[name]
        lines += ["", f"/* {name.upper()}, line {compiled.lines[name]}. */"]
        lines.append(f"#define {prefix}_HANDLER_{name.upper()} {number}u")
        for place, (word, kind) in enumerate(words):
            suffix = "_PAGES" if kind == "pages" else ""
            lines.append(f"#define {handler}_{word.upper()}{suffix} {place}u")
        lines.append(f"#define {handler}_WORDS {len(words)}u")
        if name in compiled.results:
            lines.append(f"#define {handler}_RESULT 1u /* KM_FLAG_RESULT */")
    return "\n".join(lines) + "\n"


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
    if compiled.messages:
        lines += ["", f"constexpr uint32_t kMessageWords = {MESSAGE_WORDS};"]
    for cls, words in compiled.messages.items():
        lines += [f"// {cls}: a message's words (word 0 the runtime's mark)."]
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
    return "\n".join(lines) + "\n"


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
    return lines


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
                f"{inner}km::Overlay part({kernel}, kOverlay{camel(name)});",
                f"{inner}if (!part.load()) {{",
                f"{inner}{INDENT}{gave_up}",
                f"{inner}}}",
            ]
            if kernel == "movement" and name in compiled.stamps:
                lines[-4:] = stamped(compiled, name, inner)
                lines.append(f"{pad}{INDENT}}}")
                continue
            if kernel == "movement":
                lines.append(f"{inner}return part.call(command) != 0;")
                lines.append(f"{pad}{INDENT}}}")
                continue
            lines.append(f"{inner}part.call(command);")
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
        f"km::Overlay part(movement, {overlay});",
        "const bool held = part.held();",
        "const uint64_t started = km::wall64();",
        "if (!part.load()) {",
        f"{INDENT}return false;",
        "}",
        "const uint64_t loaded = km::wall64();",
        "const bool called_ok = part.call(command) != 0;",
        "const uint64_t called = km::wall64();",
        "#if defined(COMPILE_FOR_BRISC)",
        "const uint32_t mover = 0;",
        "#else",
        "const uint32_t mover = 1;",
        "#endif",
        "const uint32_t stamp = get_write_ptr(kStamps) + mover * kStampsBytes;",
        "volatile tt_l1_ptr uint32_t* stamped = km::l1(stamp);",
        "stamped[0] = command.sequence;",
        f"stamped[1] = {overlay};",
        "stamped[2] = static_cast<uint32_t>(loaded - started);",
        "stamped[3] = static_cast<uint32_t>(called - loaded);",
        "stamped[4] = held;",
        f"const {camel(tensor)} stamps(command.a{word});",
        "if (!km::write_pages(movement, command, stamps, 2 * movement.me() + mover, 1,",
        "                     stamp)) {",
        f"{INDENT}return false;",
        "}",
        "noc_async_write_barrier();",
        "return called_ok;",
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
        '#include "komorebi/device/km_movement.hpp"',
    ]
    if "port" in compiled.includes:
        lines.append('#include "komorebi/device/km_port.hpp"')
    if compiled.includes & {"broadcast", "ask"}:
        lines.append('#include "komorebi/device/km_ask.hpp"')
    if "broadcast" in compiled.includes:
        lines.append('#include "komorebi/device/km_broadcast.hpp"')
    if compiled.overlays:
        lines.append('#include "komorebi/device/km_overlay.hpp"')
    return lines + compiled.body_includes["movement"]


def compute_includes(compiled):
    lines = ["#include <stdint.h>", ""]
    for path, needs in COMPUTE_INCLUDES:
        if needs is None or needs in compiled.includes:
            lines.append(f'#include "{path}"')
    if compiled.overlays:
        lines.append('#include "komorebi/device/km_overlay.hpp"')
    return lines + compiled.body_includes["compute"]


# A movement part's first lines: the resident case's km::Movement, as its runtime.
PART_MOVEMENT = [
    INDENT + "// the resident case's km::Movement, as its mover's runtime",
    INDENT + "km::Movement& movement = static_cast<km::Movement&>(core);",
]


def parts(compiled):
    """{path: text} of each overlay handler's parts: one movement source (built for
    each mover) and, if compute has code, a compute source."""
    files = {}
    for name in compiled.overlays:
        made = f"// {compiled.name}.{name}, made by komorebi.dsl from " + (
            f"{Path(compiled.source).name}: overlay kOverlay{camel(name)},"
        )
        files[f"{name}/movement.cpp"] = (
            "\n".join(
                [
                    made,
                    "// its part on each mover (mover A: RISCV_0, mover B: RISCV_1).",
                    *movement_includes(compiled),
                    f'#include "../{header_name(compiled)}"',
                    "",
                    "uint32_t km_overlay("
                    "km::MoverRuntime& core, const km_command_t& command) {",
                    *PART_MOVEMENT,
                    *(INDENT + line for line in tensor_aliases(compiled)),
                    *render(prune(compiled.movement[name]), 1),
                    f"{INDENT}return true;",
                    "}",
                ]
            )
            + "\n"
        )
        if has_part(compiled.compute[name]):
            files[f"{name}/compute.cpp"] = (
                "\n".join(
                    [
                        made,
                        "// its part on compute.",
                        *compute_includes(compiled),
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


def dynamic_source(compiled):
    """The compute kernel of a dynamic class: its handler's message loop (the reader
    and writer are the runtime's, messages_reader.cpp and messages_writer.cpp)."""
    name = compiled.dynamic
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// the compute kernel: the message loop of its dynamic handler.",
        "#include <stdint.h>",
        "",
        '#include "api/compute/common.h"',
        '#include "komorebi/device/km_messages.hpp"',
        '#include "komorebi/device/km_worker.hpp"',
        *compiled.body_includes["compute"],
        f'#include "{header_name(compiled)}"',
        "",
        "void kernel_main() {",
        f"{INDENT}km::Compute compute;",
        f"{INDENT}compute.run([&](const km_command_t& command) {{",
        f"{INDENT * 2}if (km::handler(command) != kHandler{camel(name)}) {{  "
        f"// line {compiled.lines[name]}",
        f"{INDENT * 3}return;",
        f"{INDENT * 2}}}",
        *render(prune(compiled.compute[name]), 2),
        f"{INDENT}}});",
        "}",
    ]
    return "\n".join(lines) + "\n"


def runtime_kernels():
    """The runtime's message movers, the reader and writer of a dynamic class."""
    folder = _env.include_directory().parent / "kernels"
    return folder / "messages_reader.cpp", folder / "messages_writer.cpp"


def movement_source(compiled):
    lines = [
        f"// {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        "// the movement source of both movers (mover A: RISCV_0, mover B: RISCV_1).",
        *movement_includes(compiled),
    ]
    lines += [f'#include "{header_name(compiled)}"', "", "void kernel_main() {"]
    lines += [INDENT + line for line in tensor_aliases(compiled)]
    lines += [
        f"{INDENT}km::Movement movement;",
        f"{INDENT}movement.run([&](const km_command_t& command) {{",
    ]
    lines += cases(compiled, compiled.movement, "return true;", 2, "movement")
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
        *compute_includes(compiled),
    ]
    lines += [f'#include "{header_name(compiled)}"', "", "void kernel_main() {"]
    reads = [b for b in compiled.buffers if b.role == "read"]
    writes = [b for b in compiled.buffers if b.role in ("write", "out")]
    others = [b for b in compiled.buffers if b.role == "broadcast"]
    first_in = (
        f"{reads[0].name}.mover_a" if reads else others[0].name if others else "0"
    )
    first_out = "16"
    if writes:
        first_out = writes[0].name
        if writes[0].index_b is not None:
            first_out += ".mover_a"
    lines += [
        f"{INDENT}km::Compute compute;",
        f"{INDENT}compute_kernel_hw_startup({first_in}, {first_out});",
        f"{INDENT}compute.run([&](const km_command_t& command) {{",
    ]
    lines += cases(compiled, compiled.compute, "break;", 2, "compute")
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
    return built.lines(error.cores) if built is not None else []
