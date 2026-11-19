"""The source text of a program's parts: the movement and compute kernels, the chain
cases and the rectangles they cover, as the program writes them."""

import ast
import re
from pathlib import Path

from komorebi import abi, layout
from komorebi.dsl.chain.chain_stamps import STAMP_BODY, STAMP_LOADED, STAMPS_PAGE
from komorebi.dsl.compile.layout import REGION_PAGE
from komorebi.dsl.files import (
    camel,
)
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import prune, render
from komorebi.errors import ProgramError

INDENT = "    "


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
            elif state["rect"] is None and state.get("ranks") is not None:
                state["args"], state["group"], state["rect"] = ranks_rectangle(
                    compiled, nocs, state["ranks"]
                )
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


def ranks_rectangle(compiled, nocs, ranks):
    """A class's default rectangle when the load's `ranks` are fewer than the grid's
    cores (a Tensix dispatcher's core is not one), as flat_rectangle's: a broadcast's
    the whole rows of ranks, else the whole grid with the ranks' group. (args, group,
    rect: None the whole grid)"""
    columns, rows = nocs.grid
    if compiled.broadcasts:
        rect = (0, 0, columns, ranks // columns)
        args, group = rectangle_args(compiled, rect, nocs)
        return args, group, rect
    args, _ = rectangle_args(compiled, (0, 0, columns, rows), nocs)
    return args, (0, ranks), None


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


def header_name(compiled):
    return f"{compiled.name.lower()}.h"


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
    # a float32 operand's tf32 source: every other unpack sets its own again
    tf32 = "tf32" in compiled.includes
    return len(unpacked) > 1 or tf32, len(packed) > 1


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


COMPUTE_INCLUDES = [
    ("api/compute/bcast.h", "bcast"),
    ("api/compute/binary_max_min.h", "max"),
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
    ("komorebi/device/core/km_tf32.hpp", "tf32"),
]


CHAIN_HANDLER = "km_chain"  # its handler's name: kHandlerKmChain


READ_MOVES = ("read stream", "word read", "fill", "broadcast's read", "table read")


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
