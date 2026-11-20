"""Kernels, slots and links of a load: kernel sizes and builds, the slots of overlays
and own code, link kernels and defines, an island's words and the groups' recipes."""

import dataclasses
import re
from pathlib import Path

from komorebi import _env, abi
from komorebi.device import chips, remembered
from komorebi.errors import ProgramError, SlowCodeError, StartError
from komorebi.program import overlay
from komorebi.program import several as programs
from komorebi.program.addon import program as added_program
from komorebi.program.checks import near as near_module
from komorebi.program.description import KernelSizes, unguarded
from komorebi.session import sizes

GROUP_DEFINE = "-DKM_PROGRAM_GROUP"  # a group's runtime kernels are built apart
GROUP_LINE = re.compile(r"^#define KM_PROGRAM_GROUP (\d+)$", re.M)
OPT_DEFINE = "KM_OPT_LEVEL"  # defined to a kernel's level when the program sets one
# A taken branch to code at or above this L1 address costs about 4 cycles more: the
# kernels and the slots go below it, the circular buffers above them.
L1_FAST_CODE = 1 << 20

TENSIX_THREADS = ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2")


def kernel_sizes(build):
    """A KernelSizes from the native kernel_build's (config bytes, buffer, text per
    thread, the runtime's message); the total from that message when it does not fit."""
    total, limit, text, error = build
    if error:
        said = re.search(r"Program size \((\d+)\) too large", error)
        total = int(said.group(1)) if said else None
    return KernelSizes(dict(zip(TENSIX_THREADS, text, strict=False)), total, limit)


def _core_args(program, placed, core):
    """(reader, writer, compute) program arguments of `core`: its group's program's
    when the run holds programs in the runtime's form."""
    if placed is None:
        return program.core_args(core)
    if len(placed.programs) > 1:
        return programs.core_args(placed, core)
    for group in placed.groups:
        if group.first <= core < group.first + group.count and group.programs:
            return placed.programs[group.programs[0]].core_args(core)
    return [], [], []


# A link CPU's own chunk on the link cores (Run._link_chunk): at most, and what is
# left besides; at least the least that holds a put ring (link_chunk_least).
LINK_CHUNK_MOST = 512 << 10


LINK_CHUNK_MARGIN = 64 << 10


# km_pages.hpp's ring: kRingChunkMin, kRingSlotsMin, kRingHeader
RING_CHUNK_MIN, RING_SLOTS_MIN, RING_HEADER = 128 << 10, 8, 128


def link_chunk_least(packet):
    """The least link chunk (whole 64 KiB) whose put ring has RING_SLOTS_MIN slots of
    `packet` bytes in its half (km_pages.hpp ring_slots): 192 KiB for 8 KiB packets."""
    ring = 2 * RING_SLOTS_MIN * (RING_HEADER + packet)
    return max(RING_CHUNK_MIN, -(-ring // (64 << 10)) * (64 << 10))


# The defines of an island run that the link cores' own kernels are built with too.
LINK_DEFINES = (
    "KM_PAGE_LINKS",
    "KM_TENSOR_TABLE",
    "KM_ASKS_BY_HOST",
    "KM_TENSORS_MAX",
    "KM_ISLANDS_MAX",
    "KM_TABLE_CACHE",
    "KM_FORWARD",
    "KM_FORWARD_CORES",
    "KM_FORWARD_CPUS",
    "KM_FORWARD_LEVELS",
    "KM_FAR_PUTS",
    "KM_LINKS_A_CORE",
    "KM_LINK_NO_FAR_DRAM",
    "KM_LINK_THIN_SECOND",
    "KM_COLL_QUIET",
)


# Every define the island load gives a program; a plan's runtime kernels get them.
ISLAND_DEFINES = (*LINK_DEFINES, "KM_WIRE")


def _link_kernels(island, program, split, cores):
    """(ranks, link cores) of a load whose link cores get the runtime's link kernels
    (the link services) and its ranks the program's (their client side): a program
    built with page links on an island of its own card, not dynamic (its link cores
    carry its messages); else (cores, 0): every core the program's kernels."""
    if (
        island is None
        or split is not None
        or program.messages is not None
        or "KM_PAGE_LINKS" not in program.defines
    ):
        return cores, 0
    graph, here = island
    links = len(graph.link_cores[here])
    return (cores - links, links) if links else (cores, 0)


def _cbs_end(cbs):
    """The end of circular buffers `cbs` ((index, format, page bytes, pages)) as
    tt-metal places them from 0, in creation order."""
    return programs.buffer_layout(cbs, 0)[1]


def check_link_defines(name, link_defines, placed):
    """Refuses link cores' kernels built without a link define a program `placed` has,
    or without the put queue (KM_LINK_QUEUE) its ranks' put lines go through."""
    have = {define for define, _ in link_defines}
    missing = sorted(
        {define for p in placed for define in p.defines if define in LINK_DEFINES}
        - have
    )
    if "KM_PAGE_LINKS" in have and "KM_LINK_QUEUE" not in have:
        missing.append("KM_LINK_QUEUE")
    if missing:
        raise ProgramError(
            f"{name}: its link cores' kernels would be built without "
            f"{', '.join(missing)}: the ranks' asks and put lines would never be served"
        )


def _link_kernel_path(kernel):
    """The runtime's link kernel of `kernel` (reader, writer, compute)."""
    return _env.include_directory().parent / "kernels" / "link" / f"{kernel}.cpp"


def _is_link_kernel(source):
    """Whether a kernel `source` of a build is one of the runtime's link kernels."""
    return any(
        Path(source) == _link_kernel_path(kernel)
        for kernel in ("reader", "writer", "compute")
    )


def _runs_no_program(source):
    """Whether a kernel `source` of a build runs no program: a link kernel, or the
    Tensix dispatcher's kernels on its own core, its TRISC thread's among them (built
    from td-thread/thread-*.cpp): no overlay is built against them."""
    dispatcher = _env.include_directory().parent / "kernels" / "dispatcher"
    path = Path(source)
    thread = path.parent.name == "td-thread" and path.name.startswith("thread-")
    return _is_link_kernel(source) or path.parent == dispatcher or thread


def kernel_build(program, kernel, defines, arch):
    """A kernel's defines (the chip `arch`'s for its writer added), its optimisation
    level (level_build: the program's, else the chip's for its writer), and its NoC
    when the program sets them."""
    level = program.kernel_opt(kernel)
    if getattr(program, "near", None) and kernel in ("reader", "writer"):
        # built without the far paths: the link cores keep the page links' defines
        kept = [define for define in defines if define[0] not in near_module.DROPPED]
        defines = [*kept, (near_module.DEFINE, "1")]
    build = chip_build(defines, level, kernel, arch, writer_level(program))
    if program.noc and kernel != "compute":
        build["noc"] = program.kernel_noc(kernel)
    return build


def chip_build(defines, level, kernel, arch, cold=0):
    """level_build of `kernel` on the chip `arch`: the writer takes the chip's defines
    and, when `level` (the program's) is None, its level (chips.Chip.writer_opt,
    writer_defines); both movers KM_WRITER_COLD at writer level `cold` above 0; compute
    its KM_DST_WINDOW (km_dst.hpp); others as given."""
    chip = chips.chip(arch)
    if kernel == "compute":
        return level_build([*defines, *compute_defines(arch)], level)
    if kernel not in ("reader", "writer"):
        return level_build(defines, level)
    if cold:
        defines = [*defines, ("KM_WRITER_COLD", str(cold))]
    if kernel == "reader":
        return level_build(defines, level)
    return level_build([*defines, *chip.writer_defines], level or chip.writer_opt)


# The writer's paths after a command's work that run from the service overlay's
# writer part in L1, by writer level (KM_WRITER_COLD; 0: every one resident).
WRITER_LEVELS = (
    (),
    ("the missed page bound",),
    ("the missed page bound", "Core::finish"),
    ("the missed page bound", "Core::finish", "complete"),
    ("the missed page bound", "Core::finish", "complete", "a chain record's marks"),
)


def writer_level(program):
    """The writer level `program` is built at (Program.writer_level; 0 when unset)."""
    return getattr(program, "writer_level", 0) or 0


def writer_over(error):
    """Whether a build's `error` is the runtime's refusal of a writer (NCRISC) ELF over
    its code memory, the error a higher writer level answers."""
    text = str(error)
    return "overflows region:0" in text and "/ncrisc/" in text


# The runtime's refusal of a writer (NCRISC) ELF over its code memory: the kernel's
# directory, the ELF's bytes and the region's
WRITER_OVER = re.compile(
    r"kernels/([^/\s]+)/\d+/+ncrisc/ncrisc\.elf: segment\[\d+\] "
    r"\[0x[0-9a-f]+,\+0x([0-9a-f]+)\) overflows region:\d+ limit of 0x([0-9a-f]+) bytes"
)


def writer_over_refusal(error, name, level):
    """A ProgramError for a build of program `name` whose writer is over its code memory
    (`error`, the runtime's) at writer level `level`, the chip's highest: the kernel,
    both sizes and the level; else None."""
    found = WRITER_OVER.search(str(error))
    if found is None:
        return None
    wanted, most = int(found[2], 16), int(found[3], 16)
    return ProgramError(
        f"{name}: the writer ({found[1]}) is {wanted:,} B of NCRISC's {most:,} B code "
        f"region at writer level {level}, the highest: {wanted - most:,} B over"
    )


# The runtime's refusal of any ELF over a region of its CPU's memory (tt_elffile)
ELF_OVER = re.compile(
    r"\S+\.elf: segment\[\d+\] \[0x[0-9a-f]+,\+0x[0-9a-f]+\) overflows region:\d+ "
    r"limit of 0x[0-9a-f]+ bytes"
)

# Each build this process's runtime refused as an ELF over its region (key: sentence):
# its ELF cache keeps an empty slot, and a later load waits forever (hazards.md)
_elf_over = {}


class AlreadyOver(RuntimeError):
    """A build the runtime refused in this process as an ELF over its region, asked
    for again: refused before the runtime (its message: the first refusal's)."""


def over_key(arch, programs, level, near):
    """The key of a build of `programs` (one load's) on `arch` at writer level `level`,
    `near` or not: what the ELFs it hands the runtime depend on (None: a program the
    remembered key does not name, which this guard cannot follow)."""
    keys = tuple(remembered.program_key(each, f"build {arch}") for each in programs)
    return None if None in keys else (arch, keys, level, near is not None)


def remember_over(key, error):
    """Keeps `error`'s sentence for build `key` when it is the runtime's refusal of an
    ELF over its region (ELF_OVER), the first one kept."""
    found = ELF_OVER.search(str(error))
    if key is not None and found is not None and not isinstance(error, AlreadyOver):
        _elf_over.setdefault(key, found[0])


def already_over(key):
    """An AlreadyOver for build `key` when the runtime refused it in this process
    (remember_over), else None."""
    said = _elf_over.get(key) if key is not None else None
    return (
        None
        if said is None
        else AlreadyOver(f"already refused in this process: {said}")
    )


def writer_over_bytes(error):
    """(the writer's bytes, its region's) of a build's `error`, the runtime's refusal
    of a writer over its code memory (WRITER_OVER)."""
    found = WRITER_OVER.search(str(error))
    return int(found[2], 16), int(found[3], 16)


def compute_defines(arch):
    """A compute kernel's defines on the chip `arch`: KM_DST_WINDOW, its dst_window."""
    return [("KM_DST_WINDOW", "1" if chips.chip(arch).dst_window else "0")]


def level_build(defines, level):
    """A kernel's defines and optimisation level `level` (None: the runtime's). A set
    level also adds a define naming it: the runtime's build cache key leaves the level
    out, and two levels must not share a cached ELF."""
    if level is None:
        return {"defines": defines}
    return {"defines": [*defines, (OPT_DEFINE, level)], "opt_level": level}


def within_indices(cbs, most, whose=""):
    """`cbs` ((index, ...) each) when every index is below `most`, the chip's buffer
    indices (Device.buffer_indices: 64 on Blackhole, 32 on Wormhole); else refused,
    `whose` buffer it is first."""
    for index, *_ in cbs:
        if not 0 <= index < most:
            raise ProgramError(
                f"{whose}buffer index {index} is past the chip's {most} buffer "
                f"indices (0 to {most - 1})"
            )
    return cbs


def tile_words_checked(programs, arch):
    """Refuses a load whose tile functions (Program.tile_words, a class's) emit a word
    the chip `arch` runs otherwise than Blackhole (chips.Chip.tile_refusal): the
    program, the function, the operation and the instruction by name."""
    refusal = chips.chip(arch).tile_refusal
    for program in programs:
        for function, runs in getattr(program, "tile_words", ()):
            for run in runs:
                for before, word in zip([None, *run], run):
                    why = refusal(before, word)
                    if why is not None:
                        raise ProgramError(
                            f"{program.name}: tile function {function}: "
                            f"{word[2] or word[0]}: {why}; on a {arch} chip this "
                            "word would compute another result"
                        )


def one_island(island, split):
    """KM_ONE_ISLAND for a load that is no island of several (no `island` given, not
    `split` over islands): its kernels take the island table's one entry as known
    (km_island.hpp), the code of several islands left out."""
    if island is not None or split is not None:
        return []
    return [("KM_ONE_ISLAND", "1")]


def own_slot_code(program):
    """Whether `program` may give the service overlay its own slot: it has overlays of
    its own, added handlers or slots(service="own"); else the service shares the
    reader's slots, the same layout with no program slots (pay for use)."""
    return (
        bool(program.overlays)
        or "KM_ADDED_HANDLERS" in program.defines
        or program.overlay_slots.service == "own"
    )


def own_slot_path(program):
    """The kernels' define of the own slot's code (km_overlay.hpp's overlay_own),
    for a program that may use it (own_slot_code); a placed load always has it."""
    return [("KM_OVERLAY_OWN", "1")] if own_slot_code(program) else []


def slots_most(program, placed=None):
    """The slots of one part the load's kernels are built for (KM_OVERLAY_SLOTS over
    4): as the overlays want them on a core, or as slots(count=) declares; a load of
    several programs: its groups' (placed_slots)."""
    fixed = overlay.constants()["kOverlaySlotsFixed"]
    if placed is not None:
        return max([fixed, *placed_slots(placed).values()])
    declared = getattr(program, "overlay_slots", None)  # None: a class not yet made
    if declared is None:
        return fixed
    own = list(program.overlays.values())
    counts = [
        declared.of(kernel, own)[0]
        for kernel in overlay.KERNELS
        if declared.reserved(kernel) or any(kernel in each.sources for each in own)
    ]
    return max([fixed, *counts])


def placed_slots(placed):
    """{kernel: count} a load of several programs wants, before its kernels are built:
    a switching run's (a declared count, else one a program, one more for an own
    overlay named ahead); else the largest group's, one an overlay with a part there.
    A merged program's spilled sources come with a declared count (komorebi.merge)."""
    declared = placed.programs[0].overlay_slots
    count = declared.count
    out = {}
    if placed.switch:
        ahead = added_program.ahead_own_kernels(placed)
        for kernel in overlay.KERNELS:
            wanted = count.get(kernel) if isinstance(count, dict) else count
            out[kernel] = wanted or len(placed.programs) + (kernel in ahead)
        return out
    single = len(placed.programs) == 1
    for group in placed.groups:
        held = [placed.programs[index] for index in group.programs]
        overlays = [
            *(
                part
                for each in held
                for part in programs.program_overlays(each, False, single)
            ),
            *(made for each in held for made in programs.own_overlays(each)),
        ]
        for kernel in overlay.KERNELS:
            if declared.reserved(kernel) or any(
                kernel in each.sources for each in overlays
            ):
                wanted = declared.of(kernel, overlays)[0]
                out[kernel] = max(out.get(kernel, 0), wanted)
    return out


def rings_dropped(backend):
    """Whether a split card's load refused for its L1 table is built again without the
    link cores' put rings: their chunks lie on cores that hold the slots too."""
    return bool((backend.link_queue_at or (0, 0))[1] and sizes.link_rows(backend))


def own_slot_declared(program):
    """`program`'s declared slots, the service shared when its kernels lack the
    own slot's code (own_slot_code)."""
    if own_slot_code(program):
        return program.overlay_slots
    return dataclasses.replace(program.overlay_slots, service="shared")


def added_path(placed):
    """The runtime's kernels' define of the path of a handler added to the live run
    (run.addon), when every program of the load names slot bytes for it, as the
    load's room requires (added.room); a program's own defines stay its parts'."""
    if placed and all("KM_ADDED_HANDLERS" in p.defines for p in placed):
        return [("KM_ADDED_HANDLERS", "1")]
    return []


def guard_path(programs):
    """The runtime's kernels' define of KM_GUARD 0 when the load's programs are built
    with guard=False (programs.same_guard: they all are, or none); a program's own
    parts get it from its defines."""
    return [("KM_GUARD", "0")] if unguarded(programs) else []


def slot_area(base, size, used, slow_code, most=None):
    """The slots' (address, bytes, buffer index): `size` bytes as the first circular
    buffer, at the L1 base `base`, on an index not in `used` (slot_index, `most` the
    chip's indices); refused (SlowCodeError) when they end above L1_FAST_CODE, unless
    `slow_code`. The buffers above start 64-byte aligned (a Wormhole's base is not)."""
    size = -(-(base + size) // overlay.ALIGN) * overlay.ALIGN - base
    if base + size > L1_FAST_CODE and not slow_code:
        raise SlowCodeError(base, size, L1_FAST_CODE)
    return base, size, slot_index(used, most=most)


def slot_index(used, reserved=(), most=None, whose=""):
    """The slots' buffer index, none of `reserved` (the link cores'): the highest
    free below the highest in `used` (the kernel config keeps its size), else the first
    free above it below `most`, the chip's indices (Device.buffer_indices); else
    refused, `whose` load named first."""
    taken = set(used) | set(reserved)
    free = set(range(max(used))) - taken
    if free:
        return max(free)
    above = set(range(max(used), most or 0)) - taken
    if not above:
        below = max(used) if most is None else most
        chip = f" (the chip's {most} buffer indices)" if most else ""
        raise ProgramError(
            f"{whose}no circular buffer index for the slots: the load's buffers and "
            f"the runtime's take every index from 0 to {below - 1}{chip}"
        )
    return min(above)


def lim_bases(lim_base, abi_headers=None):
    """The X280 local-memory addresses the kernels are given or derive from the
    runtime's base: the base, every KMD_LIM_* offset from it, the ack stride."""
    abi_headers = abi_headers or abi.headers()
    bases = {"lim_base": lim_base, "KM_ACK_STRIDE": abi_headers["KM_ACK_STRIDE"]}
    for name, offset in abi_headers.constants.items():
        if name.startswith("KMD_LIM_"):
            bases[name] = lim_base + offset
    return bases


def check_bases(bases):
    """Rejects bases that are not multiples of NOC_ALIGN: the runtime's small writes
    go slot to slot and rely on them."""
    bad = {
        name: address for name, address in bases.items() if address % sizes.NOC_ALIGN
    }
    if bad:
        named = ", ".join(f"{name} {address:#x}" for name, address in bad.items())
        raise StartError(f"not a multiple of {sizes.NOC_ALIGN} bytes: {named}")


def go_layout(panels, columns, cores, abi_headers=None):
    """KM_ISLAND_GO of an island of `cores` cores (whole rows of `columns`) over the
    grid's `panels` (column ranges, at most two, in order): what km::team_go
    multicasts its release over; 0 (one atomic a rank) when they do not tile the
    grid."""
    abi_headers = abi_headers or abi.headers()
    starts = [first for first, _ in panels]
    ends = [last for _, last in panels]
    tiled = (
        starts[:1] == [0]
        and starts[1:] == [end + 1 for end in ends[:-1]]
        and ends[-1:] == [columns - 1]
    )
    rows = cores // columns if columns else 0
    if not (len(panels) <= 2 and tiled and rows * columns == cores):
        return 0
    if not (1 <= rows < 256 and columns < 256):
        return 0
    split = panels[1][0] if len(panels) == 2 else 0
    return (
        columns
        | rows << abi_headers["KM_ISLAND_GO_ROWS"]
        | split << abi_headers["KM_ISLAND_GO_SPLIT"]
        | abi_headers["KM_ISLAND_GO_ON"]
    )


def slot_rows(slots, service=0, why=""):
    """The L1 budget table's rows of overlay slots ({part: (count, bytes)}) and of the
    service overlay's own reader slot (`service` bytes; 0: the reader's, `why`)."""
    rows = [(f"slots: service (reader), 1 x {service:,} B", service)] if service else []
    if why:
        place = "its own" if service else "in the reader slots"
        rows = [(f"slots: service (reader), {place}: {why}", 0)] + rows
    return rows + [
        (f"slots: {part}, {count} x {size:,} B", count * size)
        for part, (count, size) in slots.items()
        if count
    ]


def built_files(recipes, started):
    """(found, compiled) of the ELFs of a launch's `recipes()`: compiled when written
    from `started` (time.time()) on, else found in the build cache."""
    elves = {Path(row[1]) for _, builds in recipes for row in builds}
    compiled = sum(elf.is_file() and elf.stat().st_mtime >= started for elf in elves)
    return len(elves) - compiled, compiled


def fit_slots(built, declared, free, rows, log):
    """`built` with the undeclared slot counts lowered until `rows` (the L1 table's
    rows besides the slots) and the slots fit `free` bytes, where they can."""
    room = free - sum(size for _, size in rows)
    return overlay.choose(built, declared, room, log)


def _recipe_group(recipe):
    """The group number a runtime kernel's build was made for (KM_PROGRAM_GROUP, in
    its defines or in the header the runtime generates beside its ELF), or None."""
    for word in recipe.defines:
        if word.startswith(GROUP_DEFINE + "="):
            return int(word.split("=", 1)[1])
    header = recipe.elf.parent.parent / "defines_generated.h"
    found = GROUP_LINE.search(header.read_text()) if header.is_file() else None
    return int(found.group(1)) if found else None


def _island(island, grid, dispatcher_cores=0):
    """Session.load's `island` (graph, i) checked: island i's ranks are this card's
    first cores in core order, its link cores the rest in order (but a Tensix
    dispatcher's core, the last: `dispatcher_cores` 1), so rank first + c is core c and
    link core k is core ranks + k (the dispatcher's mapping)."""
    if island is None:
        return None
    graph, here = island
    cores = grid[0] * grid[1] - dispatcher_cores
    if not 0 <= here < len(graph.islands):
        raise ValueError(f"island {here} of {len(graph.islands)}")
    if graph.rank_cores[here] + graph.link_cores[here] != tuple(range(cores)):
        raise ValueError(
            f"island {here}: its ranks, then its link cores, are not the {cores} "
            "cores in order"
        )
    return graph, here


def _island_word(island):
    """The dispatcher's island word: first | count << 16 of its ranks; 0 for none."""
    if island is None:
        return 0
    graph, here = island
    return graph.first(here) | graph.count(here) << 16


class _Dispatches(list):
    """The dispatches of a card split into islands, closed together (a load's)."""

    close_failed = ""

    def close(self):
        failed = [d.close_failed for d in self if not d.close()]
        self.close_failed = "; ".join(failed)
        return not failed


# From this writer level a writer's parts take the writer's own level: at O2 a Wormhole
# part's km::chain_mark hangs a chain now and then (hazards.md; why is not known).
PART_OWN_LEVEL_FROM = 4


def part_recipes(recipes, arch):
    """`recipes` ({target: Recipe}) with the writer's parts at the chip `arch`'s level
    for them (chips.Chip.writer_part_opt, when it has one; the writer's own from writer
    level PART_OWN_LEVEL_FROM)."""
    level = chips.chip(arch).writer_part_opt
    if level is None or "ncrisc" not in recipes:
        return recipes
    if recipe_writer_level(recipes["ncrisc"]) >= PART_OWN_LEVEL_FROM:
        return recipes
    return {**recipes, "ncrisc": recipes["ncrisc"].at(level)}


def recipe_writer_level(recipe):
    """The writer level a kernel's Recipe was built at (its KM_WRITER_COLD; 0: none)."""
    for define in recipe.defines:
        if define.startswith("-DKM_WRITER_COLD="):
            return int(define.split("=", 1)[1])
    return 0


def _group_recipes(kernels, count):
    """{target: Recipe} of each group's runtime kernels, by group number: the
    runtime lists a program's kernels in no fixed order."""
    out = [{} for _ in range(count)]
    for source, builds in kernels:
        if _runs_no_program(source):  # the link cores', the dispatcher's core
            continue
        for row in builds:
            recipe = overlay.Recipe.from_native(row)
            number = _recipe_group(recipe)
            if number is None or not 0 <= number < count:
                raise ProgramError(f"{recipe.elf}: no group number in its build")
            out[number][recipe.target] = recipe
    return out


def _held_fields(placed, links=0):
    """A byte per core for the dispatcher's table: the program field it holds, 0xFF
    none. An island's `links` link cores, after its ranks, run the runtime's link
    kernels whatever program is loaded: they hold field 0, the field the session's
    own commands to them carry."""
    out = [0xFF] * placed.cores + [0] * links
    for group in placed.groups:
        for program_field in group.programs:
            out[group.first : group.first + group.count] = [program_field] * group.count
    return out


def _start_loads(run, placed, part=False):
    """In a run whose groups switch, each group loads its program first: the compute
    parts set the hardware up for its formats (the runtime's kernels do not); `part`:
    the run is an island's, its groups a part of the flat ones."""
    first = run.island_first  # the run takes flat ranks; an island's are a part
    handles = [
        run.load(
            placed.programs[group.programs[0]],
            (first + group.first, group.count),
            part=part,
        )
        for group in placed.groups
        if group.programs
    ]
    run.wait(handles, timeout=10.0)


def _common_slots(placed, declared, built, most):
    """A switching run's slots of the largest part any program has there: the declared
    count, else one a program up to `most` (the kernels' slots), one more where a class
    named ahead has an own overlay, and 2 at least where a loaded program has one (its
    part loads into another slot)."""
    slots = {}
    for made in built:
        for part, (_, size) in made.slots.items():
            slots[part] = max(slots.get(part, 0), size)
    count = declared.count
    service = max(made.service for made in built)
    own = added_program.ahead_own_kernels(placed)
    loaded = _loaded_own_kernels(placed)
    out = {}
    for part, size in slots.items():
        kernel = next(k for k, parts in overlay.KERNEL_PARTS.items() if part in parts)
        wanted = count.get(kernel) if isinstance(count, dict) else count
        if wanted == 1 and kernel in own | loaded:
            raise ProgramError(
                f"Program.slots(count=1) for the {kernel}: a program of this run has "
                f"an overlay of its own with a {kernel} part, which its part loads "
                "into another slot while it runs; give it 2 or more"
            )
        more = 1 if kernel in own else 0
        default = min(len(placed.programs) + more, most)
        if kernel in loaded:
            default = max(default, 2)
        out[part] = (wanted or default, size)
    return overlay.Built({}, out, service, service)


def _own_slots_kept(placed, common):
    """Refuses slots the L1 lowered to one in a kernel where a program's own overlay
    loads beside its program overlay: its first call would find no slot."""
    own = added_program.ahead_own_kernels(placed) | _loaded_own_kernels(placed)
    for part, (count, _) in common.slots.items():
        kernel = next(k for k, parts in overlay.KERNEL_PARTS.items() if part in parts)
        if kernel in own and count < 2:
            raise ProgramError(
                f"the {part} slots lowered to {count}: a program of this run loads an "
                f"overlay of its own there beside its program overlay, which needs 2; "
                "the L1 left allows no more (fewer or smaller buffers, or smaller "
                "parts)"
            )


def _loaded_own_kernels(placed):
    """The kernels where a program the run loads has a part of an overlay of its own
    (its handlers' overlays, besides its program overlay)."""
    return {
        kernel
        for program in placed.programs
        for each in program.overlays.values()
        for kernel in each.sources
    }
