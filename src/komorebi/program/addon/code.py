"""Code added to a live run (Run.addon): an overlay, or a handler whose body is one,
built against the loaded kernels, its parts written into the room the load reserved
in the overlays' DRAM tensor, and each core given its table entry by a command of the
runtime's service overlay (KM_HANDLER_ENTRY). The host writes no core's L1.
"""

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from komorebi import abi
from komorebi.device import native
from komorebi.errors import ProgramError
from komorebi.program import overlay
from komorebi.program.checks import instructions as insn_check
from komorebi.program.description import messages_header, unguarded

# The load's default room for added parts: bytes of parts, each stored in every bank.
RESERVE = 256 * 1024


class AddError(ProgramError):
    """An add the run refuses: nothing was written to the card."""


@dataclass
class Group:
    """Cores [first, first + count) with one build of the loaded kernels: its recipes
    {part: Recipe}, its slots {part: (count, bytes)}, the overlay numbers its table
    holds."""

    first: int
    count: int
    recipes: dict
    slots: dict
    numbers: set


@dataclass
class Room:
    """The room a load reserved: the overlays' DRAM tensor, its free rows `next` to
    `end` (a row: a page in each bank), the groups, the programs' handler numbers,
    whether the kernels have the added handlers' path (KM_ADDED_HANDLERS), the dynamic
    handlers that are parts (None: none), what was added {name: Added}."""

    image: object
    next: int
    end: int
    groups: list
    handlers: set
    entered: bool = True
    messages: set = None
    added: dict = field(default_factory=dict)

    @property
    def row_bytes(self):
        return self.image.page_bytes

    def left(self):
        return (self.end - self.next) * self.row_bytes


@dataclass(frozen=True)
class Added:
    """An added overlay: its number, whether it is a handler or a dynamic handler
    (`message`), its parts' bytes, and the host's times (s): the build (a look in the
    cache when `cached`), the write of its parts, the entry commands."""

    name: str
    number: int
    handler: bool
    message: bool
    parts: dict
    build_s: float
    write_s: float
    entry_s: float
    cached: bool


def room(backend, image, rows, groups, programs):
    """The Room of a load whose overlays' tensor `image` holds `rows` rows of parts,
    then its reserve; None when it reserved nothing."""
    end = image.pages // backend.card.dram_banks  # a row a period of the banks
    if end <= rows:
        return None
    handlers, messages = set(), None
    for program in programs:
        handlers |= set(program.handlers.values())
        if program.messages is not None:
            handlers |= set(program.messages.handlers)
            if program.messages.parts:
                messages = set(program.messages.handlers)
    entered = all("KM_ADDED_HANDLERS" in program.defines for program in programs)
    return Room(image, rows, end, groups, handlers, entered, messages)


def group(first, count, recipes, built):
    """A Group from a load's recipes {target: Recipe} and its Built."""
    return Group(
        first,
        count,
        {overlay.TARGETS[target]: recipe for target, recipe in recipes.items()},
        dict(built.slots),
        {number for number, _ in built.images},
    )


def _number(room, name, handler, number):
    """The added overlay's number: `number`, or the lowest free one."""
    most = overlay.constants()["kOverlayService"] - 1
    taken = {n for each in room.groups for n in each.numbers}
    taken |= {each.number for each in room.added.values()}
    refused = taken | (room.handlers if handler else set())
    if number is not None:
        if not 1 <= number <= most:
            raise AddError(f"{name}: number {number} is not in 1..{most}")
        if number in taken:
            raise AddError(f"{name}: number {number} is an overlay of the run already")
        if number in refused:
            raise AddError(
                f"{name}: number {number} is a handler of the run's programs; an added "
                "handler takes a number they do not serve"
            )
        return number
    free = [n for n in range(1, most + 1) if n not in refused]
    if not free:
        raise AddError(
            f"{name}: no overlay number is left: the run's 1 to {most} are taken "
            f"({len(taken)} by overlays and programs"
            + (f", {len(refused - taken)} by handlers" if handler else "")
            + ")"
        )
    return free[0]


def _message_number(room, name, number):
    """A dynamic handler's number, also its overlay's: `number`, or the lowest free
    one, 1 to kMessagesHandlersMax (a packet groups its messages by it)."""
    if room.messages is None:
        raise AddError(
            f"{name}: this run has no dynamic handlers as parts (Program.dynamic(..., "
            "parts=)): no message loop reads an added one"
        )
    most = messages_header()[0]["kMessagesHandlersMax"]
    taken = {n for each in room.groups for n in each.numbers}
    taken |= {each.number for each in room.added.values()}
    refused = taken | room.messages
    if number is None:
        free = [n for n in range(1, most + 1) if n not in refused]
        if not free:
            raise AddError(f"{name}: no dynamic handler number 1 to {most} is left")
        return free[0]
    if not 1 <= number <= most:
        raise AddError(f"{name}: dynamic handler {number} is not in 1..{most}")
    if number in refused:
        raise AddError(
            f"{name}: number {number} is a dynamic handler or an overlay of the run "
            "already"
        )
    return number


def _builds(run, made, log):
    """Each group's parts built against its kernels (cached), checked against its
    slots and the instruction check: [{part: Linked}], seconds, all cached. Its own
    line counts the parts found and compiled (the next load's line does not)."""
    builds, cached, started, hits = [], True, time.perf_counter(), 0
    for each in run.backend.room.groups:
        images = {}
        for kernel, part in made.parts():
            linked, hit = _part(run.backend.card, each, made, kernel, part)
            cached, hits = cached and hit, hits + hit
            images[part] = linked.image
            log(f"added {made.name}: {part} part {len(linked.image)} B")
        builds.append(images)
    parts = overlay.PARTS_BUILT
    log(
        f"added {made.name}: built: {hits + parts['found']} found, "
        f"{parts['compiled']} compiled (parts)"
    )
    parts.update(found=0, compiled=0)
    return builds, time.perf_counter() - started, cached


def _part(card, each, made, kernel, part):
    """(Linked, whether it was in the cache) of one part of `made` built against the
    kernels of group `each` and checked against its slot and the instruction check
    (the card's chip's refusals too)."""
    slots = each.slots.get(part)
    if not slots or not slots[0]:
        raise AddError(
            f"{made.name}: the load gave the {kernel} no overlay slots; "
            f"declare them with Program.slots(bytes={{{kernel!r}: n}})"
        )
    cache = native.cache_directory() / "overlays"
    try:
        linked, elf, hit = overlay.build_part_cached(
            each.recipes[part], made, kernel, part, cache, slots[1]
        )
    except overlay.OverlayError as error:
        if "do not fit its slot" not in str(error):
            raise
        raise AddError(_slot_hint(error, kernel)) from None
    if not hit and insn_check.enabled():
        insn_check.check_kernels(
            made.name,
            [(made.sources[kernel], [elf])],
            native.metal_tree(card.tree),
            native.cache_directory() / "insn-check",
            card.device.arch(),
        )
    return linked, hit


def _slot_hint(error, kernel):
    """An added part larger than its slot: what the load kept and how to keep more."""
    return (
        f"{error}: the load sized the {kernel}'s slots by the loaded programs and the "
        "classes named ahead (Program.ahead(program)); name the class ahead, or give "
        f"the loaded program more (Program.slots(bytes={{{kernel!r}: n}}))"
    )


def build_ahead(card, room, programs):
    """At a load: the parts of the code each program said a run will add
    (Program.ahead), built against each group's kernels into the parts' cache (the
    number is not in a part's key), so the add finds them built. Nothing is written
    to the card; a load that reserved no room builds none."""
    for program in programs:
        for made in getattr(program, "added_ahead", ()) if room is not None else ():
            started, hits, count = time.perf_counter(), 0, 0
            for each in room.groups:
                for kernel, part in made.parts():
                    hits += _part(card, each, made, kernel, part)[1]
                    count += 1
            card.log(
                f"ahead {made.name}: {count} parts for {len(room.groups)} groups in "
                f"{time.perf_counter() - started:.2f} s ({hits} built before)"
            )


def _rows(room, made, builds):
    """Each different part image's first row in the room ({image: row}), the row past
    the last, the images in order; refused when they do not fit."""
    width = room.row_bytes
    rows_of, at, order = {}, room.next, []
    for images in builds:
        for image in images.values():
            if image not in rows_of:
                rows_of[image] = at
                at += -(-len(image) // width)
                order.append(image)
    if at > room.end:
        asked = (at - room.next) * width
        raise AddError(
            f"{made.name}: its parts take {asked:,} B of the room the load reserved "
            f"and {room.left():,} B are left (session.load(reserve=))"
        )
    return rows_of, at, order


def _place(run, made, builds):
    """Writes each different part image into the room (the same bytes once) and
    returns, per group, {part: (bank address, bytes)}; refused when it does not fit."""
    room = run.backend.room
    banks = run.backend.card.dram_banks  # a row is a period of the device's banks
    copies = overlay.copies_in(banks)  # in banks 0 to 7: core i reads bank i % 8
    width = room.row_bytes
    rows_of, at, order = _rows(room, made, builds)
    if order:
        pages = np.zeros((banks * (at - room.next), width), np.uint8)
        for image in order:
            first = rows_of[image] - room.next
            for row in range(-(-len(image) // width)):
                chunk = np.frombuffer(image[row * width : (row + 1) * width], np.uint8)
                page = banks * (first + row)
                pages[page : page + copies, : len(chunk)] = chunk
        run.backend.card.write(room.image, pages, first=banks * room.next)
    room.next = at
    stride = room.image.aligned_page_bytes
    return [
        {
            part: (room.image.address + rows_of[image] * stride, len(image))
            for part, image in images.items()
        }
        for images in builds
    ]


def entry_words(number, handler, placed, message=False):
    """The ENTRY command's words: the number (and the handler's or the dynamic
    handler's mark), each part's DRAM address, each part's bytes / 64 two a word."""
    h = abi.headers()
    words = [0] * 10
    words[h["KM_ENTRY_NUMBER"]] = (
        number
        | (h["KM_ENTRY_HANDLER"] if handler else 0)
        | (h["KM_ENTRY_MESSAGE"] if message else 0)
    )
    for index, part in enumerate(overlay.PARTS):
        address, size = placed.get(part, (0, 0))
        words[h["KM_ENTRY_ADDRESS"] + index] = address
        words[h["KM_ENTRY_BYTES"] + index // 2] |= (size // 64) << 16 * (index % 2)
    return words


def _mirror(run, each, number, words):
    """The entry in the host's copy of the overlay area of group `each`'s cores
    (control_rows): a reset list writes the area as the host holds it."""
    rows = getattr(run.backend, "control_rows", None)
    if rows is None:
        return
    c, h = overlay.constants(), abi.headers()
    start = run.backend.control_offset + h["KM_CONTROL_BYTES"] // 4
    flags = words[h["KM_ENTRY_NUMBER"]]
    area = rows[each.first : each.first + each.count, start:]
    for part in range(len(overlay.PARTS)):
        at = (c["kOverlayTable"] + 8 * (len(overlay.PARTS) * number + part)) // 4
        lines = words[h["KM_ENTRY_BYTES"] + part // 2] >> 16 * (part % 2) & 0xFFFF
        area[:, at] = words[h["KM_ENTRY_ADDRESS"] + part]
        area[:, at + 1] = 64 * lines
    count = area[:, c["kOverlayCount"]]
    area[:, c["kOverlayCount"]] = np.maximum(count, number + 1)
    if flags & h["KM_ENTRY_HANDLER"]:
        area[:, c["kOverlayHandlers"] // 4 + (number >> 5)] |= 1 << (number & 31)
    if flags & h["KM_ENTRY_MESSAGE"]:
        area[:, c["kOverlayMessages"] // 4] |= 1 << number


def addon(
    run,
    name,
    parts,
    handler=False,
    number=None,
    defines=None,
    timeout=10.0,
    message=False,
    compile_args=None,
):
    """Run.addon: builds overlay `name` from `parts` against the loaded kernels, writes
    it into the load's room and gives every core its table entry; returns its number
    (with `handler` or `message`, also the number of the handler, or of the dynamic
    handler, whose body it is)."""
    made = prepared(run, name, parts, handler, number, defines, message, compile_args)
    return committed(run, made, timeout)


def prepared(run, name, parts, handler, number, defines, message, compile_args):
    """addon's checks and build, nothing written: (made, handler, builds, build
    seconds, cached), refused (AddError) where the add would be."""
    room = getattr(run.backend, "room", None)
    if room is None:
        raise AddError(
            f"{name}: this run's load reserved no room for added code "
            "(session.load(reserve=0))"
        )
    if not room.entered:  # the entry command is served only on that path
        raise AddError(
            f"{name}: this run's kernels were built without the path for added code; "
            "the load's program asks for it with Program.slots(bytes=...) (bytes={} "
            "keeps the slot sizes) or, as a class, with km.ANY"
        )
    if name in room.added:
        number = room.added[name].number
        raise AddError(f"{name}: added to this run already, as overlay {number}")
    if handler and message:
        raise AddError(f"{name}: a handler or a dynamic handler, not both")
    run._check_alive()
    if unguarded(run.programs):  # the added code is built as the run's programs are
        given = (defines or {}).get("KM_GUARD")
        if given is not None and str(given) != "0":
            raise AddError(
                f"{name}: defines KM_GUARD={given}, and the run's programs are built "
                "with guard=False"
            )
        defines = {**(defines or {}), "KM_GUARD": 0}
    if message:
        made = overlay.declare(
            name, parts, _message_number(room, name, number), defines
        )
        try:
            made = dataclasses.replace(made, message=True)
        except ValueError as error:
            raise AddError(str(error)) from None
    else:
        made = overlay.declare(
            name, parts, _number(room, name, handler, number), defines
        )
    if compile_args:  # a part of a program in the runtime's form: its arguments
        made = dataclasses.replace(made, compile_args=dict(compile_args))
    builds, build_s, cached = _builds(run, made, run.backend.card.log)
    _rows(room, made, builds)
    return made, handler, builds, build_s, cached


def committed(run, ready, timeout):
    """addon's writes of a `prepared` add: its parts, each core's entry; its number."""
    made, handler, builds, build_s, cached = ready
    name, message = made.name, made.message
    room, log = run.backend.room, run.backend.card.log
    started = time.perf_counter()
    placed = _place(run, made, builds)
    write_s = time.perf_counter() - started
    started = time.perf_counter()
    entry = abi.headers()["KM_HANDLER_ENTRY"]
    words = [entry_words(made.number, handler, mine, message) for mine in placed]
    part = run._flag_part if run.island_ranks != run._workers else 0
    handles = [  # the groups in flat ranks; an island's run sends its part only
        run._queue_command(
            entry, (each.first + run.island_first, each.count), mine, False, part
        )
        for each, mine in zip(room.groups, words, strict=True)
    ]
    for handle in handles:
        event = run.wait(handle, timeout)
        if event.kind != "done":
            raise AddError(f"{name}: the entry command ended {event.kind}: {event}")
    for each, mine in zip(room.groups, words, strict=True):
        _mirror(run, each, made.number, mine)
    if getattr(run, "recovery", None) is not None and not run.keep_resets():
        log(
            f"added {name}: the reset lists were not kept again: {run.recovery_refused}"
        )
    entry_s = time.perf_counter() - started
    for each in room.groups:
        each.numbers.add(made.number)
    room.added[name] = Added(
        name,
        made.number,
        handler,
        message,
        {part: len(image) for part, image in builds[0].items()},
        build_s,
        write_s,
        entry_s,
        cached,
    )
    log(
        f"added {name} as {'handler and ' if handler else ''}"
        f"{'dynamic handler and ' if message else ''}overlay {made.number}: "
        f"build {build_s * 1e3:.1f} ms{' (cached)' if cached else ''}, write "
        f"{write_s * 1e3:.1f} ms, entries {entry_s * 1e3:.1f} ms"
    )
    return made.number


def refuse_half_added(table, handlers):
    """Refuses a command naming an add a later island's device refused (IslandRun's
    `table`: ("number", n) or ("program", p) -> why), on the islands' run or one
    island's own."""
    for handler in handlers if table else ():
        if type(handler) is int or isinstance(handler, np.integer):
            why = table.get(("number", int(handler)))
        else:
            program = getattr(handler, "program", None)
            owner = getattr(handler, "owner", None)
            why = next(
                (
                    text
                    for (kind, held), text in table.items()
                    if kind == "program"
                    and (
                        program is held
                        or (owner is not None and isinstance(held, owner))
                    )
                ),
                None,
            )
        if why is not None:
            raise AddError(f"{why}: a command naming it is refused")
