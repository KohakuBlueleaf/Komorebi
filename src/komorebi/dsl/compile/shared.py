"""How a class's handlers share buffers, three ways in turn, the first that fits: one
set for the program; by shape (layout.pair); at shared indices of other pages, set at a
handler's entry (assign). None fits: refused naming the handler with the most buffers;
a chain and the cases of `third_refused` take the first two only."""

import dataclasses
from dataclasses import astuple, dataclass

from komorebi.dsl.compile import layout
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.layout import BY_SHAPE, SHARED_INDICES
from komorebi.dsl.compile.records import Buffer, Refused, Unfit
from komorebi.dsl.front.names import spelling
from komorebi.errors import L1BudgetError

ONE_SET = 1
WAYS = {
    ONE_SET: "as one set for the program",
    BY_SHAPE: "shared by shape",
    SHARED_INDICES: "at shared indices of other pages",
}
INDENT = " " * 4
UNPACKED = ("read", "field", "arrival", "broadcast")  # as mixed_formats counts them
PACKED = ("write", "out")
# The indices a buffer of one call may take at shared indices, by its role: (first,
# last, a pair), the ranges the program's buffers take theirs from (layout).
RANGES = {
    "read": (layout.FIRST_READ, layout.LAST_INDEX, True),
    "write": (layout.FIRST_WRITE, layout.LAST_INDEX, True),
    "out": (layout.FIRST_WRITE, layout.LAST_INDEX, False),
    "word": (layout.FIRST_OTHER, layout.FIRST_WRITE - 1, False),
}
OVERLAP = "commands of handlers that give an index other pages do not overlap on a core"
CHAIN_FIX = (
    "a chain's cores go on from one call to the next without waiting; give these "
    "streams the same pages, or run the calls as commands"
)


def attempted(name, build, third=lambda: None, tried_compiled=lambda: None):
    """`build(step)` for each way in turn, the first that fits; `third()`: why the
    third is not taken (a chain: "chain"; another case: its refusal's text), None when
    it is; `tried_compiled()` the last attempt's compiled class (to name its
    handlers)."""
    tried = []
    for step in WAYS:
        if step == SHARED_INDICES and third() is not None:
            break
        try:
            return build(step)
        except (Unfit, L1BudgetError) as unfit:
            tried.append((step, unfit, getattr(unfit, "compiled", None)))
            if tried[-1][2] is None:
                tried[-1] = (step, unfit, tried_compiled())
    alike = len({str(unfit) for _, unfit, _ in tried}) == 1
    if alike and (third() is None or not unshared(tried[-1][2])):
        raise tried[0][1]  # no way shares what stops it: its refusal as it is
    raise refused(name, tried, third()) from None


def plain(name, unfit):
    """A refusal's text without the class's name before it."""
    text = str(unfit)
    return text[len(name) + 2 :] if text.startswith(f"{name}: ") else text


def handler_buffers(compiled):
    """Each handler's buffers of one call (streams' pairs, tables of entries', out
    buffers) by name, a pair counted as its two buffers."""
    if compiled is None:
        return {}
    named = {}
    for buffer in compiled.buffers:
        for handler in buffer.users if buffer.scoped else ():
            named.setdefault(handler, []).append(buffer)
    return named


def most(compiled):
    """'handler h has the most buffers, n (names)', or '' when none is known."""
    named = handler_buffers(compiled)
    if not named:
        return ""
    count = {
        h: sum(2 if b.index_b is not None else 1 for b in bs) for h, bs in named.items()
    }
    handler = max(named, key=lambda h: count[h])
    names = ", ".join(b.name for b in named[handler])
    return f"; {handler} has the most buffers, {count[handler]} ({names})"


def refused(name, tried, third):
    """The class refused after the ways `tried` (step, refusal, compiled)."""
    texts = {}
    for step, unfit, _ in tried:  # ways refused alike: their text once
        texts.setdefault(plain(name, unfit), []).append(WAYS[step])
    each = "; ".join(f"{' and '.join(ways)}: {text}" for text, ways in texts.items())
    last = tried[-1][2]
    if third == "chain":
        return Refused(
            f"{name}: a class with a chain shares buffers as one set or by shape only, "
            f"and its buffers fit neither: handlers {', '.join(handler_buffers(last))}"
            f"{unshared(last)}; {CHAIN_FIX} ({each})"
        )
    if third is not None:
        return Refused(f"{name}: its buffers fit neither as one set nor by shape, and "
                       f"{third} ({each})")  # fmt: skip
    return Refused(
        f"{name}: its buffers fit neither as one set, nor by shape, nor at shared "
        f"indices of other pages{most(last)} ({each})"
    )


def unshared(compiled):
    """'; streams that could not share: h1's kRead0 (256 B) and h2's kRead0b (512 B),
    ...': two handlers' streams of one role whose formats or page bytes differ."""
    named = handler_buffers(compiled)
    pairs, seen = [], set()
    handlers = list(named)
    for at, first in enumerate(handlers):
        for second in handlers[at + 1 :]:
            for a in named[first]:
                for b in named[second]:
                    if a is b or a.index_b is None or b.index_b is None:
                        continue
                    if a.role != b.role or (a.dtype, a.page_bytes) == (
                        b.dtype,
                        b.page_bytes,
                    ):
                        continue
                    key = (first, second, a.role, a.dtype, b.dtype)
                    key += (a.page_bytes, b.page_bytes)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append(
                        f"{first}'s {a.name} ({a.dtype}, {a.page_bytes} B) and "
                        f"{second}'s {b.name} ({b.dtype}, {b.page_bytes} B)"
                    )
    return f"; streams that could not share: {', '.join(pairs)}" if pairs else ""


def third_refused(form, found, overlays, dynamic=None):
    """Why a class does not take the third way (its text in the refusal), None when
    it may: the runtime's form, a set compiled at its first use, overlay parts, a
    dynamic handler (`dynamic` its name)."""
    takes = "do not set a handler's pages at its entry (shared indices of other pages)"
    if form == "runtime":
        return (
            f'{spelling("program_form")}="runtime": a program\'s parts in the '
            f"runtime's form {takes}; make it in the kernel form, or give its streams "
            "the same pages"
        )
    if found:
        return (
            f"km.{spelling('first_use')}: a set compiled at its first use takes the "
            f"load's buffers and does not set its pages at its entry; list the sets "
            f"({spelling('handlers')}=), or give its streams the same pages"
        )
    if overlays:
        return (
            f"{', '.join(overlays)} with {spelling('overlay')}=: a handler's overlay "
            f"parts {takes}; make it without {spelling('overlay')}=, or give its "
            "streams the same pages"
        )
    if dynamic:
        return (
            f"{dynamic}, a dynamic handler: the movers of a dynamic class are the "
            f"runtime's, which {takes}; give its streams the same pages"
        )
    return None


@dataclass(frozen=True)
class Row:
    """One index of a handler's pages: the index, its pages, a page's bytes, the
    buffer (and half) that takes it and its role."""

    index: int
    pages: int
    page_bytes: int
    name: str
    role: str = "read"


@dataclass
class Pages:
    """One set of pages of the shared indices: its number, the handlers it serves,
    its rows by index."""

    number: int
    handlers: list
    rows: dict

    @property
    def mask(self):
        return sum(1 << index for index in self.rows)


@dataclass
class Shares:
    """The shared indices: each set of pages, each handler's number (0: none), each
    handler's buffers at their indices and its rows, the indices declared, each
    index's one format of compute."""

    pages: list
    of: dict
    buffers: dict
    rows: dict
    declared: list
    formats: dict


def compute_format(compiled, buffer):
    """The format compute unpacks or packs `buffer` in, or None: raw pages, words and
    the runtime's are no tiles of compute."""
    if buffer.raw or buffer.role not in UNPACKED + PACKED:
        return None
    if buffer.name in compiled.unpack_fp32:
        return f"{buffer.dtype} (to Dst as float32)"
    return buffer.dtype


def order(buffer):
    """A buffer's place among its handler's: by role, format, page bytes, block (all
    descending), so that equal buffers of two handlers take one index."""
    return (buffer.role, buffer.dtype, buffer.page_bytes, buffer.block)


def place(compiled, handler, buffer, used, formats, mine, room=True):
    """The indices of `buffer` in `handler`'s pages: the first free in its role's
    range, of its format where compute unpacks or packs it; an own field moved up
    when none is (layout.make_room, when `room`); refused naming the handler."""
    first, last, pair = RANGES[buffer.role]
    want = compute_format(compiled, buffer)
    while True:
        taken = layout.taken_indices(compiled) | set(used)
        clash = None
        index = first
        while index + pair <= last:
            group = (index, index + 1) if pair else (index,)
            if not taken & set(group):
                other = [
                    formats[i] for i in group if want and formats.get(i, want) != want
                ]
                if not other:
                    return group
                clash = clash or (group[0], other[0])
            index += 2 if pair else 1
        if not room or not layout.make_room(compiled, first, last):
            break
    if clash is not None:
        raise Unfit(
            f"{compiled.name}.{handler}: its {buffer.name} ({want}) finds no index "
            f"from {first} to {last}: index {clash[0]} is {clash[1]} in another "
            "handler, and an index compute unpacks or packs through has one format "
            "among the handlers; give the buffers one format, or split the class"
        )
    wide = len([b for b in compiled.buffers if not b.scoped])
    raise Unfit(
        f"{compiled.name}.{handler}: its buffers do not fit the indices: its "
        f"{len(mine)} buffers ({', '.join(b.name for b in mine)}) and the program's "
        f"{wide} other buffers and {len(compiled.reserved)} reserved indices; open "
        "fewer streams at once in the handler, or split the class"
    )


def placed(compiled, scoped, room=True):
    """Each handler's rows and its buffers' places ({handler: (rows, places)}) and
    each index's format of compute."""
    formats, each = {}, {}
    for handler in compiled.handlers:
        mine = sorted((b for b in scoped if handler in b.users), key=lambda b: b.name)
        mine.sort(key=order, reverse=True)  # stable: equal shapes by name
        used, rows, places = {}, {}, {}
        for buffer in mine:
            group = place(compiled, handler, buffer, used, formats, mine, room)
            want = compute_format(compiled, buffer)
            for half, index in enumerate(group):
                used[index] = buffer.name
                if want is not None:
                    formats[index] = want
                name = buffer.name + (".mover_b" if half else "")
                rows[index] = Row(
                    index, buffer.pages[half], buffer.page_bytes, name, buffer.role
                )
            places[buffer.name] = group
        each[handler] = (rows, places)
    return each, formats


def compatible(rows, other):
    """Whether two sets of rows give every index they share the same pages."""
    return all(
        (row.pages, row.page_bytes) == (other[i].pages, other[i].page_bytes)
        for i, row in rows.items()
        if i in other
    )


def largest(each):
    """Each index's largest use: {index: (bytes, row, handler)}."""
    most = {}
    for handler, (rows, _) in each.items():
        for index, row in rows.items():
            size = row.pages * row.page_bytes
            if index not in most or size > most[index][0]:
                most[index] = (size, row, handler)
    return most


def used_bytes(compiled):
    """The L1 the buffers take before the fields are sized: at shared indices, the
    program's buffers and each index's largest use (placed as assign places them)."""
    shared = compiled.step == SHARED_INDICES
    total = sum(
        sum(b.pages) * b.page_bytes
        for b in compiled.buffers
        if not (shared and b.scoped)
    )
    if not shared:
        return total
    scoped = [b for b in compiled.buffers if b.scoped]
    try:
        each, _ = placed(compiled, scoped, room=False)
    except Unfit:  # assign refuses it; until then each buffer its own bytes
        return total + sum(sum(b.pages) * b.page_bytes for b in scoped)
    return total + sum(size for size, _, _ in largest(each).values())


def assign(compiled):
    """At shared indices: each handler's buffers of one call at indices of one common
    set (placed); handlers whose pages agree on every index they share are one set of
    pages; each index declared once, as large as its largest use."""
    if compiled.step != SHARED_INDICES or compiled.shares is not None:
        return
    scoped = [b for b in compiled.buffers if b.scoped]
    compiled.buffers = [b for b in compiled.buffers if not b.scoped]
    try:
        each, formats = placed(compiled, scoped)
    except Unfit:
        compiled.buffers += scoped  # the refusal names each handler's buffers
        raise
    sets, of, buffers = [], {}, {}
    for handler, (rows, places) in each.items():
        if not rows:
            of[handler] = 0
            continue
        same = [s for s in sets if compatible(rows, s.rows)]
        if same:
            same[0].handlers.append(handler)
            same[0].rows = {**rows, **same[0].rows}
            of[handler] = same[0].number
        else:
            sets.append(Pages(len(sets) + 1, [handler], dict(rows)))
            of[handler] = len(sets)
        buffers[handler] = [
            dataclasses.replace(
                b,
                index_a=places[b.name][0],
                index_b=places[b.name][1] if len(places[b.name]) > 1 else None,
            )
            for b in scoped
            if handler in b.users
        ]
    declared = []
    for index, (size, row, handler) in sorted(largest(each).items()):
        users = [h for h, (rows, _) in each.items() if index in rows]
        dtype = formats.get(index, "").split(" ")[0] or next(
            b.dtype for b in buffers[handler] if b.name == row.name.split(".")[0]
        )
        buffer = Buffer(
            f"kShared{index}",
            "shared",
            index,
            None,
            dtype,
            row.page_bytes,
            (row.pages,),
            1,
            f"index {index} of {', '.join(users)}: {size} B, {handler}'s pages "
            "declared, a handler's set at its entry",
        )
        if "float32 (" in formats.get(index, ""):
            compiled.unpack_fp32.add(buffer.name)
        declared.append(layout.added(compiled, buffer))
    rows = {handler: each[handler][0] for handler in buffers}
    compiled.shares = Shares(sets, of, buffers, rows, declared, formats)


def namespace(handler):
    return f"km_buffers_{handler}"


# The CPUs that set a row's interface: the movers every row; unpack all but the packed
# (it waits and pops); pack the packed (it reserves and pushes); math none.
SETTERS = (
    ("TRISC_UNPACK", lambda row: row.role not in PACKED),
    ("TRISC_PACK", lambda row: row.role in PACKED),
    ("TRISC_MATH", None),
    (None, lambda row: True),
)
SHORT = 0xFFFF  # a row's fields fit 16 bits up to it


def set_rows(held, keep):
    """Each CPU's rows and tables of the sets of pages (`keep`: its rows)."""
    lines = []
    for pages in held.pages:
        rows = [r for _, r in sorted(pages.rows.items()) if keep(r)]
        if rows:
            lines.append(f"constexpr km::ShareRow kShare{pages.number}Rows[] = {{")
            lines += [
                f"{INDENT}{{{r.index}, {r.pages}, {r.page_bytes}}},  // {r.name}"
                for r in rows
            ]
            lines.append("};")
    lines.append("constexpr km::ShareTable kShareTables[] = {")
    lines.append(f"{INDENT}{{nullptr, 0, 0}},")
    for pages in held.pages:
        count = sum(keep(r) for r in pages.rows.values())
        rows = f"kShare{pages.number}Rows" if count else "nullptr"
        lines.append(f"{INDENT}{{{rows}, {count}, {pages.mask:#x}}},")
    lines.append("};")
    return lines


def header_lines(compiled, buffer_lines):
    """The header's shared indices: each CPU's rows of each set of pages (those it
    sets), the table of them, each handler's set and its buffers' names in a namespace
    of its own (`buffer_lines`: a buffer's lines as the header writes the program's)."""
    held = compiled.shares
    if held is None or not held.pages:
        return []
    fields = [n for p in held.pages for r in p.rows.values() for n in astuple(r)[:3]]
    word = "uint16_t" if max(fields) <= SHORT else "uint32_t"
    lines = [
        "",
        "// Shared indices: each handler's buffers of one call at indices of one",
        "// common set; the reader gates a command of other pages (km::share_gate),",
        "// each other CPU sets them at the handler's entry (km::share_set). A row:",
        "// index, pages, a page's bytes, of the indices the CPU sets; a table: a",
        "// set's rows, their count, its indices' mask.",
        f"#define KM_SHARE_WORD {word}",
        '#include "komorebi/device/core/km_shared_indices.hpp"',
        "",
        f"constexpr uint32_t kShares = {len(held.pages)};",
    ]
    lines += [
        f"// pages {p.number}: {', '.join(p.handlers)}; {len(p.rows)} buffers"
        for p in held.pages
    ]
    for at, (define, keep) in enumerate(SETTERS):
        if define is not None:
            lines.append(f"#{'if' if at == 0 else 'elif'} defined({define})")
        else:
            lines.append("#else  // the movers")
        if define == "TRISC_MATH":  # it sets no interface and holds no buffer
            lines.append("constexpr const km::ShareTable* kShareTables = nullptr;")
        else:
            lines += set_rows(held, keep)
    lines.append("#endif")
    lines += [
        f"constexpr uint32_t kShareOf{camel(handler)} = {number};"
        for handler, number in held.of.items()
    ]
    lines += [
        "// a command's set of pages (0: none, the CPU's stays), for the reader's gate",
        "constexpr uint32_t km_share_of(uint32_t handler) {",
        f"{INDENT}switch (handler) {{",
        *(
            f"{INDENT * 2}case kHandler{camel(handler)}: return {number};"
            for handler, number in held.of.items()
            if number
        ),
        f"{INDENT * 2}default: return 0;",
        f"{INDENT}}}",
        "}",
    ]
    for handler, buffers in held.buffers.items():
        lines += [
            "",
            f"// {handler.upper()}'s buffers (pages {held.of[handler]})",
            f"namespace {namespace(handler)} {{",
        ]
        for buffer in buffers:
            lines += buffer_lines(buffer)
        lines.append("}")
    return lines


def entry_lines(compiled, handler, kernel, pad):
    """A handler's first lines in a kernel ("movement" or "compute") at shared
    indices: its buffers' names, and its pages set when the CPU is set for others
    (the reader set its own at its gate)."""
    held = compiled.shares
    if held is None or not held.of.get(handler):
        return []
    number = f"kShareOf{camel(handler)}"
    return [
        f"{pad}using namespace {namespace(handler)};",
        f"{pad}if ({number} != km::share_now) {{",
        f"{pad}{INDENT}km::share_set({kernel}, kShareTables, {number});",
        f"{pad}}}",
    ]


def gate_lines(compiled, at):
    """The movers' source at shared indices: the define before the includes ("define":
    km_worker.hpp's reader calls the gate) and the gate after the header ("gate": a
    command of other pages waits in km::share_gate), else none."""
    if not shared_indices(compiled):
        return []
    if at == "define":
        return ["#define KM_SHARE_GATED"]
    return [
        "",
        "inline bool km::share_gated(",
        f"{INDENT}km::Reader& reader, const volatile km_command_t* job",
        ") {",
        f"{INDENT}const km_command_t& command = *const_cast<const km_command_t*>(job);",
        f"{INDENT}const uint32_t pages = km_share_of(km::handler(command));",
        f"{INDENT}return pages == 0 || pages == km::share_now ||",
        f"{INDENT * 2}km::share_gate(reader, command, kShareTables, pages);",
        "}",
    ]


def format_buffers(compiled):
    """The buffers whose formats compute unpacks and packs: the program's, and at
    shared indices each handler's."""
    held = compiled.shares
    if held is None:
        return compiled.buffers
    return [
        *compiled.buffers,
        *(b for buffers in held.buffers.values() for b in buffers),
    ]


def startup(compiled, first_in, first_out):
    """compute's start-up buffers at shared indices: a shared index compute unpacks
    when the program has no read pair of its own, one it packs when it has no write;
    else as given."""
    held = compiled.shares
    if held is None or not held.pages:
        return first_in, first_out
    unpacked = sorted(
        b.index_a
        for buffers in held.buffers.values()
        for b in buffers
        if b.role in UNPACKED and compute_format(compiled, b)
    )
    packed = sorted(
        b.index_a
        for buffers in held.buffers.values()
        for b in buffers
        if b.role in PACKED and compute_format(compiled, b)
    )
    wide = [b.name for b in compiled.buffers if b.role == "shared"]
    if unpacked and first_in in ("0", *wide):
        first_in = str(unpacked[0])
    if packed and (first_out in ("16", *wide) or first_out == first_in):
        first_out = str(packed[0])
    return first_in, first_out


def shared_by(compiled):
    """Each buffer two or more handlers share: (buffer, its handlers)."""
    return [(b, b.users) for b in compiled.buffers if b.scoped and len(b.users) > 1]


def report_lines(compiled):
    """The build report's way of sharing: which, and for the second and third which
    handlers share which indices and each index's bytes."""
    lines = [f"  buffers: {WAYS[compiled.step]} (way {compiled.step} of 3)"]
    if compiled.step == BY_SHAPE:
        for buffer, users in shared_by(compiled):
            indices = (
                f"{buffer.index_a}"
                if buffer.index_b is None
                else f"{buffer.index_a}, {buffer.index_b}"
            )
            each = " + ".join(f"{p * buffer.page_bytes} B" for p in buffer.pages)
            lines.append(
                f"    {buffer.name} (buffers {indices}, {each}): shared by "
                f"{', '.join(users)}"
            )
    held = compiled.shares
    if compiled.step == SHARED_INDICES and held is not None:
        lines[0] += f"; {OVERLAP}"
        for buffer in held.declared:
            uses = [
                f"{handler} {rows[buffer.index_a].pages} x "
                f"{rows[buffer.index_a].page_bytes} B"
                for handler, rows in held.rows.items()
                if buffer.index_a in rows
            ]
            size = buffer.pages[0] * buffer.page_bytes
            lines.append(f"    buffer {buffer.index_a}: {size} B; {', '.join(uses)}")
        lines += [
            f"    pages {pages.number} ({', '.join(pages.handlers)}): "
            f"{len(pages.rows)} buffers"
            for pages in held.pages
        ]
    return lines


def shared_indices(compiled):
    """Whether the program's handlers set their pages at their entry (the third way)."""
    return compiled.shares is not None and bool(compiled.shares.pages)
