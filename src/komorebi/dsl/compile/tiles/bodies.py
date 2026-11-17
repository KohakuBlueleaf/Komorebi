"""A handler's inline bodies (km.cxx): the C++ as written on its CPU, the names it
takes and gives by the table of the classes reference, at its kernel's start or end,
and the words of pack alone that must not steer it."""

import ast
import re
import textwrap

from komorebi.dsl.compile import (
    expressions,
    handler_buffers,
    layout,
    scope,
    tile_lookup,
)
from komorebi.dsl.compile.casts import broadcasts
from komorebi.dsl.compile.cxx import camel, pattern_uses
from komorebi.dsl.compile.dst import dst_tiles
from komorebi.dsl.compile.pages import field_fills
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import WAIT_DATA, WAIT_INPUT, check_name, refuse, wait
from komorebi.dsl.compile.tiles import writes
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Line

# An inline body's CPU (on=): a mover, one compute CPU, or compute's three; the #if of
# each one CPU's code.
COMPUTE_CPUS = ("unpack", "math", "pack")
# What compute's three CPUs run together in a body: a word of pack's alone must not
# steer it (pack_words_steer).
STEERED = re.compile(
    r"\btile_regs_\w+\s*\(|\bcopy_tile\s*\(|\bmatmul\w*\s*\(|\b(?!pack_)\w+_tile\s*\(|"
    r"\b(?:return|break|continue|goto)\b"
)
BODY_CPUS = ("mover_a", "mover_b", *COMPUTE_CPUS)
CPU_GUARDS = {
    "mover_a": "defined(COMPILE_FOR_BRISC)",
    "mover_b": "!defined(COMPILE_FOR_BRISC)",
    **{cpu: f"defined(TRISC_{cpu.upper()})" for cpu in COMPUTE_CPUS},
}
HANDOVER_WORDS = 16  # a body's words of the other mover: one 64 B page


def body_text(frame, node, call):
    """A body's text: a string, a string constant of the class's module, or a sum
    of them (one text for the bodies of several handlers)."""

    def text_of(part):
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            return part.value
        if isinstance(part, ast.Name) and isinstance(frame.names.get(part.id), str):
            return frame.names[part.id]
        if isinstance(part, ast.BinOp) and isinstance(part.op, ast.Add):
            return text_of(part.left) + text_of(part.right)
        return None

    text = text_of(call.args[0]) if call.args else None
    if text is None:
        refuse(
            frame.where,
            node,
            "an inline body is a string, a string constant of the module or a "
            "sum of them",
            'write km.cxx("..."), or km.cxx(BODY) with BODY = "..." at module ' "level",
        )
    return text


def cpu_address(frame, node, name, address, cpu, bound, before, after, release):
    """An L1 address unpack knows, for a body on one compute CPU: unpack's own;
    math or pack are handed it before the body and, with `release`, hand back a
    release after it (unpack frees nothing before)."""
    if cpu == "unpack":
        bound.append(f"const uint32_t {name} = {address};")
        return
    thread = f"ckernel::ThreadId::{cpu.capitalize()}ThreadId"
    before.append(
        Line(
            "#if defined(TRISC_UNPACK)\n"
            f"compute.hand({thread}, {address});\n"
            "#endif"
        )
    )
    bound.append(
        f"const uint32_t {name} = " "compute.handed(ckernel::ThreadId::UnpackThreadId);"
    )
    handler_buffers.handover(frame, node, f"{name}'s L1 address", "unpack", (cpu,))
    if not release:
        return
    after.append(
        Line(
            f"// {cpu} is done with {name}: unpack may free it\n"
            f"#if {CPU_GUARDS[cpu]}\n"
            "compute.hand(ckernel::ThreadId::UnpackThreadId, 1);\n"
            "#elif defined(TRISC_UNPACK)\n"
            f"(void)compute.handed({thread});\n"
            "#endif"
        )
    )
    handler_buffers.handover(frame, node, f"the release of {name}", cpu, ("unpack",))


def inline(frame, node, call, name=None):
    """`km.cxx(text, takes=[...], gives=..., on=..., include=[...])`: the C++ as
    written, on its CPU (a mover, one compute CPU, or compute's three), the names
    it takes bound before it; `name = km.cxx(..., gives=int)` the word the body
    sets, `gives=km.Tile` the tile it leaves in DST 0 (compute)."""
    text = textwrap.dedent(body_text(frame, node, call)).strip("\n")
    included = 0  # its first lines that are #include lines: the kernel's top
    lines = text.split("\n")
    while included < len(lines) and lines[included].lstrip().startswith("#include"):
        included += 1
    if any(line.lstrip().startswith("#include") for line in lines[included:]):
        refuse(
            frame.where,
            node,
            "a body's #include lines come first (they go to its kernel's top)",
            "put them on the body's first lines",
        )
    text = "\n".join(lines[included:])
    on, takes, gives, includes, at = "compute", [], None, [], None
    stated = {}  # reads= and writes=: the tensors it reads or writes, stated
    for keyword_ in call.keywords:
        value = keyword_.value
        if keyword_.arg in (spelling("body_reads"), spelling("body_writes")):
            stated[keyword_.arg] = value
        elif keyword_.arg == spelling("at"):
            at = kernel_at(frame, node, value)
        elif keyword_.arg == spelling("on"):
            if not (isinstance(value, ast.Constant) and value.value in BODY_CPUS):
                refuse(
                    frame.where,
                    node,
                    "on= names a mover or a compute CPU (a body without it is "
                    "compute's three)",
                    'write on="mover_a", "mover_b", "unpack", "math" or "pack"',
                )
            on = value.value
        elif keyword_.arg == spelling("takes"):
            if not isinstance(value, ast.List):
                refuse(
                    frame.where, node, "takes= is a list of names", "write takes=[a]"
                )
            takes = list(value.elts)
        elif keyword_.arg == spelling("include"):
            if not (
                isinstance(value, ast.List)
                and all(
                    isinstance(item, ast.Constant) and isinstance(item.value, str)
                    for item in value.elts
                )
            ):
                refuse(
                    frame.where,
                    node,
                    "include= is a list of header paths",
                    'write include=["../kernels/walk.hpp"]',
                )
            includes = [item.value for item in value.elts]
        elif keyword_.arg == spelling("gives"):
            if isinstance(value, ast.Name) and value.id == "int":
                gives = "int"
            elif isinstance(value, ast.Attribute) and value.attr == spelling(
                "tile_value"
            ):
                gives = "tile"
            else:
                refuse(
                    frame.where,
                    node,
                    "a body gives a word or a tile",
                    f"write gives=int or gives=km.{spelling('tile_value')}",
                )
        else:
            refuse(
                frame.where,
                node,
                f"km.cxx has no {keyword_.arg}=",
                f"use takes=, gives=, on=, include=, at=, "
                f"{spelling('body_reads')}= and {spelling('body_writes')}=",
            )
    stated_tensors(frame, node, stated, takes)
    if at is not None:
        kernel_takes(frame, node, takes, gives, name)
    if (gives is None) != (name is None):
        refuse(
            frame.where,
            node,
            "a body that gives a value is assigned, one that gives none is not",
            "write `x = km.cxx(..., gives=int)`, or km.cxx(...) on its own line",
        )
    mover = {"mover_a": "A", "mover_b": "B"}.get(on)
    cpu = on if on in COMPUTE_CPUS else None  # one compute CPU, or None
    if gives == "tile" and on != "compute":
        refuse(
            frame.where,
            node,
            "a tile is compute's three CPUs'",
            "drop on= for a body giving a tile",
        )
    where = mover or cpu or "compute"
    uses, bound, before, after = set(), [], [], []
    for item in takes:
        if take_constant(frame, item, bound):
            continue
        field = scope.field_name(frame, item)
        if field in frame.compiled.own:
            take_own(frame, node, field, where, text, bound)
            continue
        if field:
            take_field(frame, node, field, where, bound, before, after)
            continue
        uses |= take(frame, node, item, where, bound, before, after)
    if where == "compute":
        pack_words_steer(frame, node, text, takes)
    kernel = "compute" if mover is None else "movement"
    guard = CPU_GUARDS.get(on)
    entries = [line.strip() for line in lines[:included]]
    for path in includes:  # the body's CPU's alone
        entry = f'#include "{path}"'
        entries.append(entry if guard is None else f"#if {guard}\n{entry}\n#endif")
    for entry in entries:
        if entry not in frame.compiled.body_includes[kernel]:
            frame.compiled.body_includes[kernel].append(entry)
    start = call.args[0].lineno + 1 + included
    if not isinstance(call.args[0], ast.Constant):  # the module's: the call's line
        start = node.lineno
    body = f'#line {start} "{frame.where.file}"\n{text}\n#line RESUME'
    if bound:
        inner = "\n".join("    " + line for line in [*bound, *body.split("\n")])
        body = "{\n" + inner + "\n}"
    if gives == "int":
        check_name(frame.where, node, name)
        body = f"uint32_t {name} = 0;\n{body}"
        frame.scope[name] = Value("word", name, mover=where)
    elif gives == "tile":
        check_name(frame.where, node, name)
        tile_lookup.other_tile_line(frame, node)
        if frame.dst_live is not None or frame.dst_next != 0:
            refuse(
                frame.where,
                node,
                "a body's tile is DST 0, which another tile holds",
                "write or reduce that tile first",
            )
        frame.dst_next = 1
        frame.dst_live = name
        frame.scope[name] = Value("tile", name, dst=0)
        body = f"tile_regs_acquire();\n{body}"
    if cpu is not None:  # its CPU's alone
        body = f"#if {guard}\n{body}\n#endif"
    if at is not None:  # at the kernel's start or end, not in the handler
        if mover is not None:
            body = f"#if {guard}\n{body}\n#endif"
        frame.compiled.kernel_bodies.setdefault((at, kernel), []).append(body)
        frame.compiled.kernel_body_lines.append((at, node.lineno))
        return [], []
    lines = [*before, Line(body, uses, mover=mover), *after]
    moves, frame.body_moves = frame.body_moves, []
    if mover is None:
        return moves, lines
    return lines, []


def kernel_at(frame, node, value):
    """`at="start"` or `at="end"`: a body at its kernel's start (before the
    commands, after the start-up) or end (after the run)."""
    if not (isinstance(value, ast.Constant) and value.value in ("start", "end")):
        refuse(
            frame.where,
            node,
            "at= is the kernel's start or its end",
            'write at="start" or at="end"',
        )
    return value.value


def kernel_takes(frame, node, takes, gives, name):
    """A body at a kernel's start or end takes the class's own fields alone (no
    handler's value is made there) and gives nothing."""
    if gives is not None or name is not None:
        refuse(
            frame.where,
            node,
            "a body at a kernel's start or end gives nothing to the handler",
            "drop gives= and the name",
        )
    for item in takes:
        if scope.field_name(frame, item) not in frame.compiled.own:
            refuse(
                frame.where,
                node,
                "a body at a kernel's start or end takes the class's own fields "
                "alone: no handler's value is made there",
                f"take fields declared km.{spelling('own')}(), or drop at=",
            )


def pack_words_steer(frame, node, text, takes):
    """A written page's word in a body on compute is pack's (0 once pack's reserve
    gave up, the address on unpack and math): it may guard the packs only, and is
    refused where it steers a section, a copy, a product or a way out (pack would
    leave unpack and math held inside the unit)."""
    words = [
        item.id
        for item in takes
        if isinstance(item, ast.Name)
        and getattr(frame.scope.get(item.id), "kind", None) in ("write", "drained")
    ]
    for word in words:
        mentions = re.compile(rf"\b{re.escape(word)}\b")
        heads = []
        for line in text.splitlines():
            code = line.split("//")[0]
            steered = any(mentions.search(head) for head in heads)
            shared = STEERED.search(code)
            if shared and (steered or mentions.search(code)):
                refuse(
                    frame.where,
                    node,
                    f"{word} is pack's word (0 once pack's reserve gave up, the "
                    f"page on unpack and math): `{code.strip()}` would run on some "
                    "of compute's CPUs and not on others",
                    f"guard only pack_tile (and the packer's lines) with {word}; "
                    "keep tile_regs_*, copies, products and the way out on all "
                    "three",
                )
            for char in code:
                if char == "{":
                    heads.append(code)
                elif char == "}" and heads:
                    heads.pop()


def stated_tensors(frame, node, stated, takes):
    """`reads=[t]` and `writes=[t]` of a body: tensor parameters it takes, whose
    pages its own C++ reads or writes (a chain's description of the call; the
    build cannot check the C++ against them)."""
    taken = {item.id for item in takes if isinstance(item, ast.Name)}
    for keyword_, value in stated.items():
        if not isinstance(value, ast.List) or not all(
            isinstance(item, ast.Name) for item in value.elts
        ):
            refuse(
                frame.where,
                node,
                f"{keyword_}= is a list of the tensors the body {keyword_}",
                f"write {keyword_}=[t]",
            )
        for item in value.elts:
            if getattr(frame.scope.get(item.id), "kind", None) != "tensor":
                refuse(
                    frame.where,
                    node,
                    f"{item.id} in {keyword_}= is not a tensor parameter: "
                    f"{keyword_}= names the tensors whose pages the body "
                    f"{keyword_} in DRAM",
                    f"name tensor parameters in {keyword_}=",
                )
            if item.id not in taken:
                refuse(
                    frame.where,
                    node,
                    f"{item.id} is in {keyword_}= and not in "
                    f"{spelling('takes')}=: a body states what it {keyword_} of "
                    "what it takes",
                    f"take it: {spelling('takes')}=[..., {item.id}]",
                )


def take(frame, node, item, where, bound, before, after):
    """A name a body takes on `where` (mover "A" or "B", "compute", or a compute
    CPU), as the table of docs/reference/classes.md ("Inline bodies") says: by
    name, a stream tile's buffer or L1 address, a write stream's page on pack, a
    tensor on a mover, a written tile on mover B."""
    if isinstance(item, ast.Name):
        scope.not_ended(frame, node, item.id)
    if not isinstance(item, ast.Name) or item.id not in frame.scope:
        refuse(
            frame.where,
            node,
            "takes= names the handler's values and the class's constants",
            "take names, or self.K of a constant",
        )
    value = frame.scope[item.id]
    if value.kind in ("int", "partition", "pattern", "float"):
        return {item.id}  # a float word: a C++ float of that name
    if value.kind == "page" and where == value.mover:  # its L1 address
        return {item.id}
    if value.kind == "drained":
        return take_drained(frame, node, item.id, where, bound, before, after)
    if value.kind == "page":
        refuse(
            frame.where,
            node,
            f"{item.id} is a page mover {value.mover} reads",
            f'take it in a body on="mover_{value.mover.lower()}"',
        )
    if value.kind in ("entry", "entries"):
        return take_entry(frame, node, item.id, value, bound)
    if value.kind == "arrived":
        return take_arrived(frame, node, item.id, value, where, bound, before, after)
    if value.kind == "held":
        return take_held(frame, node, item.id, value, where, bound)
    if value.kind == "streamed" and where == "compute":  # its buffer and place
        buffer, index = spelling("tile_buffer"), spelling("tile_index")
        bound.append(f"const uint32_t {item.id}{buffer} = {value.cxx};")
        bound.append(f"const uint32_t {item.id}{index} = {value.first};")
        return set()
    if value.kind == "streamed" and where in COMPUTE_CPUS:
        address = f"get_tile_l1_byte_address({value.cxx}, {value.first})"
        cpu_address(frame, node, item.id, address, where, bound, before, after, True)
        return set()
    if value.kind == "write" and where in ("pack", "compute"):
        return writes.take_write(frame, node, item.id, value, bound, before, after)
    if value.kind == "write" and where == "B":
        take_page(frame, node, item.id, value, bound, before, after)
        return set()
    if value.kind == "write":
        refuse(
            frame.where,
            node,
            f"{item.id} is a write stream: its pages are pack's to fill",
            'take it in a body on="pack" (and after it, on="mover_b")',
        )
    if value.kind == "moved" and where in ("A", "B"):
        take_moved(frame, node, item.id, value, bound)
        return set()
    if value.kind == "moved":
        refuse(
            frame.where,
            node,
            f"{item.id} after its loop is what a mover moved",
            'take it in a body on="mover_a" or on="mover_b"',
        )
    if value.kind == "tensor" and where in ("A", "B"):
        word = f"km_word_{item.id}"
        kind = "km::Region" if item.id in frame.compiled.regions else "uint32_t"
        bound.append(f"const {kind} {word} = {item.id};")
        bound.append(f"const {camel(item.id)} {item.id}({word});")
        return {item.id}
    if value.kind == "tensor":
        refuse(
            frame.where,
            node,
            f"{item.id} is a tensor, in DRAM: the movers' to move",
            'take it in a body on="mover_a" or on="mover_b"',
        )
    if value.kind == "word" and (
        value.mover == where or value.mover == "compute" and where in COMPUTE_CPUS
    ):
        return {item.id}
    if value.kind == "word" and {value.mover, where} == {"A", "B"}:
        return take_handed(frame, node, item.id, value, where, bound, before, after)
    if value.kind in ("written", "reduced") and value.buffer is not None:
        if where != "B":
            refuse(
                frame.where,
                node,
                f"{item.id} is packed for mover B's write",
                'take it in a body on="mover_b"',
            )
        if value.popped:
            refuse(
                frame.where,
                node,
                f"{item.id} was written and its page freed above",
                "take it before the line that writes it",
            )
        if not value.waited:
            before.append(Line(handler_buffers.page_wait(frame, value)
                               + "\n    return false;\n}", mover="B"))  # fmt: skip
            value.waited = True
        bound.append(f"const uint32_t {item.id} = {value.buffer.name};")
        return set()
    if value.kind == "reduced" and value.summed:
        return take_sum(frame, node, item.id, value, where, bound, before, after)
    refuse(
        frame.where,
        node,
        f"{item.id} is a {value.kind} here: a body takes ints, partitions, "
        "patterns, words of its CPU, stream tiles, write streams (on pack), "
        "tensors (on a mover) and written tiles",
        "pass those",
    )


def take_arrived(frame, node, name, value, where, bound, before, after):
    """A block a broadcast brought, taken whole by a body on compute's three: the
    block waited for before it, `name_buffer` its buffer and `name` its first page
    there (0), freed after it."""
    if where != "compute":
        refuse(
            frame.where,
            node,
            f"{name} is a block of compute's three CPUs: each waits for it and "
            "frees it",
            "take it in a body on compute's three (no on=)",
        )
    held_in_loop = getattr(value, "held", False) and not (
        broadcasts.taken_by_field_products(frame, name)
    )
    if getattr(value, "held", False) and not held_in_loop:
        refuse(
            frame.where,
            node,
            f"{name} is taken by products of a field (waited at its cast, freed at "
            "the end of its lines), and a body takes a block whole",
            "take it in products only, or in a body only",
        )
    buffer = value.buffer.name
    if held_in_loop:  # read in place: the cast's lines wait for it and free it
        bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {buffer};")
        bound.append(f"const uint32_t {name} = 0;")
        return set()
    broadcasts.arrived_once(frame, node, name, value)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    before.append(
        Line(
            f"km_ok = km_ok && km::broadcast_receive(compute, {buffer}, "
            f"{buffer}Pages);",
            {"km_ok"},
        )
    )
    bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {buffer};")
    bound.append(f"const uint32_t {name} = 0;")
    after.append(
        Line(f"if (km_ok) {{\n    km::broadcast_done({buffer}, {buffer}Pages);\n}}")
    )
    return set()


def take_constant(frame, item, bound):
    """`self.K` of a constant of the class (not a field), taken: the body's
    `constexpr uint32_t K`, this program's value, on any CPU. False for any other
    name."""
    if not (
        isinstance(item, ast.Attribute)
        and isinstance(item.value, ast.Name)
        and item.value.id == "self"
        and item.attr not in frame.compiled.fields
        and item.attr in frame.compiled.class_constants
    ):
        return False
    bound.append(
        f"constexpr uint32_t {item.attr} = {expressions.class_constant(frame, item)};"
    )
    return True


def take_drained(frame, node, name, where, bound, before, after):
    """A drained stream a compute body fills: its page reserved before the body,
    pushed after it and counted for the end."""
    state = frame.drained[name]
    if where in ("A", "B"):
        refuse(
            frame.where,
            node,
            f"{name}'s pages are compute's to fill; mover {frame.scope[name].mover} "
            f"drains them",
            f"take it on compute, and drain it with `for page in {name}:`",
        )
    if state["ended"] is not None:
        refuse(
            frame.where,
            node,
            f"{name} ended at line {state['ended']}: no page follows",
            "take it before its end",
        )
    state["taken"] = True
    buffer = f"kField{camel(state['field'])}"
    room = f"km_room_{name}{node.lineno}"
    before.append(Line(f"const bool {room} = compute.reserve_back({buffer}, 1);"))
    bound.append(
        f"const uint32_t {name} = {room} ? "
        f"get_local_cb_interface({buffer}).fifo_wr_ptr << 4 : 0;"
    )
    bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {buffer};")
    after.append(
        Line(
            f"if ({room}) {{\n    cb_push_back({buffer}, 1);\n"
            f"    ++km_pushed_{name};\n}}"
        )
    )
    return set()


def take_entry(frame, node, name, value, bound):
    """An entry a body takes: each of its words as `name_word` (a table: its
    count as `name_count`)."""
    if value.kind == "entries":
        pairs = [(f"{name}_{spelling('entries_count')}", value.table["count"])]
    else:
        pairs = [
            (f"{name}_{word}", f"km_{name}_{word}")
            for word in value.table["class"].words
        ]
    for given, text in pairs:
        if given in frame.scope:
            refuse(
                frame.where,
                node,
                f"the body's {given} is a name of the handler too",
                "rename one",
            )
        bound.append(f"const uint32_t {given} = {text};")
    return {text for _, text in pairs} | {value.cxx}


def take_field(frame, node, field, where, bound, before, after):
    """A field an inline body takes: compute's three its buffer index, one compute
    CPU its L1 address (unpack's, handed on), mover B its address once mover A
    filled it. Freed after the body when filled here; a kept field (another
    handler reads it) stays, also in its filler (its next fill frees it)."""
    stays = field_fills.kept_field(frame, field)
    kept = stays and field not in frame.filled
    if not kept and (field not in frame.filled or frame.filled[field] is None):
        refuse(
            frame.where,
            node,
            f"self.{field} is not filled above in this handler (or a body took "
            "it), and no other handler keeps it filled",
            f"write self.{field}.fill(km.Stream(...)) before this body, or in "
            "a handler of its own",
        )
    if not kept and frame.filled[field][1] != frame.blocks[-1]:
        refuse(
            frame.where,
            node,
            f"self.{field} is filled in another block: every core that fills it "
            "must free it",
            "take it in the block of its fill (or fill it inside this one)",
        )
    if where == "A":
        refuse(
            frame.where,
            node,
            f"self.{field} is filled by mover A and handed on",
            'take it in a body on compute or on="mover_b"',
        )
    if not kept:
        frame.filled[field] = None
    buffer, pages_name = f"kField{camel(field)}", f"kField{camel(field)}Pages"
    if where == "B":
        before.append(
            Line(
                f"if (!movement.wait_front({buffer}, {pages_name})) {{\n"
                "    return false;\n}",
                mover="B",
            )
        )
        bound.append(f"const uint32_t {field} = get_read_ptr({buffer});")
        if not stays:
            after.append(Line(f"cb_pop_front({buffer}, {pages_name});", mover="B"))
        wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
        return
    before.append(
        Line(f"if (!compute.wait_front({buffer}, {pages_name})) {{\n" "    return;\n}")
    )
    if where == "compute":
        bound.append(f"const uint32_t {field} = {buffer};")
    else:
        address = f"get_tile_l1_byte_address({buffer}, 0)"
        cpu_address(frame, node, field, address, where, bound, before, after, not stays)
    if not stays:
        after.append(Line(f"cb_pop_front({buffer}, {pages_name});"))
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)


def take_handed(frame, node, name, value, where, bound, before, after):
    """A word one mover's body gave, taken by a body on the other mover: the
    giver writes the body's words into a page of the hand-over buffer and pushes
    it there; the taker waits for the page, reads them, and frees it after."""
    parts = frame.handed.setdefault(id(node), {"words": []})
    words = parts["words"]
    if len(words) == HANDOVER_WORDS:
        refuse(
            frame.where,
            node,
            f"a hand-over between the movers holds {HANDOVER_WORDS} words a body, "
            f"and this body takes {name} after {len(words)}",
            f"take at most {HANDOVER_WORDS} words of the other mover in one body",
        )
    if words and parts["giver"] != value.mover:
        refuse(frame.where, node, "a body's hand-over is from one mover", "take one's")
    words.append(name)
    buffer = layout.other_buffer(
        frame.compiled,
        "kHandover",
        "word",
        "uint32",
        4 * HANDOVER_WORDS,
        1,
        f"line {node.lineno}: a hand-over of mover {value.mover}'s words",
    ).name
    page = f"km_handed{node.lineno}"
    stores = "".join(f"\nkm::l1({page})[{n}] = {word};" for n, word in enumerate(words))
    giving = (
        f"if (!movement.reserve({buffer}, 1)) {{\n    return false;\n}}\n"
        f"const uint32_t {page} = get_write_ptr({buffer});{stores}\n"
        f"cb_push_back({buffer}, 1);"
    )
    if len(words) == 1:
        parts["giver"] = value.mover
        parts["line"] = Line(giving, set(words), mover=value.mover)
        before.append(parts["line"])
        before.append(
            Line(
                f"if (!movement.wait_front({buffer}, 1)) {{\n    return false;\n}}"
                f"\ninvalidate_l1_cache();\nconst uint32_t {page} = "
                f"get_read_ptr({buffer});",
                mover=where,
            )
        )
        after.append(Line(f"cb_pop_front({buffer}, 1);", mover=where))
    else:
        parts["line"].text = giving
        parts["line"].uses |= {name}
    bound.append(f"const uint32_t {name} = km::l1({page})[{len(words) - 1}];")
    frame.compiled.handovers.append(
        f"{frame.method.name}, line {node.lineno}: {name} from mover {value.mover} "
        f"to mover {where} by the hand-over buffer"
    )
    return set()


def take_held(frame, node, name, value, where, bound):
    """A held stream's block for a body on compute's three: `name_buffer` its
    buffer, `name` its first page there (0)."""
    if where != "compute":
        refuse(
            frame.where,
            node,
            f"{name} is a block of compute's three CPUs: each waits for it and "
            "frees it",
            "take it in a body on compute's three (no on=)",
        )
    bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {value.cxx};")
    bound.append(f"const uint32_t {name} = 0;")
    return set()


def take_moved(frame, node, name, value, bound):
    """A stream loop's name after the loop, for a body on a mover: what that mover
    moved in the loop's move (km::Moved), `name_tiles` and `name_cycles`."""
    moved, move = value.cxx, value.move
    if value.ended is not None:  # its km::Moved is declared inside the if or loop
        kind, line = value.ended
        refuse(
            frame.where,
            node,
            f"{name} after its loop is what a mover moved, and the loop is inside "
            f"the {kind} at line {line}: what it moved ends with the {kind}",
            f"take it in a body inside the {kind}, after the loop",
        )
    if not move.text.startswith("km::Moved"):
        move.text = f"km::Moved {moved} = {{0, 0}};\n" + move.text.replace(
            "movement.move(", f"movement.move({moved}, ", 1
        )
    bound.append(f"const uint32_t {name}_tiles = {moved}.tiles;")
    bound.append(f"const uint32_t {name}_cycles = {moved}.cycles;")


def take_own(frame, node, field, where, text, bound):
    """An own field a body takes: its buffer index by its name, on any CPU, no wait
    and no free. The CPU whose bodies push it (cb_push_back of the name: a mover's,
    or compute's pack CPU) is one: the field's push=, or the one the bodies show."""
    bound.append(f"const uint32_t {field} = kField{camel(field)};")
    if not re.search(rf"\bcb_push_back\s*\(\s*{field}\b", text):
        return
    cpu = {"A": "mover_a", "B": "mover_b"}.get(where, "pack")
    declared = frame.compiled.own[field]
    own = f"km.{spelling('own')}"
    if declared is None:
        refuse(
            frame.where,
            node,
            f"self.{field} is {own}(push=None): written in place, never pushed, "
            "and this body pushes it",
            f"drop the push, or declare {own}(push={cpu!r})",
        )
    if declared != "found" and declared != cpu:
        refuse(
            frame.where,
            node,
            f"self.{field} is pushed by {declared} ({own}(push={declared!r})), "
            f"and this body pushes it on {cpu}: a buffer has one producer",
            f"push it on {declared} alone",
        )
    pushers = frame.compiled.pushers.setdefault(field, {})
    others = {c: at for c, at in pushers.items() if c != cpu}
    if others:
        (other, (handler, line)), *_ = others.items()
        refuse(
            frame.where,
            node,
            f"self.{field} is pushed on {other} too ({handler}, line {line}): a "
            "buffer has one producer",
            f"push it on one CPU, and say which: {own}(push={other!r})",
        )
    pushers.setdefault(cpu, (frame.method.name, node.lineno))


def take_page(frame, node, name, value, bound, before, after):
    """A write stream's page for a body on mover B, after pack's body filled it
    outside a stream loop: mover B waits for the page, the body gets its L1
    address, and the movers write it after this body, not at pack's line."""
    if value.move is None or frame.stream_loops:
        refuse(
            frame.where,
            node,
            f"{name}'s page is mover B's after pack's body fills it, outside a "
            "stream loop",
            f"take {name} on pack first, then on mover B, outside the loop",
        )
    if not value.letters.startswith("B"):
        refuse(
            frame.where,
            node,
            f"{name}'s block is not mover B's to write",
            'give the write stream pattern="B"',
        )
    if value.waited:
        refuse(
            frame.where,
            node,
            f"{name}'s page is taken on mover B above",
            "take it in one body on mover B",
        )
    value.waited = True
    buffer, half = value.buffer.name, f"km_out_{name}"
    given = value.pattern if value.pattern in frame.scope else None
    before.append(
        Line(
            f"const uint32_t {half} = "
            f"km::block_half({buffer}, {value.pattern}, 0);\n"
            f"if (!movement.wait_front({half}, {buffer}Block)) {{\n"
            "    return false;\n}\ninvalidate_l1_cache();",
            uses=pattern_uses(given),
            mover="B",
        )
    )
    bound.append(f"const uint32_t {name} = get_read_ptr({half});")
    after.append(Line(value.move.text, set(value.move.uses)))  # both movers
    value.move.text = f"// {name}: written after mover B's body below"


def take_sum(frame, node, name, value, where, bound, before, after):
    """A reduce's sum of a field, taken by a body of compute's three at the root:
    the field's buffer and first tile bound (the group's sum, fp32), waited for
    before the body and freed after it, as the hand's operation on the sum."""
    field_ = value.summed
    if not frame.in_leader or value.popped:
        refuse(
            frame.where,
            node,
            f"{name} is the reduce's sum of self.{field_}, which only the leader "
            "holds, once",
            f"take it once in a body under `if self.{spelling('rank')} == "
            f"self.{spelling('leader')}:`",
        )
    held = dst_tiles.sum_held(frame, value)
    if not held:
        dst_tiles.same_loops(
            frame, node, name, value, "this body", *dst_tiles.once_text(frame, value)
        )
    if where != "compute":
        refuse(
            frame.where,
            node,
            f"{name} is the group's sum in compute's L1 (fp32 tiles of "
            f"self.{field_}): a body of compute's three takes it",
            f"drop {spelling('on')}= from this body",
        )
    value.popped = True
    frame.leader_writes.add(name)
    buffer = f"kField{camel(field_)}"
    before.append(
        Line(
            f"if (!compute.wait_front({buffer}, {buffer}Pages)) {{  // {name}\n"
            "    return;\n}"
        )
    )
    bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {buffer};")
    bound.append(f"const uint32_t {name}{spelling('tile_index')} = 0;")
    freed = f"cb_pop_front({buffer}, {buffer}Pages);  // {name} freed"
    if held:
        dst_tiles.sum_freed_after(frame, value, freed)
    else:
        after.append(Line(freed))
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    return set()
