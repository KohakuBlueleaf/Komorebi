"""A handler's broadcasts: km.Broadcast on its mover, the sender's cast and the
others' arrival, a loop over a block a broadcast brought, and a broadcast on a
rectangle given at each submit."""

import ast

from komorebi import abi
from komorebi.dsl import island_place
from komorebi.dsl.compile import layout
from komorebi.dsl.compile.cxx import camel, range_uses, text_uses
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import (
    WAIT_DONE,
    WAIT_FREE,
    WAIT_INPUT,
    WAIT_VALID,
    check_name,
    line_text,
    refuse,
    wait,
)
from komorebi.dsl.names import spelling
from komorebi.dsl.nodes import Block, Glue, Line


def arrived_loops(frame, node, name, value):
    """A block a broadcast brought is taken in the loops its cast stands in: the
    movers cast a block each time the cast runs, compute takes one each time the
    take runs."""
    frame.same_loops(
        node,
        name,
        value,
        "this take",
        "a cast makes one block each time its line runs and a take frees one each "
        "time its line runs, so the movers' blocks and compute's differ",
        "cast it in the loop that takes it (a block a turn), or take it once, "
        "outside the loops",
    )


def arrived_once(frame, node, name, value):
    """A block a broadcast brought is taken once: a body or a loop over it."""
    arrived_loops(frame, node, name, value)
    if value.popped:
        refuse(
            frame.where,
            node,
            "this block is taken above (a body or a loop over it), and freed",
            "take it once each block",
        )
    value.popped = True


def block_loop(frame, node, value):
    """`for tile in v:` over a block a broadcast brought: compute waits for it,
    takes its pages in order, frees its slot."""
    if not isinstance(node.target, ast.Name):
        refuse(frame.where, node, "one name a tile", "write for tile in block")
    held = getattr(value, "held", False)  # waited at its cast, freed after
    if not held:
        arrived_once(frame, node, node.iter.id, value)
    buffer = value.buffer.name
    page = f"km_page{value.buffer.index_a}"
    check_name(frame.where, node, node.target.id)
    frame.scope[node.target.id] = Value("streamed", buffer, first=page)
    body_m, body_c = frame.statements(node.body)
    frame.scope.pop(node.target.id)
    if body_m:
        refuse(
            frame.where,
            node.body[0],
            "inside a block's loop only compute's lines",
            "move the line before or after the loop",
        )
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    loop = Block(
        f"for (uint32_t {page} = 0; {page} < {buffer}Pages; {page}++) {{",
        "}",
        body_c,
    )
    if held:  # read in place: the cast's lines wait for it and free it
        return [], [Block("if (km_ok) {", "}", [loop])]
    compute = [
        Line(
            "km_ok = km_ok && km::broadcast_receive(compute, "
            f"{buffer}, {buffer}Pages);",
            {"km_ok"},
        ),
        Block(
            "if (km_ok) {",
            "}",
            [loop, Line(f"km::broadcast_done({buffer}, " f"{buffer}Pages);")],
        ),
    ]
    return [], compute


def broadcast(frame, node, name, call):
    """`b = km.Broadcast(self, along="row")`: the broadcast on its mover (a row on
    mover A, a column on mover B); a core it leaves out skips the handler."""
    if frame.method.overlay:
        refuse(
            frame.where,
            node,
            "a broadcast's handler is resident: its included test ends the "
            "command in the kernel",
            f"drop {spelling('overlay')}=True from this handler",
        )
    if node not in frame.definition.body or frame.dst_live is not None:
        refuse(
            frame.where,
            node,
            "a broadcast is made at the top of the handler",
            "make it before the tiles and loops",
        )
    if not (
        len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "self"
    ):
        refuse(
            frame.where,
            node,
            "a broadcast of this core",
            f'write km.{spelling("broadcast")}(self, along="row")',
        )
    along, window, placed = None, BROADCAST_WINDOW, {}
    for keyword_ in call.keywords:
        if keyword_.arg == spelling("along"):
            value = keyword_.value
            if not (isinstance(value, ast.Constant) and value.value in ALONG):
                refuse(
                    frame.where,
                    node,
                    "along= is a row or a column",
                    'write along="row" or along="column"',
                )
            along = value.value
        elif keyword_.arg == spelling("window"):
            window = frame.constant_int(keyword_.value)
        elif keyword_.arg in (
            spelling("broadcast_mover"),
            spelling("broadcast_length"),
            spelling("broadcast_unread"),
        ):
            placed[keyword_.arg] = keyword_.value
        else:
            refuse(
                frame.where,
                node,
                f"a broadcast has no {keyword_.arg}=",
                "use along=, window=, mover=, length= and unread=",
            )
    if along is None:
        refuse(
            frame.where,
            node,
            "a broadcast goes along a row or a column",
            'write along="row" or along="column"',
        )
    mover, length = broadcast_place(frame, node, along, placed)
    if mover in frame.movers:
        refuse(
            frame.where,
            node,
            f"{frame.movers[mover]} already runs on mover {mover}; each mover runs "
            "one broadcast (rows on mover A, columns on mover B unless mover= "
            "says)",
            "make one broadcast a mover: give the other mover=",
        )
    frame.movers[mover] = name
    index = class_mover(frame, node, mover)
    frame.compiled.broadcast_lines.setdefault(
        frame.method.name, f"line {node.lineno}, `{line_text(frame.where, node)}`"
    )
    frame.compiled.rectangle = True
    frame.compiled.rect_plain |= frame.rect_param is None
    frame.compiled.includes |= {"broadcast"}
    frame.compute_waits = True
    frame.scope[name] = Value(
        "broadcast", name, mover=mover, window=window, pattern=along, length=length
    )
    frame.scope[name].index = index  # its arguments' place among its mover's
    unread = placed.get(spelling("broadcast_unread"))
    if unread is not None:  # a word of the handler: its C++, the names it reads
        frame.scope[name].unread = (frame.integer(unread), frame.uses(unread))
    arg = "kArgBroadcast" + (f" + km::kBroadcastArgs * {index}" if index else "")
    if frame.rect_param is not None:  # its arguments computed on the core
        arg = rect_broadcast(frame, name, along, mover, window, length, index)
    movement = Line(
        f"km::Broadcast {name}(movement, command, {arg});\n"
        f"if (!{name}.ok()) {{\n    return false;\n}}\n"
        f"if (!{name}.included()) {{\n    return true;\n}}",
        {"km_grid", frame.rect_param} if frame.rect_param else set(),
        mover=mover,
    )
    counter = f"km_blocks_{name}"  # kept where casts count over loops
    counted = [
        Glue(f"uint32_t {counter} = 0;", {counter}, mover=mover),
        Glue(f"uint32_t {counter} = 0;", {counter}),
    ]
    compute = []
    if len(frame.movers) == 1:
        test, uses = included_test(frame, "compute")
        compute.append(Line(f"if ({test}) {{\n    break;\n}}", uses))
        return [movement, counted[0], other_mover_skips(frame, mover)], [
            *compute,
            counted[1],
        ]
    return [movement, counted[0]], [counted[1]]


def broadcast_branch(frame, node):
    """`if b.sender(): v = b.cast(pages.read())`, `else: v = b.arrive()`: on the
    sender the block read into its slot and cast, on the others its arrival."""
    test = node.test
    if not (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == spelling("sender")
        and isinstance(test.func.value, ast.Name)
        and frame.scope.get(test.func.value.id, Value("")).kind == "broadcast"
    ):
        return None
    name = test.func.value.id
    broadcast = frame.scope[name]
    drawn = (
        f"write `if {name}.{spelling('sender')}(): v = {name}.{spelling('cast')}"
        f"(t.pages(first, n).{spelling('read')}())`, `else: v = {name}."
        f"{spelling('arrive')}()`"
    )
    cast, arrive = (
        branch[0] if len(branch) == 1 else None for branch in (node.body, node.orelse)
    )
    if not (
        isinstance(cast, ast.Assign)
        and isinstance(arrive, ast.Assign)
        and len(cast.targets) == len(arrive.targets) == 1
        and isinstance(cast.targets[0], ast.Name)
        and isinstance(arrive.targets[0], ast.Name)
        and cast.targets[0].id == arrive.targets[0].id
        and is_cast(frame, cast.value, name)
        and is_call_of(frame, arrive.value, name, "arrive", 0)
    ):
        refuse(frame.where, node, "the sender casts a block, the others take it", drawn)
    value = cast.targets[0].id
    check_name(frame.where, cast, value)
    parts, count, dtype, page_bytes = cast_parts(frame, cast, drawn)
    buffer = broadcast.buffer
    if buffer is None:
        number = len(frame.compiled.broadcasts)
        buffer = layout.other_buffer(
            frame.compiled,
            f"kBroadcast{number}",
            "broadcast",
            dtype,
            page_bytes,
            broadcast.window * count,
            f"line {node.lineno}: {line_text(frame.where, node)}",
        )
        buffer.block = count
        broadcast.buffer = buffer
        frame.compiled.broadcasts.append(
            (
                broadcast.pattern,
                buffer,
                broadcast.window,
                broadcast.mover,
                broadcast.length,
                broadcast.index,
            )
        )
        if dtype == "float32":
            frame.compiled.unpack_fp32.add(buffer.name)
    elif (buffer.dtype, buffer.block) != (dtype, count):
        refuse(
            frame.where,
            cast,
            f"{name} casts blocks of {buffer.block} {buffer.dtype} " "pages already",
            "cast the same pages each time",
        )
    counted = counted_cast(frame, node, name, broadcast)  # None: the one loop's
    if name in frame.cast_flat:
        refuse(
            frame.where,
            node,
            f"{name} casts outside a loop already: its one block was finished "
            "there, and another cast would be block 0 again",
            "cast once outside a loop, or every time in one loop",
        )
    if counted is not None:
        block = counted
    elif frame.loops:
        loop = frame.loops[-1]
        if loop["first"] != "0":
            refuse(
                frame.where,
                node,
                "a broadcast's blocks count from 0",
                f"write for {loop['name']} in range(n)",
            )
        if broadcast.loop is not None and broadcast.loop is not loop:
            refuse(frame.where, node, f"{name} casts in two loops", "cast in one loop")
        broadcast.loop = loop
        if name not in loop["casts"]:
            loop["casts"].append(name)
        block = loop["name"]
    elif broadcast.loop is not None:
        refuse(frame.where, node, f"{name} casts in a loop already", "cast in one loop")
    else:
        block = "0"
        frame.cast_flat.add(name)
    address = f"km_address{buffer.index_a}"
    read_text = cast_reads(frame, parts, address, page_bytes)
    text = (
        f"if ({name}.sender()) {{\n"
        f"    const uint32_t {address} = {name}.reserve({block});\n"
        f"    if ({address} == 0 || !{read_text} || "
        f"!{name}.cast({block})) {{\n"
        "        return false;\n"
        "    }\n"
        f"}} else if (!{name}.arrive({block})) {{\n"
        "    return false;\n"
        "}"
    )
    uses = {name} | text_uses(block, frame)
    for part in parts:
        uses |= range_uses(part["pages"], frame)
    if broadcast.unread is not None:  # each core's slot pushed as it is, or cast
        text, uses = unread_push(frame, broadcast, text, uses)
    tensors = ",".join(sorted({part["pages"].tensor for part in parts}))
    frame.scope[value] = Value(
        "arrived",
        buffer.name,
        buffer=buffer,
        tensor=tensors,
        parts=tuple(slot_part(part) for part in parts),
        made_at=(len(frame.loops), node.lineno),
    )
    frame.scope[value].held = taken_by_field_products(frame, value) or taken_in_a_loop(
        frame, node, value
    )
    frame.scope[value].sides = tuple(  # each part's rectangle (rows, columns)
        part["pages"].sides or ("1", part["pages"].count) for part in parts
    )
    wait(frame.where, frame.compiled.waits, node, f"mover {broadcast.mover}", WAIT_FREE)
    wait(frame.where, frame.compiled.waits, node, f"mover {broadcast.mover}", WAIT_DONE)
    wait(
        frame.where,
        frame.compiled.waits,
        node,
        f"mover {broadcast.mover}",
        WAIT_VALID,
    )
    movement = [Line(text, uses, mover=broadcast.mover)]
    compute = []
    if counted is not None:  # the next block's number, on its mover and compute
        movement.append(Line(f"{counted}++;", {counted}, mover=broadcast.mover))
        compute.append(Line(f"{counted}++;", {counted}))
    if frame.scope[value].held:  # products take it: waited here, freed at the end
        w = buffer.name
        wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
        compute.append(
            Line(
                f"km_ok = km_ok && km::broadcast_receive(compute, {w}, {w}Pages);",
                {"km_ok"},
            )
        )
        frame.held_blocks.setdefault(frame.blocks[-1], []).append((value, w))
    if not frame.loops:
        frame.finished_casts[value] = broadcast.mover
        movement += end_outside_loop(frame, name)
    return movement, compute


def broadcast_end(frame, name, blocks, uses):
    """After a broadcast's last block: its mover's finish, compute's padding."""
    broadcast = frame.scope[name]
    buffer = broadcast.buffer.name
    movement = Line(
        f"if (!{name}.finish()) {{\n    return false;\n}}",
        {name},
        mover=broadcast.mover,
    )
    if broadcast.unread is not None:
        movement = unread_end(frame, broadcast, blocks, uses, movement)
    compute = Line(
        f"km_ok = km_ok && km::broadcast_end(compute, {buffer}, {buffer}Pages, "
        f"{buffer}Window, {blocks});",
        uses | {"km_ok"},
    )
    return [movement], [compute]


def broadcast_place(frame, node, along, placed):
    """A broadcast's mover and band: `mover="A"` (NoC 0: a column's sender at its
    top, a row's at its left) or `"B"` (NoC 1: bottom, right), by default its
    direction's (rows A, columns B); `length=L` bands of L cores, each with its
    sender (an int the build knows), by default the rectangle's whole side."""
    mover, length = ALONG[along], None
    given = placed.get(spelling("broadcast_mover"))
    if given is not None:
        if not (isinstance(given, ast.Constant) and given.value in ("A", "B")):
            refuse(
                frame.where,
                node,
                'a broadcast\'s mover is "A" (NoC 0) or "B" (NoC 1)',
                'write mover="A" or mover="B"',
            )
        mover = given.value
    given = placed.get(spelling("broadcast_length"))
    if given is not None:
        length = frame.known_int(given)
        if length is None or length < 1:
            refuse(
                frame.where,
                node,
                "a broadcast's length is cores a band, an int the build knows "
                "(a constant or a plan value) of 1 or more",
                "write length=L, L a module constant or a plan value",
            )
    return mover, length


def cast_parts(frame, cast, drawn):
    """A cast's parts in the slot in order, each `t.pages(first, n)` (n an int),
    `t.tiles[...]` (each side's most known) or `t.tiles[...].T` (its columns in
    turn), `.read()` or not: the parts (range, slot pages, pitch, first slot
    page, transposed), the block's pages, its format and page size."""
    parts, count, formats = [], 0, set()
    for read in cast.value.args:
        source = read
        if (
            isinstance(read, ast.Call)
            and isinstance(read.func, ast.Attribute)
            and read.func.attr == spelling("read")
            and not read.args
            and not read.keywords
        ):
            source = read.func.value
        transposed = (
            isinstance(source, ast.Attribute)
            and source.attr == spelling("transposed")
            and isinstance(source.value, ast.Subscript)
            and frame.is_tiles(source.value.value)
        )
        if transposed:
            source = source.value
        if not (
            isinstance(source, ast.Call)
            and frame.is_range(source)
            and len(source.args) == 2
            or isinstance(source, ast.Subscript)
            and frame.is_tiles(source.value)
        ):
            refuse(frame.where, cast, "a cast sends pages read from a tensor", drawn)
        pages = frame.range_of(source)
        frame.pages_only(cast, pages)
        pitch = None  # a row's pages in the slot, where the columns are a word
        if isinstance(source, ast.Subscript):
            if pages.count.isdigit():
                most = int(pages.count)
            elif None in pages.bounds:
                frame.stated_words(source, "a cast's slot")
                refuse(
                    frame.where,
                    cast,
                    "a cast's slot holds the rectangle's most rows and columns, "
                    "and the build knows no most for a side here",
                    f"write each side r:r + m or r:{spelling('min')}(r + m, e), m "
                    "a constant; rows s apart r:r + m * s:s or r:"
                    f"{spelling('min')}(r + m * s, e):s",
                )
            else:  # the slot of the most; the rows at the most columns' pitch
                most = pages.bounds[0] * pages.bounds[1]
                pitch = pages.bounds[1]
            if transposed:  # a column of the rectangle a row of the slot
                rows = pages.bounds[0] if None not in pages.bounds else None
                pitch = rows or int(pages.sides[0])
        else:
            most = frame.constant_int(source.args[1])
        formats.add(frame.tensor_format(cast, pages.tensor))
        frame.moved(cast, pages.tensor, "broadcast's read")
        parts.append(
            {
                "pages": pages,
                "most": most,
                "pitch": pitch,
                "offset": count,
                "transposed": transposed,
            }
        )
        count += most
    if len(formats) > 1:
        refuse(
            frame.where,
            cast,
            "a cast's block is of one format and page size, and its parts are "
            f"of {len(formats)}",
            "cast parts of tensors of one format",
        )
    dtype, page_bytes = formats.pop()
    return parts, count, dtype, page_bytes


def cast_reads(frame, parts, address, page_bytes):
    """The C++ that reads a cast's parts into the slot at `address`, each at its
    first slot page: one part's read as it is, several's joined by &&."""
    reads = []
    for part in parts:
        pages, pitch = part["pages"], part["pitch"]
        at = address
        if part["offset"]:
            at = f"{address} + {part['offset'] * page_bytes}"
        tensor = f"{camel(pages.tensor)}({pages.tensor})"
        if part["transposed"]:  # each column of the rectangle, a page a row
            rows, columns = pages.sides
            row_step = (
                pages.rect[2] if pages.rect is not None else
                frame.method.widths[pages.tensor]
            )  # fmt: skip
            reads.append(
                "[&]() {\n"
                f"        for (uint32_t km_column = 0; km_column < {columns}; "
                "km_column++) {\n"
                f"            for (uint32_t km_row = 0; km_row < {rows}; "
                "km_row++) {\n"
                "                if (!km::read_pages(movement, command, "
                f"{tensor}, {pages.first} + km_row * {row_step} + km_column, 1, "
                f"{at} + (km_column * {pitch} + km_row) * {page_bytes})) {{\n"
                "                    return false;\n"
                "                }\n"
                "            }\n"
                "        }\n"
                "        return true;\n"
                "    }()"
            )
        elif pages.rect is not None:  # the rectangle's rows into the slot, in order
            rows, columns, row_step = pages.rect[:3]
            pitch = pitch or columns
            reads.append(
                "[&]() {\n"
                f"        for (uint32_t km_row = 0; km_row < {rows}; km_row++) {{\n"
                "            if (!km::read_pages(movement, command, "
                f"{tensor}, {pages.first} + km_row * {row_step}, "
                f"{columns}, {at} + km_row * {pitch * page_bytes})) {{\n"
                "                return false;\n"
                "            }\n"
                "        }\n"
                "        return true;\n"
                "    }()"
            )
        else:
            count = part["most"] if pitch is None else pages.count
            reads.append(
                f"km::read_pages(movement, command, {tensor}, "
                f"{pages.first}, {count}, {at})"
            )
    return reads[0] if len(reads) == 1 else "(" + " && ".join(reads) + ")"


def class_mover(frame, node, mover):
    """The place of this handler's broadcast among its mover's in the class: a
    core holds each one's arguments after the last's. A core has KM_BROADCASTS
    broadcast areas (a broadcast's number picks one): a class past them is
    refused, naming the broadcasts it has."""
    line = f"line {node.lineno}, `{line_text(frame.where, node)}`"
    held = frame.compiled.mover_broadcasts.setdefault(mover, [])
    for index, (handler, _) in enumerate(held):
        if handler == frame.method.name:  # the handler made again (a variant)
            return index
    areas = abi.headers()["KM_BROADCASTS"]
    others = [b for each in frame.compiled.mover_broadcasts.values() for b in each]
    if len(others) >= areas:
        named = "; ".join(f"{h}, {at}" for h, at in others)
        refuse(
            frame.where,
            node,
            f"the class runs {len(others)} broadcasts already ({named}); a core "
            f"has {areas} broadcast areas, one a broadcast of the class",
            "make this handler's broadcast in a class of its own",
        )
    held.append((frame.method.name, line))
    return len(held) - 1


def counted_cast(frame, node, name, broadcast):
    """A cast inside more than one loop, or a loop not from 0: its blocks counted
    over every loop around it by a word of its mover and of compute, the finish
    after the outermost loop with the count (None: one loop from 0 counts them)."""
    if not frame.loops or len(frame.loops) == 1 and frame.loops[0]["first"] == "0":
        return None
    outer = frame.loops[0]
    if broadcast.loop is not None and broadcast.loop is not outer:
        refuse(
            frame.where,
            node,
            f"{name} casts in two loops of the handler's top",
            "cast in one loop, at any depth",
        )
    broadcast.loop = outer
    counted = outer.setdefault("counted", [])
    if name not in counted:
        counted.append(name)
    return f"km_blocks_{name}"


def counted_ends(frame, loop):
    """After the outermost loop around counted casts: each broadcast's finish and
    compute's padding, by the blocks counted."""
    movement, compute = [], []
    for name in loop.get("counted", ()):
        counter = f"km_blocks_{name}"
        m, c = broadcast_end(frame, name, counter, {counter})
        movement += m
        compute += c
    return movement, compute


def end_outside_loop(frame, name):
    """The ends of a cast outside any loop (one block): the mover's finish, which
    stays at the cast, and compute's padding pop, which goes to compute's end
    (`closing`) after the block's last use: the finish pushes the padding only once
    compute popped the block, and the pop waits for it."""
    movement_end, compute_end = broadcast_end(frame, name, "1", set())
    frame.closing += compute_end
    return movement_end


def included_test(frame, me):
    """The C++ of `the core is outside the rectangle`."""
    if frame.rect_param is not None:
        return "km_place.included == 0", {"km_place"}
    return f"{me}.arg(kArgIncluded) == 0", set()


def is_call_of(frame, node, name, kind, arguments):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == spelling(kind)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == name
        and len(node.args) == arguments
        and not node.keywords
    )


def is_cast(frame, node, name):
    """`name.cast(part, ...)`: one part or more, no keywords."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == spelling("cast")
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == name
        and len(node.args) >= 1
        and not node.keywords
    )


def other_mover_skips(frame, mover):
    """The mover a handler's first broadcast does not run skips the handler on a
    core the rectangle leaves out, as that broadcast's mover and compute do."""
    test, uses = included_test(frame, "movement")
    return Line(
        f"if ({test}) {{\n    return true;\n}}",
        uses,
        mover="B" if mover == "A" else "A",
    )


def rect_broadcast(frame, name, along, mover, window, length, index):
    """A broadcast's arguments on a rectangle given at each submit: the build's
    (its number, NoC, buffer, window, pages), the command's group, and rank,
    receivers, sender and corners computed on the core (km::rect_broadcast)."""
    args, place = f"km_args_{name}", f"{mover}{index}"
    frame.compiled.rect_broadcasts.add((mover, index))
    return (
        f"[&]() {{\n"
        f"    km::BroadcastArgs {args}{{}};\n"
        f"    {args}.broadcast = kRectId{place};\n"
        f"    {args}.group = command.group;\n"
        f"    {args}.noc = {0 if mover == 'A' else 1}u;\n"
        f"    {args}.cb = kRectCb{place};\n"
        f"    {args}.window = kRectWindow{place};\n"
        f"    {args}.pages = kRectPages{place};\n"
        f"    {args}.next = km::kBroadcastNoNext;\n"
        f"    {island_place.broadcast(args, 'movement', frame.compiled.islands)}"
        f"{frame.rect_param}, {'true' if along == 'row' else 'false'}, "
        f"{length or 0}u, km::kBroadcastNone);\n"
        f"    return {args};\n"
        f"}}()"
    )


def taken_by_field_products(frame, name):
    """Whether products with a field on the left take block `name`
    (`f(self.p, name[i][r0:r0 + n, :])`): it is then waited for at its cast and
    freed at the end of the cast's lines, for any number of them."""
    for node in ast.walk(frame.definition):
        if (
            isinstance(node, ast.Call)
            and len(node.args) == 2
            and (
                frame.field_name(node.args[0]) is not None
                or frame.held_row(node.args[0]) is not None
            )
            and any(
                isinstance(n, ast.Name) and n.id == name for n in ast.walk(node.args[1])
            )
        ):
            return True
    return False


def taken_in_a_loop(frame, cast, name):
    """Whether block `name`, cast at `cast`, is taken in a range loop the cast does
    not stand in (a block held across turns: a weight cast once and read each turn):
    it is then waited for at its cast and freed at the end of the cast's lines,
    and every take reads it in place."""
    for loop in ast.walk(frame.definition):
        if (
            not isinstance(loop, ast.For)
            or not isinstance(loop.iter, ast.Call)
            or not isinstance(loop.iter.func, ast.Name)
            or loop.iter.func.id != "range"
            or any(n is cast for n in ast.walk(loop))
        ):
            continue  # a range loop the cast does not stand in
        if any(
            isinstance(n, ast.Name) and n.id == name
            for line in loop.body
            for n in ast.walk(line)
        ):
            return True
    return False


def unread_end(frame, broadcast, blocks, uses, finish):
    """After the last block of a broadcast pushed unread: its mover pads the slots
    to the window, as the finish does; else the finish."""
    word, reads = broadcast.unread
    buffer = broadcast.buffer.name
    pad = f"km_pad_{broadcast.cxx}"
    text = (
        f"if ({word}) {{\n"
        f"    const uint32_t {pad} = ({buffer}Window - ({blocks}) % "
        f"{buffer}Window) % {buffer}Window * {buffer}Pages;\n"
        f"    if ({pad} != 0) {{\n"
        f"        if (!movement.reserve({buffer}, {pad})) {{\n"
        "            return false;\n"
        "        }\n"
        f"        cb_push_back({buffer}, {pad});\n"
        "    }\n"
        f"}} else {finish.text}"
    )
    return Line(text, finish.uses | uses | reads, mover=broadcast.mover)


def unread_push(frame, broadcast, text, uses):
    """`unread=word`: when the word is not 0, each core's mover takes the block's
    slot and pushes it as it is (no read, no cast, no arrival); else the cast."""
    word, reads = broadcast.unread
    buffer = broadcast.buffer.name
    text = (
        f"if ({word}) {{\n"
        f"    if (!movement.reserve({buffer}, {buffer}Pages)) {{\n"
        "        return false;\n"
        "    }\n"
        f"    cb_push_back({buffer}, {buffer}Pages);\n"
        f"}} else {text}"
    )
    return text, uses | reads


# A broadcast's mover: along a row on mover A (NoC 0), down a column on mover B (NoC 1).
ALONG = {"row": "A", "column": "B"}


# Blocks a broadcast holds in flight when the plan does not say.
BROADCAST_WINDOW = 2


def slot_part(part):
    """A cast part's place in the slot: (first page, rows, pitch, transposed), the
    rows and pitch ints where the build knows them (None where a word)."""
    pages, pitch = part["pages"], part["pitch"]
    sides = pages.sides if pages.sides is not None else ("1", pages.count)
    bounds = pages.bounds or (None, None)
    rows, columns = (
        int(side) if str(side).isdigit() else most for side, most in zip(sides, bounds)
    )
    if part["transposed"]:  # a column of the rectangle a row of the slot
        rows, columns = columns, (pitch or rows)
    elif pitch is not None:
        columns = pitch
    return part["offset"], rows, columns, part["transposed"]
