"""A handler's streams: a read loop over one stream or several in step, a write
stream it feeds, a held stream of one block, a drain, and a stream's options (its
block, pattern and word)."""

import ast

from komorebi.dsl.compile import expressions, handler_buffers, layout, scope
from komorebi.dsl.compile.cxx import pattern_uses, range_uses, text_uses
from komorebi.dsl.compile.dst import dst_tiles
from komorebi.dsl.compile.layout import READ_ITEMS, WRITE_ITEMS
from komorebi.dsl.compile.pages import ranges
from komorebi.dsl.compile.records import StreamOptions, Value
from komorebi.dsl.compile.source import (
    WAIT_DATA,
    WAIT_INPUT,
    WAIT_OUTPUT,
    check_name,
    line_text,
    refuse,
    wait,
)
from komorebi.dsl.front import front
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Block, Glue, Line
from komorebi.program.movement import Pattern


def block_word(frame, node, buffer, word):
    """A stream's pair whose block is given at each submit (`word`, else None):
    the handler's `<pair>Block` a word, its constant the most (the pair's pages)
    elsewhere; a pair one handler streams with two blocks refused."""
    if frame.block_words.setdefault(buffer.name, word) != word:
        refuse(
            frame.where,
            node,
            f"pair {buffer.name} is streamed with block {word or 'a constant'} "
            f"and with {frame.block_words[buffer.name] or 'a constant'} in one "
            "handler",
            "give the two streams one block, or tensors of other formats",
        )


def shared_block(frame, buffer):
    """A read pair another handler made with another block (layout.SharedPair): this
    handler's `<pair>Block` its own block (block_lines)."""
    if isinstance(buffer, layout.SharedPair):
        frame.block_shadows[buffer.name] = buffer.block


def drain(frame, node, call):
    """`km.Stream(pages, ...).drain()`: the stream read and dropped, as a loop with
    an empty body."""
    loop = ast.For(
        target=ast.Name("_", ast.Store()),
        iter=call.func.value,
        body=[ast.Pass()],
        orelse=[],
    )
    ast.copy_location(loop, node)
    loop.end_lineno = node.end_lineno
    loop.body[0].lineno = loop.body[0].end_lineno = node.lineno
    return read_loop(frame, loop, call.func.value)


def held_row(frame, arg):
    """(the held stream's Value, the row's C++) of `qs[r]`, qs a held stream;
    else None."""
    if not (isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name)):
        return None
    value = frame.scope.get(arg.value.id)
    if value is None or value.kind != "held":
        return None
    return value, arg.slice


def held_stream(frame, node, name, call):
    """`q = km.Stream(t.tiles[...], block=b)` outside a for line: a read stream of
    one block that bodies take whole. The movers move it at this line; compute
    waits for its block here and frees it at the end of the lines it stands in
    (held_ends), so the loops between may hold movers' lines too."""
    options = stream_options(frame, node, call, False)
    pages, block, pattern, letters, repeat, row, word = options
    if row != 1 or repeat != "1" or pages.items is not None:
        refuse(
            frame.where,
            node,
            "a stream bodies take is one block of pages, read once",
            f"drop {spelling('row')}=, {spelling('repeat')}= and items, or "
            f"write `for tile in km.{spelling('stream')}(...)`",
        )
    most = held_most(pages)
    if most is None or most > block:
        known = "unknown to the build" if most is None else most
        refuse(
            frame.where,
            node,
            f"a stream bodies take is one block, and its pages are at most "
            f"{known}, its block {block}",
            f"give {spelling('block')}= its most pages (a side r:min(r + n, e), or "
            "rows r:min(r + n * s, e):s), or iterate it in a for line",
        )
    if not (letters or "").startswith("A"):
        refuse(
            frame.where,
            node,
            "a stream bodies take is mover A's block (an empty one is pushed "
            "by mover A)",
            'drop pattern=, or give one that starts with "A"',
        )
    dtype, page_bytes = handler_buffers.tensor_format(
        frame, node, pages.tensor, paged=True
    )
    page_bytes = handler_buffers.ring_page(frame, pages.tensor, page_bytes)
    handler_buffers.paged_stream(frame, node, pages, block)
    index = frame.reads
    frame.reads += 1
    buffer = layout.pair(
        frame.compiled,
        "read",
        index,
        dtype,
        page_bytes,
        block,
        1,  # one block a half: the block is held until the end of its lines
        f"line {node.lineno}: {line_text(frame.where, node)}",
        frame.method.name,
    )
    shared_block(frame, buffer)
    if word is not None:
        refuse(
            frame.where,
            node,
            f"a block bodies take, given at each submit ({word}), is " "not built",
            "give block= its most pages, a constant",
        )
    pattern_text = handler_buffers.pattern_cxx(frame, buffer, pattern, letters)
    if dtype == "float32":
        frame.compiled.unpack_fp32.add(buffer.name)
    handler_buffers.moved(frame, node, pages.tensor, "read stream")
    variable, half = f"km_read{index}", f"km_held_{name}"
    stream_uses = range_uses(pages, frame) | pattern_uses(pattern)
    movement = [
        Glue(
            handler_buffers.stream_cxx(
                frame, "read", variable, pages, buffer, pattern_text, "1"
            ),
            {variable},
            stream_uses,
        ),
        *held_move(variable, half, buffer.name, pages.count),
    ]
    for line in movement[1:]:
        line.uses |= text_uses(pages.count, frame)
    wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_DATA)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    frame.compute_waits = True
    compute = []
    if pattern is not None:
        compute.append(Line(f"if (!{pattern}.valid()) {{\n    return;\n}}", {pattern}))
    compute.append(
        Line(
            f"const uint32_t {half} = km::block_half({buffer.name}, "
            f"{pattern_text}, 0);\n"
            f"km_ok = km_ok && compute.wait_front({half}, {buffer.name}Block);",
            {"km_ok"} | pattern_uses(pattern),
        )
    )
    frame.scope[name] = Value("held", half, buffer=buffer)
    bounds = getattr(pages, "bounds", None)  # a row's tiles, for products
    frame.scope[name].columns = (
        bounds[1] if bounds and isinstance(bounds[1], int) else None
    )
    frame.held.setdefault(frame.blocks[-1], []).append((name, node))
    return movement, compute


def items_alone(frame, node, row):
    """Refuses what items do not take: a stream of rows of them (`row` tiles a
    row)."""
    if row > 1:
        refuse(
            frame.where,
            node,
            "items are not tiles: a stream of them has no rows",
            f"drop {spelling('row')}=",
        )


def item_copy(frame, node, name, stream, item_name, item):
    """`out.push(item)`, an item of the loop's stream of items into a write stream of
    items of its size: no compute line; read_loop has mover A read the items into
    the loop's halves and mover B write them from there (copy_moves)."""
    if stream.items != item.items:
        refuse(
            frame.where,
            node,
            f"{item_name} is an item of {item.items} B and {name} writes items of "
            f"{stream.items} B: a mover copies an item whole, into an item of its size",
            f"give both streams size={item.items}",
        )
    state = frame.loop_states[-1]
    if item.cxx != state["first"] or len(state["letters"]) != 1:
        refuse(
            frame.where,
            node,
            f"{item_name} is copied by the movers of the loop that reads it, alone",
            f"push {item_name} in its own loop, one stream read",
        )
    if stream.pushed or state.get("copy") is not None:
        refuse(
            frame.where,
            node,
            "an item copy's loop pushes one item into one write stream, once",
            "open a loop for each copy",
        )
    loop_fed(frame, node, name, stream)
    stream.pushed = True
    state["copy"] = (node, name, stream)
    return [], []


def copy_moves(frame, node, read, state, call, body_c):
    """The movers' lines of an item copy (item_copy): mover A reads every block into
    its half of the loop's pair, mover B writes it from that half (the write stream's
    pair the read's, swapped), so the write stream's own pair is dropped when this
    line made it."""
    copy_node, name, stream = state["copy"]
    line, pages, repeat, stated, made = stream.copy
    if body_c or len(state["after"]) or len(state["inits"]):
        refuse(
            frame.where,
            copy_node,
            "an item copy's loop holds the push alone: compute takes no item",
            "move the other lines out of the loop",
        )
    if stated or any(k.arg == "pattern" for k in call.keywords):
        refuse(
            frame.where,
            copy_node,
            "an item copy is mover A's read and mover B's write: its streams take "
            "no pattern",
            "drop pattern= from both streams",
        )
    buffer = read["buffer"]
    if made:
        frame.compiled.buffers.remove(stream.buffer)
    read_pattern = handler_buffers.pattern_cxx(frame, buffer, None, "A")
    write_pattern = handler_buffers.pattern_cxx(frame, buffer, None, "B")
    swapped = f"km::BufferPair{{{buffer.name}.mover_b, {buffer.name}.mover_a}}"
    line.text = handler_buffers.stream_cxx(
        frame, "write", stream.cxx, pages, buffer, write_pattern, repeat, swapped
    )
    return handler_buffers.stream_cxx(
        frame,
        "read",
        read["variable"],
        read["pages"],
        buffer,
        read_pattern,
        read["repeat"],
    )


def loop_fed(frame, node, name, stream):
    """A write stream a read loop feeds: a block of its pages a block of the
    loop's, and its count the loop's by the same expression."""
    state = frame.loop_states[-1]
    block, count = stream.buffer.block, stream.count
    if block != state["block"] or count != state["count"]:
        refuse(
            frame.where,
            node,
            f"{name} is {count} pages in blocks of {block}, and the loop's stream "
            f"(line {state['line']}) {state['count']} in blocks of "
            f"{state['block']}: a block of the loop's is a block of {name}'s",
            f"open {name} with the loop's count, {state['count']}, and block="
            f"{state['block']}",
        )


def stream_pattern(frame, node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            Pattern(node.value)
        except ValueError as error:
            refuse(frame.where, node, str(error), "write letters A and B")
        return None, node.value
    if isinstance(node, ast.Name) and frame.scope.get(node.id, Value("")).kind == (
        "pattern"
    ):
        return node.id, None
    refuse(
        frame.where,
        node,
        "a pattern is letters or a pattern parameter",
        'write pattern="AB"',
    )


def left_first(reads, streams):
    """The reads (and their streams) with the one of the smallest block first, when
    another is zipped first: a product's left side, rows of k tiles, counts the loop
    and its last block (`zip(w_stream, x_stream)` as `zip(x_stream, w_stream)`)."""
    blocks = [read["block"] for read in reads]
    if not all(isinstance(block, int) for block in blocks):
        return reads, streams
    least = blocks.index(min(blocks))
    if blocks[least] == blocks[0]:
        return reads, streams
    order = [least, *(i for i in range(len(reads)) if i != least)]
    return [reads[i] for i in order], [streams[i] for i in order]


def read_loop(frame, node, call, streams=None):
    """A loop over one read stream, or over several read in step (`streams`:
    each tile's name and its stream's call): one move of them and of the write
    streams the loop pushes to; compute takes block k of each from its half."""
    if streams is None:
        if not isinstance(node.target, ast.Name):
            refuse(
                frame.where,
                node,
                "one name a tile",
                "write for tile in km.Stream(...)",
            )
        streams = [(node.target.id, call)]
    number = frame.reads  # the loop's own names: block, tile, valid, out, room
    block_var, tile, valid = (
        f"km_block{number}",
        f"km_tile{number}",
        f"km_valid{number}",
    )
    reads = []
    for name, stream_call in streams:
        check_name(frame.where, node, name)
        options = stream_options(frame, node, stream_call, False)
        pages, block, pattern, letters, repeat, row, word = options
        if row > 1 and len(streams) > 1:
            refuse(
                frame.where,
                node,
                "a stream of rows is read alone",
                f"write `for row in km.{spelling('stream')}(..., row=2)`",
            )
        dtype, page_bytes = handler_buffers.tensor_format(
            frame, node, pages.tensor, paged=True
        )
        page_bytes = handler_buffers.ring_page(frame, pages.tensor, page_bytes)
        handler_buffers.paged_stream(frame, node, pages, block)
        if pages.items is not None:  # the buffer's pages are the items
            page_bytes = pages.items[0]
            items_alone(frame, node, row)
        index = frame.reads
        frame.reads += 1
        buffer = layout.pair(
            frame.compiled,
            "read",
            index,
            dtype,
            page_bytes,
            block,
            READ_ITEMS,
            f"line {node.lineno}: {line_text(frame.where, node)}",
            frame.method.name if word is None else None,  # a word: a pair alone
        )
        shared_block(frame, buffer)
        block_word(frame, node, buffer, word)
        pattern_text = handler_buffers.pattern_cxx(frame, buffer, pattern, letters)
        if dtype == "float32" and pages.items is None:
            frame.compiled.unpack_fp32.add(buffer.name)
        handler_buffers.moved(frame, node, pages.tensor, "read stream")
        reads.append(
            {
                "name": name,
                "variable": f"km_read{index}",
                "half": f"km_half{index}",
                "pages": pages,
                "block": block,
                "pattern": pattern,
                "pattern_text": pattern_text,
                "letters": letters,
                "repeat": repeat,
                "buffer": buffer,
                "width": row,
                "word": word,
            }
        )
    reads, streams = left_first(reads, streams)
    first = reads[0]
    rows_in_step = False  # blocks of whole rows, as many blocks: a product's
    for other in reads[1:]:
        if (
            other["block"],
            other["pages"].count,
            other["repeat"],
            other["word"],
        ) == (first["block"], first["pages"].count, first["repeat"], first["word"]):
            continue
        if other["word"] or first["word"]:
            refuse(
                frame.where,
                node,
                "streams read in step, one with a block given at each submit, "
                "have one count and the same block parameter",
                "give each stream block= the same parameter",
            )
        if stream_units(other) & stream_units(first) and other["repeat"] == (
            first["repeat"]
        ):
            rows_in_step = True
            continue
        refuse(
            frame.where,
            node,
            "streams read in step have one count, block and repeat (or, for a "
            "product, a row of the second's rectangle a tile of the first's)",
            "give each the same count of pages, block= and repeat=, or stream "
            f"x.{spelling('tiles')}[0:1, k0:k1] with {spelling('block')}=b beside "
            f"w.{spelling('tiles')}[k0:k1, c:c + n] with {spelling('block')}=b * n",
        )
    for read in reads:
        frame.scope[read["name"]] = Value(
            "streamed",
            read["half"],
            first=tile,
            buffer=read["buffer"],
            width=read["width"],
        )
        items = read["pages"].items  # an item's bytes: no tile of compute
        frame.scope[read["name"]].items = items and items[0]
    before = set(frame.scope)
    writes_before = [v for v in frame.scope.values() if v.kind == "write"]
    pushed_before = {id(v): v.pushed for v in writes_before}
    frame.stream_loops.append(number)
    frame.loop_states.append(
        {
            "function": None,
            "other": None,
            "inits": [],
            "after": [],
            "block": reads[0]["block"],
            "first": reads[0]["half"],
            "count": reads[0]["pages"].count,  # the loop's tiles, as written
            "tensors": {r["buffer"].name: r["pages"].tensor for r in reads},
            "letters": [r["letters"] for r in reads],  # None: a pattern word
            "repeat": reads[0]["repeat"],
            "most": held_most(reads[0]["pages"]),  # the most pages, where known
            "line": node.lineno,
            "body": node.body,  # its statements: a result's later readers
        }
    )
    handler_buffers.held_cast_movers(frame, node, [r["letters"] for r in reads])
    body_m, body_c = frame.statements(node.body)
    body_c += dst_tiles.scan_end(frame, frame.loop_states[-1])
    state = frame.loop_states.pop()
    results_pushed(frame, node, state)
    frame.stream_loops.pop()
    if first["block"] < first["width"]:
        row_passes_read(frame, node, first, state, number)
    if body_m:
        refuse(
            frame.where,
            node.body[0],
            "inside a stream loop only compute's lines",
            "move the line before or after the loop",
        )
    if rows_in_step and (state["function"] != "product" or state["other"]):
        refuse(
            frame.where,
            node,
            "streams in step with blocks of different sizes are a product's "
            "operands: a tile of one is a row of the other",
            "keep the loop to `acc += self.f(a, b)`, f `return a @ b`",
        )
    scope.scoped(frame, "loop", node, before)
    for read in reads:
        frame.scope.pop(read["name"], None)
    pushed = [v for v in writes_before if v.pushed and not pushed_before[id(v)]]
    # The movers: these streams and the write streams they feed, in one move.
    movement = []
    copied = state.get("copy") is not None  # an item copy: the movers' alone
    for read in reads:
        stream_uses = (
            range_uses(read["pages"], frame)
            | pattern_uses(read["pattern"])
            | text_uses(read["repeat"], frame)
        )
        text = (
            copy_moves(frame, node, read, state, streams[0][1], body_c)
            if copied
            else handler_buffers.stream_cxx(
                frame,
                "read",
                read["variable"],
                read["pages"],
                read["buffer"],
                read["pattern_text"],
                read["repeat"],
            )
        )
        movement.append(Glue(text, {read["variable"]}, stream_uses))
    if len(reads) > 1:
        frame.compiled.zips[(frame.method.name, node.lineno)] = [
            (read["variable"], read["pages"].tensor, stated_prefetch(call))
            for read, (_, call) in zip(reads, streams, strict=True)
        ]
    variables = [read["variable"] for read in reads] + [v.cxx for v in pushed]
    moved = ", ".join(variables)
    move = Line(
        f"if (!movement.move({moved})) {{\n    return false;\n}}", set(variables)
    )
    movement.append(move)
    movement.extend(  # a sum a block into each field, as compute counts them
        Line(
            f"{sums} += ({first['pages'].count} + "
            f"{first['buffer'].name}Block - 1) / {first['buffer'].name}Block;",
            range_uses(first["pages"], frame),
            mover="B",
        )
        for sums in state.get("sums", ())
    )
    for read in reads:  # after the loop: what each mover moved (a mover's body)
        frame.scope[read["name"]] = Value("moved", f"km_moved{number}", move=move)
    wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_DATA)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    if copied:  # compute takes no item
        return movement, []
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    # Compute: block k of each stream from the half its owner filled; its tiles.
    frame.compute_waits = True
    buffer, pages, repeat = first["buffer"], first["pages"], first["repeat"]
    blocks = f"({pages.count} + {buffer.name}Block - 1) / {buffer.name}Block"
    per_turn = f"km_blocks{number}"
    count_uses = text_uses(pages.count, frame)
    compute = []
    checked = set()
    for read in reads:
        if read["pattern"] is not None and read["pattern"] not in checked:
            checked.add(read["pattern"])
            compute.append(
                Line(
                    f"if (!{read['pattern']}.valid()) {{\n    return;\n}}",
                    {read["pattern"]},
                )
            )
    compute.append(
        Glue(f"const uint32_t {per_turn} = {blocks};", {per_turn}, count_uses)
    )
    compute += state["inits"]
    total = per_turn if repeat == "1" else f"{per_turn} * {repeat}"
    inner = []
    for read in reads:
        half, name = read["half"], read["buffer"].name
        inner.append(
            Line(
                f"const uint32_t {half} = km::block_half({name}, "
                f"{read['pattern_text']}, {block_var});\n"
                f"km_ok = compute.wait_front({half}, {name}Block);\n"
                f"if (!km_ok) {{\n    break;\n}}",
                uses=pattern_uses(read["pattern"]),
            )
        )
    inner.append(
        Glue(
            f"const uint32_t {valid} = {block_var} % {per_turn} + 1 == "
            f"{per_turn} ? {pages.count} - ({per_turn} - 1) * "
            f"{buffer.name}Block : {buffer.name}Block;",
            {valid},
            count_uses,
        )
    )
    for write in pushed:
        out, room = out_names(number, write, state)
        # a row read in passes: its blocks written in the last pass alone
        written, when = state.get("written", (block_var, ""))
        inner.append(
            Line(
                f"const uint32_t {out} = km::block_half({write.buffer.name}, "
                f"{write.pattern}, {written});\n"
                f"const bool {room} = {when}compute.reserve_back({out}, "
                f"{write.buffer.name}Block);",
                uses=pattern_uses(write.pattern),
            )
        )
        wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
    if pushed and state["function"] is None and not state.get("body_pushed"):
        inner.append(
            Line(
                f"KM_UNPACK_FORMAT_A({first['half']});\n" f"copy_init({first['half']});"
            )
        )
    width = state.get("step", first["width"])
    step = f"{tile}++" if width == 1 else f"{tile} += {width}"
    loop = f"for (uint32_t {tile} = 0; {tile} < {valid}; {step}) {{"
    if body_c and state.get("whole"):  # the body takes the block whole
        inner += body_c
    elif body_c:
        inner.append(Block(loop, "}", body_c, {valid}))
    for write in pushed:
        out, room = out_names(number, write, state)
        inner.append(
            Line(
                f"if ({room}) {{\n    cb_push_back({out}, "
                f"{write.buffer.name}Block);\n}}"
            )
        )
    inner += [
        Line(f"cb_pop_front({read['half']}, {read['buffer'].name}Block);")
        for read in reads
    ]
    head = (
        f"for (uint32_t {block_var} = 0; km_ok && {block_var} < {total}; "
        f"{block_var}++) {{"
    )
    uses = {per_turn, "km_ok"} | text_uses(repeat, frame)
    compute.append(Block(head, "}", inner, uses))
    compute += state["after"]
    return movement, compute


def results_pushed(frame, node, state):
    """A loop that calls a function of several results pushes each."""
    group = state.get("results")
    if group is None or len(group["packs"]) == len(group["names"]):
        return
    left = [n for n in group["names"] if n not in group["packs"]]
    refuse(
        frame.where,
        node,
        f"the results {', '.join(group['names'])} of the call at line "
        f"{group['line']} are pushed in its loop, and {', '.join(left)} is not",
        f"push {left[0]} into a write stream in the loop",
    )


def row_passes_read(frame, node, read, state, number):
    """A row longer than its stream's block: the range read once a pass of the
    row function's (`state["passes"]`), its blocks written in the last."""
    passes = state.get("passes")
    if passes is None:
        refuse(
            frame.where,
            node,
            f"a block of {read['block']} tiles is a part of a row of "
            f"{read['width']}, and a row read in parts is a row function's",
            "push a row function's call on the row, or give block= the row",
        )
    if read["repeat"] != "1":
        refuse(
            frame.where,
            node,
            "a row read in parts is read again once a pass, by itself",
            f"drop {spelling('repeat')}=",
        )
    read["repeat"] = str(passes)
    per_turn = f"km_blocks{number}"
    state["written"] = (
        f"km_block{number} % {per_turn}",
        f"km_block{number} / {per_turn} == {passes - 1} && ",
    )


def sized_block(frame, node):
    """A stream's block= as (its pages, its word): a constant (word None), or a
    handler parameter given at each submit, whose most sizes the buffer."""
    sized = front.sized_parameters(frame.method.function)
    if isinstance(node, ast.Name) and node.id in sized:
        return sized[node.id], node.id
    return expressions.constant_int(frame, node), None


def stream_options(frame, node, call, write):
    """The stream's StreamOptions: its range, block, pattern C++, pattern letters
    or None, repeat C++, a read's tiles a row (`row=`), its block's word."""
    if len(call.args) != 1 or not ranges.is_range(frame, call.args[0]):
        refuse(
            frame.where,
            node,
            "a stream of a page range",
            "write km.Stream(tensor[part], block=8)",
        )
    pages = ranges.range_of(frame, call.args[0])
    block, pattern, letters, repeat = None, None, None, "1"
    row = 1
    word = None  # a block given at each submit: its parameter
    for keyword_ in call.keywords:
        key = keyword_.arg
        if key == spelling("block"):
            block, word = sized_block(frame, keyword_.value)
        elif key == spelling("row") and not write:
            row = expressions.constant_int(frame, keyword_.value)
        elif key == "pattern":
            pattern, letters = stream_pattern(frame, keyword_.value)
        elif key == spelling("repeat"):
            repeat = expressions.integer(frame, keyword_.value)
        elif key == spelling("write_stream"):
            continue
        elif key == spelling("prefetch") and not write:
            prefetch_checked(frame, node, keyword_.value)
        else:
            refuse(frame.where, node, f"a stream has no {key}=", "use block=, pattern=")
    block = row if block is None else block
    if block < 1 or block % row and row % block:
        refuse(
            frame.where,
            node,
            "a block of no pages, or not of whole rows nor a part of one",
            "give block= a count, a multiple of row= or a part of it",
        )
    if word is not None and (row > 1 or pages.rect is not None):
        refuse(
            frame.where,
            node,
            f"a block given at each submit ({word}) of rows of a "
            "rectangle or of row= is not built",
            "give block= a constant, or stream one run of pages",
        )
    if pattern is None and letters is None:  # unstated: the default
        letters = WRITE_PATTERN if write else READ_PATTERN
    if pages.rect is not None:
        columns = pages.rect[1]
        if isinstance(columns, str):
            refuse(
                frame.where,
                node,
                "a stream's blocks are whole rows of its rectangle, and this "
                "rectangle's columns are a word",
                f"stream {pages.tensor}.{spelling('tiles')}[r0:r1, c:c + n], n a "
                "constant, or one row of it",
            )
        if block % columns:
            refuse(
                frame.where,
                node,
                f"a block of a rectangle of {columns} columns is whole rows of it",
                f"give block= a multiple of {columns}",
            )
        pages.rect = (*pages.rect[:3], block)
    return StreamOptions(pages, block, pattern, letters, repeat, row, word)


def write_stream(frame, node, name, call):
    written = [k for k in call.keywords if k.arg == spelling("write_stream")]
    if not written or not (
        isinstance(written[0].value, ast.Constant) and written[0].value.value
    ):
        refuse(
            frame.where,
            node,
            "a stream to iterate goes in a for line",
            f"write `for tile in km.{spelling('stream')}(...)`, or "
            "write=True for a write stream",
        )
    options = stream_options(frame, node, call, True)
    pages, block, pattern, letters, repeat, _, word = options
    dtype, page_bytes = handler_buffers.tensor_format(frame, node, pages.tensor)
    page_bytes = handler_buffers.ring_page(frame, pages.tensor, page_bytes)
    if pages.items is not None:  # the buffer's pages are the items
        page_bytes = pages.items[0]
    handler_buffers.moved(frame, node, pages.tensor, "write stream")
    number = frame.writes
    frame.writes += 1
    made = len(frame.compiled.buffers)
    buffer = layout.pair(
        frame.compiled,
        "write",
        number,
        dtype,
        page_bytes,
        block,
        WRITE_ITEMS,
        f"line {node.lineno}: {line_text(frame.where, node)}",
    )
    block_word(frame, node, buffer, word)
    pattern_text = handler_buffers.pattern_cxx(frame, buffer, pattern, letters)
    variable = f"km_write{number}"
    frame.scope[name] = Value(
        "write",
        variable,
        count=pages.count,
        buffer=buffer,
        pattern=pattern_text,
        letters=letters or "",
    )
    frame.scope[name].items = pages.items and pages.items[0]  # an item's bytes
    text = handler_buffers.stream_cxx(
        frame, "write", variable, pages, buffer, pattern_text, repeat
    )
    uses = range_uses(pages, frame) | pattern_uses(pattern)
    wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_DATA)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    frame.scope[name].depth = len(frame.loops)
    fed = f"km_fed_{name}"  # compute's blocks of it, when a loop's bodies feed it
    line = Glue(text, {variable}, uses)
    stated = any(k.arg == "pattern" for k in call.keywords)
    # what an item copy (item_copy) rewrites: the line, the pages, the buffer made here
    frame.scope[name].copy = (
        line,
        pages,
        repeat,
        stated,
        len(frame.compiled.buffers) > made,
    )
    return [line], [Glue(f"uint32_t {fed} = 0;", {fed})]


# Default patterns, the reference's measured best on the whole grid for reads; a write
# of whole blocks goes by mover B's NoC 1, as the classic writer's.
READ_PATTERN = "AAAB"


WRITE_PATTERN = "B"


def held_most(pages):
    """The most pages of a range where the build knows it: a count known, or a
    rectangle's sides' most (Value.bounds), else None."""
    if pages.count.isdigit():
        return int(pages.count)
    bounds = getattr(pages, "bounds", None)
    if bounds and all(isinstance(side, int) for side in bounds):
        return bounds[0] * bounds[1]
    return None


def held_move(variable, half, buffer, count):
    """The movers' lines of a held stream: its one block moved, or with no pages an
    empty block pushed unread by mover A, block 0's (compute waits for one block a
    stream, as a hand reader pushes a job's Q block of no rows)."""
    return [
        Line(
            f"if (({count}) == 0) {{\n"
            f"    if (!movement.reserve({buffer}.mover_a, {buffer}Block)) {{\n"
            "        return false;\n"
            "    }\n"
            f"    cb_push_back({buffer}.mover_a, {buffer}Block);\n"
            f"}} else if (!movement.move({variable})) {{\n"
            "    return false;\n"
            "}",
            {variable},
            mover="A",
        ),
        Line(
            f"if (({count}) != 0 && !movement.move({variable})) {{\n"
            "    return false;\n"
            "}",
            {variable},
            mover="B",
        ),
    ]


def out_names(number, write, state):
    """The C++ names of a loop's output half and room for write stream `write`: one
    a loop, or the stream's own when results of one call go to several streams."""
    if write.cxx in state.get("own_out", ()):
        return f"{write.cxx}_out", f"{write.cxx}_room"
    return f"km_out{number}", f"km_room{number}"


def stated_prefetch(call):
    """A read stream's `prefetch=` as written (checked: prefetch_checked), else None."""
    for keyword_ in call.keywords:
        if keyword_.arg == spelling("prefetch"):
            return keyword_.value.value
    return None


def prefetch_checked(handler, node, value):
    """A read stream's `prefetch=`: True or False as written (the chain's walk reads
    it, komorebi.dsl.chain.chain_calls); anything else refused at its line."""
    if not (isinstance(value, ast.Constant) and isinstance(value.value, bool)):
        refuse(
            handler.where,
            node,
            f"{spelling('prefetch')}= is True or False as written",
            f"write {spelling('prefetch')}=False (read once the input is marked) or "
            "True (proven unwritten by the chain, or refused)",
        )


def stream_units(read):
    """What a read stream's blocks count: (its pages, its block) and, for a rectangle
    of constant columns, (its rows, its rows a block): streams in step share one."""
    pages = read["pages"]
    units = {(pages.count, read["block"])}
    sides = pages.sides
    if (
        sides is not None
        and isinstance(sides[1], int)
        and read["block"] % sides[1] == 0
    ):
        units.add((sides[0], read["block"] // sides[1]))
    return units
