"""A handler's tile products: where each operand stands (srca, srcb, turned), its
formats, and the product of a stream, a field, a held block, a broadcast or turned
tiles into DST or a field of sums."""

import ast
import math

from komorebi.dsl.compile import expressions, handler_buffers, scope, tile_lookup
from komorebi.dsl.compile.casts import broadcasts
from komorebi.dsl.compile.cxx import camel, text_uses
from komorebi.dsl.compile.dst.dst_code import (
    DST_HALF,
    sums_reserve,
    sums_room,
    sums_room_line,
)
from komorebi.dsl.compile.pages import field_fills, page_streams
from komorebi.dsl.compile.source import (
    WAIT_INPUT,
    function_line,
    line_text,
    refuse,
    source_of,
    wait,
)
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Line


def arrived_operand(frame, node, arg):
    """`w`, `kv[i]` or either `[:, c0:c0 + n]`, w a block a broadcast brought:
    (its Value, the part's slot geometry, the first column's C++, the columns or
    None for the part's); None when `arg` names no such block."""
    window = None
    if (
        isinstance(arg, ast.Subscript)
        and isinstance(arg.slice, ast.Tuple)
        and len(arg.slice.elts) == 2
    ):
        window, arg = arg.slice.elts, arg.value
    part = None
    if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
        part, arg = arg.slice, arg.value
    value = isinstance(arg, ast.Name) and frame.scope.get(arg.id)
    if not value or value.kind != "arrived":
        return None
    drawn = f"write {arg.id}[i][:, c0:c0 + n], i its part in the cast"
    if part is None and len(value.parts) > 1:
        refuse(
            frame.where,
            node,
            f"{arg.id} is a cast of {len(value.parts)} parts: a product reads one",
            drawn,
        )
    index = 0 if part is None else expressions.known_int(frame, part)
    if index is None or not 0 <= index < len(value.parts):
        refuse(
            frame.where,
            node,
            f"{arg.id} has parts 0 .. {len(value.parts) - 1}",
            drawn,
        )
    first, columns = "0", None
    if window is not None:
        rows, cols = window
        if not (
            isinstance(rows, ast.Slice)
            and rows.lower is rows.upper is rows.step is None
            and isinstance(cols, ast.Slice)
            and cols.step is None
            and cols.lower is not None
            and isinstance(cols.upper, ast.BinOp)
            and isinstance(cols.upper.op, ast.Add)
            and ast.dump(cols.upper.left) == ast.dump(cols.lower)
        ):
            refuse(frame.where, node, "a block's columns are a window c0:c0 + n", drawn)
        first = expressions.integer(frame, cols.lower)
        columns = expressions.known_int(frame, cols.upper.right)
        if columns is None:
            refuse(
                frame.where,
                node,
                "a window's width is an int the build knows",
                "write c0:c0 + n, n a constant",
            )
    return value, value.parts[index], first, columns, arg.id


def arrived_rows(frame, node, arg):
    """`w[i][r0:r0 + n, :]` (or `w[r0:r0 + n, :]`), w a block a broadcast brought:
    (its Value, the part's slot geometry, r0's C++, n); None when `arg` is not."""
    if not (
        isinstance(arg, ast.Subscript)
        and isinstance(arg.slice, ast.Tuple)
        and len(arg.slice.elts) == 2
    ):
        return None
    (rows, cols), inner = arg.slice.elts, arg.value
    part = None
    if isinstance(inner, ast.Subscript) and isinstance(inner.value, ast.Name):
        part, inner = inner.slice, inner.value
    value = isinstance(inner, ast.Name) and frame.scope.get(inner.id)
    if not value or value.kind != "arrived":
        return None
    drawn = f"write {inner.id}[i][r0:r0 + n, :], n the field's tiles"
    whole = (
        isinstance(cols, ast.Slice) and cols.lower is cols.upper is cols.step is None
    )
    sliced = isinstance(rows, ast.Slice) and rows.step is None
    known = (  # a:b, both ints the build knows
        sliced
        and None not in (rows.lower, rows.upper)
        and None
        not in (
            expressions.known_int(frame, rows.lower),
            expressions.known_int(frame, rows.upper),
        )
    )
    if not whole or not (
        known
        or sliced
        and rows.lower is not None
        and isinstance(rows.upper, ast.BinOp)
        and isinstance(rows.upper.op, ast.Add)
        and ast.dump(rows.upper.left) == ast.dump(rows.lower)
    ):
        refuse(
            frame.where,
            node,
            "a product's field on the left takes rows r0:r0 + n",
            drawn,
        )
    index = 0 if part is None else expressions.known_int(frame, part)
    if index is None or not 0 <= index < len(value.parts):
        refuse(
            frame.where,
            node,
            f"{inner.id} has parts 0 .. {len(value.parts) - 1}",
            drawn,
        )
    if known:
        low, high = expressions.known_int(frame, rows.lower), expressions.known_int(
            frame, rows.upper
        )
        return value, value.parts[index], str(low), high - low, inner.id
    count = expressions.known_int(frame, rows.upper.right)
    if count is None:
        refuse(frame.where, node, "the rows' count is an int the build knows", drawn)
    return (
        value,
        value.parts[index],
        expressions.integer(frame, rows.lower),
        count,
        inner.id,
    )


def broadcast_product(frame, node, value, function, item, arrived):
    """`acc += f(xb, w)` (or `w[i][:, c0:c0 + n]`), w a broadcast's block brought
    above the loop: tile t of block k by K row k x block + t of w's part (columns
    from c0, acc's width; transposed where the cast turned it) into acc's DST;
    w waited for before the loop's blocks, freed after them."""
    block_value, (offset, rows, pitch, transposed), first, columns, name = arrived
    state, number = frame.loop_states[-1], frame.stream_loops[-1]
    mover = frame.finished_casts.get(name)
    if mover is not None and any(
        letters is None or mover in letters for letters in state["letters"]
    ):  # compute frees the block after this loop: the cast's finish after it too
        handler_buffers.defer_finish(frame, node, name)
    width = value.width
    if columns is None:
        columns = pitch
    if columns != width:
        refuse(
            frame.where,
            node,
            f"the block's part is {columns or 'a word of'} columns and "
            f"{ast.unparse(node.target)} holds {width}",
            f"take a window of {width}: w[:, c0:c0 + {width}]",
        )
    if pitch is None:
        refuse(
            frame.where,
            node,
            "the block's rows are a word apart in its slot",
            "cast a rectangle whose columns the build knows",
        )
    if first.isdigit() and int(first) + width > pitch:
        refuse(
            frame.where,
            node,
            f"columns {first} .. {int(first) + width - 1} of a part of {pitch}",
            f"take a window inside the part's {pitch} columns",
        )
    if state["other"] is not None or state["function"] not in (None, "product"):
        refuse(
            frame.where,
            node,
            "this loop has other tile lines, and a product holds the matrix unit's "
            "setup for the loop",
            "keep the loop to the product",
        )
    buffer = block_value.buffer
    why = product_refusal(item.buffer.dtype, buffer.dtype)
    if why is not None:
        refuse(frame.where, node, why, "cast tiles of a matrix format of the stream's")
    fidelity = product_fidelity(frame, node, function)
    if placed(
        frame,
        node,
        state,
        function,
        item.buffer.dtype,
        buffer.dtype,
        ("tensor", block_value.tensor),
    ):
        refuse(
            frame.where,
            node,
            f"{function.name} turns the product, and a broadcast block's product "
            "is built plain",
            "drop the marker",
        )
    held = getattr(block_value, "held", False)  # waited at its cast instead
    if not held:
        broadcasts.arrived_once(frame, node, name, block_value)
    x, w = f"{item.buffer.name}.mover_a", buffer.name
    block = item.buffer.name + "Block"
    flip = "true" if transposed else "false"
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    if state["function"] is None:
        state["function"] = "product"
    received = (
        ""
        if held
        else f"km_ok = km_ok && km::broadcast_receive(compute, {w}, {w}Pages);\n"
    )
    state["inits"].append(
        Line(
            f"{received}"
            "// the zeros' fills (vector unit) done before the matrix unit adds\n"
            "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
            f"KM_UNPACK_FORMATS({w}, {x});\n"
            f"matmul_block_init({x}, {w}, {flip}, {width}, 1, {block});",
            {"km_ok"},
        )
    )
    if not held:
        state["after"].append(
            Line(f"if (km_ok) {{\n    km::broadcast_done({w}, {w}Pages);\n}}")
        )
    frame.compiled.includes.add("matmul")
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            width,
            block,
        )
        + (fidelity,)
    )
    row = f"(km_block{number} * {block} + {item.first})"
    text = (
        f"matmul_block({item.cxx}, {w}, {item.first}, {offset} + {row} * {pitch} "
        f"+ {first}, {value.dst}, {flip}, {width}, 1, {block});"
    )
    return [], [Line(text, text_uses(first, frame))]


def field_block_product(frame, node, value, function, field_, rows):
    """`acc += f(self.p, w[i][r0:r0 + n, :])`, self.p a row of n tiles compute
    wrote, w a block a broadcast brought (products take it: waited at its cast):
    tile k of the field by row r0 + k of w's part, acc's width a row, into acc's
    DST; outside a stream loop, any number a block."""
    block_value, (offset, _, pitch, transposed), first, count, name = rows
    width = value.width
    if frame.loop_states:
        refuse(
            frame.where,
            node,
            "a product of a field and a block runs on its own, and this stream "
            "loop holds the matrix unit's setup for its own products",
            "write it after the loop",
        )
    if field_ not in frame.packed:
        refuse(
            frame.where,
            node,
            f"self.{field_} is not written above in this handler: a product's left "
            "field is one compute writes",
            f"write self.{field_}.write(acc) before",
        )
    stored = frame.compiled.fields[field_]
    k = math.prod(stored.shape[:-2])
    if count != k:
        refuse(
            frame.where,
            node,
            f"self.{field_} is {k} tile(s) and the block's rows {count}",
            f"take rows r0:r0 + {k}",
        )
    if transposed or pitch != width:
        refuse(
            frame.where,
            node,
            f"the block's part is {pitch or 'a word of'} columns"
            f"{' (cast turned)' if transposed else ''} and "
            f"{ast.unparse(node.target)} holds {width}",
            f"cast a part of {width} columns, not turned",
        )
    buffer = block_value.buffer
    why = product_refusal(stored.format.dtype, buffer.dtype)
    if why is not None:
        refuse(
            frame.where,
            node,
            why,
            "declare the field in a matrix format of the cast's",
        )
    fidelity = product_fidelity(frame, node, function)
    if placed(
        frame,
        node,
        {},
        function,
        stored.format.dtype,
        buffer.dtype,
        ("tensor", block_value.tensor),
    ):
        refuse(
            frame.where,
            node,
            f"{function.name} turns the product, and a field's product on the left "
            "is built plain",
            "drop the marker",
        )
    frame.compiled.includes.add("matmul")
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            width,
            str(k),
        )
        + (fidelity,)
    )
    field_buffer, w = f"kField{camel(field_)}", buffer.name
    bound = ""
    if field_ not in frame.masked:  # a masked P is zero on the tail
        bound = window_inside(frame, block_value, rows[1], first, k, "rows")
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    calls = "\n".join(
        f"    matmul_block({field_buffer}, {w}, {t}, {offset} + ({first} + {t}) * "
        f"{pitch}, {value.dst}, 0, {width}, 1, {k});"
        for t in range(k)
    )
    text = (
        f"{bound}"
        f"km_ok = km_ok && compute.wait_front({field_buffer}, "
        f"{field_buffer}Pages);\n"
        "if (km_ok) {\n"
        "    // the sums' setup (vector unit) done before the matrix unit adds\n"
        "    MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
        f"    KM_UNPACK_FORMATS({w}, {field_buffer});\n"
        f"    matmul_block_init({field_buffer}, {w}, 0, {width}, 1, {k});\n"
        f"{calls}\n"
        "}"
    )
    uses = {"km_ok"} | text_uses(first, frame) | text_uses(bound, frame)
    return [], [Line(text, uses)]


def field_freed(frame, field, buffer):
    """A product's field freed once it is read: at the handler's end, or for a
    field filled each turn at the end of that turn, so the next fill finds room."""
    pop = f"cb_pop_front({buffer}, {buffer}Pages);"
    loop = fill_turn_loop(frame, field)
    if loop is None:
        frame.closing.append(Line(pop))
    else:
        freed = f"// self.{field}, this turn's fill, freed for the next\n{pop}"
        loop.setdefault("frees", []).append(Line(freed))


def field_left_product(frame, node, value, function, field_, item):
    """`acc += f(self.a, wb)`, self.a a row of K tiles compute wrote, wb a block
    of the loop's stream of K rows of acc's width: row k by tile k of the field,
    a row of products a step into acc's DST; the field waited for before the
    loop, freed by its next write (or the handler's end)."""
    state, number = frame.loop_states[-1], frame.stream_loops[-1]
    width = value.width
    if field_ not in frame.packed:
        refuse(
            frame.where,
            node,
            f"self.{field_} is not written above in this handler: a product's left "
            "field is one compute writes",
            f"write self.{field_}.write(acc) before this loop",
        )
    if state["other"] is not None or state["function"] not in (None, "product"):
        refuse(
            frame.where,
            node,
            "this loop has other tile lines, and a product holds the matrix unit's "
            "setup for the loop",
            "keep the loop to the product",
        )
    if len(frame.loop_states[-1]["tensors"]) != 1:
        refuse(
            frame.where,
            node,
            "a product with a field on the left reads one stream in its loop",
            f"write `for wb in km.{spelling('stream')}(w.tiles[0:K, 0:n], "
            "block=...)`",
        )
    stored = frame.compiled.fields[field_]
    k = math.prod(stored.shape[:-2])
    buffer, w = f"kField{camel(field_)}", item.buffer.name
    rows = state["count"]  # the loop's tiles, as written
    if (
        item.buffer.block % width
        or rows is not None
        and str(rows).isdigit()
        and int(rows) != k * width
    ):
        refuse(
            frame.where,
            node,
            f"self.{field_} is a row of {k} tile(s) and acc {width}: the stream "
            f"is {k} rows of {width} tiles, a block of whole rows",
            f"stream w.tiles[0:{k}, c:c + {width}] with a block of n x {width}",
        )
    why = product_refusal(stored.format.dtype, item.buffer.dtype)
    if why is not None:
        refuse(
            frame.where,
            node,
            why,
            "declare the field in a matrix format of the stream's",
        )
    fidelity = product_fidelity(frame, node, function)
    if placed(
        frame,
        node,
        state,
        function,
        stored.format.dtype,
        item.buffer.dtype,
        ("tensor", state["tensors"][w]),
    ):
        refuse(
            frame.where,
            node,
            f"{function.name} turns the product, and a field's product on the left "
            "is built plain",
            "drop the marker",
        )
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    state["function"], state["step"] = "product", width
    state["inits"].append(
        Line(
            f"km_ok = km_ok && compute.wait_front({buffer}, {buffer}Pages);\n"
            "// the zeros' fills (vector unit) done before the matrix unit adds\n"
            "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
            f"KM_UNPACK_FORMATS({w}.mover_a, {buffer});\n"
            f"matmul_block_init({buffer}, {w}.mover_a, 0, {width}, 1, {k});",
            {"km_ok"},
        )
    )
    frame.compiled.includes.add("matmul")
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            width,
            str(k),
        )
        + (fidelity,)
    )
    row = f"(km_block{number} * {w}Block + {item.first}) / {width}"
    text = (
        f"matmul_block({buffer}, {item.cxx}, {row}, {item.first}, {value.dst}, 0, "
        f"{width}, 1, {k});"
    )
    return [], [Line(text)]


def held_left_product(frame, node, value, function, arrived, item):
    """`acc += f(xs, wb)`, xs a row of K tiles a broadcast brought (one part, not
    turned), wb a block of the loop's one stream of K rows of acc's width: as
    field_left_product, xs waited for before the loop's blocks and freed after
    them; the cast's finish after the loop when its mover moves the stream."""
    block_value, (offset, rows, pitch, transposed), first, columns, name = arrived
    state, number = frame.loop_states[-1], frame.stream_loops[-1]
    width = value.width
    mover = frame.finished_casts.get(name)
    if mover is not None and any(
        letters is None or mover in letters for letters in state["letters"]
    ):  # compute frees the block after this loop: the cast's finish after it too
        handler_buffers.defer_finish(frame, node, name)
    drawn = f"cast a row: {name} = b.cast(x.tiles[0:1, k0:k1].read())"
    if transposed or rows != 1 or columns is not None or first != "0":
        refuse(
            frame.where,
            node,
            f"{name} on a product's left is a row of K tiles a broadcast brought "
            "(one row, not turned, the whole part)",
            drawn,
        )
    if pitch is None:
        refuse(
            frame.where,
            node,
            f"{name}'s row has a count of tiles the build does not know",
            "cast a rectangle whose columns the build knows",
        )
    if state["other"] is not None or state["function"] not in (None, "product"):
        refuse(
            frame.where,
            node,
            "this loop has other tile lines, and a product holds the matrix unit's "
            "setup for the loop",
            "keep the loop to the product",
        )
    if len(state["tensors"]) != 1:
        refuse(
            frame.where,
            node,
            f"a product with {name} on the left reads one stream in its loop",
            f"write `for wb in km.{spelling('stream')}(w.tiles[0:K, 0:n], "
            "block=...)`",
        )
    k, w = pitch, item.buffer.name
    count = state["count"]  # the loop's tiles, as written
    if (
        item.buffer.block % width
        or count is not None
        and str(count).isdigit()
        and int(count) != k * width
    ):
        refuse(
            frame.where,
            node,
            f"{name} is a row of {k} tile(s) and acc {width}: the stream is {k} "
            f"rows of {width} tiles, a block of whole rows",
            f"stream w.tiles[0:{k}, c:c + {width}] with a block of n x {width}",
        )
    buffer = block_value.buffer
    why = product_refusal(buffer.dtype, item.buffer.dtype)
    if why is not None:
        refuse(frame.where, node, why, "cast tiles of a matrix format of the stream's")
    fidelity = product_fidelity(frame, node, function)
    if placed(
        frame,
        node,
        state,
        function,
        buffer.dtype,
        item.buffer.dtype,
        ("tensor", state["tensors"][w]),
    ):
        refuse(
            frame.where,
            node,
            f"{function.name} turns the product, and a broadcast block's product on "
            "the left is built plain",
            "drop the marker",
        )
    held = getattr(block_value, "held", False)  # waited at its cast instead
    if not held:
        broadcasts.arrived_once(frame, node, name, block_value)
    x = buffer.name
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    state["function"], state["step"] = "product", width
    received = (
        ""
        if held
        else f"km_ok = km_ok && km::broadcast_receive(compute, {x}, {x}Pages);\n"
    )
    state["inits"].append(
        Line(
            f"{received}"
            "// the zeros' fills (vector unit) done before the matrix unit adds\n"
            "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
            f"KM_UNPACK_FORMATS({w}.mover_a, {x});\n"
            f"matmul_block_init({x}, {w}.mover_a, 0, {width}, 1, {k});",
            {"km_ok"},
        )
    )
    if not held:
        state["after"].append(
            Line(f"if (km_ok) {{\n    km::broadcast_done({x}, {x}Pages);\n}}")
        )
    frame.compiled.includes.add("matmul")
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            width,
            str(k),
        )
        + (fidelity,)
    )
    row = f"(km_block{number} * {w}Block + {item.first}) / {width}"
    text = (
        f"matmul_block({x}, {item.cxx}, {offset} + {row}, {item.first}, "
        f"{value.dst}, 0, {width}, 1, {k});"
    )
    return [], [Line(text)]


def field_product(frame, node, field_, call):
    """`self.part += f(a, b)`, f `return a @ b`, a and b blocks two broadcasts
    brought (oh x kc and kc x ow tiles), self.part an fp32 field of oh x ow tiles:
    the library's product by subblocks of DST, each packed into the field with
    the packer's L1 accumulation (overwriting at the first sum after a write)."""
    function = tile_lookup.tile_function_of(frame, call)
    where = function_line(function)
    if not is_product(function.function):
        refuse(
            frame.where,
            node,
            f"self.{field_} += f(...) adds a product, and the tile function "
            f"{function.name} ({where}) is not `return a @ b`",
            f"write {function.name} as `return a @ b`",
        )
    no_expressions(frame, node, call)
    items = [isinstance(arg, ast.Name) and frame.scope.get(arg.id) for arg in call.args]
    if (
        len(items) == 2
        and not call.keywords
        and all(item and in_this_loop(frame, item) for item in items)
    ):
        return stream_field_product(frame, node, field_, function, *items)
    blocks, casts = [], []
    for arg in call.args:
        item = isinstance(arg, ast.Name) and frame.scope.get(arg.id)
        if not item or item.kind != "arrived":
            refuse(
                frame.where,
                node,
                f"`{ast.unparse(arg)}` is not a block a broadcast brought: a sum "
                "in a field adds products of two broadcasts' blocks",
                "write `if b.sender(): v = b.cast(t.tiles[...].read())`, `else: "
                "v = b.arrive()` for each operand",
            )
        broadcasts.arrived_loops(frame, node, arg.id, item)
        blocks.append(item.buffer)
        casts.append(item.tensor)
    if len(blocks) != 2 or call.keywords:
        refuse(frame.where, node, "a product of two blocks", "write self.f += g(a, b)")
    stored = frame.compiled.fields[field_]
    if stored.format.dtype != "float32" or len(stored.shape) != 4:
        refuse(
            frame.where,
            node,
            f"self.{field_} holds sums: fp32 tiles, rows x columns of them",
            f"declare self.{field_}: km.L1[km.fp32, rows, columns, 32, 32]",
        )
    why = product_refusal(blocks[0].dtype, blocks[1].dtype)
    if why is not None:
        refuse(frame.where, node, why, "cast tiles of matrix formats of one family")
    left, right = blocks[0].dtype, blocks[1].dtype
    turned = placed(frame, node, {}, function, left, right, ("tensor", casts[1]))
    rows, columns = expressions.field_sides(frame, node, field_)
    a, b = blocks
    inner = a.block // rows if rows and a.block % rows == 0 else 0
    if inner == 0 or b.block != inner * columns:
        refuse(
            frame.where,
            node,
            f"self.{field_} is {rows} x {columns} tiles; the blocks are {a.block} "
            f"and {b.block} tiles, not {rows} x k and k x {columns}",
            "cast blocks of the field's rows and columns",
        )
    fidelity = function.fidelity or "HiFi4"
    if frame.compiled.fidelity not in (None, fidelity):
        refuse(
            frame.where,
            node,
            f"{function.name}'s fidelity is {fidelity} and another product's "
            f"{frame.compiled.fidelity}: the fidelity is the compute kernel's",
            "give the class's products one fidelity",
        )
    frame.compiled.fidelity = fidelity
    width = max(w for w in range(1, DST_HALF + 1) if columns % w == 0)
    height = max(
        h for h in range(1, DST_HALF // width + 1) if rows % h == 0
    )  # a subblock: height x width tiles of DST
    buffer = f"kField{camel(field_)}"
    row, column, k = (f"km_{n}{node.lineno}" for n in ("sub_row", "sub_column", "k"))
    field_fills.producer(frame, node, field_, "product")
    sums = frame.summed.get(field_)
    if sums is None:
        sums = f"km_sums_{field_}"
        frame.summed[field_] = sums
        frame.compiled.summed_fields.add(f"kField{camel(field_)}")
        frame.opening.append(Line(f"uint32_t {sums} = 0;  // self.{field_}'s"))
        frame.opening.append(Line(sums_room_line(field_)))
        frame.opening_b.append(  # mover B counts the same sums: its write's
            Line(f"uint32_t {sums} = 0;  // self.{field_}'s", mover="B")
        )
    frame.compiled.includes |= {"matmul", "pack"}
    frame.compiled.unpack_fp32.add(buffer)
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            rows * columns,
            inner,
            fidelity,
        )
    )
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    init = (
        f"    KM_UNPACK_FORMATS({b.name}, {a.name});\n"
        f"    matmul_block_init({a.name}, {b.name}, 0, {width}, {height}, "
        f"{inner});\n"
    )
    calls = (
        f"                matmul_block({a.name}, {b.name}, {row} * {inner} + {k}, "
        f"{k} * {columns} + {column}, 0, 0, {width}, {height}, {inner});\n"
    )
    turn = ""
    if turned:
        init, calls, turn = turned_blocks(
            frame, a, b, (row, column, k), inner, columns, (height, width)
        )
    room = sums_room(field_)
    packs = (
        f"            if ({room}) {{\n"
        + "\n".join(
            f"                pack_tile<true>({i * width + j}, {buffer}, ({row} + "
            f"{i}) * {columns} + {column} + {j});"
            for i in range(height)
            for j in range(width)
        )
        + "\n            }"
    )
    text = (
        "km_ok = km_ok && km::broadcast_receive(compute, "
        f"{a.name}, {a.name}Pages) &&\n"
        f"    km::broadcast_receive(compute, {b.name}, {b.name}Pages);\n"
        f"{sums_reserve(sums, room, buffer)}"
        "if (km_ok) {\n"
        f"{init}"
        f"    KM_PACK_FORMAT({buffer});\n"
        f"    if ({room}) {{\n"
        f"        pack_reconfig_l1_acc({sums} > 0 ? 1 : 0);\n"
        "    }\n"
        f"    for (uint32_t {row} = 0; {row} < {rows}; {row} += {height}) {{\n"
        f"        for (uint32_t {column} = 0; {column} < {columns}; {column} += "
        f"{width}) {{\n"
        "            tile_regs_acquire();\n"
        f"            for (uint32_t {k} = 0; {k} < {inner}; {k}++) {{\n"
        f"{calls}"
        "            }\n"
        f"{turn}"
        "            tile_regs_commit();\n"
        "            tile_regs_wait();\n"
        f"{packs}\n"
        "            tile_regs_release();\n"
        "        }\n"
        "    }\n"
        f"    if ({room}) {{\n"
        "        pack_reconfig_l1_acc(0);\n"
        "    }\n"
        f"    {sums}++;\n"
        f"    km::broadcast_done({a.name}, {a.name}Pages);\n"
        f"    km::broadcast_done({b.name}, {b.name}Pages);\n"
        "}"
    )
    return [Line(f"{sums}++;", mover="B")], [Line(text, {"km_ok"})]


def fill_turn_loop(frame, field):
    """The range loop around a product of `field` in which the field is filled
    each turn (its fill a line of that loop, the loop still open, the field read
    by this handler alone), or None."""
    loop = frame.fill_loop_of.get(field)
    if loop is None or field_fills.kept_field(frame, field):
        return None
    return loop if any(open_ is loop for open_ in frame.loops) else None


def held_block_product(frame, node, value, function, held, arrived):
    """`acc += f(qs[r], w)` (or `w[i][:, c0:c0 + n]`), qs a held stream (its
    rows of k tiles) and w a block a broadcast brought that products take: tile
    j of qs's row r by row j of w's part (acc's width of columns, transposed
    where the cast turned it) into acc's DST, outside a stream loop."""
    stream, row = held
    block_value, (offset, _, pitch, transposed), first, columns, name = arrived
    width = value.width
    if frame.loop_states:
        refuse(
            frame.where,
            node,
            "a product of a held stream's row runs on its own, and this stream "
            "loop holds the matrix unit's setup for its own products",
            "write it after the loop",
        )
    k = stream.columns
    if k is None:
        refuse(
            frame.where,
            node,
            f"{ast.unparse(node.value.args[0])}'s stream has rows of tiles the "
            "build does not know",
            "stream a rectangle whose columns are a constant: t.tiles[r:..., 0:k]",
        )
    if columns is None:
        columns = pitch
    if columns != width or pitch is None:
        refuse(
            frame.where,
            node,
            f"the block's part is {columns or 'a word of'} columns and "
            f"{ast.unparse(node.target)} holds {width}",
            f"take a window of {width}: w[i][:, c0:c0 + {width}]",
        )
    if not getattr(block_value, "held", False):
        refuse(
            frame.where,
            node,
            f"{name} is not waited for at its cast",
            "cast it in the lines around this product",
        )
    buffer = block_value.buffer
    left = stream.buffer.dtype
    why = product_refusal(left, buffer.dtype)
    if why is not None:
        refuse(frame.where, node, why, "cast tiles of a matrix format of the stream's")
    fidelity = product_fidelity(frame, node, function)
    if placed(
        frame, node, {}, function, left, buffer.dtype, ("tensor", block_value.tensor)
    ):
        refuse(
            frame.where,
            node,
            f"{function.name} turns the product, and a held stream's product is "
            "built plain",
            "drop the marker",
        )
    most = stream.buffer.block // k
    line = expressions.integer(frame, row)
    bound = (
        ""
        if line.rstrip("u").isdigit()
        else (
            f"if (!(({line}) < {most})) {{  // the held block's rows\n"
            "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
            "    return;\n}\n"
        )
    )
    if line.rstrip("u").isdigit() and not int(line.rstrip("u")) < most:
        refuse(
            frame.where,
            node,
            f"the held block has {most} rows, 0 .. {most - 1}",
            f"index it below {most}",
        )
    frame.compiled.includes.add("matmul")
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            width,
            str(k),
        )
        + (fidelity,)
    )
    if getattr(value, "row_of", None) is None:  # a mask's -inf covers the tail
        part = arrived[1]
        bound += window_inside(frame, block_value, part, first, width, "columns")
    q, w = stream.cxx, buffer.name
    flip = "true" if transposed else "false"
    calls = "\n".join(
        f"    matmul_block({q}, {w}, ({line}) * {k} + {j}, {offset} + {j} * "
        f"{pitch} + {first}, {value.dst}, {flip}, {width}, 1, {k});"
        for j in range(k)
    )
    text = (
        f"{bound}"
        "if (km_ok) {\n"
        "    // the sums' setup (vector unit) done before the matrix unit adds\n"
        "    MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
        f"    KM_UNPACK_FORMATS({w}, {q});\n"
        f"    matmul_block_init({q}, {w}, {flip}, {width}, 1, {k});\n"
        f"{calls}\n"
        "}"
    )
    uses = {"km_ok"} | expressions.uses(frame, row) | text_uses(first, frame)
    return [], [Line(text, uses | text_uses(bound, frame))]


def in_this_loop(frame, item):
    """Whether `item` is a block of the innermost stream loop's streams."""
    return (
        item.kind == "streamed"
        and bool(frame.stream_loops)
        and item.first == f"km_tile{frame.stream_loops[-1]}"
        and item.buffer is not None
    )


def k_bound(frame, node, state, field_, rows, block):
    """A product's K tiles (the loop's stream, its repeats) against the field's K
    rows: refused at build when both are numbers; else one compare a command
    before the loop, a give-up (kGaveBadItem) in place of a read past the field."""
    count, repeat = state["count"], state["repeat"]
    if count.isdigit() and repeat.isdigit():
        size = state["block"]
        tiles = (int(count) + size - 1) // size * (int(repeat) - 1) * size
        tiles += int(count)
        if tiles > rows:
            refuse(
                frame.where,
                node,
                f"self.{field_} is {rows} K rows, and the loop's stream (line "
                f"{state['line']}) is {tiles} K tiles: a tile of the stream "
                f"takes a row of self.{field_}",
                f"stream at most {rows} tiles, or declare self.{field_} with "
                f"{tiles} rows",
            )
        return []
    tiles = count
    if repeat != "1":
        tiles = (
            f"({count} + {block} - 1) / {block} * ({repeat} - 1) * {block} + "
            f"{count}"
        )
    uses = text_uses(count, frame) | text_uses(repeat, frame)
    return [
        Line(
            f"// the stream's K tiles against self.{field_}'s {rows} rows\n"
            f"if ({tiles} > {rows}) {{\n"
            "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
            "    return;\n}",
            uses,
        )
    ]


def placed(frame, node, state, function, left, right, weight):
    """Whether the product is turned (product_placement): refused when both
    operands are marked for one side, or the loop's other product is placed the
    other way; noted for the build's report of both placements."""
    placement = product_placement(function.function)
    if placement not in ("none", "plain", "turned"):
        refuse(
            frame.where,
            node,
            f"{function.name}: {placement}",
            f"mark one operand: `return a @ km.{spelling('srcb')}(b)`",
        )
    frame.compiled.placements.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            left,
            right,
            placement,
            weight,
        )
    )
    turned = placement == "turned"
    if state.setdefault("turned", turned) != turned:
        refuse(
            frame.where,
            node,
            "this loop's products are placed two ways, and a loop holds the "
            "matrix unit's setup for one",
            "mark both products' operands alike",
        )
    return turned


def no_expressions(frame, node, call):
    """Refuses an operand of a product that is computed on the line (`xb + yb`,
    `f(xb, yb)`): the matrix unit unpacks each operand from a buffer of its tiles."""
    for arg in call.args:
        if isinstance(arg, ast.BinOp | ast.UnaryOp | ast.Call | ast.Compare):
            refuse(
                frame.where,
                node,
                f"a product's operand is a block (a stream's, a held or a "
                f"broadcast's) or a field, and `{ast.unparse(arg)}` is computed on "
                "the line: the matrix unit reads each operand from a buffer of its "
                "tiles (a staged operand is not built)",
                "write it to a tensor and stream that (one more write and read of "
                "it in DRAM)",
            )


def product(frame, node, value, call):
    """`acc += f(block, self.w)`, f a tile function `return a @ b`: the library's
    product (matmul_block) of each tile of the loop's block by the field's row of
    that K tile (acc's width of tiles) added into acc's DST tiles; the field (K
    rows, K-major) held from its fill to the handler's end."""
    function = tile_lookup.tile_function_of(frame, call)
    where = function_line(function)
    if not is_product(function.function):
        refuse(
            frame.where,
            node,
            f"acc += f(...) adds a product, and the tile function {function.name} "
            f"({where}) is not `return a @ b`",
            f"write {function.name} as `return a @ b`, a the stream's block, b a "
            "field",
        )
    no_expressions(frame, node, call)
    held = len(call.args) == 2 and page_streams.held_row(frame, call.args[0])
    arrived = held and arrived_operand(frame, node, call.args[1])
    if arrived:
        return held_block_product(frame, node, value, function, held, arrived)
    left = len(call.args) == 2 and scope.field_name(frame, call.args[0])
    rows = left and arrived_rows(frame, node, call.args[1])
    if rows:
        return field_block_product(frame, node, value, function, left, rows)
    if not frame.loop_states or len(call.args) != 2 or call.keywords:
        refuse(
            frame.where,
            node,
            "a product takes a block of the loop's stream and a field",
            f"write acc += self.{function.name}(block, self.w) in "
            f"`for block in km.{spelling('stream')}(...)`",
        )
    state = frame.loop_states[-1]
    item = isinstance(call.args[0], ast.Name) and frame.scope.get(call.args[0].id)
    left = scope.field_name(frame, call.args[0])
    right = isinstance(call.args[1], ast.Name) and frame.scope.get(call.args[1].id)
    if left is not None and right and in_this_loop(frame, right):
        return field_left_product(frame, node, value, function, left, right)
    field_ = scope.field_name(frame, call.args[1])
    other = isinstance(call.args[1], ast.Name) and frame.scope.get(call.args[1].id)
    if item and other and in_this_loop(frame, item) and in_this_loop(frame, other):
        return stream_product(frame, node, value, function, item, other)
    arrived = arrived_operand(frame, node, call.args[1])
    if item and in_this_loop(frame, item) and arrived is not None:
        return broadcast_product(frame, node, value, function, item, arrived)
    left_block = arrived_operand(frame, node, call.args[0])
    if left_block is not None and other and in_this_loop(frame, other):
        return held_left_product(frame, node, value, function, left_block, other)
    if not item or item.kind != "streamed" or field_ is None:
        refuse(
            frame.where,
            node,
            f"`{ast.unparse(call.args[0])}` is not the loop's tile or "
            f"`{ast.unparse(call.args[1])}` is not a field",
            f"write acc += self.{function.name}(block, self.w)",
        )
    stored = frame.compiled.fields[field_]
    why = product_refusal(item.buffer.dtype, stored.format.dtype)
    if why is not None:
        refuse(
            frame.where,
            node,
            why,
            "stream a tensor of a matrix format and declare the field in one of "
            "its family",
        )
    columns = stored.shape[1] if len(stored.shape) == 4 else None
    if columns != value.width:
        target = ast.unparse(node.target)
        refuse(
            frame.where,
            node,
            f"self.{field_} is K rows of {columns} tiles and {target} holds "
            f"{value.width}",
            f"declare self.{field_}: km.L1[km.bf16, K, {value.width}, 32, 32] or "
            f"make {target} km.{spelling('tile')}.{spelling('zeros')}"
            f"(km.fp32, {columns})",
        )
    if state["other"] is not None or state["function"] not in (None, "product"):
        refuse(
            frame.where,
            node,
            "this loop has other tile lines, and a product holds the matrix unit's "
            "setup for the loop",
            "keep the loop to the product",
        )
    fillers = frame.compiled.field_users.get(field_, ([], []))[0]
    if (
        frame.filled.get(field_) is None
        and field_ not in frame.waited_fields
        and not (fillers and frame.method.name not in fillers)
    ):
        refuse(
            frame.where,
            node,
            f"self.{field_} is not filled above in this handler, and no other "
            "handler of the class fills it",
            f"write self.{field_}.fill(km.{spelling('stream')}(...)) before, or in "
            "a handler of its own",
        )
    filled_in = frame.fill_loops.get(field_, (0, 0))
    if filled_in[0] > 0 and fill_turn_loop(frame, field_) is None:
        refuse(
            frame.where,
            node,
            f"self.{field_} is filled in a loop (line {filled_in[1]}) and a "
            "product frees it once, at the handler's end: the next turn's fill "
            "would wait for room that is never freed",
            f"fill it before the loops, once (self.{field_}.fill(km."
            f"{spelling('stream')}(...))), or take it with a body in the loop "
            "that fills it, which frees it each turn",
        )
    fidelity = function.fidelity or "HiFi4"
    if frame.compiled.fidelity not in (None, fidelity):
        refuse(
            frame.where,
            node,
            f"{function.name}'s fidelity is {fidelity} and another product's "
            f"{frame.compiled.fidelity}: the fidelity is the compute kernel's",
            "give the class's products one fidelity",
        )
    frame.compiled.fidelity = fidelity
    weight = ("field", field_)
    left, right = item.buffer.dtype, stored.format.dtype
    turned = placed(frame, node, state, function, left, right, weight)
    number = frame.stream_loops[-1]
    buffer = f"kField{camel(field_)}"
    block = item.buffer.name + "Block"
    if turned:
        return turned_product(frame, node, value, function, item, field_, columns)
    if state["function"] is None:
        state["function"] = "product"
        waited = ""
        if field_ not in frame.waited_fields:  # once a handler
            frame.waited_fields.add(field_)
            wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
            waited = (
                f"if (!compute.wait_front({buffer}, {buffer}Pages)) {{\n"
                "    return;\n}\n"
            )
            if not field_fills.kept_field(frame, field_):
                field_freed(frame, field_, buffer)
        state["inits"].append(
            Line(
                f"{waited}"
                "// the zeros' fills (vector unit) done before the matrix unit "
                "adds\n"
                "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
                f"KM_UNPACK_FORMATS({buffer}, {item.buffer.name}.mover_a);\n"
                f"matmul_block_init({item.buffer.name}.mover_a, {buffer}, 0, "
                f"{columns}, 1, {block});"
            )
        )
    state["inits"] += k_bound(frame, node, state, field_, stored.shape[0], block)
    frame.compiled.includes.add("matmul")
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            columns,
            block,
            fidelity,
        )
    )
    text = (
        f"matmul_block({item.cxx}, {buffer}, {item.first}, (km_block{number} * "
        f"{block} + {item.first}) * {columns}, {value.dst}, 0, {columns}, 1, "
        f"{block});"
    )
    return [], [Line(text)]


def product_fidelity(frame, node, function):
    """The product's fidelity, one for the class's products (the compute
    kernel's)."""
    fidelity = function.fidelity or "HiFi4"
    if frame.compiled.fidelity not in (None, fidelity):
        refuse(
            frame.where,
            node,
            f"{function.name}'s fidelity is {fidelity} and another product's "
            f"{frame.compiled.fidelity}: the fidelity is the compute kernel's",
            "give the class's products one fidelity",
        )
    frame.compiled.fidelity = fidelity
    return fidelity


def stream_field_product(frame, node, field_, function, a, b):
    """`self.part += f(xb, wb)` in a zip of two streams, self.part an fp32 field of
    1 x n tiles: each block's product (its valid K tiles) by subblocks of DST,
    packed into the field with the packer's L1 accumulation; each W block used
    as it arrives and freed. The field is as wide as L1 holds, not as DST."""
    sums = sums_field(frame, node, field_)
    rows, columns = expressions.field_sides(frame, node, field_)
    if rows != 1:
        refuse(
            frame.where,
            node,
            f"self.{field_} is {rows} rows of tiles; a stream's block is one row "
            "of X's tiles",
            f"declare self.{field_}: km.L1[km.fp32, 1, n, 32, 32]",
        )
    stream_operands(frame, node, function, a, b, columns, f"self.{field_}")
    state = frame.loop_states[-1]
    state["whole"] = True  # the product takes the block, not its tiles
    # mover B counts the loop's blocks into each field the loop adds to
    state.setdefault("sums", []).append(sums)
    buffer = f"kField{camel(field_)}"
    inner = a.buffer.block
    width = max(w for w in range(1, DST_HALF + 1) if columns % w == 0)
    valid = f"km_valid{frame.stream_loops[-1]}"
    column, k = (f"km_{n}{node.lineno}" for n in ("sub_column", "k"))
    weight = ("tensor", state["tensors"][b.buffer.name])
    left, right = a.buffer.dtype, b.buffer.dtype
    turned = placed(frame, node, state, function, left, right, weight)
    call = (
        f"matmul_block({a.cxx}, {b.cxx}, {k}, {k} * {columns} + {column}, 0, 0, "
        f"{width}, 1, {inner});"
    )
    init = (
        f"{operand_formats(frame, b, a)}\n"
        f"matmul_block_init({a.buffer.name}.mover_a, {b.buffer.name}.mover_a, 0, "
        f"{width}, 1, {inner});"
    )
    turn = ""
    if turned:
        plain_tf32(frame, node, a, b)
        call, init, turn = turned_sums(
            frame, a, b, f"{k} * {columns} + {column}", k, width
        )
    if state["function"] is None:
        state["function"] = "product"
        state["product_init"] = init
        state["inits"].append(Line(f"{init}\nKM_PACK_FORMAT({buffer});"))
    own, restore = product_setups(state.get("product_init"), init)
    frame.compiled.includes |= {"pack"}
    frame.compiled.unpack_fp32.add(buffer)
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            columns,
            inner,
            frame.compiled.fidelity,
        )
    )
    room = sums_room(field_)
    packs = (
        f"        if ({room}) {{\n"
        + "\n".join(
            f"            pack_tile<true>({j}, {buffer}, {column} + {j});"
            for j in range(width)
        )
        + "\n        }"
    )
    text = (
        f"{sums_reserve(sums, room, buffer)}"
        "if (km_ok) {\n"
        f"{own}"
        f"    if ({room}) {{\n"
        f"        pack_reconfig_l1_acc({sums} > 0 ? 1 : 0);\n"
        "    }\n"
        f"    for (uint32_t {column} = 0; {column} < {columns}; {column} += "
        f"{width}) {{\n"
        "        tile_regs_acquire();\n"
        f"        for (uint32_t {k} = 0; {k} < {valid}; {k}++) {{\n"
        f"            {call}\n"
        "        }\n"
        f"{turn}"
        "        tile_regs_commit();\n"
        "        tile_regs_wait();\n"
        f"{packs}\n"
        "        tile_regs_release();\n"
        "    }\n"
        f"    if ({room}) {{\n"
        "        pack_reconfig_l1_acc(0);\n"
        "    }\n"
        f"{restore}"
        f"    {sums}++;\n"
        "}"
    )
    return [], [Line(text, {"km_ok", valid})]


def operand_formats(frame, srca, srcb):
    """The sources' formats before a product of two streams' blocks (`srca` into
    SrcA, `srcb` into SrcB): the library's from the buffers, or, with a float32
    side, that side read as tf32 (km_tf32.hpp), every later unpack then setting
    its own again (mixed_formats)."""
    sides = [item.buffer.dtype == "float32" for item in (srca, srcb)]
    a, b = (f"{item.buffer.name}.mover_a" for item in (srca, srcb))
    if not any(sides):
        return f"KM_UNPACK_FORMATS({a}, {b});"
    frame.compiled.includes.add("tf32")
    flags = ", ".join("true" if side else "false" for side in sides)
    return f"km::tf32_formats<{flags}>({a}, {b});"


def plain_tf32(frame, node, a, b):
    """Refuses a float32 operand of a turned product: its tf32 setup is the plain
    product's (operand_formats)."""
    for side, item in (("left", a), ("right", b)):
        if item.buffer.dtype == "float32":
            refuse(
                frame.where,
                node,
                f"the product's {side} operand is float32 and the product is "
                "turned: a float32 operand is read as tf32 in a plain product only",
                "drop the marker, or stream a bfloat16 copy of the tensor",
            )


def stream_operands(frame, node, function, a, b, columns, target):
    """Two streams' blocks as a product's operands: bfloat16, b's block k rows
    of `columns` for each of a's k tiles (a: one row of tiles a block)."""
    why = product_refusal(a.buffer.dtype, b.buffer.dtype, tf32=True)
    if why is not None:
        refuse(frame.where, node, why, "stream tensors of matrix formats of one family")
    if a.width != 1 or b.buffer.block != a.buffer.block * columns:
        refuse(
            frame.where,
            node,
            f"{target} is {columns} tiles wide: each block of the second stream "
            f"is the first's {a.buffer.block} K tiles by {columns} columns "
            f"({a.buffer.block * columns} tiles), and it is {b.buffer.block}",
            f"stream w.{spelling('tiles')}[k0:k1, c:c + {columns}] with "
            f"{spelling('block')}={a.buffer.block * columns} beside x's "
            f"{spelling('block')}={a.buffer.block}",
        )
    state = frame.loop_states[-1]
    if a.cxx != state["first"] and a.buffer.block != state["block"]:
        refuse(
            frame.where,
            node,
            "the loop counts the blocks of its stream of the smallest block, and "
            "the product's left operand is not of that block",
            "zip x's stream (rows of k tiles) beside w's (k rows of n)",
        )
    if state["other"] is not None or state["function"] not in (None, "product"):
        refuse(
            frame.where,
            node,
            "this loop has other tile lines, and a product holds the matrix unit's "
            "setup for the loop",
            "keep the loop to the product",
        )
    product_fidelity(frame, node, function)
    frame.compiled.includes.add("matmul")


def stream_product(frame, node, value, function, a, b):
    """`acc += f(xb, wb)` in `for xb, wb in zip(km.Stream(x..), km.Stream(w..))`:
    each tile k of xb by row k of wb (acc's width of tiles) added into acc's DST
    tiles by the library's product; each W block used as it arrives and freed."""
    target = ast.unparse(node.target)
    stream_operands(frame, node, function, a, b, value.width, target)
    state = frame.loop_states[-1]
    block = f"{a.buffer.name}Block"
    weight = ("tensor", state["tensors"][b.buffer.name])
    left, right = a.buffer.dtype, b.buffer.dtype
    if placed(frame, node, state, function, left, right, weight):
        plain_tf32(frame, node, a, b)
        return turned_stream_product(frame, node, value, function, a, b)
    if state["function"] is None:
        state["function"] = "product"
        state["inits"].append(
            Line(
                "// the zeros' fills (vector unit) done before the matrix unit "
                "adds\n"
                "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));\n"
                f"{operand_formats(frame, b, a)}\n"
                f"matmul_block_init({a.buffer.name}.mover_a, {b.buffer.name}"
                f".mover_a, 0, {value.width}, 1, {block});"
            )
        )
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            value.width,
            block,
            frame.compiled.fidelity,
        )
    )
    text = (
        f"matmul_block({a.cxx}, {b.cxx}, {a.first}, {a.first} * {value.width}, "
        f"{value.dst}, 0, {value.width}, 1, {block});"
    )
    return [], [Line(text)]


def sums_field(frame, node, field_):
    """A field of sums: fp32, rows x columns of tiles; its count of sums (a
    name of compute's, 0: the next sum overwrites)."""
    stored = frame.compiled.fields[field_]
    if stored.format.dtype != "float32" or len(stored.shape) != 4:
        refuse(
            frame.where,
            node,
            f"self.{field_} holds sums: fp32 tiles, rows x columns of them",
            f"declare self.{field_}: km.L1[km.fp32, rows, columns, 32, 32]",
        )
    field_fills.producer(frame, node, field_, "product")
    sums = frame.summed.get(field_)
    if sums is None:
        sums = f"km_sums_{field_}"
        frame.summed[field_] = sums
        frame.compiled.summed_fields.add(f"kField{camel(field_)}")
        frame.opening.append(Line(f"uint32_t {sums} = 0;  // self.{field_}'s"))
        frame.opening.append(Line(sums_room_line(field_)))
        frame.opening_b.append(  # mover B counts the same sums: its write's
            Line(f"uint32_t {sums} = 0;  // self.{field_}'s", mover="B")
        )
    return sums


def turned_blocks(frame, a, b, names, inner, columns, sides):
    """The broadcasts' product turned: its setup, a K tile's calls (W's tiles into
    SrcB, each row of X's subblock read transposed: DST i * width .. hold row i's
    transposed sums) and the lines after the K tiles (DST turned back, the setup
    again)."""
    row, column, k = names
    height, width = sides
    init = (
        f"KM_UNPACK_FORMATS({a.name}, {b.name});\n"
        f"matmul_block_init({b.name}, {a.name}, 1, 1, {width}, 1);"
    )
    calls = "".join(
        f"                matmul_block({b.name}, {a.name}, {k} * {columns} + "
        f"{column}, ({row} + {i}) * {inner} + {k}, {i * width}, 1, 1, {width}, "
        "1);\n"
        for i in range(height)
    )
    frame.compiled.includes.add("transpose_dest")
    lines = turned_tiles(height * width, 0).text.split("\n") + init.split("\n")[1:]
    turn = "".join(f"            {line}\n" for line in lines)
    return "".join(f"    {line}\n" for line in init.split("\n")), calls, turn


def turned_product(frame, node, value, function, item, field_, columns):
    """The field product turned: W's tiles (stored transposed) into SrcB, the
    block's tile read transposed into SrcA, acc's tiles transposed before the
    loop and after it, so DST sums (X W)^T between and X W outside."""
    state, number = frame.loop_states[-1], frame.stream_loops[-1]
    buffer = f"kField{camel(field_)}"
    block = item.buffer.name + "Block"
    x = f"{item.buffer.name}.mover_a"
    turn = turned_tiles(value.width, value.dst)
    if state["function"] is None:
        state["function"] = "product"
        waited = ""
        if field_ not in frame.waited_fields:  # once a handler
            frame.waited_fields.add(field_)
            wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
            waited = (
                f"if (!compute.wait_front({buffer}, {buffer}Pages)) {{\n"
                "    return;\n}\n"
            )
            if not field_fills.kept_field(frame, field_):
                field_freed(frame, field_, buffer)
        state["inits"].append(
            Line(
                f"{waited}"
                "// the zeros' fills (vector unit) done before the matrix unit "
                "adds\n"
                "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));"
            )
        )
    state["inits"] += [
        turn,
        Line(
            f"KM_UNPACK_FORMATS({x}, {buffer});\n"
            f"matmul_block_init({buffer}, {x}, 1, 1, {columns}, 1);"
        ),
    ]
    state["after"].append(turn)
    stored = frame.compiled.fields[field_]
    state["inits"] += k_bound(frame, node, state, field_, stored.shape[0], block)
    frame.compiled.includes |= {"matmul", "transpose_dest"}
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            columns,
            block,
            frame.compiled.fidelity,
        )
    )
    text = (
        f"matmul_block({buffer}, {item.cxx}, (km_block{number} * {block} + "
        f"{item.first}) * {columns}, {item.first}, {value.dst}, 1, 1, "
        f"{columns}, 1);"
    )
    return [], [Line(text)]


def turned_stream_product(frame, node, value, function, a, b):
    """The stream product turned: each W block (its tiles stored transposed)
    into SrcB, X's tile read transposed into SrcA, acc's tiles transposed before
    the loop and after it (turned_product's)."""
    state = frame.loop_states[-1]
    x, w = f"{a.buffer.name}.mover_a", f"{b.buffer.name}.mover_a"
    turn = turned_tiles(value.width, value.dst)
    if state["function"] is None:
        state["function"] = "product"
        state["inits"].append(
            Line(
                "// the zeros' fills (vector unit) done before the matrix unit "
                "adds\n"
                "MATH(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU));"
            )
        )
    state["inits"] += [
        turn,
        Line(
            f"KM_UNPACK_FORMATS({x}, {w});\n"
            f"matmul_block_init({w}, {x}, 1, 1, {value.width}, 1);"
        ),
    ]
    state["after"].append(turn)
    frame.compiled.includes.add("transpose_dest")
    frame.compiled.products.append(
        (
            frame.method.name,
            node.lineno,
            line_text(frame.where, node),
            function,
            value.width,
            f"{a.buffer.name}Block",
            frame.compiled.fidelity,
        )
    )
    text = (
        f"matmul_block({b.cxx}, {a.cxx}, {a.first} * {value.width}, {a.first}, "
        f"{value.dst}, 1, 1, {value.width}, 1);"
    )
    return [], [Line(text)]


def turned_sums(frame, a, b, w_tile, x_tile, width):
    """The field sums' product turned: its call (W's tile `w_tile` into SrcB, X's
    `x_tile` read transposed into SrcA, DST tiles 0.. the transposed sums), its
    setup, and the lines after a subblock's K tiles: its DST tiles transposed
    back, the setup again."""
    x, w = f"{a.buffer.name}.mover_a", f"{b.buffer.name}.mover_a"
    call = f"matmul_block({b.cxx}, {a.cxx}, {w_tile}, {x_tile}, 0, 1, 1, {width}, 1);"
    init = (
        f"KM_UNPACK_FORMATS({x}, {w});\n"
        f"matmul_block_init({w}, {x}, 1, 1, {width}, 1);"
    )
    frame.compiled.includes.add("transpose_dest")
    lines = turned_tiles(width, 0).text.split("\n") + init.split("\n")[1:]
    return call, init, "".join(f"        {line}\n" for line in lines)


def window_inside(frame, block_value, part, first, width, along):
    """A run-time check that a window of `width` from `first` along the part's
    rows ("rows") or columns ("columns") of the slot stays inside the cast's
    rectangle (a short block leaves the slot's tail holding an earlier block):
    "" where the build knows it, else a give-up past it."""
    index = block_value.parts.index(part)
    rows, columns = block_value.sides[index]
    if part[3]:  # turned: the slot's rows are the rectangle's columns
        rows, columns = columns, rows
    valid = str(rows if along == "rows" else columns)
    if valid.isdigit() and str(first).isdigit():
        return ""
    return (
        f"if (!(({first}) + {width} <= ({valid}))) {{  // the window inside the "
        "cast's rectangle\n"
        "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
        "    return;\n}\n"
    )


def is_product(function):
    """Whether a tile function is `return a @ b` of its two parameters, either one
    marked km.srcb(...) or km.srca(...)."""
    tree = source_of(function)[0]
    body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    names = [a.arg for a in tree.args.args]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or len(names) != 2:
        return False
    value = body[0].value
    return (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.MatMult)
        and [marked_operand(value.left)[0], marked_operand(value.right)[0]] == names
    )


def product_placement(function):
    """A product tile function's placement: "none" (no marker; the left operand in
    SrcB, the right in SrcA), "plain" (marked so) or "turned" (marked the other way:
    the right operand in SrcB); a str of why when both are marked for one side."""
    value = source_of(function)[0].body[-1].value
    left, right = marked_operand(value.left)[1], marked_operand(value.right)[1]
    if left is not None and left == right:
        return (
            f"both operands are marked km.{spelling(left)}: a product has one of each"
        )
    if left is None and right is None:
        return "none"
    return "turned" if left == "srca" or right == "srcb" else "plain"


def product_refusal(left, right, tf32=False):
    """None, or why a product of a `left` tile with a `right` tile is refused: each
    side of PRODUCT_FAMILIES, both of one family; float32 on either side only where
    the product sets its source as tf32 (`tf32`: two streams in step, plain)."""
    matrix = (*PRODUCT_FAMILIES["b"], *PRODUCT_FAMILIES["a"])
    for side, dtype in (("left", left), ("right", right)):
        if dtype == "float32" and tf32:
            continue
        if dtype == "float32":
            return (
                f"a product's {side} operand is float32: a class product takes "
                f"{', '.join(matrix)} (a float32 operand is read as tf32 only in a "
                "product of two streams in step); stream a bfloat16 copy of the "
                "tensor, or declare the field km.bf16"
            )
        if dtype not in matrix:
            return (
                f"a product's {side} operand is {dtype}: the matrix unit takes "
                f"{', '.join(matrix)}"
            )
    families = {
        f for f, names in PRODUCT_FAMILIES.items() for d in (left, right) if d in names
    }
    if len(families) > 1:
        return (
            f"a product of {left} and {right}: its operands are of one exponent "
            "family (bfloat16 and bfp*_b, or float16 and bfp*)"
        )
    return None


def product_setups(first, init):
    """A loop's later product of sums whose setup (`init`: SrcA's and SrcB's formats,
    the block's shape, a turn) is not the loop's first product's (`first`, set once
    before the blocks): the lines that set its own before its K tiles and the first's
    again after them; none when they are one."""
    if first is None or first == init:
        return "", ""
    own = "".join(f"    {line}\n" for line in init.split("\n"))
    return own, "".join(f"    {line}\n" for line in first.split("\n"))


def turned_tiles(tiles, first):
    """The C++ that transposes DST tiles first.. (`tiles` of them) in place: the
    matrix unit's transposed product's sums to and from the plain layout."""
    return Line(
        "transpose_dest_init<DST_ACCUM_MODE, true>();\n"
        + "\n".join(
            f"transpose_dest<DST_ACCUM_MODE, true>({first + j});" for j in range(tiles)
        )
    )


# A product's operand formats (each side its own), by exponent family; float32 refused
# (only a hand kernel's float32 product is proven: experiments/core/formats).
PRODUCT_FAMILIES = {
    "b": ("bfloat16", "bfp8_b", "bfp4_b", "bfp2_b"),
    "a": ("float16", "bfp8", "bfp4", "bfp2"),
}


def marked_operand(node):
    """An operand of `@`: its name and its marker ("srcb", "srca"; None: unmarked)."""
    sides = {spelling("srcb"): "srcb", spelling("srca"): "srca"}
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "km"
        and node.func.attr in sides
        and len(node.args) == 1
        and not node.keywords
    ):
        return getattr(node.args[0], "id", None), sides[node.func.attr]
    return getattr(node, "id", None), None
