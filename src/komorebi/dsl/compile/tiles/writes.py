"""A handler's writes: a value's pages written by mover B, of DST, a pair, sums, a
row or a field, a step's result, and the reads that feed a write."""

import ast
import inspect
import math

from komorebi.dsl.compile import expressions, handler_buffers, scope, tile_lookup
from komorebi.dsl.compile.cxx import camel, pattern_uses, range_uses
from komorebi.dsl.compile.dst import dst_tiles
from komorebi.dsl.compile.dst.dst_code import DST_FULL, DST_HALF, pack, sums_room
from komorebi.dsl.compile.pages import field_fills, page_streams, page_words, ranges
from komorebi.dsl.compile.source import (
    WAIT_DATA,
    WAIT_INPUT,
    WAIT_OUTPUT,
    line_text,
    refuse,
    wait,
)
from komorebi.dsl.compile.tiles import functions, products, steps
from komorebi.dsl.compile.tiles.functions import tile_format_refusal
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Line


def dst_pair_of(frame, call):
    """The two DST sums held together that a tile function's call takes (`f(a,
    b)`, a and b the pair of a second zeros, each once), or None."""
    if (
        frame.dst_pair is None
        or len(call.args) != 2
        or call.keywords
        or not all(isinstance(arg, ast.Name) for arg in call.args)
    ):
        return None
    names = tuple(arg.id for arg in call.args)
    return names if set(names) == set(frame.dst_pair) else None


def dst_sums_of(frame, call):
    """The name of the DST sums a tile function's call takes (`f(acc)`, acc a
    km.tile.zeros sum compute holds), or None."""
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
        return None
    value = frame.scope.get(call.args[0].id)
    if value is None or value.kind != "tile" or value.batch is not None:
        return None
    return call.args[0].id


def dst_field_of(frame, call):
    """(acc, field) of a tile function's call `f(acc, self.b)`, acc DST sums and
    self.b a field, or None."""
    if len(call.args) != 2 or call.keywords:
        return None
    acc, operand = call.args
    value = isinstance(acc, ast.Name) and frame.scope.get(acc.id)
    if not value or value.kind != "tile" or value.batch is not None:
        return None
    field_ = scope.field_name(frame, operand)
    return (acc.id, field_) if field_ else None


def fed_moves(frame, node, loop, body_m):
    """The moves of the write streams a loop's bodies feed (take_fed), after the
    loop; refused when the loop has movers' lines too (each would wait on the
    other)."""
    moves = loop.get("moves", [])
    if moves and body_m:
        refuse(
            frame.where,
            node,
            "a write stream fed a block a turn is moved after the loop, and this "
            "loop has movers' lines too: the movers would wait on each other",
            "feed it in a loop of compute's lines alone",
        )
    return list(moves)


def row_written(frame, call):
    """Whether `call` is a tile function's call on a row of more than 2 tiles."""
    first = call.args[0] if call.args else None
    value = isinstance(first, ast.Name) and frame.scope.get(first.id)
    return bool(value) and value.kind == "streamed" and (value.width or 1) > 2


def take_fed(frame, node, name, value, bound, before, after):
    """A write stream made outside the loops around a body on compute or pack:
    one block of it each time the body runs (compute counts them), reserved
    before the body and pushed after it; the movers move the whole stream after
    the outermost of those loops (fed_moves)."""
    if value.move is not None and not value.table:
        refuse(
            frame.where,
            node,
            f"{name} is fed above outside a loop",
            "open a write stream for each",
        )
    buffer, fed = value.buffer.name, f"km_fed_{name}"
    out, room = f"km_out_{name}", f"km_room_{name}"
    given = value.pattern if value.pattern in frame.scope else None
    before.append(
        Line(
            f"const uint32_t {out} = km::block_half({buffer}, {value.pattern}, "
            f"{fed});\n"
            f"const bool {room} = compute.reserve_back({out}, {buffer}Block);",
            uses=pattern_uses(given) | {fed},
        )
    )
    bound.append(
        f"const uint32_t {name} = {room} ? "
        f"get_local_cb_interface({out}).fifo_wr_ptr << 4 : 0;"
    )
    bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {out};")
    after.append(
        Line(
            f"if ({room}) {{\n    cb_push_back({out}, {buffer}Block);\n}}\n"
            f"{fed}++;",
            {fed},
        )
    )
    if value.move is None:
        value.move = Line(
            f"if (!movement.move({value.cxx})) {{\n    return false;\n}}",
            {value.cxx},
        )
        value.table = {"fed": True}
        frame.loops[value.depth].setdefault("moves", []).append(value.move)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    return set()


def take_write(frame, node, name, value, bound, before, after):
    """A write stream's page for a body on pack: in a stream loop the loop's
    page (reserved with the block, pushed after it); outside one, the stream's
    one block, reserved, pushed and moved at this line. The page's L1 address,
    0 when compute gave up."""
    if value.pushed:
        refuse(
            frame.where,
            node,
            f"{name} is fed above (a push or another body)",
            "open a write stream for each",
        )
    value.pushed = True
    buffer = value.buffer.name
    if frame.stream_loops:
        state = frame.loop_states[-1]
        if value.buffer.block != state["block"]:
            refuse(
                frame.where,
                node,
                f"{name}'s block is {value.buffer.block} page(s), the loop's "
                f"{state['block']}",
                "give the write stream the loop's block=",
            )
        page_streams.loop_fed(frame, node, name, value)
        state["body_pushed"] = True
        number = frame.stream_loops[-1]
        out, room, tile = f"km_out{number}", f"km_room{number}", f"km_tile{number}"
        ring = f"get_local_cb_interface({out})"
        bound.append(
            f"const uint32_t {name} = {room} ? ({ring}.fifo_wr_ptr + {tile} * "
            f"{ring}.fifo_page_size) << 4 : 0;"
        )
        bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {out};")
        return set()
    if value.depth < len(frame.loops):  # a block each turn of the loops inside
        return take_fed(frame, node, name, value, bound, before, after)
    if value.count != str(value.buffer.block):
        refuse(
            frame.where,
            node,
            f"{name} outside a stream loop is one block: {value.count} page(s), "
            f"its block {value.buffer.block}",
            "give the write stream one block, or fill it in a stream loop",
        )
    out, room = f"km_out_{name}", f"km_room_{name}"
    given = value.pattern if value.pattern in frame.scope else None
    before.append(
        Line(
            f"const uint32_t {out} = "
            f"km::block_half({buffer}, {value.pattern}, 0);\n"
            f"const bool {room} = compute.reserve_back({out}, {buffer}Block);",
            uses=pattern_uses(given),
        )
    )
    bound.append(
        f"const uint32_t {name} = {room} ? "
        f"get_local_cb_interface({out}).fifo_wr_ptr << 4 : 0;"
    )
    bound.append(f"const uint32_t {name}{spelling('tile_buffer')} = {out};")
    after.append(Line(f"if ({room}) {{\n    cb_push_back({out}, {buffer}Block);\n}}"))
    value.move = Line(
        f"if (!movement.move({value.cxx})) {{\n    return false;\n}}", {value.cxx}
    )
    frame.body_moves.append(value.move)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    return set()


def write(frame, node, call):
    target = call.func.value
    if scope.field_name(frame, target) is not None:
        return write_field(frame, node, call, scope.field_name(frame, target))
    if isinstance(target, ast.Subscript) and scope.field_name(frame, target.value):
        return write_row(
            frame, node, call, scope.field_name(frame, target.value), target.slice
        )
    if not ranges.is_range(frame, target):
        refuse(
            frame.where,
            node,
            "write to a page range",
            "write tensor.pages(first, 1).write(tile)",
        )
    pages = ranges.range_of(frame, target)
    handler_buffers.pages_only(frame, node, pages)
    summed = call.args[0] if len(call.args) == 1 else None
    if scope.field_name(frame, summed) in frame.summed:
        return write_sums(frame, node, pages, scope.field_name(frame, summed))
    if (
        tile_lookup.tile_function_of(frame, summed) is not None
        and summed.args
        and all(scope.field_name(frame, arg) in frame.summed for arg in summed.args)
    ):
        return write_function_of_sums(frame, node, pages, summed)
    if tile_lookup.tile_function_of(frame, summed) is not None and dst_pair_of(
        frame, summed
    ):
        return write_function_of_pair(frame, node, call, summed)
    if tile_lookup.tile_function_of(frame, summed) is not None and dst_sums_of(
        frame, summed
    ):
        return write_function_of_dst(frame, node, call, summed)
    if tile_lookup.tile_function_of(frame, summed) is not None and steps.step_of_dst(
        frame, summed
    ):
        return write_step_of_dst(frame, node, call, summed)
    if tile_lookup.tile_function_of(frame, summed) is not None and dst_field_of(
        frame, summed
    ):
        return write_function_of_field(frame, node, call, summed)
    if scope.field_name(frame, summed) in frame.stepped:
        return steps.write_state(frame, node, pages, scope.field_name(frame, summed))
    if tile_lookup.tile_function_of(frame, summed) is not None:
        refuse(
            frame.where,
            node,
            "a write of a tile function takes DST sums: f(acc); f(a, b) of a pair "
            "held together (a second zeros); f(acc, self.s[r]), a step; f(acc, "
            "self.b), self.b a field",
            "write pages.write(f(acc)) of sums added in the loop above",
        )
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
        refuse(frame.where, node, "write one tile", "write pages.write(total)")
    name = call.args[0].id
    value = frame.scope.get(name)
    if value is not None and value.kind == "word":
        return page_words.write_word(frame, node, pages, value)
    dst_tiles.not_paired(frame, node, name)
    if getattr(value, "dropped", None) is not None:
        refuse(
            frame.where,
            node,
            f"{name} is written on one branch of the if at line {value.dropped} "
            "only, and freed on the other",
            f"write {name} on each branch of that if, or after it alone",
        )
    if getattr(value, "moved", None) is not None:  # written above: once more
        return write_again(frame, node, pages, name, value)
    carried = dst_tiles.carries(frame, node, value)
    held_sum = dst_tiles.sum_held(frame, value)
    if (
        value is not None
        and value.kind in ("tile", "reduced")
        and not value.popped
        and not carried
        and not (held_sum and not value.summed)
    ):
        dst_tiles.same_loops(
            frame, node, name, value, "this write", *dst_tiles.once_text(frame, value)
        )
    if value is not None and value.kind == "reduced" and value.summed:
        if not frame.in_leader or value.popped:
            refuse(
                frame.where,
                node,
                f"{name} is the reduce's sum of self.{value.summed}, which only "
                "the leader holds, once",
                f"write it once under `if self.{spelling('rank')} == "
                f"self.{spelling('leader')}:`",
            )
        value.popped = True
        frame.leader_writes.add(name)
        return write_sums(frame, node, pages, value.summed, pushed=True)
    if value is None or value.kind not in ("tile", "reduced") or value.popped:
        refuse(
            frame.where,
            node,
            f"{name} is not a tile of compute (or is written already)",
            "write a tile made by km.tile.zeros and added to, once",
        )
    width = value.width
    if pages.rect is not None:
        refuse(
            frame.where,
            node,
            "a page write of a rectangle is one row of it",
            f"write {pages.tensor}.{spelling('tiles')}[r:r + 1, c0:c1].write(...)",
        )
    if pages.count != str(width):
        refuse(frame.where, node, *width_written(name, width, pages))
    frame.compiled.includes.add("pack")
    compute = []
    converted = False
    if value.kind == "tile":
        dtype, page_bytes = handler_buffers.tensor_format(frame, node, pages.tensor)
        handler_buffers.tile_packed(frame, node, pages.tensor)
        buffer = handler_buffers.out_buffer(
            frame,
            dtype,
            page_bytes,
            f"line {node.lineno}: {line_text(frame.where, node)}",
            width,
        )
        if carried:
            compute.append(
                Line(dst_tiles.carried_write(frame, node, value, buffer), {"km_ok"})
            )
            value.buffer = buffer
            wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
            wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
            movement, _ = write_moved(frame, node, pages, value, buffer, width)
            return movement, compute
        compute.append(Line(pack(value.dst, buffer.name, width), {"km_ok"}))
        if getattr(value, "state", None) is not None:  # started from a field
            steps.state_packed(frame, compute, *value.state)
        steps.row_given_back(frame, compute, value)
        frame.dst_live = None
        value.kind = "written"
        value.buffer = buffer
        wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
    else:
        if not frame.in_leader and not (
            value.everywhere
            and not held_sum
            and handler_buffers.tensor_format(frame, node, pages.tensor)[0]
            == value.buffer.dtype
        ):
            refuse(
                frame.where,
                node,
                (
                    f"{name} is km.{spelling('all_reduce')}'s sum, float32 at "
                    "every rank; packed again in another format by compute, whose "
                    "buffer front at a rank but the leader is the part it sent"
                    if value.everywhere
                    else f"{name} is the reduce's sum, which only the leader holds"
                ),
                ("write it to a float32 tensor, or" if value.everywhere else "write it")
                + f" under `if self.{spelling('rank')} == "
                f"self.{spelling('leader')}:`",
            )
        buffer = value.buffer
        frame.leader_writes.add(name)
        dtype, page_bytes = handler_buffers.tensor_format(frame, node, pages.tensor)
        handler_buffers.tile_packed(frame, node, pages.tensor)
        if held_sum:
            buffer = handler_buffers.out_buffer(
                frame,
                dtype,
                page_bytes,
                f"line {node.lineno}: {line_text(frame.where, node)}",
                width,
            )
            compute.append(dst_tiles.held_sum_copy(frame, value, buffer, width))
            converted = True
        elif dtype != buffer.dtype:  # the sum packed again in the tensor's format
            source = buffer
            buffer = handler_buffers.out_buffer(
                frame,
                dtype,
                page_bytes,
                f"line {node.lineno}: {line_text(frame.where, node)}",
                width,
            )
            frame.compiled.unpack_fp32.add(source.name)
            copies = "\n".join(
                f"copy_tile({source.name}, {t}, {t});" for t in range(width)
            )
            again = f"{copies}\ncb_pop_front({source.name}, {width});\n" + pack(
                0, buffer.name, width
            )
            body = "\n".join("    " + line for line in again.split("\n"))
            compute.append(
                Line(
                    f"// the sum in {source.dtype}, packed again in {dtype}\n"
                    "km_ok = km_ok && compute.wait_front("
                    f"{source.name}, {width});\n"
                    "if (km_ok) {\n"
                    f"    KM_UNPACK_FORMAT_A({source.name});\n"
                    f"    copy_init({source.name});\n"
                    "    tile_regs_acquire();\n"
                    f"{body}\n}}",
                    {"km_ok"},
                )
            )
            converted = True
    return write_moved(frame, node, pages, value, buffer, width, converted, compute)


def write_field(frame, node, call, field_):
    """`self.f.write(acc)`, acc DST sums (or `f(acc)`, `f(acc, self.s[r])` a
    function of them): acc's tiles packed into the field's pages for products
    that take it as their left side. A write frees the last one's pages (their
    readers are done), the handler's end frees the last."""
    written = call.args[0] if len(call.args) == 1 else None
    if tile_lookup.tile_function_of(frame, written) is not None and row_written(
        frame, written
    ):
        return write_row_value(frame, node, field_, written)
    if tile_lookup.tile_function_of(frame, written) is not None and dst_sums_of(
        frame, written
    ):
        return write_function_of_dst(frame, node, call, written)
    if tile_lookup.tile_function_of(frame, written) is not None and steps.step_of_dst(
        frame, written
    ):
        return write_step_of_dst(frame, node, call, written)
    value = isinstance(written, ast.Name) and frame.scope.get(written.id)
    if not value or value.kind != "tile" or value.popped:
        refuse(
            frame.where,
            node,
            f"self.{field_} takes a tile of compute's DST sums (or a tile function "
            "of them), once",
            f"write self.{field_}.write(acc), acc made by km.tile.zeros and added "
            "to",
        )
    dst_tiles.same_loops(
        frame,
        node,
        written.id,
        value,
        "this write",
        *dst_tiles.once_text(frame, value),
    )
    stored = frame.compiled.fields[field_]
    pages = math.prod(stored.shape[:-2])
    fillers = frame.compiled.field_users.get(field_, ([], []))[0]
    if (
        fillers
        or field_ in frame.compiled.own
        or field_ in frame.summed
        or field_ in frame.stepped
        or len(stored.shape) < 3
    ):
        refuse(
            frame.where,
            node,
            f"self.{field_} is filled, summed into or a body's: compute writes "
            "a field of its own",
            f"declare a field for the write: km.L1[km.bf16, {value.width}, 32, 32]",
        )
    if pages != value.width:
        refuse(
            frame.where,
            node,
            f"self.{field_} holds {pages} tile(s) and {written.id} is "
            f"{value.width}",
            f"declare self.{field_}: km.L1[km.bf16, {value.width}, 32, 32]",
        )
    field_fills.producer(frame, node, field_, "product")
    if getattr(value, "row_of", None) is not None:  # sums started from a mask
        frame.masked.add(field_)
    else:
        frame.masked.discard(field_)
    buffer, held = f"kField{camel(field_)}", f"km_packed_{field_}"
    frame.compiled.summed_fields.add(buffer)  # pack writes it: one of its formats
    if field_ not in frame.packed:
        frame.packed[field_] = held
        frame.opening.append(
            Line(f"bool {held} = false;  // self.{field_}'s pages pushed, not freed")
        )
        frame.closing.append(
            Line(f"if ({held}) {{\n    cb_pop_front({buffer}, {buffer}Pages);\n}}")
        )
    frame.compiled.includes.add("pack")
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
    value.kind, value.popped = "written", True
    frame.dst_live = None
    state = getattr(value, "state", None)
    text = (
        f"if ({held}) {{  // the last write's pages: their products are done\n"
        f"    cb_pop_front({buffer}, {buffer}Pages);\n"
        "}\n"
        f"{pack(value.dst, buffer, value.width)}\n"
        f"{held} = km_ok;"
    )
    compute = [Line(text, {"km_ok"})]
    if state is not None:
        steps.state_packed(frame, compute, *state)
    steps.row_given_back(frame, compute, value)
    return [], compute


def write_function_of_dst(frame, node, write_call, call):
    """`pages.write(f(acc))`, acc DST sums (a product's over K, added in the
    loop): f's body run on each of acc's DST tiles in place on the math CPU, then
    acc written as `pages.write(acc)` writes it."""
    function = tile_lookup.tile_function_of(frame, call)
    name = dst_sums_of(frame, call)
    value = frame.scope[name]
    unit, cxx = functions.tile_unit(frame, node, function, ["float32"])
    if unit.products:
        refuse(
            frame.where,
            node,
            f"{function.name} has a matrix product (`@`), and its operand here is "
            f"{name}, sums in DST that the matrix unit does not read",
            f"add the product in the loop and write {function.name}({name}) of "
            "its sums",
        )
    calls = "\n".join(
        f"MATH((km_tile::{cxx}({value.dst + k})));" for k in range(value.width)
    )
    body = f"MATH((km_tile::{cxx}_init()));  // {function.name}'s setup\n{calls}"
    frame.compiled.tile_calls.append(
        (frame.method.name, node.lineno, line_text(frame.where, node), function, cxx)
    )
    write_call.args[0] = ast.Name(id=name, ctx=ast.Load())
    movement, compute = write(frame, node, write_call)
    return movement, [Line(body), *compute]


def write_function_of_field(frame, node, write_call, call):
    """`pages.write(f(acc, self.b))`, acc n DST sums, self.b a field of n tiles or
    one for each: the field's tiles (or their row or column, broadcast) into the DST
    tiles after acc's, f on tile t of each as the pair form runs it, acc written;
    the field waited for once and freed at the handler's end, as a product's is."""
    function = tile_lookup.tile_function_of(frame, call)
    name, field_ = dst_field_of(frame, call)
    value = frame.scope[name]
    stored = frame.compiled.fields[field_]
    held = math.prod(stored.shape[:-2])
    if held not in (1, value.width):
        refuse(
            frame.where,
            node,
            f"{function.name} takes {name} ({value.width} tiles) and self.{field_} "
            f"({held} tiles): tile t of each, or the field's one tile for each",
            f"declare self.{field_} of {value.width} tiles, or of 1",
        )
    first = max(frame.dst_next, value.dst + value.width)
    if first + value.width > DST_FULL:
        refuse(
            frame.where,
            node,
            f"{name} is {value.width} tiles of DST from tile {value.dst}, and "
            f"self.{field_}'s copies take {value.width} more: DST holds {DST_FULL} "
            "fp32 tiles",
            f"sum at most {(DST_FULL - value.dst) // 2} tiles in {name}",
        )
    if first + value.width > DST_HALF:  # past a half: DST in full-sync mode
        frame.compiled.full_dst = True
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
    if filled_in[0] > 0 and products.fill_turn_loop(frame, field_) is None:
        refuse(
            frame.where,
            node,
            f"self.{field_} is filled in a loop (line {filled_in[1]}) and this "
            "write frees it once, at the handler's end: the next turn's fill "
            "would wait for room that is never freed",
            f"fill it before the loops, once (self.{field_}.fill(km."
            f"{spelling('stream')}(...)))",
        )
    parameters = list(inspect.signature(function.function).parameters)
    tiles = dict(zip(parameters, (0, first - value.dst), strict=False))
    unit, cxx = functions.tile_unit(
        frame, node, function, ["float32", "float32"], tiles=tiles
    )
    if unit.products:
        refuse(
            frame.where,
            node,
            f"{function.name} has a matrix product (`@`), and its operand here is "
            f"{name}, sums in DST that the matrix unit does not read",
            f"add the product in the loop and write {function.name} of its sums",
        )
    along = field_reads(frame, node, function, unit, parameters)
    buffer = f"kField{camel(field_)}"
    waited = ""
    if field_ not in frame.waited_fields:  # once a handler
        frame.waited_fields.add(field_)
        wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
        waited = f"km_ok = km_ok && compute.wait_front({buffer}, {buffer}Pages);\n"
        if not field_fills.kept_field(frame, field_):
            products.field_freed(frame, field_, buffer)
    if stored.format.dtype == "float32":
        frame.compiled.unpack_fp32.add(buffer)
    if along is None:
        lines = [f"KM_UNPACK_FORMAT_A({buffer});", f"copy_init({buffer});"] + [
            f"copy_tile({buffer}, {t if held > 1 else 0}, {first + t});"
            for t in range(value.width)
        ]
    else:  # row 0 to every row (`.row()`), or column 0 to every column (`.col()`)
        frame.compiled.includes.add("bcast")
        kind = f"BroadcastType::{along}"
        lines = [
            f"KM_UNPACK_FORMATS({buffer}, {buffer});",
            f"unary_bcast_init<{kind}>({buffer});",
            *[
                f"unary_bcast<{kind}>({buffer}, {t if held > 1 else 0}, {first + t});"
                for t in range(value.width)
            ],
            f"unary_bcast_uninit<{kind}>({buffer});",
        ]
    copies = "".join(f"    {line}\n" for line in lines)
    calls = "\n".join(
        f"MATH((km_tile::{cxx}({value.dst + k})));" for k in range(value.width)
    )
    body = (
        f"// self.{field_}'s tiles into DST {first} .. {first + value.width - 1}\n"
        f"{waited}"
        "if (km_ok) {\n"
        f"{copies}"
        "}\n"
        f"MATH((km_tile::{cxx}_init()));  // {function.name}'s setup\n{calls}"
    )
    frame.compiled.tile_calls.append(
        (frame.method.name, node.lineno, line_text(frame.where, node), function, cxx)
    )
    write_call.args[0] = ast.Name(id=name, ctx=ast.Load())
    movement, compute = write(frame, node, write_call)
    return movement, [Line(body, {"km_ok"}), *compute]


def field_reads(frame, node, function, unit, parameters):
    """How a write's tile function reads its field operand: whole (None), by
    `.row()` ("ROW") or by `.col()` ("COL"), one way; its sums operand whole."""
    kinds = {}
    for k, operand in enumerate(unit.operands):
        kind = "ROW" if k in unit.row_operands else None
        kinds.setdefault(operand, set()).add("COL" if k in unit.col_operands else kind)
    sums, field_ = parameters[0], parameters[1]
    if kinds.get(sums, {None}) != {None}:
        refuse(
            frame.where,
            node,
            f"{function.name} reads {sums}, DST sums, by `.row()` or `.col()`: a "
            "row or column of a field is broadcast, and sums are whole tiles",
            f"read {sums} whole, and the field ({field_}) by `.row()` or `.col()`",
        )
    if len(kinds.get(field_, {None})) > 1:
        refuse(
            frame.where,
            node,
            f"{function.name} reads {field_} in two ways (whole, `.row()`, "
            "`.col()`): its DST tiles hold one copy of the field",
            f"read {field_} one way",
        )
    return next(iter(kinds.get(field_, {None})))


def write_function_of_pair(frame, node, write_call, call):
    """`pages.write(f(a, b))`, a and b DST sums held together (a second zeros):
    f's body run on DST tile t of the first of them with the other operand read
    its distance apart, for each of their tiles, then the first's tiles written as
    `pages.write(acc)` writes them (DST released)."""
    function = tile_lookup.tile_function_of(frame, call)
    values = [frame.scope[arg.id] for arg in call.args]
    if values[0].width != values[1].width:
        refuse(
            frame.where,
            node,
            f"{function.name} takes {call.args[0].id} ({values[0].width} tiles) "
            f"and {call.args[1].id} ({values[1].width} tiles): tile t of each",
            "make the two sums one width",
        )
    for value in values:
        dst_tiles.same_loops(
            frame,
            node,
            value.cxx,
            value,
            "this write",
            *dst_tiles.once_text(frame, value),
        )
    base = min(values, key=lambda v: v.dst)
    parameters = list(inspect.signature(function.function).parameters)
    tiles = {p: v.dst - base.dst for p, v in zip(parameters, values, strict=False)}
    unit, cxx = functions.tile_unit(
        frame, node, function, ["float32", "float32"], tiles=tiles
    )
    if unit.products:
        refuse(
            frame.where,
            node,
            f"{function.name} has a matrix product (`@`), and its operands here "
            "are sums in DST that the matrix unit does not read",
            f"add the products in the loops and write {function.name} of the sums",
        )
    if unit.row_operands or unit.col_operands:
        refuse(
            frame.where,
            node,
            f"{function.name} reads an operand by `.row()` or `.col()`, and its "
            "operands here are DST sums, whole tiles: a row or column is broadcast "
            "from a field",
            f"read both sums whole, or write {function.name}(acc, self.b) of a field",
        )
    calls = "\n".join(
        f"MATH((km_tile::{cxx}({base.dst + k})));" for k in range(base.width)
    )
    body = f"MATH((km_tile::{cxx}_init()));  // {function.name}'s setup\n{calls}"
    frame.compiled.tile_calls.append(
        (frame.method.name, node.lineno, line_text(frame.where, node), function, cxx)
    )
    other = values[1] if base is values[0] else values[0]
    other.kind = "written"  # its sums end with the write of the first's
    frame.dst_pair = None
    frame.dst_live = base.cxx
    write_call.args[0] = ast.Name(id=base.cxx, ctx=ast.Load())
    movement, compute = write(frame, node, write_call)
    return movement, [Line(body), *compute]


def write_function_of_sums(frame, node, pages, call):
    """`t.tiles[r0:r1, c0:c1].write(f(self.h, self.g))`, f a tile function of
    fields of sums of one shape: each tile's fields copied into DST k (the unit's
    operand order), f's body, DST 0 packed in t's format and written a row at a
    time by mover B as `write(self.h)` is; the fields freed, the next sums start."""
    function = tile_lookup.tile_function_of(frame, call)
    fields = [scope.field_name(frame, arg) for arg in call.args]
    sides = {expressions.field_sides(frame, node, f) for f in fields}
    if len(sides) > 1 or len(set(fields)) != len(fields):
        refuse(
            frame.where,
            node,
            f"{function.name} takes fields of sums of one shape, each once, and "
            f"is given {', '.join('self.' + f for f in fields)}",
            "give it fields of one shape",
        )
    ((rows, columns),) = sides
    formats = [frame.compiled.fields[f].format.dtype for f in fields]
    why = tile_format_refusal(formats)
    if why is not None:
        refuse(frame.where, node, f"{function.name}: {why}", "give it fields of those")
    unit, cxx = functions.tile_unit(frame, node, function, formats)
    if unit.products:
        refuse(
            frame.where,
            node,
            f"{function.name} has a matrix product (`@`), and its operands here "
            "are fields of sums in L1, which the unit reads from DST",
            f"add the product into a field (self.f += self.p(a, b)) and write "
            f"{function.name} of the fields",
        )
    rect = pages.rect
    bounds = pages.bounds or (None, None)
    written = rect[:2] if rect is not None else ("1", pages.count)
    fits = all(
        side_fits(side, bound, most)
        for side, bound, most in zip(written, bounds, (rows, columns), strict=True)
    )
    if not fits:
        refuse(
            frame.where,
            node,
            f"the fields are {rows} x {columns} tiles; write a rectangle of them",
            f"write {pages.tensor}.{spelling('tiles')}[r:r + {rows}, c:c + "
            f"{columns}].write({function.name}(...))",
        )
    dtype, page_bytes = handler_buffers.tensor_format(frame, node, pages.tensor)
    name = f"kOutRows{columns}"
    held = [b.pages[0] for b in frame.compiled.buffers if b.name == name]
    out = handler_buffers.out_buffer(
        frame,
        dtype,
        page_bytes,
        f"line {node.lineno}: {line_text(frame.where, node)}",
        columns,
        name,
    )
    out.pages = (max(held + [columns * min(rows, 2)]),)
    parameters = list(inspect.signature(function.function).parameters)
    buffers = {f: f"kField{camel(f)}" for f in fields}
    counts = [frame.summed[f] for f in fields]
    made = " && ".join(f"{sums} > 0" for sums in counts)
    tile, out_room = f"km_tile{node.lineno}", f"km_room{node.lineno}"
    lines = [f"// {line_text(frame.where, node)}: {function.name} of the fields' sums"]
    for f in fields:
        buffer, sums = buffers[f], frame.summed[f]
        lines += [
            f"if (km_ok && {sums} > 0) {{",
            f"    if ({sums_room(f)}) {{  // pack's: its pages held",
            f"        cb_push_back({buffer}, {buffer}Pages);",
            "    }",
            f"    km_ok = compute.wait_front({buffer}, {buffer}Pages);",
            "}",
        ]
    first = buffers[fields[0]]
    lines += [
        f"if (km_ok && {made}) {{",
        f"    KM_UNPACK_FORMAT_A({first});",
        f"    copy_init({first});",
        f"    MATH((km_tile::{cxx}_init()));  // {function.name}'s setup",
        f"    bool {out_room} = true;  // pack's alone: it guards the packs only",
        f"    for (uint32_t {tile} = 0; km_ok && {tile} < {rows * columns}; "
        f"{tile}++) {{",
        "        tile_regs_acquire();",
    ]
    for dst, operand in enumerate(unit.operands):
        buffer = buffers[fields[parameters.index(operand)]]
        lines.append(f"        copy_tile({buffer}, {tile}, {dst});")
    lines += [
        f"        MATH((km_tile::{cxx}(0)));",
        "        tile_regs_commit();",
        "        tile_regs_wait();",
        f"        {out_room} = {out_room} && compute.reserve_back({out.name}, 1);",
        f"        if ({out_room}) {{",
        f"            KM_PACK_FORMAT({out.name});",
        f"            pack_tile(0, {out.name});",
        f"            cb_push_back({out.name}, 1);",
        "        }",
        "        tile_regs_release();",
        "    }",
        *[f"    cb_pop_front({b}, {b}Pages);" for b in buffers.values()],
        "}",
        *[f"{sums} = 0;" for sums in counts],
    ]
    tensor = f"{camel(pages.tensor)}({pages.tensor})"
    row = f"km_row{node.lineno}"
    row_step = rect[2] if rect is not None else columns
    whole = str(written[0]) == str(rows)
    part = "" if whole else f"{row} < ({written[0]}) &&\n    "
    movement = [
        f"if ({made}) {{",
        f"    for (uint32_t {row} = 0; {row} < {rows}; {row}++) {{",
        f"        if (!movement.wait_front({out.name}, {columns}) ||",
        f"            {'(' if part else ''}{part}!km::write_pages(movement, "
        f"command, {tensor}, {pages.first} + {row} * {row_step}, {written[1]}, "
        f"get_read_ptr({out.name})){')' if part else ''}) {{",
        "            return false;",
        "        }",
        "        noc_async_writes_flushed();",
        f"        cb_pop_front({out.name}, {columns});",
        "    }",
        "}",
        *[f"{sums} = 0;" for sums in counts],
    ]
    frame.compiled.tile_calls.append(
        (frame.method.name, node.lineno, line_text(frame.where, node), function, cxx)
    )
    frame.compiled.includes |= {"pack", "ask"}
    handler_buffers.moved(frame, node, pages.tensor, "page write")
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    return [Line("\n".join(movement), range_uses(pages, frame), mover="B")], [
        Line("\n".join(lines), {"km_ok"})
    ]


def write_moved(frame, node, pages, value, buffer, width, converted=False, compute=()):
    """Mover B's page write of `value`'s `width` pages from `buffer`, and
    `compute`'s lines with it."""
    # A page write: mover B waits for compute's page, writes it, frees it once
    # sent (as a hand writer does); the runtime settles writes before it reports.
    tensor = f"{camel(pages.tensor)}({pages.tensor})"
    written = (
        f"km::write_pages(movement, command, {tensor}, {pages.first}, {width}, "
        f"get_read_ptr({buffer.name}))"
    )
    if value.waited:
        opening = f"if (!{written}) {{"
    elif converted:
        opening = (
            f"if (!movement.wait_front({buffer.name}, {width}) ||\n"
            f"    !{written}) {{"
        )
    else:
        opening = (
            handler_buffers.page_wait(frame, value, width)[:-3]
            + f" ||\n    !{written}) {{"
        )
    text = f"{opening}\n    return false;\n}}\n" + freed(buffer, width)
    movement = [Line(text, range_uses(pages, frame), mover="B")]
    if value.cxx not in frame.carried:  # a running total stays live
        value.waited = value.popped = True
        dtype = handler_buffers.tensor_format(frame, node, pages.tensor)
        value.moved = (movement[0], buffer, width, frame.blocks[-1], dtype, node.lineno)
    handler_buffers.moved(frame, node, pages.tensor, "page write")
    frame.compiled.includes.add("ask")
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    return movement, list(compute)


def width_written(name, width, pages, first=None):
    """The refusal of a write whose pages are not the tile's width: the count or
    slice the build could not work out by name, where that is the cause."""
    at = "" if first is None else f", written as {width} page(s) at line {first}"
    if not pages.counted:
        return f"{name} is {width} tile(s), {width} page(s){at}", (
            f"write pages(first, {width})"
        )
    return (
        f"{name} is {width} tile(s){at}, and the build cannot work out the length "
        f"of {pages.counted}",
        f"write pages(first, {width}), or a slice c : c + {width}",
    )


def freed(buffer, width):
    """Mover B's free of a written tile's pages, its writes flushed."""
    return f"noc_async_writes_flushed();\ncb_pop_front({buffer.name}, {width});"


def write_again(frame, node, pages, name, value):
    """A written tile's pages written once more by mover B, before it frees them
    (compute packed them once): the free moves from the write above to this one."""
    line, buffer, width, block, dtype, first = value.moved
    if block != frame.blocks[-1]:
        refuse(
            frame.where,
            node,
            f"{name} is written at line {first} in another statement list: a "
            "tile's writes are lines of one list (an if's branch writes it on "
            "each branch alone)",
            f"write {name} again beside line {first}",
        )
    if handler_buffers.tensor_format(frame, node, pages.tensor) != dtype:
        refuse(
            frame.where,
            node,
            f"{name}'s pages are packed once, as {dtype[0]} for its write at line "
            f"{first}, and {pages.tensor} is of another format",
            f"write it to a tensor of {dtype[0]}, or write a copy of the tile",
        )
    if pages.rect is not None or pages.count != str(width):
        refuse(frame.where, node, *width_written(name, width, pages, first))
    line.text = line.text.removesuffix("\n" + freed(buffer, width))
    tensor = f"{camel(pages.tensor)}({pages.tensor})"
    text = (
        f"if (!km::write_pages(movement, command, {tensor}, {pages.first}, {width}, "
        f"get_read_ptr({buffer.name}))) {{\n    return false;\n}}\n"
        + freed(buffer, width)
    )
    value.moved = (Line(text, range_uses(pages, frame), mover="B"), *value.moved[1:])
    handler_buffers.moved(frame, node, pages.tensor, "page write")
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    return [value.moved[0]], []


def write_row(frame, node, call, field_, row):
    """`self.o[j].write(acc)`, acc started from a row of self.o: its tiles packed
    into row j in place, the token pushed, then the state packed (acc's)."""
    written = call.args[0] if len(call.args) == 1 else None
    value = isinstance(written, ast.Name) and frame.scope.get(written.id)
    if not value or value.kind != "tile" or (value.row_of or (None,))[0] != field_:
        refuse(
            frame.where,
            node,
            f"a row of self.{field_} takes sums started from one of its rows "
            "(their token orders its reads and writes)",
            f"write acc = f(self.{field_}[i], self.s[r]) above, then "
            f"self.{field_}[j].write(acc)",
        )
    dst_tiles.same_loops(
        frame,
        node,
        written.id,
        value,
        "this write",
        *dst_tiles.once_text(frame, value),
    )
    _, source, token_o = value.row_of
    line = expressions.integer(frame, row)
    items = frame.compiled.fields[field_].shape[0]
    width = value.width
    bound = (
        ""
        if line.rstrip("u").isdigit()
        else (
            f"if (!(({line}) < {items})) {{  // self.{field_}'s rows\n"
            "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
            "    return;\n}\n"
        )
    )
    if line.rstrip("u").isdigit() and not int(line.rstrip("u")) < items:
        refuse(
            frame.where,
            node,
            f"self.{field_} holds {items} rows, 0 .. {items - 1}",
            f"index it below {items}",
        )
    packs = "".join(
        f"    pack_tile<true>({value.dst + t}, {source}, ({line}) * {width} + "
        f"{t});\n"
        for t in range(width)
    )
    text = (
        f"{bound}"
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"if (km_ok && compute.reserve_back({token_o}, 1)) {{  // pack's alone\n"
        f"    KM_PACK_FORMAT({source});\n"
        f"{packs}"
        f"    cb_push_back({token_o}, 1);\n"
        "}\n"
        "tile_regs_release();"
    )
    frame.compiled.includes.add("pack")
    value.kind, value.popped = "written", True
    frame.dst_live = None
    compute = [Line(text, {"km_ok"} | expressions.uses(frame, row))]
    steps.state_packed(frame, compute, *value.state)
    return [], compute


def write_row_value(frame, node, field_, written):
    """`self.part.write(f(row))`, f's result a per-row value (`s.col()`): the
    row's sum passes, then the value's tile packed into the field, one tile of
    compute's own, for a later loop's `.col()` (a write frees the last one)."""
    stored = frame.compiled.fields[field_]
    if (
        frame.compiled.field_users.get(field_, ([], []))[0]
        or field_ in frame.compiled.own
        or math.prod(stored.shape[:-2]) != 1
        or stored.format.dtype != "float32"
    ):
        refuse(
            frame.where,
            node,
            f"self.{field_} takes a row's per-row value: one float32 tile of "
            "compute's own",
            f"declare self.{field_}: km.L1[km.fp32, 1, 32, 32], not filled",
        )
    field_fills.producer(frame, node, field_, "product")
    buffer, held = f"kField{camel(field_)}", f"km_packed_{field_}"
    frame.compiled.summed_fields.add(buffer)
    if field_ not in frame.packed:
        frame.packed[field_] = held
        frame.opening.append(Line(f"bool {held} = false;  // self.{field_} pushed"))
        frame.closing.append(
            Line(f"if ({held}) {{\n    cb_pop_front({buffer}, {buffer}Pages);\n}}")
        )
    frame.compiled.includes.add("pack")
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
    frame.loop_states[-1]["value"] = (buffer, held)
    _, made = functions.tile_call(frame, node, written, None)
    frame.scope.pop("<tile function>", None)
    return [], made


def write_sums(frame, node, pages, field_, pushed=False):
    """`t.tiles[r0:r1, c0:c1].write(self.part)`, self.part a field of sums: its
    tiles copied into DST, packed in t's format a row at a time, each row of the
    rectangle written by mover B; the field's pages freed, the next sum starts.
    `pushed`: a reduce's sum, the field handed on already."""
    rows, columns = expressions.field_sides(frame, node, field_)
    rect = pages.rect
    bounds = pages.bounds or (None, None)
    if rect is not None:  # rows of the rectangle; a side a word: at most the field's
        sides = rect[:2]
    elif pages.count == str(rows * columns):
        sides = (str(rows), columns)
    else:  # one row
        sides = ("1", pages.count)
    fits = all(
        side_fits(side, bound, most)
        for side, bound, most in zip(sides, bounds, (rows, columns), strict=True)
    )
    if not fits:
        refuse(
            frame.where,
            node,
            f"self.{field_} is {rows} x {columns} tiles; write a rectangle of them "
            "(a side a word: the build must know it is at most the field's)",
            f"write {pages.tensor}.{spelling('tiles')}[r:r + {rows}, c:c + "
            f"{columns}].write(self.{field_}), or c:{spelling('min')}(c + "
            f"{columns}, e)",
        )
    written_rows, written_columns = sides
    dtype, page_bytes = handler_buffers.tensor_format(frame, node, pages.tensor)
    name = f"kOutRows{columns}"  # its rows' pages stay whole: one width a buffer
    held = [b.pages[0] for b in frame.compiled.buffers if b.name == name]
    out = handler_buffers.out_buffer(
        frame,
        dtype,
        page_bytes,
        f"line {node.lineno}: {line_text(frame.where, node)}",
        columns,
        name,
    )
    # a row packed while the one before is written (one row: the row alone)
    out.pages = (max(held + [columns * min(rows, 2)]),)
    buffer, sums = f"kField{camel(field_)}", frame.summed[field_]
    room, out_room = sums_room(field_), f"km_room{node.lineno}"
    step = max(w for w in range(1, DST_HALF + 1) if columns % w == 0)
    tile = f"km_tile{node.lineno}"
    copies = "\n".join(
        f"        copy_tile({buffer}, {tile} + {j}, {j});" for j in range(step)
    )
    packs = "\n".join(f"            pack_tile({j}, {out.name});" for j in range(step))
    taken = (
        f"km_ok = km_ok && compute.wait_front({buffer}, {buffer}Pages);\n"
        "if (km_ok) {\n"
        if pushed
        else f"if (km_ok && {sums} > 0) {{\n"
        f"    if ({room}) {{  // pack's: its pages held\n"
        f"        cb_push_back({buffer}, {buffer}Pages);\n"
        "    }\n"
        f"    km_ok = compute.wait_front({buffer}, {buffer}Pages);\n"
        "}\n"
        f"if (km_ok && {sums} > 0) {{\n"
    )
    compute = (
        f"// self.{field_}'s sums, packed again in {dtype}\n"
        f"{taken}"
        f"    KM_UNPACK_FORMAT_A({buffer});\n"
        f"    copy_init({buffer});\n"
        f"    bool {out_room} = true;  // pack's alone: it guards the packs only\n"
        f"    for (uint32_t {tile} = 0; km_ok && "
        f"{tile} < {rows * columns}; {tile} += "
        f"{step}) {{\n"
        "        tile_regs_acquire();\n"
        f"{copies}\n"
        "        tile_regs_commit();\n"
        "        tile_regs_wait();\n"
        f"        {out_room} = {out_room} && compute.reserve_back({out.name}, "
        f"{step});\n"
        f"        if ({out_room}) {{\n"
        f"            KM_PACK_FORMAT({out.name});\n"
        f"{packs}\n"
        f"            cb_push_back({out.name}, {step});\n"
        "        }\n"
        "        tile_regs_release();\n"
        "    }\n"
        f"    cb_pop_front({buffer}, {buffer}Pages);\n"
        "}\n"
        f"{sums} = 0;"
    )
    tensor = f"{camel(pages.tensor)}({pages.tensor})"
    row = f"km_row{node.lineno}"
    row_step = rect[2] if rect is not None else columns
    written = (  # the rectangle's rows of the field's, where they are a word
        ""
        if str(written_rows) == str(rows)
        else f"{row} < ({written_rows}) &&\n        "
    )
    movement = (
        f"for (uint32_t {row} = 0; {row} < {rows}; {row}++) {{\n"
        f"    if (!movement.wait_front({out.name}, {columns}) ||\n"
        f"        {'(' if written else ''}{written}!km::write_pages(movement, "
        f"command, {tensor}, {pages.first} + {row} * {row_step}, "
        f"{written_columns}, get_read_ptr({out.name})){')' if written else ''}) "
        "{\n"
        "        return false;\n"
        "    }\n"
        "    noc_async_writes_flushed();\n"
        f"    cb_pop_front({out.name}, {columns});\n"
        "}"
    )
    if not pushed:  # under compute's condition of the pack: a sum was made
        inner = "\n".join("    " + line for line in movement.split("\n"))
        movement = f"if ({sums} > 0) {{\n{inner}\n}}\n{sums} = 0;"
    frame.compiled.includes |= {"pack", "ask"}
    handler_buffers.moved(frame, node, pages.tensor, "page write")
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    return [Line(movement, range_uses(pages, frame), mover="B")], [
        Line(compute, {"km_ok"})
    ]


def step_on_carried(frame, node, write_call, call, value, carried):
    """`pages.write(f(acc, self.s[r]))`, acc started from a row of a field with
    state r (`acc = g(self.m[i], self.s[r])`): state r is in DST after acc's
    tiles already, so f runs on it there, and acc's write packs it back."""
    function = tile_lookup.tile_function_of(frame, call)
    name, field_, index = steps.step_of_dst(frame, call)
    if carried[0] != f"kField{camel(field_)}" or ast.dump(carried[4]) != ast.dump(
        index
    ):
        refuse(
            frame.where,
            node,
            f"{name} holds a state of its start in DST, and this step takes "
            f"self.{field_}[{ast.unparse(index)}]",
            f"step on the state {name} started with",
        )
    unit, cxx = steps.step_unit(frame, node, function, value.width)
    before = (
        f"// {function.name}'s step on the state in DST {value.width}\n"
        "if (km_ok) {\n"
        f"{functions.step_calls(frame, cxx)}"
        "}"
    )
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    frame.compiled.tile_calls.append(
        (frame.method.name, node.lineno, line_text(frame.where, node), function, cxx)
    )
    write_call.args[0] = ast.Name(id=name, ctx=ast.Load())
    movement, compute = write(frame, node, write_call)
    return movement, [Line(before, {"km_ok"}), *compute]


def write_step_of_dst(frame, node, write_call, call):
    """`pages.write(f(acc, self.s[r]))`, f a step-form function of acc (1 to 3 DST
    tiles) and state r of the fp32 field self.s: the state into the DST tile after
    the row, f, the row written as acc is and the state packed back into page r,
    ordered by a token pack pushes and unpack takes (primed at the handler's start).
    """
    function = tile_lookup.tile_function_of(frame, call)
    name, field_, index = steps.step_of_dst(frame, call)
    value = frame.scope[name]
    width = value.width
    if value.dst != 0 or not 1 <= width <= 3:
        refuse(
            frame.where,
            node,
            f"{name} is {width} tile(s) of DST from tile {value.dst}; a step takes "
            "a row of 1 to 3 tiles from DST 0 and its state in the tile after them",
            "sum the row in km.tile.zeros(km.fp32, n), n at most 3",
        )
    carried = getattr(value, "state", None)
    if carried is not None:
        return step_on_carried(frame, node, write_call, call, value, carried)
    buffer, token, slot, bound = steps.step_state(frame, node, field_, index)
    unit, cxx = steps.step_unit(frame, node, function, width)
    before = (
        f"{bound}"
        f"// {function.name}'s step: self.{field_}[{ast.unparse(index)}] into "
        f"DST {width}\n"
        f"{steps.state_taken(buffer, token, slot, width)}"
        f"{functions.step_calls(frame, cxx)}"
        "}"
    )
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    frame.compiled.tile_calls.append(
        (frame.method.name, node.lineno, line_text(frame.where, node), function, cxx)
    )
    write_call.args[0] = ast.Name(id=name, ctx=ast.Load())
    movement, compute = write(frame, node, write_call)
    steps.state_packed(frame, compute, buffer, token, slot, width, index)
    return movement, [
        Line(before, {"km_ok"} | expressions.uses(frame, index)),
        *compute,
    ]


def side_fits(side, bound, most):
    """Whether a rectangle's side (an int, or C++ text) writes a field's side of
    `most` tiles: the same constant, or a word the build knows is at most it."""
    if isinstance(side, int) or side.isdigit():
        return int(side) == most
    return bound is not None and bound <= most
