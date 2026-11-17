"""A handler's tile function calls: the unit of a call over DST tiles, its rows in
chunks and passes, its results pushed, a step's calls, a timed call, and where a
traced call's operands stand."""

import ast
import inspect
import linecache
from pathlib import Path

from komorebi.dsl.compile import layout, scope, tile_lookup
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.dst import dst_tiles
from komorebi.dsl.compile.dst.dst_code import DST_HALF, pack
from komorebi.dsl.compile.pages import page_streams
from komorebi.dsl.compile.pages.page_streams import out_names
from komorebi.dsl.compile.pages.stamp_pages import SPANS_FIRST
from komorebi.dsl.compile.records import Buffer, Value
from komorebi.dsl.compile.source import (
    WAIT_INPUT,
    check_name,
    function_line,
    line_text,
    refuse,
    source_of,
    wait,
)
from komorebi.dsl.compile.tiles.products import marked_operand, product_refusal
from komorebi.dsl.front import front
from komorebi.dsl.front.names import NAMES, spelling
from komorebi.dsl.front.nodes import Line
from komorebi.dsl.tile import trace, unit_program
from komorebi.dsl.tile.forms.rows import chunk_program, row_program
from komorebi.dsl.tile.ir import (
    A_DTYPES,
    BF16,
    BLOCK_B,
    F32,
    KernelIR,
    PlanError,
    TensorParam,
)
from komorebi.dsl.tile.unit import UnitError

TILE_FORMATS = {"float32": F32, "bfloat16": BF16, **{b: b for b in BLOCK_B}}
SCALAR = "scalar"  # a tile function's argument that is a float word, not a tile


def fields_by_row(frame, node, function, unit, given):
    """A field in the one-tile form is a plain row, read by `.row()` alone, or a
    plain column, read by `.col()` alone (`given`: each field's parameter)."""
    kept = {*unit.row_operands, *unit.col_operands}
    row = {unit.operands[k] for k in kept}
    whole = {o for k, o in enumerate(unit.operands) if k not in kept}
    for field_, parameter in given.items():
        if parameter not in row or parameter in whole:
            refuse(
                frame.where,
                node,
                f"self.{field_} in a call on a stream's tiles is a row (row 0 of "
                f"its tile) read by `{parameter}.row()` alone, or a column (column "
                f"0) read by `{parameter}.col()` alone",
                f"write `{parameter}.row()` or `{parameter}.col()` in "
                f"{function.name}, or call it on a row of a stream of rows",
            )


def held_field(frame, node, field):
    """A field a tile function holds for its loop: filled above in the loop's
    block, waited for once before the blocks (and freed after them)."""
    filled = frame.filled.get(field)
    if field in frame.packed:  # a row's per-row value compute wrote above
        filled = (None, frame.blocks[-2]) if len(frame.blocks) > 1 else None
    if filled is None or len(frame.blocks) < 2 or filled[1] != frame.blocks[-2]:
        refuse(
            frame.where,
            node,
            f"self.{field} is not filled before this loop, in its block: a field "
            "is filled for the loop whose tile function holds it",
            f"write self.{field}.fill(km.{spelling('stream')}(...)) before the " "loop",
        )
    frame.filled[field] = None
    buffer = f"kField{camel(field)}"
    if frame.compiled.fields[field].format.dtype == "float32":
        frame.compiled.unpack_fp32.add(buffer)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    return Line(
        f"if (!compute.wait_front({buffer}, {buffer}Pages)) {{\n    return;\n}}"
    )


def held_not_traced(frame, node, call, function):
    """Refuses a tile function's call on a held stream: its operands are a
    stream loop's tiles, and a held block has no loop of its tiles."""
    for arg in call.args:
        if isinstance(arg, ast.Name) and (
            frame.scope.get(arg.id, Value("")).kind == "held"
        ):
            refuse(
                frame.where,
                node,
                f"{arg.id} is a read stream that bodies take, one block: the tile "
                f"function {function.name} takes a stream loop's tiles",
                f"take {arg.id} in a body, or iterate its pages with "
                f"`for tile in km.{spelling('stream')}(...)`",
            )


def push(frame, node, call):
    target = call.func.value
    if not isinstance(target, ast.Name) or target.id not in frame.scope:
        refuse(frame.where, node, "push into a write stream", "write out.push(tile)")
    stream = frame.scope[target.id]
    if stream.kind != "write":
        refuse(
            frame.where,
            node,
            f"{target.id} is not a write stream",
            "open it with km.Stream(..., write=True)",
        )
    for arg in call.args:  # the item pushed named first, then the stream
        if isinstance(arg, ast.Name) and arg.id in frame.scope:
            scope.whole_tile(frame, node, arg.id, frame.scope[arg.id], "a push")
    scope.whole_tile(frame, node, target.id, stream, "a push")
    pushed = call.args and tile_expression(frame, node, call.args[0])
    if len(call.args) == 1 and pushed:
        call.args[0] = pushed
    if (
        len(call.args) == 1
        and tile_lookup.tile_function_of(frame, call.args[0]) is not None
    ):
        _, made = tile_call(frame, node, call.args[0], None)
        item = frame.scope.pop("<tile function>")
        if item.kind == "chunked":  # a wide row's call packs its own chunks
            return push_chunked(frame, node, stream, target.id, made)
        _, pushed = push_made(frame, node, stream, item, target.id)
        return [], made + pushed
    if len(call.args) == 1 and isinstance(call.args[0], ast.BinOp):
        refuse(
            frame.where,
            node,
            "a tile expression is a tile function's",
            "write it in a @km.tile_function and push its call: "
            "out.push(self.f(tile))",
        )
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
        refuse(frame.where, node, "push one tile of the loop", "write out.push(tile)")
    item = frame.scope.get(call.args[0].id)
    if item is not None and item.kind == "tile" and frame.dst_live == item.cxx:
        return push_made(frame, node, stream, item, target.id)
    if item is not None and item.kind == "result":
        return push_results(frame, node, stream, item, target.id)
    tile_lookup.other_tile_line(frame, node)
    if item is None or item.kind != "streamed" or not item.cxx.startswith("km_half"):
        refuse(
            frame.where,
            node,
            "push a tile of the stream loop",
            "write out.push(tile)",
        )
    if frame.dst_live is not None:
        refuse(frame.where, node, f"{frame.dst_live} holds DST", "write it first")
    if stream.pushed:
        refuse(
            frame.where,
            node,
            "a write stream fed by two loops",
            "open a write stream for each",
        )
    page_streams.loop_fed(frame, node, target.id, stream)
    stream.pushed = True
    number = frame.stream_loops[-1]
    text = (
        "tile_regs_acquire();\n"
        f"copy_tile({item.cxx}, {item.first}, 0);\n"
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"if (km_room{number}) {{\n"
        f"    KM_PACK_FORMAT(km_out{number});\n"
        f"    pack_tile<true>(0, km_out{number}, {item.first});\n"
        "}\n"
        "tile_regs_release();"
    )
    return [], [Line(text)]


def push_chunked(frame, node, stream, name, made):
    """`out.push(f(row, ...))` of a wide row: the call packed its chunks."""
    if stream.pushed:
        refuse(
            frame.where,
            node,
            "a write stream fed by two loops",
            "open a write stream for each",
        )
    page_streams.loop_fed(frame, node, name, stream)
    stream.pushed = True
    return [], made


def push_made(frame, node, stream, item, name):
    """`out.push(t)`, t the tile a body made in DST 0 in a stream loop: packed into
    the write stream at the loop tile's place."""
    if not frame.stream_loops:
        refuse(
            frame.where,
            node,
            "a pushed tile is pushed in a stream loop",
            "push it there",
        )
    if stream.pushed:
        refuse(
            frame.where,
            node,
            "a write stream fed by two loops",
            "open a write stream for each",
        )
    page_streams.loop_fed(frame, node, name, stream)
    stream.pushed = True
    number = frame.stream_loops[-1]
    frame.dst_live, frame.dst_next = None, 0
    item.kind = "pushed"
    packs = [f"    pack_tile<true>(0, km_out{number}, km_tile{number});"]
    if item.batch is not None:  # each call's DST 0 to its tile's place
        slot, count = f"km_slot{number}", item.batch[1]
        packs = [
            f"    for (uint32_t {slot} = 0; "
            f"{slot} < km_batch{number}; {slot}++) {{\n"
            f"        pack_tile<true>({count} * {slot}, km_out{number}, "
            f"km_tile{number} + {slot});\n    }}"
        ]
    if item.width > 1:  # a row: DST 0 .. width - 1 to the row's places
        packs = [
            f"    pack_tile<true>({t}, km_out{number}, km_tile{number} + {t});"
            for t in range(item.width)
        ]
    text = (
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"if (km_room{number}) {{\n    KM_PACK_FORMAT(km_out{number});\n"
        + "\n".join(packs)
        + "\n}\n"
        + dst_tiles.scan_carried(frame, item)  # a scan's state: to the next tile
        + "tile_regs_release();"
    )
    return [], [Line(text, {"km_ok"} if item.scan is not None else set())]


def push_results(frame, node, stream, item, name):
    """`out.push(y)`, y a result of a call of several: its DST tile packed into
    the write stream; the commit, the packs and the release at the last push of
    the call's results."""
    state = frame.loop_states[-1] if frame.loop_states else {}
    group = state.get("results")
    if group is None or item.cxx not in group["names"]:
        refuse(
            frame.where,
            node,
            f"{item.cxx} is a result of a tile function's call, pushed in its loop",
            "push it in the loop of the call",
        )
    if stream.pushed:
        refuse(
            frame.where,
            node,
            "a write stream fed by two loops",
            "open a write stream for each",
        )
    page_streams.loop_fed(frame, node, name, stream)
    stream.pushed = True
    state.setdefault("own_out", set()).add(stream.cxx)
    number = frame.stream_loops[-1]
    out, room = out_names(number, stream, state)
    if group["wide"] is not None:  # a wide row's: its chunks packed in its loop
        group["packs"][item.cxx] = (out, room)
        item.kind = "pushed"
        if len(group["packs"]) < len(group["names"]):
            return [], []
        return [], [Line(group["wide"]([group["packs"][n] for n in group["names"]]))]
    pack = f"    pack_tile<true>({item.dst}, {out}, km_tile{number});"
    if item.batch is not None:  # each call's DST k to its tile's place
        slot, count = f"km_slot{number}", item.batch[1]
        pack = (
            f"    for (uint32_t {slot} = 0; {slot} < km_batch{number}; {slot}++) "
            f"{{\n        pack_tile<true>({count} * {slot} + {item.dst}, {out}, "
            f"km_tile{number} + {slot});\n    }}"
        )
    group["packs"][item.cxx] = f"if ({room}) {{\n    KM_PACK_FORMAT({out});\n{pack}\n}}"
    item.kind = "pushed"
    if len(group["packs"]) < len(group["names"]):
        return [], []
    packs = [group["packs"][n] for n in group["names"]]
    frame.dst_live, frame.dst_next = None, 0
    text = "\n".join(["tile_regs_commit();", "tile_regs_wait();", *packs])
    return [], [Line(text + "\ntile_regs_release();")]


def results_named(frame, node, function, unit, name):
    """A call of a function of several results names each (`a, b = f(...)`)."""
    count = len(name) if isinstance(name, list) else 1
    if count != unit.results:
        refuse(
            frame.where,
            node,
            f"the tile function {function.name} returns {unit.results} tile(s), "
            f"and the line takes {count}",
            "name each result: `a, b = f(...)`, then push each",
        )


def row_chunks(frame, node, function, unit, cxx, item, fields, name):
    """A pushed call on a row of more than 2 tiles: one acquire for the sums'
    passes over its chunks of 2 (DST 0, 1) and the per-row values (DST 2, packed
    to a page of their own); then an acquire a chunk: the chunk, those values and
    each field's chunk (DST 4 + k) in, the body, the chunk packed in place."""
    if name is not None and not isinstance(name, list):
        refuse(
            frame.where,
            node,
            f"a call of {function.name} on a row of {item.width} tiles writes the "
            "row in chunks, pushed as it goes",
            f"push the call: `out.push({function.name}(row, ...))`",
        )
    if frame.method.stamps is not None and not frame.method.overlay:
        refuse(
            frame.where,
            node,
            "stamps time one call a tile or a row, and a wide row's call is a "
            "call a chunk",
            "time it without stamps=, or on rows of 2 tiles",
        )
    rows = chunk_rows_buffer(frame.compiled, cxx, node)
    number = frame.stream_loops[-1]
    chunk, tile, first = f"km_chunk{number}", f"km_tile{number}", item.first
    count = item.width // 2
    loop = f"for (uint32_t {chunk} = 0; {chunk} < {count}; {chunk}++) {{"
    copy = [
        f"    copy_tile({item.cxx}, {first} + 2 * {chunk}, 0);",
        f"    copy_tile({item.cxx}, {first} + 2 * {chunk} + 1, 1);",
    ]
    lines = ["tile_regs_acquire();", f"MATH((km_tile::{cxx}_start(0)));"]
    last = f"copy_tile({item.cxx}, {first} + {item.width - 1}, 0);"  # an odd row's
    back = f"{item.buffer.name}.mover_a"

    def sum_fields(index):
        return sum_field_copies(unit, fields, back, index)

    for part in unit.parts:
        if part.startswith("sum") and part.endswith("_one"):
            lines += [last, *sum_fields(count), f"MATH((km_tile::{cxx}_{part}(0)));"]
        elif part.startswith("sum"):
            call = f"    MATH((km_tile::{cxx}_{part}(0)));"
            lines += [loop, *copy, *("    " + x for x in sum_fields(chunk)), call, "}"]
        elif part.startswith("finish"):
            lines.append(f"MATH((km_tile::{cxx}_{part}(0)));")
    value = frame.loop_states[-1].get("value")
    if value is not None and "value" not in unit.parts:
        refuse(
            frame.where,
            node,
            f"{function.name}'s result reads the row; a field takes a per-row "
            "value alone",
            f"return the value broadcast: `return s.col()` in {function.name}",
        )
    if value is not None:  # the per-row value's tile into a field, no writes
        frame.compiled.tile_calls.append(
            (
                frame.method.name,
                node.lineno,
                line_text(frame.where, node),
                function,
                cxx,
            )
        )
        return [], [Line("\n".join([*lines, *value_packed(cxx, *value)]))]
    head = lines
    lines = [
        "tile_regs_commit();",
        "tile_regs_wait();",
        f"cb_reserve_back({rows}, 1);",
        f"KM_PACK_FORMAT({rows});",
        f"pack_tile(2, {rows});",
        f"cb_push_back({rows}, 1);",
        "tile_regs_release();",
        f"cb_wait_front({rows}, 1);",
        loop,
        "    tile_regs_acquire();",
        *copy,
        f"    KM_UNPACK_FORMAT_A({rows});",
        f"    copy_init({rows});",
        f"    copy_tile({rows}, 0, 2);",
        *[f"    {x}" for x in field_copies(frame.compiled, fields, rows, chunk)],
        f"    KM_UNPACK_FORMAT_A({back});",
        f"    copy_init({back});",
        f"    MATH((km_tile::{cxx}(0)));",
        "    tile_regs_commit();",
        "    tile_regs_wait();",
        "@packs",
        "    tile_regs_release();",
        "}",
    ]
    body_ = lines

    def text(outs):  # the call's lines, each result packed into its stream
        lines = [*head]
        for x in body_:
            lines += (
                chunk_packs(unit, outs, tile, chunk, "    ") if x == "@packs" else [x]
            )
        if item.width % 2:  # the last tile: a chunk of one
            at = lines.index(loop, lines.index(f"cb_wait_front({rows}, 1);"))
            body = [x[4:] for x in lines[at + 1 : -1] if "+ 1" not in x[4:]]
            body = [
                x.replace(f"{first} + 2 * {chunk}", f"{first} + {item.width - 1}")
                .replace(f"{tile} + 2 * {chunk}", f"{tile} + {item.width - 1}")
                .replace(f", {chunk}, ", f", {count}, ")
                .replace(f"km_tile::{cxx}(0)", f"km_tile::{cxx}_apply_one(0)")
                for x in body
            ]
            lines += body
        lines.append(f"cb_pop_front({rows}, 1);")
        return "\n".join(lines)

    frame.compiled.tile_calls.append(
        (frame.method.name, node.lineno, line_text(frame.where, node), function, cxx)
    )
    if isinstance(name, list):  # several results: packed at the last push
        frame.loop_states[-1]["wide"] = text
        frame.scope["<results>"] = Value("tile", "<results>", width=item.width)
        return [], []
    frame.scope["<tile function>"] = Value("chunked", cxx, width=item.width)
    return [], [Line(text([(f"km_out{number}", f"km_room{number}")]))]


def row_passes(frame, node, function, unit, cxx, item, fields, name):
    """A pushed call on a row longer than its block: the range read once a sum
    level, then for the writes. A sum pass holds DST a row, adds its chunks (DST
    0, 1) and packs the values' and sums' tiles (DST 2, 3) to a page pair of the
    row's, unpacked at the next pass; the last writes as on a row in one block."""
    state = frame.loop_states[-1]
    block, count = state["block"], known_count(state["count"])
    if count is None and state.get("most"):  # a word: the per-row pages' most
        count = state["most"]
    refused = (
        "a wide row read in parts is whole chunks of 2 a block, and its rows "
        "a constant count or a stated most"
    )
    if block % 2 or count is None or count % item.width:
        refuse(
            frame.where,
            node,
            refused,
            "give block= an even part of the row, and the rows a count or a most: "
            f"x.{spelling('tiles')}[r:min(r + MOST, r + rows), 0:W]",
        )
    row_chunks(frame, node, function, unit, cxx, item, fields, name)  # its checks
    number = frame.stream_loops[-1]
    levels = sum(1 for p in unit.parts if p.startswith("sum"))
    pair = 2 if levels > 1 else 1  # one sum pass: the values' tile alone a row
    rows = chunk_rows_buffer(frame.compiled, cxx, node, pair * count // item.width)
    value = state.get("value")  # a per-row value into a field: no write pass
    state["passes"] = levels + (value is None)
    at, part = f"km_pass{number}", f"km_part{number}"
    per_turn = f"km_blocks{number}"
    chunk, tile, first = f"km_chunk{number}", f"km_tile{number}", item.first
    loop = f"for (uint32_t {chunk} = 0; {chunk} < {block // 2}; {chunk}++) {{"
    copy = [
        f"    copy_tile({item.cxx}, {first} + 2 * {chunk}, 0);",
        f"    copy_tile({item.cxx}, {first} + 2 * {chunk} + 1, 1);",
    ]
    back = f"{item.buffer.name}.mover_a"
    last = item.width // block - 1

    def by_pass(name_):  # the level's part of the pass this block is in
        return " else ".join(
            f"if ({at} == {k}) {{\n    MATH((km_tile::{cxx}_{name_}{k}(0)));\n}}"
            for k in range(levels)
        ).split("\n")

    lines = [
        f"const uint32_t {at} = km_block{number} / {per_turn};",
        f"const uint32_t {part} = km_block{number} % {per_turn} % {last + 1};",
        f"if ({at} < {levels}) {{",
        f"    if ({part} == 0) {{",
        "        tile_regs_acquire();",
        f"        if ({at} == 0) {{",
        f"            MATH((km_tile::{cxx}_start(0)));",
        "        } else {",
        f"            cb_wait_front({rows}, 2);",
        f"            KM_UNPACK_FORMAT_A({rows});",
        f"            copy_init({rows});",
        f"            copy_tile({rows}, 0, 2);",
        f"            copy_tile({rows}, 1, 3);",
        f"            cb_pop_front({rows}, 2);",
        f"            KM_UNPACK_FORMAT_A({back});",
        f"            copy_init({back});",
        "        }",
        "    }",
        f"    {loop}",
        *[f"    {x}" for x in copy],
        *[
            f"        {x}"
            for x in sum_field_copies(
                unit, fields, back, f"{part} * {block // 2} + {chunk}"
            )
        ],
        *[f"        {x}" for x in by_pass("sum")],
        "    }",
        f"    if ({part} == {last}) {{",
        *[f"        {x}" for x in by_pass("finish")],
        *(
            []
            if value is None
            else [
                f"        if ({at} == {levels - 1}) {{",
                *[f"            {x}" for x in value_packed(cxx, *value)],
                "        } else {",
            ]
        ),
        "        tile_regs_commit();",
        "        tile_regs_wait();",
        f"        cb_reserve_back({rows}, {pair});",
        f"        KM_PACK_FORMAT({rows});",
        f"        pack_tile(2, {rows});",
        *([f"        pack_tile(3, {rows});"] if pair == 2 else []),
        f"        cb_push_back({rows}, {pair});",
        "        tile_regs_release();",
        *([] if value is None else ["        }"]),
        "    }",
        "} else {",
        f"    if ({part} == 0) {{",
        f"        cb_wait_front({rows}, {pair});",
        "    }",
        f"    {loop}",
        "        tile_regs_acquire();",
        *[f"    {x}" for x in copy],
        f"        KM_UNPACK_FORMAT_A({rows});",
        f"        copy_init({rows});",
        f"        copy_tile({rows}, 0, 2);",
        *[
            f"        {x}"
            for x in field_copies(
                frame.compiled, fields, rows, f"{part} * {block // 2} + {chunk}"
            )
        ],
        f"        KM_UNPACK_FORMAT_A({back});",
        f"        copy_init({back});",
        f"        MATH((km_tile::{cxx}(0)));",
        "        tile_regs_commit();",
        "        tile_regs_wait();",
        "@packs",
        "        tile_regs_release();",
        "    }",
        f"    if ({part} == {last}) {{",
        f"        cb_pop_front({rows}, {pair});",
        "    }",
        "}",
    ]
    if value is not None:  # no write pass: the else above never runs
        lines = lines[: lines.index("} else {")] + ["}"]

    def text(outs):  # each result packed into its stream in the last pass
        packs = chunk_packs(unit, outs, tile, chunk, "        ")
        return "\n".join(y for x in lines for y in (packs if x == "@packs" else [x]))

    if isinstance(name, list):  # several results: packed at the last push
        state["wide"] = text
        return [], []
    return [], [Line(text([(f"km_out{number}", f"km_room{number}")]))]


def scalar_argument(frame, node, arg):
    """Whether a tile function's argument is a float word (a run-time scalar);
    an int word is refused, naming the float annotation."""
    value = isinstance(arg, ast.Name) and frame.scope.get(arg.id)
    if value and value.kind == "int":
        refuse(
            frame.where,
            node,
            f"`{arg.id}` is an int word, and a tile function's run-time scalar is "
            "a float word (a float32's bits, loaded by the unit program's setup)",
            f"annotate it `{arg.id}: float` and pass the number",
        )
    return bool(value) and value.kind == "float"


def step_calls(frame, cxx):
    """A step's setup and body on the math CPU, each line indented once; with
    stamps (a resident handler), each timed by timed_call: two spans a step,
    its setup's and its body's."""
    timed = frame.method.stamps is not None and not frame.method.overlay
    lines = [
        *timed_call(frame, [f"MATH((km_tile::{cxx}_init()));"], timed),
        *timed_call(frame, [f"MATH((km_tile::{cxx}(0)));"], timed),
    ]
    if timed:  # each span's start in its own scope
        lines = [
            *(["{", *("    " + x for x in lines[:7]), "}"]),
            *(["{", *("    " + x for x in lines[7:]), "}"]),
        ]
    return "".join(f"    {line}\n" for line in lines)


def tile_call(frame, node, call, name):
    """`t = self.f(a, b)` (or pushed): a, b the loop's stream tiles (or a row of
    `row=` tiles and fields held as resident rows), copied into DST as f's unit
    program reads them, the emitted body on the math CPU, the result in DST 0 (a
    row: 0 .. width - 1). Returns (no movement, compute's lines)."""
    function = tile_lookup.tile_function_of(frame, call)
    held_not_traced(frame, node, call, function)
    traced = traced_placement(frame, node, function)
    if not frame.loop_states:
        refuse(
            frame.where,
            node,
            f"the tile function {function.name} is called on a stream's tiles, "
            "in the stream's loop",
            f"call it inside `for tile in km.{spelling('stream')}(...)`",
        )
    state = frame.loop_states[-1]
    if state["other"] is not None:
        refuse(
            frame.where,
            node,
            f"this loop has other tile lines (line {state['other'].lineno}), and "
            "a tile function's setup holds the vector unit for the loop",
            "write the loop's tile arithmetic in the tile function",
        )
    if call.keywords or not call.args:
        refuse(frame.where, node, "a tile function takes tiles", "call f(tile, ...)")
    if function.fidelity is not None and matrix_product(function.function) is None:
        refuse(
            frame.where,
            node,
            f"fidelity= is a product's (the matrix unit's), and {function.name} "
            "is the vector unit's",
            f"drop fidelity= from {function.name}",
        )
    operands = []  # (Value of a stream tile, or None, field name or None)
    scalars = {}  # {the function's parameter: the float word passed for it}
    names = list(inspect.signature(function.function).parameters)
    scanned = dst_tiles.scan_state(frame, node, call, name, state)
    for arg in call.args:
        if scanned is not None and isinstance(arg, ast.Name) and arg.id == name:
            operands.append((scanned, None))  # the scan's state, from its carry
            continue
        field = scope.field_name(frame, arg)
        if field is not None:
            operands.append((None, field))
            continue
        if scalar_argument(frame, node, arg):
            scalars[names[len(operands)]] = arg.id
            operands.append((None, None))
            continue
        item = isinstance(arg, ast.Name) and frame.scope.get(arg.id)
        if item:
            scope.whole_tile(frame, node, arg.id, item, "a tile function")
        if not item or item.kind != "streamed" or not item.cxx.startswith("km_half"):
            refuse(
                frame.where,
                node,
                f"a tile function takes the tiles of the loop's streams, and "
                f"`{ast.unparse(arg)}` is not one",
                "pass the loop's tiles: f(a, b) in `for a, b in "
                f"{spelling('zip')}(km.{spelling('stream')}(...), ...)`",
            )
        operands.append((item, None))
    streamed = [item for item, _ in operands if item not in (None, scanned)]
    width = max(item.width for item in streamed) if streamed else 1
    fields = [field for _, field in operands if field is not None]
    if scanned is not None and (width != 1 or fields or not streamed):
        refuse(
            frame.where,
            node,
            f"{name} is a scan's state, stepped a tile of the loop: its function "
            "takes it and the loop's tiles of one tile each, no field",
            f"call {function.name} on {name} and tiles of a stream of tiles",
        )
    if frame.dst_live is not None or frame.dst_next != 0:
        refuse(
            frame.where,
            node,
            "a tile function's result is DST 0, which another tile holds",
            "write or push that tile first",
        )
    if scalars and width > 1:
        refuse(
            frame.where,
            node,
            f"{', '.join(scalars.values())}: a run-time scalar of a tile function "
            "called on a row: the row form's numbers are immediates",
            "call it on a stream's tiles, one tile a side",
        )
    formats = []
    for item, field in operands:
        if item is not None:
            formats.append(item.buffer.dtype)
        elif field is not None:
            formats.append(frame.compiled.fields[field].format.dtype)
        else:
            formats.append(SCALAR)
    why = tile_format_refusal([f for f in formats if f != SCALAR])
    if why is not None:
        refuse(frame.where, node, f"{function.name}: {why}", "give it tiles of those")
    unit, cxx = tile_unit(frame, node, function, formats, width, fields)
    if scanned is not None and (unit.products or unit.results > 1):
        refuse(
            frame.where,
            node,
            f"{name} is a scan's state, and {function.name} has "
            + ("a matrix product (`@`)" if unit.products else "several results"),
            f"step {name} with a function of the vector unit and one result",
        )
    if width == 1 and fields:  # each field's parameter
        given = {f: names[i] for i, (_, f) in enumerate(operands) if f is not None}
        fields_by_row(frame, node, function, unit, given)
    if unit.row_operands or unit.col_operands:  # unpacked broadcast (unary_bcast)
        frame.compiled.includes.add("bcast")
    if state["function"] not in (None, cxx):
        refuse(
            frame.where,
            node,
            f"this loop calls the tile function {state['function']} already; a "
            "loop calls one",
            "make one tile function of the two",
        )
    parameters = list(inspect.signature(function.function).parameters)
    products = [  # each product's operands: (a's tile, w's tile)
        (operands[parameters.index(p.a)][0], operands[parameters.index(p.w)][0])
        for p in unit.products
    ]
    for (a, w), p in zip(products, unit.products):
        why = product_refusal(a.buffer.dtype, w.buffer.dtype)
        if why is not None:
            refuse(
                frame.where,
                node,
                f"{why}, in {function.name}",
                "stream tensors of matrix formats of one family into it",
            )
        frame.compiled.placements.append(
            (
                frame.method.name,
                node.lineno,
                line_text(frame.where, node),
                a.buffer.dtype,
                w.buffer.dtype,
                traced.get((p.a, p.w), "none"),
                None,
            )
        )
    if products:
        fidelity = unit.fidelity
        if frame.compiled.fidelity not in (None, fidelity):
            refuse(
                frame.where,
                node,
                f"{function.name}'s fidelity is {fidelity} and another product's "
                f"{frame.compiled.fidelity}: the fidelity is the compute kernel's",
                "give the class's products one fidelity",
            )
        frame.compiled.fidelity = fidelity
        frame.compiled.includes.add("matmul_custom")
    if state["function"] is None:
        state["function"] = cxx
        for field in fields:
            state["inits"].append(held_field(frame, node, field))
            buffer = f"kField{camel(field)}"
            if field not in frame.packed:  # compute's own: its next write frees it
                state["after"].append(Line(f"cb_pop_front({buffer}, {buffer}Pages);"))
        init = []
        if any(operand is not None for operand in unit.operands):
            init += [
                f"KM_UNPACK_FORMAT_A({streamed[0].buffer.name}.mover_a);",
                f"copy_init({streamed[0].buffer.name}.mover_a);",
            ]
        if products:  # the first product's buffers: a into SrcB, w into SrcA
            a, w = (item.buffer.name + ".mover_a" for item in products[0])
            init += [
                f"KM_UNPACK_FORMATS({w}, {a});",
                f"mm_no_mop_init_short({a}, {w}, false, 1, 1, 1);",
            ]
        init.append(f"MATH((km_tile::{cxx}_init()));  // {function.name}'s setup")
        state["inits"].append(Line("\n".join(init)))
    timed = frame.method.stamps is not None and not frame.method.overlay
    results_named(frame, node, function, unit, name)
    count = max(len(unit.operands), unit.results)  # DST tiles a call
    slots = DST_HALF // count if width == 1 and scanned is None else 1
    copied = [
        operands[parameters.index(operand)][0]
        for operand in unit.operands
        if operand is not None and operands[parameters.index(operand)][0]
    ]
    switch = SrcaSwitch(products, copied)
    number = frame.stream_loops[-1]
    if width > 2 and state["block"] < width:  # the row read in passes
        return row_passes(frame, node, function, unit, cxx, streamed[0], fields, name)
    if width > 2:
        return row_chunks(frame, node, function, unit, cxx, streamed[0], fields, name)
    lines = ["tile_regs_acquire();"]
    if slots > 1:  # the calls of `slots` tiles an acquire, each its own DST tiles
        state["step"] = slots
        batch, slot = f"km_batch{number}", f"km_slot{number}"
        left = f"km_valid{number} - km_tile{number}"
        lines = [
            f"const uint32_t {batch} = {left} < {slots} ? {left} : {slots};",
            "tile_regs_acquire();",
            f"for (uint32_t {slot} = 0; {slot} < {batch}; {slot}++) {{",
        ]
        for a, w in products:  # unpacked in the unit's issue order
            lines += [f"    {line}" for line in switch.product(a, w)]
            lines.append(
                f"    UNPACK((llk_unpack_AB_matmul({a.cxx}, {w.cxx}, {a.first} + "
                f"{slot}, {w.first} + {slot}, 1, 1, 1)));"
            )
        for dst, operand in enumerate(unit.operands):
            if operand is None:  # a product's Dst tile
                continue
            item, field = operands[parameters.index(operand)]
            if dst in unit.row_operands or dst in unit.col_operands:
                lines += [
                    f"    {line}"
                    for line in row_broadcast(
                        item,
                        field,
                        slot,
                        f"{count} * {slot} + {dst}",
                        copied,
                        "COL" if dst in unit.col_operands else "ROW",
                    )
                ]
                continue
            lines += [f"    {line}" for line in switch.copy(item)]
            lines.append(
                f"    copy_tile({item.cxx}, {item.first} + {slot}, "
                f"{count} * {slot} + {dst});"
            )
        words = scalar_words(unit, scalars)
        call_ = [f"MATH((km_tile::{cxx}({count} * {slot}{words})));"]
        lines += ["    " + line for line in timed_call(frame, call_, timed)] + ["}"]
    elif width == 1:
        for a, w in products:  # unpacked in the unit's issue order
            lines += switch.product(a, w)
            lines.append(
                f"UNPACK((llk_unpack_AB_matmul({a.cxx}, {w.cxx}, {a.first}, "
                f"{w.first}, 1, 1, 1)));"
            )
        for dst, operand in enumerate(unit.operands):
            if operand is None:  # a product's Dst tile
                continue
            item, field = operands[parameters.index(operand)]
            if dst in unit.row_operands or dst in unit.col_operands:
                along = "COL" if dst in unit.col_operands else "ROW"
                lines += row_broadcast(item, field, None, dst, copied, along)
                continue
            if item is scanned and item is not None:
                lines += dst_tiles.scan_operand(
                    state["scan"], item, dst, streamed[0].buffer.name
                )
                continue
            lines += switch.copy(item)
            lines.append(f"copy_tile({item.cxx}, {item.first}, {dst});")
    else:  # the row form: the row in DST 0 .. width - 1, the field's tile after
        item = streamed[0]
        lines += [
            f"copy_tile({item.cxx}, {item.first} + {t}, {t});" for t in range(width)
        ]
        for k, field in enumerate(fields):  # resident row k: DST 2 width + k
            lines.append(f"copy_tile(kField{camel(field)}, 0, {2 * width + k});")
    if slots == 1:
        words = scalar_words(unit, scalars)
        lines += timed_call(frame, [f"MATH((km_tile::{cxx}(0{words})));"], timed)
    frame.compiled.tile_calls.append(
        (frame.method.name, node.lineno, line_text(frame.where, node), function, cxx)
    )
    frame.dst_next = 1
    held = "<results>" if isinstance(name, list) else name or "<tile function>"
    frame.dst_live = held
    frame.scope[held] = Value(
        "tile",
        held,
        dst=0,
        width=width,
        batch=(number, count) if slots > 1 else None,
        scan=None if scanned is None else scanned.buffer,
    )
    return [], [Line("\n".join(lines), uses=set(scalars.values()))]


def tile_expression(frame, node, value):
    """`out.push(x * y)`, `t = km.silu(x) * y`: an expression of the loop's tiles,
    numbers and km's tile arithmetic, made a tile function of those tiles (its
    parameters in the order the expression names them) and returned as that
    function's call; None when `value` names no tile of the loop."""
    value, inner = _composed(frame, value)
    if not isinstance(value, (ast.BinOp, ast.UnaryOp, ast.Call)):
        return None
    if tile_lookup.tile_function_of(frame, value) is not None and not inner:
        return None
    arithmetic = {spelling(k) for k in NAMES if k.startswith("math_")}
    for call in ast.walk(value):
        if isinstance(call, ast.Call) and not (
            (isinstance(call.func, ast.Name) and call.func.id == "abs")
            or (isinstance(call.func, ast.Name) and call.func.id in inner)
            or (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "km"
                and call.func.attr in arithmetic
            )
        ):
            return None  # a call of no tile arithmetic: not a tile expression
    names = [
        n for n in ast.walk(value) if isinstance(n, ast.Name) and n.id not in inner
    ]
    tiles = []  # the function's parameters: the loop's tiles and float words
    for n in names:
        kind = frame.scope.get(n.id, Value("")).kind
        if kind in ("streamed", "float") and n.id not in tiles:
            tiles.append(n.id)
    if not any(frame.scope[t].kind == "streamed" for t in tiles):
        return None
    for n in names:
        kind = frame.scope.get(n.id, Value("")).kind if n.id in frame.scope else None
        if kind == "int":
            refuse(
                frame.where,
                node,
                f"`{n.id}` is an int word, and a tile expression's run-time "
                "scalar is a float word (a float32's bits, loaded by the unit "
                "program's setup)",
                f"annotate it `{n.id}: float` and pass the number",
            )
        if n.id == "self" or kind not in (None, "streamed", "float"):
            what = "self" if n.id == "self" else f"`{n.id}` ({kind})"
            refuse(
                frame.where,
                node,
                f"a tile expression reads the loop's tiles, numbers, module "
                f"constants and km's tile arithmetic, and {what} is none of them: "
                "the unit program's numbers are immediates built into its code",
                "write a number, a module constant or a float word there",
            )
    text = ast.unparse(value)
    name = f"tile_line_{node.lineno}"
    filename = (
        f"<the tile expression of {Path(frame.where.file).name}, "
        f"line {node.lineno}>"
    )
    source = f"def {name}({', '.join(tiles)}): return {text}\n"
    lines = ["\n"] * (node.lineno - 1) + [source]
    linecache.cache[filename] = (len(source), None, lines, filename)
    names_of = {**frame.names, **inner}
    exec(compile("".join(lines), filename, "exec"), names_of)
    function = front.TileFunction(names_of[name])
    call = ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=[ast.Name(id=t, ctx=ast.Load()) for t in tiles],
        keywords=[],
    )
    ast.copy_location(call, value)
    frame.expression_calls[id(call)] = (call, function)
    return call


def _composed(frame, value):
    """`value` with each composed result's name replaced by its expression and each
    tile function it calls bound to a name of its own (`self.f(...)` too), and those
    names' tile functions; `value` and {} when it reads no composed result and calls
    no tile function inside another call or an expression."""
    if not isinstance(value, ast.AST):
        return value, {}
    held = [
        n
        for n in ast.walk(value)
        if isinstance(n, ast.Name)
        and frame.scope.get(n.id, Value("")).kind == "composed"
    ]
    calls = [
        n
        for n in ast.walk(value)
        if isinstance(n, ast.Call) and tile_lookup.tile_function_of(frame, n)
    ]
    if not held and not any(call is not value for call in calls):
        return value, {}
    inner = {}

    class Composed(ast.NodeTransformer):
        def visit_Name(self, name):
            result = frame.scope.get(name.id)
            if result is None or result.kind != "composed":
                return name
            return self.visit(_copied(result.expression, name))

        def visit_Call(self, call):
            function = tile_lookup.tile_function_of(frame, call)
            self.generic_visit(call)
            if isinstance(function, front.TileFunction):
                named = f"km_tile_{function.name}"
                inner[named] = function
                call.func = ast.copy_location(ast.Name(id=named, ctx=ast.Load()), call)
            return call

    return Composed().visit(_copied(value, value)), inner


def _copied(expression, at):
    """A copy of `expression` placed at `at`'s line."""
    copied = ast.parse(ast.unparse(expression), mode="eval").body
    for node in ast.walk(copied):
        ast.copy_location(node, at)
    return copied


def composed_result(frame, node, name, value):
    """`t = f(a, b)` (or a tile expression) whose name later tile lines of its loop
    read as an operand and never push: no code here, the name held as its expression
    and inlined where it is read (one unit program). False for any other line."""
    state = frame.loop_states[-1] if frame.loop_states else None
    body = state.get("body") if state else None
    if not body or not any(line is node for line in body):
        return False
    if not isinstance(value, (ast.BinOp, ast.UnaryOp, ast.Call)):
        return False
    tile = tile_lookup.tile_function_of(frame, value) is not None
    if not tile and not any(
        isinstance(n, ast.Name) and frame.scope.get(n.id, Value("")).kind == "streamed"
        for n in ast.walk(value)
    ):
        return False
    later = body[[id(line) for line in body].index(id(node)) + 1 :]
    bare = read = 0
    for line in later:
        for n in ast.walk(line):
            if isinstance(n, ast.Name) and n.id == name:
                read += 1
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in (spelling("push"), spelling("write"))
                and len(n.args) == 1
                and isinstance(n.args[0], ast.Name)
                and n.args[0].id == name
            ):
                bare += 1
    if read == bare:  # pushed or written as it is, or unread: a call of its own
        return False
    if bare:
        refuse(
            frame.where,
            node,
            f"{name} is pushed and read by a later tile line: a loop calls one tile "
            "function, and a result read by another is composed into it",
            f"push the later line's result only, or compute {name} again there",
        )
    if tile_expression(frame, node, value) is None and not tile:
        return False
    frame.scope[name] = Value("composed", name, expression=value)
    return True


def tile_results(frame, node, targets, call):
    """`s, y = self.f(a, b)`, f returning several tiles: the call as `tile_call`
    makes it, result k in DST k of the call, each name pushed on its own."""
    names = [t.id if isinstance(t, ast.Name) else None for t in targets.elts]
    if None in names:
        refuse(
            frame.where,
            node,
            "a tile function's results are names",
            "write a, b = ...",
        )
    for name in names:
        check_name(frame.where, node, name)
        scope.not_parameter(frame, node, name)
    _, made = tile_call(frame, node, call, names)
    group = frame.scope.pop("<results>")
    state = frame.loop_states[-1]
    state["results"] = {"names": names, "packs": {}, "line": node.lineno}
    state["results"]["wide"] = state.pop("wide", None)  # a wide row's lines
    for k, name in enumerate(names):
        frame.scope[name] = Value("result", name, dst=k, batch=group.batch)
    return [], made


def tile_unit(frame, node, function, formats, width=1, fields=(), tiles=None):
    """The unit program of `function` on tiles of `formats` (traced, its rules
    applied, emitted at the write; `width` > 1: the row form over rows of that
    many tiles, `fields` its resident rows) and its C++ name; refused by name and
    line when the tile compiler does not lower it."""
    made = (function.function, tuple(formats), width)
    if tiles is not None:  # operands already in Dst, apart (a unit of its own)
        made += (tuple(sorted(tiles.items())),)
    for cxx, (key, unit) in frame.compiled.units.items():
        if key == made:
            return unit, cxx
    where = function_line(function)
    parameters = list(inspect.signature(function.function).parameters)
    if len(parameters) != len(formats):
        refuse(
            frame.where,
            node,
            f"the tile function {function.name} ({where}) takes "
            f"{len(parameters)} tiles, {len(formats)} given",
            "pass one tile a parameter",
        )
    product = matrix_product(function.function)
    if product is not None and width > 1:
        refuse(
            frame.where,
            node,
            f"the tile function {function.name}, line {product}, has a matrix "
            "product (`@`) and is called on a row: a product of one K step of "
            "32 x 32 tiles a side is lowered, a K loop over a row is not",
            "call it on a stream's tiles, one tile a side, or make the row's "
            "product with `acc += self.f(block, self.w)`",
        )
    cxx = function.name
    while cxx in frame.compiled.units:
        cxx += "_"
    dest = tile_dest(formats)
    if dest != "fp32" and width > 1:
        refuse(
            frame.where,
            node,
            f"the tile function {function.name} ({where}) takes tiles of "
            f"{', '.join(sorted(set(formats)))} (the A family, a 16-bit Dst) and "
            "is called on a row: the row form keeps each row's value and its sums "
            "in Dst between its levels, which a 16-bit Dst would cut to fp16",
            "call it on a stream's tiles, one tile a side, or stream tensors of "
            "the B family",
        )
    kinds = {**TILE_FORMATS, **{f: f for f in A_DTYPES}}
    tensors = [
        TensorParam(p, False, kinds[f])
        for p, f in zip(parameters, formats)
        if f != SCALAR
    ]
    values = [  # the run-time scalars: float words of the call
        trace.ValueParam(p, "float") for p, f in zip(parameters, formats) if f == SCALAR
    ]
    result = TensorParam("result", True, F32)
    ir = KernelIR(cxx, [*tensors, result], values, {})
    held = len(parameters) - len(fields)  # the fields are the last parameters
    try:
        with trace.Tracer(ir):
            proxies = [trace.TensorProxy(t) for t in tensors]
            out = trace.TensorProxy(result)
            if width == 1:
                for t in trace.grid(trace.num_tiles(proxies[0])):
                    loads = iter([trace.load(p.tiles[t]) for p in proxies])
                    given = iter(values)
                    args = [next(given if f == SCALAR else loads) for f in formats]
                    stored = function.function(*args)
                    if isinstance(stored, tuple):  # its results: DST 0, 1, ...
                        traced_results(ir, out, t, stored)
                    else:
                        trace.store(out.tiles[t], stored)
            else:
                rows = [trace.resident(p.tiles[0, :]) for p in proxies[held:]]
                for t in trace.grid(trace.num_tile_rows(proxies[0])):
                    loads = [trace.load(p.tiles[t, :]) for p in proxies[:held]]
                    value = function.function(*loads, *rows)
                    if isinstance(value, tuple) and width > 2:  # chunk form's
                        traced_results(ir, out, (t, slice(None)), value)
                        continue
                    if isinstance(value, tuple):
                        raise PlanError(
                            f"{len(value)} results; a row of 2 tiles stores one "
                            "(several results are the one-tile form's and a "
                            "wide row's)"
                        )
                    trace.store(out.tiles[t, :], value)
        if width == 1:
            fidelity = function.fidelity or "HiFi4"  # its products'
            unit = unit_program(
                ir, dest, load_macro=True, fidelity=fidelity, tiles=tiles
            )
        else:
            unit = (  # a wide row: in chunks of 2 tiles (row_chunks)
                chunk_program(ir, width)
                if width > 2
                else row_program(ir, width, width, 2 * width)
            )
            frame.compiled.full_dst = True
    except (PlanError, UnitError) as error:
        refuse(
            frame.where,
            node,
            f"the tile function {function.name} ({where}) is not lowered by the "
            f"tile compiler: {error}",
            "write this body inline: km.cxx(..., gives=km.Tile)",
        )
    frame.compiled.units[cxx] = (made, unit)
    frame.compiled.body_includes["compute"].append(f'#include "tile_{cxx}.hpp"')
    return unit, cxx


def timed_call(frame, call, timed):
    """A call's lines; `timed` (stamps): between two tensix_sync(), its span on
    the math CPU's wall clock to the stamps page."""
    if not timed:
        return call
    last = frame.spans_most - 1
    return [
        "MATH(tensix_sync());",
        "MATH(const uint32_t km_called = km::wall());",
        *call,
        "MATH(tensix_sync());",
        f"MATH(km_spans[{SPANS_FIRST} + (km_spans[1] < {last} ? km_spans[1] : "
        f"{last})] = km::wall() - km_called);",
        "MATH(km_spans[1] += 1);",
        'MATH(asm volatile("fence" ::: "memory"));',
    ]


def traced_placement(frame, node, function):
    """Each product's placement ("none" or "plain") in a traced tile function, by
    its operands' names; refused at the call's line when a marker turns one (the
    unit's vector code reads the products in DST in one body, and the transpose
    back is a library call) or marks both operands for one side."""
    tree = source_of(function.function)[0]
    placements = {}
    for product in ast.walk(tree):
        if not (isinstance(product, ast.BinOp) and isinstance(product.op, ast.MatMult)):
            continue
        left = marked_operand(product.left)[1]
        right = marked_operand(product.right)[1]
        where = f"{function.name}'s `{ast.unparse(product)}` (line {product.lineno})"
        if left is not None and left == right:
            refuse(
                frame.where,
                node,
                f"{where}: both operands are marked km.{spelling(left)}",
                "mark one operand",
            )
        if left == "srca" or right == "srcb":
            refuse(
                frame.where,
                node,
                f"{where} turns the product (its right operand in SrcB), and a "
                "tile function the tile compiler traces runs its vector code on "
                "the product in DST in one body: the turned product would leave "
                "(a @ w)^T there, and DST's transpose back is a library call the "
                "body's words do not hold",
                "mark it the plain way (`km.srcb(a) @ w`), or make the product "
                "a tile function `return a @ km.srcb(w)` of its own that a "
                "handler's product line adds",
            )
        names = (marked_operand(product.left)[0], marked_operand(product.right)[0])
        placements[names] = "none" if left is None and right is None else "plain"
    return placements


def scalar_words(unit, scalars):
    """The call's run-time scalar arguments after `dst`: each float word's bits, in
    the unit's order (`unit.scalars`: `<parameter>_bits`)."""
    return "".join(
        f", __builtin_bit_cast(uint32_t, {scalars[name.removesuffix('_bits')]})"
        for name in unit.scalars
    )


def tile_format_refusal(formats):
    """None, or why a tile function is refused tiles of `formats`: float32 and the B
    family run on an fp32 Dst, the A family on a 16-bit one (tile_dest); a mix of
    the two is refused (an fp32 Dst reads the A family 2^-15 off, a 16-bit Dst cuts
    float32 to fp16)."""
    if all(f in TILE_FORMATS for f in formats) or all(f in A_DTYPES for f in formats):
        return None
    a = [f for f in formats if f in A_DTYPES]
    other = [f for f in formats if f not in A_DTYPES]
    return (
        f"a tile function takes tiles of float32 or the B family ("
        f"{', '.join(TILE_FORMATS)}), run on an fp32 Dst, or of the A family ("
        f"{', '.join(A_DTYPES)}), run on a 16-bit Dst; given {', '.join(a)} with "
        f"{', '.join(other)}: an fp32 Dst reads the A family 2^-15 off, and a 16-bit "
        "Dst holds no float32"
    )


class SrcaSwitch:
    """A tile call's SrcA format set again before each unpack into it (a product's w,
    each copied operand) when those are not of one format: the call's setup sets it
    once, for the first of them, and the unpacker reads every later tile as that."""

    def __init__(self, products, copied):
        read = [w.buffer.dtype for _, w in products] + [c.buffer.dtype for c in copied]
        self.on = len(set(read)) > 1

    def product(self, a, w):
        if not self.on:
            return []
        return [f"KM_UNPACK_FORMATS({w.buffer.name}.mover_a, {a.buffer.name}.mover_a);"]

    def copy(self, item):
        if not self.on:
            return []
        return [f"KM_UNPACK_FORMAT_A({item.buffer.name}.mover_a);"]


def tile_dest(formats):
    """The Dst a tile function on tiles of `formats` runs on: "fp16a" for the A
    family, else "fp32"."""
    return "fp16a" if all(f in A_DTYPES for f in formats) else "fp32"


def traced_results(ir, out, t, values):  # t: a tile, or (a row, slice(None))
    """Each result of a tile function's trace stored into its own output tensor (the
    first into `out`, the others added to the IR as result1, result2, ...)."""
    for k, value in enumerate(values):
        if k == 0:
            trace.store(out.tiles[t], value)
            continue
        result = TensorParam(f"result{k}", True, F32)
        ir.tensors.append(result)
        trace.store(trace.TensorProxy(result).tiles[t], value)


def chunk_packs(unit, outs, tile, chunk, indent):
    """A wide row's chunk packed: result k's two Dst tiles from unit.chunk_results[k]
    into its stream's half (outs[k]: its out and room)."""
    lines = []
    for base, (out, room) in zip(unit.chunk_results, outs):
        lines += [
            f"if ({room}) {{",
            f"    KM_PACK_FORMAT({out});",
            f"    pack_tile<true>({base}, {out}, {tile} + 2 * {chunk});",
            f"    pack_tile<true>({base + 1}, {out}, {tile} + 2 * {chunk} + 1);",
            "}",
        ]
    return [indent + x for x in lines]


def value_packed(cxx, buffer, held):
    """A row's per-row value (its part `value`, Dst 0) packed into its field: the
    last write's page freed first."""
    return [
        f"if ({held}) {{",
        f"    cb_pop_front({buffer}, {buffer}Pages);",
        "}",
        f"MATH((km_tile::{cxx}_value(0)));",
        *pack(0, buffer).split("\n"),
        f"{held} = km_ok;",
    ]


def sum_field_copies(unit, fields, back, index):
    """A wide row's field tiles into DST 4 + k in a sum pass, when its reductions
    read a field's row (`.row()`): each field unpacked in its own setup, the row's
    (`back`) again after; none otherwise."""
    if not unit.chunk_fields_in_sums:
        return []
    lines = []
    for k, f in enumerate(fields):
        cb = f"kField{camel(f)}"
        lines += [
            f"KM_UNPACK_FORMAT_A({cb});",
            f"copy_init({cb});",
            f"copy_tile({cb}, {index}, {4 + k});",
        ]
    return [*lines, f"KM_UNPACK_FORMAT_A({back});", f"copy_init({back});"]


def field_copies(compiled, fields, rows, index):
    """A wide row's field tiles into DST 4 + k, tile `index` of each: a field of
    another format than the values' fp32 page unpacked in its own setup (a bf16
    field: half the L1), the page's again after."""
    lines, now = [], rows
    for k, f in enumerate(fields):
        cb = f"kField{camel(f)}"
        own = cb if compiled.fields[f].format.dtype != "float32" else rows
        if own != now:
            lines += [f"KM_UNPACK_FORMAT_A({own});", f"copy_init({own});"]
            now = own
        lines.append(f"copy_tile({cb}, {index}, {4 + k});")
    return lines


def chunk_rows_buffer(compiled, cxx, node, pages=1):
    """The page a wide row's per-row values go through between its two parts (one
    a tile function), or `pages` (a row read in passes: a pair of the values' and
    the sums' tiles a row): pushed by the pack CPU, read back by the unpacker, fp32."""
    name = f"kChunkRows{camel(cxx)}"
    for buffer in compiled.buffers:  # a row read in passes holds its rows' pairs
        if buffer.name == name and buffer.pages[0] < pages:
            buffer.pages = (pages,)
    if all(b.name != name for b in compiled.buffers):
        layout.added(
            compiled,
            Buffer(
                name,
                "field",
                layout.next_index(compiled),
                None,
                "float32",
                4096,
                (pages,),
                1,
                f"the per-row values of {cxx}'s wide rows (line {node.lineno})",
            ),
        )
        compiled.unpack_fp32.add(name)
    return name


def row_broadcast(item, field, slot, dst, copied, along="ROW"):
    """The unpack of a per-column operand into DST tile `dst`: row 0 of its tile (a
    field's, or the loop tile of `item`) copied to every row (a per-row operand,
    `along` "COL": column 0 to every column), then the copy's setup again for the
    call's next tiles."""
    if field is not None:
        cb, tile = f"kField{camel(field)}", "0"
        formats = cb
    else:  # the block's half; the formats of half A (the pair's one format)
        cb, formats = item.cxx, f"{item.buffer.name}.mover_a"
        tile = item.first if slot is None else f"{item.first} + {slot}"
    lines = [  # a 16-bit tile goes through SrcB (an fp32 one unpacked to DST)
        f"KM_UNPACK_FORMATS({formats}, {formats});",
        f"unary_bcast_init<BroadcastType::{along}>({cb});",
        f"unary_bcast<BroadcastType::{along}>({cb}, {tile}, {dst});",
        f"unary_bcast_uninit<BroadcastType::{along}>({cb});",
    ]
    if copied:
        back = f"{copied[0].buffer.name}.mover_a"
        lines += [f"KM_UNPACK_FORMAT_A({back});", f"copy_init({back});"]
    return lines


def matrix_product(function):
    """The line of the first `@` in a function's source, or None."""
    tree, _, _ = source_of(function)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return node.lineno
    return None


def known_count(text):
    """A count's C++ text of integer constants (`(2 * 32)`) as an int, else None."""
    ops = {ast.Add: int.__add__, ast.Sub: int.__sub__, ast.Mult: int.__mul__}
    ops[ast.Div] = int.__floordiv__

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](value(node.left), value(node.right))
        raise ValueError(node)

    try:
        return value(ast.parse(str(text), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
