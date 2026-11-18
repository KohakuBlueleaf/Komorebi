"""A handler's steps: a tile function stepping a state through a field's rows, f(x,
self.s[r]), its state taken and packed back, from DST or from a field."""

import ast
import inspect
import math

from komorebi.dsl.compile import (
    expressions,
    handler_buffers,
    layout,
    scope,
    tile_lookup,
)
from komorebi.dsl.compile.cxx import camel, range_uses
from komorebi.dsl.compile.dst.dst_code import DST_HALF, pack
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import (
    WAIT_DATA,
    WAIT_INPUT,
    WAIT_OUTPUT,
    function_line,
    line_text,
    refuse,
    wait,
)
from komorebi.dsl.compile.tiles import functions
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Line
from komorebi.dsl.tile import trace
from komorebi.dsl.tile.forms.steps import step_program
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam, way_of
from komorebi.dsl.tile.unit import UnitError

# A step field's tokens: two, so a take of another index need not wait for the last
# push; a take of the index the last take took waits for both (every push done).
TOKENS = 2


def row_given_back(frame, compute, value):
    """Sums started from a row of a field (started_from_row) written elsewhere
    leave the row as it was: its token pushed back before the last compute line's
    DST release, so the row's next read finds it."""
    row_of = getattr(value, "row_of", None)
    if row_of is None:
        return
    given = (
        f"if (km_ok && compute.reserve_back({row_of[2]}, 1)) {{  // the row as "
        "it was\n"
        f"    cb_push_back({row_of[2]}, 1);\n"
        "}\n"
        "tile_regs_release();"
    )
    last = compute[-1]
    at = last.text.rindex("tile_regs_release();")
    last.text = last.text[:at] + given + last.text[at + len("tile_regs_release();") :]


def started_from_field(frame, node, name, call):
    """`acc = f(self.o, self.s[r])`, f a step-form function, self.o a field
    compute writes (1 to 3 tiles): DST sums that start from the field's tiles in
    DST 0 .., state r in the DST tile after them, f's body run on both; products
    then add into acc, and acc's write packs the state back into page r."""
    function = tile_lookup.tile_function_of(frame, call)
    field_, state, index, row = step_of_field(frame, call)
    if row is not None:
        return started_from_row(frame, node, name, call, field_, state, index, row)
    if frame.dst_live is not None:
        refuse(
            frame.where,
            node,
            f"{frame.dst_live} still holds DST; a step's row starts at DST 0",
            "write it first",
        )
    if field_ not in frame.packed:
        refuse(
            frame.where,
            node,
            f"self.{field_} is not written above in this handler: sums start from "
            "a field compute writes",
            f"write self.{field_}.write(acc) before, e.g. of km.tile.zeros",
        )
    stored = frame.compiled.fields[field_]
    width = math.prod(stored.shape[:-2])
    if not 1 <= width <= 3 or stored.format.dtype != "float32":
        refuse(
            frame.where,
            node,
            f"self.{field_} is {width} {stored.format.dtype} tile(s); a step's row "
            "is 1 to 3 fp32 tiles",
            f"declare self.{field_}: km.L1[km.fp32, n, 32, 32], n at most 3",
        )
    buffer, token, slot, bound = step_state(frame, node, state, index)
    unit, cxx = step_unit(frame, node, function, width)
    source = f"kField{camel(field_)}"
    frame.compiled.unpack_fp32.add(source)
    copies = "".join(f"    copy_tile({source}, {t}, {t});\n" for t in range(width))
    text = (
        f"{bound}"
        f"// {function.name}'s step: self.{field_}'s tiles into DST 0 .. "
        f"{width - 1}, self.{state}[{ast.unparse(index)}] into DST {width}\n"
        f"km_ok = km_ok && compute.wait_front({source}, {source}Pages);\n"
        "tile_regs_acquire();\n"
        f"{state_taken(buffer, token, slot, width)}"
        f"    KM_UNPACK_FORMAT_A({source});\n"
        f"    copy_init({source});\n"
        f"{copies}"
        f"{functions.step_calls(frame, cxx)}"
        "}"
    )
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    frame.compiled.tile_calls.append(
        (frame.method.name, node.lineno, line_text(frame.where, node), function, cxx)
    )
    frame.dst_next = max(frame.dst_next, width + 1)  # the state's tile too
    if frame.dst_next > DST_HALF:
        frame.compiled.full_dst = True
    frame.dst_live = name
    value = Value(
        "tile", name, dst=0, width=width, made_at=(len(frame.loops), node.lineno)
    )
    value.state = (buffer, token, slot, width, index)  # packed at acc's write
    value.state_kept = unit.state_kept
    frame.scope[name] = value
    return [], [Line(text, {"km_ok"} | expressions.uses(frame, index))]


def started_from_row(frame, node, name, call, field_, state, index, row):
    """`acc = f(self.o[i], self.s[r])`: started_from_field from row i of a field
    of rows filled above (n x w fp32 tiles), read in place by index after its
    token, as a step's states are; acc's write `self.o[j].write(acc)` packs it
    back in place and pushes the token."""
    function = tile_lookup.tile_function_of(frame, call)
    if frame.dst_live is not None:
        refuse(
            frame.where,
            node,
            f"{frame.dst_live} still holds DST; a step's row starts at DST 0",
            "write it first",
        )
    stored = frame.compiled.fields[field_]
    width = stored.shape[1] if len(stored.shape) == 4 else 0
    if not 1 <= width <= 3:
        refuse(
            frame.where,
            node,
            f"self.{field_}'s rows are {width or 'not'} tiles; a step's row is 1 "
            "to 3 fp32 tiles",
            f"declare self.{field_}: km.L1[km.fp32, rows, n, 32, 32], n at most 3",
        )
    source, token_o, line_o, bound_o = step_state(frame, node, field_, row, width)
    buffer, token, slot, bound = step_state(frame, node, state, index)
    unit, cxx = step_unit(frame, node, function, width)
    base = f"({line_o}) * {width}"
    copies = "".join(  # tile 0 copied with the token's take
        f"    copy_tile({source}, {base} + {t}, {t});\n" for t in range(1, width)
    )
    text = (
        f"{bound_o}{bound}"
        f"// {function.name}'s step: self.{field_}[{ast.unparse(row)}] into DST 0 "
        f".. {width - 1}, self.{state}[{ast.unparse(index)}] into DST {width}\n"
        "tile_regs_acquire();\n"
        f"{state_taken(buffer, token, slot, width)}"
        "}\n"
        f"{state_taken(source, token_o, f'{base}', 0)}"
        f"{copies}"
        f"{functions.step_calls(frame, cxx)}"
        "}"
    )
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    frame.compiled.tile_calls.append(
        (frame.method.name, node.lineno, line_text(frame.where, node), function, cxx)
    )
    frame.dst_next = max(frame.dst_next, width + 1)
    if frame.dst_next > DST_HALF:
        frame.compiled.full_dst = True
    frame.dst_live = name
    value = Value(
        "tile", name, dst=0, width=width, made_at=(len(frame.loops), node.lineno)
    )
    value.state = (buffer, token, slot, width, index)
    value.state_kept = unit.state_kept
    value.row_of = (field_, source, token_o)  # written back in place
    frame.scope[name] = value
    uses = {"km_ok"} | expressions.uses(frame, index) | expressions.uses(frame, row)
    return [], [Line(text, uses)]


def state_packed(frame, compute, buffer, token, slot, dst, index, kept=False):
    """DST `dst` packed into state `slot` and the token pushed, before the last
    compute line's DST release (after the row's packs); `kept` (the steps return
    the state as they took it): the token pushed alone, the page as it was."""
    packed = (
        f"if (km_ok && compute.reserve_back({token}, 1)) {{  // the state as it was\n"
        f"    cb_push_back({token}, 1);\n"
        "}\n"
        "tile_regs_release();"
        if kept
        else f"if (km_ok && compute.reserve_back({token}, 1)) {{  // pack's alone\n"
        f"    KM_PACK_FORMAT({buffer});\n"
        f"    pack_tile<true>({dst}, {buffer}, {slot});\n"
        f"    cb_push_back({token}, 1);\n"
        "}\n"
        "tile_regs_release();"
    )
    last = compute[-1]
    at = last.text.rindex("tile_regs_release();")
    last.text = last.text[:at] + packed + last.text[at + len("tile_regs_release();") :]
    last.uses = last.uses | expressions.uses(frame, index)


def state_taken(buffer, token, slot, dst):
    """The token taken and state `slot` copied into DST `dst` (an open block): one
    token when the last take was of another index, else all of them."""
    took = took_name(token)
    return (
        f"km_ok = km_ok && compute.wait_front({buffer}, {buffer}Pages) &&\n"
        f"    compute.wait_front<false>({token}, ({slot}) == {took} ? {TOKENS} : 1);\n"
        "if (km_ok) {\n"
        f"    {took} = {slot};\n"
        f"    cb_pop_front({token}, 1);\n"
        f"    KM_UNPACK_FORMAT_A({buffer});\n"
        f"    copy_init({buffer});\n"
        f"    copy_tile({buffer}, {slot}, {dst});\n"
    )


def took_name(token):
    """The word that holds the index the last take of `token` took."""
    return f"km_took_{token}"


def step_of_dst(frame, call):
    """(acc, field, index) of a step function's call `f(acc, self.s[r])`, acc DST
    sums and self.s a field, or None."""
    if len(call.args) != 2 or call.keywords:
        return None
    acc, slot = call.args
    value = isinstance(acc, ast.Name) and frame.scope.get(acc.id)
    if not value or value.kind != "tile" or value.batch is not None:
        return None
    field_ = isinstance(slot, ast.Subscript) and scope.field_name(frame, slot.value)
    return (acc.id, field_, slot.slice) if field_ else None


def step_of_field(frame, call):
    """(field, state field, index) of `f(self.o, self.s[r])`, f a tile function,
    self.s a field indexed; else None."""
    if tile_lookup.tile_function_of(frame, call) is None or len(call.args) != 2:
        return None
    row, slot = call.args
    index = None
    if isinstance(row, ast.Subscript):  # a row of a field of rows
        row, index = row.value, row.slice
    field_ = scope.field_name(frame, row)
    state = isinstance(slot, ast.Subscript) and scope.field_name(frame, slot.value)
    return (field_, state, slot.slice, index) if field_ and state else None


def step_state(frame, node, field_, index, width=None):
    """State `index` of a step's field (or row `index` of a field of rows of
    `width` tiles): (its buffer, its token, the index's C++, a run-time bound
    check or ""); at the field's first use its token buffer, the token primed at
    the handler's start and taken at its end."""
    stored = frame.compiled.fields[field_]
    if width is None:
        pages = stored.shape[0] if len(stored.shape) == 3 else None
        if stored.format.dtype != "float32" or not isinstance(pages, int):
            refuse(
                frame.where,
                node,
                f"self.{field_} holds a step's states: fp32 tiles, a page each",
                f"declare self.{field_}: km.L1[km.fp32, states, 32, 32]",
            )
        items, what = pages, "states"
    else:
        shape = stored.shape
        if stored.format.dtype != "float32" or len(shape) != 4 or shape[1] != width:
            refuse(
                frame.where,
                node,
                f"self.{field_} holds rows of {width} fp32 tile(s), a row an index",
                f"declare self.{field_}: km.L1[km.fp32, rows, {width}, 32, 32]",
            )
        items, what = shape[0], "rows"
        pages = items * width
    buffer, token = f"kField{camel(field_)}", f"kStepToken{camel(field_)}"
    filled_in = frame.fill_loops.get(field_, (0, 0))
    if filled_in[0] > 0:
        refuse(
            frame.where,
            node,
            f"self.{field_} is filled in a loop (line {filled_in[1]}) and its "
            f"{what} are freed once, at the handler's end: the next turn's fill "
            "would wait for room that is never freed",
            f"fill it before the loops, once (self.{field_}.fill(km."
            f"{spelling('stream')}(...)))",
        )
    if field_ not in frame.stepped:
        if frame.filled.get(field_) is None:
            refuse(
                frame.where,
                node,
                f"self.{field_} is not filled above in this handler: a step's "
                f"{what} start from a fill",
                f"write self.{field_}.fill(km.{spelling('stream')}(t.pages(first, "
                f"{pages}))) before the loop",
            )
        frame.filled[field_] = None
        frame.stepped[field_] = pages
        layout.other_buffer(
            frame.compiled,
            token,
            "token",
            "float32",
            16,
            TOKENS,
            f"line {node.lineno}: the {what}' tokens",
        )
        frame.compiled.summed_fields.add(buffer)  # pack writes the states
        frame.compiled.unpack_fp32.add(buffer)
        frame.opening.append(
            Line(
                f"for (uint32_t km_token = 0; km_token < {TOKENS}; km_token++) {{\n"
                f"    if (compute.reserve_back({token}, 1)) {{  // self.{field_}'s "
                f"{what}: the fill's\n"
                f"        cb_push_back({token}, 1);\n    }}\n}}\n"
                f"uint32_t {took_name(token)} = 0xFFFFFFFFu;  // no index taken yet"
            )
        )
        frame.closing.append(
            Line(
                f"if (km_ok && compute.wait_front<false>({token}, {TOKENS})) {{\n"
                f"    cb_pop_front({token}, {TOKENS});\n}}\n"
                f"cb_pop_front({buffer}, {buffer}Pages);",
                {"km_ok"},
            )
        )
    slot = expressions.integer(frame, index)
    known = slot.rstrip("u").isdigit()
    if known and not int(slot.rstrip("u")) < items:
        refuse(
            frame.where,
            node,
            f"self.{field_} holds {items} {what}, 0 .. {items - 1}",
            f"index it below {items}",
        )
    bound = (
        ""
        if known
        else (
            f"if (!(({slot}) < {items})) {{  // self.{field_}'s {what}\n"
            "    compute.gave_up(km::kGaveBadItem, km::kWhereCInput);\n"
            "    return;\n}\n"
        )
    )
    return buffer, token, slot, bound


def step_unit(frame, node, function, width):
    """The step form's unit of `function(row, state)` over a row of `width` DST
    tiles and the state tile after them: it returns (the row, the state's
    per-row values) or (the row, the values, a word of word_max)."""
    made = (function.function, ("step",), width)
    for cxx, (key, unit) in frame.compiled.units.items():
        if key == made:
            return unit, cxx
    where = function_line(function)
    drawn = "return (row, (m, l, ...)) or (row, (m, l), km.word_max(...))"
    if len(inspect.signature(function.function).parameters) != 2:
        refuse(
            frame.where,
            node,
            f"the step function {function.name} ({where}) takes a row and its " "state",
            f"def {function.name}(row, state): ... {drawn}",
        )
    cxx = function.name
    while cxx in frame.compiled.units:
        cxx += "_"
    tensors = [
        TensorParam(n, out, F32)
        for n, out in (
            ("row", False),
            ("state", False),
            ("result", True),
            ("state_out", True),
        )
    ]
    ir = KernelIR(cxx, tensors, [], {})
    try:
        with trace.Tracer(ir):
            rows, states, results, outs = (trace.TensorProxy(t) for t in tensors)
            for t in trace.grid(trace.num_tile_rows(rows)):
                got = function.function(
                    trace.load(rows.tiles[t, :]), trace.load_state(states.tiles[t])
                )
                if not isinstance(got, tuple) or len(got) not in (2, 3):
                    refuse(
                        frame.where,
                        node,
                        f"the step function {function.name} ({where}) returns "
                        "the row and its state",
                        drawn,
                    )
                trace.store(results.tiles[t, :], got[0])
                word = got[2] if len(got) == 3 else None
                trace.store_state(outs.tiles[t], got[1], word=word)
        fidelity = function.fidelity or "HiFi4"
        unit = step_program(ir, width, width, fidelity=fidelity, matrix_sum=False)
    except (PlanError, UnitError) as error:
        refuse(
            frame.where,
            node,
            f"the step function {function.name} ({where}) is not lowered by the "
            f"tile compiler: {error}",
            way_of(error, ir),
        )
    # every slot it stores is the slot it read, and no word: the state as it was
    stored = [s.value for s in ir.stores if s.value.kind == "state"][0]
    unit.state_kept = not stored.attrs["word"] and all(
        n.kind == "slot" and n.attrs["slot"] == k for k, n in enumerate(stored.args)
    )
    frame.compiled.units[cxx] = (made, unit)
    frame.compiled.body_includes["compute"].append(f'#include "tile_{cxx}.hpp"')
    return unit, cxx


def write_state(frame, node, pages, field_):
    """`t.pages(first, n).write(self.s)`, self.s a step's n states: after the last
    step's token, each state copied into DST 0 and packed in t's format, written
    by mover B a page at a time."""
    count = frame.stepped[field_]
    if pages.rect is not None or pages.count != str(count):
        refuse(
            frame.where,
            node,
            f"self.{field_} is {count} states, a page each",
            f"write {pages.tensor}.{spelling('pages')}(first, {count})"
            f".write(self.{field_})",
        )
    dtype, page_bytes = handler_buffers.tensor_format(frame, node, pages.tensor)
    handler_buffers.tile_packed(frame, node, pages.tensor)
    out = handler_buffers.out_buffer(
        frame, dtype, page_bytes, f"line {node.lineno}: {line_text(frame.where, node)}"
    )
    buffer, token = f"kField{camel(field_)}", f"kStepToken{camel(field_)}"
    tile = f"km_state{node.lineno}"
    packed = "\n".join("        " + line for line in pack(0, out.name).split("\n"))
    compute = (
        f"// self.{field_}'s states, after the last step's\n"
        f"km_ok = km_ok && compute.wait_front<false>({token}, {TOKENS});\n"
        "if (km_ok) {\n"
        f"    KM_UNPACK_FORMAT_A({buffer});\n"
        f"    copy_init({buffer});\n"
        f"    for (uint32_t {tile} = 0; {tile} < {count}; {tile}++) {{\n"
        "        tile_regs_acquire();\n"
        f"        copy_tile({buffer}, {tile}, 0);\n"
        f"{packed}\n"
        "    }\n"
        "}"
    )
    tensor = f"{camel(pages.tensor)}({pages.tensor})"
    movement = (
        f"for (uint32_t {tile} = 0; {tile} < {count}; {tile}++) {{\n"
        f"    if (!movement.wait_front({out.name}, 1) ||\n"
        f"        !km::write_pages(movement, command, {tensor}, {pages.first} + "
        f"{tile}, 1, get_read_ptr({out.name}))) {{\n"
        "        return false;\n"
        "    }\n"
        "    noc_async_writes_flushed();\n"
        f"    cb_pop_front({out.name}, 1);\n"
        "}"
    )
    frame.compiled.includes |= {"pack", "ask"}
    handler_buffers.moved(frame, node, pages.tensor, "page write")
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    return [Line(movement, range_uses(pages, frame), mover="B")], [
        Line(compute, {"km_ok"})
    ]
