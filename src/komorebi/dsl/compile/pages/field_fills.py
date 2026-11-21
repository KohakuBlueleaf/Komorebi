"""A field's fill: mover A's read of pages into the field (or both movers', by a
pattern), compute's zero pages, the one producer of a field's pages, and a field kept
across commands."""

import ast

from komorebi.dsl.compile import expressions, handler_buffers
from komorebi.dsl.compile.cxx import camel, pattern_uses, range_uses
from komorebi.dsl.compile.pages import page_streams, ranges
from komorebi.dsl.compile.source import (
    WAIT_DATA,
    WAIT_INPUT,
    WAIT_OUTPUT,
    line_text,
    refuse,
    wait,
)
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Line


def fill(frame, node, call, field):
    """`self.f.fill(km.Stream(t.pages(first, n)))`: mover A reads the n pages into
    the field's L1 and hands it on; one km.cxx body takes it, then it is free."""
    if field in frame.compiled.own:
        refuse(
            frame.where,
            node,
            f"self.{field} is an own field: only bodies use it, no line fills it",
            f"declare it without km.{spelling('own')}() to fill it from a stream",
        )
    stream = call.args[0] if len(call.args) == 1 and not call.keywords else None
    if not (
        expressions.construct(frame, stream) == "stream"
        and len(stream.args) == 1
        and {k.arg for k in stream.keywords} <= {spelling("block"), "pattern"}
        and ranges.is_range(frame, stream.args[0])
    ):
        refuse(
            frame.where,
            node,
            "a field is filled from a stream of a page range (block= and "
            "pattern= the only options)",
            f"write self.{field}.fill(km.Stream(tensor.pages(first, count)))",
        )
    if frame.filled.get(field) is not None:
        refuse(
            frame.where,
            node,
            f"self.{field} is filled above and no body took it",
            "fill it once, before the body that takes it",
        )
    pages = ranges.range_of(frame, stream.args[0])
    handler_buffers.pages_only(frame, node, pages)
    dtype, page_bytes = handler_buffers.tensor_format(frame, node, pages.tensor)
    value = frame.compiled.fields[field]
    if (dtype, page_bytes) != (value.format.dtype, value.format.tile_bytes):
        refuse(
            frame.where,
            node,
            f"{pages.tensor} has {dtype} pages and self.{field} "
            f"{value.format.dtype} tiles",
            "fill a field from a tensor of its format",
        )
    buffer, pages_name = f"kField{camel(field)}", f"kField{camel(field)}Pages"
    pitch = None  # a row's tiles in the field: the rectangle's columns, or the
    if pages.rect is not None and isinstance(pages.rect[1], str):  # field's
        pitch = expressions.field_sides(frame, node, field)[1]
        if pages.bounds[1] is None or pages.bounds[1] > pitch:
            expressions.stated_words(frame, stream.args[0], "a fill of a field")
            refuse(
                frame.where,
                node,
                f"the rectangle's columns are a word and self.{field}'s rows are "
                f"{pitch} tiles: the build must know they are at most {pitch}",
                f"write {pages.tensor}.{spelling('tiles')}[r0:r1, "
                f"c:{spelling('min')}(c + {pitch}, e)]",
            )
    count = pages.count if pages.rect is None else ("rect", pitch or pages.rect[1])
    tensor = f"{camel(pages.tensor)}({pages.tensor})"
    movers = movers_fill(frame, node, stream, pages, tensor, buffer, pages_name)
    producer(frame, node, field, "fill")
    frame.compiled.fills.append((field, count, frame.method.name, node.lineno))
    frame.compiled.includes.add("ask")
    frame.filled[field] = (node, frame.blocks[-1])
    frame.fill_loops[field] = (len(frame.loops), node.lineno)
    frame.fill_loop_of[field] = frame.loops[-1] if frame.loops else None
    handler_buffers.moved(frame, node, pages.tensor, "fill")
    text = (
        f"if (!movement.reserve({buffer}, {pages_name}) ||\n"
        f"    !km::read_pages(movement, command, {tensor}, {pages.first}, "
        f"{pages_name}, get_write_ptr({buffer}))) {{\n"
        "    return false;\n"
        "}\n"
        "noc_async_read_barrier();\n"
        f"cb_push_back({buffer}, {pages_name});"
    )
    if pages.rect is not None:  # each row of the rectangle, its run of pages
        rows, columns, row_step = pages.rect[:3]
        row = f"km_row_{field}"
        text = (
            f"// the rectangle's rows, at most the field's\n"
            f"if (({rows}) * {pitch or columns} > {pages_name} ||\n"
            f"    !movement.reserve({buffer}, {pages_name})) {{\n"
            "    return false;\n"
            "}\n"
            f"for (uint32_t {row} = 0; {row} < ({rows}); {row}++) {{\n"
            f"    if (!km::read_pages(movement, command, {tensor}, {pages.first} + "
            f"{row} * {row_step}, {columns}, get_write_ptr({buffer}) + {row} * "
            f"{pitch or columns} * {page_bytes})) {{\n"
            "        return false;\n"
            "    }\n"
            "}\n"
            "noc_async_read_barrier();\n"
            f"cb_push_back({buffer}, {pages_name});"
        )
    if movers is not None:
        text = movers[0]
        handler_buffers.held_cast_movers(frame, node, ["B"], "fill")
        wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_DATA)
    compute = []
    if kept_field(frame, field):
        held = f"km_held_{field}"
        wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
        free = (
            f"    if (!compute.wait_front({buffer}, {pages_name})) {{\n"
            "        return;\n    }\n"
            f"    cb_pop_front({buffer}, {pages_name});\n"
            "}\n"
        )
        compute.append(
            Line(
                f"// self.{field} stays between commands: the last fill freed "
                "before this one\n"
                "#if defined(KM_OVERLAY_PART)  // a part's statics are zeroed at "
                "each call\n"
                f"uint32_t& {held} = km::fields_held_word();\n"
                f"if (({held} >> {buffer}) & 1u) {{\n"
                f"{free}"
                f"{held} |= 1u << {buffer};\n"
                "#else\n"
                f"static bool {held} = false;\n"
                f"if ({held}) {{\n"
                f"{free}"
                f"{held} = true;\n"
                "#endif"
            )
        )
    if movers is None:
        return [Line(text, range_uses(pages, frame), mover="A")], compute
    uses = range_uses(pages, frame) | movers[1]
    return [Line(text, uses, mover="A"), Line(text, uses, mover="B")], compute


def movers_fill(frame, node, stream, pages, tensor, buffer, pages_name):
    """`fill(km.Stream(pages, block=b, pattern=p))`: both movers read their blocks
    into the field (km::fill_field); its C++ and uses, or None without the options."""
    if not stream.keywords:
        return None
    options = {k.arg: k.value for k in stream.keywords}
    if len(options) != 2:
        refuse(
            frame.where,
            node,
            "a fill on both movers names its block= and its pattern=",
            f'write km.Stream({ast.unparse(stream.args[0])}, block=b, pattern="AB")',
        )
    if pages.rect is not None:
        refuse(
            frame.where,
            node,
            "a fill of a rectangle of tiles is mover A's: block= and pattern= "
            "take a page range",
            f"write {pages.tensor}.{spelling('pages')}(first, count), or drop "
            "block= and pattern=",
        )
    block = expressions.constant_int(frame, options[spelling("block")])
    if block < 1:
        refuse(frame.where, node, f"a block of {block} pages", "give block= a count")
    pattern, letters = page_streams.stream_pattern(frame, options["pattern"])
    word = pattern if letters is None else f'km::pattern("{letters}")'
    text = (
        f"if (!km::fill_field(movement, command, {tensor}, {pages.first}, "
        f"{pages_name}, {block}, {word}, {buffer})) {{\n"
        "    return false;\n"
        "}"
    )
    return text, pattern_uses(pattern)


def kept_field(frame, field):
    """Whether a field is filled by one handler and read by another: it stays in
    L1 between commands (its next fill frees it)."""
    fillers, readers = frame.compiled.field_users.get(field, ([], []))
    return bool(fillers and readers)


def producer(frame, node, field, kind):
    """A field's pages are pushed by one producer: mover A's fill, or compute's
    adds of a product, in every handler of the class."""
    other = frame.compiled.producers.setdefault(
        field, (kind, frame.method.name, node.lineno, line_text(frame.where, node))
    )
    if other[0] != kind:
        _, handler, line, text = other
        pusher = "mover A" if other[0] == "fill" else "compute"
        refuse(
            frame.where,
            node,
            f"self.{field} is pushed by {pusher}"
            f" too ({handler}, line {line}, `{text}`): a buffer has one producer",
            "fill it, or add products into it, not both: give each its own field",
        )


def zero(frame, node, call, field):
    """`self.f.zero()`: compute packs a zero tile into each of the field's pages and
    pushes them, as a fill would (no DRAM read); its lines then read it as a filled
    field."""
    if field in frame.compiled.own:
        refuse(
            frame.where,
            node,
            f"self.{field} is an own field: only bodies use it, no line zeroes it",
            f"declare it without km.{spelling('own')}() to zero it",
        )
    if call.args or call.keywords:
        refuse(
            frame.where,
            node,
            "a zero takes no arguments",
            f"write self.{field}.zero()",
        )
    if frame.filled.get(field) is not None:
        refuse(
            frame.where,
            node,
            f"self.{field} is filled above and no body took it",
            "zero it once, before the lines that read it",
        )
    if frame.dst_live is not None:
        refuse(
            frame.where,
            node,
            f"{frame.dst_live} still holds DST, and a zero packs DST 0",
            "write it first",
        )
    buffer, pages = f"kField{camel(field)}", f"kField{camel(field)}Pages"
    producer(frame, node, field, "zero")
    frame.compiled.includes |= {"fill", "pack"}
    frame.compiled.summed_fields.add(buffer)  # pack writes it: its pack format
    frame.filled[field] = (node, frame.blocks[-1])
    frame.fill_loops[field] = (len(frame.loops), node.lineno)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_OUTPUT)
    text = (
        f"// self.{field} zeroed on the core: a zero tile packed into each page\n"
        "tile_regs_acquire();\n"
        "fill_tile_init();\n"
        "fill_tile(0, 0.0f);\n"
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"if (km_ok && compute.reserve_back({buffer}, {pages})) {{\n"
        f"    KM_PACK_FORMAT({buffer});\n"
        f"    for (uint32_t km_page = 0; km_page < {pages}; km_page++) {{\n"
        f"        pack_tile<true>(0, {buffer}, km_page);\n"
        "    }\n"
        f"    cb_push_back({buffer}, {pages});\n"
        "}\n"
        "tile_regs_release();"
    )
    return [], [Line(text, {"km_ok"})]
