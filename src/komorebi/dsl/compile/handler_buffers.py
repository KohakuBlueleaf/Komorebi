"""A handler's buffers and pages: the buffer compute packs and mover B writes, its
wait, a stream's C++, a tensor's page format, and the moves and hand-overs the build
reports."""

from komorebi.dsl.compile import layout
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.layout import REGION_PAGE, WRITE_ITEMS, block_step
from komorebi.dsl.compile.records import Move
from komorebi.dsl.compile.source import line_text, refuse
from komorebi.dsl.front import front
from komorebi.dsl.front.names import spelling


def handover(frame, node, what, sender, receivers):
    """A value handed between compute CPUs by mailbox, for the build's report."""
    frame.compiled.handovers.append(
        f"{frame.method.name}, line {node.lineno}: {what} from {sender} to "
        f"{' and '.join(receivers)} by mailbox"
    )


def moved(frame, node, tensor, how):
    frame.compiled.moves.append(
        Move(frame.method.name, node.lineno, line_text(frame.where, node), tensor, how)
    )
    mover = {"page write": "B", "fill": "A"}.get(how)
    if mover is not None:
        held_cast_movers(frame, node, [mover], how)


def out_buffer(frame, dtype, page_bytes, what, width=1, name=None):
    """The buffer of the page(s) compute packs and mover B writes: the program's
    next one of this handler (handlers share them), two writes of `width`; or the
    one `name`d (rows of a field's sums: rows of one width a buffer)."""
    if name is None:
        name = f"kOut{frame.writes}"
        frame.writes += 1
    pages = WRITE_ITEMS * width
    return layout.out_buffer(frame.compiled, name, dtype, page_bytes, pages, what)


def page_wait(frame, value, pages=1):
    """The opening of mover B's wait for the page(s) compute packs of `value`."""
    if value.kind == "reduced":
        return f"if (!km::reduce_result(movement, {value.buffer.name}, {pages})) {{"
    return f"if (!movement.wait_front({value.buffer.name}, {pages})) {{"


def pages_only(frame, node, pages):
    """Refuses a range of items where a line takes pages."""
    if pages.items is not None:
        refuse(
            frame.where,
            node,
            f"{pages.tensor}.{spelling('items')}(...) is a stream's: this line "
            "moves whole pages",
            f"write {pages.tensor}.{spelling('pages')}(first, count), or stream "
            "the items",
        )


def pattern_cxx(frame, buffer, pattern, letters):
    """The C++ of a stream's pattern: a parameter's, or the pair's constant."""
    if pattern is not None:
        return pattern
    constant = f"{buffer.name}Pattern"
    known = buffer.patterns.get(constant)
    if known is not None and known != letters:
        constant = f"{buffer.name}Pattern{len(buffer.patterns)}"
    buffer.patterns[constant] = letters
    return constant


def stream_cxx(frame, kind, variable, pages, buffer, pattern, repeat):
    tensor = f"{camel(pages.tensor)}({pages.tensor})"
    if pages.items is not None:  # items of a page: an item a buffer page
        tensor = f"km::items({tensor}, {pages.items[0]}, {pages.items[1]})"
    direction = "km::kMoveRead" if kind == "read" else "km::kMoveWrite"
    if pages.rect is not None:  # blocks of whole rows of a rectangle
        rows, columns, row_step, block = pages.rect
        per = block // columns  # rows a block
        blocks = f"(({rows}) + {per - 1}) / {per}"
        between = block_step(per, row_step)
        walk = (
            f"km::Walk{{{pages.first}, {blocks}, {between}, {per}, "
            f"{row_step}, {columns}, 1, {columns}, {block}, ({rows}) - ({blocks} "
            f"- 1) * {per}, {columns}}}"
        )
        return (
            f"// the rectangle's rows, {columns} tiles each, {row_step} apart\n"
            f"const km::Stream {variable}(\n"
            f"    {direction}, {tensor}, {walk}, {repeat}, "
            f"{buffer.name}, {pattern}\n"
            ");"
        )
    return (
        f"const km::Stream {variable}(\n"
        f"    {direction}, {tensor}, {pages.first}, "
        f"{pages.count}, {buffer.name}Block, {repeat}, {buffer.name}, {pattern}\n"
        ");"
    )


def tensor_format(frame, node, tensor):
    if tensor in frame.compiled.regions:  # a thread's region: words, in its pages
        return "uint32", REGION_PAGE
    example = frame.tensors.get(tensor)
    if not front.is_tensor(example):
        refuse(
            frame.where,
            node,
            f"the program is made without an example tensor for {tensor}",
            f"make it with {tensor}=a_tensor (its format and page size)",
        )
    return example.dtype, example.page_bytes


def held_cast_movers(frame, node, letters, how="stream"):
    """Refuses a stream (or a page write, a fill) on the mover of a block cast
    outside a loop that products take: the mover waits at the cast for compute
    to free the block, which compute does at the end of the cast's lines, after
    this line."""
    for ends in frame.held_blocks.values():
        for name, _ in ends:
            mover = frame.finished_casts.get(name)
            if mover is not None and any(
                each is None or mover in each for each in letters
            ):
                other = "A" if mover == "B" else "B"
                refuse(
                    frame.where,
                    node,
                    f"{name} was cast outside a loop and products take it, so "
                    f"mover {mover} waits at the cast for compute to free it, "
                    f"which compute does after this {how}, which mover {mover} "
                    "moves",
                    (
                        f'read the stream on mover {other}: pattern="{other}"'
                        if how == "stream"
                        else f'cast it on mover {other} (mover="{other}"), or in '
                        "the loop around this line"
                    ),
                )
