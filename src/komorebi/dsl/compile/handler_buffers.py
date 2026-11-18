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
from komorebi.formats import tile_bytes


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
    tiles = page_tiles(frame, pages.tensor)
    if tiles != 1:  # a read a page of k tiles; the walk stays in tiles
        tensor = f"km::pages_of<{tiles}>({tensor})"
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


def tensor_format(frame, node, tensor, paged=False):
    """A tensor's format and the bytes of a buffer page of it: its page, or for a
    tensor in pages of k tiles, a tile (`paged`: a read stream of its tiles, the one
    line built for it; any other refused)."""
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
    tiles = page_tiles(frame, tensor)
    frame.compiled.page_tiles[(frame.method.name, tensor)] = tiles
    if tiles == 1:
        return example.dtype, example.page_bytes
    if not paged:
        refuse(
            frame.where,
            node,
            f"{tensor} is in pages of {tiles} tiles (page_tiles={tiles}): only a read "
            f"stream of its tiles is built for it, not this line",
            f"read it as km.{spelling('stream')}({tensor}.{spelling('tiles')}[r0:r1, "
            "c0:c1], block=...), or make it with page_tiles=1",
        )
    return example.dtype, example.page_bytes // tiles


def page_tiles(frame, tensor):
    """The tiles a DRAM page of `tensor`'s example (1: a page a tile or a row)."""
    if tensor in frame.compiled.regions:
        return 1
    return getattr(frame.tensors.get(tensor), "page_tiles", 1)


def paged_stream(frame, node, pages, block):
    """Refuses a read stream of a tensor in pages of k tiles that would split a page:
    not a rectangle of its tiles, or a block, a first column or a width the build
    knows that is not whole pages (a word's: the mover's check)."""
    tiles = page_tiles(frame, pages.tensor)
    if tiles == 1:
        return
    tensor = pages.tensor
    if not pages.tiles:
        refuse(
            frame.where,
            node,
            f"{tensor} is in pages of {tiles} tiles: a stream of it reads its tiles",
            f"write km.{spelling('stream')}({tensor}.{spelling('tiles')}[r0:r1, c0:c1]"
            ", ...)",
        )
    if block % tiles:
        refuse(
            frame.where,
            node,
            f"{spelling('block')}={block} over {tensor} in pages of {tiles} tiles: a "
            f"block is whole pages, a multiple of {tiles}",
            f"give a {spelling('block')} that is a multiple of {tiles}",
        )
    width = pages.sides[1]
    if isinstance(width, str):
        width = int(width) if width.isdigit() else None
    for what, known in (("first column", pages.columns_first), ("width", width)):
        if known is not None and known % tiles:
            refuse(
                frame.where,
                node,
                f"{tensor}.{spelling('tiles')}[...]: its {what} {known} over pages of "
                f"{tiles} tiles is not a multiple of {tiles} (a page is not split)",
                f"give columns c0:c1 with c0 and c1 multiples of {tiles}",
            )


def held_cast_movers(frame, node, letters, how="stream"):
    """A stream (or a page write, a fill) on the mover of a block cast outside a
    loop that products take: the mover would wait at the cast for compute to free
    the block, which compute does after this line, so the cast's finish goes to
    the handler's end (defer_finish)."""
    for ends in frame.held_blocks.values():
        for name, _ in ends:
            mover = frame.finished_casts.get(name)
            if mover is not None and any(
                each is None or mover in each for each in letters
            ):
                defer_finish(frame, node, name, how)


def defer_finish(frame, node, name, how="stream"):
    """Block `name`, cast outside a loop, followed by this line on its mover: the
    mover's finish (compute's free; on the sender every receiver's DONE) goes to
    the handler's end. Refused for a cast under a condition, which the finish
    would leave; a second cast on the mover is refused at the cast."""
    if name in frame.deferred_casts:
        return
    mover = frame.finished_casts[name]
    if frame.cast_finishes.get(name) is None:
        other = "A" if mover == "B" else "B"
        refuse(
            frame.where,
            node,
            f"{name} was cast outside a loop under a condition, and products take "
            f"it, so mover {mover} waits at the cast for compute to free it, which "
            f"compute does after this {how}, which mover {mover} moves",
            (
                f'read the stream on mover {other}: pattern="{other}"'
                if how == "stream"
                else f'cast it on mover {other} (mover="{other}")'
            )
            + ", or cast it at the handler's top level",
        )
    frame.deferred_casts.append(name)


def deferred_finishes(frame, movement):
    """`movement` without the deferred casts' finishes, which come at its end."""
    moved = []
    for name in frame.deferred_casts:
        nodes = frame.cast_finishes[name]
        drop_nodes(movement, {id(node) for node in nodes})
        moved += nodes
    return movement + moved


def drop_nodes(nodes, ids):
    """Removes the nodes of `ids` from the tree `nodes`, in place."""
    nodes[:] = [node for node in nodes if id(node) not in ids]
    for node in nodes:
        children = getattr(node, "children", None)
        if children is not None:
            drop_nodes(children, ids)


def tile_packed(frame, node, tensor):
    """Refuses a tile packed into a page of a tensor in rows whose page is not one
    tile's bytes: pack_tile writes a whole tile, past the page in L1."""
    example = frame.tensors.get(tensor)
    if getattr(example, "layout", "tile") == "tile":
        return
    tile = tile_bytes(example.dtype)
    if example.page_bytes == tile:
        return
    refuse(
        frame.where,
        node,
        f"{tensor} is in rows of {example.page_bytes} B, not tiles of compute "
        f"({tile} B of {example.dtype}): a write packs a whole tile into each "
        "page, past the page in L1",
        "make the tensor in tiles, or write its pages from a body on a mover "
        '(`km.cxx(..., on="mover_b")`: noc_async_write_page)',
    )
