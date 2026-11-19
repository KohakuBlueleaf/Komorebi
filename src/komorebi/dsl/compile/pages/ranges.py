"""A handler's page ranges: a tensor's pages, a slice of them, a core's part by rank
or along a broadcast, a rectangle of tiles, items packed in pages, and their steps
and bounds."""

import ast

from komorebi.dsl.compile import expressions, handler_buffers
from komorebi.dsl.compile.cxx import text_uses
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import refuse
from komorebi.dsl.front import front
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Glue


def is_range(frame, node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in (
            spelling("pages"),
            spelling("items"),
        ) and is_tensor(frame, node.func.value)
    if isinstance(node, ast.Subscript):
        return is_tensor(frame, node.value) or is_tiles(frame, node.value)
    if isinstance(node, ast.Attribute):
        return node.attr == spelling("pages") and is_tensor(frame, node.value)
    if isinstance(node, ast.Name):
        return frame.scope.get(node.id, Value("")).kind == "range"
    return False


def is_tensor(frame, node):
    return (
        isinstance(node, ast.Name)
        and frame.scope.get(node.id, Value("")).kind == "tensor"
    )


def is_tiles(frame, node):
    """`tensor.tiles`, a tensor's tiles as a grid."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == spelling("tiles")
        and is_tensor(frame, node.value)
    )


def items_moved(frame, node, tensor, size):
    """Refuses a stream of items the movers cannot move: of a DRAM tensor in items not
    a multiple of 64 B (on islands another island's item is asked for by its bytes,
    which land at their offset in a 64-B line; on one card the chip's DRAM read
    alignment). An L1 tensor's items on islands are this island's rows, any size."""
    islands = frame.compiled.islands[0]
    example = frame.tensors.get(tensor)
    in_dram = tensor not in frame.compiled.regions and front.is_tensor(example)
    in_dram = in_dram and example.in_dram
    if islands > 1 and in_dram and size % 64:
        refuse(
            frame.where,
            node,
            f"items of {size} B of {tensor}, and the class is built for {islands} "
            "islands: an item another island holds is asked for by its bytes, and "
            "they land at their offset in a 64-B line, not at the start of the "
            "buffer page",
            f"give a {spelling('item_size')} that is a multiple of 64",
        )
    if in_dram and size % 64:
        refuse(
            frame.where,
            node,
            f"items of {size} B of {tensor}, a DRAM tensor: a core reads DRAM in "
            "aligned lines (64 B on Blackhole, 32 B on Wormhole), so an item that "
            "does not start a 64-B line lands at its offset in that line, not at the "
            "start of its buffer page",
            f"give a {spelling('item_size')} that is a multiple of 64",
        )


def items_of(frame, node):
    """`t.items(first, count, size=B)`: items of B bytes packed in t's pages, a
    whole number a page; a stream moves them as pages of B bytes."""
    tensor = node.func.value.id
    if not node.args:  # first=, count=, size=, step=: the keyword form
        return stepped_items(frame, node, tensor)
    sizes = [k.value for k in node.keywords if k.arg == spelling("item_size")]
    if len(node.args) != 2 or len(sizes) != 1 or len(node.keywords) != 1:
        refuse(
            frame.where,
            node,
            f"{spelling('items')}(first, count, {spelling('item_size')}=bytes)",
            "give first, count and the bytes an item",
        )
    size = expressions.constant_int(frame, sizes[0])
    _, page_bytes = handler_buffers.tensor_format(frame, node, tensor)
    if size < 1 or page_bytes % size:
        refuse(
            frame.where,
            node,
            f"items of {size} B in pages of {page_bytes} B: a page holds a whole "
            "number of items",
            f"give a {spelling('item_size')} that divides {page_bytes}",
        )
    items_moved(frame, node, tensor, size)
    first, count = (expressions.integer(frame, arg) for arg in node.args)
    used = expressions.uses(frame, node.args[0]) | expressions.uses(frame, node.args[1])
    at_submit(
        frame,
        node,
        (
            "items",
            tensor,
            size,
            side_text(frame, node.args[0]),
            side_text(frame, node.args[1]),
            "1",
        ),
    )
    return Value(
        "range",
        first=first,
        count=count,
        tensor=tensor,
        cxx=" ".join(sorted(used)),
        items=(size, page_bytes // size),
    )


def partition(frame, node, name, call):
    """`part = self.partition(pages)`: this core's part by rank; `part =
    b.partition(pages)`: by its place along broadcast b's row or column."""
    index, length = "km_rank", "km_size"
    if isinstance(call.func.value, ast.Name) and call.func.value.id != "self":
        along = frame.scope[call.func.value.id].pattern
        index, length = (
            ("km_self_x", "km_rect_columns")
            if along == "row"
            else ("km_self_y", "km_rect_rows")
        )
    if len(call.args) != 1 or call.keywords:
        refuse(frame.where, node, "partition(pages) takes one range", "give one")
    argument = call.args[0]
    if is_range(frame, argument):
        pages = range_of(frame, argument)
        handler_buffers.pages_only(frame, node, pages)
        if pages.rect is not None and str(pages.rect[0]) != "1":
            refuse(
                frame.where,
                node,
                "a part is a run of pages, and a rectangle of rows is not one",
                f"partition one row ({spelling('tiles')}[r:r + 1, c0:c1]), whole "
                "rows, or a page range",
            )
        first, count = pages.first, pages.count
    else:
        first, count = "0", expressions.integer(frame, argument)
    frame.scope[name] = Value("partition", name, first=first)
    text = f"const km::Partition {name} = km::partition({count}, {index}, {length});"
    uses = text_uses(count, frame) | {index, length}
    return [Glue(text, {name}, uses)], [Glue(text, {name}, uses)]


def range_of(frame, node):
    """A page range: tensor.pages, tensor.pages(first, count), tensor[part]."""
    if isinstance(node, ast.Name):
        return frame.scope[node.id]
    if isinstance(node, ast.Attribute):
        tensor = node.value.id
        return Value(
            "range", first="0", count=frame.counts[tensor], tensor=tensor, cxx=tensor
        )
    if isinstance(node, ast.Call) and node.func.attr == spelling("items"):
        return items_of(frame, node)
    if isinstance(node, ast.Call):
        tensor = node.func.value.id
        if len(node.args) != 2 or node.keywords:
            refuse(frame.where, node, "pages(first, count)", "give both")
        first = expressions.integer(frame, node.args[0])
        known = expressions.folded_int(frame, node.args[1])  # WIDE // 2: 32
        if known is not None and known >= 0:
            count, counted = str(known), ""
        else:
            count = expressions.integer(frame, node.args[1])
            counted = (
                f"pages(first, {ast.unparse(node.args[1])}): its count is not ints "
                "the build knows (+ - * // of literals and constants)"
            )
        used = expressions.uses(frame, node.args[0]) | expressions.uses(
            frame, node.args[1]
        )
        return Value(
            "range",
            first=first,
            count=count,
            tensor=tensor,
            cxx=" ".join(sorted(used)),
            counted=counted,
            written=ast.unparse(node),
        )
    if is_tiles(frame, node.value):
        return rectangle(frame, node)
    tensor = node.value.id
    part = node.slice
    if isinstance(part, ast.Name) and part.id in frame.scope:
        value = frame.scope[part.id]
        if value.kind == "partition":
            return Value(
                "range",
                first=value.first_text(value.cxx),
                count=f"{value.cxx}.len",
                tensor=tensor,
                cxx=part.id,
            )
    refuse(
        frame.where,
        node,
        "a tensor is in DRAM: a slice is not a value",
        f"take a part: {tensor}[self.{spelling('partition')}(...)], or use a "
        f"stream or {spelling('read')}()",
    )


def rectangle(frame, node):
    """`t.tiles[r0:r1, c0:c1]`: the range of the rectangle's tiles, row-major (a
    row: one run of pages; rows: runs of its columns, a row of t apart);
    `t.tiles[r0:r1:s, c0:c1]` its rows s rows of t apart (stepped_side)."""
    tensor = node.value.value.id
    example = frame.tensors.get(tensor)
    sides = node.slice.elts if isinstance(node.slice, ast.Tuple) else []
    if (
        len(sides) != 2
        or not all(isinstance(side, ast.Slice) for side in sides)
        or sides[1].step is not None
        or not all(side.lower is not None and side.upper is not None for side in sides)
    ):
        refuse(
            frame.where,
            node,
            "a rectangle of tiles is two slices, rows then columns, each with "
            "its start and end, a step on the rows alone",
            f"write {tensor}.{spelling('tiles')}[r0:r1, c0:c1] or [r0:r1:s, " "c0:c1]",
        )
    if (
        not front.is_tensor(example)
        or example.layout != "tile"
        or len(example.shape) < 2
    ):
        refuse(
            frame.where,
            node,
            f"{tensor}'s tiles: the program is made without an example tensor "
            "in tiles of at least two dimensions",
            f'make it with {tensor}=a tensor of the shape, layout "tile"',
        )
    columns_of = getattr(frame, "column_words", {}).get(tensor)
    if columns_of is None:
        columns_of = example.shape[-1] // 32
        frame.widths[tensor] = columns_of  # the host's check at submit
    texts, lengths, bounds = [], [], []
    for what, side in zip(("rows", "columns"), sides):
        low, high = expressions.integer(frame, side.lower), expressions.integer(
            frame, side.upper
        )
        texts.append(low)
        known = side_length(frame, side)
        if known is not None and known < 1:
            refuse(
                frame.where,
                node,
                f"the {what} {ast.unparse(side.lower)} : {ast.unparse(side.upper)} "
                f"of {tensor}'s tiles are {known} long: a slice whose end is before "
                "its start (reversed) or at it (empty) is no tiles",
                "write the start before the end: r0 < r1, c0 < c1",
            )
        if known is None:  # its length checked at submit where the call gives it
            at_submit(
                frame,
                node,
                (
                    "side",
                    what,
                    side_text(frame, side.lower),
                    side_text(frame, side.upper),
                ),
            )
        lengths.append((known, f"(({high}) - ({low}))"))
        bounds.append(known if known is not None else side_bound(frame, side))
    step = row_step(frame, node, sides[0], columns_of)
    if step is not None:  # rows s apart: their count, its most
        lengths[0], bounds[0] = stepped_side(frame, sides[0])
    rows, columns = lengths
    first = f"(({texts[0]}) * {columns_of} + ({texts[1]}))"
    if rows[0] is not None and columns[0] is not None:
        count = str(rows[0] * columns[0])
    elif rows[0] == 1:  # one row: its columns
        count = columns[1]
    else:
        count = (
            f"({rows[1] if rows[0] is None else rows[0]} * "
            f"{columns[1] if columns[0] is None else columns[0]})"
        )
    used = {columns_of} if isinstance(columns_of, str) else set()
    for side in sides:
        used |= expressions.uses(frame, side.lower) | expressions.uses(
            frame, side.upper
        )
    for part in (sides[0].step,) if step is not None else ():
        used |= expressions.uses(frame, part)
    if step is None and (rows[0] == 1 or columns[0] == columns_of):  # one run
        # (a width given at each submit is never known to equal a side)
        rect = None
    else:  # rows of columns; either side a word, read at run time
        rect = (
            rows[1] if rows[0] is None else str(rows[0]),
            columns[1] if columns[0] is None else columns[0],
            columns_of if step is None else step,
        )
    return Value(
        "range",
        first=first,
        count=count,
        tensor=tensor,
        cxx=" ".join(sorted(used)),
        rect=rect,
        bounds=tuple(bounds),
        sides=(
            rows[1] if rows[0] is None else str(rows[0]),
            columns[1] if columns[0] is None else columns[0],
        ),
        tiles=True,
        columns_first=known_end(frame, sides[1].lower),
        counted=" and ".join(
            f"the {what} {ast.unparse(side.lower)} : {ast.unparse(side.upper)}: its "
            "ends are not one name plus ints"
            for what, side, length in zip(("rows", "columns"), sides, lengths)
            if length[0] is None
        ),
        written=ast.unparse(node),
        corner=(
            f"{ast.unparse(sides[0].lower)} : {ast.unparse(sides[0].upper)}",
            ast.unparse(sides[1].lower),
        ),
    )


SIGNS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//"}


def side_text(frame, node):
    """`node` as Python of the handler's int parameters, every other part an int
    the build knows, for a check at submit; None if it reads anything else (a
    name of its lines, a word, the rank)."""
    if isinstance(node, ast.Name) and any(
        p.name == node.id and p.kind == "int" for p in frame.params
    ):
        return node.id
    known = expressions.folded_int(frame, node)
    if known is not None:
        return str(known)
    if isinstance(node, ast.BinOp) and type(node.op) in expressions.FOLDS:
        left, right = side_text(frame, node.left), side_text(frame, node.right)
        if left is not None and right is not None:
            return f"({left} {SIGNS[type(node.op)]} {right})"
    return None


def at_submit(frame, node, check):
    """Keeps `check` for the handler's submit (front.HandlerMethod.bounds_checked)
    where every part of it is known there and its line runs at every call (in no
    if, loop body or branch of the handler)."""
    if None in check or not every_call(frame, node):
        return
    entry = [ast.unparse(node), *check]
    if entry not in frame.bounds:
        frame.bounds.append(entry)


def every_call(frame, node):
    """Whether `node` runs at every call of the handler: not in the body or else of
    an if, a loop or a conditional expression (a loop's iterable runs)."""
    if frame.definition is None:
        return False
    for outer in ast.walk(frame.definition):
        if isinstance(outer, (ast.If, ast.For, ast.While)):
            parts = [*outer.body, *outer.orelse]
        elif isinstance(outer, ast.IfExp):
            parts = [outer.body, outer.orelse]
        else:
            continue
        if any(inner is node for part in parts for inner in ast.walk(part)):
            return False
    return True


def known_end(frame, end):
    """A slice end the build knows as an int, else None."""
    try:
        value = ast.literal_eval(end)
    except ValueError:
        return expressions.known_int(frame, end)
    return value if isinstance(value, int) else None


def row_step(frame, node, side, columns_of):
    """A rectangle's rows' step, s of `r0:r1:s` (an int or a word): the tensor
    pages from one of its rows to the next (s x t's columns), or None for no step
    or a step of 1."""
    if side.step is None or expressions.known_int(frame, side.step) == 1:
        return None
    known = expressions.known_int(frame, side.step)
    if known is not None and known < 1:
        refuse(
            frame.where, node, "a rectangle's rows' step is 1 or more", "give s >= 1"
        )
    if known is not None and isinstance(columns_of, int):
        return known * columns_of
    if known is not None:  # the tensor's width given at each submit
        return f"({known} * {columns_of})"
    return f"(({expressions.integer(frame, side.step)}) * {columns_of})"


def side_bound(frame, side):
    """The most a slice whose length is a word can be, where the build knows it:
    n of `c:min(c + n, e)` (either order of min's ints), else None."""
    upper = side.upper
    if not (
        isinstance(upper, ast.Call)
        and isinstance(upper.func, ast.Name)
        and upper.func.id == spelling("min")
    ):
        return None
    found = [
        expressions.known_int(frame, arg.right)
        for arg in upper.args
        if isinstance(arg, ast.BinOp)
        and isinstance(arg.op, ast.Add)
        and ast.dump(arg.left) == ast.dump(side.lower)
    ]
    known = [n for n in found if n is not None]
    return min(known) if known else None


def side_length(frame, side):
    """A slice's length where the build knows it: its ends one part the build does
    not know (or none) plus ints, the same part at both (`a:a + n`, `1:J + 1`,
    `j - 1:j + 1`)."""
    if side.lower is None or side.upper is None:
        return None
    low, first = end_parts(frame, side.lower)
    high, last = end_parts(frame, side.upper)
    return last - first if low == high else None


def end_parts(frame, end):
    """A slice's end as (the part the build does not know, as ast.dump, or None; the
    int added to it): `a`, `a + 2`, `a - 1`, `2 + a`, `J + 1`."""
    known = expressions.folded_int(frame, end)
    if known is not None:
        return None, known
    if isinstance(end, ast.BinOp) and isinstance(end.op, (ast.Add, ast.Sub)):
        right = expressions.folded_int(frame, end.right)
        if right is not None:
            part, known = end_parts(frame, end.left)
            return part, known + right if isinstance(end.op, ast.Add) else known - right
        left = expressions.folded_int(frame, end.left)
        if left is not None and isinstance(end.op, ast.Add):
            part, known = end_parts(frame, end.right)
            return part, left + known
    return ast.dump(end), 0


def stepped_items(frame, node, tensor):
    """`t.items(first=f, count=n, size=B, step=s)`: items f, f + s, ... (n of
    them), a `km::Walk` of one item a row and a row pitch of s items."""
    names = ("entries_first", "entries_count", "item_size", "item_step")
    given = {k.arg: k.value for k in node.keywords}
    words = [spelling(n) for n in names]
    if set(given) - set(words) or not set(words[:3]) <= set(given):
        refuse(
            frame.where,
            node,
            f"{spelling('items')}({', '.join(w + '=' for w in words)}): the first "
            "item, their count, the bytes an item and the items from one to the "
            f"next ({spelling('item_step')}= 1 if left out)",
            "give first=, count= and size= (and step=), and no other keyword",
        )
    size = expressions.constant_int(frame, given[words[2]])
    _, page_bytes = handler_buffers.tensor_format(frame, node, tensor)
    if size < 1 or page_bytes % size:
        refuse(
            frame.where,
            node,
            f"items of {size} B in pages of {page_bytes} B: a page holds a whole "
            "number of items",
            f"give a {spelling('item_size')} that divides {page_bytes}",
        )
    items_moved(frame, node, tensor, size)
    bounded = {}
    for word in (words[1], words[3]):
        if word not in given:
            continue
        value = given[word]
        known = expressions.known_int(frame, value)
        if known is None and not expressions.is_word(frame, value):
            refuse(
                frame.where,
                node,
                f"{word}= {ast.unparse(value)}: the build bounds an int or a word "
                "of the handler, not this",
                f"give {word}= an int or a word (a parameter, an entry's word)",
            )
        if known is not None and known < 1:
            refuse(
                frame.where,
                node,
                f"{word}= {known}: 1 or more",
                f"give {word}= >= 1",
            )
        bounded[word] = known
    first = expressions.integer(frame, given[words[0]])
    count = expressions.integer(frame, given[words[1]])
    step = expressions.integer(frame, given[words[3]]) if words[3] in given else "1"
    used = set()
    for value in given.values():
        used |= expressions.uses(frame, value)
    rect = None if step == "1" else (count, 1, step)
    at_submit(
        frame,
        node,
        (
            "items",
            tensor,
            size,
            side_text(frame, given[words[0]]),
            side_text(frame, given[words[1]]),
            side_text(frame, given[words[3]]) if words[3] in given else "1",
        ),
    )
    return Value(
        "range",
        first=first,
        count=count,
        tensor=tensor,
        cxx=" ".join(sorted(used)),
        items=(size, page_bytes // size),
        rect=rect,
        bounds=(bounded[words[1]], 1),
        sides=(count, 1),
    )


def stepped_side(frame, side):
    """The rows of `r0:r1:s`, (their count where the build knows it, its C++),
    and their most where the build knows it: r1 = r0 + n * s (n rows), r1 =
    min(r0 + n * s, e) (at most n), or a step known and r1 - r0 known or bound
    (the rows of that many)."""
    low, high = expressions.integer(frame, side.lower), expressions.integer(
        frame, side.upper
    )
    step = expressions.integer(frame, side.step)
    text = f"(((({high}) - ({low})) + ({step}) - 1) / ({step}))"

    def rows_of(end):  # n of r0 + n * s (either order of the product), or None
        if not (
            isinstance(end, ast.BinOp)
            and isinstance(end.op, ast.Add)
            and ast.dump(end.left) == ast.dump(side.lower)
            and isinstance(end.right, ast.BinOp)
            and isinstance(end.right.op, ast.Mult)
        ):
            return None
        factors = (end.right.left, end.right.right)
        for factor, other in (factors, factors[::-1]):
            if ast.dump(other) == ast.dump(side.step):
                return expressions.known_int(frame, factor)
        return None

    known = rows_of(side.upper)
    if known is not None:
        return (known, text), known
    upper = side.upper
    if (
        isinstance(upper, ast.Call)
        and isinstance(upper.func, ast.Name)
        and upper.func.id == spelling("min")
    ):
        found = [n for n in map(rows_of, upper.args) if n is not None]
        if found:
            return (None, text), min(found)
    each = expressions.known_int(frame, side.step)
    length = side_length(frame, side)
    if each is not None and length is not None:
        return (-(-length // each), text), -(-length // each)
    bound = side_bound(frame, side)
    if each is not None and bound is not None:
        return (None, text), -(-bound // each)
    return (None, text), None
