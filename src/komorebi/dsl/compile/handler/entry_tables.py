"""A page's table of entries (pages.entries(...)): the page read on every CPU, a loop
over its entries, an entry's words."""

import ast

from komorebi.dsl.compile import expressions, handler_buffers, layout, scope
from komorebi.dsl.compile.casts import broadcasts
from komorebi.dsl.compile.cxx import camel, range_uses, text_uses
from komorebi.dsl.compile.pages import ranges
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import (
    WAIT_DATA,
    WAIT_INPUT,
    check_name,
    line_text,
    refuse,
    wait,
)
from komorebi.dsl.compile.tiles import writes
from komorebi.dsl.front import front
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Block, Glue, Group, Line, has_line


def entries_call(frame, node):
    """`node` when it is `<a page range>.entries(...)`, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == spelling("entries")
        and ranges.is_range(frame, node.func.value)
    ):
        return node
    return None


def entries_loop(frame, node):
    """`for e in pages.entries(...)`, `for e in jobs` or `for e in jobs[a:b]`:
    the same loop on each CPU, its entry's words ints of each (`e.word`); a
    slice's ends clamped to the count, as Python's; one entry known at build is
    a block of no loop."""
    iterator = node.iter
    call = entries_call(frame, iterator)
    lower = upper = pop = None
    if call is not None:
        table, movement, compute, pop = entries_table(frame, node, call)
    else:
        table = entries_named(frame, iterator)
        movement, compute = [], []
        if isinstance(iterator, ast.Subscript):
            part = iterator.slice
            if not isinstance(part, ast.Slice) or part.step is not None:
                refuse(
                    frame.where,
                    node,
                    "a part of a table is a slice of its entries, with no step",
                    f"write for e in {iterator.value.id}[a:b]",
                )
            lower, upper = part.lower, part.upper
    if not isinstance(node.target, ast.Name):
        refuse(frame.where, node, "one name an entry", "write for e in table")
    name = node.target.id
    check_name(frame.where, node, name)
    if name in frame.scope:
        refuse(frame.where, node, f"{name} is already defined", "use a new name")
    info, source = table.table, table.cxx
    count = info["count"]
    uses = {source} | text_uses(count, frame)

    def clamped(bound):
        text = expressions.integer(frame, bound)
        uses.update(expressions.uses(frame, bound))
        return f"({text} < {count} ? {text} : {count})"

    low = "0" if lower is None else clamped(lower)
    high = count if upper is None else clamped(upper)
    index = f"km_{name}_at"
    cls = info["class"]
    frame.scope[name] = Value("entry", name, table=info)
    for word in cls.words:  # generated: read by text_uses in ranges' C++
        frame.scope[f"km_{name}_{word}"] = Value("int", f"km_{name}_{word}")
    loop = {"name": index, "first": low, "end": high, "casts": []}
    frame.loops.append(loop)
    before = set(frame.scope)
    body_m, body_c = frame.statements(node.body)
    scope.scoped(frame, "loop", node, before)
    frame.loops.pop()
    frame.scope.pop(name)
    for word in cls.words:
        frame.scope.pop(f"km_{name}_{word}")
    mover = None
    if call is not None:  # a table read at this line: only where it is used
        movement, compute, pop, mover = entries_users(
            frame, movement, compute, pop, body_m, body_c
        )
    alone = count == "1" and lower is None and upper is None
    offset = (
        str(info["first"]) if alone else (f"{info['first']} + {info['size']} * {index}")
    )
    entry = [
        Glue(
            f"const volatile tt_l1_ptr uint32_t* {name} = {source} + {offset};",
            {name},
            {source},
        )
    ]
    entry += [
        Glue(
            f"const uint32_t km_{name}_{word} = "
            f"{name}[k{cls.__name__}{camel(word)}];",
            {f"km_{name}_{word}"},
            {name},
        )
        for word in cls.words
    ]
    head = head_c = "{"
    if not alone:
        head = f"for (uint32_t {index} = {low}; {index} < {high}; {index}++) {{"
        head_c = head
        if frame.compute_waits:
            head_c = (
                f"for (uint32_t {index} = {low}; km_ok && {index} < {high}; "
                f"{index}++) {{"
            )
    if body_m:
        movement.append(Block(head, "}", [*entry, *body_m], uses, mover=mover))
    if body_c:
        compute.append(Block(head_c, "}", [*entry, *body_c], uses | {"km_ok"}))
    if pop is not None:
        compute.append(pop)
    for broadcast in loop["casts"]:
        m, c = broadcasts.broadcast_end(frame, broadcast, high, uses)
        movement += m
        compute += c
    m, c = broadcasts.counted_ends(frame, loop)
    movement, compute = movement + m, compute + c
    movement += writes.fed_moves(frame, node, loop, body_m)
    return movement, compute


def entries_made(frame, node, name, call):
    """`jobs = pages.entries(...)` at the handler's top: the table, read at this
    line on every CPU and kept to the handler's end (compute frees its page)."""
    if len(frame.blocks) != 1:
        refuse(
            frame.where,
            node,
            "a named table is read at the handler's top and kept to its end",
            f"name it before the loops and ifs, or loop over "
            f"pages.{spelling('entries')}(...) itself",
        )
    check_name(frame.where, node, name)
    value, movement, compute, pop = entries_table(frame, node, call)
    frame.scope[name] = value
    frame.closing.append(pop)
    return movement, compute


def entries_named(frame, node):
    """The table of entries `node` names (`jobs` or `jobs[a:b]`), or None."""
    base = node.value if isinstance(node, ast.Subscript) else node
    if isinstance(base, ast.Name):
        value = frame.scope.get(base.id)
        if value is not None and value.kind == "entries":
            return value
    return None


def entries_table(frame, node, call):
    """`pages.entries(Job, first=F, size=S, count=C)` (or fixed=N for count=):
    the one page read on every CPU, mover A's pushed to compute, mover B's its
    own; (the table's Value, movement nodes, compute nodes, compute's pop)."""
    if frame.dynamic:
        refuse(
            frame.where,
            node,
            "a dynamic handler is compute's lines alone: no mover reads a table",
            "read the table in a handler of another class",
        )
    cls = None
    if len(call.args) == 1 and isinstance(call.args[0], ast.Name):
        found = frame.names.get(call.args[0].id)
        if (
            isinstance(found, type)
            and issubclass(found, front.Entry)
            and found is not front.Entry
        ):
            cls = found
    keys = {k.arg: k.value for k in call.keywords}
    first_key, size_key, count_key, fixed_key = (
        spelling(f"entries_{kind}") for kind in ("first", "size", "count", "fixed")
    )
    if (
        cls is None
        or set(keys) - {first_key, size_key, count_key, fixed_key}
        or size_key not in keys
        or (count_key in keys) == (fixed_key in keys)
    ):
        refuse(
            frame.where,
            node,
            f"entries name their class (a km.{spelling('entry')} of the module), "
            f"their {size_key}= and either the word of their count ({count_key}=) "
            f"or their number ({fixed_key}=)",
            f"write {spelling('entries')}(Job, {first_key}=8, {size_key}=8, "
            f"{count_key}=0)",
        )
    size = expressions.constant_int(frame, keys[size_key])
    first = expressions.constant_int(frame, keys[first_key]) if first_key in keys else 0
    pages = ranges.range_of(frame, call.func.value)
    handler_buffers.pages_only(frame, node, pages)
    if pages.count != "1":
        refuse(frame.where, node, "a table of entries is one page", "read pages(i, 1)")
    dtype, page_bytes = handler_buffers.tensor_format(frame, node, pages.tensor)
    words = page_bytes // 4
    if size < len(cls.words):
        refuse(
            frame.where,
            node,
            f"{cls.__name__} names {len(cls.words)} words, and an entry is " f"{size}",
            f"give {size_key}= at least {len(cls.words)}",
        )
    most = (words - first) // size if words > first else 0
    if most < 1:
        refuse(
            frame.where,
            node,
            f"a page of {words} words holds no entry of {size} from word {first}",
            "give a first word and a size inside the page",
        )
    if count_key in keys:
        word = expressions.constant_int(frame, keys[count_key])
        if word >= words or first <= word < first + most * size:
            refuse(
                frame.where,
                node,
                f"the count's word {word} is not a word of the page before or "
                f"after its entries (words {first} to {first + most * size - 1} "
                f"of {words})",
                f"give {count_key}= a word outside the entries",
            )
    else:
        fixed = expressions.constant_int(frame, keys[fixed_key])
        if not 1 <= fixed <= most:
            refuse(
                frame.where,
                node,
                f"{fixed} entries in a page that holds {most}",
                f"give {fixed_key}= 1 to {most}",
            )
    known = frame.compiled.entries.setdefault(cls.__name__, cls.words)
    if known != cls.words:
        refuse(frame.where, node, f"two entry classes named {cls.__name__}", "name one")
    number = frame.tables
    frame.tables += 1
    name = f"km_entries{number}"
    what = f"line {node.lineno}: {line_text(frame.where, node)}"
    pushed = layout.other_buffer(
        frame.compiled,
        f"kEntries{number}", "word", dtype, page_bytes, 1, f"{what} (mover A's, "
        "to compute)",
    )  # fmt: skip
    own = layout.other_buffer(
        frame.compiled,
        f"kEntries{number}B",
        "word",
        dtype,
        page_bytes,
        1,
        f"{what} (mover B's)",
    )
    for buffer in (pushed, own):
        buffer.page_bytes = max(buffer.page_bytes, page_bytes)
    tensor = f"{camel(pages.tensor)}({pages.tensor})"

    def read(buffer, reserve):
        opening = f"!movement.reserve({buffer}, 1) ||\n    " if reserve else ""
        return (
            f"if ({opening}!km::read_pages(movement, command, {tensor}, "
            f"{pages.first}, 1,\n                    get_write_ptr({buffer}))) "
            "{\n    return false;\n}\n"
            "noc_async_read_barrier();\n"
            "invalidate_l1_cache();\n"
            f"const volatile tt_l1_ptr uint32_t* {name} =\n"
            f"    km::l1(get_write_ptr({buffer}));"
        )

    uses = range_uses(pages, frame)
    movement = [
        Line(
            read(pushed.name, True) + f"\ncb_push_back({pushed.name}, 1);",
            uses,
            mover="A",
        ),
        Glue(read(own.name, False), {name}, uses, mover="B"),
    ]
    compute = [
        Line(
            f"if (!compute.wait_front({pushed.name}, 1)) {{\n    return;\n}}\n"
            f"const volatile tt_l1_ptr uint32_t* {name} =\n"
            f"    km::l1(get_tile_address({pushed.name}, 0));\n"
            "invalidate_l1_cache();"
        )
    ]
    if count_key in keys:
        count = f"{name}_count"
        text = (
            f"const uint32_t {count} =\n"
            f"    {name}[{word}] <= {most} ? {name}[{word}] : 0;"
        )
        movement.append(Glue(text, {count}, {name}))
        compute.append(Glue(text, {count}, {name}))
        frame.scope[count] = Value("int", count)  # generated: read by text_uses
    else:
        count = str(fixed)
    handler_buffers.moved(frame, node, pages.tensor, "table read")
    frame.compiled.includes.add("ask")
    wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_DATA)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    table = {"class": cls, "first": first, "size": size, "count": count}
    pop = Line(f"cb_pop_front({pushed.name}, 1);")
    return Value("entries", name, table=table), movement, compute, pop


def entries_users(frame, movement, compute, pop, body_m, body_c):
    """A table's read kept on the CPUs whose lines in its loop use it: mover B
    reads its own copy, mover A one for itself and compute; (movement, compute,
    compute's pop, the loop's mover: "A", "B" or None for both)."""

    def mover_has_line(nodes, mover, inherited=None):
        for node in nodes:
            own = node.mover or inherited
            if isinstance(node, Line) and own in (None, mover):
                return True
            if (
                isinstance(node, Block | Group)
                and own in (None, mover)
                and mover_has_line(node.children, mover, own)
            ):
                return True
        return False

    on_compute = any(has_line(node) for node in body_c)
    on_a = on_compute or mover_has_line(body_m, "A")
    on_b = mover_has_line(body_m, "B")
    if on_a == on_b:  # both movers (or no line at all): the read as it is
        return movement, compute, pop, None
    mover = "A" if on_a else "B"
    kept = []
    for node in movement:
        if node.mover not in (None, mover):
            continue
        node.mover = mover
        kept.append(node)
    if not on_compute:
        compute, pop = [], None
    return kept, compute, pop, mover
