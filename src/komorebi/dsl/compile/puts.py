"""A mover's put of a page or an item into another core's buffer, and the bump of its
count."""

import ast

from komorebi.dsl.compile import expressions, handler_buffers, scope
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import check_name, refuse
from komorebi.dsl.names import spelling
from komorebi.dsl.nodes import Line


def bump_line(frame, node, name, call):
    """`slot = self.f.bump(i)`: word i of field f, then one more, on the CPU that
    owns f's words: an own field's push= mover, or a word field's unpack CPU."""
    field_ = scope.field_name(frame, call.func.value)
    if len(call.args) != 1 or call.keywords:
        refuse(
            frame.where, node, f"{spelling('bump')}(i): one word's index", "give one"
        )
    check_name(frame.where, node, name)
    uses = expressions.uses(frame, call.args[0])
    if field_ in frame.compiled.word_fields:
        index = expressions.integer(frame, call.args[0])
        frame.scope[name] = Value("word", name, mover="unpack")
        word = f"km_field_{field_}[{index}]"
        text = f"uint32_t {name} = 0;\nUNPACK(({name} = {word}++));"
        return [], [Line(text, uses | {f"km_field_{field_}"})]
    mover = {"mover_a": "A", "mover_b": "B"}.get(frame.compiled.own.get(field_))
    if field_ not in frame.compiled.own or mover is None:
        refuse(
            frame.where,
            node,
            f"self.{field_}'s words are no one CPU's: a bump runs on the CPU that "
            "owns them",
            f"declare it km.{spelling('own')}(push=\"mover_a\") or "
            '(push="mover_b"), or a field of km.u32 words (unpack\'s)',
        )
    index = expressions.mover_integer(frame, node, call.args[0], mover)
    frame.scope[name] = Value("word", name, mover=mover)
    text = (
        f"uint32_t {name} = 0;\n{{\n    volatile tt_l1_ptr uint32_t* km_bumped = "
        f"km::l1(get_write_ptr(kField{camel(field_)}));\n    {name} = "
        f"km_bumped[{index}]++;\n}}"
    )
    return [Line(text, uses, mover=mover)], []


def put_line(frame, node, call):
    """`t.items(size=B).put(self.f, at=i)`: B bytes of own field f (or of its item
    k, `self.f[k]`) at item i of t, by a mover (on=, mover B unstated); unwaited,
    the mover's puts waited for at the handler's end (km::put_bytes, put_wait)."""
    items = call.func.value
    if not (
        isinstance(items, ast.Call)
        and isinstance(items.func, ast.Attribute)
        and items.func.attr == spelling("items")
        and isinstance(items.func.value, ast.Name)
        and getattr(frame.scope.get(items.func.value.id), "kind", None) == "tensor"
        and not items.args
        and [k.arg for k in items.keywords] == [spelling("item_size")]
    ):
        refuse(
            frame.where,
            node,
            f"a put is `t.{spelling('items')}({spelling('item_size')}=B)."
            f"{spelling('put')}(source, {spelling('put_at')}=i)`, t a tensor or a "
            "region parameter",
            "give the tensor's items by their size alone",
        )
    tensor = items.func.value.id
    size = expressions.constant_int(frame, items.keywords[0].value)
    _, page_bytes = handler_buffers.tensor_format(frame, node, tensor)
    if size < 4 or size % 4 or page_bytes % size:
        refuse(
            frame.where,
            node,
            f"items of {size} B in pages of {page_bytes} B: a put is whole words, "
            "a page a whole number of items",
            f"give a {spelling('item_size')} of 4 B steps dividing {page_bytes}",
        )
    keywords = {k.arg: k.value for k in call.keywords}
    if (
        len(call.args) != 1
        or set(keywords) - {spelling("put_at"), "on"}
        or (spelling("put_at") not in keywords)
    ):
        refuse(
            frame.where,
            node,
            f"{spelling('put')}(source, {spelling('put_at')}=i), on= a mover",
            "give the source, the item and (if not mover B) the mover",
        )
    on = keywords.get("on")
    on = on.value if isinstance(on, ast.Constant) else "mover_b" if on is None else ""
    mover = {"mover_a": "A", "mover_b": "B"}.get(on)
    if mover is None:
        refuse(
            frame.where,
            node,
            "a put is a mover's NoC write",
            'write on="mover_a" or on="mover_b" (mover B when not given)',
        )
    expressions.mover_word(frame, node, call.args[0], mover)
    source, uses = put_source(frame, node, call.args[0], size)
    at = keywords[spelling("put_at")]
    index = expressions.mover_integer(frame, node, at, mover)
    uses |= expressions.uses(frame, at)
    per = page_bytes // size
    if mover not in frame.putters:
        frame.putters.append(mover)
    text = (
        f"if (!km::put_bytes(movement, command, {camel(tensor)}({tensor}), "
        f"({index}) / {per}, (({index}) % {per}) * {size}, {size}, {source})) {{\n"
        "    return false;\n}"
    )
    return [Line(text, uses | {tensor}, mover=mover)], []


def put_source(frame, node, source, size):
    """A put's source: an own field's L1 start, or item k of it (`self.f[k]`)."""
    item = None
    if (
        isinstance(source, ast.Name)
        and getattr(frame.scope.get(source.id), "kind", None) == "page"
    ):  # the page a mover reads of a stream
        return source.id, {source.id}
    if isinstance(source, ast.Subscript):
        source, item = source.value, source.slice
    field_ = scope.field_name(frame, source)
    if field_ not in frame.compiled.own:
        refuse(
            frame.where,
            node,
            "a put's source is an own field (km.own(): the bodies' buffer), whole "
            "or an item of it",
            f"put self.f or self.f[k], f declared = km.{spelling('own')}(...)",
        )
    address = f"get_write_ptr(kField{camel(field_)})"
    if item is None:
        return address, set()
    return (
        f"{address} + ({expressions.integer(frame, item)}) * {size}",
        expressions.uses(frame, item),
    )
