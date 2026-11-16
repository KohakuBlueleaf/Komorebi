"""A page's words: one page read as a word or as words of every kernel, and a mover's
word written as one page."""

import ast

from komorebi.dsl.compile import layout
from komorebi.dsl.compile.cxx import camel, range_uses
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import (
    WAIT_DATA,
    WAIT_INPUT,
    check_name,
    line_text,
    refuse,
    wait,
)
from komorebi.dsl.names import spelling
from komorebi.dsl.nodes import Glue, Line


def read_word(frame, node, name, pages):
    """`word = pages.read()` of one page: its first word, on mover B, from its own
    L1 page (read, settled, the L1 cache invalidated)."""
    frame.pages_only(node, pages)
    if pages.count != "1":
        refuse(frame.where, node, "a word is one page", "read pages(first, 1)")
    check_name(frame.where, node, name)
    dtype, page_bytes = frame.tensor_format(node, pages.tensor)
    buffer = layout.other_buffer(
        frame.compiled,
        "kWordB", "word", dtype, page_bytes, 1, f"line {node.lineno}: "
        f"{line_text(frame.where, node)} (mover B's word page)",
    )  # fmt: skip
    buffer.page_bytes = max(buffer.page_bytes, page_bytes)
    tensor = f"{camel(pages.tensor)}({pages.tensor})"
    text = (
        f"uint32_t {name} = 0;\n"
        f"if (!km::read_pages(movement, command, {tensor}, {pages.first}, 1,\n"
        f"                    get_write_ptr({buffer.name}))) {{\n"
        "    return false;\n"
        "}\n"
        "noc_async_read_barrier();\n"
        "invalidate_l1_cache();\n"
        f"{name} = km::l1(get_write_ptr({buffer.name}))[0];"
    )  # fmt: skip
    frame.moved(node, pages.tensor, "word read")
    frame.compiled.includes.add("ask")
    frame.scope[name] = Value("word", name, mover="B")
    return [Line(text, range_uses(pages, frame), mover="B")], []


def read_words(frame, node, target, value):
    """`a, b = pages.read()[0:2]` of one page: its words 0 and 1 as ints of
    every kernel. Mover A reads the page into a page it hands compute (unpack
    takes the words and hands them to math and pack by mailbox); mover B reads
    its own."""
    call = value.value if isinstance(value, ast.Subscript) else None
    part = value.slice if call is not None else None
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == spelling("read")
        and not call.args
        and not call.keywords
        and frame.is_range(call.func.value)
        and isinstance(part, ast.Slice)
        and part.step is None
        and part.upper is not None
        and all(isinstance(name, ast.Name) for name in target.elts)
    ):
        refuse(
            frame.where,
            node,
            "names are given a page's words, by a constant slice",
            f"write `first, count = t.{spelling('pages')}(i, 1)."
            f"{spelling('read')}()[0:2]`",
        )
    lower = 0 if part.lower is None else frame.constant_int(part.lower)
    upper = frame.constant_int(part.upper)
    names = [name.id for name in target.elts]
    pages = frame.range_of(call.func.value)
    frame.pages_only(node, pages)
    if pages.count != "1":
        refuse(frame.where, node, "words are of one page", "read pages(first, 1)")
    dtype, page_bytes = frame.tensor_format(node, pages.tensor)
    if not 0 <= lower < upper <= page_bytes // 4 or upper - lower != len(names):
        refuse(
            frame.where,
            node,
            f"words {lower} to {upper - 1} of a page of {page_bytes // 4} words "
            f"for {len(names)} names",
            "give a slice of the page with a name each word",
        )
    for name in names:
        check_name(frame.where, node, name)
        frame.scope[name] = Value("int", name)
    what = f"line {node.lineno}: {line_text(frame.where, node)}"
    handed = layout.other_buffer(
        frame.compiled,
        "kWords",
        "word",
        dtype,
        page_bytes,
        1,
        f"{what} (mover A's, to compute)",
    )
    own = layout.other_buffer(
        frame.compiled,
        "kWordB",
        "word",
        dtype,
        page_bytes,
        1,
        f"{what} (mover B's word page)",
    )
    for buffer in (handed, own):
        buffer.page_bytes = max(buffer.page_bytes, page_bytes)
    declared = "\n".join(f"[[maybe_unused]] uint32_t {name} = 0;" for name in names)
    taken = "\n".join(
        f"    {name} = km_page_words[{lower + n}];" for n, name in enumerate(names)
    )
    tensor = f"{camel(pages.tensor)}({pages.tensor})"

    def mover(buffer, reserve):
        opening = (
            f"if (!movement.reserve({buffer}, 1) ||\n    " if reserve else ("if (")
        )
        return (
            f"{declared}\n"
            f"{opening}!km::read_pages(movement, command, {tensor}, "
            f"{pages.first}, 1,\n                    get_write_ptr({buffer}))) {{\n"
            "    return false;\n"
            "}\n"
            "noc_async_read_barrier();\n"
            "invalidate_l1_cache();\n"
            "{\n"
            "    volatile tt_l1_ptr uint32_t* km_page_words =\n"
            f"        km::l1(get_write_ptr({buffer}));\n"
            f"{taken}\n"
            "}"
        )

    uses = range_uses(pages, frame)
    movement = [
        Line(mover("kWords", True) + "\ncb_push_back(kWords, 1);", uses, mover="A"),
        Glue(mover("kWordB", False), set(names), uses, mover="B"),
    ]
    hands = "\n".join(
        f"    compute.hand(ckernel::ThreadId::{cpu}ThreadId, {name});"
        for name in names
        for cpu in ("Math", "Pack")
    )
    heard = "\n".join(
        f"{name} = compute.handed(ckernel::ThreadId::UnpackThreadId);" for name in names
    )
    compute = [
        Line(
            f"{declared}\n"
            "if (!compute.wait_front(kWords, 1)) {\n"
            "    return;\n"
            "}\n"
            "#if defined(TRISC_UNPACK)\n"
            "{\n"
            "    const volatile uint32_t* km_page_words =\n"
            "        reinterpret_cast<const volatile uint32_t*>(\n"
            "            get_tile_l1_byte_address(kWords, 0)\n"
            "        );\n"
            '    asm volatile("fence" ::: "memory");\n'
            f"{taken}\n"
            f"{hands}\n"
            "}\n"
            "#else\n"
            f"{heard}\n"
            "#endif\n"
            "cb_pop_front(kWords, 1);"
        )
    ]
    frame.moved(node, pages.tensor, "word read")
    frame.compiled.includes.add("ask")
    frame.handover(node, ", ".join(names), "unpack", ("math", "pack"))
    wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_DATA)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_DATA)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    return movement, compute


def write_word(frame, node, pages, value):
    """`pages.write(word)`: a mover's word as one page (the word first, zeros
    after) from its own L1 page, written as a page write is, settled."""
    if value.mover not in ("A", "B"):
        refuse(
            frame.where,
            node,
            f"{value.cxx} is compute's word; a page of words is a mover's",
            'make the word on="mover_a" or on="mover_b"',
        )
    if pages.count != "1":
        refuse(frame.where, node, "a word is one page", "write pages(first, 1)")
    dtype, page_bytes = frame.tensor_format(node, pages.tensor)
    if page_bytes % 4:
        refuse(
            frame.where,
            node,
            f"a page of {page_bytes} B is not whole words",
            "use words",
        )
    buffer = layout.other_buffer(
        frame.compiled,
        f"kWord{value.mover}", "word", dtype, page_bytes, 1, f"line {node.lineno}: "
        f"{line_text(frame.where, node)} (mover {value.mover}'s word page)",
    )  # fmt: skip
    buffer.page_bytes = max(buffer.page_bytes, page_bytes)
    tensor = f"{camel(pages.tensor)}({pages.tensor})"
    text = (
        "{\n"
        "    volatile tt_l1_ptr uint32_t* km_word_page =\n"
        f"        km::l1(get_write_ptr({buffer.name}));\n"
        f"    km_word_page[0] = {value.cxx};\n"
        "    for (uint32_t km_rest = 1; "
        f"km_rest < {page_bytes // 4}; km_rest++) {{\n"
        "        km_word_page[km_rest] = 0;\n"
        "    }\n"
        f"    if (!km::write_pages(movement, command, {tensor}, {pages.first}, 1,\n"
        f"                         get_write_ptr({buffer.name}))) {{\n"
        "        return false;\n"
        "    }\n"
        "    noc_async_write_barrier();\n"
        "}"
    )  # fmt: skip
    frame.moved(node, pages.tensor, "page write")
    frame.compiled.includes.add("ask")
    uses = range_uses(pages, frame) | {value.cxx}
    return [Line(text, uses, mover=value.mover)], []
