"""A handler's stamps page: an overlay handler's cycles on each mover, a resident
handler's math cycles of each tile function call, and the compute CPUs' words of a
handler of steps."""

import ast

from komorebi.dsl.compile import handler_buffers, layout, scope, tile_lookup
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.source import WAIT_DATA, refuse, wait
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Block, Glue, Group, Line

TIMED_WAITS = (  # (the call's start, the counter, an expression or a statement)
    ("compute.wait_front(", "km_waited_in", True),
    ("km::broadcast_receive(compute, ", "km_waited_in", True),
    ("compute.reserve_back(", "km_waited_out", True),
    ("tile_regs_acquire();", "km_waited_out", False),
    ("tile_regs_wait();", "km_waited_in", False),
)


def calls_steps(frame):
    """Whether the handler calls a step (`f(x, self.s[r])`, f a tile function, a
    field indexed second): its stamps then add the three CPUs' words."""
    return any(
        isinstance(node, ast.Call)
        and tile_lookup.tile_function_of(frame, node) is not None
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Subscript)
        and scope.field_name(frame, node.args[1].value) is not None
        for node in ast.walk(frame.definition)
    )


def stamps(frame):
    """`stamps="t"`: the overlay's resident case on each mover writes the
    command's sequence, the overlay, its load and call cycles and whether it was
    held to page 2 x core + mover of tensor parameter t (mover A 0, B 1)."""
    stamps = frame.method.stamps
    decorator = frame.definition.decorator_list[0]
    found = [p for p in frame.params if p.name == stamps and p.kind == "tensor"]
    if not frame.method.overlay or not found:
        refuse(
            frame.where,
            decorator,
            f"stamps={stamps!r} names no tensor parameter of an overlay handler",
            'write @km.handler(overlay=True, stamps="t") with t: km.Tensor',
        )
    dtype, page_bytes = handler_buffers.tensor_format(frame, decorator, stamps)
    if page_bytes < 20 or page_bytes % 4:
        refuse(
            frame.where,
            decorator,
            f"a stamps page is five words; {stamps}'s page is {page_bytes} B",
            f"make {stamps} a tensor of uint32 pages of at least five words",
        )
    slot = -(-page_bytes // 64) * 64  # mover B's page at the 64-B line a write keeps
    buffer = layout.other_buffer(
        frame.compiled,
        "kStamps", "word", dtype, slot, 2, "stamps: a page a mover"
    )  # fmt: skip
    buffer.page_bytes = max(buffer.page_bytes, slot)
    frame.compiled.stamps[frame.method.name] = (stamps, found[0].word)
    handler_buffers.moved(frame, decorator, stamps, "page write")


def tile_stamps(frame):
    """`stamps="t"` of a resident handler: the math CPU's cycles of each tile
    function call to page `rank` of tensor parameter t (the sequence, the calls,
    the empty spans, each call's span); compute's opening and closing lines and
    mover B's write of the page."""
    stamps = frame.method.stamps
    decorator = frame.definition.decorator_list[0]
    found = [p for p in frame.params if p.name == stamps and p.kind == "tensor"]
    if not found:
        refuse(
            frame.where,
            decorator,
            f"stamps={stamps!r} names no tensor parameter",
            f'write @km.handler(stamps="t") with t: km.{spelling("tensor")}',
        )
    dtype, page_bytes = handler_buffers.tensor_format(frame, decorator, stamps)
    if page_bytes < 4 * (SPANS_FIRST + 1) or page_bytes % 4:
        refuse(
            frame.where,
            decorator,
            f"a stamps page holds {SPANS_FIRST + 1} words or more; {stamps}'s "
            f"page is {page_bytes} B",
            f"make {stamps} a tensor of uint32 pages of {SPANS_FIRST + 1} words "
            "or more",
        )
    buffer = layout.other_buffer(
        frame.compiled,
        "kSpans", "word", dtype, page_bytes, 1, "stamps: the math CPU's spans"
    )  # fmt: skip
    steps = calls_steps(frame)  # then the three CPUs' words end the page
    least = SPANS_FIRST + 1 + CPU_WORDS
    if steps and page_bytes < 4 * least:
        refuse(
            frame.where,
            decorator,
            f"a stamps page of a handler of steps holds {least} words or more "
            f"(the compute CPUs' {CPU_WORDS} at its end); {stamps}'s page is "
            f"{page_bytes} B",
            f"make {stamps} a tensor of uint32 pages of {least} words or more",
        )
    frame.spans_most = page_bytes // 4 - SPANS_FIRST - (CPU_WORDS if steps else 0)
    cpus = page_bytes // 4 - CPU_WORDS
    handler_buffers.moved(frame, decorator, stamps, "page write")
    wait(frame.where, frame.compiled.waits, decorator, "mover B", WAIT_DATA)
    waits = (
        "\n// each compute CPU: its span and its waits (CPU_WORDS)\n"
        "#define KM_WAITED(counter, call) ({ const uint32_t km_w0 = km::wall(); "
        "const bool km_r = (call); counter += km::wall() - km_w0; km_r; })\n"
        "#define KM_WAITED_DO(counter, call) do { const uint32_t km_w0 = "
        "km::wall(); call; counter += km::wall() - km_w0; } while (0)\n"
        "uint32_t km_waited_in = 0;\n"
        "uint32_t km_waited_out = 0;\n"
        "const uint32_t km_opened = km::wall();"
        if steps
        else ""
    )
    opening = Group(
        f"stamps: the math CPU's cycles of each tile function call, to {stamps} "
        "page rank",
        [
            Line(
                "const uint32_t km_spans_page "
                f"= get_tile_address({buffer.name}, 0);\n"
                "MATH(volatile tt_l1_ptr uint32_t* km_spans "
                "= km::l1(km_spans_page));\n"
                "MATH(km_spans[0] = command.sequence);\n"
                "MATH(km_spans[1] = 0);\n"
                "for (uint32_t km_sync = 0; "
                f"km_sync < {SPANS_EMPTY}; km_sync++) {{\n"
                "    MATH(const uint32_t km_called = km::wall());\n"
                "    MATH(tensix_sync());\n"
                "    MATH(km_spans[2 + km_sync] = km::wall() - km_called);\n"
                "}" + waits
            )
        ],
    )
    words = "\n".join(
        f"{cpu}(km::l1(km_spans_page)[{cpus + 3 * at} + {k}] = {value});"
        for at, cpu in enumerate(("UNPACK", "MATH", "PACK"))
        for k, value in enumerate(
            ("km::wall() - km_opened", "km_waited_in", "km_waited_out")
        )
    )
    closing = Group(
        f"stamps: the page to mover B, for {stamps} page rank",
        [
            *([Line(words)] if steps else []),
            Line(
                f"if (km_ok && compute.reserve_back({buffer.name}, 1)) {{\n"
                f"    cb_push_back({buffer.name}, 1);\n"
                "}",
                {"km_ok"},
            ),
        ],
    )
    tensor = f"{camel(stamps)}({stamps})"
    write = Group(
        f"stamps: compute's page to {stamps} page rank",
        [
            Line(
                f"if (!movement.wait_front({buffer.name}, 1) ||\n"
                f"    !km::write_pages(movement, command, {tensor}, km_rank, 1,\n"
                f"                     get_read_ptr({buffer.name}))) {{\n"
                "    return false;\n"
                "}\n"
                "noc_async_write_barrier();\n"
                f"cb_pop_front({buffer.name}, 1);",
                {"km_rank", stamps},
                mover="B",
            )
        ],
    )
    frame.compiled.includes.add("ask")
    return opening, closing, write


# A stamped handler of steps: the page's last words, for unpack, math and pack: the
# span, the waits for inputs and the waits for room (TIMED_WAITS sorts each wait).
CPU_WORDS = 9


# A tile function's stamps page: the sequence, the calls, 16 empty spans (a wall-clock
# pair around one tensix_sync), then each call's span from word 18.
SPANS_EMPTY, SPANS_FIRST = 16, 18


def timed_waits(nodes):
    """A stamped handler's compute lines with each wait (TIMED_WAITS) timed into
    its CPU's counter, in place."""
    for node in nodes:
        if isinstance(node, (Line, Glue)):
            node.text = timed_text(node.text)
        elif isinstance(node, Block):
            node.opening = timed_text(node.opening)
            timed_waits(node.children)
        elif isinstance(node, Group):
            timed_waits(node.children)


def timed_text(text):
    """`text` with its waits wrapped in KM_WAITED / KM_WAITED_DO."""
    for start, counter, expression in TIMED_WAITS:
        out, at = [], 0
        while (found := text.find(start, at)) >= 0:
            out.append(text[at:found])
            if not expression:
                out.append(f"KM_WAITED_DO({counter}, {start[:-1]});")
                at = found + len(start)
                continue
            depth, opened = 0, found + start.index("(")  # the call's own parenthesis
            for end in range(opened, len(text)):  # its close
                depth += {"(": 1, ")": -1}.get(text[end], 0)
                if depth == 0:
                    break
            out.append(f"KM_WAITED({counter}, {text[found : end + 1]})")
            at = end + 1
        text = "".join(out) + text[at:]
    return text
