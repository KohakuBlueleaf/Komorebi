"""A drained stream: pages of an own field that compute's bodies fill and push, which
a mover's loop drains a page a turn up to the end compute states."""

import ast

from komorebi.dsl.compile import expressions, layout, scope
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import WAIT_DATA, check_name, refuse, wait
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Block, Line, has_line


def drain_loop(frame, node, stream):
    """`for page in out:` the mover's loop over a drained stream's pages, a page
    a turn, until the count compute's end handed over."""
    state = frame.drained[stream]
    mover = frame.scope[stream].mover
    if state["ended"] is None:
        refuse(
            frame.where,
            node,
            f"no line ends {stream} above: the mover would wait for pages forever",
            f"end it: {stream}.{spelling('stream_end')}() after the loop that "
            "fills it, or fill it in a with block",
        )
    if not state["taken"]:
        refuse(
            frame.where,
            node,
            f"nothing pushes {stream}'s pages: no compute body takes it",
            f"fill it: km.{spelling('cxx')}(..., takes=[{stream}])",
        )
    drained_loops(frame, node, stream, "its drain")
    if not isinstance(node.target, ast.Name):
        refuse(frame.where, node, "one name a page", f"write `for page in {stream}:`")
    buffer = f"kField{camel(state['field'])}"
    end = end_buffer(frame, node, stream)
    taken, total = f"km_taken_{stream}", f"km_total_{stream}"
    opening = (
        f"if (movement.poll([&] {{\n    if ({total} == 0xFFFFFFFFu && "
        f"cb_pages_available_at_front({end}, 1)) {{\n        "
        f"invalidate_l1_cache();\n        {total} = km::l1(get_read_ptr({end}))[0];"
        f"\n        cb_pop_front({end}, 1);\n    }}\n    return {taken} == {total} "
        f"|| cb_pages_available_at_front({buffer}, 1);\n}}, "
        "movement.args().wait_limit) != km::kGaveNone) {\n    return false;\n}\n"
        f"if ({taken} == {total}) {{\n    break;\n}}"
    )
    counts = Line(
        f"uint32_t {taken} = 0, {total} = 0xFFFFFFFFu;  // the end's count",
        mover=mover,
    )
    loop = page_loop(
        frame, node, node.target.id, state["field"], mover, opening, f"++{taken};", ""
    )
    return [counts, *loop[0]], []


def drained_end(frame, node, name):
    """`out.end()`: compute's line saying no page follows, a hand-over of the
    count of pages pushed to the mover that drains them."""
    state = frame.drained[name]
    if state["within"]:
        refuse(
            frame.where,
            node,
            f"{name} ends where its with block closes",
            f"drop {name}.{spelling('stream_end')}(), or the with",
        )
    if state["ended"] is not None:
        refuse(
            frame.where,
            node,
            f"{name} ended at line {state['ended']} already",
            "end it once",
        )
    drained_loops(frame, node, name, "its end")
    state["ended"] = node.lineno
    end = end_buffer(frame, node, name)
    text = (
        f"if (compute.reserve_back({end}, 1)) {{\n"
        f"    PACK((reinterpret_cast<volatile uint32_t*>(get_local_cb_interface("
        f"{end}).fifo_wr_ptr << 4)[0] = km_pushed_{name}));\n"
        f"    cb_push_back({end}, 1);\n}}"
    )
    return [], [Line(text)]


def drained_loops(frame, node, name, what):
    """A drained stream's make, end and drain stand in the same loops: compute
    pushes one end (a count) where the end line runs, the mover reads one where
    the drain runs, and a stream made in a loop is a new stream each turn."""
    loops, line = frame.drained[name]["loops"]
    if loops == len(frame.loops):
        return
    once = ""
    if loops > len(frame.loops):  # made each turn, drained after: all turns' pages
        once = (
            f"; for every turn's pages drained once, make {name} before the loop "
            f"({name} = km.{spelling('stream')}(...)), its body lines in the loop, "
            f"and {name}.end() and the drain after it"
        )
    refuse(
        frame.where,
        node,
        f"{name} was made at line {line} in {_loops(loops)} and {what} stands in "
        f"{_loops(len(frame.loops))}: the end's count is pushed where the end line "
        "runs and read where the drain runs, so a turn's pages or a count are "
        "lost or waited for for good",
        f"make, end and drain {name} in the same loop (a with block per turn, and "
        f"`for page in {name}:` after it in that loop), or all three outside the "
        f"loops{once}",
    )


def drained_stream(frame, node, name, call):
    """`out = km.Stream(self.f, mover="B")`: pages of own field f that compute's
    bodies fill and the line pushes, drained by the mover named (drain_loop), up
    to the end compute states (drained_end)."""
    field_ = scope.field_name(frame, call.args[0])
    keywords = {k.arg: k.value for k in call.keywords}
    named = keywords.get("mover")
    mover = named.value if isinstance(named, ast.Constant) else None
    if mover not in ("A", "B") or set(keywords) != {"mover"} or len(call.args) != 1:
        refuse(
            frame.where,
            node,
            "a stream of an own field's pages is compute's to fill and one "
            "mover's to drain: mover= names it",
            f'write `{name} = km.{spelling("stream")}(self.f, mover="B")`',
        )
    if field_ not in frame.compiled.own or frame.compiled.own[field_] in (
        "mover_a",
        "mover_b",
    ):
        refuse(
            frame.where,
            node,
            f"self.{field_} is no own field compute fills (a mover pushes it, or "
            "it is filled by a line)",
            f'declare it km.{spelling("own")}(push="pack")',
        )
    check_name(frame.where, node, name)
    frame.scope[name] = Value("drained", name, mover=mover)
    frame.drained[name] = {
        "field": field_,
        "ended": None,
        "taken": False,
        "loops": (len(frame.loops), node.lineno),
    }
    frame.drained[name]["within"] = False
    pushed = Line(f"uint32_t km_pushed_{name} = 0;  // the pages pushed")
    return [], [pushed]


def drained_with(frame, node):
    """`with km.Stream(self.f, mover="B") as out:`: the stream, its block's
    lines, and its end where the block closes."""
    item = node.items[0] if len(node.items) == 1 else None
    if (
        item is None
        or not isinstance(item.optional_vars, ast.Name)
        or expressions.construct(frame, item.context_expr) != "stream"
        or not scope.field_name(
            frame, item.context_expr.args[0] if item.context_expr.args else None
        )
    ):
        refuse(
            frame.where,
            node,
            "a with block is a drained stream's, ended where it closes",
            f'write `with km.{spelling("stream")}(self.f, mover="B") as out:`',
        )
    name = item.optional_vars.id
    movement, compute = drained_stream(frame, node, name, item.context_expr)
    frame.drained[name]["within"] = True
    body_m, body_c = frame.statements(node.body)
    frame.drained[name]["within"] = False
    end_m, end_c = drained_end(frame, node, name)
    frame.drained[name]["ended"] = node.end_lineno
    return movement + body_m + end_m, compute + body_c + end_c


def end_buffer(frame, node, name):
    """The page of a drained stream's end: the count of pages compute pushed."""
    return layout.other_buffer(
        frame.compiled,
        f"kEnd{camel(name)}",
        "word",
        "uint32",
        16,
        1,
        f"line {node.lineno}: the end of {name} (pack's, to mover "
        f"{frame.scope[name].mover})",
    ).name


def page_loop(frame, node, name, field_, mover, opening, closing, _):
    """A mover's loop over pages of own field f, a page a turn: `opening` waits
    for one (or breaks), the body's lines (the mover's alone), the page freed
    once its writes read it, then `closing`; `name` the page's L1 address."""
    check_name(frame.where, node, name)
    frame.scope[name] = Value("page", name, mover=mover)
    body_m, body_c = frame.statements(node.body)
    frame.scope.pop(name)

    def movers(nodes, inherited=None):  # the mover of each line of the nodes
        for item in nodes:
            held = getattr(item, "mover", None) or inherited
            if isinstance(item, Line):
                yield held
            else:
                yield from movers(getattr(item, "children", []), held)

    if any(has_line(n) for n in body_c) or set(movers(body_m)) - {mover}:
        refuse(
            frame.where,
            node.body[0],
            f"inside a stream mover {mover} drains, mover {mover}'s lines alone",
            "move the other CPU's lines before or after the loop",
        )
    wait(frame.where, frame.compiled.waits, node, f"mover {mover}", WAIT_DATA)
    buffer = f"kField{camel(field_)}"
    loop = Block(
        "for (;;) {",
        "}",
        [
            Line(
                f"{opening}\ninvalidate_l1_cache();\nconst uint32_t {name} = "
                f"get_read_ptr({buffer});\nvolatile tt_l1_ptr uint32_t* "
                f"{name}_words = km::l1({name});",
                mover=mover,
            ),
            *body_m,
            Line(
                "noc_async_writes_flushed();  // the page read by its writes\n"
                f"cb_pop_front({buffer}, 1);\n{closing}",
                mover=mover,
            ),
        ],
        mover=mover,
    )
    return [loop], []


def _loops(count):
    """A loop depth in words, for a refusal."""
    return "no loop" if count == 0 else "1 loop" if count == 1 else f"{count} loops"
