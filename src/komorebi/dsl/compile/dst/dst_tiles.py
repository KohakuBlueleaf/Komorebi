"""DST's tiles in a handler: a running total and its copy, the sums a product leaves,
the scratch tile of an add, zeros, a field's tile, a total carried back across a loop
and a scan's state carried between its loop's tiles."""

import ast
import math

from komorebi.dsl.compile import expressions, handler_buffers, scope, tile_lookup
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.dst.dst_code import DST_FULL, DST_HALF, add_tile, pack
from komorebi.dsl.compile.layout import PAGE_BYTES
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import WAIT_INPUT, refuse, wait
from komorebi.dsl.front.names import kind_of, spelling
from komorebi.dsl.front.nodes import Block, Line


def accumulate(frame, node, value):
    """`total += tile` inside a stream loop, or `total += block` of a broadcast."""
    source = node.value
    if not isinstance(source, ast.Name) or source.id not in frame.scope:
        refuse(frame.where, node, "add a tile of a stream", "write total += tile")
    item = frame.scope[source.id]
    scope.whole_tile(frame, node, source.id, item, "an add")
    tile_lookup.other_tile_line(frame, node)
    if item.kind == "streamed":
        frame.compiled.includes |= {"add"}
        text = add_tile(item.cxx, item.first, value.dst, scratch(frame))
        return [], [Line(text, {item.cxx, item.first})]
    if item.kind == "block":
        frame.compiled.includes |= {"add"}
        loop = (
            f"for (uint32_t {item.first} = 0; {item.first} < {item.count}; "
            f"{item.first}++) {{"
        )
        body = add_tile(item.cxx, item.first, value.dst, scratch(frame))
        return [], [Block(loop, "}", [Line(body)], {item.cxx})]
    refuse(frame.where, node, f"{source.id} is a {item.kind}", "add a tile of a stream")


def carried_back(frame, loop):
    """A running total written or reduced in `loop`'s body: read back from its
    carry into its DST tiles as the body's last line, for the next turn."""
    back = loop.pop("back", None)
    if back is None:
        return []
    value, loaded = back
    if frame.dst_live is not None:
        refuse(
            frame.where,
            loop["body"][-1],
            f"{frame.dst_live} holds DST at the end of the turn, where the running "
            f"total {value.cxx} comes back to it",
            f"write or reduce {frame.dst_live} before the turn ends",
        )
    frame.dst_live = value.cxx
    return [Line(f"// {value.cxx}, the running total, back in DST\n{loaded}")]


def carried_end(frame):
    """A running total (or a scan's state) still in DST at the handler's end (no
    line after its loop wrote it): DST released."""
    live = frame.scope.get(frame.dst_live) if frame.dst_live is not None else None
    scanned = live is not None and live.scan is not None
    if frame.dst_live is None or not (frame.dst_live in frame.carried or scanned):
        return []
    return [
        Line(
            f"// {frame.dst_live}, the running total, left: DST released\n"
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            "tile_regs_release();"
        )
    ]


def carried_reduce(frame, node, value):
    """A running total's reduce: (before it, the tiles carried and read back with
    the carry kept; after the reduce's pack, the own total read back again and the
    carry freed), so the reduce's sums never enter the total."""
    buffer = carry_buffer(frame, node, value)
    saved, kept = carry_lines(frame, buffer, value, pop=False)
    _, loaded = carry_lines(frame, buffer, value)
    before = (
        f"// {value.cxx} is a running total: carried before the reduce\n"
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"{saved}"
        "tile_regs_release();\n"
        f"{kept}"
    )
    frame.loops[-1]["back"] = (value, loaded)
    frame.dst_live = None
    return before


def carried_write(frame, node, value, out):
    """A running total's page write: its tiles packed into `out` and the carry,
    DST released, then read back from the carry into the same tiles."""
    buffer = carry_buffer(frame, node, value)
    saved, loaded = carry_lines(frame, buffer, value)
    frame.loops[-1]["back"] = (value, loaded)
    frame.dst_live = None
    packs = "".join(
        f"    pack_tile({value.dst + t}, {out.name});\n" for t in range(value.width)
    )
    return (
        f"// {value.cxx} is a running total: written and carried\n"
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"if (km_ok && compute.reserve_back({out.name}, {value.width})) {{\n"
        f"    KM_PACK_FORMAT({out.name});\n"
        f"{packs}"
        f"    cb_push_back({out.name}, {value.width});\n"
        "}\n"
        f"{saved}"
        "tile_regs_release();"
    )


def carries(frame, node, value):
    """True when `value` is a running total: a tile of DST made by zeros outside
    the range loop whose body writes or reduces it at `node` (a page a turn, the
    sum so far); its tiles go through a carry buffer there and are back in DST at
    the end of that loop's body."""
    if (
        value is None
        or value.kind != "tile"
        or value.made_at is None
        or value.batch is not None
        or getattr(value, "state", None) is not None
        or getattr(value, "row_of", None) is not None
        or frame.dst_live != value.cxx
    ):
        return False
    if value.made_at[0] >= len(frame.loops):
        return False
    if node not in frame.loops[-1].get("body", ()) or frame.loops[-1].get("back"):
        refuse(
            frame.where,
            node,
            f"{value.cxx} is a running total (made at line {value.made_at[1]}, "
            "outside this loop): it is written or reduced once a turn, as a line "
            "of the range loop's own body",
            "move the line out of the if or inner loop into the range loop's "
            "body, or make the tile in the loop",
        )
    frame.carried.add(value.cxx)
    return True


def carry_buffer(frame, node, value):
    """The fp32 buffer a running total's tiles pass through (compute's alone:
    pack writes it, unpack reads it back to DST unchanged)."""
    buffer = handler_buffers.out_buffer(
        frame,
        "float32",
        PAGE_BYTES["float32"],
        f"line {node.lineno}: the running total {value.cxx}, carried",
        value.width,
        f"kCarry{camel(value.cxx)}",
    )
    buffer.pages = (value.width,)
    frame.compiled.unpack_fp32.add(buffer.name)
    return buffer


def carry_lines(frame, buffer, value, pop=True):
    """(the pack of the total's DST tiles into `buffer`, the read of them back
    into the same DST tiles after a new acquire, freed when `pop`)."""
    name, dst, width = buffer.name, value.dst, value.width
    packs = "".join(f"    pack_tile({dst + t}, {name});\n" for t in range(width))
    saved = (
        f"if (km_ok && compute.reserve_back({name}, {width})) {{  // the total\n"
        f"    KM_PACK_FORMAT({name});\n"
        f"{packs}"
        f"    cb_push_back({name}, {width});\n"
        "}\n"
    )
    copies = "".join(f"    copy_tile({name}, {t}, {dst + t});\n" for t in range(width))
    loaded = (
        f"km_ok = km_ok && compute.wait_front({name}, {width});\n"
        "tile_regs_acquire();\n"
        "if (km_ok) {  // the total back in DST\n"
        f"    KM_UNPACK_FORMAT_A({name});\n"
        f"    copy_init({name});\n"
        f"{copies}" + (f"    cb_pop_front({name}, {width});\n" if pop else "") + "}"
    )
    return saved, loaded


def from_field(frame, node, name, field_):
    """`h = self.s`, s a float32 field of one tile filled above in this block: its
    tile copied into a DST tile, as km.tile.zeros makes one (a scan's state started
    from the field), and the field's page freed."""
    stored = frame.compiled.fields[field_]
    pages = math.prod(stored.shape[:-2])
    if pages != 1 or stored.format.dtype != "float32":
        refuse(
            frame.where,
            node,
            f"{name} = self.{field_} starts a DST tile from a float32 field of one "
            f"tile, and self.{field_} is {pages} tile(s) of {stored.format.dtype}",
            f"declare self.{field_} km.L1[km.fp32, 1, 32, 32] (a scan's state is "
            "carried in float32)",
        )
    filled = frame.filled.get(field_)
    if filled is None or filled[1] != frame.blocks[-1]:
        refuse(
            frame.where,
            node,
            f"self.{field_} is not filled above, in this block",
            f"write self.{field_}.fill(km.{spelling('stream')}(...)) before the line",
        )
    if frame.dst_live is not None:
        refuse(
            frame.where,
            node,
            f"{frame.dst_live} still holds DST; level 1 keeps one tile there",
            "write it, or reduce it, first",
        )
    frame.filled[field_] = None
    buffer = f"kField{camel(field_)}"
    frame.compiled.unpack_fp32.add(buffer)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    dst = frame.dst_next
    frame.dst_next += 1
    frame.dst_live = name
    frame.scope[name] = Value(
        "tile", name, dst=dst, width=1, made_at=(len(frame.loops), node.lineno)
    )
    text = (
        f"km_ok = km_ok && compute.wait_front({buffer}, 1);\n"
        "tile_regs_acquire();\n"
        f"if (km_ok) {{  // self.{field_}'s tile into DST\n"
        f"    KM_UNPACK_FORMAT_A({buffer});\n"
        f"    copy_init({buffer});\n"
        f"    copy_tile({buffer}, 0, {dst});\n"
        f"    cb_pop_front({buffer}, 1);\n"
        "}"
    )
    return [], [Line(text, {"km_ok"})]


def held_sum_copy(frame, value, buffer, width):
    """A turn's write of a held sum: its tiles copied (fp32, exact) from the sum's
    buffer into DST and packed into the write's own `buffer`, the sum kept for
    the next turn."""
    source = value.buffer
    frame.compiled.unpack_fp32.add(source.name)
    copies = "\n".join(f"copy_tile({source.name}, {t}, {t});" for t in range(width))
    again = f"{copies}\n" + pack(0, buffer.name, width)
    body = "\n".join("    " + line for line in again.split("\n"))
    sum_freed_after(frame, value, f"cb_pop_front({source.name}, {width});")
    return Line(
        f"// {value.cxx}, the sum held across the loop, copied for this turn\n"
        f"km_ok = km_ok && compute.wait_front({source.name}, {width});\n"
        "if (km_ok) {\n"
        f"    KM_UNPACK_FORMAT_A({source.name});\n"
        f"    copy_init({source.name});\n"
        "    tile_regs_acquire();\n"
        f"{body}\n}}",
        {"km_ok"},
    )


def not_paired(frame, node, name):
    """Refuses a write or a reduce of one of two DST sums held together: its
    pack releases DST, and the other's sums with it."""
    if frame.dst_pair is not None and name in frame.dst_pair:
        a, b = frame.dst_pair
        refuse(
            frame.where,
            node,
            f"{a} and {b} hold DST together, and writing {name} alone releases "
            "the other's sums",
            f"write a tile function of both: pages.write(f({a}, {b}))",
        )


def once_text(frame, value):
    """The reason and the right forms a refusal of a tile or a reduce's sum used
    in other loops than the line that made it gives."""
    if value.kind == "tile":
        return (
            "a tile holds DST from its zeros to its one write or reduce, which "
            "frees it, so the next turn of a loop would add into a freed tile "
            "nothing zeroed",
            "make it in the loop that writes it (`total = km.tile.zeros("
            "km.fp32)` as its first line there), or add into it in the loop and "
            "write it after the loop",
        )
    return (
        "a reduce makes one sum each time it runs, and a write or a body frees "
        "it, so a turn that follows would wait for a sum that is gone",
        "write it where it is reduced, in the same loop (under the leader's "
        "`if`), or reduce and write both outside the loop",
    )


def same_loops(frame, node, name, value, using, why, instead):
    """Refuses `using` a once-only object (`value.made_at`: a tile of DST, a
    reduce's sum, a cast block) in other loops than the line that made it: a loop
    around the one line and not the other runs it a turn more, and the movers'
    and compute's counts differ."""
    if value.made_at is None:
        return
    loops, line = value.made_at
    if loops == len(frame.loops):
        return
    where = (
        f"outside the loop {using} stands in"
        if loops < len(frame.loops)
        else f"inside a loop that has ended where {using} stands"
    )
    refuse(
        frame.where, node, f"{name} was made at line {line}, {where}: {why}", instead
    )


def scratch(frame):
    """The DST tile a copy lands in before an add: the one after the sums."""
    return frame.dst_next


def second_sums(frame, node, name, count):
    """Whether `name = km.tile.zeros(km.fp32, count)` is a second set of DST sums
    held beside the live one (two sums a tile function of both writes: `pages.
    write(f(a, b))`); refused by name when they would not fit DST together."""
    first = frame.scope.get(frame.dst_live)
    if (
        frame.dst_pair is not None
        or first is None
        or first.kind != "tile"
        or first.made_at is None
        or first.made_at[0] != len(frame.loops)
        or first.batch is not None
        or getattr(first, "state", None) is not None
        or getattr(first, "row_of", None) is not None
        or first.cxx in frame.carried
    ):
        return False
    if first.dst + first.width + count > DST_FULL:
        refuse(
            frame.where,
            node,
            f"{first.cxx} ({first.width} tiles) and {name} ({count} tiles) take "
            f"{first.dst + first.width + count} of DST's {DST_FULL} fp32 tiles",
            f"make each {DST_FULL // 2} or fewer",
        )
    return True


def sum_freed_after(frame, value, freed):
    """A held sum's free: after the outermost loop the sum was made outside of,
    under the leader's condition (only the leader holds the sum)."""
    condition, uses = frame.leader_condition
    loop = frame.loops[value.made_at[0]]
    loop.setdefault("after", []).append(
        Line(
            f"if ({condition}) {{  // {value.cxx}, held across the loop, freed\n"
            f"    {freed}\n}}",
            set(uses),
        )
    )


def sum_held(frame, value):
    """True when `value` is a reduce's sum made outside the range loop a line
    writes it or a body takes it in, under the leader's if: the sum is held
    (waited for each turn, never freed in it) and freed after that loop."""
    return (
        value is not None
        and value.kind == "reduced"
        and not value.popped
        and value.made_at is not None
        and value.made_at[0] < len(frame.loops)
        and frame.in_leader
        and frame.leader_condition is not None
    )


def zeros(frame, node, name, call):
    fmt = call.args[0] if call.args else None
    if not (isinstance(fmt, ast.Attribute) and kind_of(fmt.attr, ("fp32",)) == "fp32"):
        refuse(
            frame.where,
            node,
            "a tile accumulates in DST in fp32",
            f"write km.{spelling('tile')}.{spelling('zeros')}(km.fp32)",
        )
    count = expressions.constant_int(frame, call.args[1]) if len(call.args) > 1 else 1
    second = frame.dst_live is not None and second_sums(frame, node, name, count)
    if frame.dst_live is not None and not second:
        refuse(
            frame.where,
            node,
            f"{frame.dst_live} still holds DST; level 1 keeps one tile there",
            "write it, or reduce it, first",
        )
    if not 1 <= count <= DST_FULL - frame.dst_next:
        refuse(
            frame.where,
            node,
            f"{count} tiles of sums: DST holds {DST_FULL} fp32 tiles",
            f"write km.{spelling('tile')}.{spelling('zeros')}(km.fp32, n), n up "
            f"to {DST_FULL - frame.dst_next}",
        )
    if frame.dst_next + count > DST_HALF:  # past a half: DST in full-sync mode
        frame.compiled.full_dst = True
    dst = frame.dst_next
    frame.dst_next += count
    if second:
        frame.dst_pair = (frame.dst_live, name)
    frame.dst_live = name
    frame.scope[name] = Value(
        "tile", name, dst=dst, width=count, made_at=(len(frame.loops), node.lineno)
    )
    frame.compiled.includes |= {"fill"}
    fills = "\n".join(f"fill_tile({dst + t}, 0.0f);" for t in range(count))
    if second:  # DST is held already: the second sums' tiles zeroed after the first
        return [], [Line(f"fill_tile_init();\n{fills}")]
    text = f"tile_regs_acquire();\nfill_tile_init();\n{fills}"
    return [], [Line(text)]


def scan_state(frame, node, call, name, state):
    """`h = f(x, h)` in a stream loop, h a DST tile made before the loop (a scan):
    h packed into an fp32 carry before the loop's blocks, and returned as f's
    operand there (each tile reads it back from the carry, scan_operand); None
    when the line is no scan."""
    if not isinstance(name, str):
        return None
    read = [a for a in call.args if isinstance(a, ast.Name) and a.id == name]
    if not read:
        return None
    value = frame.scope.get(name)
    if value is None:
        refuse(
            frame.where,
            node,
            f"{name} is read before it is set: a scan's state is a DST tile made "
            "before its loop",
            f"write {name} = km.{spelling('tile')}.{spelling('zeros')}(km.fp32) "
            f"(or {name} = self.<field>) before the loop",
        )
    if (
        value.kind != "tile"
        or frame.dst_live != name
        or value.made_at is None
        or value.made_at[1] >= state["line"]
    ):
        return None
    if value.width != 1 or len(read) > 1:
        refuse(
            frame.where,
            node,
            f"{name} is a scan's state: one DST tile, passed once"
            + (f", and it is {value.width} tiles" if value.width != 1 else ""),
            f"make {name} = km.{spelling('tile')}.{spelling('zeros')}(km.fp32) and "
            f"pass it once",
        )
    buffer = carry_buffer(frame, node, value)
    saved, _ = carry_lines(frame, buffer, value)
    state["inits"].append(
        Line(
            f"// {name}, a scan's state, into its carry\n"
            "tile_regs_commit();\n"
            "tile_regs_wait();\n"
            f"{saved}"
            "tile_regs_release();",
            {"km_ok"},
        )
    )
    frame.dst_live, frame.dst_next = None, 0
    state["scan"] = (name, value.made_at)
    return Value("carried", buffer.name, buffer=buffer)


def scan_operand(state, carried, dst, streams):
    """A scan's state read back from its carry into DST tile `dst` as its tile's
    operand, and the streams' format (`streams`: a stream's buffer) set again."""
    name = carried.cxx
    return [
        f"km_ok = km_ok && compute.wait_front({name}, 1);",
        f"if (km_ok) {{  // {state[0]}, the scan's state",
        f"    KM_UNPACK_FORMAT_A({name});",
        f"    copy_init({name});",
        f"    copy_tile({name}, 0, {dst});",
        f"    cb_pop_front({name}, 1);",
        "}",
        f"KM_UNPACK_FORMAT_A({streams}.mover_a);",
        f"copy_init({streams}.mover_a);",
    ]


def scan_carried(frame, value):
    """A scan's step's end: its state (DST 0) packed into the carry for the next
    tile, before the release; '' when `value` is no scan's state."""
    if value.scan is None:
        return ""
    saved, _ = carry_lines(frame, value.scan, value)
    return saved


def scan_end(frame, state):
    """A stream loop's body end: a scan's state packed into its carry when no push
    did, and after the loop read back from the carry into DST 0, the tile its name
    holds there (released at the handler's end if nothing writes it)."""
    if "scan" not in state:
        return []
    name, made_at = state["scan"]
    value = frame.scope[name]
    lines = []
    if frame.dst_live == name:  # not pushed: carried at the tile's end
        lines.append(
            Line(
                "tile_regs_commit();\n"
                "tile_regs_wait();\n"
                f"{scan_carried(frame, value)}"
                "tile_regs_release();",
                {"km_ok"},
            )
        )
    back = Value("tile", name, dst=0, width=1, made_at=made_at, scan=value.scan)
    _, loaded = carry_lines(frame, value.scan, back)
    state["after"].append(
        Line(f"// {name}, the scan's state, back in DST\n{loaded}", {"km_ok"})
    )
    frame.dst_live, frame.dst_next = name, 1
    frame.scope[name] = back
    return lines
