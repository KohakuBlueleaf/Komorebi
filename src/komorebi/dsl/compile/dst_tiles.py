"""DST's tiles in a handler: a running total and its copy, the sums a product leaves,
the scratch tile of an add, zeros, and a total carried back across a loop."""

import ast

from komorebi.dsl.compile import handler_buffers
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.dst_code import DST_FULL, DST_HALF, add_tile, pack
from komorebi.dsl.compile.layout import PAGE_BYTES
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import refuse
from komorebi.dsl.names import kind_of, spelling
from komorebi.dsl.nodes import Block, Line


def accumulate(frame, node, value):
    """`total += tile` inside a stream loop, or `total += block` of a broadcast."""
    source = node.value
    if not isinstance(source, ast.Name) or source.id not in frame.scope:
        refuse(frame.where, node, "add a tile of a stream", "write total += tile")
    item = frame.scope[source.id]
    frame.other_tile_line(node)
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
    """A running total still in DST at the handler's end (no line after its loop
    wrote it): DST released."""
    if frame.dst_live is None or frame.dst_live not in frame.carried:
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
    count = frame.constant_int(call.args[1]) if len(call.args) > 1 else 1
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
