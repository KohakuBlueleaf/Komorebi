"""A handler's reduces: a tile's sum over the command's group by owner or by tree, of
a value or a field, the arrival buffer and its slots, and a merge's reduce."""

import ast
import math
import textwrap

from komorebi.dsl.compile import expressions, handler_buffers, layout, scope
from komorebi.dsl.compile.cxx import camel
from komorebi.dsl.compile.dst import dst_tiles
from komorebi.dsl.compile.dst.dst_code import (
    DST_FULL,
    DST_HALF,
    add_tile,
    pack,
    sums_reserve,
    sums_room,
)
from komorebi.dsl.compile.layout import PAGE_BYTES
from komorebi.dsl.compile.records import Value
from komorebi.dsl.compile.source import (
    WAIT_EMIT,
    WAIT_INPUT,
    WAIT_PORT,
    line_text,
    refuse,
    wait,
)
from komorebi.dsl.compile.tiles import bodies
from komorebi.dsl.compile.tiles.merge_token import tokened
from komorebi.dsl.front.names import spelling
from komorebi.dsl.front.nodes import Line
from komorebi.dsl.islands import island_reduce


def all_reduce(frame, node, name, call, field_, merging):
    """`whole = km.all_reduce(total)`: reduce's sum, cast back by rank 0 to every
    rank of the group in the same command (km_tree.hpp); `km.all_reduce(self.part)`
    of a field compute wrote: every rank's field then holds the group's sum."""
    kind = spelling("all_reduce")
    if merging:
        refuse(
            frame.where,
            node,
            f"km.{kind} adds; a merge of states is km.reduce's "
            f"{spelling('combine')}=",
            f"write km.reduce(self.state, ..., {spelling('combine')}=MERGE)",
        )
    if field_ is not None and name is not None:
        refuse(
            frame.where,
            node,
            f"km.{kind}(self.{field_}) leaves the sum in self.{field_} at every "
            "rank: it gives no value",
            f"write km.{kind}(self.{field_}) on a line of its own",
        )
    if field_ is None and name is None:
        refuse(
            frame.where,
            node,
            f"km.{kind}'s sum is a value: it is assigned",
            f"write whole = km.{kind}(total), or km.{kind}(self.part)",
        )
    tile = reduced_tile(frame, node, call, "all_reduce") if field_ is None else None
    form, form_uses, size, group, uses_group = reduce_keywords(frame, node, call)
    if field_ is not None:
        return field_all_reduce(
            frame, node, field_, (form, form_uses), size, (group, uses_group)
        )
    return tile_reduce(
        frame,
        node,
        name,
        call,
        tile,
        (form, form_uses),
        size,
        (group, uses_group),
        everywhere=True,
    )


def arrival_buffer(frame, number, slots, per, what):
    """kArrival<number>, which the handlers' reduces of that number share: a slot
    of the largest part (`per` pages) and as many slots as the most a reduce takes.
    """
    name = f"kArrival{number}"
    found = next((b for b in frame.compiled.buffers if b.name == name), None)
    if found is not None:
        per, slots = max(per, found.block), max(slots, found.pages[0] // found.block)
        found.block, found.pages = per, (slots * per,)
        return found
    arrival = layout.other_buffer(
        frame.compiled,
        name,
        "arrival",
        "float32",
        PAGE_BYTES["float32"],
        slots * per,
        what,
    )
    arrival.block = per
    return arrival


def field_all_reduce(frame, node, field_, form, size, group):
    """`km.all_reduce(self.part)` of a field compute wrote: compute adds the
    arrivals to it in DST and packs the sum for mover B's send (rank 0: its cast
    back, then freed); rank 0 packs it into its field, each other rank the sum cast
    back, which a later loop's tile function then takes at every rank."""
    form, form_uses = form
    group, uses_group = group
    kind = spelling("all_reduce")
    held = frame.packed.get(field_)
    stored = frame.compiled.fields[field_]
    if held is None or stored.format.dtype != "float32":
        refuse(
            frame.where,
            node,
            f"km.{kind}(self.{field_}) sums a float32 field compute wrote above",
            f"write self.{field_}.write(...) before it, self.{field_} declared "
            "km.L1[km.fp32, ...]",
        )
    tiles = math.prod(stored.shape[:-2])
    if tiles + 1 > DST_HALF:
        refuse(
            frame.where,
            node,
            f"self.{field_}'s {tiles} tiles and the arrivals' tile are more than "
            f"DST's {DST_HALF}",
            f"all-reduce a field of up to {DST_HALF - 1} tiles",
        )
    if form == "km::kReduceOwner":
        if size is None:
            refuse(
                frame.where,
                node,
                f"a reduce of self.{field_} by owner holds a slot of its {tiles} "
                "tiles for each other core of the group, and the group's size is "
                "the command's",
                f'give {spelling("group")}=S, or reduce by tree: form="tree"',
            )
        slots = max(1, size - 1)
    elif form == "km::kReduceTree":
        slots = tree_slots(frame, size)
    else:
        refuse(
            frame.where,
            node,
            f"an all-reduce of self.{field_} sizes its slots by the form",
            'write form="owner" or form="tree"',
        )
    number = frame.arrivals
    frame.arrivals += 1
    arrival = arrival_buffer(
        frame,
        number,
        slots,
        tiles,
        f"line {node.lineno}: {line_text(frame.where, node)}",
    )
    buffer = f"kField{camel(field_)}"
    out = handler_buffers.out_buffer(
        frame,
        "float32",
        PAGE_BYTES["float32"],
        f"line {node.lineno}: {line_text(frame.where, node)}",
        tiles,
    ).name
    frame.compiled.unpack_fp32 |= {arrival.name, buffer}
    frame.compiled.includes |= {"port", "tree", "add", "pack"}
    frame.compiled.port_handlers.add(frame.method.name)
    uses = uses_group | form_uses
    moving = group.format(cpu="movement")
    computing = group.format(cpu="compute")
    movement = [
        Line(
            f"if (!km::all_reduce_arrive(movement, command, {arrival.name}, "
            f"{moving}, {form})) {{\n    return false;\n}}",
            uses,
            mover="A",
        ),
        Line(
            f"if (!km::all_reduce_emit(movement, command, {arrival.name}, "
            f"{moving}, {form}, {out}, {tiles})) {{\n    return false;\n}}\n"
            f"if (km::all_reduce_keeps({moving})) {{  // the cast's, kept by it\n"
            f"    cb_pop_front({out}, {tiles});\n}}",
            uses,
            mover="B",
        ),
    ]
    adds = "\n".join(
        add_tile(
            f"{arrival.name}.cb",
            f"km::slot_page({arrival.name}, km_slot)" + (f" + {t}" if t else ""),
            t,
            tiles,
        )
        for t in range(tiles)
    )
    adds = "\n".join("    " + line for line in adds.split("\n"))
    taken = "".join(f"    copy_tile({buffer}, {t}, {t});\n" for t in range(tiles))
    cast = "".join(
        f"    copy_tile({arrival.name}.cb, {t}, {t});\n" for t in range(tiles)
    )
    packs = "".join(f"    pack_tile({t}, {out});\n" for t in range(tiles))
    # another island's leader casts the sum on from the out buffer as well
    keeps = (
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"if (km_ok && compute.reserve_back({buffer}, {buffer}Pages)) {{\n"
        f"    KM_PACK_FORMAT({buffer});\n"
        + "".join(f"    pack_tile({t}, {buffer});\n" for t in range(tiles))
        + f"    cb_push_back({buffer}, {buffer}Pages);\n"
        "}\n"
        f"if (km_ok && km::all_reduce_keeps({computing}) &&\n"
        f"    compute.reserve_back({out}, {tiles})) {{\n"
        f"    KM_PACK_FORMAT({out});\n{packs}"
        f"    cb_push_back({out}, {tiles});\n"
        "}\n"
        "tile_regs_release();"
    )
    packed_keeps = "\n".join("    " + line for line in keeps.split("\n"))
    root_packs = "".join(f"    pack_tile({t}, {buffer});\n" for t in range(tiles))
    text = (
        f"// self.{field_} summed over the group, the sum at every rank\n"
        f"km_ok = km_ok && compute.wait_front({buffer}, {buffer}Pages);\n"
        "tile_regs_acquire();\n"
        "if (km_ok) {\n"
        f"    KM_UNPACK_FORMAT_A({buffer});\n"
        f"    copy_init({buffer});\n{taken}"
        f"    cb_pop_front({buffer}, {buffer}Pages);\n"
        "}\n"
        f"{held} = false;\n"
        f"km_ok = km_ok && km::reduce(compute, {arrival.name}, {computing}, "
        f"{form}, [&](uint32_t km_slot) {{\n{adds}\n}});\n"
        "tile_regs_commit();\n"
        "tile_regs_wait();\n"
        f"if (km_ok && compute.reserve_back({out}, {tiles})) {{  // for mover B\n"
        f"    KM_PACK_FORMAT({out});\n{packs}"
        f"    cb_push_back({out}, {tiles});\n"
        "}\n"
        f"if (km_ok && km::reduce_send({computing}, {form}).root &&\n"
        f"    compute.reserve_back({buffer}, {buffer}Pages)) {{  // rank 0's\n"
        f"    KM_PACK_FORMAT({buffer});\n{root_packs}"
        f"    cb_push_back({buffer}, {buffer}Pages);\n"
        "}\n"
        "tile_regs_release();\n"
        f"km_ok = km_ok && km::all_reduce_take(compute, {arrival.name}, "
        f"{computing}, [&] {{\n"
        "    tile_regs_acquire();\n"
        f"    KM_UNPACK_FORMAT_A({arrival.name}.cb);\n"
        f"    copy_init({arrival.name}.cb);\n{cast}{packed_keeps}\n"
        "});\n"
        f"{held} = km_ok;"
    )
    frame.compute_waits = True
    wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_PORT)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_EMIT)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    return movement, [Line(text, uses | {"km_ok"})]


def field_reduce(frame, node, name, field_, form, size, group):
    """`km.reduce(self.part, form=form, group=S)` of a field of sums, added at the
    group's rank 0 in L1: mover A takes the arrivals (a slot a sender or a round),
    compute packs each onto the field with L1 accumulation, mover B sends the
    field on. Its size: the field and the slots in L1, not DST."""
    form, form_uses = form
    group, uses_group = group
    sums = frame.summed.get(field_)
    if sums is None:
        refuse(
            frame.where,
            node,
            f"self.{field_} holds no sums above in this handler",
            f"add products into it first: self.{field_} += self.f(a, b)",
        )
    rows, columns = expressions.field_sides(frame, node, field_)
    tiles = rows * columns
    if form == "km::kReduceOwner":
        if size is None:
            refuse(
                frame.where,
                node,
                f"a reduce of self.{field_} by owner holds a slot of its {tiles} "
                "tiles for each other core of the group, and the group's size is "
                "the command's",
                f'give {spelling("group")}=S, or reduce by tree: form="tree"',
            )
        slots = max(1, size - 1)
    elif form == "km::kReduceTree":
        slots = tree_slots(frame, size)
    else:
        refuse(
            frame.where,
            node,
            f"a reduce of self.{field_} sizes its slots by the form",
            'write form="owner" or form="tree"',
        )
    number = frame.arrivals
    frame.arrivals += 1
    arrival = arrival_buffer(
        frame,
        number,
        slots,
        tiles,
        f"line {node.lineno}: {line_text(frame.where, node)}",
    )
    frame.compiled.unpack_fp32.add(arrival.name)
    buffer = f"kField{camel(field_)}"
    frame.compiled.includes |= {"port", "pack"}
    frame.compiled.port_handlers.add(frame.method.name)
    uses = uses_group | form_uses
    moving = group.format(cpu="movement")
    movement = [
        Line(
            f"if (!km::reduce_arrive(movement, command, {arrival.name}, {moving}, "
            f"{form})) {{\n    return false;\n}}",
            uses,
            mover="A",
        ),
        Line(
            f"if (!km::reduce_emit(movement, command, {arrival.name}, {moving}, "
            f"{form}, {buffer}, {buffer}Pages)) {{\n    return false;\n}}",
            uses,
            mover="B",
        ),
    ]
    tile = f"km_tile{node.lineno}"
    room = sums_room(field_)
    combine = (
        f"KM_UNPACK_FORMAT_A({arrival.name}.cb);\n"
        f"copy_init({arrival.name}.cb);\n"
        f"KM_PACK_FORMAT({buffer});\n"
        f"if ({room}) {{\n"
        f"    pack_reconfig_l1_acc({sums} > 0 ? 1 : 0);\n"
        "}\n"
        f"for (uint32_t {tile} = 0; {tile} < {tiles}; {tile}++) {{\n"
        "    tile_regs_acquire();\n"
        f"    copy_tile({arrival.name}.cb, km::slot_page({arrival.name}, "
        f"km_slot) + {tile}, 0);\n"
        "    tile_regs_commit();\n"
        "    tile_regs_wait();\n"
        f"    if ({room}) {{\n"
        f"        pack_tile<true>(0, {buffer}, {tile});\n"
        "    }\n"
        "    tile_regs_release();\n"
        "}\n"
        f"if ({room}) {{\n"
        "    pack_reconfig_l1_acc(0);\n"
        "}\n"
        f"{sums}++;"
    )
    body = "\n".join("    " + line for line in combine.split("\n"))
    compute = [
        Line(
            f"// self.{field_}'s sums and the group's: each arrival added into it\n"
            f"{sums_reserve(sums, room, buffer)}"
            f"km_ok = km_ok && km::reduce(compute, {arrival.name}, "
            f"{group.format(cpu='compute')}, {form}, "
            f"[&](uint32_t km_slot) {{\n{body}\n}});\n"
            f"if (km_ok && {room}) {{  // to mover B's send, or to the leader's "
            "write\n"
            f"    cb_push_back({buffer}, {buffer}Pages);\n"
            "}\n"
            + sent_field_acked(buffer, group.format(cpu="compute"), form)
            + f"{sums} = 0;",
            uses | {"km_ok"},
        ),
    ]
    frame.compute_waits = True
    frame.scope[name] = Value(
        "reduced",
        name,
        buffer=None,
        width=tiles,
        summed=field_,
        made_at=(len(frame.loops), node.lineno),
    )
    frame.reduced.append(name)
    wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_PORT)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_EMIT)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    return movement, compute


def merge_reduce(frame, node, name, field_, form, size, group, combine):
    """`km.reduce(self.state, form="tree", group=S, combine=MERGE)`: an own fp32
    field written in place, merged up the group: the MERGE body on compute's three
    for each arrival slot (`arrival_buffer`, `arrival_index`, the field by name); a
    sender pushes the state for mover B's send, the root's stays in place."""
    combine_ = spelling("combine")
    if combine is None:
        refuse(
            frame.where,
            node,
            "a reduce's sum is a value: it is assigned",
            f"write whole = km.reduce(...), or merge a state in place with "
            f"{combine_}=",
        )
    if name is not None:
        refuse(
            frame.where,
            node,
            f"a reduce with {combine_}= leaves the merged state in place, at the "
            "group's rank 0: it gives no value",
            f"write km.reduce(self.state, ..., {combine_}=MERGE) on its own line",
        )
    own = f"km.{spelling('own')}(push=None)"
    stored = frame.compiled.fields.get(field_) if field_ else None
    if (
        stored is None
        or field_ not in frame.compiled.own
        or frame.compiled.own[field_] is not None
    ):
        refuse(
            frame.where,
            node,
            f"{combine_}= merges a state written in place: an own field of fp32 "
            "tiles that only bodies write",
            f"declare it state: km.L1[km.fp32, n, 32, 32] = {own}",
        )
    if stored.format.dtype != "float32":
        refuse(
            frame.where,
            node,
            f"self.{field_} is {stored.format.dtype}: a merged state is fp32 "
            "tiles, as the arrival's slots are",
            f"declare self.{field_}: km.L1[km.fp32, ...] = {own}",
        )
    tiles = 1
    for side in stored.shape[:-2]:
        tiles *= side
    merge = tokened(  # a token between pack and unpack, or refused here
        frame,
        node,
        field_,
        textwrap.dedent(
            bodies.body_text(
                frame, node, ast.Call(func=None, args=[combine], keywords=[])
            )
        ).strip("\n"),
    )
    form, form_uses = form
    group, uses_group = group
    if form == "km::kReduceOwner":
        if size is None:
            refuse(
                frame.where,
                node,
                f"a merge by owner holds a slot of self.{field_}'s {tiles} tiles "
                "for each other core of the group, and the group's size is the "
                "command's",
                f'give {spelling("group")}=S, or merge by tree: form="tree"',
            )
        slots = max(1, size - 1)
    elif form == "km::kReduceTree":
        slots = tree_slots(frame, size)
    else:
        refuse(
            frame.where,
            node,
            f"a merge of self.{field_} sizes its slots by the form",
            'write form="owner" or form="tree"',
        )
    number = frame.arrivals
    frame.arrivals += 1
    arrival = arrival_buffer(
        frame,
        number,
        slots,
        tiles,
        f"line {node.lineno}: {line_text(frame.where, node)}",
    )
    frame.compiled.unpack_fp32.add(arrival.name)
    frame.compiled.unpack_fp32.add(f"kField{camel(field_)}")
    buffer = f"kField{camel(field_)}"
    frame.compiled.includes |= {"port", "pack"}
    frame.compiled.port_handlers.add(frame.method.name)
    uses = uses_group | form_uses
    moving = group.format(cpu="movement")
    movement = [
        Line(
            f"if (!km::reduce_arrive(movement, command, {arrival.name}, {moving}, "
            f"{form})) {{\n    return false;\n}}",
            uses,
            mover="A",
        ),
        Line(
            f"if (!km::reduce_emit(movement, command, {arrival.name}, {moving}, "
            f"{form}, {buffer}, {buffer}Pages)) {{\n    return false;\n}}",
            uses,
            mover="B",
        ),
    ]
    start = combine.lineno + 1 if isinstance(combine, ast.Constant) else node.lineno
    body = (
        f"const uint32_t arrival{spelling('tile_buffer')} = {arrival.name}.cb;\n"
        f"const uint32_t arrival{spelling('tile_index')} = "
        f"km::slot_page({arrival.name}, km_slot);\n"
        f"const uint32_t {field_} = {buffer};\n"
        f'#line {start} "{frame.compiled.source}"\n{merge}\n#line RESUME'
    )
    inner = "\n".join("    " + line for line in body.split("\n"))
    compute_group = group.format(cpu="compute")
    compute = [
        Line(
            f"// self.{field_} merged with the group's: each arrival by the "
            f"{combine_} body\n"
            f"km_ok = km_ok && km::reduce(compute, {arrival.name}, "
            f"{compute_group}, {form}, [&](uint32_t km_slot) {{\n{inner}\n}});\n"
            f"if (km_ok && !km::reduce_send({compute_group}, {form}).root) {{  "
            "// to mover B's send\n"
            f"    const bool km_room_{field_} = compute.reserve_back({buffer}, "
            f"{buffer}Pages);\n"
            f"    if (km_room_{field_}) {{\n"
            f"        cb_push_back({buffer}, {buffer}Pages);\n"
            "    }\n"
            "}",
            uses | {"km_ok"},
        ),
    ]
    frame.compute_waits = True
    wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_PORT)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_EMIT)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    return movement, compute


def reduce(frame, node, name, call, everywhere=False):
    """`whole = km.reduce(total, form=form)`: total's n tiles summed over the
    group (by owner or by tree) at its rank 0, a slot of n tiles a sender;
    `km.reduce(self.part, ...)`: a field of sums, the same in L1. `everywhere`
    (km.all_reduce): the sum then cast to every rank of the group (km_tree.hpp)."""
    field_ = scope.field_name(frame, call.args[0]) if len(call.args) == 1 else None
    merging = any(keyword_.arg == spelling("combine") for keyword_ in call.keywords)
    if everywhere:
        return all_reduce(frame, node, name, call, field_, merging)
    if name is None and not merging:
        refuse(
            frame.where,
            node,
            "a reduce's sum is a value: it is assigned",
            f"write whole = km.reduce(...), or merge a state in place with "
            f"{spelling('combine')}=",
        )
    if field_ is None and merging:
        refuse(
            frame.where,
            node,
            f"a reduce with {spelling('combine')}= merges a state written in "
            "place: an own field of fp32 tiles",
            f"write km.reduce(self.state, ..., {spelling('combine')}=MERGE)",
        )
    tile = reduced_tile(frame, node, call, "reduce") if field_ is None else None
    form, form_uses, size, group, uses_group = reduce_keywords(frame, node, call)
    combine = next(
        (k.value for k in call.keywords if k.arg == spelling("combine")), None
    )
    if combine is not None or name is None:
        return merge_reduce(
            frame,
            node,
            name,
            field_,
            (form, form_uses),
            size,
            (group, uses_group),
            combine,
        )
    if field_ is not None:
        return field_reduce(
            frame, node, name, field_, (form, form_uses), size, (group, uses_group)
        )
    return tile_reduce(
        frame, node, name, call, tile, (form, form_uses), size, (group, uses_group)
    )


def reduce_keywords(frame, node, call):
    """A reduce's form= and group=: (form, its uses, size or None, the group's
    C++ text, its uses); group= sets the leader to each group's rank 0."""
    form, size = None, None
    for keyword_ in call.keywords:
        if keyword_.arg == spelling("combine"):
            continue
        if keyword_.arg == spelling("group"):
            size = expressions.constant_int(frame, keyword_.value)
            if size < 1:
                refuse(
                    frame.where,
                    node,
                    "a reduce's groups are of 1 rank or more",
                    "give group= a size",
                )
            continue
        if keyword_.arg != spelling("form"):
            refuse(
                frame.where,
                node,
                f"a reduce has no {keyword_.arg}=",
                f"use form=, group= and {spelling('combine')}=",
            )
        value = keyword_.value
        if isinstance(value, ast.Constant) and value.value in ("owner", "tree"):
            form = "km::kReduceOwner" if value.value == "owner" else "km::kReduceTree"
            form_uses = set()
        else:
            form = expressions.integer(frame, value)
            form_uses = expressions.uses(frame, value)
    if form is None:
        form, form_uses = "km::kReduceTree", set()
    group = "km_group"
    uses_group = {"km_group"}
    if size is not None:  # consecutive ranks in groups of `size`, rank 0 owns
        if frame.method.name in frame.compiled.reduce_groups:
            refuse(
                frame.where,
                node,
                "a handler reduces in groups once",
                "reduce the handler's sums in one line",
            )
        frame.compiled.reduce_groups[frame.method.name] = (
            size,
            f"line {node.lineno}, `{line_text(frame.where, node)}`",
        )
        group = f"km::digit_group(command, {{cpu}}.me(), 1, {size})"
        uses_group = set()
        frame.leader_text = f"(km_rank - km_rank % {size})"
    return form, form_uses, size, group, uses_group


def reduced_tile(frame, node, call, kind):
    """The DST tile `km.<kind>(total)` sums, checked."""
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
        refuse(
            frame.where,
            node,
            f"{spelling(kind)} a tile, or a field of sums",
            f"write km.{spelling(kind)}(total, form=form) or "
            f"km.{spelling(kind)}(self.part, form=form)",
        )
    tile = frame.scope.get(call.args[0].id)
    dst_tiles.not_paired(frame, node, call.args[0].id)
    if tile is None or tile.kind != "tile" or frame.dst_live != call.args[0].id:
        refuse(
            frame.where,
            node,
            f"{spelling(kind)} the tile held in DST",
            f"write km.{spelling(kind)}(total, form=form)",
        )
    if not dst_tiles.carries(frame, node, tile):
        dst_tiles.same_loops(
            frame,
            node,
            call.args[0].id,
            tile,
            f"this {spelling(kind)}",
            *dst_tiles.once_text(frame, tile),
        )
    return tile


def tile_reduce(frame, node, name, call, tile, form, size, group, everywhere=False):
    """`whole = km.reduce(total)` of DST tiles (reduce's); `everywhere`: rank 0
    casts the sum back and every other rank packs it into the same buffer."""
    form, form_uses = form
    group, uses_group = group
    width = tile.width
    slots = REDUCE_SLOTS  # by owner, or a form the command chooses
    if width > 1 and form != "km::kReduceTree":
        refuse(
            frame.where,
            node,
            f"a reduce of {width} tiles by owner holds a slot of them for each "
            "other core of the group",
            'reduce them by tree: form="tree"',
        )
    if form == "km::kReduceTree":  # a tree's: one slot a round
        slots = tree_slots(frame, size)
    if tile.dst + width + 1 > DST_FULL:  # the sums and a tile for the arrivals
        refuse(
            frame.where,
            node,
            f"{call.args[0].id}'s {width} tiles and the arrivals' tile are more "
            f"than DST's {DST_FULL}",
            f"reduce up to {DST_FULL - 1 - tile.dst} tiles",
        )
    if tile.dst + width + 1 > DST_HALF:
        frame.compiled.full_dst = True
    number = frame.arrivals
    frame.arrivals += 1
    arrival = arrival_buffer(
        frame,
        number,
        slots,
        width,
        f"line {node.lineno}: {line_text(frame.where, node)}",
    )
    frame.compiled.unpack_fp32.add(arrival.name)
    out = handler_buffers.out_buffer(
        frame,
        "float32",
        PAGE_BYTES["float32"],
        f"line {node.lineno}: {line_text(frame.where, node)}",
        width,
    )
    frame.compiled.includes |= {"port", "add", "pack"}
    frame.compiled.port_handlers.add(frame.method.name)
    uses = uses_group | form_uses
    moving = group.format(cpu="movement")
    call_ = "km::all_reduce" if everywhere else "km::reduce"
    movement = [
        Line(
            f"if (!{call_}_arrive(movement, command, {arrival.name}, {moving}, "
            f"{form})) {{\n    return false;\n}}",
            uses,
            mover="A",
        ),
        Line(
            f"if (!{call_}_emit(movement, command, {arrival.name}, {moving}, "
            f"{form}, {out.name}, {width})) {{\n    return false;\n}}",
            uses,
            mover="B",
        ),
    ]
    if everywhere:
        frame.compiled.includes.add("tree")
    combine = "\n".join(
        add_tile(
            f"{arrival.name}.cb",
            f"km::slot_page({arrival.name}, km_slot)" + (f" + {t}" if t else ""),
            tile.dst + t,
            tile.dst + width,
        )
        for t in range(width)
    )
    body = "\n".join("    " + line for line in combine.split("\n"))
    compute = [
        Line(
            f"km_ok = km_ok && km::reduce(compute, {arrival.name}, "
            f"{group.format(cpu='compute')}, {form}, "
            f"[&](uint32_t km_slot) {{\n{body}\n}});",
            uses | {"km_ok"},
        ),
        Line(pack(tile.dst, out.name, width), {"km_ok"}),
    ]
    if everywhere:  # the sum cast back: slot 0's tiles packed as rank 0's were
        copies = "".join(
            f"    copy_tile({arrival.name}.cb, {t}, {tile.dst + t});\n"
            for t in range(width)
        )
        cast = (
            "tile_regs_acquire();\n"
            f"KM_UNPACK_FORMAT_A({arrival.name}.cb);\n"
            f"copy_init({arrival.name}.cb);\n{copies}" + pack(tile.dst, out.name, width)
        )
        inner = "\n".join("    " + line for line in cast.split("\n"))
        compute.append(
            Line(
                f"km_ok = km_ok && km::all_reduce_take(compute, {arrival.name}, "
                f"{group.format(cpu='compute')}, [&] {{\n{inner}\n}});",
                uses | {"km_ok"},
            )
        )
    frame.compute_waits = True
    if tile.cxx in frame.carried and frame.dst_live == tile.cxx:
        compute = [
            Line(dst_tiles.carried_reduce(frame, node, tile), {"km_ok"}),
            *compute,
        ]
        frame.scope[name] = Value(
            "reduced",
            name,
            buffer=out,
            width=width,
            made_at=(len(frame.loops), node.lineno),
            everywhere=everywhere,
        )
        if not everywhere:  # its sum is the leader's alone
            frame.reduced.append(name)
        wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_PORT)
        wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_EMIT)
        wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
        return movement, compute
    frame.dst_live = None
    tile.kind = "written"
    frame.scope[name] = Value(
        "reduced",
        name,
        buffer=out,
        width=width,
        made_at=(len(frame.loops), node.lineno),
        everywhere=everywhere,
    )
    if not everywhere:
        frame.reduced.append(name)
    wait(frame.where, frame.compiled.waits, node, "mover A", WAIT_PORT)
    wait(frame.where, frame.compiled.waits, node, "mover B", WAIT_EMIT)
    wait(frame.where, frame.compiled.waits, node, "compute", WAIT_INPUT)
    return movement, compute


def tree_slots(frame, size):
    """A tree reduce's arrival slots: a round each over a group of `size` (on
    islands island_reduce's); the command's group (None): TREE_ROUNDS, and over
    islands the root part's rounds and a slot each other part, whichever is more."""
    if size is not None:
        return island_reduce.group_slots(size, frame.compiled.islands)
    parts, most = frame.compiled.islands
    if parts <= 1:
        return TREE_ROUNDS
    return max(TREE_ROUNDS, max(1, (most - 1).bit_length()) + parts - 1)


# A reduce's arrival buffer: one slot for every other core of the largest group a
# reduce by owner takes on a 120-core grid (119 cores), one tile a slot.
REDUCE_SLOTS = 118


# A reduce of n tiles by tree: a slot of n tiles for each round (km::kTreeRounds).
TREE_ROUNDS = 7


def sent_field_acked(buffer, group, form):
    """A field of sums is emptied by compute's unpack on the group's root and by mover
    B's send elsewhere; unpack keeps its own count of the pages taken, so a core that
    sent counts mover B's pop there too, or its next turn as root rewinds the count
    (the hand's K split does the same). The field is one whole item of its buffer."""
    return (
        f"if (km_ok && !km::reduce_send({group}, {form}).root) {{  // mover B pops it\n"
        f"    UNPACK((get_local_cb_interface({buffer}).tiles_acked += "
        f"{buffer}Pages));\n"
        "}\n"
    )
