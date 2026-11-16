"""The row form: a row of `width` whole tiles in Dst, one exact sum along the row
(`reduce_sum(f(row), exact=True)`), a per-row value computed from it, and the stored
row computed from the row, that value broadcast with `.col()` and a per-column value
broadcast with `.row()`. Layout rule: a per-row value is lanes of one tile."""

from dataclasses import dataclass

from komorebi.dsl.tile import chain as chains
from komorebi.dsl.tile.ir import TILE_DTYPES, PlanError, fail
from komorebi.dsl.tile.lower import (
    NOT_CHECKED,
    STALL_SFPU,
    TILE,
    WAIT_MATH,
    Body,
    _checks,
    _value,
    allocate,
    fold_negation,
    fuse_mad,
    hoist,
)
from komorebi.dsl.tile.unit import (
    OPS,
    PROGRAMMABLE,
    WORK,
    Call,
    Check,
    Insn,
    Loop,
    Pass,
    Runtime,
    Sync,
    UnitError,
    UnitProgram,
)

GROUPS = [(h, q) for h in range(2) for q in range(4)]  # row groups: 4 rows each
ROTATE = 3  # SFPSHFT2 mode: rotate lanes right by one inside each group of 8


@dataclass(frozen=True)
class Layout:
    """Where a value's vectors live in Dst. "tile": vector (t, h, c, q, p) of tile t,
    rows 16 h + 4 q .. + 3, columns 16 c + p + 2 k; "rows": per-row slot (h, q), lane l
    holding row 16 h + 4 q + l / 8 in all 8 lanes of its lane row; "cols": per-column
    slot (t, c, p), lane l holding column 32 t + 16 c + 2 (l % 8) + p in every row."""

    kind: str
    tile: int
    offset: int = 0  # "rows": the slot's offset in its tile (steps.SLOT)

    def address(self, t, h, c, q, p):
        """The Dst address of the vector a tile vector (t, h, c, q, p) meets."""
        base = TILE * self.tile
        if self.kind == "tile":
            return base + TILE * t + 32 * h + 16 * c + 4 * q + 2 * p
        if self.kind == "rows":
            return base + 4 * (4 * h + q) + self.offset
        return base + 16 * c + 8 * t + 2 * p


TILE_VALUES = 1024  # a 32 x 32 tile in face order


def row_operand(row, width):
    """The per-column operand tile a `.row()` of a resident row reads (Dst tile
    `cols_tile` of `row_program`), from a plain row of 32 `width` values: 1024 values in
    face order, face c rows 8 t .. 8 t + 3 holding columns 32 t + 16 c .. + 15, the rest
    0. The host writes it once, as the stream's resident page."""
    if not 1 <= width <= 2:
        fail("the row form takes rows of 1 or 2 tiles (a per-column tile holds 2)")
    row = list(row)
    if len(row) != 32 * width:
        fail(f"a row of {width} tile(s) has {32 * width} values, not {len(row)}")
    out = [0.0] * TILE_VALUES
    for t in range(width):
        for c in range(2):
            for r in range(8 * t, 8 * t + 4):
                at = 256 * c + 16 * r
                out[at : at + 16] = row[32 * t + 16 * c : 32 * t + 16 * c + 16]
    return out


def row_program(ir, width, rows_tile, cols_tile, dest="fp32", rows=False):
    """The unit program of row kernel `ir` over a row of `width` tiles in Dst tiles
    0 .. width - 1 (the result stored over them), its per-row value in Dst tile
    `rows_tile` and its per-column operand read from Dst tile `cols_tile`; `rows`:
    the band, the body's live rows a runtime argument (`band_groups`)."""
    if dest != "fp32":
        raise PlanError("komorebi.dsl.tile: the unit program is built for fp32 Dst")
    if not 1 <= width <= 2:
        fail("the row form takes rows of 1 or 2 tiles (a per-column tile holds 2)")
    if not {t.dtype for t in ir.tensors} <= set(TILE_DTYPES):
        fail(
            f"{ir.name}: tensors are fp32 or of the B family "
            f"({', '.join(TILE_DTYPES[1:])})"
        )
    if len(ir.stores) != 1:
        fail(f"{ir.name}: the row form takes one store")
    store = ir.stores[0]
    shapes = chains._Shapes(True)
    nodes = ir.nodes()
    for n in nodes:
        shapes(n)
    reduces = [n for n in nodes if n.kind == "reduce"]
    if not reduces:
        fail(f"{ir.name}: the row form takes a reduction")
    for red in reduces:
        if red.attrs["op"] != "sum" or not red.attrs["exact"] or red.attrs["dim"] != -1:
            fail(
                "the row form's reduction is reduce_sum(..., exact=True) along the row"
            )
    cols = [n for n in nodes if n.kind == "col"]
    if any(shapes(c.args[0]) != "col" for c in cols):
        fail("a per-row value meets the row through .col()")
    rows_ = [n for n in nodes if n.kind == "load" and n.attrs["view"].kind == "row"]
    if len({id(n.attrs["view"].tensor) for n in rows_}) != 1:
        fail("the row form reads one row tensor (loaded once)")
    residents = {id(n): n for n in nodes if n.kind == "resident"}
    if len(reduces) > 1 or len(cols) != 1 or len(residents) > 1:
        return level_program(ir, width, rows_tile, cols_tile, shapes, rows_[0], rows)
    red, stat = reduces[0], cols[0].args[0]

    def leaf_of(node, offset, kind):
        if offset is not None:
            return None
        if node.kind in ("load", "resident") and node.attrs["view"].kind == "row":
            return chains.Leaf(node, kind, None)
        if node is red or node is stat:
            return chains.Leaf(node, kind, None)
        return None

    compiler = chains._Compiler(ir.name, shapes, {})
    reduced = compiler.chain(red.args[0], leaf_of)
    per_row = compiler.chain(stat, leaf_of)
    stored = compiler.chain(store.value, leaf_of)
    for c in (reduced, per_row, stored):
        if c.values or c.prev is not None:
            fail(f"{ir.name}: the row form takes no runtime scalars and one stage")
    layouts = {
        "copy": Layout("tile", 0),
        "bcast_col": Layout("rows", rows_tile),
        "bcast_row": Layout("cols", cols_tile),
    }
    for c in (reduced, stored):
        for leaf in c.operands:
            if leaf.kind == "copy" and leaf.source.kind != "load":
                fail("the row's own operand is the loaded row")
            if leaf.kind == "bcast_row" and leaf.source.kind != "resident":
                fail(".row() broadcasts a resident row (a weight)")
    body = Body()
    unit = UnitProgram(ir.name, dest)
    unit.operands = [rows_[0].attrs["view"].tensor.name]
    pairs = _stat_phase(body, reduced, per_row, layouts["bcast_col"], width)
    pairs += _apply_phase(body, stored, layouts, width)
    unit.passes.append(Pass("algebra", "fold_negation", fold_negation(body.code)))
    unit.passes.append(Pass("algebra", "fuse_mad", fuse_mad(body.code)))
    unit.passes.append(
        Pass(
            "layout",
            "lanes",
            [
                f"the per-row value: 8 vectors of lanes in tile {rows_tile}; the "
                f"per-column value: {4 * width} vectors in tile {cols_tile}"
            ],
        )
    )
    _check_lanes(unit, body.code, layouts["bcast_col"])
    moved = 0
    for pair in pairs:
        at = [i for i, x in enumerate(body.code) if id(x) in pair]
        moved += interleave(body.code, at[0], at[-1] + 1)
    unit.passes.append(
        Pass("schedule", "interleave", [f"{moved} instructions moved in {len(pairs)}"])
    )
    _check_broadcast(unit, body.code, layouts, width)
    setup = []
    unit.passes.append(Pass("schedule", "hoist", hoist(body.code, setup)))
    unit.passes.append(Pass("layout", "allocate", allocate(body.code, body.prog)))
    allocate(setup, body.prog)
    unit.constants = {
        PROGRAMMABLE[i]: value for i, value in enumerate(body.prog.values())
    }
    unit.init["math"] = [
        Insn(
            "SFPCONFIG",
            {"imm16_math": 0, "config_dest": 15, "instr_mod1": 1},
            "lane configuration",
        ),
        *setup,
    ]
    copies = [f"copy_tile(x, {t}, dst + {t})" for t in range(width)]
    copies.append(f"copy_tile(cols, 0, dst + {cols_tile})")
    unit.tile["unpack"] = [
        Sync("unpack", "reader", "cb_wait_front(x, width)", "the row's tiles"),
        *[Call(c, "into Dst (fp32, unpacked to Dst)") for c in copies],
    ]
    unit.tile["math"] = [
        Sync("math", "pack", "tile_regs_acquire()", "a free Dst half"),
        *[Call(c, "the math part of the copy (none to Dst)") for c in copies],
        Call(
            "ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, "
            "UnpackDestination::SrcRegs>(dst)",
            "Dst base of tile dst (RISC-V writes)",
        ),
        Insn(
            "STALLWAIT",
            {"stall_res": STALL_SFPU, "wait_res": WAIT_MATH},
            "SFPU waits for the matrix unit",
        ),
        *body.code,
        Sync("math", "pack", "tile_regs_commit()", "hands Dst to the pack CPU"),
    ]
    unit.tile["pack"] = [
        Sync("pack", "math", "tile_regs_wait()", "Dst from the math CPU"),
        *[Call(f"pack_tile(dst + {t}, out)", "the row's tile t") for t in range(width)],
        Sync("pack", "math", "tile_regs_release()", "frees the Dst half"),
    ]
    _checks(unit, body.code)
    # The checks' repairs went into body.code: the tile program takes it again.
    unit.tile["math"][-1 - len(body.code) + _repairs(unit) : -1] = body.code
    if rows:
        at = len(unit.tile["math"]) - 1 - len(body.code)
        items, done = band_groups(body.code, layouts)
        unit.tile["math"][at:-1] = items
        unit.band = True
        unit.passes.append(Pass("schedule", "band", done))
    unit.not_checked = [r for r in NOT_CHECKED if "load-macro" not in r]
    return unit


BAND_ROWS = 8  # the row form's band: two row groups (scheduled together)


def _store_rows(insn, layouts):
    """The first row of the band a store writes: a row's vector or a per-row slot."""
    a = insn.args["dest_reg_addr"]
    slots = layouts["bcast_col"]
    if slots.tile * TILE <= a < (slots.tile + 1) * TILE:
        group = (a - slots.tile * TILE - slots.offset) // 4
    else:
        group = 4 * (a % TILE // 32) + a % 16 // 4
    return 4 * group // BAND_ROWS * BAND_ROWS


def band_groups(code, layouts):
    """Rule band in the row form: an instruction runs while the first band its stores
    write (followed back through the registers) holds a live row: runs of one band
    under `if`, an SFPNOP after a run's last MAD (the next run may not run). Refused
    with lane flags (a flag set in a run would change the lanes of the next)."""
    if any(x.sets_flags() or OPS[x.op].kind == "flags" for x in code):
        raise UnitError("the row form's band takes no lane flags")
    first = [None] * len(code)
    need = {}
    for i in range(len(code) - 1, -1, -1):
        x = code[i]
        if x.op == "SFPSTORE":
            first[i] = _store_rows(x, layouts)
        else:
            found = [need.pop(w) for w in x.writes() if w in need]
            first[i] = min(found) if found else None
        for r in x.reads():
            if first[i] is not None:
                need[r] = min(need.get(r, first[i]), first[i])
    start = 0
    for i in range(len(code)):
        start = start if first[i] is None else first[i]
        first[i] = start
    runs = []
    for x, row in zip(code, first):
        if runs and runs[-1][0] == row:
            runs[-1][1].append(x)
        else:
            runs.append((row, [x]))
    items, nops = [], 0
    for k, (row, run) in enumerate(runs):
        if OPS[run[-1].op].kind == "mad" and k + 1 < len(runs):
            run.append(Insn("SFPNOP", {}, "the next run may not run"))
            nops += 1
        if row == 0:
            items += run
            continue
        test = f"rows - 1u >= {row}u"
        guard = Runtime(test, lambda r, row=row: int(not 1 <= r <= row), guard=True)
        items.append(Loop("band", guard, run))
    done = [
        f"{len(runs)} runs in bands of {BAND_ROWS} rows, "
        f"{sum(isinstance(x, Loop) for x in items)} guarded; {nops} SFPNOP"
    ]
    return items, done


def _repairs(unit):
    """SFPNOPs the mad-reader check inserted (its verdict "repaired: n SFPNOP")."""
    verdict = next(c.verdict for c in unit.checks if c.rule == "mad-reader")
    return int(verdict.split()[1]) if verdict.startswith("repaired") else 0


def _vectors(width):
    """(t, c, p) of every vector of one row group across the row."""
    return [(t, c, p) for t in range(width) for c in range(2) for p in range(2)]


def _stat_phase(b, reduced, per_row, rows, width):
    """Per row group: f of every vector of the row summed lane by lane (a product
    into a multiply-add), the 8 lanes of each lane row summed by rotations, the
    per-row value computed and stored to its slot. Returns the instructions of each
    pair of groups (ids), for the interleave pass."""
    top = reduced.fn.result
    pairs, start = [], 0
    for h, q in GROUPS:
        if q % 2 == 0:
            start = len(b.code)
        acc = None
        for t, c, p in _vectors(width):
            address = Layout("tile", 0).address(t, h, c, q, p)
            loaded = {}

            def source(i, address=address, loaded=loaded, at=(t, h, c, q, p)):
                if i not in loaded:
                    loaded[i] = b.load(address, f"row {at}")
                return loaded[i]

            if acc is not None and top.kind == "mul":
                memo = {}
                a = _value(b, top.args[0], memo, source)
                m = _value(b, top.args[1], memo, source)
                acc = b.mad(a, m, acc, "sum += f")
            else:
                v = _value(b, top, {}, source)
                acc = v if acc is None else b.add(acc, v, "sum += f")
        for k in (1, 2, 4):
            turned = acc
            for _ in range(k):
                turned = b.op(
                    "SFPSHFT2",
                    f"lanes turned (step {k})",
                    lreg_src_c=turned,
                    instr_mod1=ROTATE,
                )
            acc = b.add(acc, turned, f"sum of {2 * k} lanes")
        value = _value(b, per_row.fn.result, {}, lambda i, acc=acc: acc)
        b.store(value, rows.address(0, h, 0, q, 0), f"per-row value, group {h} {q}")
        if q % 2:
            pairs.append({id(x) for x in b.code[start:]})
    return pairs


def _apply_phase(b, stored, layouts, width):
    """Per half of the row: its 4 per-row slots loaded once; per vector column (t,
    c, p) the per-column vector loaded once; the stored function of two row groups
    at a time written over the row. Returns the instructions of each pair (ids)."""
    kinds = [leaf.kind for leaf in stored.operands]
    pairs = []
    for h in range(2):
        slots = {}
        for t, c, p in _vectors(width):
            column = None
            for pair in ((0, 1), (2, 3)):
                start = len(b.code)
                for q in pair:
                    address = layouts["copy"].address(t, h, c, q, p)
                    loaded = {}

                    def source(
                        i,
                        address=address,
                        loaded=loaded,
                        slots=slots,
                        at=(t, h, c, q, p),
                    ):
                        nonlocal column
                        kind = kinds[i]
                        if kind == "bcast_col":
                            if at[3] not in slots:
                                where = layouts[kind].address(*at)
                                note = f"per-row slot {at[1]} {at[3]}"
                                slots[at[3]] = b.load(where, note)
                            return slots[at[3]]
                        if kind == "bcast_row":
                            if column is None:
                                where = layouts[kind].address(*at)
                                note = f"per-column {at[0]} {at[2]} {at[4]}"
                                column = b.load(where, note)
                            return column
                        if i not in loaded:
                            loaded[i] = b.load(address, f"row {at}")
                        return loaded[i]

                    value = _value(b, stored.fn.result, {}, source)
                    b.store(value, address, f"row {(t, h, c, q, p)}")
                pairs.append({id(x) for x in b.code[start:]})
    return pairs


def interleave(code, start, end):
    """Schedule: reorders code[start:end] (a run under lane flags moves as one) so a
    reader of the MAD result issued just before it waits behind another ready word;
    a constant's loads go in front of its first reader. Returns how many moved."""
    whole = code[start:end]
    loaded = {}
    for x in code:
        for v in x.writes():
            loaded[v] = loaded.get(v, True) and x.op == "SFPLOADI"
    consts = {}
    for x in whole:
        v = x.args.get("lreg_ind") if x.op == "SFPLOADI" else None
        if isinstance(v, str) and v.startswith("v") and loaded.get(v):
            consts.setdefault(v, []).append(x)
    # a constant register's setup (loads into L0, SFPCONFIG) stays whole, in front
    setup = []
    for i, x in enumerate(whole):
        if x.op == "SFPCONFIG":
            j = i
            while j > 0 and whole[j - 1].op == "SFPLOADI":
                if whole[j - 1].args["lreg_ind"] != "L0":
                    break
                j -= 1
            setup += whole[j : i + 1]
    held = {id(x) for ws in consts.values() for x in ws} | {id(x) for x in setup}
    block = [x for x in whole if id(x) not in held]
    units = _units(block)
    if units is None:
        return 0
    preds = []
    for i, unit in enumerate(units):
        mine = set()
        for j in range(i):
            if any(_depends(x, y) for x in units[j] for y in unit):
                mine.add(j)
        preds.append(mine)
    height = [1] * len(units)
    for i in range(len(units) - 1, -1, -1):
        for j in range(i + 1, len(units)):
            if i in preds[j]:
                height[i] = max(height[i], height[j] + 1)
    later = {v for x in code[end:] for v in x.reads()}
    written = {v for x in block for v in x.writes()}
    outside = {
        v
        for x in block
        for v in x.reads()
        if _work(v) and v not in written and v not in consts
    }
    outside |= {v for x in code[:start] for v in x.writes() if v in later}
    left = {}
    for x in block:
        for v in x.reads():
            left[v] = left.get(v, 0) + 1
    start_live = {v for v in outside if left.get(v) or v in later}
    # a constant loaded at its reader is live beside the estimate: a tighter bound
    for limit in range(WORK, 2, -1):
        order = _schedule(
            units, preds, height, set(start_live), dict(left), later, limit
        )
        placed, out = set(), []
        for x in (x for i in order for x in units[i]):
            for v in x.reads():
                if v in consts and v not in placed:
                    out += consts[v]
                    placed.add(v)
            out.append(x)
        unread = [w for v, ws in consts.items() if v not in placed for w in ws]
        code[start:end] = setup + unread + out
        if _pressure(code) <= WORK:
            break
    else:
        code[start:end] = whole
        return 0
    return sum(1 for k, i in enumerate(order) if k != i)


def _schedule(units, preds, height, live, left, later, limit):
    """The issue order of `units`: the tallest ready one that keeps at most `limit`
    values live, else the one leaving fewest; not a reader of the MAD just issued."""
    order, issued, previous = [], set(), None
    while len(order) < len(units):
        ready = [i for i in range(len(units)) if i not in issued and preds[i] <= issued]
        ready.sort(key=lambda i: (-height[i], i))
        after = {i: len(_unit_after(units[i], live, left, later)) for i in ready}
        fits = [i for i in ready if after[i] <= limit]
        # none fits: the one that leaves the fewest values live
        ready = fits or sorted(ready, key=lambda i: after[i])
        pick = ready[0]
        if previous is not None and OPS[previous.op].kind == "mad":
            for i in ready:
                if not set(previous.writes()) & set(units[i][0].reads()):
                    pick = i
                    break
        live = _unit_after(units[pick], live, left, later, issue=True)
        order.append(pick)
        issued.add(pick)
        previous = units[pick][-1]
    return order


def _units(block):
    """The block as units that move whole: an instruction, or a run under lane flags
    (None when a run does not close inside the block)."""
    units, i = [], 0
    while i < len(block):
        if not block[i].sets_flags():
            units.append([block[i]])
            i += 1
            continue
        depth, j = 0, i
        while True:
            if j == len(block):
                return None
            x = block[j]
            if x.sets_flags():
                depth = 0 if x.op == "SFPENCC" else depth + 1
            j += 1
            if depth == 0:
                break
        units.append(block[i:j])
        i = j
    return units


def _depends(other, insn):
    """True when `insn` must stay after `other` (registers or a Dst row)."""
    raw = set(other.writes()) & set(insn.reads())
    war = set(other.reads()) & set(insn.writes())
    waw = set(other.writes()) & set(insn.writes())
    dst = "dest_reg_addr" in other.args and "dest_reg_addr" in insn.args
    same = dst and other.args["dest_reg_addr"] == insn.args["dest_reg_addr"]
    touches = "SFPSTORE" in (other.op, insn.op)
    return bool(raw or war or waw or (same and touches))


def _unit_after(unit, live, left, later, issue=False):
    """The live work values once every instruction of `unit` has issued; with
    `issue`, `left` counts their reads off."""
    out, counts = set(live), dict(left)
    for x in unit:
        for v in x.reads():
            counts[v] -= 1
        out = _after(x, out, counts, later, issue=True)
    if issue:
        left.update(counts)
    return out


def _pressure(code):
    """The most work registers the allocator needs at one instruction of `code` (as
    `allocate` counts: a value whose last read is an instruction frees its register
    for that instruction's result)."""
    first, last = {}, {}
    for i, insn in enumerate(code):
        for v in insn.writes():
            if _work(v):
                first.setdefault(v, i)
        for v in insn.reads():
            if _work(v):
                last[v] = i
    starts, ends = [0] * (len(code) + 2), [0] * (len(code) + 2)
    for v, i in first.items():
        starts[i] += 1
        end = last.get(v, i)
        if end == i or v in code[end].writes():  # written there: held through it
            end += 1
        ends[end] += 1
    most = live = 0
    for i in range(len(code)):
        live += starts[i] - ends[i]
        most = max(most, live)
    return most


def _work(value):
    """True for a value that takes a work register (not a fixed or constant one)."""
    return isinstance(value, str) and value.startswith("v")


def _after(insn, live, left, later, issue=False):
    """The live work values once `insn` has issued (`left`: reads still to come in
    the block, before this one's when not `issue`)."""
    out = set(live)
    for v in insn.reads():
        rest = left.get(v, 0) - (0 if issue else 1)
        if rest <= 0 and v not in later:
            out.discard(v)
    out |= {v for v in insn.writes() if _work(v)}
    return out


def _check_broadcast(unit, code, layouts, width):
    """broadcast: every stored row vector was computed from the row vector at its
    own address, the per-row slot of its row group and the per-column vector of its
    columns, followed back through the values (before register allocation)."""
    writer = {}
    sources = {}
    checked = depth = 0
    for insn in code:
        if insn.sets_flags():
            depth = 0 if insn.op == "SFPENCC" else depth + 1
        if insn.op == "SFPLOAD":
            sources[insn.writes()[0]] = {insn.args["dest_reg_addr"]}
        elif insn.op == "SFPSTORE":
            address = insn.args["dest_reg_addr"]
            value = insn.args["lreg_ind"]
            index = _index(address, width)
            if index is None:
                continue
            want = {layouts[k].address(*index) for k in layouts}
            got = sources.get(value, set())
            if not got <= want or layouts["copy"].address(*index) not in got:
                raise UnitError(
                    f"broadcast: the row vector at Dst + {address} reads {sorted(got)}"
                    f", not its own slots {sorted(want)}"
                )
            checked += 1
        else:
            got = set()
            for v in insn.reads():
                got |= sources.get(v, set())
            for v in insn.writes():
                # a write under lane flags keeps what the other lanes held
                sources[v] = got | (sources.get(v, set()) if depth else set())
                writer[v] = insn
    unit.checks.append(
        Check(
            "broadcast",
            "a stored row vector meets the per-row slot of its rows and the "
            "per-column vector of its columns",
            f"ok: {checked} vectors followed back to their loads",
        )
    )


def _check_lanes(unit, code, rows):
    """lanes: a value stored into the per-row tile is complete, every lane of a lane
    row holding the sum of all 8 (a + a turned by its span doubles the span; values
    the same in every lane have span 8). Before register allocation."""
    span, turned = {}, {}
    stores = 0
    for insn in code:
        reads = insn.reads()
        if insn.op == "SFPLOAD":
            # a per-row slot holds complete values
            whole = insn.args["dest_reg_addr"] // TILE == rows.tile
            span[insn.writes()[0]] = 8 if whole else 1
            continue
        if insn.op == "SFPSTORE":
            address = insn.args["dest_reg_addr"]
            if address // TILE == rows.tile:
                got = span.get(insn.args["lreg_ind"], 8)
                if got != 8:
                    raise UnitError(
                        f"lanes: the per-row value stored at Dst + {address} sums "
                        f"{got} of the 8 lanes of a row"
                    )
                stores += 1
            continue
        out = insn.writes()
        if not out:
            continue
        if insn.op == "SFPSHFT2":
            base, shift = turned.get(reads[0], (reads[0], 0))
            turned[out[0]] = (base, shift + 1)
            span[out[0]] = span.get(base, 8)
            continue
        values = [r for r in reads if isinstance(r, str) and r in span]
        width = min((span[v] for v in values), default=8)
        for a in values:
            for b in values:
                if turned.get(b) == (a, span[a]):
                    width = 2 * span[a]
        span[out[0]] = width
    unit.checks.append(
        Check(
            "lanes",
            "a per-row value is summed over all 8 lanes of its row before a store "
            "broadcasts it",
            f"ok: {stores} slots, each complete",
        )
    )


def _index(address, width):
    """(t, h, c, q, p) of a row vector's Dst address, or None outside the row."""
    if not 0 <= address < TILE * width:
        return None
    t, rest = divmod(address, TILE)
    h, rest = divmod(rest, 32)
    c, rest = divmod(rest, 16)
    q, rest = divmod(rest, 4)
    return t, h, c, q, rest // 2


ROW_SLOTS = (0, 2, 32, 34)  # the per-row slots of a group in the rows tile (Dst rows)
DST_TILES = 8  # fp32 tiles in all of Dst: the row form takes it whole


def _below(node, kinds, stop):
    """The nodes of `kinds` under `node`, not descending past a node of `stop`."""
    found, seen, todo = [], set(), [node]
    while todo:
        n = todo.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if n.kind in kinds:
            found.append(n)
        if n.kind not in stop or n is node:
            todo += list(n.args)
    return found


def _levels(reduces, stats):
    """Each reduction's level (a pass over the row: one more than the per-row values
    its argument reads by .col()) and each per-row value's (its last reduction's)."""
    level = {}

    def of(node):
        if id(node) not in level:
            if node.kind == "reduce":
                under = [c.args[0] for c in _below(node.args[0], ("col",), ("col",))]
                level[id(node)] = 1 + max((of(p) for p in under), default=-1)
            else:
                under = _below(node, ("reduce",), ("reduce",))
                level[id(node)] = max(of(r) for r in under)
        return level[id(node)]

    for n in [*reduces, *stats]:
        of(n)
    return level


def level_program(ir, width, rows_tile, cols_tile, shapes, row, rows=False):
    """The row form with several reductions in levels, a pass over the row a level: a
    level's sums may read the per-row values of the levels before it by .col(); each
    per-row value kept across a level (and each sum a later value reads) in a slot of
    the rows tile; one per-column tile a resident row (Dst tile cols_tile + k)."""
    if rows:
        fail(f"{ir.name}: the band takes the row form of one reduction")
    nodes = ir.nodes()
    reduces = [n for n in nodes if n.kind == "reduce"]
    stats = list({id(c.args[0]): c.args[0] for c in nodes if c.kind == "col"}.values())
    residents = list({id(n): n for n in nodes if n.kind == "resident"}.values())
    level = _levels(reduces, stats)
    kept = [
        r
        for r in reduces
        if all(r is not p for p in stats)
        and any(
            level[id(p)] > level[id(r)]
            for p in stats
            if any(x is r for x in _below(p, ("reduce",), ("reduce",)))
        )
    ]
    held = [*stats, *kept]
    if len(held) > len(ROW_SLOTS):
        fail(
            f"{ir.name}: {len(held)} per-row values kept across the row's passes; the "
            f"rows tile holds {len(ROW_SLOTS)} (a slot a value a row group)"
        )
    if cols_tile + len(residents) > DST_TILES:
        fail(
            f"{ir.name}: {len(residents)} resident rows; Dst's {DST_TILES} fp32 tiles "
            f"hold {DST_TILES - cols_tile} per-column tiles after the row's"
        )
    slot = {id(v): Layout("rows", rows_tile, ROW_SLOTS[k]) for k, v in enumerate(held)}
    cols = {id(r): Layout("cols", cols_tile + k) for k, r in enumerate(residents)}

    def leaf_of(node, offset, kind):
        if offset is not None:
            return None
        if node.kind in ("load", "resident") and node.attrs["view"].kind == "row":
            return chains.Leaf(node, kind, None)
        if node.kind == "reduce" or id(node) in slot:
            return chains.Leaf(node, kind, None)
        return None

    compiler = chains._Compiler(ir.name, shapes, {})
    limit = chains.SLOTS + len(residents) + len(stats)
    summed = {id(r): compiler.chain(r.args[0], leaf_of, limit=limit) for r in reduces}
    made = {
        id(p): compiler.chain(p, leaf_of, limit=limit)
        for p in stats
        if p.kind != "reduce"
    }
    stored = compiler.chain(ir.stores[0].value, leaf_of, limit=limit)
    for c in summed.values():
        if any(leaf.kind not in ("copy", "bcast_col") for leaf in c.operands):
            fail(f"{ir.name}: a sum along the row reads the row and per-row values")
    for c in [*summed.values(), *made.values(), stored]:
        if c.values or c.prev is not None:
            fail(f"{ir.name}: the row form takes no runtime scalars and one stage")
        for leaf in c.operands:
            if leaf.kind == "copy" and leaf.source.kind == "resident":
                fail(".row() broadcasts a resident row (a weight)")
            if leaf.kind == "bcast_row" and leaf.source.kind != "resident":
                fail(".row() broadcasts a resident row (a weight)")
    body = Body()
    unit = UnitProgram(ir.name, "fp32")
    unit.operands = [row.attrs["view"].tensor.name]
    pairs = []
    for at in range(max(level[id(r)] for r in reduces) + 1):
        mine = [r for r in reduces if level[id(r)] == at]
        done = [p for p in stats if level[id(p)] == at]
        pairs += _level_phase(body, mine, done, summed, made, slot, width)
    pairs += _apply_slots(body, stored, slot, cols, width)
    unit.passes.append(Pass("algebra", "fold_negation", fold_negation(body.code)))
    unit.passes.append(Pass("algebra", "fuse_mad", fuse_mad(body.code)))
    levels = 1 + max(level.values())
    unit.passes.append(
        Pass(
            "layout",
            "lanes",
            [
                f"{len(reduces)} sum(s) in {levels} pass(es) over the row; {len(held)} "
                f"per-row value(s) in slots of tile {rows_tile}; {len(residents)} "
                f"per-column value(s) in tile(s) from {cols_tile}"
            ],
        )
    )
    rows_layout = Layout("rows", rows_tile)
    _check_lanes(unit, body.code, rows_layout)
    moved = 0
    for pair in pairs:
        at = [i for i, x in enumerate(body.code) if id(x) in pair]
        if at:
            moved += interleave(body.code, at[0], at[-1] + 1)
    unit.passes.append(
        Pass("schedule", "interleave", [f"{moved} instructions moved in {len(pairs)}"])
    )
    layouts = {"copy": Layout("tile", 0)}
    layouts.update({f"slot{k}": v for k, v in enumerate(slot.values())})
    layouts.update({f"cols{k}": v for k, v in enumerate(cols.values())})
    _check_broadcast(unit, body.code, layouts, width)
    setup = []
    unit.passes.append(Pass("schedule", "hoist", hoist(body.code, setup)))
    unit.passes.append(Pass("layout", "allocate", allocate(body.code, body.prog)))
    allocate(setup, body.prog)
    unit.constants = {
        PROGRAMMABLE[i]: value for i, value in enumerate(body.prog.values())
    }
    unit.init["math"] = [
        Insn(
            "SFPCONFIG",
            {"imm16_math": 0, "config_dest": 15, "instr_mod1": 1},
            "lane configuration",
        ),
        *setup,
    ]
    copies = [f"copy_tile(x, {t}, dst + {t})" for t in range(width)]
    copies += [
        f"copy_tile(cols{k}, 0, dst + {cols_tile + k})" for k in range(len(cols))
    ]
    unit.tile["unpack"] = [
        Sync("unpack", "reader", "cb_wait_front(x, width)", "the row's tiles"),
        *[Call(c, "into Dst (fp32, unpacked to Dst)") for c in copies],
    ]
    unit.tile["math"] = [
        Sync("math", "pack", "tile_regs_acquire()", "a free Dst half"),
        *[Call(c, "the math part of the copy (none to Dst)") for c in copies],
        Call(
            "ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, "
            "UnpackDestination::SrcRegs>(dst)",
            "Dst base of tile dst (RISC-V writes)",
        ),
        Insn(
            "STALLWAIT",
            {"stall_res": STALL_SFPU, "wait_res": WAIT_MATH},
            "SFPU waits for the matrix unit",
        ),
        *body.code,
        Sync("math", "pack", "tile_regs_commit()", "hands Dst to the pack CPU"),
    ]
    unit.tile["pack"] = [
        Sync("pack", "math", "tile_regs_wait()", "Dst from the math CPU"),
        *[Call(f"pack_tile(dst + {t}, out)", "the row's tile t") for t in range(width)],
        Sync("pack", "math", "tile_regs_release()", "frees the Dst half"),
    ]
    _checks(unit, body.code)
    unit.tile["math"][-1 - len(body.code) + _repairs(unit) : -1] = body.code
    unit.not_checked = [r for r in NOT_CHECKED if "load-macro" not in r]
    return unit


def _level_phase(b, mine, done, summed, made, slot, width):
    """A level, per row group: the level's sums (`mine`) over every vector of the
    row (a product into a multiply-add), their lanes summed by rotations, then the
    per-row values the level completes (`done`) and the kept sums stored to their
    slots. Returns the instructions of each pair of groups (ids)."""
    pairs, start = [], 0
    for h, q in GROUPS:
        if q % 2 == 0:
            start = len(b.code)
        held = {}  # per-row values of earlier levels, loaded once a group

        def kept_value(node, h=h, q=q, held=held):
            if id(node) not in held:
                where = slot[id(node)].address(0, h, 0, q, 0)
                held[id(node)] = b.load(where, f"per-row slot, group {h} {q}")
            return held[id(node)]

        accs = {}
        for t, c, p in _vectors(width):
            address = Layout("tile", 0).address(t, h, c, q, p)
            loaded = {}
            for r in mine:
                chain = summed[id(r)]

                def source(i, chain=chain, address=address, loaded=loaded):
                    leaf = chain.operands[i]
                    if leaf.kind == "bcast_col":
                        return kept_value(leaf.source)
                    if "row" not in loaded:
                        loaded["row"] = b.load(address, f"row {address}")
                    return loaded["row"]

                top, acc = chain.fn.result, accs.get(id(r))
                if acc is not None and top.kind == "mul":
                    memo = {}
                    x = _value(b, top.args[0], memo, source)
                    m = _value(b, top.args[1], memo, source)
                    accs[id(r)] = b.mad(x, m, acc, "sum += f")
                else:
                    v = _value(b, top, {}, source)
                    accs[id(r)] = v if acc is None else b.add(acc, v, "sum += f")
        for r in mine:
            acc = accs[id(r)]
            for k in (1, 2, 4):
                turned = acc
                for _ in range(k):
                    turned = b.op(
                        "SFPSHFT2",
                        f"lanes turned (step {k})",
                        lreg_src_c=turned,
                        instr_mod1=ROTATE,
                    )
                acc = b.add(acc, turned, f"sum of {2 * k} lanes")
            accs[id(r)] = acc
            if id(r) in slot:  # read by .col() itself, or by a later level's value
                where = slot[id(r)].address(0, h, 0, q, 0)
                b.store(acc, where, f"per-row sum, group {h} {q}")
        for p in done:
            if id(p) not in made:
                continue  # a sum read by .col() itself: stored above

            def stat_source(i, p=p, accs=accs, kept_value=kept_value):
                leaf = made[id(p)].operands[i]
                if id(leaf.source) in accs:
                    return accs[id(leaf.source)]
                return kept_value(leaf.source)

            value = _value(b, made[id(p)].fn.result, {}, stat_source)
            b.store(value, slot[id(p)].address(0, h, 0, q, 0), f"per-row value {h} {q}")
        if q % 2:
            pairs.append({id(x) for x in b.code[start:]})
    return pairs


def _apply_slots(b, stored, slot, cols, width):
    """Per vector column (t, c, p) and pair of row groups: each per-row value loaded
    from its slot once a group, each per-column vector once a vector column, the
    stored function written over the row. Returns the instructions of each pair."""
    pairs = []
    for h in range(2):
        for t, c, p in _vectors(width):
            column = {}
            for pair in ((0, 1), (2, 3)):
                start = len(b.code)
                for q in pair:
                    address = Layout("tile", 0).address(t, h, c, q, p)
                    loaded = {}

                    def source(
                        i,
                        address=address,
                        loaded=loaded,
                        at=(t, h, c, q, p),
                        column=column,
                    ):
                        leaf = stored.operands[i]
                        key = id(leaf.source)
                        if leaf.kind == "bcast_col":
                            if key not in loaded:
                                where = slot[key].address(*at)
                                loaded[key] = b.load(
                                    where, f"per-row slot {at[1]} {at[3]}"
                                )
                            return loaded[key]
                        if leaf.kind == "bcast_row":
                            if key not in column:
                                where = cols[key].address(*at)
                                column[key] = b.load(
                                    where, f"per-column {at[0]} {at[2]} {at[4]}"
                                )
                            return column[key]
                        if "row" not in loaded:
                            loaded["row"] = b.load(address, f"row {at}")
                        return loaded["row"]

                    value = _value(b, stored.fn.result, {}, source)
                    b.store(value, address, f"row {(t, h, c, q, p)}")
                pairs.append({id(x) for x in b.code[start:]})
    return pairs
