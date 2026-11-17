"""The row form: a row of `width` whole tiles in Dst, one exact sum along the row
(`reduce_sum(f(row), exact=True)`), a per-row value computed from it, and the stored
row computed from the row, that value broadcast with `.col()` and a per-column value
broadcast with `.row()`. Layout rule: a per-row value is lanes of one tile."""

from dataclasses import dataclass
from functools import partial

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
    maximum,
    minimum,
)
from komorebi.dsl.tile.schedule import interleave
from komorebi.dsl.tile.unit import (
    OPS,
    PROGRAMMABLE,
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
# The reductions along the row, each with the value a chunk form's slot starts at
START = {"sum": 0.0, "max": float("-inf"), "min": float("inf")}
SUMMED = ("copy", "bcast_col", "bcast_row")  # what a level's reduction reads


def _fold(b, op, acc, v, note):
    """acc and v combined by reduction `op`: their sum, larger or smaller."""
    if op == "max":
        return maximum(b, acc, v)
    if op == "min":
        return minimum(b, acc, v)
    return b.add(acc, v, note)


def _lanes(b, op, acc):
    """The 8 lanes of each lane row of `acc` combined by `op` (rotations by 1, 2 and 4
    lanes), the value in every lane."""
    for k in (1, 2, 4):
        turned = acc
        for _ in range(k):
            turned = b.op(
                "SFPSHFT2",
                f"lanes turned (step {k})",
                lreg_src_c=turned,
                instr_mod1=ROTATE,
            )
        acc = _fold(b, op, acc, turned, f"sum of {2 * k} lanes")
    return acc


def _reduced(red):
    """Refused unless `red` is reduce_sum, reduce_max or reduce_min, exact, along
    the row."""
    if red.attrs["op"] not in START or not red.attrs["exact"] or red.attrs["dim"] != -1:
        fail(
            "the row form's reduction is reduce_sum, reduce_max or reduce_min(..., "
            "exact=True) along the row"
        )


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
        _reduced(red)
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
    pairs = _stat_phase(
        body, reduced, per_row, layouts["bcast_col"], width, red.attrs["op"]
    )
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


def _stat_phase(b, reduced, per_row, rows, width, op="sum"):
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

            if acc is not None and top.kind == "mul" and op == "sum":
                memo = {}
                a = _value(b, top.args[0], memo, source)
                m = _value(b, top.args[1], memo, source)
                acc = b.mad(a, m, acc, "sum += f")
            else:
                v = _value(b, top, {}, source)
                acc = v if acc is None else _fold(b, op, acc, v, "sum += f")
        acc = _lanes(b, op, acc)
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


def _check_broadcast(unit, code, layouts, width, bases=(0,)):
    """broadcast: every stored row vector (the row's, or a result's from a tile of
    `bases`) was computed from the row vector at its own address, the per-row slot
    of its row group and the per-column vector of its columns (before allocation)."""
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
            index = next(
                (
                    _index(address - TILE * base, width)
                    for base in bases
                    if _index(address - TILE * base, width) is not None
                ),
                None,
            )
            if index is None:
                continue
            want = {layouts[k].address(*index) for k in layouts}
            got = sources.get(value, set())
            own = layouts["copy"].address(*index)
            slots = {  # the per-row values' (the row form's: "bcast_col")
                layouts[k].address(*index)
                for k in layouts
                if k[:4] == "slot" or k == "bcast_col"
            }
            alone = bool(got) and got <= slots  # a per-row value alone (part value)
            if not got <= want or own not in got and not alone:
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


def _check_lanes(unit, code, rows, sums=None):
    """lanes: a value stored into the per-row tile (or the chunk form's sums tile,
    `sums`) is complete, every lane of a lane row holding the sum of all 8 (a + a
    turned by its span doubles the span; values the same in every lane have span 8).
    Before register allocation."""
    tiles = {rows.tile} if sums is None else {rows.tile, sums}
    span, turned, copied = {}, {}, {}  # copied: a copy's source (max and min's)
    stores = 0
    for insn in code:
        reads = [copied.get(r, r) for r in insn.reads()]
        if insn.op == "SFPLOAD":
            # a per-row slot holds complete values
            whole = insn.args["dest_reg_addr"] // TILE in tiles
            span[insn.writes()[0]] = 8 if whole else 1
            continue
        if insn.op == "SFPSTORE":
            address = insn.args["dest_reg_addr"]
            if address // TILE in tiles:
                got = span.get(insn.args["lreg_ind"], 8)
                if got != 8:
                    raise UnitError(
                        f"lanes: the per-row value stored at Dst + {address} sums "
                        f"{got} of the 8 lanes of a row"
                    )
                stores += 1
            continue
        out = insn.writes()
        for written in out:
            copied.pop(written, None)
        if not out:
            continue
        if insn.op == "SFPSHFT2":
            base, shift = turned.get(reads[0], (reads[0], 0))
            turned[out[0]] = (base, shift + 1)
            span[out[0]] = span.get(base, 8)
            continue
        if insn.op == "SFPMOV" and reads[0] in span and not insn.args["instr_mod1"]:
            copied[out[0]] = reads[0]
            continue
        values = [r for r in reads if isinstance(r, str) and r in span]
        width = min((span[v] for v in values), default=8)
        for a in values:
            for b in values:
                if turned.get(b) == (a, span[a]):
                    width = 2 * span[a]
        for written in out:  # a swap writes both its registers
            span[written] = width
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
        if any(leaf.kind not in SUMMED for leaf in c.operands):
            fail(
                f"{ir.name}: a sum along the row reads the row, per-row values and "
                "resident rows"
            )
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
        pairs += _level_phase(body, mine, done, summed, made, slot, width, cols)
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


def _level_phase(b, mine, done, summed, made, slot, width, cols=None):
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
            loaded, at = {}, (t, h, c, q, p)
            for r in mine:
                chain = summed[id(r)]

                def source(i, chain=chain, address=address, loaded=loaded, at=at):
                    leaf = chain.operands[i]
                    if leaf.kind == "bcast_col":
                        return kept_value(leaf.source)
                    if leaf.kind == "bcast_row":  # a resident row's columns
                        key = id(leaf.source)
                        if key not in loaded:
                            where = cols[key].address(*at)
                            loaded[key] = b.load(where, f"per-column {where}")
                        return loaded[key]
                    if "row" not in loaded:
                        loaded["row"] = b.load(address, f"row {address}")
                    return loaded["row"]

                top, acc, op = chain.fn.result, accs.get(id(r)), r.attrs["op"]
                if acc is not None and top.kind == "mul" and op == "sum":
                    memo = {}
                    x = _value(b, top.args[0], memo, source)
                    m = _value(b, top.args[1], memo, source)
                    accs[id(r)] = b.mad(x, m, acc, "sum += f")
                else:
                    v = _value(b, top, {}, source)
                    accs[id(r)] = v if acc is None else _fold(b, op, acc, v, "sum += f")
        for r in mine:
            acc = _lanes(b, r.attrs["op"], accs[id(r)])
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


def _apply_slots(b, stored, slot, cols, width, bases=(0,)):
    """Per vector column (t, c, p) and pair of row groups: each per-row value loaded
    from its slot once a group, each per-column vector once a vector column, stored
    result k written at Dst tile bases[k] (result 0 over the row; `stored` one chain
    or a list). Returns the instructions of each pair."""
    chains_ = stored if isinstance(stored, list) else [stored]
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
                        chain=None,
                    ):
                        leaf = chain.operands[i]
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

                    values = [  # each result from the one load of the row vector
                        _value(b, c_.fn.result, {}, partial(source, chain=c_))
                        for c_ in chains_
                    ]
                    for base, value in zip(bases, values):
                        to = Layout("tile", base).address(t, h, c, q, p)
                        b.store(value, to, f"row {(t, h, c, q, p)} of tile {base}")
                pairs.append({id(x) for x in b.code[start:]})
    return pairs


CHUNK = 2  # tiles of a wide row a call takes (a per-column tile holds 2)
CHUNK_ROWS, CHUNK_SUMS, CHUNK_COLS = 2, 3, 4  # the chunk form's Dst tiles


def chunk_program(ir, width):
    """The row form over `width` tiles (over 2) in chunks of 2 (Dst 0, 1): part
    `start` zeroes the sums (Dst 3), `sum<L>` adds a chunk's share of level L's sums,
    `finish<L>` makes level L's per-row values (Dst 2); the body writes a chunk with
    resident row k's chunk in Dst 4 + k; an odd row's last tile: parts `*_one`."""
    if width <= CHUNK:
        fail(f"a wide row is more than {CHUNK} tiles: {width} tiles given")
    if not {t.dtype for t in ir.tensors} <= set(TILE_DTYPES):
        fail(
            f"{ir.name}: tensors are fp32 or of the B family "
            f"({', '.join(TILE_DTYPES[1:])})"
        )
    shapes = chains._Shapes(True)
    nodes = ir.nodes()
    for n in nodes:
        shapes(n)
    reduces = [n for n in nodes if n.kind == "reduce"]
    if not reduces:
        fail(f"{ir.name}: the row form takes a reduction")
    for red in reduces:
        _reduced(red)
    cols_ = [n for n in nodes if n.kind == "col"]
    if any(shapes(c.args[0]) != "col" for c in cols_):
        fail("a per-row value meets the row through .col()")
    row = [n for n in nodes if n.kind == "load" and n.attrs["view"].kind == "row"]
    if len({id(n.attrs["view"].tensor) for n in row}) != 1:
        fail("the row form reads one row tensor (loaded once)")
    stats = list({id(c.args[0]): c.args[0] for c in cols_}.values())
    residents = list({id(n): n for n in nodes if n.kind == "resident"}.values())
    level = _levels(reduces, stats)
    for what, count, most in (
        ("per-row values read by .col()", len(stats), len(ROW_SLOTS)),
        ("sums along the row", len(reduces), len(ROW_SLOTS)),
        ("resident rows", len(residents), DST_TILES - CHUNK_COLS),
    ):
        if count > most:
            fail(
                f"{ir.name}: {count} {what}; a wide row's call holds {most} (the "
                f"chunk, the values' tile and the sums' tile take {CHUNK_COLS} of "
                f"Dst's {DST_TILES} fp32 tiles; a tile holds {len(ROW_SLOTS)} slots)"
            )
    slot = {
        id(v): Layout("rows", CHUNK_ROWS, ROW_SLOTS[k]) for k, v in enumerate(stats)
    }
    sums = {
        id(r): Layout("rows", CHUNK_SUMS, ROW_SLOTS[k]) for k, r in enumerate(reduces)
    }
    cols = {id(r): Layout("cols", CHUNK_COLS + k) for k, r in enumerate(residents)}
    free = CHUNK_COLS + len(residents)  # result k > 0: its own 2 tiles from here
    bases = [0, *range(free, free + CHUNK * (len(ir.stores) - 1), CHUNK)]
    if free + CHUNK * (len(ir.stores) - 1) > DST_TILES:
        fail(
            f"{ir.name}: {len(ir.stores)} results; a wide row's call holds "
            f"{1 + (DST_TILES - free) // CHUNK} with {len(residents)} resident "
            f"row(s) (result 0 over the chunk, each other {CHUNK} tiles of Dst's "
            f"{DST_TILES} past the residents)"
        )

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
    stored = [compiler.chain(st.value, leaf_of, limit=limit) for st in ir.stores]
    for c in summed.values():
        if any(leaf.kind not in SUMMED for leaf in c.operands):
            fail(
                f"{ir.name}: a sum along the row reads the row, per-row values and "
                "resident rows"
            )
    for c in [*summed.values(), *made.values(), *stored]:
        if c.values or c.prev is not None:
            fail(f"{ir.name}: the row form takes no runtime scalars and one stage")
        for leaf in c.operands:
            if leaf.kind == "copy" and leaf.source.kind == "resident":
                fail(".row() broadcasts a resident row (a weight)")
            if leaf.kind == "bcast_row" and leaf.source.kind != "resident":
                fail(".row() broadcasts a resident row (a weight)")
    b = Body()
    marks, pairs = [], []  # (part, its first instruction's index), interleave pairs

    def part(name):
        b.op("SFPNOP", f"part {name}")  # a marker, cut out below
        marks.append(name)

    part("start")
    zero = b.const(0.0, "0")
    ops = {r.attrs["op"] for r in reduces} - {"sum"}
    first = {"sum": zero, **{op: b.const(START[op], f"{op} start") for op in ops}}
    for h, q in GROUPS:
        for r in reduces:
            op = r.attrs["op"]
            note = f"sum := 0, {h} {q}" if op == "sum" else f"{op} := its start"
            b.store(first[op], sums[id(r)].address(0, h, 0, q, 0), note)
    levels = 1 + max(level.values())
    for at in range(levels):
        mine = [r for r in reduces if level[id(r)] == at]
        done = [p for p in stats if level[id(p)] == at]
        for tail, tiles in [("", CHUNK), ("_one", 1)][: 1 + width % CHUNK]:
            part(f"sum{at}{tail}")  # an odd row's last chunk: one tile
            pairs += _chunk_sums(b, mine, summed, slot, sums, tiles, cols)
        part(f"finish{at}")
        _chunk_finish(b, done, made, slot, sums)
    part("apply")
    pairs += _apply_slots(b, stored, slot, cols, CHUNK, bases)
    if width % CHUNK:
        part("apply_one")
        pairs += _apply_slots(b, stored, slot, cols, 1, bases)
    if len(stored) == 1 and all(
        leaf.kind == "bcast_col" for leaf in stored[0].operands
    ):
        part("value")  # a per-row value alone: one tile of it, for a field
        pairs += _apply_slots(b, stored, slot, cols, 1)
    unit = UnitProgram(ir.name, "fp32")
    unit.results, unit.chunk_results = len(stored), bases
    unit.chunk_fields_in_sums = any(
        leaf.kind == "bcast_row" for c in summed.values() for leaf in c.operands
    )
    unit.operands = [row[0].attrs["view"].tensor.name]
    unit.passes.append(Pass("algebra", "fold_negation", fold_negation(b.code)))
    unit.passes.append(Pass("algebra", "fuse_mad", fuse_mad(b.code)))
    unit.passes.append(
        Pass(
            "layout",
            "chunks",
            [
                f"a row of {width} tiles in {width // CHUNK} chunks of {CHUNK}; "
                f"{len(reduces)} sum(s) in {levels} pass(es) over the chunks, in slots "
                f"of tile {CHUNK_SUMS}; {len(stats)} per-row value(s) in slots of "
                f"tile {CHUNK_ROWS}; {len(residents)} per-column chunk(s) from tile "
                f"{CHUNK_COLS}"
            ],
        )
    )
    _check_lanes(unit, b.code, Layout("rows", CHUNK_ROWS), CHUNK_SUMS)
    moved = 0
    for pair in pairs:
        at = [i for i, x in enumerate(b.code) if id(x) in pair]
        if at:
            moved += interleave(b.code, at[0], at[-1] + 1)
    unit.passes.append(
        Pass("schedule", "interleave", [f"{moved} instructions moved in {len(pairs)}"])
    )
    layouts = {"copy": Layout("tile", 0)}
    layouts.update({f"slot{k}": v for k, v in enumerate(slot.values())})
    layouts.update({f"sum{k}": v for k, v in enumerate(sums.values())})
    layouts.update({f"cols{k}": v for k, v in enumerate(cols.values())})
    _check_broadcast(unit, b.code, layouts, CHUNK, bases)
    setup = []
    unit.passes.append(Pass("schedule", "hoist", hoist(b.code, setup)))
    unit.passes.append(Pass("layout", "allocate", allocate(b.code, b.prog)))
    allocate(setup, b.prog)
    unit.constants = {PROGRAMMABLE[i]: value for i, value in enumerate(b.prog.values())}
    unit.init["math"] = [
        Insn(
            "SFPCONFIG",
            {"imm16_math": 0, "config_dest": 15, "instr_mod1": 1},
            "lane configuration",
        ),
        *setup,
    ]
    head = [
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
    ]
    unit.tile["math"] = [*head, *b.code]
    _checks(unit, b.code)
    cuts = [
        i for i, x in enumerate(b.code) if x.op == "SFPNOP" and x.note[:5] == "part "
    ]
    for k, (name, at) in enumerate(zip(marks, cuts)):
        end = cuts[k + 1] if k + 1 < len(cuts) else len(b.code)
        items = [*head, *b.code[at + 1 : end]]
        if name == "apply":
            unit.tile["math"] = items
        else:
            unit.parts[name] = items
    unit.tile["unpack"] = [
        Sync("unpack", "reader", "cb_wait_front(x, width)", "the row's tiles"),
        Call("copy_tile(x, 2 c + t, dst + t)", "chunk c into Dst (fp32)"),
    ]
    unit.tile["pack"] = [
        Sync("pack", "math", "tile_regs_wait()", "Dst from the math CPU"),
        Call("pack_tile(dst + t, out)", "the chunk's tile t"),
        Sync("pack", "math", "tile_regs_release()", "frees the Dst half"),
    ]
    unit.not_checked = [r for r in NOT_CHECKED if "load-macro" not in r]
    return unit


def chunk_cost(unit, width):
    """(instructions, cycles) of a wide row on the math CPU: the sum parts and the
    body once a chunk, the others once."""
    chunks, count, cycles = width // CHUNK, 0, 0.0
    for name, items in [("apply", unit.tile["math"]), *unit.parts.items()]:
        probe = UnitProgram(unit.name, unit.dest)
        probe.tile["math"] = items
        n, c, _ = probe.cost("math")
        full = name == "apply" or name.startswith("sum") and "_" not in name
        times = chunks if full else 1
        count, cycles = count + n * times, cycles + c * times
    return count, round(cycles, 1)


def _chunk_sums(b, mine, summed, slot, sums, tiles, cols=None):
    """Part sum<L>, per row group: each of the level's sums over the chunk's vectors,
    its lanes summed by rotations, added into its slot. Returns the interleave
    pairs."""
    pairs, start = [], 0
    for h, q in GROUPS:
        if q % 2 == 0:
            start = len(b.code)
        held = {}

        def kept_value(node, h=h, q=q, held=held):
            if id(node) not in held:
                where = slot[id(node)].address(0, h, 0, q, 0)
                held[id(node)] = b.load(where, f"per-row slot, group {h} {q}")
            return held[id(node)]

        accs = {}
        for t, c, p in _vectors(tiles):
            address = Layout("tile", 0).address(t, h, c, q, p)
            loaded, at = {}, (t, h, c, q, p)
            for r in mine:
                chain = summed[id(r)]

                def source(i, chain=chain, address=address, loaded=loaded, at=at):
                    leaf = chain.operands[i]
                    if leaf.kind == "bcast_col":
                        return kept_value(leaf.source)
                    if leaf.kind == "bcast_row":  # a resident row's columns
                        key = id(leaf.source)
                        if key not in loaded:
                            where = cols[key].address(*at)
                            loaded[key] = b.load(where, f"per-column {where}")
                        return loaded[key]
                    if "row" not in loaded:
                        loaded["row"] = b.load(address, f"row {address}")
                    return loaded["row"]

                top, acc, op = chain.fn.result, accs.get(id(r)), r.attrs["op"]
                if acc is not None and top.kind == "mul" and op == "sum":
                    memo = {}
                    x = _value(b, top.args[0], memo, source)
                    m = _value(b, top.args[1], memo, source)
                    accs[id(r)] = b.mad(x, m, acc, "sum += f")
                else:
                    v = _value(b, top, {}, source)
                    accs[id(r)] = v if acc is None else _fold(b, op, acc, v, "sum += f")
        for r in mine:
            acc = _lanes(b, r.attrs["op"], accs[id(r)])
            where = sums[id(r)].address(0, h, 0, q, 0)
            so_far = b.load(where, f"sum so far, group {h} {q}")
            total = _fold(b, r.attrs["op"], so_far, acc, "sum +=")
            b.store(total, where, f"sum so far, group {h} {q}")
        if q % 2:
            pairs.append({id(x) for x in b.code[start:]})
    return pairs


def _chunk_finish(b, done, made, slot, sums):
    """Part finish<L>, per row group: the per-row values level L completes, from the
    sums' slots and the earlier values' slots, stored to their slots."""
    for h, q in GROUPS:
        loaded = {}

        def value_of(node, h=h, q=q, loaded=loaded):
            if id(node) not in loaded:
                where = (sums if node.kind == "reduce" else slot)[id(node)]
                note = f"{'sum' if node.kind == 'reduce' else 'per-row value'} {h} {q}"
                loaded[id(node)] = b.load(where.address(0, h, 0, q, 0), note)
            return loaded[id(node)]

        for p in done:
            if id(p) not in made:  # a sum read by .col() itself
                value = value_of(p) if p.kind == "reduce" else None
            else:
                chain = made[id(p)]
                value = _value(
                    b,
                    chain.fn.result,
                    {},
                    lambda i, chain=chain: value_of(chain.operands[i].source),
                )
            b.store(value, slot[id(p)].address(0, h, 0, q, 0), f"per-row value {h} {q}")
