"""The row form: a row of `width` whole tiles in Dst, one exact sum along the row
(`reduce_sum(f(row), exact=True)`), a per-row value computed from it, and the stored
row computed from the row, that value broadcast with `.col()` and a per-column value
broadcast with `.row()`. Layout rule: a per-row value is lanes of one tile."""

from dataclasses import dataclass

from komorebi.dsl.tile import chain as chains
from komorebi.dsl.tile.ir import BF16, F32, PlanError, fail
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
    Pass,
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


def row_program(ir, width, rows_tile, cols_tile, dest="fp32"):
    """The unit program of row kernel `ir` over a row of `width` tiles in Dst tiles
    0 .. width - 1 (the result stored over them), its per-row value in Dst tile
    `rows_tile` and its per-column operand read from Dst tile `cols_tile`."""
    if dest != "fp32":
        raise PlanError("komorebi.dsl.tile: the unit program is built for fp32 Dst")
    if not 1 <= width <= 2:
        fail("the row form takes rows of 1 or 2 tiles (a per-column tile holds 2)")
    if not {t.dtype for t in ir.tensors} <= {BF16, F32}:
        fail(f"{ir.name}: tensors are bf16 or fp32")
    if len(ir.stores) != 1:
        fail(f"{ir.name}: the row form takes one store")
    store = ir.stores[0]
    shapes = chains._Shapes(True)
    nodes = ir.nodes()
    for n in nodes:
        shapes(n)
    reduces = [n for n in nodes if n.kind == "reduce"]
    if len(reduces) != 1:
        fail(f"{ir.name}: the row form takes one reduction")
    red = reduces[0]
    if red.attrs["op"] != "sum" or not red.attrs["exact"] or red.attrs["dim"] != -1:
        fail("the row form's reduction is reduce_sum(..., exact=True) along the row")
    cols = [n for n in nodes if n.kind == "col"]
    if len(cols) != 1 or shapes(cols[0].args[0]) != "col":
        fail("the per-row value meets the row through one .col()")
    stat = cols[0].args[0]
    rows_ = [n for n in nodes if n.kind == "load" and n.attrs["view"].kind == "row"]
    if len({id(n.attrs["view"].tensor) for n in rows_}) != 1:
        fail("the row form reads one row tensor (loaded once)")

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
    unit.not_checked = [r for r in NOT_CHECKED if "load-macro" not in r]
    return unit


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
    """Schedule: reorders code[start:end] (straight line; a run under lane flags, from
    the flag-setting instruction to its SFPENCC, moves as one) so that an
    instruction reading the MAD result issued just before it waits behind another
    ready one. Returns how many instructions moved."""
    block = code[start:end]
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
    outside = {v for x in block for v in x.reads() if _work(v) and v not in written}
    outside |= {v for x in code[:start] for v in x.writes() if v in later}
    left = {}
    for x in block:
        for v in x.reads():
            left[v] = left.get(v, 0) + 1
    live = {v for v in outside if left.get(v) or v in later}
    order, issued, previous = [], set(), None
    while len(order) < len(units):
        ready = [i for i in range(len(units)) if i not in issued and preds[i] <= issued]
        ready.sort(key=lambda i: (-height[i], i))
        after = {i: len(_unit_after(units[i], live, left, later)) for i in ready}
        fits = [i for i in ready if after[i] <= WORK]
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
    code[start:end] = [x for i in order for x in units[i]]
    if _pressure(code) > WORK:
        code[start:end] = block
        return 0
    return sum(1 for k, i in enumerate(order) if k != i)


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
    """The most work registers the allocator needs at one instruction of `code`."""
    first, last = {}, {}
    for i, insn in enumerate(code):
        for v in insn.writes():
            if _work(v):
                first.setdefault(v, i)
        for v in insn.reads():
            if _work(v):
                last[v] = i
    starts, ends = [0] * (len(code) + 1), [0] * (len(code) + 1)
    for v, i in first.items():
        starts[i] += 1
        ends[max(i, last.get(v, i))] += 1
    most = live = 0
    for i in range(len(code)):
        live += starts[i]
        most = max(most, live)
        live -= ends[i]
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
