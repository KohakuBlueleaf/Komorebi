"""The step form: a row of `width` tiles and one state tile in Dst. Exact reductions
along the row come in levels (a reduction reads per-row values of the levels before
it); per-row values are lanes of the state tile (layout rule: slot k of row group
(h, q) at 16 h + 4 q + SLOT[k]); a per-row value reduced to one word (tile.word_max)
takes the last slot; the row is stored in place. A pass over the row per level, row
group by row group, so a group's per-row values feed its own next pass."""

import copy

from komorebi.dsl.tile import chain as chains
from komorebi.dsl.tile.ir import BF16, F32, PlanError, fail
from komorebi.dsl.tile.lower import (
    ADDR_MOD,
    NOT_CHECKED,
    STALL_SFPU,
    TILE,
    WAIT_MATH,
    Body,
    _checks,
    _templates,
    _value,
    allocate,
    check_mad_readers,
    constant_registers,
    drop_copies,
    fold_negation,
    fuse_mad,
    hoist,
)
from komorebi.dsl.tile.rows import (
    GROUPS,
    ROTATE,
    Layout,
    _check_broadcast,
    _check_lanes,
    _repairs,
    _vectors,
    interleave,
)
from komorebi.dsl.tile.unit import (
    OPS,
    PROGRAMMABLE,
    Call,
    Insn,
    Pass,
    Sync,
    UnitProgram,
)

SLOT = (0, 2, 32, 34)  # slot k: even / odd columns of faces 0-1, then of faces 2-3
WORD_SLOT = 3
# The word's place in the packed fp32 state tile: lane 0 of slot 3, row group (0, 0)
# (face 2, row 0, column 1); every lane of that vector holds it.
WORD = 2 * 256 + 1
MAX_SWAP = 1  # SFPSWAP mod1: the larger into VC, the smaller into VD


def step_program(ir, width, state_tile, dest="fp32"):
    """The unit program of step kernel `ir` over a row of `width` tiles in Dst tiles
    0 .. width - 1 (stored over them) and its state tile in Dst tile `state_tile`
    (read and written in place; see the module text for its layout)."""
    if dest != "fp32":
        raise PlanError("komorebi.dsl.tile: the unit program is built for fp32 Dst")
    if not 1 <= width <= 3 or not width <= state_tile <= 3:
        fail("the step form takes 1 to 3 row tiles and the state in a tile after them")
    if not {t.dtype for t in ir.tensors} <= {BF16, F32}:
        fail(f"{ir.name}: tensors are bf16 or fp32")
    shapes = chains._Shapes(True)
    nodes = ir.nodes()
    for n in nodes:
        shapes(n)
    rows_ = [s for s in ir.stores if shapes(s.value) == "row"]
    states = [s for s in ir.stores if s.value.kind == "state"]
    if len(rows_) != 1 or len(states) != 1 or len(ir.stores) != 2:
        fail(f"{ir.name}: the step form takes one row store and one state store")
    row_store, state = rows_[0].value, states[0].value
    loads = [n for n in nodes if n.kind == "load" and n.attrs["view"].kind == "row"]
    if len({id(n.attrs["view"].tensor) for n in loads}) != 1:
        fail("the step form reads one row tensor")
    slots = [n for n in nodes if n.kind == "slot"]
    if len({id(n.args[0]) for n in slots}) > 1:
        fail("the step form reads one state tile")
    reduces = [n for n in nodes if n.kind == "reduce"]
    for r in reduces:
        if r.attrs["op"] not in ("sum", "max") or not r.attrs["exact"]:
            fail("the step form's reductions are exact=True sums or maxima")
    stored = list(state.args[: len(state.args) - state.attrs["word"]])
    word = state.args[-1] if state.attrs["word"] else None
    level = _levels(nodes)
    held = _held(nodes, shapes, stored, word, level)
    for s in slots:
        k = s.attrs["slot"]
        written = level[id(stored[k])] if k < len(stored) else None
        if written is not None and any(
            level[id(u)] > written for u in _users(s, nodes)
        ):
            fail(f"state slot {k} is read after the level that writes it")
    compiler = chains._Compiler(ir.name, shapes, {})
    stops = {id(n) for n in held} | {id(n) for n in slots}

    def leaf_of(node, offset, kind):
        if offset is not None:
            return None
        if node.kind == "load" and node.attrs["view"].kind == "row":
            return chains.Leaf(node, "copy", None)
        if id(node) in stops or node.kind == "reduce":
            return chains.Leaf(node, "bcast_col", None)
        return None

    def chain_of(root):
        c = compiler.chain(root, leaf_of, stops=stops - {id(root)}, limit=8)
        if c.values or c.prev is not None:
            fail(f"{ir.name}: the step form takes no runtime scalars and one stage")
        return c

    unit = UnitProgram(ir.name, dest)
    unit.operands = [loads[0].attrs["view"].tensor.name]
    plan = _Plan(width, state_tile, level, stored, word, held, slots)
    plan.reductions = []
    for r in reduces:
        operand, scale = _scale_out(r)
        plan.reductions.append((r, operand, chain_of(operand), scale))
    plan.store = (row_store, chain_of(row_store))
    plan.per_row = [(n, chain_of(n)) for n in held]
    unit.passes.append(Pass("algebra", "templates", _templates(compiler.templates)))
    scaled = sum(1 for *_, s in plan.reductions if s is not None)
    unit.passes.append(
        Pass(
            "algebra",
            "max_scale",
            (
                [
                    f"{scaled} positive constant factor(s) out of a maximum "
                    "(rounding is monotonic: exact)"
                ]
                if scaled
                else []
            ),
        )
    )
    # Constants kept in a work register for a group's level, most loads first, while
    # the values still fit the 8 work registers (each trial builds the body anew).
    best = _attempt(ir.name, dest, plan, set())
    if best is None:
        fail(f"{ir.name}: the step's values do not fit the 8 work registers")
    kept, loads = set(), best[0].loaded
    for bits in sorted(loads, key=lambda b: -loads[b]):
        got = _attempt(ir.name, dest, plan, kept | {bits})
        if got is not None and _estimate(got[0].code) < _estimate(best[0].code):
            kept.add(bits)
            best = got
    body, probe, passes, setup = best
    cached = len(kept)
    note = f"{cached} constant(s) loaded once a group and level, the rest once a read"
    unit.passes += passes
    unit.passes.append(
        Pass(
            "layout",
            "lanes",
            [
                f"{len(stored)} per-row values in slots of tile {state_tile}"
                + (", the word in the last" if word is not None else ""),
                note,
            ],
        )
    )
    unit.checks += probe.checks
    if word is not None:
        body.code += _word_tail(TILE * state_tile + SLOT[WORD_SLOT])
    unit.constants = {PROGRAMMABLE[i]: v for i, v in enumerate(body.prog.values())}
    unit.init["math"] = [
        Insn(
            "SFPCONFIG",
            {"imm16_math": 0, "config_dest": 15, "instr_mod1": 1},
            "lane configuration",
        ),
        *setup,
    ]
    copies = [f"copy_tile(x, {t}, dst + {t})" for t in range(width)]
    copies.append(f"copy_tile(state, 0, dst + {state_tile})")
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
    packs = [f"pack_tile(dst + {t}, out)" for t in range(width)]
    packs.append(f"pack_tile(dst + {state_tile}, state_out)")
    unit.tile["pack"] = [
        Sync("pack", "math", "tile_regs_wait()", "Dst from the math CPU"),
        *[Call(c, "a tile of the row, then the state") for c in packs],
        Sync("pack", "math", "tile_regs_release()", "frees the Dst half"),
    ]
    _checks(unit, body.code)
    unit.tile["math"][-1 - len(body.code) + _repairs(unit) : -1] = body.code
    unit.not_checked = [r for r in NOT_CHECKED if "load-macro" not in r]
    return unit


def _attempt(name, dest, plan, cacheable):
    """(body, checks, passes, setup) of the body with constants `cacheable` loaded
    once a group and level, or None when its values do not fit the registers."""
    body, probe = Body(), UnitProgram(name, dest)
    body.cacheable, body.pairs = set(cacheable), []
    _build(body, plan)
    passes = [
        Pass("algebra", "fold_negation", _tally(fold_negation(body.code))),
        Pass("algebra", "fuse_mad", _tally(fuse_mad(body.code))),
        Pass("algebra", "drop_copies", drop_copies(body.code)),
    ]
    moved = 0
    for pair in body.pairs:
        at = [i for i, x in enumerate(body.code) if id(x) in pair]
        moved += interleave(body.code, at[0], at[-1] + 1)
    done = [f"{moved} instructions moved in {len(body.pairs)} pairs of row vectors"]
    passes.append(Pass("schedule", "interleave", done))
    layouts = plan.layouts()
    _check_lanes(probe, body.code, layouts["slot0"])
    _check_broadcast(probe, body.code, layouts, plan.width)
    done = constant_registers(body.code, body.prog)
    passes.append(Pass("layout", "constant_registers", done))
    setup = []
    passes.append(Pass("schedule", "hoist", hoist(body.code, setup)))
    try:
        passes.append(Pass("layout", "allocate", allocate(body.code, body.prog)))
    except ValueError:
        return None
    allocate(setup, body.prog)
    return body, probe, passes, setup


def _estimate(code):
    """Cycles of `code` (allocated) once the mad-reader check has repaired it."""
    trial = copy.deepcopy(code)
    check_mad_readers([trial])
    unit = UnitProgram("estimate", "fp32")
    unit.tile["math"] = trial
    return unit.cost("math")[1]


def _tally(done):
    """A pass's report with equal lines counted once."""
    counts = {}
    for line in done:
        counts[line] = counts.get(line, 0) + 1
    return [f"{line} (x {n})" if n > 1 else line for line, n in counts.items()]


class _Plan:
    """What the builder needs: the levels, the per-row values held (their chains),
    the reductions (node, chain of the operand, factor taken out) and the store."""

    def __init__(self, width, state_tile, level, stored, word, held, slots):
        self.width, self.state_tile, self.level = width, state_tile, level
        self.stored, self.word, self.held, self.slots = stored, word, held, slots
        self.reductions, self.store, self.per_row = [], None, []

    def levels(self):
        """The levels the body walks: 0 to the highest reduction, per-row value or
        row pass (a row value of level l is stored in pass l + 1)."""
        top = [self.level[id(r)] for r, *_ in self.reductions]
        top += [self.level[id(n)] for n, _ in self.per_row]
        top.append(self.level[id(self.store[0])] + 1)
        return range(max(top) + 1)

    def slot_of(self, node):
        """The slot a per-row value lives in: its state slot, else None."""
        for k, v in enumerate(self.stored):
            if v is node:
                return k
        return None

    def per_row_of(self, lv):
        """The per-row values of level `lv` and their chains, in order."""
        return [(n, c) for n, c in self.per_row if self.level[id(n)] == lv]

    def address(self, k, h, q):
        return Layout("rows", self.state_tile, SLOT[k]).address(0, h, 0, q, 0)

    def layouts(self):
        out = {"copy": Layout("tile", 0)}
        for k in range(len(SLOT)):
            out[f"slot{k}"] = Layout("rows", self.state_tile, SLOT[k])
        return out


def _levels(nodes):
    """Level of each node: 0 for loads and constants, a reduction one more than its
    operand, any other node the highest of its arguments."""
    level = {}
    for n in nodes:
        if n.kind == "reduce":
            level[id(n)] = 1 + level[id(n.args[0])]
        else:
            level[id(n)] = max((level[id(a)] for a in n.args), default=0)
    return level


def _users(node, nodes):
    return [u for u in nodes if any(a is node for a in u.args)]


def _held(nodes, shapes, stored, word, level):
    """The per-row values computed once a group and kept: stored ones, the word's,
    those broadcast into the row and those read by two or more. A later level reads
    a kept value back from its state slot."""
    out = []
    for n in nodes:
        if shapes(n) != "col" or n.kind in ("slot", "reduce"):
            continue
        users = _users(n, nodes)
        if (
            any(n is v for v in stored)
            or (word is not None and n is word.args[0])
            or any(u.kind == "col" for u in users)
            or len(users) > 1
        ):
            out.append(n)
    return out


def _scale_out(r):
    """(operand, factor): a maximum of x c with c a positive constant is the maximum
    of x times c; else (the operand, None)."""
    x = r.args[0]
    if r.attrs["op"] != "max" or x.kind != "mul":
        return x, None
    for i in (0, 1):
        c = x.args[i]
        if c.kind == "const" and c.attrs["value"] > 0:
            return x.args[1 - i], c.attrs["value"]
    return x, None


def _build(b, plan):
    """The body: per row group, per level, the row's pass then the per-row values.
    A per-row value of an earlier level is read back from its state slot; the
    state slots a level reads are loaded before it writes any."""
    for h, q in GROUPS:
        for lv in plan.levels():
            b.cache = {}
            values = {}  # id(node) -> this level's per-row values, reductions

            def per_row(node, h=h, q=q, values=values):
                if id(node) not in values:
                    k = (
                        node.attrs["slot"]
                        if node.kind == "slot"
                        else plan.slot_of(node)
                    )
                    if k is None:
                        fail("a per-row value read a level later has no state slot")
                    note = f"slot {k} ({h} {q})"
                    values[id(node)] = b.load(plan.address(k, h, q), note)
                return values[id(node)]

            _row_pass(b, plan, lv, h, q, per_row, values)
            # the row pass writes no slot: the per-row values' reads load here
            for _, c in plan.per_row_of(lv):
                for leaf in c.operands:
                    if leaf.kind != "copy" and leaf.source.kind == "slot":
                        per_row(leaf.source)
            for node, c in plan.per_row_of(lv):
                v = _value(b, c.fn.result, {}, _source(c, per_row, None))
                values[id(node)] = v
                k = plan.slot_of(node)
                if k is not None:
                    b.store(v, plan.address(k, h, q), f"slot {k} ({h} {q})")
                if plan.word is not None and node is plan.word.args[0]:
                    _word(b, plan, v, (h, q) == GROUPS[0])
    b.cache = None


def _source(c, per_row, vector):
    """source(index) of chain `c`: a row vector (`vector(leaf)`) or a per-row value."""

    def source(i):
        leaf = c.operands[i]
        if leaf.kind == "copy":
            return vector(leaf)
        return per_row(leaf.source)

    return source


def _row_pass(b, plan, lv, h, q, per_row, values):
    """Level `lv`'s pass over the group's row vectors: the reductions of this level
    (lane by lane, then the 8 lanes of each lane row by rotations) and the row's
    store when its value is of the level before."""
    mine = [x for x in plan.reductions if plan.level[id(x[0])] == lv]
    node, chain = plan.store
    storing = plan.level[id(node)] + 1 == lv
    if not mine and not storing:
        return
    accs, start = {}, len(b.code)
    for n, (t, c, p) in enumerate(_vectors(plan.width)):
        if n % 2 == 0:
            start = len(b.code)
        address = Layout("tile", 0).address(t, h, c, q, p)
        loaded, results = {}, {}

        def vector(leaf, address=address, loaded=loaded, at=(t, h, c, q, p)):
            if "v" not in loaded:
                loaded["v"] = b.load(address, f"row {at}")
            return loaded["v"]

        def result(root, ch, results=results, vector=vector):
            if id(root) not in results:
                source = _source(ch, per_row, vector)
                results[id(root)] = _value(b, ch.fn.result, {}, source)
            return results[id(root)]

        for r, operand, ch, _ in mine:
            v = result(operand, ch)
            if id(r) not in accs:
                accs[id(r)] = _lane_copy(b, v)
            else:
                accs[id(r)] = _combine(b, r.attrs["op"], accs[id(r)], v)
        if storing:
            b.store(result(node, chain), address, f"row {(t, h, c, q, p)}")
        if n % 2:
            b.pairs.append({id(x) for x in b.code[start:]})
    for r, _, _, scale in mine:
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
            acc = _combine(b, r.attrs["op"], acc, turned)
        if scale is not None:
            acc = b.mul(acc, b.const(scale), "the factor taken out of the maximum")
        values[id(r)] = acc


def _lane_copy(b, v):
    """A value the reduction may update in place."""
    return b.op("SFPMOV", "reduction: first vector", lreg_c=v)


def _combine(b, op, acc, v):
    """acc + v, or the larger of acc and v into acc (in place, SFPSWAP)."""
    if op == "sum":
        return b.add(acc, v, "sum")
    other = b.op("SFPMOV", "max: the other", lreg_c=v)
    b.op("SFPSWAP", "max", lreg_src_c=acc, lreg_dest=other, instr_mod1=MAX_SWAP)
    return acc


def _word(b, plan, v, first):
    """The word's running maximum over the groups, kept in the word slot of group
    (0, 0) (the tail reduces its 32 lanes)."""
    address = TILE * plan.state_tile + SLOT[WORD_SLOT]
    if first:
        b.store(v, address, "word: group (0, 0)")
        return
    acc = b.load(address, "word so far")
    other = b.op("SFPMOV", "word: this group", lreg_c=v)
    b.op("SFPSWAP", "word: max", lreg_src_c=acc, lreg_dest=other, instr_mod1=MAX_SWAP)
    b.store(acc, address, "word so far")


def _word_tail(address):
    """The word slot's 32 lanes to their maximum, in every lane (after register
    allocation: SFPTRANSP works on LReg 0-3, nothing else is live here)."""
    out = [_insn("SFPLOAD", lreg_ind=0, dest_reg_addr=address, note="the word's lanes")]
    out += [_insn("SFPMOV", lreg_c=0, lreg_dest=r, note="a copy") for r in (1, 2, 3)]
    out.append(_insn("SFPTRANSP", note="lane rows across LReg 0-3"))
    for r in (1, 2, 3):
        out.append(
            _insn(
                "SFPSWAP",
                lreg_src_c=0,
                lreg_dest=r,
                instr_mod1=MAX_SWAP,
                note="max over the lane rows",
            )
        )
    for k in (1, 2, 4):
        src = 0
        for _ in range(k):
            out.append(
                _insn(
                    "SFPSHFT2",
                    lreg_src_c=src,
                    lreg_dest=1,
                    instr_mod1=ROTATE,
                    note=f"lanes turned (step {k})",
                )
            )
            src = 1
        out.append(
            _insn(
                "SFPSWAP",
                lreg_src_c=0,
                lreg_dest=1,
                instr_mod1=MAX_SWAP,
                note="max over the lanes",
            )
        )
    out.append(_insn("SFPSTORE", lreg_ind=0, dest_reg_addr=address, note="the word"))
    return out


def _insn(op, note="", **args):
    for arg, _ in OPS[op].args:
        args.setdefault(arg, ADDR_MOD if arg == "sfpu_addr_mode" else 0)
    return Insn(op, args, note)
