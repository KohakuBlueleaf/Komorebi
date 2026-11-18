"""The hazard checks of a lowered unit program: a value read after its register is
written again, Dst window reads, the MAD readers, configuration NOPs and the saved
register."""

from komorebi.dsl.tile.unit import (
    OPS,
    PROGRAMMABLE,
    SAVED,
    WORK,
    Check,
    Insn,
    Loop,
    UnitError,
    encode,
    insns,
)

TILE = 64  # Dst address units of a tile in either Dst: 4 faces of 16 rows
# The readers of a MAD-class result that the unit does not hold back (rule
# mad-reader: the ISA's SFPMAD page; SFPGT, SFPLE from tt-metal's ckernel_sfpu_recip.h).
NOT_HELD = ("SFPIADD", "SFPSHFT", "SFPAND", "SFPOR", "SFPSWAP", "SFPGT", "SFPLE")
SHFT2_LANES = (2, 3, 4)  # SFPSHFT2 lane modes: their reads are not seen at all
STALL_SFPU, WAIT_MATH = 0x100, 0x10
NEG_ONE_BF16 = 0xBF80


def _order(items):
    for item in items:
        if isinstance(item, Loop):
            yield from _order(item.body)
        elif isinstance(item, Insn):
            yield item


def _named_spans(code):
    """{register: [(first, last)]}: where a register the code names ("L<n>") holds
    a value, from each write to its last read before the next write."""
    spans = {}
    for i, insn in enumerate(code):
        for v in insn.reads():
            if isinstance(v, str) and v[0] == "L":
                r = int(v[1:])
                if r < WORK and spans.get(r):
                    spans[r][-1][1] = i
                elif r < WORK:
                    spans[r] = [[i, i]]
        for v in insn.writes():
            if isinstance(v, str) and v[0] == "L" and int(v[1:]) < WORK:
                r = int(v[1:])
                if not (spans.get(r) and spans[r][-1][1] == i):
                    spans.setdefault(r, []).append([i, i])
    return {r: [tuple(s) for s in v] for r, v in spans.items()}


def check_mad_readers(blocks_):
    """mad-reader: an SFPNOP between a multiply-add and a reader the unit does not
    hold back, across each block's wrap too (a loop body's last instruction runs
    before its first). Returns the repairs made."""
    repairs = 0
    for code in blocks_:
        i = 1
        while i < len(code):
            prev, cur = code[i - 1], code[i]
            if OPS[prev.op].kind == "mad" and not_held(cur, set(prev.writes())):
                code.insert(i, Insn("SFPNOP", {}, "hazard mad-reader: MAD result"))
                repairs += 1
            i += 1
        last = code[-1] if code else None
        if (
            last
            and OPS[last.op].kind == "mad"
            and not_held(code[0], set(last.writes()))
        ):
            code.append(Insn("SFPNOP", {}, "hazard mad-reader: MAD result, the wrap"))
            repairs += 1
    return repairs


def check_dst_window(unit, tiles):
    """dst-window: the vector code's loads and stores address Dst tiles below `tiles`,
    those its caller gives a call (a walk of rows covers the one tile its address
    names: 4 faces of 16 rows). Raises UnitError past them."""
    items = [*unit.tile["math"], *(x for part in unit.parts.values() for x in part)]
    addresses = [
        x.args["dest_reg_addr"]
        for x in insns(items)
        if x.op in ("SFPLOAD", "SFPSTORE", "SFPLOADMACRO")
    ]
    reach = max(addresses, default=-1) // TILE + 1
    if reach > tiles:
        raise UnitError(
            f"{unit.name}: the vector code reaches Dst tile {reach - 1}; a call is "
            f"given tiles 0 .. {tiles - 1}"
        )
    unit.checks.append(
        Check("dst-window", "Dst accesses inside a call's tiles", f"{reach} of {tiles}")
    )


def not_held(insn, written):
    """True when `insn` reads (or half-writes) a register of `written` the cycle after
    a multiply-add and the unit does not stall it (the ISA's SFPMAD page)."""
    reads = set(insn.reads())
    if insn.op in NOT_HELD or insn.op == "SFPLOADI":
        return bool(reads & written)
    if insn.op == "SFPSHFT2":
        return insn.args["instr_mod1"] in SHFT2_LANES and bool(reads & written)
    if insn.op in ("SFPSETSGN", "SFPSETEXP", "SFPSETMAN"):
        return not insn.args["instr_mod1"] & 1 and insn.args["lreg_dest"] in written
    if insn.op == "SFPCONFIG":
        return bool({"L0", 0} & written)
    return False


def check_config_nop(code):
    """config-nop: an SFPCONFIG writing LReg 11 to 14 is followed by an SFPNOP."""
    repairs = 0
    for i in range(len(code) - 1, -1, -1):
        insn = code[i]
        if (
            isinstance(insn, Insn)
            and insn.op == "SFPCONFIG"
            and 11 <= insn.args["config_dest"] <= 14
        ):
            after = code[i + 1] if i + 1 < len(code) else None
            if not (isinstance(after, Insn) and after.op == "SFPNOP"):
                code.insert(
                    i + 1, Insn("SFPNOP", {}, "hazard config-nop: after SFPCONFIG")
                )
                repairs += 1
    return repairs


def _check_saved(flat):
    """LReg 11 in the body's code `flat`: set before its first read, restored to -1.0
    (SFPLOADI of 0xBF80 into LReg 0, then SFPCONFIG to 11) after its last. Returns the
    check's text."""
    sets = [
        i
        for i, x in enumerate(flat)
        if x.op == "SFPCONFIG" and x.args["config_dest"] == SAVED
    ]
    if not sets:
        return "LReg 11 not written"
    reads = [i for i, x in enumerate(flat) if SAVED in x.reads()]
    last = sets[-1]
    loads = [x for x in flat[:last] if x.op == "SFPLOADI" and x.args["lreg_ind"] == 0]
    restore = (
        loads
        and (loads[-1].args["instr_mod0"], loads[-1].args["imm16"]) == (0, NEG_ONE_BF16)
        and not any(0 in x.writes() for x in flat[flat.index(loads[-1]) + 1 : last])
    )
    if not restore or (reads and reads[-1] > last):
        raise UnitError("LReg 11 is read after its restore, or not restored to -1.0")
    if reads and len(sets) < 2 or (reads and reads[0] < sets[0]):
        raise UnitError("LReg 11 is read before the body sets it")
    return f"LReg 11 set by the body, read {len(reads)} times, restored to -1.0 after"


def _checks(unit, row, plan=None):
    """The hazard checks of the math CPU's program (`row`: the per-row loop body;
    `plan`: its load-macro plan, or None)."""
    math_init, math_tile = unit.init["math"], unit.tile["math"]
    n = check_mad_readers([row])
    unit.checks.append(
        Check(
            "mad-reader",
            "SFPNOP between a MAD and a reader not held back",
            f"repaired: {n} SFPNOP" if n else "ok",
        )
    )
    n = check_config_nop(math_init) + check_config_nop(row)
    unit.checks.append(
        Check(
            "config-nop",
            "SFPNOP after an SFPCONFIG of LReg 11 to 14",
            f"repaired: {n} SFPNOP" if n else "ok",
        )
    )
    saved = _check_saved(list(_order(math_tile)))
    written = sorted(
        x.args["config_dest"]
        for x in insns(math_init)
        if x.op == "SFPCONFIG" and 11 <= x.args["config_dest"] <= 14
    )
    read = sorted({r for x in insns(math_tile) for r in x.reads() if r in PROGRAMMABLE})
    if set(read) - set(written):
        raise UnitError(f"LReg {read} read but not set by the setup")
    if SAVED in written:
        raise UnitError(
            "LReg 11 is written by the setup: it is set and restored by the body"
        )
    unit.clobbers = [f"LReg{r}" for r in written]
    unit.checks.append(
        Check(
            "constants",
            "constant registers set before use, LReg 11 restored to -1.0",
            f"ok: LReg {written} set by the setup and declared clobbered; " + saved,
        )
    )
    depth = 0
    for insn in row:
        if insn.sets_flags():
            depth = 0 if insn.op == "SFPENCC" else depth + 1
    if depth:
        raise UnitError("a condition is open at the end of the row code")
    unit.checks.append(
        Check(
            "predication",
            "lane predication restored (every lane on) at the end of the code",
            "ok",
        )
    )
    flat = list(_order(math_tile))
    first = next(i for i, x in enumerate(flat) if x.op in ("SFPLOAD", "SFPLOADMACRO"))
    stalled = any(
        x.op == "STALLWAIT" and x.args["stall_res"] & STALL_SFPU for x in flat[:first]
    )
    if not stalled:
        stall = {"stall_res": STALL_SFPU, "wait_res": WAIT_MATH}
        math_tile.insert(0, Insn("STALLWAIT", stall, "SFPU waits for the matrix unit"))
    unit.checks.append(
        Check(
            "dst-load",
            "matrix-unit writes to Dst finished before the first SFPLOAD",
            "ok: STALLWAIT(SFPU on MATH) first" if stalled else "repaired: STALLWAIT",
        )
    )
    built = 0
    for insn in insns(math_init + math_tile):
        encode(insn.op, insn.args)
        built += bool(insn.runtime)
    unit.checks.append(
        Check(
            "immediates",
            "every immediate and register field inside its width",
            (
                f"ok: every word encoded; {built} immediate(s) of run-time scalars "
                "built on the core (TT_SFPLOADI of a 16-bit half)"
                if built
                else "ok: every word encoded (all immediates are compile-time)"
            ),
        )
    )
