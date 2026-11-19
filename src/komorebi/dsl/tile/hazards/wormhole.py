"""Wormhole's forms of the words it would run otherwise (chips.wormhole_tile_refusal):
`forms(unit)` gives each such Insn of the unit's math CPU its `wormhole` words, which
compute the same there; emit writes them under ARCH_WORMHOLE, unit.wormhole_run walks
them. A word without a form stays, and a load on Wormhole refuses it by name."""

from komorebi.device import chips
from komorebi.dsl.tile.lower import CC_RESET
from komorebi.dsl.tile.unit import (
    FIXED,
    IADD_NO_CC,
    IADD_SUB,
    SHIFT_IMM_VC,
    Insn,
    Loop,
    Product,
    Sync,
)

WORK = range(7)  # LRegs a copy may take (7: Blackhole's indirect SrcA, Dst register)
COMPSIGN = 1  # SFPMOV instr_mod1: the copy's sign complemented
SHIFT_IMM = 1  # SFPSHFT instr_mod1 on Wormhole: lreg_dest shifted by the immediate
# NEGATE_VA, NEGATE_VC: the fields a copy may negate (the product's sign: either)
NEGATED = {1: ("lreg_src_a", "lreg_src_b"), 2: ("lreg_src_c",)}
SOURCES = ("lreg_src_a", "lreg_src_b", "lreg_src_c")
# end a scan for a dead register (a stack of flags the scan does not follow)
UNKNOWN = ("SFPLOADMACRO", "SFPCONFIG", "SFPPUSHC", "SFPPOPC", "SFPCOMPC")
ZERO = FIXED["zero"]
LT0, NE0, GTE0 = 0, 2, 4  # SFPSETCC instr_mod1: the integer view < 0, != 0, >= 0
# SFPARECIP's estimate on Wormhole: p(m) ~ -1 / 2m on [1, 2) (cubic, bf16 coefficients
# for SFPMULI / SFPADDI, highest first; relative error 2.5e-3, the LUT's 5.6e-3)
RECIP_P = (0x3DE2, 0xBF2A, 0x3FBC, 0xBFB5)  # 0.1104, -0.6641, 1.4688, -1.4141
INT_ABS, INT_SM = 2, 3  # Blackhole's SFPCAST instr_mod1: integer abs, to sign-magnitude


def _words(items):
    for item in items:
        if isinstance(item, Loop):
            yield from _words(item.body)
        elif isinstance(item, Product):
            yield from (*item.insns, *item.wormhole)
        elif isinstance(item, Insn):
            yield item


def forms(unit):
    """Each word of `unit`'s math functions that Wormhole runs otherwise given its
    form there (once); a copy goes to the word's own destination, an LReg 0 .. 6 no
    word of the unit names, or one dead after the word wherever it runs (a recorded
    word: after each play)."""
    functions = [unit.init["math"], unit.tile["math"], *unit.parts.values()]
    words = [x for function in functions for x in _words(function)]
    named = {r for x in words for r in (*x.reads(), *x.writes()) if isinstance(r, int)}
    free = [r for r in WORK if r not in named]
    places = {}  # id: (the word, the registers dead after it)
    for function in functions:
        for block in _blocks(function, {}):
            for i, x in enumerate(block):
                if x is None or x.wormhole:
                    continue
                dead = {r for r in WORK if _dead(r, block[i + 1 :], x.sets_flags())}
                places[id(x)] = (x, dead & places.get(id(x), (x, dead))[1])
    for x, dead in places.values():
        x.wormhole = tuple(_form(x, [*free, *sorted(dead)]))


def _blocks(items, recorded):
    """The straight-line blocks of a loop nest, each its words in run order (a replay
    as its recorded words; None where a loop or a call comes between)."""
    block, i = [], 0
    while i < len(items):
        item = items[i]
        if isinstance(item, Loop):
            block.append(None)
            yield from _blocks(item.body, recorded)
        elif isinstance(item, Insn) and item.op == "REPLAY":
            start, count = item.args["start_idx"], item.args["len"]
            if item.args["load_mode"]:
                recorded[start] = items[i + 1 : i + 1 + count]
            else:
                block += recorded.get(start, [None])
        elif isinstance(item, Insn):
            block.append(item)
        elif not isinstance(item, (Product, Sync)):
            block.append(None)
        i += 1
    yield block


def _dead(register, after, masked=False):
    """`register` not read in `after` (a block's rest) before a write to every lane a
    copy took: one with the flags unchanged since the copy, or one after an SFPENCC
    that turns every lane on (`masked`: the flags changed by the word itself)."""
    for x in after:
        if x is None or x.op in UNKNOWN or register in x.reads():
            return False
        every = (x.args.get("imm12_math"), x.args.get("instr_mod1")) == CC_RESET
        if x.op == "SFPENCC" and every:
            masked = False
        elif x.sets_flags():
            masked = True
        elif register in x.writes() and not masked:
            return True
    return False


def _move(source, target, mod1, note=""):
    args = {"imm12_math": 0, "lreg_c": source, "lreg_dest": target, "instr_mod1": mod1}
    return Insn("SFPMOV", args, note)


def _form(x, free):
    """Wormhole's words for `x`, or none (the same word, or no form: a load macro's
    template store, which runs nothing and stores this one word)."""
    if x.args.get("lreg_dest", 0) >= chips.TEMPLATE_VD and x.op not in UNKNOWN:
        return []
    mod1 = x.args.get("instr_mod1", 0)
    if x.op == "SFPSHFT" and mod1 == SHIFT_IMM_VC:
        c, d = x.args["lreg_c"], x.args["lreg_dest"]
        shift = {**x.args, "lreg_c": 0, "instr_mod1": SHIFT_IMM}
        copy = [] if c == d else [_move(c, d, 0, "Wormhole: the shifted value")]
        return [*copy, Insn("SFPSHFT", shift, x.note, x.uses)]
    if x.op in ("SFPMAD", "SFPADD", "SFPMUL") and mod1 & 3:
        return _negated(x, free)
    if x.op == "SFPCAST" and mod1 == INT_ABS:
        args = {"imm12_math": 0, "lreg_c": x.args["lreg_src_c"]}
        return [
            Insn("SFPABS", {**args, "lreg_dest": x.args["lreg_dest"], "instr_mod1": 0})
        ]
    if x.op == "SFPCAST" and mod1 == INT_SM:
        return _sign_magnitude(x, free)
    if x.op == "SFPGT" and mod1 == 1:
        return _less(x, free)
    if x.op == "SFPARECIP" and mod1 == 0:
        return _recip(x, free)
    return []


def _word(op, c, d, mod1=0, imm=0, note=""):
    return Insn(
        op, {"imm12_math": imm, "lreg_c": c, "lreg_dest": d, "instr_mod1": mod1}, note
    )


def _less(x, free):
    """SFPGT's flags, VC < VD as sign-magnitude integers (FP32's total order): against
    0 the sign (SFPSETCC); else SFPSETCC of r < 0, r made without the flags (VC - VD,
    negated where VC < 0, VC where the signs differ: r ^ ((VC ^ VD) & (r ^ VC))), or
    with one register, by the lanes of the two signs (the flags kept on their stack)."""
    c, d = x.args["lreg_c"], x.args["lreg_dest"]
    if d == ZERO:
        return [_word("SFPSETCC", c, 0, LT0, note=x.note)]
    if c == ZERO:  # 0 < VD: VD >= 0 and VD != 0 (the second in the first's lanes)
        return [
            _word("SFPSETCC", d, 0, GTE0, note=x.note),
            _word("SFPSETCC", d, 0, NE0),
        ]
    a, b = _two(free, c, d)
    if b is None:
        return _less_by_lanes(x, next((r for r in free if r not in (c, d)), None))
    sub, add = IADD_SUB | IADD_NO_CC, IADD_NO_CC
    return [
        _move(c, a, 0, "Wormhole: VC's sign"),
        _word("SFPSHFT", 0, a, SHIFT_IMM, (-31) & 0xFFF),
        _word("SFPIADD", ZERO, a, sub, note="m = -sign: 0 or -1"),
        _move(d, b, 0, "Wormhole: VD"),
        _word("SFPIADD", c, b, sub, note="r = VC - VD"),
        _word("SFPIADD", a, b, add),
        _word("SFPXOR", a, b, note="r negated where VC < 0"),
        _move(c, a, 0),
        _word("SFPXOR", d, a, note="VC ^ VD"),
        _word("SFPXOR", c, b),
        _word("SFPAND", b, a),
        _word("SFPXOR", c, b),
        _word("SFPXOR", a, b, note="VC's sign where the signs differ"),
        _word("SFPSETCC", b, 0, LT0, note=x.note),
    ]


def _less_by_lanes(x, a):
    """r in register `a` (dead in every lane) lane by lane, every lane on: VC - VD
    where both are positive, VC where VC >= 0 > VD or VD >= 0 > VC, VD - VC where both
    are negative (the magnitudes' order reversed); then the flags of before (SFPPUSHC,
    SFPPOPC) and SFPSETCC of r < 0 in their lanes, as SFPGT sets them."""
    if a is None:
        return []  # no register: refused on Wormhole
    c, d = x.args["lreg_c"], x.args["lreg_dest"]
    imm, mod1 = CC_RESET
    sub = IADD_SUB | IADD_NO_CC
    return [
        _word("SFPPUSHC", 0, 0, note="Wormhole: the lanes' flags kept"),
        _word("SFPENCC", 0, 0, mod1, imm, "every lane on"),
        _move(d, a, 0, "VD"),
        _word("SFPSETCC", c, 0, GTE0, note="VC >= 0"),
        _word("SFPIADD", c, a, sub, note="r = VC - VD"),
        _word("SFPSETCC", d, 0, LT0, note="VD < 0"),
        _move(c, a, 0, "r = VC (not below)"),
        _word("SFPENCC", 0, 0, mod1, imm, "every lane on"),
        _word("SFPSETCC", c, 0, LT0, note="VC < 0"),
        _move(c, a, 0, "r = VC (below)"),
        _word("SFPSETCC", d, 0, LT0, note="VD < 0"),
        _word("SFPIADD", d, a, sub, note="r = VD - VC"),
        _word("SFPPOPC", 0, 0, note="the lanes' flags of before"),
        _word("SFPSETCC", a, 0, LT0, note=x.note),
    ]


def _two(free, *taken):
    """Two of `free` but `taken`, (None, None) without."""
    pick = [r for r in free if r not in taken][:2]
    return (pick + [None, None])[:2] if len(pick) == 2 else (None, None)


def _recip(x, free):
    """SFPARECIP (sign + a 7-bit estimate of 1 / |x|) as p(m) 2^(128 - e): m = x's
    mantissa in [1, 2), p of RECIP_P, 2^(128 - e) = ~x with its mantissa cleared (x's
    sign flipped, p < 0: the product's is x's). As the LUT, inf below 2^-126 and 0 at
    inf and NaN; 1 / x up to 2^127 (the LUT: 0 from 2^126)."""
    c, d = x.args["lreg_c"], x.args["lreg_dest"]
    m, t = (
        (d, next((r for r in free if r not in (c, d)), None))
        if c != d
        else _two(free, c)
    )
    if t is None:
        return []  # no registers: refused on Wormhole
    out = [
        _word("SFPSETEXP", c, m, 1, 127, "Wormhole: x's mantissa in [1, 2)"),
        _word("SFPSETSGN", m, m, 1, 0, "m = |that|"),
        _move(m, t, 0, "p(m)"),
    ]
    for k, coefficient in enumerate(RECIP_P):
        op = "SFPMULI" if k == 0 else "SFPADDI"
        out.append(
            Insn(op, {"imm16_math": coefficient, "lreg_dest": t, "instr_mod1": 0})
        )
        if 0 < k < len(RECIP_P) - 1:
            out.append(Insn("SFPMUL", _mad(t, m, ZERO, t)))
        if k == len(RECIP_P) - 2:
            out.append(_word("SFPNOT", c, m, note="~x: 2^(128 - e)"))
    out.append(_word("SFPSETMAN", m, m, 1, 0))
    out.append(Insn("SFPMUL", _mad(t, m, ZERO, d), x.note, x.uses))
    return out


def _mad(a, b, c, d):
    args = {"lreg_src_a": a, "lreg_src_b": b, "lreg_src_c": c, "lreg_dest": d}
    return {**args, "instr_mod1": 0}


def _sign_magnitude(x, free):
    """Blackhole's two's complement to sign-magnitude cast: the integer magnitude
    (SFPABS) into a copy register, and onto it the sign of a copy of the value."""
    c, d = x.args["lreg_src_c"], x.args["lreg_dest"]
    t = next((r for r in free if r != d), None)
    if t is None:
        return []  # no register: refused on Wormhole
    copy = [] if c == d else [_move(c, d, 0, "Wormhole: the sign's source")]
    magnitude = {"imm12_math": 0, "lreg_c": c, "lreg_dest": t, "instr_mod1": 0}
    sign = {"imm12_math": 0, "lreg_c": t, "lreg_dest": d, "instr_mod1": 0}
    return [
        *copy,
        Insn("SFPABS", magnitude, "Wormhole: the magnitude"),
        Insn("SFPSETSGN", sign, x.note, x.uses),
    ]


def _negated(x, free):
    """NEGATE_VA, NEGATE_VC as a sign-complemented copy each, read in the field's
    place (the product's or the addend's sign changed exactly as Blackhole's). Without
    the registers, the addend -0 (NEGATE_VC of the constant 0) is read as 0: the one
    result that differs is a product of -0, +0 on Wormhole and -0 on Blackhole."""
    bits = [b for b in NEGATED if x.args["instr_mod1"] & b]
    registers = [*dict.fromkeys((x.args["lreg_dest"], *free))]
    found = _copies(dict(x.args), bits, registers)
    if found is None and 2 in bits and x.args["lreg_src_c"] == ZERO:
        found = _copies(dict(x.args), [b for b in bits if b != 2], registers)
    if found is None:
        return []  # no register: refused on Wormhole
    args, words = found
    args["instr_mod1"] = x.args["instr_mod1"] & ~3
    return [*words, Insn(x.op, args, x.note, x.uses, x.runtime)]


def _copies(args, bits, registers):
    """(the fields after the copies, the copies' SFPMOVs) for the negations `bits`,
    each copy in one of `registers` that the word reads in no other field; or None."""
    if not bits:
        return args, []
    for field in NEGATED[bits[0]]:
        others = {args[f] for f in SOURCES if f != field}
        for r in registers:
            if r in others:
                continue
            note = f"Wormhole: -{field}"
            rest = _copies(
                {**args, field: r}, bits[1:], [x for x in registers if x != r]
            )
            if rest is not None:
                return rest[0], [_move(args[field], r, COMPSIGN, note), *rest[1]]
    return None
