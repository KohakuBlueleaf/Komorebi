"""Wormhole's forms of the words it would run otherwise (chips.wormhole_tile_refusal):
`forms(unit)` gives each such Insn of the unit's math CPU its `wormhole` words, which
compute the same there; emit writes them under ARCH_WORMHOLE, unit.wormhole_run walks
them. A word without a form stays, and a load on Wormhole refuses it by name."""

from komorebi.device import chips
from komorebi.dsl.tile.unit import SHIFT_IMM_VC, Insn, Loop, Product, Sync

WORK = range(7)  # LRegs a copy may take (7: Blackhole's indirect SrcA, Dst register)
COMPSIGN = 1  # SFPMOV instr_mod1: the copy's sign complemented
SHIFT_IMM = 1  # SFPSHFT instr_mod1 on Wormhole: lreg_dest shifted by the immediate
# NEGATE_VA, NEGATE_VC: the fields a copy may negate (the product's sign: either)
NEGATED = {1: ("lreg_src_a", "lreg_src_b"), 2: ("lreg_src_c",)}
SOURCES = ("lreg_src_a", "lreg_src_b", "lreg_src_c")
UNKNOWN = ("SFPLOADMACRO", "SFPCONFIG")  # end a scan for a dead register
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
    word of the unit names, or one the word's block writes next without reading."""
    functions = [unit.init["math"], unit.tile["math"], *unit.parts.values()]
    words = [x for function in functions for x in _words(function)]
    named = {r for x in words for r in (*x.reads(), *x.writes()) if isinstance(r, int)}
    free = [r for r in WORK if r not in named]
    for function in functions:
        for block in _blocks(function, {}):
            for i, x in enumerate(block):
                if x is not None and not x.wormhole:
                    dead = [r for r in WORK if _dead(r, block[i + 1 :])]
                    x.wormhole = tuple(_form(x, [*free, *dead]))


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


def _dead(register, after):
    """`register` written next in `after` (a block's rest) before any read, with the
    lanes' flags unchanged up to that write (it writes every lane the copy did)."""
    for x in after:
        if x is None or x.op in UNKNOWN or x.sets_flags() or register in x.reads():
            return False
        if register in x.writes():
            return True
    return False


def _move(source, target, mod1, note):
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
    return []


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
    place (the product's or the addend's sign changed exactly as Blackhole's)."""
    bits = [b for b in NEGATED if x.args["instr_mod1"] & b]
    found = _copies(dict(x.args), bits, [*dict.fromkeys((x.args["lreg_dest"], *free))])
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
