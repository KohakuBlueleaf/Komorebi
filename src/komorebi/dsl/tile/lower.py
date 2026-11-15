"""A traced tile expression over a stream of tiles to its unit program: the chain
compiler's vector function (`chain.Chain`) is written as vector-unit instructions on
named values, the rules run as passes (algebra, schedule, layout), then the hazard
checks. Built on fp32 Dst: whole-tile operands, constants, +, -, *, sigmoid, rsqrt."""

import copy
import struct

from komorebi.dsl.tile import chain as chains
from komorebi.dsl.tile import macro, sfpu
from komorebi.dsl.tile.ir import BF16, F32, PlanError, fail
from komorebi.dsl.tile.unit import (
    FIXED,
    IADD_IMM,
    IADD_NO_CC,
    IADD_SUB,
    OPS,
    PROGRAMMABLE,
    REPLAY_ENTRIES,
    SHIFT_IMM_VC,
    WORK,
    Call,
    Check,
    Insn,
    Loop,
    Pass,
    Sync,
    UnitError,
    UnitProgram,
    encode,
    insns,
)

# Dst access through ADDR_MOD_7 (no increments), the format the registers hold;
# ADDR_MOD_6 steps Dst by 2 (one row group) after a load macro.
ADDR_MOD, MACRO_ADDR_MOD = 7, 6
ROWS, FACES = 8, 4
TILE = 64  # Dst address units of one fp32 tile: 4 faces of 16 rows
# Operand tiles a whole-tile stream body reads (Dst tiles 0 ..).
OPERANDS = 3
# The readers of a MAD-class result that the unit does not hold back (rule
# mad-reader: the ISA's SFPMAD page; SFPGT, SFPLE from tt-metal's ckernel_sfpu_recip.h).
NOT_HELD = ("SFPIADD", "SFPSHFT", "SFPAND", "SFPOR", "SFPSWAP", "SFPGT", "SFPLE")
SHFT2_LANES = (2, 3, 4)  # SFPSHFT2 lane modes: their reads are not seen at all
STALL_SFPU, WAIT_MATH = 0x100, 0x10
SET_D, CR_D = 0x4, 0x4
CC_RESET = (3, 10)  # SFPENCC imm, mod1: condition on, every lane's flag set
INF_BF16 = 0x7F80
# The rules a compute body must obey that no check here covers yet.
NOT_CHECKED = (
    "interleave: two independent MAD chains interleaved (a speed rule)",
    "load-macro: SFPLOADMACRO delays counted in instructions, drained by SFPNOPs",
    "exp-range: the biased exp argument kept inside its range",
    "indirect: indirect register forms",
    "stop-word: the stop word polled in every wait and unit loop (the caller's)",
    "dst-window: Dst accesses inside the window (the walk of rows is not checked)",
    "dst-tiles: Dst tiles per acquire (the caller's index)",
    "placement: code placement and the kernel buffer's size",
    "src-banks: an unpack of more faces than Src has banks (2) waits for a release",
)


class Body:
    """The instructions of one row group (4 rows x 8 columns of each face, the
    vector unit's 32 lanes) on named values; `self.code` in issue order."""

    def __init__(self):
        self.code, self.count, self.depth = [], 0, 0
        self.prog, self.bits = {}, {}
        # {bits: value}: constants loaded once and read again (None: one per read),
        # those of `cacheable`; `loaded`: loads made of each constant
        self.cache, self.cacheable, self.loaded = None, set(), {}

    def fresh(self):
        self.count += 1
        return f"v{self.count}"

    def op(self, name, note="", **args):
        """Appends `name`; a written register argument left out gets a new value.
        Returns the value written (or None)."""
        for arg in OPS[name].writes:
            args.setdefault(arg, self.fresh())
        for arg, _ in OPS[name].args:
            args.setdefault(arg, 0)
        insn = Insn(name, args, note)
        self.code.append(insn)
        written = insn.writes()
        return written[0] if written else None

    def load(self, address=0, note=""):
        return self.op(
            "SFPLOAD",
            note or f"Dst + {address}",
            instr_mod0=0,
            sfpu_addr_mode=ADDR_MOD,
            dest_reg_addr=address,
        )

    def store(self, value, address=0, note="the result to Dst"):
        self.op(
            "SFPSTORE",
            note,
            lreg_ind=value,
            sfpu_addr_mode=ADDR_MOD,
            dest_reg_addr=address,
        )

    def const(self, value, note=""):
        """`value` (fp32) into a new value: one SFPLOADI when bf16 or fp16 holds it
        exactly, else the low then the high half."""
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        if self.cache is not None and bits in self.cache:
            return self.cache[bits]
        out = self.fresh()
        self.loaded[bits] = self.loaded.get(bits, 0) + 1
        if self.cache is not None and bits in self.cacheable:
            self.cache[bits] = out
        note = note or f"{value!r}"
        if bits & 0xFFFF == 0:
            self.op("SFPLOADI", note, lreg_ind=out, instr_mod0=0, imm16=bits >> 16)
            return out
        half = struct.pack("<e", value) if abs(value) < 65504 else None
        if half is not None and struct.unpack("<e", half)[0] == _fp32(value):
            imm = struct.unpack("<H", half)[0]
            self.op("SFPLOADI", note, lreg_ind=out, instr_mod0=1, imm16=imm)
            return out
        self.op("SFPLOADI", note, lreg_ind=out, instr_mod0=2, imm16=bits & 0xFFFF)
        self.op("SFPLOADI", note, lreg_ind=out, instr_mod0=8, imm16=bits >> 16)
        return out

    def register_const(self, value, bits=None):
        """A programmable constant register holding fp32 `value` (or the integer
        `bits`), set once by the setup after the schedule pass hoists it: its name
        until the layout pass."""
        if bits is None:
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
        if bits in self.bits:
            return self.bits[bits]
        if len(self.prog) == len(PROGRAMMABLE):
            raise UnitError("more constants than the 3 programmable registers")
        name = f"p{len(self.prog)}"
        self.bits[bits] = name
        self.prog[name] = value if value is not None else hex(bits)
        self.code += _config(name, bits, f"{self.prog[name]}")
        return name

    def mad(self, a, b, c, note="", neg_a=False, neg_c=False, into=None):
        """±(a b) ± c, rounded once."""
        mod = int(neg_a) | int(neg_c) << 1
        args = {"lreg_dest": into} if into else {}
        return self.op(
            "SFPMAD",
            note,
            lreg_src_a=a,
            lreg_src_b=b,
            lreg_src_c=c,
            instr_mod1=mod,
            **args,
        )

    def mul(self, a, b, note=""):
        return self.op("SFPMUL", note, lreg_src_a=a, lreg_src_b=b, lreg_src_c="zero")

    def add(self, a, b, note=""):
        return self.op("SFPADD", note, lreg_src_a="one", lreg_src_b=a, lreg_src_c=b)

    def neg(self, a, note=""):
        return self.op("SFPMOV", note or f"-{a}", lreg_c=a, instr_mod1=1)

    def when(self, name, note, **args):
        """A flag-setting instruction: the lanes where it holds run what follows
        until `end`."""
        self.depth += 1
        return self.op(name, note, **args)

    def end(self):
        self.depth = 0
        imm, mod = CC_RESET
        self.op("SFPENCC", "every lane on", imm12_math=imm, instr_mod1=mod)


def _config(name, bits, note):
    """SFPLOADI of `bits` into LReg 0 (one word when the low half is 0) and the
    SFPCONFIG that copies it into constant register `name`."""
    out = []
    if bits & 0xFFFF:
        args = {"lreg_ind": "L0", "instr_mod0": 2, "imm16": bits & 0xFFFF}
        out.append(Insn("SFPLOADI", args, note))
        args = {"lreg_ind": "L0", "instr_mod0": 8, "imm16": bits >> 16}
    else:
        args = {"lreg_ind": "L0", "instr_mod0": 0, "imm16": bits >> 16}
    out.append(Insn("SFPLOADI", args, note))
    config = {"imm16_math": 0, "config_dest": name, "instr_mod1": 0}
    out.append(Insn("SFPCONFIG", config, f"constant register {name}"))
    return out


def _fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def exp(b, a):
    """e^a in fp32 (tt-metal's `_sfpu_exp_fp32_accurate_`, guarded form): round a / ln 2
    to an integer, a two-part Cody-Waite reduction, a degree-6 polynomial, then the
    exponent added back with overflow to infinity and underflow to 0."""
    j = b.mul(a, b.const(1.442695), "a / ln 2")
    r = b.const(1.37805939e-3)
    sm = b.op(
        "SFP_STOCH_RND",
        "round to sign-magnitude 16",
        lreg_src_b="zero",
        lreg_src_c=j,
        instr_mod1=7,
    )
    j = b.op("SFPCAST", "back to fp32", lreg_src_c=sm)
    f = b.mad(j, b.const(-6.93145752e-1), a, "f = a - j ln 2 (high)")
    f = b.mad(j, b.const(-1.42860677e-6), f, "f -= j ln 2 (low)")
    for c in (8.37312452e-3, 4.16695364e-2, 1.66664720e-1, 4.99999851e-1):
        r = b.mad(r, f, b.const(c), "polynomial")
    i = b.op("SFPABS", "|j| as an integer", lreg_c=sm)
    y = b.mad(r, f, "one", "polynomial")
    i = b.op("SFPSETSGN", "j's sign on |j|", lreg_c=i, lreg_dest=j)
    r = b.mad(y, f, "one", "e^f")
    y = b.op("SFPMULI", "overflow: inf", imm16_math=INF_BF16, lreg_dest=y)
    e = b.op("SFPEXEXP", "biased exponent of e^f", lreg_c=r, instr_mod1=1)
    e = b.op("SFPIADD", "+ j", lreg_c=i, lreg_dest=e, instr_mod1=IADD_NO_CC)
    b.when(
        "SFPIADD",
        "exponent < 255",
        imm12_math=(-255) & 0xFFF,
        lreg_c=e,
        instr_mod1=IADD_IMM,
    )
    b.op("SFPMOV", "", lreg_c=e, lreg_dest=y)
    b.op("SFPSETEXP", "2^j e^f", lreg_c=r, lreg_dest=y)
    b.when(
        "SFPIADD",
        "exponent < 1",
        imm12_math=(-1) & 0xFFF,
        lreg_c=e,
        lreg_dest=e,
        instr_mod1=IADD_IMM,
    )
    b.op("SFPMOV", "underflow: 0", lreg_c="zero", lreg_dest=y)
    b.end()
    return y


def recip(b, x):
    """1 / x (tt-metal's `sfpu_reciprocal_iter<2>`): the unit's estimate and two
    Newton steps, the second only where the first is not NaN (t < 0)."""
    two = b.register_const(2.0)
    y = b.op("SFPARECIP", "estimate of 1 / x", lreg_c=x)
    t = b.mad(x, y, two, "t = x y - 2", neg_c=True)
    y1 = b.mad(y, b.neg(t), b.const(-0.0), "y1 = y (-t) - 0")
    b.when("SFPGT", "t < 0", lreg_c=t, lreg_dest="zero", instr_mod1=1)
    t = b.mad(x, y1, two, "t = x y1 - 2", neg_c=True)
    b.mad(y1, b.neg(t), b.const(-0.0), "y = y1 (-t) - 0", into=y)
    b.end()
    return y


def sigmoid(b, x):
    """1 / (1 + e^-x) on fp32 registers, as tt-metal's `_sfpu_sigmoid_<true>`."""
    return recip(b, b.add(exp(b, b.neg(x)), "one", "1 + e^-x"))


def rsqrt(b, x):
    """1 / sqrt(x) for a positive finite x (tt-metal's `_calculate_sqrt_body_`, its
    23-bit reciprocal form, without the guards for 0, infinity and x < 0): the
    initial guess from x's bits, a degree-2 correction, then one Newton step."""
    magic = b.register_const(None, bits=0x5F1110A0)
    c1, c2 = b.register_const(2.2825186), b.register_const(2.2533049)
    i = b.op(
        "SFPSHFT",
        "x's bits >> 1",
        imm12_math=(-1) & 0xFFF,
        lreg_c=x,
        instr_mod1=SHIFT_IMM_VC,
    )
    y = b.op(
        "SFPIADD",
        "y = magic - (x >> 1)",
        lreg_c=magic,
        lreg_dest=i,
        instr_mod1=IADD_SUB | IADD_NO_CC,
    )
    xy = b.mul(x, y, "x y")
    c = b.mad(y, xy, "zero", "c = -y x y", neg_a=True)
    y = b.mul(y, b.mad(c, b.add(c, c2, "c2 + c"), c1, "c1 + c (c2 + c)"), "y")
    xy = b.mul(x, y, "x y")
    e = b.mad(y, xy, "one", "1 - y x y", neg_a=True)
    half = b.op("SFPDIVP2", "y / 2", imm12_math=0xFF, lreg_c=y, instr_mod1=1)
    return b.mad(e, half, y, "y + (1 - y x y) y / 2")


def flash_exp2(b, u):
    """2^(u - 639) as Kohaku-TT's flash softmax computes it (sfpu.flash_exp2): u
    clamped to 512 from below, its bits << 9 give the exponent and the fraction f,
    a degree-2 polynomial of 1.f gives the mantissa."""
    floor = b.register_const(sfpu.FLASH_FLOOR)
    c = b.op("SFPMOV", "u (clamped in place)", lreg_c=u)
    b.op("SFPSWAP", "max(u, 512)", lreg_src_c=c, lreg_dest=floor, instr_mod1=1)
    e = b.op("SFPSHFT", "bits << 9", imm12_math=9, lreg_c=c, instr_mod1=SHIFT_IMM_VC)
    m = b.op("SFPSETEXP", "1.f", imm12_math=127, lreg_c=e, instr_mod1=1)
    p = b.mad(m, b.const(sfpu.FLASH_B2), b.const(sfpu.FLASH_B1), "B2 m + B1")
    p = b.mad(p, m, b.const(sfpu.FLASH_B0), "(B2 m + B1) m + B0")
    return b.op("SFPSETEXP", "2^n p", lreg_c=p, lreg_dest=e, instr_mod1=2)


def maximum(b, x, y):
    """The larger of x and y in every lane: one SFPSWAP on copies (the copy of a
    value read no more is dropped by the pass drop_copies)."""
    big = b.op("SFPMOV", "max: first", lreg_c=x)
    small = b.op("SFPMOV", "max: second", lreg_c=y)
    b.op("SFPSWAP", "max, min", lreg_src_c=big, lreg_dest=small, instr_mod1=1)
    return big


FUNCTIONS = {"exp": exp, "sigmoid": sigmoid, "rsqrt": rsqrt}
# Lane tests of a condition: (instruction, mod1, the operands swapped)
CONDITIONS = {"gt": ("SFPGT", 1, False), "lt": ("SFPGT", 1, True)}
ZERO_TESTS = {"eq": 6, "ne": 2}  # SFPSETCC on the integer view: -0.0 is not 0


def _value(b, node, done, source=None):
    """The value of sfpu node `node` (an operand, a constant, an operation);
    `source(index)` gives operand `index` (default: a load of Dst tile `index`)."""
    key = id(node)
    if key in done:
        return done[key]
    k = node.kind
    if k == "operand":
        if source is None and (node.value[1] is not None or node.along):
            fail("the unit program reads whole-tile operands")
        if source is None:
            if node.value[0] >= OPERANDS:
                fail(f"the unit program reads at most {OPERANDS} operand tiles")
            out = b.load(TILE * node.value[0], f"operand {node.value[0]}")
        else:
            out = source(node.value[0])
    elif k == "const":
        fixed = {0: "zero", 0x3F800000: "one"}
        out = fixed.get(struct.unpack("<I", struct.pack("<f", node.value))[0])
        out = out or b.const(node.value)
    elif k == "max":
        x, y = (_value(b, a, done, source) for a in node.args)
        out = maximum(b, x, y)
    elif k == "mad":
        out = b.mad(*(_value(b, a, done, source) for a in node.args))
    elif k == "flash_exp2":
        out = flash_exp2(b, _value(b, node.args[0], done, source))
    elif k == "select":
        then, other = (_value(b, a, done, source) for a in node.args)
        out = b.op("SFPMOV", "select: the other value", lreg_c=other)
        _when(b, node.value, done, source)
        b.op("SFPMOV", "select: where it holds", lreg_c=then, lreg_dest=out)
        b.end()
    elif k == "call":
        if node.value not in FUNCTIONS:
            fail(f"vector function {node.value} is not built in the unit program")
        out = FUNCTIONS[node.value](b, _value(b, node.args[0], done, source))
    elif k in ("mul", "add", "sub", "neg"):
        args = [_value(b, a, done, source) for a in node.args]
        if k == "mul":
            out = b.mul(*args, "x sigmoid(x)" if node.args[1].kind == "call" else "")
        elif k == "add":
            out = b.add(*args)
        elif k == "sub":
            out = b.add(args[0], b.neg(args[1]))
        else:
            out = b.neg(args[0])
    else:
        fail(f"vector operation {k} is not built in the unit program")
    done[key] = out
    return out


def _when(b, cond, done, source):
    """Lane flags from comparison `cond`: the lanes where it holds run what follows."""
    x, y = (_value(b, v, done, source) for v in (cond.a, cond.b))
    if cond.kind in ZERO_TESTS and y == "zero":
        mod = ZERO_TESTS[cond.kind]
        b.when("SFPSETCC", f"x {cond.kind} 0", lreg_c=x, instr_mod1=mod)
        return
    if cond.kind not in CONDITIONS:
        fail(f"the comparison {cond.kind} is not built in the unit program")
    name, mod, swap = CONDITIONS[cond.kind]
    x, y = (y, x) if swap else (x, y)
    b.when(name, f"x {cond.kind} y", lreg_c=y, lreg_dest=x, instr_mod1=mod)


def drop_copies(code):
    """Algebra: a plain copy (SFPMOV) of a value read no more after it is no copy:
    its readers take the value itself. Returns what it did."""
    first, last = {}, {}
    for i, insn in enumerate(code):
        for v in insn.reads() + insn.writes():
            first.setdefault(v, i)
            last[v] = i
    alias, keep, depth = {}, [], 0
    for i, insn in enumerate(code):
        for arg, _ in OPS[insn.op].args:
            value = insn.args[arg]
            if isinstance(value, str) and value in alias:
                insn.args[arg] = alias[value]
        if insn.sets_flags():
            depth = 0 if insn.op == "SFPENCC" else depth + 1
        a = insn.args
        source, out = a.get("lreg_c"), a.get("lreg_dest")
        if (
            insn.op == "SFPMOV"
            and a["instr_mod1"] == 0
            and not depth
            and isinstance(source, str)
            and source.startswith("v")
            and last.get(source) == i
            and first.get(out) == i
        ):
            alias[out] = source
            last[source] = last.get(out, i)
            continue
        keep.append(insn)
    done = len(code) - len(keep)
    code[:] = keep
    return [f"{done} copies of values read no more"] if done else []


def common_values(code):
    """Algebra: an instruction computing what an earlier one computed from the same
    values is dropped, its readers take the earlier result (a per-row value computed
    once a block, not once a vector). Values written once only. Returns what it did."""
    writes, loaders = {}, {}
    for insn in code:
        for v in insn.writes():
            writes[v] = writes.get(v, 0) + 1
            loaders.setdefault(v, []).append(insn)
    # a constant is known by its bits, whichever value loads it
    bits = {
        v: _bits_of(words)
        for v, words in loaders.items()
        if all(w.op == "SFPLOADI" for w in words)
    }
    seen, alias, keep, depth = {}, {}, [], 0
    for insn in code:
        for arg, _ in OPS[insn.op].args:
            value = insn.args[arg]
            if isinstance(value, str) and value in alias:
                insn.args[arg] = alias[value]
        if insn.sets_flags():
            depth = 0 if insn.op == "SFPENCC" else depth + 1
        # constants stay loaded where read (the constant passes place them)
        out, kind = insn.writes(), OPS[insn.op].kind
        if (
            not depth
            and not insn.sets_flags()
            and kind in ("mad", "simple")
            and insn.op != "SFPSWAP"
            and len(out) == 1
            and isinstance(out[0], str)
            and out[0].startswith("v")
            and writes[out[0]] == 1
            and all(writes.get(v, 0) <= 1 or bits.get(v) for v in insn.reads())
        ):
            fields = [a for a, _ in OPS[insn.op].args if a not in OPS[insn.op].writes]
            known = [
                ("bits", bits[v]) if bits.get(v) else v
                for v in (insn.args[a] for a in fields)
            ]
            key = (insn.op, *zip(fields, known, strict=True))
            if key in seen:
                alias[out[0]] = seen[key]
                continue
            seen[key] = out[0]
        keep.append(insn)
    read = {v for insn in keep for v in insn.reads() if insn.op != "SFPLOADI"}
    keep = [x for x in keep if x.op != "SFPLOADI" or x.writes()[0] in read]
    done = len(code) - len(keep)
    code[:] = keep
    return [f"{done} instructions computed once for their readers"] if done else []


def fold_negation(code):
    """Algebra: a negation (SFPMOV mod1 1) disappears into a multiply-add, into its
    producer (one reader: both signs flipped) or its consumer (one reader, as a
    product operand: the product's sign flipped). Returns what it did."""
    done, i = [], 0
    while i < len(code):
        insn = code[i]
        if insn.op != "SFPMOV" or insn.args["instr_mod1"] != 1:
            i += 1
            continue
        source, out = insn.args["lreg_c"], insn.args["lreg_dest"]
        made = [x for x in code if source in x.writes()]
        readers = [x for x in code if source in x.reads()]
        users = [x for x in code if out in x.reads()]
        if len(made) == 1 and made[0].op == "SFPMAD" and readers == [insn]:
            made[0].args["instr_mod1"] ^= 3
            made[0].args["lreg_dest"] = out
            made[0].note = f"-({made[0].note})"
            done.append(f"a negation into its producer ({made[0].note})")
            del code[i]
            continue
        if len(users) == 1 and users[0].op == "SFPMAD":
            user = users[0].args
            slots = [s for s in ("lreg_src_a", "lreg_src_b") if user[s] == out]
            if len(slots) == 1 and user["lreg_src_c"] != out:
                if slots == ["lreg_src_b"]:
                    user["lreg_src_b"] = user["lreg_src_a"]
                user["lreg_src_a"] = source
                user["instr_mod1"] ^= 1
                done.append(f"a negation into its consumer ({users[0].note})")
                del code[i]
                continue
        if users and all(_addend_only(x, out) for x in users):
            for x in users:
                x.args["lreg_src_c"] = source
                x.args["instr_mod1"] ^= 2
            done.append(f"a negation into the addend of {len(users)} multiply-add(s)")
            del code[i]
            continue
        i += 1
    return done


def _addend_only(insn, value):
    """True when multiply-add `insn` reads `value` once, as its addend (VC)."""
    if insn.op not in ("SFPMAD", "SFPADD"):
        return False
    reads = [insn.args[f] for f in ("lreg_src_a", "lreg_src_b", "lreg_src_c")]
    return reads.count(value) == 1 and insn.args["lreg_src_c"] == value


def fuse_mad(code):
    """Algebra: a product read only by an add becomes one multiply-add (one rounding
    in place of two). Returns what it did."""
    done, i = [], 0
    while i < len(code):
        insn = code[i]
        if insn.op != "SFPMUL" or insn.args["lreg_src_c"] != "zero":
            i += 1
            continue
        out = insn.args["lreg_dest"]
        users = [x for x in code if out in x.reads()]
        made = [x for x in code if out in x.writes()]
        add = users[0] if len(users) == 1 and len(made) == 1 else None
        if add is None or add.op != "SFPADD" or add.args["lreg_src_a"] != "one":
            i += 1
            continue
        b, c = add.args["lreg_src_b"], add.args["lreg_src_c"]
        if (b == out) == (c == out):
            i += 1
            continue
        add.op = "SFPMAD"
        add.args["lreg_src_a"] = insn.args["lreg_src_a"]
        add.args["lreg_src_b"] = insn.args["lreg_src_b"]
        add.args["lreg_src_c"] = c if b == out else b
        if c == out and add.args["instr_mod1"] & 2:
            # the add negated the product: the product's sign flips, not the addend's
            add.args["instr_mod1"] ^= 3
        add.args["instr_mod1"] ^= insn.args["instr_mod1"] & 1
        add.note = f"{insn.note or 'a b'} + c, one rounding".strip()
        done.append(f"a product into its add ({add.note})")
        del code[i]
    return done


def constant_registers(code, prog):
    """Layout: the constants the row code loads (SFPLOADI) move into the free
    programmable registers, most loader words saved first; equal constants share
    one. Returns what it did."""
    found = {}
    for insn in code:
        if insn.op == "SFPLOADI" and isinstance(insn.args["lreg_ind"], str):
            value = insn.args["lreg_ind"]
            if value.startswith("v"):
                found.setdefault(value, []).append(insn)
    by_bits = {}
    for value, words in found.items():
        writers = [x for x in code if value in x.writes()]
        if len(writers) != len(words) or any(x.op != "SFPLOADI" for x in writers):
            continue
        bits = _bits_of(words)
        if bits is None:
            continue
        by_bits.setdefault(bits, []).append(value)
    saved = {
        bits: sum(len(found[v]) for v in values) for bits, values in by_bits.items()
    }
    done = []
    for bits in sorted(saved, key=lambda b: -saved[b]):
        if len(prog) == len(PROGRAMMABLE):
            break
        name = f"p{len(prog)}"
        values = by_bits[bits]
        note = found[values[0]][0].note
        prog[name] = note
        for value in values:
            for word in found[value]:
                code.remove(word)
            for insn in code:
                for arg, _ in OPS[insn.op].args:
                    if insn.args[arg] == value:
                        insn.args[arg] = name
        code[:0] = _config(name, bits, note)
        done.append(f"{note} in {name} ({saved[bits]} words a row)")
    return done


def negated_zero(code):
    """Algebra: a -0.0 loaded only as the addend of multiply-adds becomes the fixed
    zero with the addend's sign flipped (one SFPLOADI less). Returns what it did."""
    done = []
    for insn in [x for x in code if x.op == "SFPLOADI"]:
        value = insn.args["lreg_ind"]
        if (insn.args["instr_mod0"], insn.args["imm16"]) != (0, 0x8000):
            continue
        readers = [x for x in code if value in x.reads()]
        writers = [x for x in code if value in x.writes()]
        if writers != [insn] or not all(
            x.op == "SFPMAD"
            and x.read_fields().count("lreg_src_c") == 1
            and [x.args[f] for f in x.read_fields()].count(value) == 1
            and x.args["lreg_src_c"] == value
            for x in readers
        ):
            continue
        for x in readers:
            x.args["lreg_src_c"] = "zero"
            x.args["instr_mod1"] ^= 2
        code.remove(insn)
        done.append(f"-0.0 as -zero ({len(readers)} reader(s))")
    return done


def resident_constants(code, prog):
    """Layout: constants the row still loads move into work registers that stay set
    across the rows, most words saved first, while the row's values still fit; equal
    constants share one. Returns (what it did, the loads, the pinned values)."""
    found = {}
    for insn in code:
        if insn.op == "SFPLOADI" and isinstance(insn.args["lreg_ind"], str):
            found.setdefault(insn.args["lreg_ind"], []).append(insn)
    by_bits = {}
    for value, words in found.items():
        writers = [x for x in code if value in x.writes()]
        bits = _bits_of(words)
        if writers == words and bits is not None and value.startswith("v"):
            by_bits.setdefault(bits, []).append(value)
    saved = {b: sum(len(found[v]) for v in vs) for b, vs in by_bits.items()}
    done, loads, pinned = [], [], []
    for bits in sorted(saved, key=lambda b: -saved[b]):
        trial = copy.deepcopy(code)
        first, *others = by_bits[bits]
        _share(trial, by_bits[bits], first)
        words = [x for x in trial if x.op == "SFPLOADI" and x.args["lreg_ind"] == first]
        for w in words:
            trial.remove(w)
        try:
            allocate(copy.deepcopy(loads + words + trial), prog, [*pinned, first])
        except ValueError:
            continue
        code[:] = trial
        loads += words
        pinned.append(first)
        done.append(f"{words[0].note} held across rows ({saved[bits]} words a row)")
    return done, loads, pinned


def _share(code, values, name):
    """Every value of `values` read as `name`; their loads but `name`'s dropped."""
    for insn in list(code):
        held = insn.args.get("lreg_ind")
        if insn.op == "SFPLOADI" and held in values and held != name:
            code.remove(insn)
        for arg, _ in OPS[insn.op].args:
            if insn.args[arg] in values:
                insn.args[arg] = name


def _bits_of(words):
    """The 32-bit value SFPLOADI words build (FLOATB, FLOATA, USHORT, then UPPER),
    or None for another form."""
    bits = None
    for w in words:
        mod, imm = w.args["instr_mod0"], w.args["imm16"]
        if mod == 0:
            bits = imm << 16
        elif mod == 1:
            half = struct.unpack("<e", struct.pack("<H", imm))[0]
            bits = struct.unpack("<I", struct.pack("<f", half))[0]
        elif mod == 2:
            bits = imm
        elif mod == 8 and bits is not None:
            bits = (bits & 0xFFFF) | imm << 16
        else:
            return None
    return bits


def hoist(code, setup):
    """Schedule: the constant registers' setup (SFPLOADI into L0, SFPCONFIG) moves
    out of the per-row code into the setup, once per stream."""
    done = []
    i = len(code) - 1
    while i >= 0:
        if code[i].op == "SFPCONFIG":
            j = i
            while j > 0 and code[j - 1].op == "SFPLOADI":
                if code[j - 1].args["lreg_ind"] != "L0":
                    break
                j -= 1
            setup[:0] = code[j : i + 1]
            done.append(f"constant register {code[i].args['config_dest']}")
            del code[j : i + 1]
            i = j
        i -= 1
    return list(reversed(done))


def replay(loop):
    """Schedule: an unrolled loop's first pass records the first entries of its body
    into the replay buffer as it runs; the other passes replay them and issue only
    the rest (rule replay). Returns the loop nest that replaces `loop`."""
    if not loop.unroll or loop.count < 2:
        return [loop], []
    if any(x.op == "REPLAY" for x in insns(loop.body)):
        return [loop], []
    n = min(REPLAY_ENTRIES, len(loop.body))
    record = Insn("REPLAY", _replay(n, 1), f"record {n} while they run")
    play = Insn("REPLAY", _replay(n, 0), f"replay {n}")
    rest = Loop(loop.name, loop.count - 1, [play, *loop.body[n:]], unroll=True)
    done = [f"{n} of {len(loop.body)} instructions of each {loop.name} recorded once"]
    return [record, *loop.body, rest], done


def _replay(count, load):
    return {
        "start_idx": 0,
        "len": count,
        "execute_while_loading": load,
        "load_mode": load,
    }


def allocate(code, prog, pinned=()):
    """Layout: each value gets an LReg 0 .. 7 for its live range (a result takes the
    register of an operand that dies there, else the lowest free one); the fixed
    constants take 9 / 10 / 11, the programmable ones 12 upward. `pinned` values live
    to the end of `code` (they are read again by the next row)."""
    regs = {name: FIXED[name] for name in FIXED}
    regs.update({name: PROGRAMMABLE[i] for i, name in enumerate(prog)})
    last = {}
    for i, insn in enumerate(code):
        for v in insn.reads() + insn.writes():
            if isinstance(v, str) and v not in regs:
                last[v] = len(code) if v in pinned else i
    free = list(range(WORK))
    for i, insn in enumerate(code):
        dying = [
            v
            for v in insn.reads()
            if isinstance(v, str) and last.get(v) == i and v not in insn.writes()
        ]
        free += [regs[v] for v in dict.fromkeys(dying) if regs.get(v, WORK) < WORK]
        for v in dict.fromkeys(insn.writes()):
            if isinstance(v, str) and v not in regs:
                taken = [regs[d] for d in dying if regs.get(d, WORK) < WORK]
                if not taken and not free:
                    raise ValueError("more live values than the 8 work registers")
                regs[v] = taken[0] if taken else min(free)
                free.remove(regs[v])
                if last.get(v, i) == i:
                    free.append(regs[v])
            elif (
                isinstance(v, str)
                and last.get(v) == i
                and regs[v] < WORK
                and regs[v] not in free
            ):
                # rewritten in place (SFPSWAP's smaller value) and read no more
                free.append(regs[v])
    for insn in code:
        for arg, _ in OPS[insn.op].args:
            value = insn.args[arg]
            if isinstance(value, str):
                insn.args[arg] = int(value[1:]) if value[0] == "L" else regs[value]
    used = sorted({r for v, r in regs.items() if v not in FIXED and r < WORK})
    return [f"{len(used)} of {WORK} work registers"]


def check_mad_readers(blocks_):
    """mad-reader: an SFPNOP between a multiply-add and a reader the unit does not
    hold back. Returns the repairs made."""
    repairs = 0
    for code in blocks_:
        i = 1
        while i < len(code):
            prev, cur = code[i - 1], code[i]
            if OPS[prev.op].kind == "mad" and not_held(cur, set(prev.writes())):
                code.insert(i, Insn("SFPNOP", {}, "hazard mad-reader: MAD result"))
                repairs += 1
            i += 1
    return repairs


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
    n = check_config_nop(math_init)
    unit.checks.append(
        Check(
            "config-nop",
            "SFPNOP after an SFPCONFIG of LReg 11 to 14",
            f"repaired: {n} SFPNOP" if n else "ok",
        )
    )
    written = sorted(
        x.args["config_dest"]
        for x in insns(math_init)
        if x.op == "SFPCONFIG" and 11 <= x.args["config_dest"] <= 14
    )
    read = sorted({r for x in insns(math_tile) for r in x.reads() if r in PROGRAMMABLE})
    if set(read) - set(written):
        raise UnitError(f"LReg {read} read but not set by the setup")
    if 11 in written:
        raise UnitError("LReg 11 is written: its -1.0 restore is not built")
    unit.clobbers = [f"LReg{r}" for r in written]
    unit.checks.append(
        Check(
            "constants",
            "constant registers set before use, LReg 11 restored to -1.0",
            f"ok: LReg {written} set by the setup and declared clobbered; "
            "LReg 11 not written",
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
    for insn in insns(math_init + math_tile):
        encode(insn.op, insn.args)
    unit.checks.append(
        Check(
            "immediates",
            "every immediate and register field inside its width",
            "ok: every word encoded (all immediates are compile-time)",
        )
    )


def _order(items):
    for item in items:
        if isinstance(item, Loop):
            yield from _order(item.body)
        elif isinstance(item, Insn):
            yield item


def stream_chain(ir):
    """The Chain of a stream kernel's one store (tile in, tile out), as the stream
    form plans it: each load is a register operand."""
    if len(ir.stores) != 1:
        fail(f"{ir.name}: the unit program takes one store")
    shapes = chains._Shapes(False)
    nodes = ir.nodes()
    for n in nodes:
        shapes(n)
    loads = {id(n): n for n in nodes if n.kind in ("load", "resident")}

    def leaf_of(node, offset, kind):
        if id(node) not in loads or kind not in ("copy", "bcast_col") or offset:
            return None
        return chains.Leaf(node, kind, offset)

    compiler = chains._Compiler(ir.name, shapes, {})
    return compiler.chain(ir.stores[0].value, leaf_of)


def _templates(chosen):
    """The templates record: each transcendental's chosen form and its bound."""
    return [f"{k}: {v}" for k, v in chosen.items()]


def unit_program(ir, dest="fp32", load_macro=False):
    """The unit program of stream kernel `ir` on `dest` (fp32) Dst registers: operand
    k read from Dst tile k (relative to the caller's `dst`), the result stored into
    tile 0. `load_macro`: try the load-macro rule (a refusal is a pass's record)."""
    if dest != "fp32":
        raise PlanError("komorebi.dsl.tile: the unit program is built for fp32 Dst")
    chain = stream_chain(ir)
    if chain.values or chain.prev is not None:
        fail(f"{ir.name}: the unit program takes no runtime scalars")
    if len(chain.operands) > OPERANDS:
        fail(f"{ir.name}: the unit program reads at most {OPERANDS} operand tiles")
    dtypes = {t.dtype for t in ir.tensors}
    if not dtypes <= {BF16, F32}:
        fail(f"{ir.name}: tensors are bf16 or fp32")
    body = Body()
    per_row = [k for k, leaf in enumerate(chain.operands) if leaf.kind == "bcast_col"]
    if per_row:
        # A per-row operand (a state tile, each row's value in all its lanes): a row
        # step takes a 4-row block's even and odd vectors, the state loaded once.
        shared = {}

        def source(k, p):
            if k in per_row:
                if k not in shared:
                    shared[k] = body.load(TILE * k, f"per-row operand {k}")
                return shared[k]
            return body.load(TILE * k + 2 * p, f"operand {k}, {'odd' if p else 'even'}")

        results = [
            _value(body, chain.fn.result, {}, lambda k, p=p: source(k, p))
            for p in (0, 1)
        ]
        for p, result in enumerate(results):
            body.store(result, 2 * p, f"the result to Dst ({'odd' if p else 'even'})")
    else:
        result = _value(body, chain.fn.result, {})
        body.store(result)
    unit = UnitProgram(ir.name, dest)
    unit.operands = [leaf.source.attrs["view"].tensor.name for leaf in chain.operands]
    setup = []
    unit.passes.append(Pass("algebra", "templates", _templates(chain.templates)))
    unit.passes.append(Pass("algebra", "fold_negation", fold_negation(body.code)))
    unit.passes.append(Pass("algebra", "fuse_mad", fuse_mad(body.code)))
    if per_row:
        # The even and odd vectors of a block are two independent chains.
        from komorebi.dsl.tile.rows import interleave  # noqa: PLC0415 (rows imports us)

        unit.passes.append(Pass("algebra", "drop_copies", drop_copies(body.code)))
        unit.passes.append(Pass("algebra", "common_values", common_values(body.code)))
        moved = interleave(body.code, 0, len(body.code))
        done = [f"{moved} instructions moved in the even and odd chains"]
        unit.passes.append(Pass("schedule", "interleave", done))
    unit.passes.append(Pass("algebra", "negated_zero", negated_zero(body.code)))
    done = constant_registers(body.code, body.prog)
    unit.passes.append(Pass("layout", "constant_registers", done))
    unit.passes.append(Pass("schedule", "hoist", hoist(body.code, setup)))
    done, held, pinned = resident_constants(body.code, body.prog)
    unit.passes.append(Pass("layout", "resident_constants", done))
    whole = held + body.code
    unit.passes.append(Pass("layout", "allocate", allocate(whole, body.prog, pinned)))
    body.code[:] = whole[len(held) :]
    held_regs = {x.args["lreg_ind"] for x in held}
    clobbered = [x.op for x in body.code if held_regs & set(x.writes())]
    unit.checks.append(
        Check(
            "resident",
            "a register held across rows is written by no row instruction",
            f"refused: {clobbered}" if clobbered else f"ok: LReg {sorted(held_regs)}",
        )
    )
    if clobbered:
        raise UnitError(f"{ir.name}: a held constant register is overwritten")
    allocate(setup, body.prog)
    unit.constants = {
        PROGRAMMABLE[i]: value for i, value in enumerate(body.prog.values())
    }
    plan = None
    if load_macro and not per_row:
        try:
            plan = macro.fit(body.code)
        except macro.Refused as e:
            unit.passes.append(Pass("schedule", "load_macro", [f"refused: {e}"]))
    unit.init["math"] = [
        Insn(
            "SFPCONFIG",
            {"imm16_math": 0, "config_dest": 15, "instr_mod1": 1},
            "lane configuration",
        ),
        Call(
            "ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, "
            ".dest = {.incr = 0}}.set(ADDR_MOD_7)",
            "address mode 7: no increments (3 SETC16)",
        ),
        Insn("SETRWC", {**_setrwc(), "BitMask": 0xF}, "reset the counters"),
        *setup,
    ]
    if plan is None and per_row:
        row = [*body.code, Insn("INCRWC", _rwc(rwc_d=4), "next 4-row block")]
        face = Loop("row", ROWS // 2, row, unroll=True)
        drained = []
    elif plan is None:
        row = [*body.code, Insn("INCRWC", _rwc(rwc_d=2), "next 4 rows")]
        face = Loop("row", ROWS, row, unroll=True)
        drained = []
    else:
        row = macro.rows(plan, MACRO_ADDR_MOD)
        if ROWS % plan.rows:
            raise UnitError(f"{plan.rows} rows a macro turn do not divide a face")
        face = Loop("row", ROWS // plan.rows, row, unroll=True)
        unit.init["math"] += [
            Call(
                "ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, "
                ".dest = {.incr = 2}}.set(ADDR_MOD_6)",
                "address mode 6: Dst + 2 after a macro (3 SETC16)",
            ),
            *macro.setup(plan),
        ]
        drained = macro.drain(plan)
        unit.passes.append(
            Pass(
                "schedule",
                "load_macro",
                [
                    f"{len(plan.chain)} operation(s) and the store in one SFPLOADMACRO "
                    f"a row (store delay {plan.store_delay}); {len(plan.others)} "
                    f"plain load(s) a row; registers in turns of {plan.loaded_turns} "
                    f"(loaded) and {plan.plain_turns} (plain)"
                ],
            )
        )
    step = Insn(
        "SETRWC",
        {**_setrwc(), "rwc_cr": CR_D, "rwc_d": 8, "BitMask": SET_D},
        "Dst + 8 rows",
    )
    copies = [f"copy_tile(in{k}, 0, dst + {k})" for k in range(len(chain.operands))]
    unit.tile["unpack"] = [
        Sync("unpack", "reader", "cb_wait_front(in, 1)", "the input tiles"),
        *[Call(c, "the tile into Dst (fp32, unpacked to Dst)") for c in copies],
        Call("cb_pop_front(in, 1)", "frees the input tiles"),
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
        *held,
        Loop("face", FACES, [face, step, Insn(step.op, dict(step.args), step.note)]),
        *drained,
        Insn("SETRWC", {**_setrwc(), "BitMask": SET_D}, "Dst counter back to 0"),
        Sync("math", "pack", "tile_regs_commit()", "hands Dst to the pack CPU"),
    ]
    unit.tile["pack"] = [
        Sync("pack", "math", "tile_regs_wait()", "Dst from the math CPU"),
        Call("pack_tile(dst, out)", "Dst tile dst to the output buffer"),
        Sync("pack", "math", "tile_regs_release()", "frees the Dst half"),
        Sync("pack", "writer", "cb_push_back(out, 1)", "an output tile"),
    ]
    _checks(unit, row, plan)
    rows, done = replay(face)
    unit.passes.append(Pass("schedule", "replay", done))
    faces = next(x for x in unit.tile["math"] if isinstance(x, Loop))
    faces.body[:1] = rows
    unit.not_checked = [r for r in NOT_CHECKED if plan is None or "load-macro" not in r]
    return unit


def _rwc(**args):
    return {"rwc_cr": 0, "rwc_d": 0, "rwc_b": 0, "rwc_a": 0, **args}


def _setrwc():
    return {"clear_ab_vld": 0, **_rwc(), "BitMask": 0}
