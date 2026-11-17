"""A traced tile expression over a stream of tiles to its unit program: the chain
compiler's vector function (`chain.Chain`) is written as vector-unit instructions on
named values, the rules run as passes (algebra, schedule, layout), then the hazard
checks. Built on fp32 Dst: whole-tile operands, constants, +, -, *, sigmoid, rsqrt."""

import copy
import struct

from komorebi.dsl.tile import chain as chains
from komorebi.dsl.tile import macro, sfpu
from komorebi.dsl.tile.ir import A_DTYPES, TILE_DTYPES, PlanError, fail
from komorebi.dsl.tile.matrix import PHASES, tile_products
from komorebi.dsl.tile.schedule import interleave
from komorebi.dsl.tile.unit import (
    FIXED,
    IADD_IMM,
    IADD_NO_CC,
    IADD_SUB,
    OPS,
    PROGRAMMABLE,
    REPLAY_ENTRIES,
    SAVED,
    SHIFT_IMM_VC,
    WORK,
    Call,
    Check,
    Insn,
    Loop,
    Pass,
    Product,
    Runtime,
    Sync,
    UnitError,
    UnitProduct,
    UnitProgram,
    encode,
    insns,
)

# Dst access through ADDR_MOD_7 (no increments), the format the registers hold;
# ADDR_MOD_6 steps Dst by 2 (one row group) after a load macro.
ADDR_MOD, MACRO_ADDR_MOD = 7, 6
ROWS, FACES = 8, 4
BANDS = 4  # bands of 4 rows a face (rule band: a body runs the bands of live rows)
TILE = 64  # Dst address units of a tile in either Dst: 4 faces of 16 rows
# SFPLOAD / SFPSTORE's instr_mod0 by Dst: its own format (fp32), FP16A (a 16-bit Dst)
DEST_MODES = {"fp32": 0, "fp16a": 1}
# Operand tiles a whole-tile stream body reads: Dst tiles 0 .. 3 (half the fp32 Dst).
OPERANDS = 4
# The readers of a MAD-class result that the unit does not hold back (rule
# mad-reader: the ISA's SFPMAD page; SFPGT, SFPLE from tt-metal's ckernel_sfpu_recip.h).
NOT_HELD = ("SFPIADD", "SFPSHFT", "SFPAND", "SFPOR", "SFPSWAP", "SFPGT", "SFPLE")
SHFT2_LANES = (2, 3, 4)  # SFPSHFT2 lane modes: their reads are not seen at all
STALL_SFPU, WAIT_MATH = 0x100, 0x10
SET_D, CR_D = 0x4, 0x4
CC_RESET = (3, 10)  # SFPENCC imm, mod1: condition on, every lane's flag set
INF_BF16 = 0x7F80
# The constant registers in the order the layout pass fills them: 12 to 14, then 11
# (set and restored by the body itself, rule constant_registers' fourth register).
CONSTANT_REGISTERS = (*PROGRAMMABLE, SAVED)
SAVED_NAME = f"p{len(PROGRAMMABLE)}"
NEG_ONE_BF16 = 0xBF80
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
        self.tiles = None  # operand k's Dst tile, when not k (unit_program's tiles)
        self.prog, self.bits = {}, {}
        # {bits: value}: constants loaded once and read again (None: one per read),
        # those of `cacheable`; `loaded`: loads made of each constant
        self.cache, self.cacheable, self.loaded = None, set(), {}
        # constants of `whole` are loaded once for the whole unit (their values in
        # `held`, which the layout pass places in the highest work registers)
        self.whole, self.whole_cache, self.held = set(), {}, set()
        self.mode = 0  # SFPLOAD / SFPSTORE's format: Dst's own (FP16A on a 16-bit Dst)
        self.scalars = {}  # {run-time scalar index: its value, held across the rows}

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
            instr_mod0=self.mode,
            sfpu_addr_mode=ADDR_MOD,
            dest_reg_addr=address,
        )

    def store(self, value, address=0, note="the result to Dst"):
        self.op(
            "SFPSTORE",
            note,
            lreg_ind=value,
            instr_mod0=self.mode,
            sfpu_addr_mode=ADDR_MOD,
            dest_reg_addr=address,
        )

    def const(self, value, note=""):
        """`value` (fp32) into a new value: one SFPLOADI when bf16 or fp16 holds it
        exactly, else the low then the high half."""
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        if bits in self.whole_cache:
            return self.whole_cache[bits]
        if self.cache is not None and bits in self.cache:
            return self.cache[bits]
        out = self.fresh()
        self.loaded[bits] = self.loaded.get(bits, 0) + 1
        if bits in self.whole:
            self.whole_cache[bits] = out
            self.held.add(out)
        elif self.cache is not None and bits in self.cacheable:
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

    def scalar(self, index):
        """Run-time scalar `index`'s value: loaded once before the rows by its two
        run-time SFPLOADI (`scalar_loads`) and held."""
        if index not in self.scalars:
            self.scalars[index] = self.fresh()
        return self.scalars[index]

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
    a degree-2 polynomial of 1.f gives the mantissa. The swap's VD is a copy of 512:
    with the back door open a VD of 12 or more stores a template and runs nothing."""
    floor = b.register_const(sfpu.FLASH_FLOOR)
    c = b.op("SFPMOV", "u (clamped in place)", lreg_c=u)
    low = b.op("SFPMOV", "512 (the swap's VD takes the smaller)", lreg_c=floor)
    b.op("SFPSWAP", "max(u, 512)", lreg_src_c=c, lreg_dest=low, instr_mod1=1)
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


def sqrt(b, x):
    """The square root of x >= 0: x rsqrt(x) (0 at 0: rsqrt's guess from 0's bits is
    finite)."""
    return b.mul(x, rsqrt(b, x), "x rsqrt(x)")


def minimum(b, x, y):
    """The smaller of x and y in every lane: the other result of maximum's swap."""
    big = b.op("SFPMOV", "min: first", lreg_c=x)
    small = b.op("SFPMOV", "min: second", lreg_c=y)
    b.op("SFPSWAP", "max, min", lreg_src_c=big, lreg_dest=small, instr_mod1=1)
    return small


# tanh below |x| = TANH_SMALL: x + x t p(t), t = x^2 (a fit of tanh(x) / x - 1 over t)
TANH_SMALL = 0.625
TANH_P = (-0.33333284, 0.13331491, -0.053743497, 0.020650586, -0.0057170433)
# erf: x N(x^2) / D(x^2) on x clamped to [-10, 10] (tt-metal's fp32 erf rational)
ERF_N = (
    1.1283791065,
    2.1477432549e-01,
    6.2133435160e-02,
    5.6230435148e-03,
    6.1307044234e-04,
    1.7678321456e-05,
    2.7384647439e-08,
    -2.8632063387e-10,
)
ERF_D = (
    1.0,
    5.2367275953e-01,
    1.2961706519e-01,
    1.9642570987e-02,
    1.9545555115e-03,
    1.3179056987e-04,
    1.3156344494e-06,
    -3.5153888689e-09,
    -6.7350725691e-12,
)
# log(1 + f) = f + f^2 r(f) for f in [-0.25, 0.5] (Norbert Juffa's logf, as tt-metal)
LOG_P = tuple(
    float.fromhex(c)
    for c in (
        "-0x1.92cp-5",
        "0x1.b84p-4",
        "-0x1.0c4p-3",
        "0x1.274p-3",
        "-0x1.55p-3",
        "0x1.998p-3",
        "-0x1.00001ap-2",
        "0x1.555572p-2",
        "-0x1p-1",
    )
)[::-1]
LN2_FP32 = 0.693147182


def _horner(b, x, coefficients, note):
    """c[0] + x (c[1] + x (...)), one multiply-add a coefficient."""
    p = b.const(coefficients[-1])
    for c in reversed(coefficients[:-1]):
        p = b.mad(p, x, b.const(c), note)
    return p


def tanh(b, x):
    """tanh(x): below |x| = 0.625 the odd polynomial x + x t p(t), above it
    (1 - e^-2|x|) / (1 + e^-2|x|) with the accurate exp and x's sign."""
    t = b.mul(x, x, "t = x^2")
    p = _horner(b, t, TANH_P, "tanh polynomial")
    small = b.mad(b.mul(x, t, "x t"), p, x, "x + x t p(t)")
    a = b.op("SFPABS", "|x|", lreg_c=x, instr_mod1=1)
    out = b.op("SFPMOV", "x (its sign)", lreg_c=x)
    e = exp(b, b.mul(a, b.const(-2.0), "-2 |x|"))
    num = b.mad(e, "one", "one", "1 - e", neg_a=True)
    big = b.mul(num, recip(b, b.add(e, "one", "1 + e")), "(1 - e) / (1 + e)")
    out = b.op("SFPSETSGN", "the quotient with x's sign", lreg_c=big, lreg_dest=out)
    small_x = b.const(TANH_SMALL)
    b.when("SFPGT", "|x| < 0.625", lreg_c=a, lreg_dest=small_x, instr_mod1=1)
    b.op("SFPMOV", "tanh: the polynomial", lreg_c=small, lreg_dest=out)
    b.end()
    return out


def erf(b, x):
    """erf(x): x N(x^2) / D(x^2) of x clamped to [-10, 10] (tt-metal's fp32 rational),
    the quotient by `recip`, the result clamped to [-1, 1]."""
    c = minimum(b, maximum(b, x, b.const(-10.0)), b.const(10.0))
    s = b.mul(c, c, "s = x^2")
    n = b.mul(c, _horner(b, s, ERF_N, "erf numerator"), "x N(s)")
    y = b.mul(n, recip(b, _horner(b, s, ERF_D, "erf denominator")), "N / D")
    return minimum(b, maximum(b, y, b.const(-1.0)), "one")


def log(b, x):
    """The natural log of a normal x > 0: x = 2^k m, m in [0.75, 1.5) (m above 1.5
    halved, k + 1), log(m) = f + f^2 r(f) with f = m - 1, then k ln 2 + log(m)."""
    e = b.op("SFPEXEXP", "x's exponent - 127", lreg_c=x)
    m = b.op("SFPSETEXP", "m in [1, 2)", imm12_math=127, lreg_c=x, instr_mod1=1)
    e = b.op("SFPCAST", "k as sign-magnitude", lreg_src_c=e, instr_mod1=3)
    k = b.op("SFPCAST", "k as fp32", lreg_src_c=e)
    b.when("SFPGT", "m > 1.5", lreg_c=b.const(1.5), lreg_dest=m, instr_mod1=1)
    b.mad(m, b.const(0.5), "zero", "m / 2", into=m)
    b.mad(k, "one", "one", "k + 1", into=k)
    b.end()
    f = b.add(m, b.const(-1.0), "f = m - 1 (exact)")
    r = _horner(b, f, LOG_P, "log polynomial")
    r = b.mad(r, b.mul(f, f, "f^2"), f, "f + f^2 r")
    return b.mad(k, b.const(LN2_FP32), r, "k ln 2 + log(m)")


FUNCTIONS = {
    "exp": exp,
    "sigmoid": sigmoid,
    "rsqrt": rsqrt,
    "recip": recip,
    "sqrt": sqrt,
    "tanh": tanh,
    "erf": erf,
    "log": log,
}
# Lane tests of a condition: (instruction, mod1, the operands swapped)
CONDITIONS = {"gt": ("SFPGT", 1, False), "lt": ("SFPGT", 1, True)}
ZERO_TESTS = {"eq": 6, "ne": 2}  # SFPSETCC on the integer view: -0.0 is not 0
# The tests made of SFPGT's: the tests whose lanes take the other value over the
# value where the test holds (ne: the reverse), one SFPGT each
INVERSES = {"ge": ("lt",), "le": ("gt",), "eq": ("gt", "lt"), "ne": ("gt", "lt")}


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
            tile = b.tiles[node.value[0]] if b.tiles else node.value[0]
            out = b.load(TILE * tile, f"operand {node.value[0]}")
        else:
            out = source(node.value[0])
    elif k == "const":
        fixed = {0: "zero", 0x3F800000: "one"}
        out = fixed.get(struct.unpack("<I", struct.pack("<f", node.value))[0])
        out = out or b.const(node.value)
    elif k == "max":
        x, y = (_value(b, a, done, source) for a in node.args)
        out = maximum(b, x, y)
    elif k == "min":
        x, y = (_value(b, a, done, source) for a in node.args)
        out = minimum(b, x, y)
    elif k == "abs":
        x = _value(b, node.args[0], done, source)
        out = b.op("SFPABS", "|x|", lreg_c=x, instr_mod1=1)  # 1: the float form
    elif k == "mad":
        out = b.mad(*(_value(b, a, done, source) for a in node.args))
    elif k == "scalar":
        out = b.scalar(node.value)
    elif k == "flash_exp2":
        out = flash_exp2(b, _value(b, node.args[0], done, source))
    elif k == "select" and node.value.kind in INVERSES and not _zero_test(node.value):
        out = _select_inverse(b, node, done, source)
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


def _zero_test(cond):
    """Whether `cond` is == or != against the constant +0.0 (an SFPSETCC)."""
    y = cond.b
    return (
        cond.kind in ZERO_TESTS
        and y.kind == "const"
        and struct.pack("<f", y.value) == bytes(4)
    )


def _select_inverse(b, node, done, source):
    """A select whose test is the inverse of SFPGT's: the value taken where the
    test holds first, the other written over it in the lanes of each test of
    INVERSES (x >= y: where x < y; x == y: where x > y, then where x < y)."""
    cond = node.value
    then, other = (_value(b, a, done, source) for a in node.args)
    first, over = (other, then) if cond.kind == "ne" else (then, other)
    out = b.op("SFPMOV", f"select: x {cond.kind} y", lreg_c=first)
    for kind in INVERSES[cond.kind]:
        _when(b, sfpu.Cond(kind, cond.a, cond.b), done, source)
        b.op("SFPMOV", f"select: where x {kind} y", lreg_c=over, lreg_dest=out)
        b.end()
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


def constant_registers(code, prog, registers=None):
    """Layout: the constants the row code loads (SFPLOADI) move into the free
    programmable registers, most loader words saved first; equal constants share
    one. `registers` 4: LReg 11 too, when no code reads its -1.0 (the body sets it
    and restores it); None: the programmable ones. Returns what it did."""
    if registers is None or any("neg_one" in x.reads() for x in code):
        registers = min(registers or len(PROGRAMMABLE), len(PROGRAMMABLE))
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
        if any(x.args.get("lreg_dest") == value for x in code):
            continue  # read as VD (SFPGT's left side): VD 12 .. 14 stores a template
        bits = _bits_of(words)
        if bits is None:
            continue
        by_bits.setdefault(bits, []).append(value)
    saved = {
        bits: sum(len(found[v]) for v in values) for bits, values in by_bits.items()
    }
    done = []
    for bits in sorted(saved, key=lambda b: -saved[b]):
        if len(prog) >= registers:
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


def resident_constants(code, prog, loads=(), pinned=()):
    """Layout: constants the row still loads move into work registers that stay set
    across the rows, most words saved first, while the row's values still fit; equal
    constants share one; `loads` and `pinned` are held already (run-time scalars).
    Returns (what it did, the loads, the pinned values)."""
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
    done, loads, pinned = [], list(loads), list(pinned)
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


def scalar_loads(ir, chain, body):
    """The run-time scalars' loads (each a float handler word, `<name>_bits` of the
    body's call): two SFPLOADI a scalar whose immediates the core builds from the
    argument, before the rows; refused for a reciprocal or an integer size."""
    loads, names = [], []
    for index, scalar in enumerate(chain.values):
        value = scalar.value
        if getattr(value, "kind", None) != "float":
            fail(f"{ir.name}: a run-time scalar of the unit program is a float word")
        if scalar.recip:
            fail(
                f"{ir.name}: a division by the run-time scalar {value.name}: the unit "
                f"program multiplies; pass its reciprocal word and multiply by it"
            )
        name = f"{value.name}_bits"
        names.append(name)
        held = body.scalars.get(index)
        if held is None:
            continue
        for mod, text in ((2, f"{name} & 0xFFFFu"), (8, f"{name} >> 16")):
            insn = Insn(
                "SFPLOADI",
                {"lreg_ind": held, "instr_mod0": mod, "imm16": 0},
                f"{value.name}, the call's word",
            )
            insn.runtime = text
            loads.append(insn)
    return loads, names


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


def hoist(code, setup, saved=None):
    """Schedule: the constant registers' setup (SFPLOADI into L0, SFPCONFIG) moves out
    of the rows into the setup, once per stream; LReg 11's to the code's start (it is
    restored at each unit's end), or with `saved` (a dict) to saved["head"], its
    restore to saved["tail"]."""
    done, front = [], []
    i = len(code) - 1
    while i >= 0:
        if code[i].op == "SFPCONFIG" and code[i].args["config_dest"] == SAVED_NAME:
            j = i
            while j > 0 and code[j - 1].op == "SFPLOADI":
                if code[j - 1].args["lreg_ind"] != "L0":
                    break
                j -= 1
            front[:0] = code[j : i + 1]
            del code[j : i + 1]
            i = j - 1
            continue
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
    if front and saved is not None:
        saved["head"], saved["tail"] = front, restore_saved()
        done.append("LReg 11: set before the rows, -1.0 again after them")
    elif front:
        code[:0] = front
        code += restore_saved()
        done.append("LReg 11: set first in the unit, -1.0 again at its end")
    return list(reversed(done))


def restore_saved():
    """LReg 11 back to -1.0 (tt-metal's `_restore_lconst_neg1_`), then the SFPNOP an
    SFPCONFIG of a constant register takes."""
    return [
        Insn(
            "SFPLOADI",
            {"lreg_ind": "L0", "instr_mod0": 0, "imm16": NEG_ONE_BF16},
            "-1.0 into LReg 11 again",
        ),
        Insn(
            "SFPCONFIG",
            {"imm16_math": 0x5555, "config_dest": SAVED, "instr_mod1": 8},
            "LReg 11 restored",
        ),
        Insn("SFPNOP", {}, "hazard config-nop: after SFPCONFIG"),
    ]


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


def allocate(code, prog, pinned=(), high=()):
    """Layout: each value an LReg 0 .. 7 for its live range (an operand's that dies
    there, else the lowest free; `high` values the highest); fixed constants 9 / 10 /
    11, the programmable 12 up, then 11. `pinned` values live to the end of `code`; a
    register the code names ("L<n>") is not given to a value live while it is."""
    regs = {name: FIXED[name] for name in FIXED}
    regs.update({name: CONSTANT_REGISTERS[i] for i, name in enumerate(prog)})
    last = {}
    for i, insn in enumerate(code):
        for v in insn.reads() + insn.writes():
            if isinstance(v, str) and v not in regs and v[0] != "L":
                last[v] = len(code) if v in pinned else i
    named = _named_spans(code)

    def fits(r, i, v):
        end = last.get(v, i)
        return all(e < i or s > end for s, e in named.get(r, ()))

    free = list(range(WORK))
    for i, insn in enumerate(code):
        dying = [
            v
            for v in insn.reads()
            if isinstance(v, str) and last.get(v) == i and v not in insn.writes()
        ]
        free += [regs[v] for v in dict.fromkeys(dying) if regs.get(v, WORK) < WORK]
        for v in dict.fromkeys(insn.writes()):
            if isinstance(v, str) and v[0] == "L":
                continue
            if isinstance(v, str) and v not in regs:
                taken = [regs[d] for d in dying if regs.get(d, WORK) < WORK]
                taken = [r for r in taken if r in free and fits(r, i, v)]
                fit = sorted(r for r in free if fits(r, i, v))
                if not taken and not fit:
                    raise UnitError(
                        "komorebi.dsl.tile: more live values than the 8 work registers"
                    )
                if v in high:
                    regs[v] = fit[-1] if fit else taken[0]
                else:
                    regs[v] = taken[0] if taken else fit[0]
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


def _order(items):
    for item in items:
        if isinstance(item, Loop):
            yield from _order(item.body)
        elif isinstance(item, Insn):
            yield item


def products_of(ir, shapes, nodes):
    """The tile products (`a @ w`) of a stream kernel, each refused unless the unpack
    can feed it: both operands a 32 x 32 tile the unpacker reads from a buffer (a
    load or a resident view, not a value the vector unit computed)."""
    out = []
    for n in nodes:
        if n.kind != "matmul":
            continue
        for side, arg in zip(("left", "right"), n.args):
            if arg.kind not in ("load", "resident"):
                made = (
                    "a product, its result in Dst"
                    if arg.kind == "matmul"
                    else f"a {arg.kind}, computed on the vector unit"
                )
                fail(
                    f"{ir.name}: a product after vector code: the {side} operand of "
                    f"`@` is {made}, and the matrix unit reads its operands unpacked "
                    "from a buffer into SrcA and SrcB; give that value to a second "
                    "body (pack, then unpack)",
                    n.line,
                )
            if shapes(arg) != "tile":
                fail(
                    f"{ir.name}: the unpack feeds a product one 32 x 32 tile a side "
                    f"(x.tiles[t]); the {side} operand of `@` is a row block",
                    n.line,
                )
        out.append(n)
    return out


def stream_chain(ir):
    """The Chain of a stream kernel's one store (tile in, tile out), as the stream
    form plans it: each load is a register operand, and so is each tile product (its
    result in a Dst tile of its own, `products_of`)."""
    if len(ir.stores) != 1:
        fail(f"{ir.name}: the unit program takes one store")
    return _stream_chains(ir)[0]


def stream_results(ir):
    """The Chains of a stream kernel's stores, each with the map of its operand k to
    the kernel's operand (one list over the stores: an operand two stores read is
    one), and that list. Result k is stored into Dst tile k, over operand k."""
    if len(ir.stores) > OPERANDS:
        fail(
            f"{ir.name}: {len(ir.stores)} results; a call holds {OPERANDS} Dst tiles "
            "(half of the fp32 Dst) and stores result k into tile k"
        )
    operands, keys, out = [], {}, []
    for chain in _stream_chains(ir):
        if chain.values or chain.prev is not None:
            fail(
                f"{ir.name}: a function of several results takes tiles and numbers "
                f"(no run-time scalar), at most {OPERANDS} tiles"
            )
        index = []
        for leaf in chain.operands:
            if leaf.kind != "copy":
                fail(f"{ir.name}: a function of several results reads whole tiles")
            key = id(leaf.source)
            if key not in keys:
                keys[key] = len(operands)
                operands.append(leaf)
            index.append(keys[key])
        out.append((chain, index))
    if len(operands) > OPERANDS:
        fail(f"{ir.name}: the unit program reads at most {OPERANDS} operand tiles")
    return out, operands


def _stream_chains(ir):
    """One Chain a store of a stream kernel (one compiler: one templates record)."""
    shapes = chains._Shapes(False)
    nodes = ir.nodes()
    products = {id(n): n for n in products_of(ir, shapes, nodes)}  # refused first
    for n in nodes:
        shapes(n)
    loads = {id(n): n for n in nodes if n.kind in ("load", "resident")}

    def leaf_of(node, offset, kind):
        if id(node) in products and kind == "copy" and not offset:
            return chains.Leaf(node, kind, offset)
        if kind == "bcast_row" and node.kind == "load" and not offset:
            # a per-column operand: the caller's unpack copies row 0 to every row,
            # and the unit reads it as a whole tile
            return chains.Leaf(node, kind, offset)
        if id(node) not in loads or kind not in ("copy", "bcast_col") or offset:
            return None
        return chains.Leaf(node, kind, offset)

    compiler = chains._Compiler(ir.name, shapes, {})
    return [
        compiler.chain(store.value, leaf_of, stops=set(products)) for store in ir.stores
    ]


def _dst_base(index):
    """The RISC-V's write of the Dst base: tile `index` (C++ text) of the caller's."""
    return Call(
        "ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, "
        f"UnpackDestination::SrcRegs>({index})",
        (
            "Dst base of tile dst (RISC-V writes)"
            if index == "dst"
            else f"Dst base of tile {index} (RISC-V writes)"
        ),
    )


def _products(unit, leaves, fidelity):
    """The math CPU's lines of the unit's tile products (`leaves`: (Dst tile, the
    matmul node)), in operand order, each the no-MOP product's words after the Dst
    base of its tile; records unit.products and the unpack CPU's calls."""
    lines = []
    for i, (k, node) in enumerate(leaves):
        a, w = (arg.attrs["view"].tensor.name for arg in node.args)
        pairs = tile_products(fidelity, 1, 1, k)
        unit.products.append(UnitProduct(k, a, w))
        lines += [
            _dst_base("dst" if k == 0 else f"dst + {k}"),
            Product(
                f"{a} @ {w} into Dst tile dst + {k} at {fidelity}",
                [x for x, _ in pairs],
                "1 cycle a word; a block's two words 8 apart",
                synced=i == len(leaves) - 1,  # the vector unit's stall waits for it
                insns=[insn for _, insn in pairs],
            ),
        ]
        unit.tile["unpack"].append(
            Call(
                f"llk_unpack_AB_matmul({a}, {w}, 0, 0, 1, 1, 1)",
                f"{a} into SrcB, {w} into SrcA (the product of Dst tile {k})",
            )
        )
    unit.fidelity = fidelity
    unit.init["unpack"].append(
        Call(
            f"mm_no_mop_init_short({leaves[0][1].args[0].attrs['view'].tensor.name}, "
            f"{leaves[0][1].args[1].attrs['view'].tensor.name}, false, 1, 1, 1)",
            f"the products' unpack and address modes, at math_fidelity {fidelity}",
        )
    )
    return lines


def _product_alone(ir, node, fidelity, band):
    """The unit program of a tile function that is one product (`a @ w` stored): the
    no-MOP product's words alone, as `matrix.product_program` emits them."""
    if band:
        fail(f"{ir.name}: the band runs vector code; a product alone runs whole tiles")
    unit = UnitProgram(ir.name, "fp32")
    unit.operands = [None]
    unit.tile["unpack"] = [
        Sync("unpack", "reader", "cb_wait_front(in, 1)", "the operand tiles")
    ]
    lines = _products(unit, [(0, node)], fidelity)
    unit.tile["math"] = [
        Sync("math", "pack", "tile_regs_acquire()", "a free Dst half"),
        *lines,
        Sync("math", "pack", "tile_regs_commit()", "hands Dst to the pack CPU"),
    ]
    unit.tile["pack"] = [
        Sync("pack", "math", "tile_regs_wait()", "Dst from the math CPU"),
        Call("pack_tile(dst, out)", "Dst tile dst to the output buffer"),
        Sync("pack", "math", "tile_regs_release()", "frees (and clears) the Dst half"),
        Sync("pack", "writer", "cb_push_back(out, 1)", "an output tile"),
    ]
    words = len(unit.tile["math"][2].words)
    unit.passes.append(
        Pass("schedule", "products", [f"1 product at {fidelity}: {words} words"])
    )
    return unit


def _templates(chosen):
    """The templates record: each transcendental's chosen form and its bound."""
    return [f"{k}: {v}" for k, v in chosen.items()]


# Address mode the row's last store steps Dst with (rule store_step), when no load
# macro takes ADDR_MOD_6 (`MACRO_ADDR_MOD`, Dst + 2 after each macro).
STEP_ADDR_MOD = MACRO_ADDR_MOD
BLOCKS = FACES * ROWS // 2  # 4-row blocks of a tile (the per-row form's rows)


def per_row_prepass(code, addresses):
    """Rule per_row_prepass: the values of row code `code` made from the per-row
    operands' loads (at `addresses`) and constants alone, computed once a block in a
    pass over the operand tile first, stored over it (in all its lanes). Returns (the
    pass's code or None, its report); `code` then loads the value."""
    loads = [
        x for x in code if x.op == "SFPLOAD" and x.args["dest_reg_addr"] in addresses
    ]
    consts = {
        v
        for x in code
        if x.op == "SFPLOADI"
        for v in x.writes()
        if all(w.op == "SFPLOADI" for w in code if v in w.writes())
    }
    fixed = set(FIXED)
    only = {x.writes()[0]: x for x in loads}  # value -> its load
    part = []
    for x in code:
        if x in loads or x.op in ("SFPLOAD", "SFPSTORE", "SFPLOADI") or x.sets_flags():
            continue
        reads = set(x.reads())
        fits = reads & set(only) and reads <= set(only) | consts | fixed
        if fits and len(x.writes()) == 1:
            only[x.writes()[0]] = x
            part.append(x)
    if not part:
        return None, ["nothing: no value of the per-row operands alone"]
    computed = {x.writes()[0] for x in part}
    rest = [x for x in code if x not in part and x not in loads]
    frontier = [v for v in computed if any(v in x.reads() for x in rest)]
    if len(loads) != 1 or len(frontier) != 1:
        return None, [
            "refused: one per-row operand and one value of it read by the rows "
            f"({len(loads)} operand(s), {len(frontier)} value(s))"
        ]
    load, value = loads[0], frontier[0]
    if any(load.writes()[0] in x.reads() for x in rest):
        return None, ["refused: the rows read the operand itself too"]
    needed = {v for x in part for v in x.reads()} & consts
    setup = [x for x in code if x.op == "SFPLOADI" and set(x.writes()) & needed]
    for x in part:
        code.remove(x)
    for x in setup:
        if not any(set(x.writes()) & set(y.reads()) for y in code if y not in setup):
            code.remove(x)
    operand = dict(load.args)
    load.args["lreg_ind"] = value
    load.note = f"{load.note}: the per-row pass's value"
    address = load.args["dest_reg_addr"]
    pre = [
        *[copy.deepcopy(x) for x in setup],
        Insn("SFPLOAD", operand, "per-row operand"),
        *part,
        Insn(
            "SFPSTORE",
            {
                "lreg_ind": value,
                "instr_mod0": operand["instr_mod0"],
                "sfpu_addr_mode": ADDR_MOD,
                "dest_reg_addr": address,
            },
            "the per-row value over the operand",
        ),
    ]
    return pre, [f"{len(part)} instruction(s) and {len(setup)} load(s) a block, once"]


def _results(body, results):
    """Each result's value from its chain (operand k loaded once a row, from Dst tile
    k), then result k stored into Dst tile k: every load of a row before its stores."""
    loaded = {}

    def source(k):
        if k not in loaded:
            loaded[k] = body.load(TILE * k, f"operand {k}")
        return loaded[k]

    values = [
        _value(body, chain.fn.result, {}, lambda i, index=index: source(index[i]))
        for chain, index in results
    ]
    for k, value in enumerate(values):
        body.store(value, TILE * k, f"result {k} to Dst tile {k}")


def store_step(row, incr):
    """Rule store_step: the row's closing INCRWC (Dst + `incr`) folded into its last
    Dst access when that is a store: the store takes STEP_ADDR_MOD (Dst + `incr`).
    Returns whether it did."""
    if not row or row[-1].op != "INCRWC" or row[-1].args["rwc_d"] != incr:
        return False
    touching = [x for x in row if "dest_reg_addr" in x.args]
    if not touching or touching[-1].op != "SFPSTORE":
        return False
    touching[-1].args["sfpu_addr_mode"] = STEP_ADDR_MOD
    touching[-1].note = f"{touching[-1].note}, then Dst + {incr}"
    del row[-1]
    return True


def _step_mode(incr):
    return Call(
        "ckernel::addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, "
        f".dest = {{.incr = {incr}}}}}.set(ADDR_MOD_{STEP_ADDR_MOD})",
        f"address mode {STEP_ADDR_MOD}: Dst + {incr} after the row's store (3 SETC16)",
    )


def unit_program(
    ir, dest="fp32", load_macro=False, rows=False, fidelity="HiFi4", tiles=None
):
    """The unit program of stream kernel `ir` on `dest` Dst (DEST_MODES): operand k (a
    load, or a product `a @ w` at `fidelity`) in Dst tile k of `dst` (or `tiles[name]`,
    its parameter's: sums apart in Dst), the result in tile 0. `load_macro`: try the
    load-macro rule (a refusal is a pass's record); `rows`: live rows a run-time word.
    """
    if dest not in DEST_MODES:
        raise PlanError(f"komorebi.dsl.tile: no unit program for a {dest} Dst")
    if fidelity not in PHASES:
        fail(f"{ir.name}: no fidelity {fidelity!r}: one of {', '.join(PHASES)}")
    band = bool(rows)
    several = len(ir.stores) > 1
    if several:
        results, operands = stream_results(ir)
        chain = results[0][0]
        if tiles:
            fail(f"{ir.name}: several results are stored over operands 0 .. n - 1")
    else:
        chain = stream_chain(ir)
        operands = chain.operands
    root = ir.stores[0].value
    if root.kind == "matmul" and not several:
        return _product_alone(ir, root, fidelity, band)
    if chain.prev is not None:  # more blocks than the chain's slots: run as stages
        fail(f"{ir.name}: the unit program reads at most {OPERANDS} operand tiles")
    leaves = [
        (k, leaf.source)
        for k, leaf in enumerate(operands)
        if leaf.source.kind == "matmul"
    ]
    if len(leaves) > OPERANDS:
        fail(
            f"{ir.name}: {len(leaves)} products: Dst holds {OPERANDS} operand tiles "
            "of a body (a product's result stays in Dst for the vector code)"
        )
    if len(operands) > OPERANDS:
        fail(f"{ir.name}: the unit program reads at most {OPERANDS} operand tiles")
    dtypes = {t.dtype for t in ir.tensors}
    read = {t.dtype for t in ir.tensors if not t.output}
    if dest == "fp16a" and not read <= set(A_DTYPES):
        fail(f"{ir.name}: a 16-bit Dst's tensors are of the A family")
    if dest == "fp32" and not dtypes <= set(TILE_DTYPES):
        fail(
            f"{ir.name}: tensors are fp32 or of the B family "
            f"({', '.join(TILE_DTYPES[1:])})"
        )
    body = Body()
    body.mode = DEST_MODES[dest]
    if tiles:  # operand k's Dst tile: its parameter's
        body.tiles = [tiles[leaf.source.attrs["view"].tensor.name] for leaf in operands]
    per_row = [k for k, leaf in enumerate(operands) if leaf.kind == "bcast_col"]
    if per_row and tiles:
        fail(f"{ir.name}: a per-row operand is read from Dst tile k, not apart")
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
    elif several:
        _results(body, results)
    else:
        result = _value(body, chain.fn.result, {})
        body.store(result)
    unit = UnitProgram(ir.name, dest)
    unit.operands = [
        None if leaf.source.kind == "matmul" else leaf.source.attrs["view"].tensor.name
        for leaf in operands
    ]
    unit.results = len(ir.stores)
    unit.row_operands = [
        k for k, leaf in enumerate(operands) if leaf.kind == "bcast_row"
    ]
    unit.col_operands = per_row
    setup = []
    unit.passes.append(Pass("algebra", "templates", _templates(chain.templates)))
    unit.passes.append(Pass("algebra", "fold_negation", fold_negation(body.code)))
    unit.passes.append(Pass("algebra", "fuse_mad", fuse_mad(body.code)))
    pre = None
    if several:  # a value two results compute from the same operands: computed once
        unit.passes.append(Pass("algebra", "common_values", common_values(body.code)))
    if per_row:
        # The even and odd vectors of a block are two independent chains.
        unit.passes.append(Pass("algebra", "drop_copies", drop_copies(body.code)))
        unit.passes.append(Pass("algebra", "common_values", common_values(body.code)))
        trial = copy.deepcopy(body.code)
        pre, done = per_row_prepass(trial, {TILE * k for k in per_row})
        if pre is not None:
            pre_unit = UnitProgram("estimate", dest)
            pre_unit.tile["math"] = _prepass_loop(pre, {})
            spent = pre_unit.cost("math")[1]
            saved = BLOCKS * (len(body.code) - len(trial))
            done.append(
                f"{saved} words a tile saved against {spent:.1f} cycles of the pass"
            )
            if spent < saved:
                body.code[:] = trial
            else:
                pre = None
                done.append("kept in the rows (cheaper)")
        unit.passes.append(Pass("algebra", "per_row_prepass", done))
        moved = interleave(body.code, 0, len(body.code))
        done = [f"{moved} instructions moved in the even and odd chains"]
        unit.passes.append(Pass("schedule", "interleave", done))
    scalars, unit.scalars = scalar_loads(ir, chain, body)
    unit.passes.append(Pass("algebra", "negated_zero", negated_zero(body.code)))
    done = constant_registers(body.code, body.prog, len(CONSTANT_REGISTERS))
    unit.passes.append(Pass("layout", "constant_registers", done))
    saved = {"head": [], "tail": []}
    unit.passes.append(Pass("schedule", "hoist", hoist(body.code, setup, saved)))
    allocate(saved["head"] + saved["tail"], body.prog)
    done, held, pinned = resident_constants(
        body.code, body.prog, scalars, [x.args["lreg_ind"] for x in scalars]
    )
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
        CONSTANT_REGISTERS[i]: value for i, value in enumerate(body.prog.values())
    }
    plan = None
    if load_macro and dest != "fp32":  # the macro's load and store: fp32 Dst's
        unit.passes.append(Pass("schedule", "load_macro", [f"refused: {dest} Dst"]))
    elif load_macro and not per_row:
        try:
            plan = macro.fit(body.code)
            if ROWS % plan.rows:
                raise macro.Refused(
                    f"{plan.rows} rows a macro turn do not divide a face"
                )
            if band and (ROWS // plan.rows) % BANDS:
                raise macro.Refused(
                    f"a macro turn of {plan.rows} row steps; a band is 2"
                )
        except macro.Refused as e:
            plan = None
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
    if plan is None:
        incr = 4 if per_row else 2
        note = "next 4-row block" if per_row else "next 4 rows"
        row = [*body.code, Insn("INCRWC", _rwc(rwc_d=incr), note)]
        stepped = store_step(row, incr)
        if stepped:
            unit.init["math"].append(_step_mode(incr))
        done = [f"the INCRWC into the last store (Dst + {incr})"] if stepped else []
        unit.passes.append(Pass("schedule", "store_step", done))
        face = Loop("row", ROWS // 2 if per_row else ROWS, row, unroll=True)
        drained = []
    else:
        row = macro.rows(plan, MACRO_ADDR_MOD)
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
    copies = [
        f"copy_tile(in{k}, 0, dst + {k})"
        for k, name in enumerate(unit.operands)
        if name is not None
    ]
    unit.tile["unpack"] = [
        Sync("unpack", "reader", "cb_wait_front(in, 1)", "the input tiles")
    ]
    multiplied = _products(unit, leaves, fidelity) if leaves else []
    unit.tile["unpack"] += [
        *[Call(c, "the tile into Dst (fp32, unpacked to Dst)") for c in copies],
        Call("cb_pop_front(in, 1)", "frees the input tiles"),
    ]
    if leaves:
        words = sum(len(x.words) for x in multiplied if isinstance(x, Product))
        unit.passes.append(
            Pass(
                "schedule",
                "products",
                [
                    f"{len(leaves)} product(s) at {fidelity} before the vector code "
                    f"({words} matrix words), the vector unit's stall on the matrix "
                    "unit after them"
                ],
            )
        )
    unit.tile["math"] = [
        Sync("math", "pack", "tile_regs_acquire()", "a free Dst half"),
        *multiplied,
        *[Call(c, "the math part of the copy (none to Dst)") for c in copies],
        _dst_base("dst"),
        Insn(
            "STALLWAIT",
            {"stall_res": STALL_SFPU, "wait_res": WAIT_MATH},
            "SFPU waits for the matrix unit" + (" (the products)" if leaves else ""),
        ),
        *saved["head"],
        *([] if pre is None else _prepass_loop(pre, body.prog, unit)),
        *held,
        Loop("face", FACES, [face, step, Insn(step.op, dict(step.args), step.note)]),
        *drained,
        *saved["tail"],
        Insn("SETRWC", {**_setrwc(), "BitMask": SET_D}, "Dst counter back to 0"),
        Sync("math", "pack", "tile_regs_commit()", "hands Dst to the pack CPU"),
    ]
    unit.tile["pack"] = [
        Sync("pack", "math", "tile_regs_wait()", "Dst from the math CPU"),
        Call("pack_tile(dst, out)", "Dst tile dst to the output buffer"),
        Sync("pack", "math", "tile_regs_release()", "frees the Dst half"),
        Sync("pack", "writer", "cb_push_back(out, 1)", "an output tile"),
    ]
    if several:
        unit.tile["pack"][1:2] = [
            Call(f"pack_tile(dst + {k}, out{k})", f"result {k} to its output buffer")
            for k in range(unit.results)
        ]
    _checks(unit, row, plan)
    rows, done = replay(face)
    unit.passes.append(Pass("schedule", "replay", done))
    faces = next(x for x in unit.tile["math"] if isinstance(x, Loop))
    if band:
        at = unit.tile["math"].index(faces)
        unit.tile["math"][at : at + 1] = band_faces(face, faces.body[1:])
        unit.band = True
        done = [
            f"{face.count // BANDS} pass(es) of the face's rows a band of 4 live "
            "rows, faces 2 and 3 past 16 rows"
        ]
        if pre is not None:
            done.append("the per-row pass runs the whole tile")
        if leaves:
            done.append("the products run the whole tile")
        unit.passes.append(Pass("schedule", "band", done))
    else:
        faces.body[:1] = rows
    unit.not_checked = [r for r in NOT_CHECKED if plan is None or "load-macro" not in r]
    return unit


# The band's C++ (`rows`: the body's argument, 1 .. 32; 0 or past 32: the whole tile):
# bands in faces 0 and 1, whether faces 2 and 3 run, and the bands in them.
LOWER = "(rows - 1u < 16u ? (rows + 3u) / 4u : 4u)"
UPPER_FACES = "(rows - 1u >= 16u ? 2u : 0u)"
UPPER = "(rows - 17u < 16u ? (rows - 13u) / 4u : 4u)"


def _live(rows):
    return rows if 1 <= rows <= 32 else 32


def _lower(rows):
    return min(4, (_live(rows) + 3) // 4)


def _upper(rows):
    return max(0, (_live(rows) - 13) // 4)


def band_faces(face, steps):
    """Rule band: the face loop (`face`: a face's row loop, `steps`: the two SETRWC
    to the next face) on the live rows: faces 0, 1, then 2, 3 past 16 rows, each a
    recorded pass and a runtime loop of its other bands, replayed; each face's Dst set
    from its base by the SETRWC, whatever rows the face before ran."""
    per_band = face.count // BANDS
    row = face.body
    n = min(REPLAY_ENTRIES, len(row))
    record = Insn("REPLAY", _replay(n, 1), f"record {n} while they run")
    play = [Insn("REPLAY", _replay(n, 0), f"replay {n}"), *row[n:]]

    def one(text, of):
        more = Loop(
            "band",
            Runtime(f"{text} - 1u", lambda r: of(r) - 1),
            play * per_band,
        )
        return [record, *row, *play * (per_band - 1), more, *steps]

    return [
        Loop("face", 2, one(LOWER, _lower)),
        Loop(
            "face",
            Runtime(UPPER_FACES, lambda r: 2 if _live(r) > 16 else 0),
            one(UPPER, _upper),
        ),
    ]


def _rwc(**args):
    return {"rwc_cr": 0, "rwc_d": 0, "rwc_b": 0, "rwc_a": 0, **args}


def _setrwc():
    return {"clear_ab_vld": 0, **_rwc(), "BitMask": 0}


def _prepass_loop(pre, prog, unit=None):
    """The per-row pass over the operand tile's 16 blocks (allocated, its constants
    loaded once before the walk, the step in its store), then Dst back to 0. With
    `unit`: its init sets the store's address mode and its checks record the pass."""
    code = copy.deepcopy(pre)
    consts = [x for x in code if x.op == "SFPLOADI"]
    allocate(code, prog, pinned={v for x in consts for v in x.writes()})
    walk = [x for x in code if x.op != "SFPLOADI"]
    walk.append(Insn("INCRWC", _rwc(rwc_d=4), "next 4-row block"))
    store_step(walk, 4)
    if unit is not None:
        unit.passes.append(
            Pass(
                "schedule", "per_row_prepass", [f"{len(walk)} words a block, 16 blocks"]
            )
        )
        n = check_mad_readers([walk])
        unit.checks.append(
            Check(
                "mad-reader (per-row pass)",
                "SFPNOP between a MAD and a reader not held back",
                f"repaired: {n} SFPNOP" if n else "ok",
            )
        )
        if not any(
            isinstance(x, Call) and "dest = {.incr = 4}" in x.text
            for x in unit.init["math"]
        ):
            unit.init["math"].append(_step_mode(4))
    else:
        check_mad_readers([walk])
    step = Insn(
        "SETRWC",
        {**_setrwc(), "rwc_cr": CR_D, "rwc_d": 8, "BitMask": SET_D},
        "Dst + 8 rows",
    )
    return [
        *consts,
        Loop(
            "face",
            FACES,
            [
                Loop("row", ROWS // 2, walk, unroll=True),
                step,
                Insn(step.op, dict(step.args), step.note),
            ],
        ),
        Insn("SETRWC", {**_setrwc(), "BitMask": SET_D}, "Dst counter back to 0"),
    ]
