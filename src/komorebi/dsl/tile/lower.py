"""A traced tile expression over a stream of tiles to its unit program: the chain
compiler's vector function (`chain.Chain`) is written as vector-unit instructions on
named values, the rules run as passes (algebra, schedule, layout), then the hazard
checks. Built: the fp32-Dst SiLU chain (operands, constants, +, *, -, sigmoid)."""

import struct

from komorebi.dsl.tile import chain as chains
from komorebi.dsl.tile.ir import BF16, F32, PlanError, fail
from komorebi.dsl.tile.unit import (
    FIXED,
    IADD_IMM,
    IADD_NO_CC,
    OPS,
    PROGRAMMABLE,
    REPLAY_ENTRIES,
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

# Dst access through ADDR_MOD_7 (no increments), the format the registers hold.
ADDR_MOD = 7
ROWS, FACES = 8, 4
# The readers of a MAD-class result that the unit does not hold back (rule
# mad-reader; SFPGT and SFPLE added from tt-metal's ckernel_sfpu_recip.h).
NOT_HELD = ("SFPIADD", "SFPSHFT", "SFPAND", "SFPOR", "SFPSWAP", "SFPGT", "SFPLE")
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
)


class Body:
    """The instructions of one row group (4 rows x 8 columns of each face, the
    vector unit's 32 lanes) on named values; `self.code` in issue order."""

    def __init__(self):
        self.code, self.count, self.depth = [], 0, 0
        self.prog = {}

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

    def load(self):
        return self.op("SFPLOAD", "x from Dst", instr_mod0=0, sfpu_addr_mode=ADDR_MOD)

    def store(self, value):
        self.op(
            "SFPSTORE",
            "the result to Dst",
            lreg_ind=value,
            sfpu_addr_mode=ADDR_MOD,
        )

    def const(self, value, note=""):
        """`value` (fp32) into a new value: one SFPLOADI when bf16 or fp16 holds it
        exactly, else the low then the high half."""
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        out = self.fresh()
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

    def register_const(self, value):
        """A programmable constant register holding `value` (set once by the setup,
        after the schedule pass hoists it): its name until the layout pass."""
        name = f"p{len(self.prog)}"
        self.prog[name] = value
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        if bits & 0xFFFF:
            raise UnitError(f"a constant register value {value!r} is not bf16")
        self.op("SFPLOADI", f"{value!r}", lreg_ind="L0", instr_mod0=0, imm16=bits >> 16)
        self.op("SFPCONFIG", f"constant register {name}", config_dest=name)
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


FUNCTIONS = {"sigmoid": sigmoid}


def _value(b, node, done):
    """The value of sfpu node `node` (an operand, a constant, an operation)."""
    key = id(node)
    if key in done:
        return done[key]
    k = node.kind
    if k == "operand":
        if node.value != (0, None) or node.along:
            fail("the unit program reads one whole-tile operand")
        out = b.load()
    elif k == "const":
        out = b.const(node.value)
    elif k == "call":
        if node.value not in FUNCTIONS:
            fail(f"vector function {node.value} is not built in the unit program")
        out = FUNCTIONS[node.value](b, _value(b, node.args[0], done))
    elif k in ("mul", "add", "sub", "neg"):
        args = [_value(b, a, done) for a in node.args]
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
        i += 1
    return done


def hoist(code, setup):
    """Schedule: the constant registers' setup (SFPLOADI into L0, SFPCONFIG) moves
    out of the per-row code into the setup, once per stream."""
    done = []
    for i in range(len(code) - 1, -1, -1):
        if code[i].op == "SFPCONFIG":
            setup[:0] = [code[i - 1], code[i]]
            done.append(f"constant register {code[i].args['config_dest']}")
            del code[i - 1 : i + 1]
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


def allocate(code, prog):
    """Layout: each value gets an LReg 0 .. 7 for its live range (a result takes the
    register of an operand that dies there, else the lowest free one); the fixed
    constants take 9 / 10 / 11, the programmable ones 12 upward."""
    regs = {name: FIXED[name] for name in FIXED}
    regs.update({name: PROGRAMMABLE[i] for i, name in enumerate(prog)})
    last = {}
    for i, insn in enumerate(code):
        for v in insn.reads() + insn.writes():
            if isinstance(v, str) and v not in regs:
                last[v] = i
    free = list(range(WORK))
    for i, insn in enumerate(code):
        dying = [
            v
            for v in insn.reads()
            if isinstance(v, str) and last.get(v) == i and v not in insn.writes()
        ]
        free += [regs[v] for v in dict.fromkeys(dying) if regs.get(v, WORK) < WORK]
        for v in insn.writes():
            if isinstance(v, str) and v not in regs:
                taken = [regs[d] for d in dying if regs.get(d, WORK) < WORK]
                regs[v] = taken[0] if taken else min(free)
                free.remove(regs[v])
                if last.get(v, i) == i:
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
            if (
                OPS[prev.op].kind == "mad"
                and cur.op in NOT_HELD
                and set(prev.writes()) & set(cur.reads())
            ):
                code.insert(i, Insn("SFPNOP", {}, "hazard mad-reader: MAD result"))
                repairs += 1
            i += 1
    return repairs


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


def _checks(unit, row):
    """The hazard checks of the math CPU's program (`row`: the per-row loop body)."""
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
    first = next(i for i, x in enumerate(flat) if x.op == "SFPLOAD")
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
        if id(node) not in loads or kind != "copy" or offset is not None:
            return None
        return chains.Leaf(node, kind, offset)

    compiler = chains._Compiler(ir.name, shapes, {})
    return compiler.chain(ir.stores[0].value, leaf_of)


def unit_program(ir, dest="fp32"):
    """The unit program of stream kernel `ir` on `dest` (fp32) Dst registers."""
    if dest != "fp32":
        raise PlanError("komorebi.dsl.tile: the unit program is built for fp32 Dst")
    chain = stream_chain(ir)
    if len(chain.operands) != 1 or chain.values or chain.prev is not None:
        fail(f"{ir.name}: the unit program takes one operand and no scalars")
    dtypes = {t.dtype for t in ir.tensors}
    if not dtypes <= {BF16, F32}:
        fail(f"{ir.name}: tensors are bf16 or fp32")
    body = Body()
    result = _value(body, chain.fn.result, {})
    body.store(result)
    unit = UnitProgram(ir.name, dest)
    setup = []
    unit.passes.append(Pass("algebra", "fold_negation", fold_negation(body.code)))
    unit.passes.append(Pass("schedule", "hoist", hoist(body.code, setup)))
    unit.passes.append(Pass("layout", "allocate", allocate(body.code, body.prog)))
    allocate(setup, body.prog)
    unit.constants = {
        PROGRAMMABLE[i]: value for i, value in enumerate(body.prog.values())
    }
    row = [*body.code, Insn("INCRWC", _rwc(rwc_d=2), "next 4 rows")]
    face = Loop("row", ROWS, row, unroll=True)
    step = Insn(
        "SETRWC",
        {**_setrwc(), "rwc_cr": CR_D, "rwc_d": 8, "BitMask": SET_D},
        "Dst + 8 rows",
    )
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
    unit.tile["unpack"] = [
        Sync("unpack", "reader", "cb_wait_front(in, 1)", "an input tile"),
        Call("copy_tile(in, 0, dst)", "the tile into Dst (fp32, unpacked to Dst)"),
        Call("cb_pop_front(in, 1)", "frees the input tile"),
    ]
    unit.tile["math"] = [
        Sync("math", "pack", "tile_regs_acquire()", "a free Dst half"),
        Call("copy_tile(in, 0, dst)", "the math part of the copy (none to Dst)"),
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
        Loop("face", FACES, [face, step, Insn(step.op, dict(step.args), step.note)]),
        Insn("SETRWC", {**_setrwc(), "BitMask": SET_D}, "Dst counter back to 0"),
        Sync("math", "pack", "tile_regs_commit()", "hands Dst to the pack CPU"),
    ]
    unit.tile["pack"] = [
        Sync("pack", "math", "tile_regs_wait()", "Dst from the math CPU"),
        Call("pack_tile(dst, out)", "Dst tile dst to the output buffer"),
        Sync("pack", "math", "tile_regs_release()", "frees the Dst half"),
        Sync("pack", "writer", "cb_push_back(out, 1)", "an output tile"),
    ]
    _checks(unit, row)
    rows, done = replay(face)
    unit.passes.append(Pass("schedule", "replay", done))
    faces = next(x for x in unit.tile["math"] if isinstance(x, Loop))
    faces.body[:1] = rows
    unit.not_checked = list(NOT_CHECKED)
    return unit


def _rwc(**args):
    return {"rwc_cr": 0, "rwc_d": 0, "rwc_b": 0, "rwc_a": 0, **args}


def _setrwc():
    return {"clear_ab_vld": 0, **_rwc(), "BitMask": 0}
