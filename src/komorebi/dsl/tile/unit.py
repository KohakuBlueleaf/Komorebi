"""The unit program: for each compute CPU (unpack, math, pack) a loop nest of
coprocessor instructions with named registers, the Dst rows they walk and the sync
points between the CPUs, with a cost a tile. Rules are passes over it (`Pass`): kind
"algebra", "layout" or "schedule"; checks of the hazard rules run after them.
"""

from dataclasses import dataclass, field

CPUS = ("unpack", "math", "pack")
KINDS = ("algebra", "layout", "schedule")
# Fixed vector-unit registers: 0.0, 1.0, -1.0; 12 to 14 are set by SFPCONFIG.
FIXED = {"zero": 9, "one": 10, "neg_one": 11}
PROGRAMMABLE = (12, 13, 14)
WORK = 8  # LReg 0 .. 7 hold values
COST_SOURCE = "estimate (an instruction a cycle, a MAD result after 2): not measured"
REPLAY_SOURCE = "estimate (a played instruction a cycle): not measured"
REPLAY_ENTRIES = 32  # entries of one thread's replay buffer


@dataclass(frozen=True)
class Op:
    """One coprocessor instruction: its opcode, its macro's arguments (name, width in
    bits) from the high field down to bit `low`, the register arguments it reads and
    writes, and its class: "mad" (the 2-cycle multiply-add unit), "simple", "load",
    "store", "flags", "rwc", "stall"."""

    code: int
    args: tuple
    reads: tuple = ()
    writes: tuple = ()
    kind: str = "simple"
    low: int = 0


_R = (("imm12_math", 12), ("lreg_c", 4), ("lreg_dest", 4), ("instr_mod1", 4))
_MAD = (
    ("lreg_src_a", 4),
    ("lreg_src_b", 4),
    ("lreg_src_c", 4),
    ("lreg_dest", 4),
    ("instr_mod1", 4),
)
_LS = (("lreg_ind", 4), ("instr_mod0", 4), ("sfpu_addr_mode", 3), ("dest_reg_addr", 13))
OPS = {
    "SFPLOAD": Op(0x70, _LS, (), ("lreg_ind",), "load"),
    "SFPSTORE": Op(0x72, _LS, ("lreg_ind",), (), "store"),
    "SFPLOADI": Op(
        0x71,
        (("lreg_ind", 4), ("instr_mod0", 4), ("imm16", 16)),
        (),
        ("lreg_ind",),
        "load",
    ),
    "SFPMAD": Op(
        0x84, _MAD, ("lreg_src_a", "lreg_src_b", "lreg_src_c"), ("lreg_dest",), "mad"
    ),
    "SFPADD": Op(
        0x85, _MAD, ("lreg_src_a", "lreg_src_b", "lreg_src_c"), ("lreg_dest",), "mad"
    ),
    "SFPMUL": Op(
        0x86, _MAD, ("lreg_src_a", "lreg_src_b", "lreg_src_c"), ("lreg_dest",), "mad"
    ),
    "SFPMULI": Op(
        0x74,
        (("imm16_math", 16), ("lreg_dest", 4), ("instr_mod1", 4)),
        ("lreg_dest",),
        ("lreg_dest",),
        "mad",
    ),
    "SFPMOV": Op(0x7C, _R, ("lreg_c",), ("lreg_dest",)),
    "SFPABS": Op(0x7D, _R, ("lreg_c",), ("lreg_dest",)),
    "SFPEXEXP": Op(0x77, _R, ("lreg_c",), ("lreg_dest",)),
    "SFPARECIP": Op(0x99, _R, ("lreg_c",), ("lreg_dest",)),
    "SFPSETSGN": Op(0x89, _R, ("lreg_c", "lreg_dest"), ("lreg_dest",)),
    "SFPSETEXP": Op(0x82, _R, ("lreg_c", "lreg_dest"), ("lreg_dest",)),
    "SFPIADD": Op(0x79, _R, ("lreg_c", "lreg_dest"), ("lreg_dest",)),
    "SFPGT": Op(0x97, _R, ("lreg_c", "lreg_dest"), (), "flags"),
    "SFPENCC": Op(0x8A, _R, (), (), "flags"),
    "SFPCAST": Op(
        0x90,
        (("lreg_src_c", 4), ("lreg_dest", 4), ("instr_mod1", 4)),
        ("lreg_src_c",),
        ("lreg_dest",),
    ),
    "SFP_STOCH_RND": Op(
        0x8E,
        (
            ("rnd_mode", 3),
            ("imm8_math", 5),
            ("lreg_src_b", 4),
            ("lreg_src_c", 4),
            ("lreg_dest", 4),
            ("instr_mod1", 4),
        ),
        ("lreg_src_b", "lreg_src_c"),
        ("lreg_dest",),
    ),
    "SFPNOP": Op(0x8F, ()),
    "SFPCONFIG": Op(0x91, (("imm16_math", 16), ("config_dest", 4), ("instr_mod1", 4))),
    "INCRWC": Op(
        0x38,
        (("rwc_cr", 6), ("rwc_d", 4), ("rwc_b", 4), ("rwc_a", 4)),
        kind="rwc",
        low=6,
    ),
    "SETRWC": Op(
        0x37,
        (
            ("clear_ab_vld", 2),
            ("rwc_cr", 4),
            ("rwc_d", 4),
            ("rwc_b", 4),
            ("rwc_a", 4),
            ("BitMask", 6),
        ),
        kind="rwc",
    ),
    "REPLAY": Op(
        0x04,
        (
            ("start_idx", 10),
            ("len", 10),
            ("execute_while_loading", 3),
            ("load_mode", 1),
        ),
        kind="replay",
    ),
    "STALLWAIT": Op(0xA2, (("stall_res", 9), ("wait_res", 15)), kind="stall"),
}
# SFPIADD's immediate form reads only lreg_c; SFPIADD mod1 bit 2 set: no flags.
IADD_IMM, IADD_NO_CC = 1, 4
CODES = {op.code: name for name, op in OPS.items()}


def encode(name, args):
    """The 32-bit instruction word of `name` with `args`; a value wider than its
    field raises UnitError (an immediate is never let spill into the opcode)."""
    word, shift = 0, OPS[name].low
    for arg, width in reversed(OPS[name].args):
        value = args[arg]
        if not isinstance(value, int) or not 0 <= value < 1 << width:
            raise UnitError(f"{name} {arg}={value!r} does not fit {width} bits")
        word |= value << shift
        shift += width
    return OPS[name].code << 24 | word


def decode(word):
    """(name, args) of a 32-bit instruction word, or None for an opcode not in OPS."""
    name = CODES.get(word >> 24)
    if name is None:
        return None
    args, shift = {}, OPS[name].low
    for arg, width in reversed(OPS[name].args):
        args[arg] = word >> shift & ((1 << width) - 1)
        shift += width
    return name, dict(reversed(args.items()))


@dataclass(eq=False)
class Insn:
    """Instruction `op` with its macro's arguments `args` (a register argument is a
    value's name until the layout pass, then an LReg number) and what it computes."""

    op: str
    args: dict
    note: str = ""

    def reads(self):
        """The register arguments read (value names or LReg numbers)."""
        o = OPS[self.op]
        names = list(o.reads)
        if self.op == "SFPIADD" and self.args["instr_mod1"] & IADD_IMM:
            names.remove("lreg_dest")
        if self.op == "SFPLOADI" and self.args["instr_mod0"] in (8, 10):
            names.append("lreg_ind")
        if self.op in ("SFPSETSGN", "SFPSETEXP") and self.args["instr_mod1"] & 1:
            names.remove("lreg_dest")
        return [self.args[n] for n in names]

    def writes(self):
        """The register arguments written."""
        return [self.args[n] for n in OPS[self.op].writes]

    def sets_flags(self):
        """True when it changes the lanes' condition flags."""
        if self.op == "SFPIADD":
            return not self.args["instr_mod1"] & IADD_NO_CC
        if self.op == "SFPGT":
            return bool(self.args["instr_mod1"] & 1)
        return self.op == "SFPENCC"


def _arg(value):
    return value if isinstance(value, str) else str(value)


@dataclass(eq=False)
class Call:
    """A line of the library that is not one coprocessor instruction (a RISC-V write
    of a configuration register, a library unit call): its C++ text and what it
    does. Not priced."""

    text: str
    note: str


@dataclass(eq=False)
class Loop:
    """`count` passes over `body`; `unroll`: the C++ loop is unrolled."""

    name: str
    count: int
    body: list
    unroll: bool = False


@dataclass(eq=False)
class Sync:
    """A sync point between CPUs: `cpu` waits for (or hands over to) `other`, through
    the library call `text`."""

    cpu: str
    other: str
    text: str
    note: str


@dataclass(eq=False)
class Pass:
    """A rule applied: its kind (algebra, layout, schedule), name and what it did."""

    kind: str
    name: str
    done: list = field(default_factory=list)


@dataclass(eq=False)
class Check:
    """A hazard rule checked: its id, the rule, and the verdict ("ok", "repaired: ...",
    or "refused: ..."); a refusal raises before emission."""

    rule: str
    text: str
    verdict: str


class UnitError(ValueError):
    """A unit program that breaks a hazard rule the checks cannot repair."""


@dataclass(eq=False)
class UnitProgram:
    """One tile function on one core: per CPU its setup (`init`, once per stream) and
    its program for one tile (`tile`), loop nests of Insn / Call / Loop / Sync."""

    name: str
    dest: str
    init: dict = field(default_factory=lambda: {cpu: [] for cpu in CPUS})
    tile: dict = field(default_factory=lambda: {cpu: [] for cpu in CPUS})
    constants: dict = field(default_factory=dict)
    clobbers: list = field(default_factory=list)
    passes: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    not_checked: list = field(default_factory=list)

    def cost(self, cpu):
        """(instructions, cycles, terms) of one tile on `cpu`: coprocessor
        instructions and their cycles; each term (what, count, cycles, source),
        cycles None for a library call or a sync (not priced)."""
        terms = []
        _price(self.tile[cpu], 1, terms, [])
        priced = [t for t in terms if t[2] is not None]
        count = sum(t[1] for t in priced if t[0] in OPS)
        return count, sum(t[2] for t in priced), terms

    def dump(self):
        """The unit program as text: setup, per-tile loop nests, passes, checks and
        the cost a tile."""
        lines = [f"unit program {self.name} ({self.dest} Dst)"]
        lines += [
            f"  constant LReg{r} = {v!r} (set by the setup)"
            for r, v in self.constants.items()
        ]
        if self.clobbers:
            lines.append(f"  clobbers {', '.join(self.clobbers)}")
        for cpu in CPUS:
            if self.init[cpu]:
                lines.append(f"  {cpu} setup:")
                lines += _dump(self.init[cpu], 4)
            lines.append(f"  {cpu} per tile:")
            lines += _dump(self.tile[cpu], 4)
        for p in self.passes:
            lines.append(f"  pass {p.kind} {p.name}: {'; '.join(p.done) or 'nothing'}")
        lines += [f"  check {c.rule} ({c.text}): {c.verdict}" for c in self.checks]
        lines += [f"  not checked {rule}" for rule in self.not_checked]
        for cpu in CPUS:
            count, cycles, terms = self.cost(cpu)
            lines.append(
                f"  cost {cpu}: {count} instructions, {cycles} cycles a tile, and "
                "the calls not priced"
            )
            lines += [
                f"    {what}: {n} x, {'-' if c is None else c} cycles ({source})"
                for what, n, c, source in _merge(terms)
            ]
        return "\n".join(lines)


def _dump(items, depth):
    pad, out = " " * depth, []
    for item in items:
        if isinstance(item, Loop):
            how = ", unrolled" if item.unroll else ""
            out.append(f"{pad}loop {item.name} < {item.count}{how}:")
            out += _dump(item.body, depth + 2)
        elif isinstance(item, Insn):
            args = " ".join(f"{n}={_arg(item.args[n])}" for n, _ in OPS[item.op].args)
            note = f"  ; {item.note}" if item.note else ""
            out.append(f"{pad}{item.op} {args}{note}")
        elif isinstance(item, Sync):
            out.append(
                f"{pad}sync {item.cpu} <- {item.other}: {item.text}  ; {item.note}"
            )
        else:
            out.append(f"{pad}call {item.text}  ; {item.note}")
    return out


def _price(items, times, terms, previous):
    """Appends (what, count, cycles, source) per item, `times` passes; `previous`
    is the last instruction issued (a MAD-class result read next costs a stall)."""
    for item in items:
        if isinstance(item, Loop):
            _price(item.body, times * item.count, terms, previous)
        elif isinstance(item, Insn) and item.op == "REPLAY":
            # Recording with execution runs what follows (priced there); a play
            # runs `len` recorded instructions, a cycle each.
            plays = 0 if item.args["load_mode"] else item.args["len"]
            terms.append(("REPLAY", times, times * plays, REPLAY_SOURCE))
            previous[:] = []
        elif isinstance(item, Insn):
            stall = 0
            if previous and OPS[previous[0].op].kind == "mad":
                written = set(previous[0].writes())
                stall = 1 if written & set(item.reads()) else 0
            terms.append((item.op, times, times * (1 + stall), COST_SOURCE))
            if stall:
                terms.append(("stall on a MAD result", times, 0, COST_SOURCE))
            previous[:] = [item]
        elif isinstance(item, Call):
            terms.append((item.text, times, None, "library call, not priced"))
        else:
            terms.append((f"sync {item.text}", times, None, "sync, not priced"))


def _merge(terms):
    """Terms of the same name summed, in first-seen order."""
    out = {}
    for what, n, cycles, source in terms:
        if what in out:
            _, m, c, s = out[what]
            both = None if c is None or cycles is None else c + cycles
            out[what] = (what, m + n, both, s)
        else:
            out[what] = (what, n, cycles, source)
    return list(out.values())


def insns(items):
    """Every Insn of a loop nest, loop bodies once, in order."""
    for item in items:
        if isinstance(item, Loop):
            yield from insns(item.body)
        elif isinstance(item, Insn):
            yield item


def blocks(items):
    """The straight-line instruction lists of a loop nest (each loop body's own)."""
    run = []
    for item in items:
        if isinstance(item, Loop):
            if run:
                yield run
            run = []
            yield from blocks(item.body)
        elif isinstance(item, Insn):
            run.append(item)
    if run:
        yield run
