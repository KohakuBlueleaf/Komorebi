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
# LReg 11 as a fourth constant register: set at the start of a unit's body and set
# back to -1.0 at its end (rule constant_registers, the step form).
SAVED = 11
WORK = 8  # LReg 0 .. 7 hold values
REPLAY_ENTRIES = 32  # entries of one thread's replay buffer
# Cycles (AICLK) of the math CPU's vector code: one-core Blackhole measurements
# (experiments/tile_probes/README.md, section 8) or, where marked, the ISA's word.
PRICES = {
    "word": (1.0, "a ready word pushed: 1 cycle (probe 8: 1.128 in a loop of 16)"),
    "loop": (2.0, "a pass of a loop the core runs: N + 2 cycles (probe 12)"),
    "replayed": (1.004, "a replayed word: 1.004 cycles (probe 8)"),
    "mad_stall": (1.0, "a MAD result read next: 1 stall (probe 9: 4.03 against 3.04)"),
    # read by a store: no stall (experiments/tile_rules: add, scale, SiLU on the card)
    "shft2_stall": (1.0, "SFPSHFT2 lane modes stall the unit 1 cycle (stated)"),
    "built_word": (4.14, "a word built at run time: 4.14 cycles (probe 8)"),
    "table_word": (9.37, "a word loaded from an L1 table: 9.37 cycles (probe 8)"),
    "macro_row": (1.046, "one SFPLOADMACRO a row: 1.046 cycles (probe 9)"),
    "swap_stall": (1.0, "SFPSWAP: the unit takes only SFPNOP the next cycle (stated)"),
}
# SFPSHFT2 modes that take 2 cycles and stall the unit one (VectorUnit table).
SHFT2_SLOW = (2, 3, 4)
# The matrix unit (probe 11: Dst fp32, bf16 sources; one word a phase of fidelity).
MATRIX_PRICES = {
    "matrix_word": (1.0, "MVMUL, ELWMUL, ELWADD, MOVA2D, MOVB2D: 1 cycle (probe 11)"),
    "dst_raw": (
        6.0,
        "a word adding into a Dst block waits 6 cycles after its last write",
    ),
    "matrix_sync": (8.0, "one MVMUL pushed alone: 8 cycles push to sync (probe 11)"),
}
# The words that add into their Dst block (the moves overwrite it: no wait).
DST_READERS = ("MVMUL", "ELWMUL", "ELWADD")


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
    "SFPSETCC": Op(0x7B, _R, ("lreg_c",), (), "flags"),
    # mod1 1: VC gets the larger, VD the smaller (a VD of 8 and up is not written)
    "SFPSWAP": Op(
        0x92,
        (("imm12_math", 12), ("lreg_src_c", 4), ("lreg_dest", 4), ("instr_mod1", 4)),
        ("lreg_src_c", "lreg_dest"),
        ("lreg_src_c", "lreg_dest"),
    ),
    # LReg 0-3 (and 4-7): per column, the 4 lane rows across the 4 registers
    "SFPTRANSP": Op(0x8C, _R),
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
    "SFPSHFT": Op(0x7A, _R, ("lreg_c",), ("lreg_dest",)),
    "SFPSHFT2": Op(
        0x94,
        (("imm12_math", 12), ("lreg_src_c", 4), ("lreg_dest", 4), ("instr_mod1", 4)),
        ("lreg_src_c",),
        ("lreg_dest",),
    ),
    "SFPDIVP2": Op(0x76, _R, ("lreg_c",), ("lreg_dest",)),
    "SFPLOADMACRO": Op(0x93, _LS, (), (), "macro"),
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
    # Dst[8 rows] += SrcB[8 rows] @ SrcA[16 x 16]; the address mode steps the counters
    "MVMUL": Op(
        0x26,
        (("clear_dvalid", 2), ("instr_mod19", 3), ("addr_mode", 5), ("dst", 14)),
        kind="matrix",
    ),
}
# SFPIADD's immediate form reads only lreg_c; mod1 bit 1: lreg_c - lreg_dest; bit 2
# set: no flags. SFPSHFT mod1 5: lreg_c shifted by the immediate.
IADD_IMM, IADD_SUB, IADD_NO_CC = 1, 2, 4
SHIFT_IMM_VC = 5
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
    # registers it reads and writes beside its fields (a load macro's loaded register
    # and the register its template names), LReg numbers
    uses: tuple = ()

    def reads(self):
        """The register arguments read (value names or LReg numbers)."""
        return [self.args[n] for n in self.read_fields()] + list(self.uses)

    def read_fields(self):
        """The names of the register fields read."""
        o = OPS[self.op]
        names = list(o.reads)
        if self.op == "SFPIADD" and self.args["instr_mod1"] & IADD_IMM:
            names.remove("lreg_dest")
        if self.op == "SFPLOADI" and self.args["instr_mod0"] in (8, 10):
            names.append("lreg_ind")
        if self.op in ("SFPSETSGN", "SFPSETEXP") and self.args["instr_mod1"] & 1:
            names.remove("lreg_dest")
        if self.op == "SFPSHFT" and self.args["instr_mod1"] & 5 != 5:
            names = ["lreg_dest"] + ([] if self.args["instr_mod1"] & 1 else names)
        return names

    def writes(self):
        """The register arguments written (SFPSWAP writes only work registers)."""
        out = [self.args[n] for n in OPS[self.op].writes]
        if self.op == "SFPSWAP":
            out = [v for v in out if _work_register(v)]
        return out + list(self.uses)

    def sets_flags(self):
        """True when it changes the lanes' condition flags."""
        if self.op == "SFPIADD":
            return not self.args["instr_mod1"] & IADD_NO_CC
        if self.op == "SFPGT":
            return bool(self.args["instr_mod1"] & 1)
        return self.op in ("SFPENCC", "SFPSETCC")


def _work_register(value):
    """True for LReg 0-7, by number, by "L<n>" or by a value's name (not a fixed or
    programmable constant's)."""
    if isinstance(value, int):
        return value < WORK
    if value[0] == "L":
        return int(value[1:]) < WORK
    return value not in FIXED and not value.startswith("p")


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
class Product:
    """Matrix-unit words issued by a library call (`text`): their (op, Dst block) in
    issue order, priced by `matrix_cost`; `synced`: a sync follows the last. With
    `insns` the words are emitted (their instructions) in place of the call."""

    text: str
    words: list
    note: str
    synced: bool = True
    insns: list = field(default_factory=list)


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
    operands: list = field(default_factory=list)
    init: dict = field(default_factory=lambda: {cpu: [] for cpu in CPUS})
    tile: dict = field(default_factory=lambda: {cpu: [] for cpu in CPUS})
    # the math CPU's later parts of a unit (`<name>_<part>(dst)`), after the matrix
    # unit's work between them; priced with the tile's math program
    parts: dict = field(default_factory=dict)
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
        state = {"previous": None, "recording": None, "buffer": {}}
        _price(self.tile[cpu], 1, terms, state)
        if cpu == "math":
            for items in self.parts.values():
                state["previous"] = None
                _price(items, 1, terms, state)
        priced = [t for t in terms if t[2] is not None]
        count = sum(t[1] for t in priced if t[0].split()[0] in OPS)
        return count, round(sum(t[2] for t in priced), 1), terms

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
            if cpu == "math":
                for part, items in self.parts.items():
                    lines.append(f"  math per tile, part {part}:")
                    lines += _dump(items, 4)
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
                f"    {what}: {n} x, {'-' if c is None else round(c, 1)} cycles "
                f"({source})"
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
        elif isinstance(item, Product):
            out.append(
                f"{pad}product {item.text}: {len(item.words)} words  ; {item.note}"
            )
        else:
            out.append(f"{pad}call {item.text}  ; {item.note}")
    return out


def _price(items, times, terms, state):
    """Appends (what, count, cycles, source) per item, `times` passes. `state`: the
    last instruction issued (a stall follows a MAD result read next), the words being
    recorded into the replay buffer, and the recorded sequences by start index."""
    for item in items:
        if isinstance(item, Loop):
            if not item.unroll:
                cycles, source = PRICES["loop"]
                passes = times * item.count
                terms.append(("loop pass", passes, passes * cycles, source))
            _price(item.body, times * item.count, terms, state)
        elif isinstance(item, Insn) and item.op == "REPLAY":
            a = item.args
            if a["load_mode"]:
                # Recording with execution: what follows runs as pushed words.
                state["recording"] = (a["start_idx"], a["len"])
                state["buffer"][a["start_idx"]] = []
                _issue(item, times, terms, state, "word")
                continue
            played = state["buffer"].get(a["start_idx"], [])[: a["len"]]
            if len(played) < a["len"]:
                raise UnitError(f"REPLAY of {a['len']} plays words never recorded")
            for insn in played:
                _issue(insn, times, terms, state, "replayed")
        elif isinstance(item, Insn):
            _issue(item, times, terms, state, "word")
            if state["recording"]:
                start, count = state["recording"]
                state["buffer"][start].append(item)
                if len(state["buffer"][start]) == count:
                    state["recording"] = None
        elif isinstance(item, Call):
            terms.append((item.text, times, None, "library call, not priced"))
        elif isinstance(item, Product):
            cycles = matrix_cost(item.words, item.synced)
            source = "matrix_cost of the expanded words (probe 11)"
            terms.append(
                ("matrix words", times * len(item.words), times * cycles, source)
            )
            state["previous"] = None
        else:
            terms.append((f"sync {item.text}", times, None, "sync, not priced"))


def _issue(insn, times, terms, state, how):
    """The price of `insn` issued `times` (pushed: "word", or "replayed") and of the
    stall in front of it, if any."""
    previous = state["previous"]
    stall = None
    if (
        previous is not None
        and OPS[previous.op].kind == "mad"
        and insn.op != "SFPSTORE"
        and set(previous.writes()) & set(insn.reads())
    ):
        stall = "mad_stall"
    if (
        previous is not None
        and previous.op == "SFPSHFT2"
        and previous.args["instr_mod1"] in SHFT2_SLOW
    ):
        stall = "shft2_stall"
    if previous is not None and previous.op == "SFPSWAP" and insn.op != "SFPNOP":
        stall = "swap_stall"
    if stall:
        cycles, source = PRICES[stall]
        terms.append((f"stall ({stall})", times, times * cycles, source))
    if insn.op == "SFPLOADMACRO":
        how = "macro_row"
    cycles, source = PRICES[how]
    name = insn.op if how == "word" else f"{insn.op} {how}"
    terms.append((name, times, times * cycles, source))
    state["previous"] = insn


def matrix_cost(words, synced=True):
    """Cycles of matrix-unit `words` ((op, Dst block) in issue order): one a cycle, a
    word adding into a block waits 6 cycles after that block's last write (K words
    apart: max(1, 6 / K) a word), the last result 8 cycles to a sync."""
    gap, _ = MATRIX_PRICES["dst_raw"]
    t, written = -1.0, {}
    for op, block in words:
        t += MATRIX_PRICES["matrix_word"][0]
        if op in DST_READERS and block in written:
            t = max(t, written[block] + gap)
        written[block] = t
    if not words:
        return 0.0
    return t + (MATRIX_PRICES["matrix_sync"][0] if synced else 1.0)


def rotate_blocks(words):
    """Schedule rule: `words` reordered so a word adding into a Dst block comes as late
    after that block's last write as the order allows (each block's own words keep
    their order: its sum rounds the same). Returns the new order."""
    queues, order = {}, []
    for w in words:
        queues.setdefault(w[1], []).append(w)
    last = dict.fromkeys(queues, -len(words))
    for i in range(len(words)):
        block = min((b for b in queues if queues[b]), key=lambda b: (last[b], b))
        order.append(queues[block].pop(0))
        last[block] = i
    return order


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
