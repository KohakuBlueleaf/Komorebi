"""Schedule rule "load-macro chain": a row of loads, at most one operation for each of
the vector unit's simple, MAD and round sub-units, and one store of the result where
an operand was loaded becomes one SFPLOADMACRO a row (beside the other operands'
plain loads). The fit is what the ISA's SFPLOADMACRO page states; a row that does not
fit is refused with the reason (`Refused`)."""

from dataclasses import dataclass, field
from math import lcm

from komorebi.dsl.tile.unit import OPS, Insn

# The sub-unit each instruction a macro slot may arm runs on (the ISA's allocation
# table). Flag and lane-movement instructions are left out: see `fit`.
UNITS = {
    "simple": (
        "SFPABS",
        "SFPAND",
        "SFPARECIP",
        "SFPCAST",
        "SFPDIVP2",
        "SFPEXEXP",
        "SFPEXMAN",
        "SFPIADD",
        "SFPLZ",
        "SFPMOV",
        "SFPNOT",
        "SFPOR",
        "SFPSETEXP",
        "SFPSETMAN",
        "SFPSETSGN",
        "SFPSHFT",
        "SFPXOR",
    ),
    "mad": ("SFPADD", "SFPADDI", "SFPMAD", "SFPMUL", "SFPMULI"),
    "round": ("SFP_STOCH_RND", "SFPSHFT2"),
}
# A sequence word holds one byte a slot, from bit 0: simple, MAD, round, store.
SLOT_SHIFT = {"simple": 0, "mad": 8, "round": 16, "store": 24}
STORE_SLOT = 3  # slot selector: SFPSTORE of the loaded register
TEMPLATE_SLOT = 4  # selectors 4 .. 7: instruction template 0 .. 3
PATCH_VB = 0x80  # the loaded register replaces VB (else VC)
MACROS = 4  # sequences, and instruction templates
MAX_DELAY = 7
LOADED = 4  # the loaded register is LReg 0 .. 3 (the macro's two low VD bits)
FIRST_PLAIN = 4  # plain-loaded operands rotate in LReg 4 .. 7
COMMUTES = ("SFPMAD", "SFPMUL", "SFPADD")  # VA * VB: the operands may swap


class Refused(ValueError):
    """A row the load-macro rule does not take, and why."""


@dataclass(eq=False)
class Plan:
    """A row as a macro: the load it carries, the other loads, the chain of
    (instruction, sub-unit, patched slot, delay), the store's delay, the rotations
    of the loaded register and of the plain-loaded ones."""

    carried: Insn
    others: list
    chain: list = field(default_factory=list)
    store_delay: int = 0
    loaded_turns: int = 1
    plain_turns: int = 1

    @property
    def rows(self):
        """Rows in one turn of every rotation."""
        return lcm(self.loaded_turns, self.plain_turns)


def _latency(insn, unit):
    if unit == "mad":
        return 2
    if insn.op == "SFPSHFT2" and insn.args["instr_mod1"] in (2, 3, 4):
        return 2
    return 1


def fit(row):
    """The load-macro plan of allocated row code `row` (LReg numbers), or Refused."""
    loads = [x for x in row if x.op == "SFPLOAD"]
    stores = [x for x in row if x.op == "SFPSTORE"]
    work = [x for x in row if x.op not in ("SFPLOAD", "SFPSTORE")]
    if len(stores) != 1:
        raise Refused(f"the row stores {len(stores)} values; the store slot stores one")
    store = stores[0]
    address = store.args["dest_reg_addr"]
    carried = [x for x in loads if x.args["dest_reg_addr"] == address]
    if not carried:
        raise Refused(
            "the result goes where no operand was loaded; the macro's store writes "
            "the row its load read"
        )
    plan = Plan(carried[0], [x for x in loads if x is not carried[0]])
    plain = {x.args["lreg_ind"] for x in plan.others}
    if not work:
        raise Refused("the row computes nothing")
    of = {
        id(x): next((u for u, ops in UNITS.items() if x.op in ops), None) for x in work
    }
    counted = [of[id(x)] for x in work]
    if len(work) > 3 or any(counted.count(u) > 1 for u in UNITS):
        mads = counted.count("mad")
        raise Refused(
            f"the row computes with {len(work)} instructions ({mads} MAD-class); "
            "a macro arms at most one a sub-unit (simple, MAD, round)"
        )
    value, delay, units = plan.carried.args["lreg_ind"], 0, set()
    for insn in work:
        unit = of[id(insn)]
        if insn.sets_flags() or insn.op in ("SFPENCC", "SFPGT", "SFPSETCC"):
            raise Refused(f"{insn.op} sets lane flags; a macro's slots run unguarded")
        if unit is None:
            raise Refused(f"{insn.op} is not an instruction a macro slot arms")
        if insn.op == "SFPSHFT2" and insn.args["instr_mod1"] in (0, 1, 2):
            raise Refused("an SFPSHFT2 register shuffle touches LReg 0 .. 3")
        dest = OPS[insn.op].writes[0]
        if dest in insn.read_fields():
            raise Refused(f"{insn.op} reads its destination, which a template fixes")
        slot = _slot(insn, value)
        others = [r for r in insn.reads() if r != value]
        if any(r not in plain and r < 9 for r in others):
            raise Refused(
                f"{insn.op} reads LReg {others}: a template's other registers are "
                "constants or a plain load of the same row"
            )
        plan.chain.append((insn, unit, slot, delay))
        units.add(unit)
        delay += _latency(insn, unit)
        value = insn.args[dest]
    if store.args["lreg_ind"] != value:
        raise Refused("the row stores a value other than the chain's last")
    plan.store_delay = delay
    if delay > MAX_DELAY:
        raise Refused(f"the store comes {delay} cycles after the load; at most 7")
    _rotations(plan)
    return plan


def _slot(insn, value):
    """The field the loaded register patches (VC, or VB with PATCH_VB); swaps a
    product's operands when the chain value is VA. Refused when neither fits."""
    reads = insn.reads()
    if reads.count(value) != 1:
        raise Refused(f"{insn.op} does not read the previous result exactly once")
    args = insn.args
    if "lreg_src_a" in args:
        if args["lreg_src_a"] == value:
            if insn.op not in COMMUTES:
                raise Refused(f"{insn.op} reads the previous result as VA")
            args["lreg_src_a"], args["lreg_src_b"] = args["lreg_src_b"], value
        if args["lreg_src_b"] == value:
            return "lreg_src_b"
        return "lreg_src_c"
    if args.get("lreg_c") == value:
        return "lreg_c"
    raise Refused(f"{insn.op} reads the previous result in no patched field")


def _rotations(plan):
    """How many registers the loaded value and each plain-loaded operand rotate
    over, so that no load overwrites a register a scheduled instruction has yet to
    read: a row issues 1 + len(others) instructions."""
    width = 1 + len(plan.others)
    turns = 1
    while turns * width <= plan.store_delay + 1:
        turns += 1
    plan.loaded_turns = turns
    last_read = max(
        (d for insn, _, _, d in plan.chain if set(insn.reads()) & _plain(plan)),
        default=None,
    )
    turns = 1
    if last_read is not None:
        while turns * width <= width + last_read:
            turns += 1
    plan.plain_turns = turns
    templates = len(plan.chain) * plan.plain_turns
    if plan.loaded_turns > LOADED or len(plan.others) * turns > 8 - FIRST_PLAIN:
        raise Refused("the rotation needs more registers than the macro can name")
    if templates > MACROS or plan.plain_turns > MACROS:
        raise Refused(f"{templates} instruction templates; the unit holds 4")


def _plain(plan):
    return {x.args["lreg_ind"] for x in plan.others}


def setup(plan):
    """The instructions that write the templates (through the back door: VD 12 + k
    stores the word), the sequences and the store format, once per stream; LaneConfig
    must leave the back door open (the setup's first SFPCONFIG writes it 0)."""
    out = [Insn("SFPNOP", {}, "LaneConfig settles"), Insn("SFPNOP", {}, "")]
    for v in range(plan.plain_turns):
        sequence = STORE_SLOT | plan.store_delay << 3
        sequence <<= SLOT_SHIFT["store"]
        for k, (insn, unit, slot, delay) in enumerate(plan.chain):
            index = k * plan.plain_turns + v
            args = dict(insn.args)
            args[slot] = 0
            args[OPS[insn.op].writes[0]] = 12 + index
            for x, reg in zip(plan.others, _registers(plan, v)):
                for name, _ in OPS[insn.op].args:
                    if name != slot and insn.args[name] == x.args["lreg_ind"]:
                        args[name] = reg
            out.append(Insn(insn.op, args, f"template {index}: {insn.note}"))
            bits = TEMPLATE_SLOT + index | delay << 3
            bits |= PATCH_VB if slot == "lreg_src_b" else 0
            sequence |= bits << SLOT_SHIFT[unit]
        low = {"lreg_ind": 0, "instr_mod0": 10, "imm16": sequence & 0xFFFF}
        high = {"lreg_ind": 0, "instr_mod0": 8, "imm16": sequence >> 16}
        config = {"imm16_math": 0, "config_dest": 4 + v, "instr_mod1": 0}
        out += [
            Insn("SFPLOADI", low, f"sequence {v} low"),
            Insn("SFPLOADI", high, f"sequence {v} high"),
            Insn("SFPCONFIG", config, f"sequence {v}"),
            Insn("SFPNOP", {}, ""),
            Insn("SFPNOP", {}, ""),
        ]
    mask = ((1 << plan.plain_turns) - 1) << 4
    misc = {"imm16_math": mask, "config_dest": 8, "instr_mod1": 1}
    out += [
        Insn("SFPCONFIG", misc, "macro stores use their load's format"),
        Insn("SFPNOP", {}, ""),
        Insn("SFPNOP", {}, ""),
    ]
    return out


def _registers(plan, variant):
    return [
        FIRST_PLAIN + k * plan.plain_turns + variant for k in range(len(plan.others))
    ]


def rows(plan, addr_mod):
    """The code of `plan.rows` rows: each row's plain loads, then its macro, whose
    address mode `addr_mod` steps Dst to the next row group."""
    out = []
    for r in range(plan.rows):
        v, loaded = r % plan.plain_turns, r % plan.loaded_turns
        for x, reg in zip(plan.others, _registers(plan, v)):
            out.append(Insn("SFPLOAD", {**x.args, "lreg_ind": reg}, x.note))
        args = dict(plan.carried.args)
        args["lreg_ind"] = v << 2 | loaded
        args["sfpu_addr_mode"] = addr_mod
        out.append(Insn("SFPLOADMACRO", args, f"macro {v}: load, chain, store"))
    return out


def drain(plan):
    """SFPNOPs after the last macro until its store has run."""
    return [Insn("SFPNOP", {}, "load-macro drain") for _ in range(plan.store_delay + 2)]
