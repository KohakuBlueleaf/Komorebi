"""Today's per-row passes (drop_copies, common_values, per_row_prepass, interleave) on
code of a few words, the branches the corpus (tile_units.py) does not reach: each
case's code (`<case>.in`, one word a line) and today's result (`<case>.out`: the
report, the code after the pass, the pre-pass's code). Usage: <this> <folder>."""

import sys
from pathlib import Path

from komorebi.dsl.tile.lower import common_values, drop_copies, per_row_prepass
from komorebi.dsl.tile.schedule import interleave
from komorebi.dsl.tile.unit import OPS, Insn


def w(op, note="", **args):
    """`op` with the fields given, the others 0."""
    return Insn(op, {a: args.get(a, 0) for a, _ in OPS[op].args}, note)


def load(v, address):
    return w("SFPLOAD", f"Dst + {address}", lreg_ind=v, dest_reg_addr=address)


def store(v, address):
    return w("SFPSTORE", "store", lreg_ind=v, dest_reg_addr=address)


def const(v, imm, mod=0):
    return w("SFPLOADI", "const", lreg_ind=v, instr_mod0=mod, imm16=imm)


def mul(a, b, out):
    return w(
        "SFPMUL", "a b", lreg_src_a=a, lreg_src_b=b, lreg_src_c="zero", lreg_dest=out
    )


def add(a, b, out):
    return w(
        "SFPADD", "a + b", lreg_src_a="one", lreg_src_b=a, lreg_src_c=b, lreg_dest=out
    )


def mov(a, out, mod=0):
    return w("SFPMOV", "copy", lreg_c=a, lreg_dest=out, instr_mod1=mod)


def gt(a, b):
    return w("SFPGT", "a gt b", lreg_c=b, lreg_dest=a, instr_mod1=1)


ENCC = w("SFPENCC", "every lane on", imm12_math=3, instr_mod1=10)
# nine loads of one Dst row and its stores after them: their order is fixed
NINE = [load(f"v{i}", 0) for i in range(1, 10)]
CASES = [
    # interleave: a constant read in a run under lane flags and outside it
    (
        "outside",
        "interleave",
        [
            const("v1", 0x3F00),
            load("v2", 0),
            gt("v2", "zero"),
            mov("v1", "v2"),
            ENCC,
            mul("v2", "v1", "v3"),
            store("v3", 0),
        ],
    ),
    # a run under lane flags that does not close: nothing moved
    ("open_run", "interleave", [load("v1", 0), gt("v1", "zero"), mov("v1", "v2")]),
    # nine values live at once at any order: kept as it was
    (
        "pressure",
        "interleave",
        [*NINE, *[add(f"v{i}", f"v{i + 1}", f"v{i + 10}") for i in range(1, 9)]]
        + [store(f"v{i}", 0) for i in range(11, 19)],
    ),
    # a constant register's setup (loads into L0, SFPCONFIG) held in front
    (
        "setup",
        "interleave",
        [
            load("v1", 0),
            mul("v1", "v1", "v2"),
            const("L0", 0x3F00),
            w("SFPCONFIG", "constant register p0", config_dest="p0"),
            add("v2", "p0", "v3"),
            store("v3", 0),
        ],
    ),
    # drop_copies: a copy under lane flags stays, a plain one goes
    (
        "copies",
        "drop_copies",
        [
            load("v1", 0),
            mov("v1", "v2"),
            gt("v2", "zero"),
            mov("v2", "v3"),
            ENCC,
            store("v3", 0),
        ],
    ),
    # common_values: two constants of the same bits are one, under flags nothing
    (
        "bits",
        "common_values",
        [
            load("v1", 0),
            const("v2", 0x3F00),
            const("v3", 0x3800, 1),
            mul("v1", "v2", "v4"),
            mul("v1", "v3", "v5"),
            gt("v1", "zero"),
            mul("v1", "v2", "v6"),
            ENCC,
            add("v4", "v5", "v7"),
            add("v7", "v6", "v8"),
            store("v8", 0),
        ],
    ),
    # per_row_prepass: the value of the per-row operand and a constant, the constant
    # read by the rows too (its load stays)
    (
        "prepass",
        "per_row_prepass",
        [
            load("v1", 64),
            const("v2", 0x3F00),
            mul("v1", "v2", "v3"),
            add("v3", "one", "v4"),
            load("v5", 0),
            mul("v5", "v4", "v6"),
            mul("v6", "v2", "v7"),
            store("v7", 0),
        ],
    ),
    # two per-row operands
    (
        "two_operands",
        "per_row_prepass",
        [
            load("v1", 64),
            load("v2", 128),
            add("v1", "v2", "v3"),
            load("v4", 0),
            mul("v4", "v3", "v5"),
            store("v5", 0),
        ],
    ),
]


def line(x):
    """A word as the `.in` and `.out` files write it."""
    fields = " ".join(f"{a}={x.args[a]}" for a, _ in OPS[x.op].args)
    return f"{x.op} {fields} ; {x.note}"


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for case, rule, code in CASES:
        (folder / f"{case}.in").write_text(
            f"{rule}\n" + "".join(f"{line(x)}\n" for x in code)
        )
        pre = None
        if rule == "interleave":
            done = [f"{interleave(code, 0, len(code))} moved"]
        elif rule == "drop_copies":
            done = drop_copies(code)
        elif rule == "common_values":
            done = common_values(code)
        else:
            pre, done = per_row_prepass(code, {64, 128})
        out = ["; ".join(done) or "nothing", *map(line, code)]
        if pre is not None:
            out += ["PRE", *map(line, pre)]
        (folder / f"{case}.out").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
