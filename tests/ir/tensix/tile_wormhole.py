"""Today's Wormhole forms, emit and its checks on units of a few words each, the ones
the corpus (tile_units.py) does not reach: writes each unit as data (`<case>.unit`)
and what today makes of it (`.out`): the forms, each check's verdict, the header or
the refusal. Usage: <this> <folder>."""

import copy
import sys
from pathlib import Path

from tile_units import unit_lines

from komorebi.dsl.tile import emit
from komorebi.dsl.tile.emit import check_buffers
from komorebi.dsl.tile.hazards import wormhole
from komorebi.dsl.tile.macro import check_back_door, check_templates
from komorebi.dsl.tile.unit import (
    OPS,
    Call,
    Insn,
    Loop,
    Product,
    Runtime,
    Sync,
    UnitError,
    UnitProgram,
)


def w(op, values, note="", uses=(), runtime=""):
    """`op` with its fields' `values` in the table's order."""
    args = dict(zip([a for a, _ in OPS[op].args], values, strict=True))
    return Insn(op, args, note, tuple(uses), runtime)


def loads(registers):
    return [w("SFPLOAD", [r, 0, 3, 0]) for r in registers]


def stores(registers):
    return [w("SFPSTORE", [r, 0, 3, 0]) for r in registers]


def chain(n):
    """`n` multiply-adds, each reading the one before (Wormhole holds none)."""
    return [w("SFPMAD", [0, 10, 9, 0, 0], f"m{k}") for k in range(n)]


def sequence(k, word):
    """Macro `k`'s sequence `word`, as an init sets it."""
    return [
        w("SFPLOADI", [0, 10, word & 0xFFFF]),
        w("SFPLOADI", [0, 8, word >> 16]),
        w("SFPCONFIG", [0, 4 + k, 0]),
    ]


def macro(k):
    return w("SFPLOADMACRO", [4 * k, 0, 3, 0])


def wait(other, text):
    return Sync("unpack", other, text, "")


TEMPLATE = w("SFPMAD", [1, 2, 4, 12, 0], "template 0")
EVERY = range(7)
RUNS = [
    w("SFPLOADI", [0, 2, 0], "k", runtime="k_bits & 0xFFFFu"),
    w("REPLAY", [0, 34, 0, 1]),
    *chain(34),
    Loop("face", 4, [w("REPLAY", [0, 34, 0, 0]), w("SFPSTORE", [0, 0, 7, 0])], True),
    Loop("band", Runtime("rows / 4", lambda rows: rows // 4), [w("SFPNOP", [])]),
    Loop("tail", Runtime("rows > 16", lambda rows: int(rows > 16), True), chain(1)),
    Product("matmul()", [("MVMUL", 0)], "p", False, chain(1), loads([1])),
    w("STALLWAIT", [0x100, 0x10], "stall"),
]
# (case, {"<init|tile>_<cpu>": items}): each kind of word's form, the run on Wormhole
# (`runs`), each check's verdicts, a float in the text
CASES = [
    ("shft", {"tile_math": [w("SFPSHFT", [3, 1, 2, 5], "shift")]}),
    ("cast_abs", {"tile_math": [w("SFPCAST", [1, 2, 2], "abs")]}),
    ("cast_sm", {"tile_math": [w("SFPCAST", [1, 2, 3], "sm")]}),
    ("gt_zero_d", {"tile_math": [w("SFPGT", [0, 1, 9, 1], "gt")]}),
    ("gt_zero_c", {"tile_math": [w("SFPGT", [0, 9, 2, 1], "gt")]}),
    ("gt", {"tile_math": [w("SFPGT", [0, 1, 2, 1], "gt")]}),
    (
        "gt_lanes",
        # one register dead after it (3): the lanes' form
        {
            "tile_math": [
                *loads(EVERY),
                w("SFPGT", [0, 1, 2, 1]),
                *stores([0, 1, 2, 4, 5, 6]),
            ]
        },
    ),
    (
        # 3 read after the next tile writes it, every lane on: dead after the flags
        "gt_lanes_next_tile",
        {
            "tile_math": [
                *loads(EVERY),
                *stores([3]),
                w("SFPGT", [0, 1, 2, 1]),
                *stores([0, 1, 2, 4, 5, 6]),
            ]
        },
    ),
    (
        "gt_none",
        {"tile_math": [*loads(EVERY), w("SFPGT", [0, 1, 2, 1]), *stores(EVERY)]},
    ),
    ("recip", {"tile_math": [w("SFPARECIP", [0, 1, 2, 0], "r")]}),
    ("recip_same", {"tile_math": [w("SFPARECIP", [0, 1, 1, 0], "r")]}),
    ("mad_negated", {"tile_math": [w("SFPMAD", [1, 2, 3, 4, 3], "m", uses=(5,))]}),
    (
        "mad_zero",
        {"tile_math": [*loads(EVERY), w("SFPMAD", [1, 2, 9, 1, 2]), *stores(EVERY)]},
    ),
    (
        "template",
        {
            "tile_math": [
                w("SFPMAD", [1, 2, 4, 12, 2], "template 0"),
                *loads([4]),
                macro(0),
            ]
        },
    ),
    (
        "template_read",
        {"tile_math": [w("SFPMAD", [1, 2, 4, 12, 2], "t"), *loads([4]), *stores([4])]},
    ),
    (
        "runs",
        {
            "init_math": [w("SFPCONFIG", [0, 15, 1]), Call("ckernel::f()", "")],
            "tile_math": RUNS,
        },
    ),
    (
        "buffers",
        {
            "tile_unpack": [
                wait("reader", "cb_wait_front(in0, 1); cb_wait_front(  in1, 1)")
            ],
            "tile_pack": [
                Call("pack_tile(0, out)", ""),
                Loop("i", 2, [Call("cb_push_back(out2, 1)", "")]),
            ],
        },
    ),
    (
        "two_producers",
        {
            "tile_unpack": [wait("writer", "cb_wait_front(out, 1)")],
            "tile_pack": [Call("pack_tile(dst + 1, out)", "")],
        },
    ),
    ("back_door_init", {"init_math": [TEMPLATE, w("SFPSTORE", [13, 0, 3, 0], "x")]}),
    (
        "back_door_tile",
        {
            "init_math": [TEMPLATE],
            "tile_math": [Loop("f", 4, [w("SFPMOV", [0, 1, 14, 0])])],
        },
    ),
    ("no_sequence", {"tile_math": [macro(1)]}),
    (
        "not_stored",
        {"init_math": [*sequence(1, 0x0504), TEMPLATE], "tile_math": [macro(1)]},
    ),
    (
        "stored_by_a_word",
        {
            "init_math": [*sequence(1, 0x0504), TEMPLATE, w("SFPMOV", [0, 1, 13, 0])],
            "tile_math": [macro(1)],
        },
    ),
    (
        "stored",
        {
            "init_math": [
                *sequence(0, 0x0404),
                *sequence(2, 0x050000),
                TEMPLATE,
                w("SFPMOV", [0, 1, 13, 0], "template 1"),
            ],
            "tile_math": [macro(0), macro(2), macro(0)],
        },
    ),
    (
        "overwritten",
        {
            "init_math": [*sequence(0, 0x04), TEMPLATE],
            "tile_math": [macro(0), w("SFPMOV", [0, 1, 12, 0], "x")],
        },
    ),
    (
        "sequence_cut",
        {
            "init_math": [
                w("SFPLOADI", [0, 10, 4]),
                w("SFPMOV", [0, 1, 0, 0]),
                w("SFPLOADI", [0, 8, 0]),
                w("SFPCONFIG", [0, 4, 0]),
            ],
            "tile_math": [macro(0)],
        },
    ),
    *[
        (f"float_{k}", {"tile_math": [Call(text, "")]})
        for k, text in enumerate(
            ["x(1.5f)", "x(1e5)", "float y", "x(2.)", "a1.5", "x(1.5fy)", "x(12e+3)"]
        )
    ],
]


def unit(case, cpus):
    """The unit of `case`, its items fresh."""
    made = UnitProgram(case, "fp32", scalars=["k_bits"] if case == "runs" else [])
    for key, items in cpus.items():
        part, cpu = key.split("_")
        getattr(made, part)[cpu] = copy.deepcopy(items)
    return made


def text(x):
    """A word as `op fields ; note`."""
    return f"{x.op} {','.join(str(x.args[a]) for a, _ in OPS[x.op].args)} ; {x.note}"


def words(items):
    for x in items:
        if isinstance(x, Loop):
            yield from words(x.body)
        elif isinstance(x, Product):
            yield from (*x.insns, *x.wormhole)
        elif isinstance(x, Insn):
            yield x


def tried(f, *args):
    try:
        return f(*args)
    except UnitError as error:
        return f"refused: {error}"


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for case, cpus in CASES:
        (folder / f"{case}.unit").write_text(
            "\n".join(unit_lines(unit(case, cpus))) + "\n"
        )
        made = unit(case, cpus)
        wormhole.forms(made)
        functions = [made.init["math"], made.tile["math"], *made.parts.values()]
        out = [
            "\t".join(["FORM", text(x), *map(text, x.wormhole)])
            for function in functions
            for x in words(function)
            if x.wormhole
        ]
        for name, check in (
            ("one-producer", check_buffers),
            ("back-door", check_back_door),
            ("macro-templates", check_templates),
        ):
            out.append(f"CHECK\t{name}\t{tried(check, unit(case, cpus))}")
        out.append(f"EMIT\n{tried(emit, unit(case, cpus))}")
        (folder / f"{case}.out").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
