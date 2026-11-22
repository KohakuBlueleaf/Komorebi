"""Today's hazard checks (hazards/checks.py) on units of a few words each, the branches
the corpus (tile_units.py) does not reach: writes each unit as data (`<case>.unit`)
and today's verdicts on its math words (`.out`): the repairs, LReg 11's check, what
the unit holds back after each word, the Dst window. Usage: <this> <folder>."""

import copy
import sys
from pathlib import Path

from tile_units import unit_lines
from tile_wormhole import w

from komorebi.dsl.tile.hazards.checks import (
    _check_saved,
    check_config_nop,
    check_dst_window,
    check_mad_readers,
    not_held,
)
from komorebi.dsl.tile.unit import Loop, UnitError, UnitProgram, insns


def mad(dest):
    return w("SFPMAD", [3, 4, 9, dest, 0])


def iadd(read):
    return w("SFPIADD", [0, read, 5, 5])


def load(imm):
    return w("SFPLOADI", [0, 0, imm])


def config(dest):
    return w("SFPCONFIG", [0, dest, 0])


def sgn(mod1):
    return w("SFPSETSGN", [0, 3, 1, mod1])


READ = w("SFPMAD", [11, 1, 9, 2, 0])  # reads LReg 11
CASES = [
    ("mad_reader", [iadd(2), mad(1), iadd(1), mad(2), w("SFPMOV", [0, 2, 7, 0])]),
    ("mad_wrap", [iadd(2), mad(2)]),
    ("held_sgn", [mad(1), sgn(0)]),
    ("held_sgn_imm", [mad(1), sgn(1)]),
    ("held_config", [w("SFPMOV", [0, 1, 0, 0]), config(12)]),
    ("held_shft2", [mad(1), w("SFPSHFT2", [0, 1, 1, 3])]),
    ("held_mov", [mad(1), w("SFPMOV", [0, 1, 2, 0])]),
    ("saved", [load(0x3F00), config(11), READ, load(0xBF80), config(11)]),
    ("saved_unwritten", [READ]),
    ("saved_unrestored", [load(0x3F00), config(11), READ]),
    ("saved_read_first", [READ, load(0x3F00), config(11), load(0xBF80), config(11)]),
    (
        "saved_clobbered",
        [load(0x3F00), config(11), READ, load(0xBF80), w("SFPMOV", [0, 1, 0, 0])]
        + [config(11)],
    ),
    ("config_nop", [config(12), w("SFPMOV", [0, 1, 2, 0]), config(11)] + [config(13)]),
    (
        "window",
        [w("SFPLOAD", [0, 0, 0, 0]), Loop("x", 2, [w("SFPSTORE", [0, 0, 0, 130])])],
    ),
    ("window_none", []),
]


def tried(f, *args):
    try:
        return f(*args)
    except UnitError as error:
        return f"refused: {error}"


def notes(code):
    return " ".join(f"{x.op}:{x.note}" for x in code)


def window(unit, tiles):
    """The dst-window check's verdict at `tiles` tiles a call."""
    unit.checks.clear()
    refused = tried(check_dst_window, unit, tiles)
    return refused or unit.checks[0].verdict


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for case, tile in CASES:
        unit = UnitProgram(case, "fp32")
        unit.tile["math"] = copy.deepcopy(tile)
        (folder / f"{case}.unit").write_text("\n".join(unit_lines(unit)) + "\n")
        flat = list(insns(unit.tile["math"]))
        held = [not_held(x, set(b.writes())) for b, x in zip(flat, flat[1:])]
        code = copy.deepcopy(flat)
        out = [f"MAD\t{check_mad_readers([code])}\t{notes(code)}"]
        code = copy.deepcopy(flat)
        out.append(f"CONFIG\t{check_config_nop(code)}\t{notes(code)}")
        out.append(f"SAVED\t{tried(_check_saved, flat)}")
        out.append("HELD\t" + " ".join(str(h) for h in held))
        out.append("WINDOW\t" + "\t".join(window(unit, t) for t in (1, 2, 3)))
        (folder / f"{case}.out").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
