"""The Dst window check: a load past a call's tiles is refused, one inside passes;
"dst-window" is no longer among the rules not checked."""

from komorebi.dsl.tile.lower import NOT_CHECKED, check_dst_window
from komorebi.dsl.tile.unit import Insn, Loop, UnitError, UnitProgram


def unit(tile):
    u = UnitProgram("hand", "fp32")
    load = {
        "lreg_ind": 0,
        "instr_mod0": 0,
        "sfpu_addr_mode": 0,
        "dest_reg_addr": 64 * tile,
    }
    u.tile["math"] = [Loop("face", 4, [Insn("SFPLOAD", load)])]
    return u


assert not any(r.startswith("dst-window") for r in NOT_CHECKED)
u = unit(3)
check_dst_window(u, 4)
print("load at tile 3 of 4:", u.checks[-1].verdict)
try:
    check_dst_window(unit(4), 4)
    raise SystemExit("load at tile 4 of 4 passed")
except UnitError as error:
    print("load at tile 4 of 4:", error)
print("OK")
