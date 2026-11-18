"""The multiply-add reader check covers a loop body's wrap: a body that ends in SFPMAD
and starts with a reader of its result gets an SFPNOP; one that starts with a load
does not."""

from komorebi.dsl.tile.lower import check_mad_readers
from komorebi.dsl.tile.unit import Insn

mad = {
    "lreg_src_a": 0,
    "lreg_src_b": 1,
    "lreg_src_c": 2,
    "lreg_dest": 3,
    "instr_mod1": 0,
}
iadd = {"imm12_math": 0, "lreg_c": 3, "lreg_dest": 4, "instr_mod1": 0}
load = {"lreg_ind": 3, "instr_mod0": 0, "sfpu_addr_mode": 7, "dest_reg_addr": 0}
for first, want in ((("SFPIADD", iadd), 1), (("SFPLOAD", load), 0)):
    code = [Insn(*first), Insn("SFPMAD", mad)]
    n = check_mad_readers([code])
    print(f"[{first[0]}, SFPMAD]: {n} SFPNOP, {[x.op for x in code]}")
    assert n == want and (code[-1].op == "SFPNOP") == bool(want)
print("OK")
