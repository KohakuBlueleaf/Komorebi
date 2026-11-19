"""Wormhole's forms (hazards.wormhole): a negated operand as a sign-complemented copy,
a shift of another register as a copy and Wormhole's shift, the integer casts as
SFPABS and SFPSETSGN; a copy takes a register only where it is dead (no flags change
before its next write), and with none the word keeps no form (refused by name)."""

from types import SimpleNamespace

from komorebi.device import chips
from komorebi.dsl.tile.hazards import wormhole
from komorebi.dsl.tile.unit import Insn

WORMHOLE = chips.chip("wormhole_b0").tile_refusal


def mad(op="SFPMAD", a=1, b=2, c=3, d=4, mod1=0):
    args = {"lreg_src_a": a, "lreg_src_b": b, "lreg_src_c": c, "lreg_dest": d}
    return Insn(op, {**args, "instr_mod1": mod1})


def word(op, c, d, mod1=0, imm=0):
    return Insn(
        op, {"imm12_math": imm, "lreg_c": c, "lreg_dest": d, "instr_mod1": mod1}
    )


def formed(*words):
    unit = SimpleNamespace(init={"math": []}, tile={"math": list(words)}, parts={})
    wormhole.forms(unit)
    return [(x.op, x.args) for x in words[0].wormhole]


def busy(*registers):
    """A store after the word that reads `registers` (they stay live)."""
    return [word("SFPABS", r, r) for r in registers]


def test_negate_vc_is_a_negated_copy_in_the_destination():
    x = mad(a=1, b=2, c=3, d=4, mod1=2)
    (move, move_args), (op, args) = formed(x, *busy(0, 1, 2, 3, 5, 6))
    assert (move, move_args["lreg_c"], move_args["lreg_dest"]) == ("SFPMOV", 3, 4)
    assert move_args["instr_mod1"] == 1  # the sign complemented
    assert (op, args["lreg_src_c"], args["instr_mod1"]) == ("SFPMAD", 4, 0)


def test_both_negations_use_the_destination_and_a_dead_register():
    """-(a b) - c with d = a: the product's sign through b (d takes -b), the addend's
    through a register written next without a read between (no flags change)."""
    x = mad(a=4, b=3, c=9, d=6, mod1=3)
    rest = [mad(a=1, b=2, c=9, d=3), *busy(0, 1, 2, 4, 5, 6)]
    words = formed(x, *rest)
    assert [w[0] for w in words] == ["SFPMOV", "SFPMOV", "SFPMAD"]
    assert words[2][1]["instr_mod1"] == 0
    assert all(WORMHOLE(None, (op, args, "", "", [], [])) is None for op, args in words)


def test_a_write_after_a_flag_change_does_not_free_a_register():
    """The write after SFPGT keeps the lanes it skips: their old value stays live."""
    x = mad(a=4, b=3, c=9, d=6, mod1=3)
    flags = Insn(
        "SFPSETCC", {"imm12_math": 0, "lreg_c": 4, "lreg_dest": 0, "instr_mod1": 0}
    )
    assert formed(x, flags, mad(a=1, b=2, c=9, d=3), *busy(0, 1, 2, 4, 5, 6)) == []


def test_the_shift_of_another_register_is_a_copy_then_wormhole_s_shift():
    assert formed(word("SFPSHFT", 1, 1, 5, 9)) == [
        ("SFPSHFT", {"imm12_math": 9, "lreg_c": 0, "lreg_dest": 1, "instr_mod1": 1})
    ]
    words = formed(word("SFPSHFT", 2, 1, 5, 9))
    assert [w[0] for w in words] == ["SFPMOV", "SFPSHFT"]


def test_blackhole_s_integer_casts_are_sfpabs_and_sfpsetsgn():
    cast = Insn("SFPCAST", {"lreg_src_c": 2, "lreg_dest": 3, "instr_mod1": 3})
    words = formed(cast, *busy(2, 3))
    assert [w[0] for w in words] == ["SFPMOV", "SFPABS", "SFPSETSGN"]
    assert words[2][1]["lreg_dest"] == 3 and words[1][1]["instr_mod1"] == 0
    cast = Insn("SFPCAST", {"lreg_src_c": 2, "lreg_dest": 3, "instr_mod1": 2})
    assert [w[0] for w in formed(cast)] == ["SFPABS"]


def test_a_template_store_keeps_no_form():
    """VD 12 + k stores load macro template k and runs nothing: a copy before it would
    run once and the stored word would lose its negation; it stays refused."""
    assert formed(mad("SFPADD", a=10, b=3, c=2, d=12, mod1=2), *busy(0, 1)) == []
