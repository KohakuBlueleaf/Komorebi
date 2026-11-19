"""Wormhole's forms (hazards.wormhole): a negated operand as a sign-complemented copy,
a shift of another register as a copy and Wormhole's shift, the integer casts as
SFPABS and SFPSETSGN; a copy takes a register only where it is dead (no flags change
before its next write), and with none the word keeps no form (refused by name)."""

import random
import struct
from types import SimpleNamespace

import numpy as np

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
    x = mad(a=4, b=3, c=5, d=6, mod1=3)
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


def test_a_negated_zero_addend_is_read_as_zero_without_a_register():
    """-(y t) - 0 with every register live: the product's sign by a copy in the
    destination, the addend 0 (only a product of -0 differs: +0, not -0)."""
    words = formed(
        mad(a=4, b=3, c=wormhole.ZERO, d=6, mod1=3), *busy(0, 1, 2, 3, 4, 5, 6)
    )
    assert [w[0] for w in words] == ["SFPMOV", "SFPMAD"]
    assert words[1][1]["lreg_src_c"] == wormhole.ZERO and words[1][1]["instr_mod1"] == 0


def test_a_write_after_every_lane_is_on_frees_a_register():
    x = mad(a=4, b=3, c=5, d=6, mod1=3)
    flags = Insn(
        "SFPSETCC", {"imm12_math": 0, "lreg_c": 4, "lreg_dest": 0, "instr_mod1": 0}
    )
    imm, mod1 = wormhole.CC_RESET
    every = Insn(
        "SFPENCC", {"imm12_math": imm, "lreg_c": 0, "lreg_dest": 0, "instr_mod1": mod1}
    )
    words = formed(x, flags, every, mad(a=1, b=2, c=9, d=3), *busy(0, 1, 2, 4, 5, 6))
    assert [w[0] for w in words] == ["SFPMOV", "SFPMOV", "SFPMAD"]


# The lanes of a few vector words (Wormhole's functional model), one value a lane
M32 = 0xFFFFFFFF


def run_words(words, regs, lanes, flags=None):
    """The lanes' flags after `words` from `flags` (default every lane on)."""
    flags, use, stack = list(flags or [True] * lanes), True, []
    for op, a in words:
        if op == "SFPPUSHC":
            stack.append((list(flags), use))
            continue
        if op == "SFPPOPC":
            flags, use = stack.pop()
            flags = list(flags)
            continue
        on = [not use or f for f in flags]
        c, d, mod1, imm = (
            a.get("lreg_c"),
            a.get("lreg_dest"),
            a.get("instr_mod1"),
            a.get("imm12_math"),
        )
        for i in range(lanes):
            if op == "SFPENCC":
                use, flags[i] = bool(imm & 1), bool(imm & 2)
                continue
            if not on[i]:
                continue
            vc = regs[c][i] if c is not None and c in regs else None
            if op == "SFPMOV":
                regs[d][i] = vc ^ (0x80000000 if mod1 & 1 else 0)
            elif op == "SFPSHFT":
                k = imm - 4096 if imm & 0x800 else imm
                regs[d][i] = (regs[d][i] << k) & M32 if k >= 0 else regs[d][i] >> -k
            elif op == "SFPIADD":
                vd = regs[d][i]
                regs[d][i] = (vc - vd if mod1 & 2 else vc + vd) & M32
            elif op in ("SFPXOR", "SFPAND"):
                regs[d][i] = regs[d][i] ^ vc if op == "SFPXOR" else regs[d][i] & vc
            elif op == "SFPSETCC":
                s = vc - (1 << 32) if vc >> 31 else vc
                flags[i] = {0: s < 0, 2: s != 0, 4: s >= 0}[mod1]
            else:
                raise AssertionError(op)
    return [not use or f for f in flags]


def sign_mag_less(c, d):
    return _key(c) < _key(d)


def _key(x):
    """x's place in the sign-magnitude order, as a signed integer."""
    return (x ^ (0x7FFFFFFF if x >> 31 else 0)) - (1 << 32 if x >> 31 else 0)


EDGES = [
    0,
    0x80000000,
    1,
    0x80000001,
    0x3F800000,
    0xBF800000,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7FFFFFFF,
    0xFFFFFFFF,
    0x00800000,
    0x80800000,
]


def compare_lanes():
    rng = random.Random(7)
    values = EDGES + [rng.getrandbits(32) for _ in range(40)]
    pairs = [(c, d) for c in values for d in values]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def test_sfpgt_is_exact_with_two_registers_or_one_in_the_lanes_it_runs_in():
    """Blackhole's flags: VC < VD in the lanes on before, the others off."""
    cs, ds = compare_lanes()
    before = [i % 3 != 0 for i in range(len(cs))]
    want = [b and sign_mag_less(c, d) for b, c, d in zip(before, cs, ds)]
    gt = Insn("SFPGT", {"imm12_math": 0, "lreg_c": 1, "lreg_dest": 2, "instr_mod1": 1})
    for free in ([5, 6], [5]):
        words = [(w.op, w.args) for w in wormhole._form(gt, free)]
        assert words and words[-1][0] == "SFPSETCC"
        assert ("SFPPUSHC" in dict(words)) == (len(free) == 1)
        regs = {
            1: list(cs),
            2: list(ds),
            5: [0] * len(cs),
            6: [0] * len(cs),
            wormhole.ZERO: [0] * len(cs),
        }
        assert run_words(words, regs, len(cs), before) == want
        assert regs[1] == cs and regs[2] == ds  # VC and VD as they were
    assert wormhole._form(gt, []) == []


def test_sfpgt_against_zero_is_the_sign_bit_or_two_tests():
    cs, _ = compare_lanes()
    for c, d, want in (
        (1, wormhole.ZERO, [sign_mag_less(v, 0) for v in cs]),
        (wormhole.ZERO, 1, [sign_mag_less(0, v) for v in cs]),
    ):
        gt = Insn(
            "SFPGT", {"imm12_math": 0, "lreg_c": c, "lreg_dest": d, "instr_mod1": 1}
        )
        words = [(w.op, w.args) for w in wormhole._form(gt, [])]
        assert {op for op, _ in words} == {"SFPSETCC"}
        assert (
            run_words(words, {1: list(cs), wormhole.ZERO: [0] * len(cs)}, len(cs))
            == want
        )


def bf16(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def test_sfparecip_s_estimate_is_a_cubic_within_its_stated_error():
    """The words' value of p(m) 2^(128 - e) in float32: within 2.6e-3 of 1 / x over
    2^-126 .. 2^126 (the LUT's 5.6e-3); inf below, 0 at inf and NaN, x's sign."""
    recip = Insn(
        "SFPARECIP", {"imm12_math": 0, "lreg_c": 5, "lreg_dest": 3, "instr_mod1": 0}
    )
    ops = [w.op for w in wormhole._form(recip, [4])]
    assert "SFPARECIP" not in ops and ops[-1] == "SFPMUL"
    assert wormhole._form(recip, []) == []
    p = [np.float32(bf16(c)) for c in wormhole.RECIP_P]

    def estimate(x):
        u = np.asarray(x, np.float32).view(np.uint32)
        m = ((u & 0x7FFFFF) | (127 << 23)).astype(np.uint32).view(np.float32)
        t = m * p[0]
        for k, c in enumerate(p[1:]):
            t = (t + c).astype(np.float32)
            t = (t * m).astype(np.float32) if k < len(p) - 2 else t
        scale = (~u & 0xFF800000).astype(np.uint32).view(np.float32)
        return (t * scale).astype(np.float32)

    rng = np.random.default_rng(3)
    x = (rng.integers(0x00800000, 0x7E800000, 200000, dtype=np.uint32) | 0).view(
        np.float32
    )
    x = np.concatenate([x, -x])
    with np.errstate(over="ignore", invalid="ignore"):
        rel = np.abs(estimate(x).astype(np.float64) * x - 1)
        edge = estimate(
            np.array([0.0, -0.0, 1e-40, np.inf, -np.inf, np.nan], np.float32)
        )
    assert rel.max() < 2.6e-3
    assert list(edge[:3]) == [np.inf, -np.inf, np.inf] and list(edge[3:]) == [
        0.0,
        -0.0,
        0.0,
    ]
    assert np.signbit(edge[4]) and not np.signbit(edge[3])
