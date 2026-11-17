"""The card's tile formats, one tile a case (README.md): the bits each fidelity
multiplies a side, the A family through unpack and pack, the host's block encode
against the packer, odd offsets, the transposed product. `build` builds every case on
a mock device and checks the model's own numbers; `card` runs them on one card."""

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.device import native
from komorebi.program import several
from komorebi.session import kernels as session_kernels

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
RESULTS = HERE / "results"
# The runtime's kernel cache: `build` fills it on the mock device and a card job
# reads it (the same build key on the card), so no kernel is compiled in a job.
JIT_CACHE = HERE.parents[2] / "build" / "ttcache-formats"
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
CB_A, CB_B, CB_OUT, CB_INFO = 0, 1, 16, 24  # layout.hpp
NONCE, ERROR, COUNT, PRODUCT, TRANSPOSE, SYNC, SYNCS = 0, 1, 2, 16, 32, 48, 8
TILE_WORDS = 1024
FIDELITIES = ("LoFi", "HiFi2", "HiFi3", "HiFi4")
PHASES = {"LoFi": (0,), "HiFi2": (0, 1), "HiFi3": (0, 1, 2), "HiFi4": (0, 1, 2, 3)}
# A format's explicit mantissa bits as the registers hold them (fp32 is truncated to
# tf32's 10), and its family (None: goes with either).
EXPLICIT = {
    "float32": 10,
    "bfloat16": 7,
    "float16": 10,
    "bfp8_b": 6,
    "bfp4_b": 2,
    "bfp2_b": 0,
    "bfp8": 6,
    "bfp4": 2,
    "bfp2": 0,
}
BLOCK = {"bfp8": 8, "bfp4": 4, "bfp2": 2}
PROBE_BITS = 12  # the bits set one at a time (fp32's 11 and 12 are cut by tf32)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def tile_bytes(fmt):
    base = fmt.removesuffix("_b")
    if base in BLOCK:
        return formats.bfp_tile_bytes(BLOCK[base])
    return {"float32": 4096, "bfloat16": 2048, "float16": 2048}[fmt]


def encode(tiles, fmt):
    """[N, 1024] float32 tiles (face order) as [N, tile bytes] uint8 of `fmt`."""
    tiles = np.ascontiguousarray(tiles, np.float32).reshape(-1, TILE_WORDS)
    base = fmt.removesuffix("_b")
    if base in BLOCK:
        return formats.bfp_encode(tiles, BLOCK[base], family_a=not fmt.endswith("_b"))
    if fmt == "bfloat16":
        bits = formats.bf16_bits(tiles)
    elif fmt == "float16":
        bits = tiles.astype(np.float16)
    else:
        bits = tiles
    return np.ascontiguousarray(bits).view(np.uint8).reshape(len(tiles), -1)


def decode(raw, fmt):
    """[N, tile bytes] uint8 of `fmt` as [N, 1024] float32 tiles."""
    raw = np.ascontiguousarray(raw, np.uint8).reshape(-1, tile_bytes(fmt))
    base = fmt.removesuffix("_b")
    if base in BLOCK:
        return formats.bfp_decode(raw, BLOCK[base], family_a=not fmt.endswith("_b"))
    if fmt == "bfloat16":
        return formats.bf16_values(raw.view(np.uint16))
    if fmt == "float16":
        return raw.view(np.float16).astype(np.float32)
    return raw.view(np.float32).copy()


def tile(matrix):
    return formats.tilize(np.asarray(matrix, np.float32))


def matrix(tiles):
    return formats.untilize(np.asarray(tiles).reshape(1, TILE_WORDS), 32, 32)


# --- The host model of the report's tables ---------------------------------------


def significand(values):
    """(sign, 11-bit significand of the register's datum, exponent): the value is
    sign * m * 2 ** (e - 10), the mantissa truncated to the register's 10 bits."""
    v = np.asarray(values, np.float32)
    bits = v.view(np.uint32).astype(np.int64)
    exp = (bits >> 23) & 0xFF
    m = np.where(exp == 0, 0, (1 << 10) | ((bits >> 13) & 0x3FF))
    return np.where(bits >> 31 == 1, -1.0, 1.0), m, exp - 127


def slices(values, side):
    """The two slices of each datum a phase multiplies, as values: SrcA the top 5
    significand bits and the next 5 (its lowest never), SrcB the top 7 and the low 4."""
    sign, m, e = significand(values)
    scale = sign * np.ldexp(1.0, e - 10)
    if side == "a":
        return (m >> 6 << 6) * scale, ((m >> 1) & 0x1F) * 2.0 * scale
    return (m >> 4 << 4) * scale, (m & 0xF) * scale


def model_product(left, right, fidelity):
    """C = left @ right as the matrix unit's phases make it: left in SrcB, right in
    SrcA; phase 0 a_hi*b_hi, 1 a_lo*b_hi, 2 a_hi*b_lo, 3 a_lo*b_lo (float64 sums)."""
    a_hi, a_lo = slices(right, "a")
    b_hi, b_lo = slices(left, "b")
    pairs = {0: (a_hi, b_hi), 1: (a_lo, b_hi), 2: (a_hi, b_lo), 3: (a_lo, b_lo)}
    out = np.zeros((left.shape[0], right.shape[1]))
    for phase in PHASES[fidelity]:
        a, b = pairs[phase]
        out += b @ a
    return out


def bits_tiles(left_fmt, right_fmt):
    """The probe's left and right matrices: rows 0..11 one left bit each against a
    right row of ones; row 12.. a left one against a right row of one bit a column;
    rows 16..27 one left bit each against that row (the cross terms)."""
    left, right = np.zeros((32, 32)), np.zeros((32, 32))
    lbits = min(PROBE_BITS, 12 if left_fmt == "float32" else EXPLICIT[left_fmt])
    rbits = min(PROBE_BITS, 12 if right_fmt == "float32" else EXPLICIT[right_fmt])
    for i in range(12):
        left[i, 0] = 1 + 2.0 ** -(i + 1) if i < lbits else 1.0
        left[16 + i, 2] = 1 + 2.0 ** -(i + 1) if i < lbits else 1.0
    left[12:16, 1] = 1.0
    right[0, :] = 1.0
    right[1, :] = right[2, :] = 1.0
    for j in range(12):
        if j < rbits:
            right[1, j] = right[2, j] = 1 + 2.0 ** -(j + 1)
    return left, right


def bits_used(c, left, right):
    """The bits the card multiplied: each left bit (rows 0..11, column 0) and each
    right bit (row 12, columns 0..11) that reached C; and the cross table."""
    left_used = [m + 1 for m in range(12) if left[m, 0] != 1 and c[m, 0] == left[m, 0]]
    right_used = [
        m + 1 for m in range(12) if right[1, m] != 1 and c[12, m] == right[1, m]
    ]
    cross = (c[16:28, :12] - left[16:28, 2:3] * right[2:3, :12]) == 0
    return left_used, right_used, cross


def card_bfp(tiles, bits, family_a, early):
    """The report's packer: the row's largest exponent shared, each datum rounded
    to the 7-bit magnitude (nearest, ties away), then cut to the width; `early`
    rounds each datum to 7 significant bits at its own exponent first."""
    words = np.ascontiguousarray(tiles, np.float32).view(np.uint32)
    words = words.reshape(-1, 64, 16).astype(np.int64)
    exp = (words >> 23) & 0xFF
    man = (1 << 23) | (words & 0x7FFFFF)
    sign = words >> 31
    zero = exp == 0
    if early:  # E8M6: 7 significant bits, nearest, ties away
        man = (man + (1 << 16)) >> 17 << 17
        carry = man >> 24
        man >>= carry
        exp = exp + carry
    if family_a:
        exp = np.clip(exp - 127 + 15, 0, 31)
    shared = np.where(zero, 0, exp).max(axis=-1, keepdims=True)
    man = man >> np.minimum(shared - exp, 31)
    m7 = np.minimum((man + (1 << 16)) >> 17, 127)
    m = m7 >> (7 - formats.BFP_WIDTH[bits])
    m = np.where(zero, 0, m)
    sign = np.where(m == 0, 0, sign)
    codes = (sign << formats.BFP_WIDTH[bits]) | m
    return formats.bfp_pack_codes(shared[..., 0], codes.reshape(-1, 1024), bits)


# --- The cases --------------------------------------------------------------------


def case(name, kind, left, right, out, fidelity="HiFi4", **extra):
    return {
        "name": name,
        "kind": kind,
        "left": left,
        "right": right,
        "out": out,
        "fidelity": fidelity,
        **extra,
    }


def cases():
    """Every program, the known path (bf16 x bf16) first."""
    pairs = [("bfloat16", "bfloat16")]
    pairs += [("bfloat16", f) for f in ("bfp8_b", "bfp4_b", "bfp2_b")]
    pairs += [(f, "bfloat16") for f in ("bfp8_b", "bfp4_b", "bfp2_b")]
    pairs += [("float32", "float32"), ("float16", "float16")]
    out = [
        case(f"bits {left} x {right} {fid}", "bits", left, right, "float32", fid)
        for left, right in pairs
        for fid in FIDELITIES
    ]
    out.append(case("bits bfp8 x float16 HiFi4", "bits", "bfp8", "float16", "float32"))
    out += [
        case(
            f"offset bfloat16 x {fmt}",
            "offset",
            "bfloat16",
            fmt,
            "float32",
            first=1,
            count=2,
            right_tiles=3,
        )
        for fmt in ("bfp4_b", "bfp2_b")
    ]
    out += [
        case(f"{form} {fid}", form, "bfloat16", "bfloat16", "float32", fid)
        for fid in ("LoFi", "HiFi2", "HiFi4")
        for form in ("plain", "transposed")
    ]
    copies = [("float16", "float32"), ("float16", "float16"), ("bfp8", "float32")]
    copies.append(("bfp8", "bfp8"))
    out += [case(f"copy {s} to {d}", "copy", s, None, d) for s, d in copies]
    packs = [(d, p) for d in ("bfp8_b", "bfp4_b", "bfp2_b") for p in (False, True)]
    packs.append(("bfp8", False))
    out += [
        case(
            f"pack float32 to {d} precise {int(p)}",
            "pack",
            "float32",
            None,
            d,
            precise=p,
        )
        for d, p in packs
    ]
    return out


def cases_a():
    """The second set: the copy path's controls in the B family, the A family with a
    16-bit Dst, fp32 Dst packed to fp16, and products of A operands of every scale."""
    out = [
        case("copy bfloat16 to float32", "copy", "bfloat16", None, "float32"),
        case("copy bfloat16 to bfloat16", "copy", "bfloat16", None, "bfloat16"),
        case("pack float32 to float16", "pack", "float32", None, "float16"),
        case(
            "copy float16 to float32 unpacked to Dst",
            "copy",
            "float16",
            None,
            "float32",
            to_dest=True,
        ),
    ]
    copies = [("float16", "float16"), ("float16", "float32"), ("bfp8", "bfp8")]
    copies.append(("bfp8", "float16"))
    out += [
        case(f"copy {s} to {d} dst16", "copy", s, None, d, dst32=False)
        for s, d in copies
    ]
    out += [
        case("product float16 x float16", "product", "float16", "float16", "float32"),
        case("product bfp8 x float16", "product", "bfp8", "float16", "float32"),
        case(
            "product float16 x float16 dst16",
            "product",
            "float16",
            "float16",
            "float16",
            dst32=False,
        ),
    ]
    return out


SETS = {"1": cases, "2": cases_a}


def p4_values():
    """An fp16 tile: ordinary values of three scales, and the extremes in face 0's
    first row (largest, smallest normal, subnormals, infinities, NaN, signed zero)."""
    rng = np.random.default_rng(4)
    values = rng.standard_normal(TILE_WORDS) * np.repeat([1.0, 100.0, 0.01, 1.0], 256)
    extremes = [65504.0, -65504.0, 2.0**-14, -(2.0**-14), 2.0**-24, 2.0**-20, 1e-5]
    extremes += [np.inf, -np.inf, np.nan, -0.0, 0.0, 1 / 3, 2.0**15, 0.5 + 2**-11, 7.0]
    values[:16] = extremes
    return values.astype(np.float16).astype(np.float32).reshape(1, TILE_WORDS)


def bfp8a_values():
    """A tile for the A family's bfp8: ordinary rows, and rows at its exponent's ends
    (65024 its largest, 2 ** -14 and below its smallest, above the largest)."""
    rng = np.random.default_rng(5)
    values = rng.standard_normal(TILE_WORDS).reshape(64, 16)
    values[0] = [65024.0, 1.0, -2.0, 0.5] + [0.0] * 12
    values[1] = [2.0**-14, 2.0**-15, 2.0**-16, -(2.0**-14)] + [0.0] * 12
    values[2] = [70000.0, -1e6, 1.0, 3.0] + [0.0] * 12
    values[3] = rng.standard_normal(16) * 1e-3
    return values.astype(np.float32).reshape(1, TILE_WORDS)


def p6_values():
    """An fp32 tile for the packers: random rows; rows of ties for bfp8 and for
    bfp4 / bfp2 under a largest of 1; small negatives that round to zero; a wide
    spread of exponents; magnitudes that round over the largest; low fp32 bits."""
    rng = np.random.default_rng(6)
    rows = rng.standard_normal((64, 16))
    for r in range(16, 24):  # bfp8 ties: (k + 1/2) / 64 under a largest of 1
        k = rng.integers(0, 63, 15)
        rows[r] = [1.0, *((k + 0.5) / 64 * rng.choice([-1, 1], 15))]
    for r in range(24, 32):  # bfp4 ties at (k + 1/2) / 4, bfp2 ties at 1/2
        k = rng.integers(0, 3, 15)
        rows[r] = [1.0, *((k + 0.5) / 4 * rng.choice([-1, 1], 15))]
        rows[r, 1:4] = [0.5, -0.5, 0.5]
    for r in range(32, 40):  # negatives under half a step, at half, over half
        rows[r] = [
            1.0,
            -(2.0**-8),
            -(2.0**-7),
            -(2.0**-7) * 1.01,
            -(2.0**-3),
        ] + [-(2.0**-9)] * 11
    for r in range(40, 48):
        rows[r] = rng.standard_normal(16) * 10.0 ** rng.integers(-3, 4, 16)
        rows[r, 3] = 0.0
    for r in range(48, 56):  # a largest just under 2: rounds up over 127/64
        rows[r] = [1.999, 1.996, -1.9999, *rng.uniform(-2, 2, 13)]
    for r in range(56, 64):  # bits under bf16's: the packer sees Dst's fp32
        rows[r] = (1.0 + rng.integers(0, 2**23, 16) * 2.0**-23) * rng.choice(
            [-1, 1], 16
        )
    return rows.astype(np.float32).reshape(1, TILE_WORDS)


def inputs(c, seed=1):
    """The case's input bytes (left, right), the values they hold, and its model."""
    kind = c["kind"]
    if kind == "bits":
        left, right = bits_tiles(c["left"], c["right"])
        lt, rt = encode(tile(left), c["left"]), encode(tile(right), c["right"])
        lv, rv = matrix(decode(lt, c["left"])), matrix(decode(rt, c["right"]))
        ref = model_product(lv, rv, c["fidelity"])
        return {"left": lt, "right": rt, "lv": lv, "rv": rv, "ref": [ref]}
    rng = np.random.default_rng(seed)
    if kind == "offset":
        lv = matrix(
            decode(encode(tile(rng.standard_normal((32, 32))), "bfloat16"), "bfloat16")
        )
        rt = encode(
            np.stack([tile(rng.standard_normal((32, 32)))[0] for _ in range(3)]),
            c["right"],
        )
        rvs = [formats.untilize(decode(rt, c["right"])[t], 32, 32) for t in range(3)]
        refs = [model_product(lv, rvs[t], "HiFi4") for t in (1, 2)]
        return {
            "left": encode(tile(lv), "bfloat16"),
            "right": rt,
            "ref": refs,
            "exact": [lv @ rvs[t] for t in (1, 2)],
        }
    if kind in ("plain", "transposed"):
        x = matrix(
            decode(encode(tile(rng.standard_normal((32, 32))), "bfloat16"), "bfloat16")
        ).astype(np.float64)
        w = matrix(
            decode(encode(tile(rng.standard_normal((32, 32))), "bfloat16"), "bfloat16")
        ).astype(np.float64)
        if kind == "plain":
            left, right = x, w
            ref = model_product(x, w, c["fidelity"])
        else:
            left, right = w.T, x  # the right tile is read transposed: X^T
            ref = model_product(w.T, x.T, c["fidelity"]).T
        return {
            "left": encode(tile(left), "bfloat16"),
            "right": encode(tile(right), "bfloat16"),
            "ref": [ref],
            "exact": [x @ w],
        }
    if kind == "product":  # values of four scales a face, none under fp16's normals
        scale = np.repeat([1.0, 30.0, 0.01, 1e-3], 256)
        lt = encode(rng.standard_normal((1, TILE_WORDS)) * scale, c["left"])
        rt = encode(rng.standard_normal((1, TILE_WORDS)) * scale[::-1], c["right"])
        lv, rv = matrix(decode(lt, c["left"])), matrix(decode(rt, c["right"]))
        return {
            "left": lt,
            "right": rt,
            "ref": [model_product(lv, rv, "HiFi4")],
            "exact": [lv.astype(np.float64) @ rv],
        }
    if kind == "copy" and c["left"] == "bfloat16":
        values = rng.standard_normal((1, TILE_WORDS)) * 10.0 ** rng.integers(
            -6, 6, 1024
        )
        raw = encode(values, "bfloat16")
        return {"left": raw, "right": raw, "values": decode(raw, "bfloat16")}
    if kind == "pack" and c["out"] == "float16":
        values = (p4_values() * 1.0001).astype(np.float32)
        values = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
        return {"left": encode(values, "float32"), "right": None, "values": values}
    if kind == "copy":
        values = p4_values() if c["left"] == "float16" else bfp8a_values()
        raw = encode(values, c["left"])
        return {"left": raw, "right": raw, "values": decode(raw, c["left"])}
    values = p6_values()
    return {"left": encode(values, "float32"), "right": None, "values": values}


def launch(session, c, data):
    """One program on core (0, 0) for case `c`: (the Launch, out, info, nonce)."""
    nonce = secrets.randbits(32) | 1
    count = c.get("count", 1)
    right_tiles = c.get("right_tiles", 1) if c["right"] else 0
    left = session.tensor(data["left"], "uint8")
    right = session.tensor(data["right"] if c["right"] else data["left"], "uint8")
    out_bytes = tile_bytes(c["out"])
    t_out = session.tensor(np.zeros((count, out_bytes), np.uint8), "uint8")
    t_info = session.tensor(np.zeros((1, TILE_WORDS), np.uint32), "uint32")
    core = [(0, 0, 0, 0)]
    run = session.launch()
    page = tile_bytes(c["left"])
    run.cb(core, CB_A, c["left"], page, page)
    if c["right"]:
        page = tile_bytes(c["right"])
        run.cb(core, CB_B, c["right"], page, right_tiles * page)
    run.cb(core, CB_OUT, c["out"], out_bytes, 2 * out_bytes)
    run.cb(core, CB_INFO, "float32", 4 * TILE_WORDS, 4 * TILE_WORDS)
    accessors = left.accessor_args() + right.accessor_args()
    words = [left.address, right.address, 1, right_tiles]
    run.kernel(str(KERNELS / "reader.cpp"), core, 1, accessors, [], [], [(0, 0, words)])
    accessors = t_out.accessor_args() + t_info.accessor_args()
    words = [t_out.address, t_info.address, count]
    run.kernel(str(KERNELS / "writer.cpp"), core, 0, accessors, [], [], [(0, 0, words)])
    kind = c["kind"]
    form = {"transposed": 2, "copy": 1, "pack": 1}.get(kind, 0)
    options = {"math_fidelity": c["fidelity"], "fp32_dest_acc_en": c.get("dst32", True)}
    if c.get("precise"):
        options["bfp8_pack_precise"] = True
    if kind == "pack" or c.get("to_dest"):
        options["unpack_to_dest_mode"] = {CB_A: "UnpackToDestFp32"}
    run.kernel(
        str(KERNELS / "compute.cpp"),
        core,
        2,
        [],
        [("CASE", str(form))],
        [],
        [(0, 0, [nonce, c.get("first", 0), count])],
        **several.compute_config(options),
    )
    return run, t_out, t_info, nonce


# --- The checks -------------------------------------------------------------------


def _spans(info, nonce, count, transposed):
    empty = float(np.median(info[SYNC : SYNC + SYNCS]))
    out = {
        "nonce_ok": int(info[NONCE]) == nonce,
        "error": int(info[ERROR]),
        "empty_span": empty,
        "product_span": [int(v) - empty for v in info[PRODUCT : PRODUCT + count]],
    }
    if transposed:
        out["transpose_span"] = [
            int(v) - empty for v in info[TRANSPOSE : TRANSPOSE + count]
        ]
    return out


def _rel(got, ref):
    err = np.abs(np.asarray(got, np.float64) - ref)
    scale = np.sqrt(np.mean(np.square(ref)))
    return {
        "rms_rel": float(np.sqrt(np.mean(err**2)) / scale),
        "max_rel": float(err.max() / scale),
    }


def check(c, data, raw):
    """The record of case `c` from its output bytes [count, tile bytes]."""
    kind, rec = c["kind"], {}
    if kind == "bits":
        got = matrix(decode(raw, c["out"])).astype(np.float64)
        left_used, right_used, cross = bits_used(got, data["lv"], data["rv"])
        rec["left_bits"] = left_used
        rec["right_bits"] = right_used
        rec["left_significand"] = 1 + len(left_used)
        rec["right_significand"] = 1 + len(right_used)
        rec["cross_terms"] = int(cross.sum())
        rec["model_mismatch"] = int(np.count_nonzero(got != data["ref"][0]))
        rec["first_mismatch"] = [
            [int(i), int(j), float(got[i, j]), float(data["ref"][0][i, j])]
            for i, j in np.argwhere(got != data["ref"][0])[:4]
        ]
        return rec
    if kind == "pack" and c["out"] == "float16":
        got = decode(raw, "float16")[0]
        want = data["values"][0].astype(np.float16)  # IEEE half, nearest even
        rec["differ_from_ieee"] = int(np.count_nonzero(got != want.astype(np.float32)))
        cut = (data["values"][0].view(np.uint32) >> 13 << 13).view(np.float32)
        rec["differ_from_truncation"] = int(
            np.count_nonzero(got != cut.astype(np.float16).astype(np.float32))
        )
        rec["examples"] = [
            [float(data["values"][0][i]), float(got[i])] for i in range(16)
        ]
        return rec
    if kind in ("offset", "plain", "transposed", "product"):
        outs = decode(raw, c["out"])
        rec["tiles"] = []
        for t, ref in enumerate(data["ref"]):
            got = formats.untilize(outs[t], 32, 32).astype(np.float64)
            rec["tiles"].append(
                {"model": _rel(got, ref), "exact": _rel(got, data["exact"][t])}
            )
        return rec
    if kind == "copy":
        got = decode(raw, c["out"])[0]
        want = data["values"][0]
        same = (got == want) | (np.isnan(got) & np.isnan(want))
        same &= np.signbit(got) == np.signbit(want)
        rec["differ"] = int(np.count_nonzero(~same))
        rec["examples"] = [
            [float(want[i]), float(got[i])] for i in np.flatnonzero(~same)[:12]
        ]
        if c["out"] == c["left"]:
            rec["bytes_differ"] = int(np.count_nonzero(raw[0] != data["left"][0]))
        if c["left"] == "float16":
            rec["extremes"] = [[float(want[i]), float(got[i])] for i in range(16)]
        return rec
    bits = BLOCK[c["out"].removesuffix("_b")]
    family_a = not c["out"].endswith("_b")
    card_exp, card_codes = formats.bfp_unpack_codes(raw, bits)
    width = formats.BFP_WIDTH[bits]
    rec["negative_zero"] = int(np.count_nonzero(card_codes == 1 << width))
    models = {
        "host": formats.bfp_encode(data["values"], bits, family_a),
        "away": card_bfp(data["values"], bits, family_a, early=False),
        "early_away": card_bfp(data["values"], bits, family_a, early=True),
    }
    for name, want in models.items():
        exp, codes = formats.bfp_unpack_codes(want, bits)
        diff = np.flatnonzero(codes[0] != card_codes[0])
        rec[name] = {
            "exponents_differ": int(np.count_nonzero(exp != card_exp)),
            "datums_differ": int(len(diff)),
            "rows": sorted({int(i) // 16 for i in diff})[:16],
        }
    got = formats.bfp_decode(raw, bits, family_a)[0]
    rec["values_error"] = _rel(got, data["values"][0].astype(np.float64))
    return rec


def fake_output(c, data):
    """The build's stand-in for the card: the model's own bytes."""
    if c["kind"] in ("bits", "offset", "plain", "transposed", "product"):
        return np.concatenate([encode(tile(r), c["out"]) for r in data["ref"]])
    if c["kind"] == "copy" or c["out"] == "float16":
        return encode(data["values"], c["out"])
    bits = BLOCK[c["out"].removesuffix("_b")]
    return formats.bfp_encode(data["values"], bits, not c["out"].endswith("_b"))


def native_against_numpy(session):
    """tt-metal's bound block-float encode and decode against formats.py's, bit for
    bit, on the probe's tiles (a mock device: the bound calls ask the runtime's HAL)."""
    module = native.load_module(session.tree)
    out = {}
    tiles = np.concatenate([p6_values(), bfp8a_values(), p4_values()])
    tiles = np.where(np.isfinite(tiles), tiles, 0.0).astype(np.float32)
    for bits in (8, 4, 2):
        for family_a in (False, True):
            ours = formats.bfp_encode(tiles, bits, family_a)
            theirs = np.asarray(module.bfp_encode(tiles.reshape(-1), bits, family_a))
            theirs = theirs.view(np.uint8).reshape(ours.shape)
            back = np.asarray(
                module.bfp_decode(theirs.reshape(-1).view(np.uint32), bits, family_a)
            )
            key = f"bfp{bits}{'' if family_a else '_b'}"
            out[key] = {
                "encode_bytes_differ": int(np.count_nonzero(ours != theirs)),
                "decode_differ": int(
                    np.count_nonzero(
                        back.reshape(-1)
                        != formats.bfp_decode(ours, bits, family_a).reshape(-1)
                    )
                ),
            }
    return out


def build(args):
    """Mock device: every case built by the runtime's JIT (its kernel buffer), and
    the checks run on the model's own bytes."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    out = {}
    with km.Session() as session:
        if args.set == "1":
            out["native"] = native_against_numpy(session)
            log(f"native against numpy: {json.dumps(out['native'])}")
        for c in SETS[args.set]():
            data = inputs(c)
            run, *_ = launch(session, c, data)
            _, *built = run.build()
            got = session_kernels.kernel_sizes(built)
            rec = check(c, data, fake_output(c, data))
            log(f"built {c['name']}: {got.total} of {got.limit} B; {json.dumps(rec)}")
    print(json.dumps(out), flush=True)
    return 0


def card(args):
    """One core: every case in turn; its record and spans."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    started = time.time()
    records, saved = {}, {}
    with km.Session() as session:
        log(f"device open {time.time() - started:.1f}s")
        for c in SETS[args.set]():
            data = inputs(c)
            run, t_out, t_info, nonce = launch(session, c, data)
            log(f"launch {c['name']}")
            run.run()
            session.device.finish()
            raw = t_out.read().reshape(c.get("count", 1), -1).copy()
            info = t_info.read().reshape(-1).copy()
            rec = check(c, data, raw)
            rec.update(
                _spans(info, nonce, c.get("count", 1), c["kind"] == "transposed")
            )
            log(f"{c['name']}: {json.dumps(rec)}")
            records[c["name"]] = rec
            saved[c["name"]] = raw
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"card{args.set}.json", "w") as f:
        json.dump(
            {"cases": records, "wall_s": round(time.time() - started, 1)}, f, indent=1
        )
    if args.save:
        np.savez(args.save, **saved)
    clean = all(r["nonce_ok"] and r["error"] == 0 for r in records.values())
    return 0 if clean else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["build", "card"])
    p.add_argument("--set", choices=SETS, default="1", help="the set of cases")
    p.add_argument("--save", help="card: the output bytes into this .npz file")
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
