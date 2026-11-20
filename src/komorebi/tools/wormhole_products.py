"""A bit-exact host model of a Wormhole tile product of bfloat16 values into an fp32 Dst
as the tile compiler issues it (in0 to SrcB; per K tile, each phase's terms 0-15 then
16-31), with the chip's erratum: an accumulator of exactly -1 unit writes -16 units
(formats.md, "A sum that cancels to -1 unit"). Usage: product(x, w, "HiFi4").
"""

import numpy as np

PHASES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}


def operands(values):
    """(sign, biased exponent, 10-bit mantissa) of bfloat16 values (float32 arrays)."""
    bits = np.ascontiguousarray(values, np.float32).view(np.uint32)
    if np.any(bits & 0xFFFF):
        raise ValueError("the model's operands are bfloat16 values")
    sign = (bits >> 31).astype(np.int64)
    exponent = ((bits >> 23) & 0xFF).astype(np.int64)
    mantissa = (((bits >> 16) & 0x7F) << 3).astype(np.int64)
    return sign, exponent, mantissa


def _srca_bits(mantissa, phase):  # 5 bits of SrcA, 4 of them fractional
    return (mantissa >> 1) & 0x1F if phase & 1 else 0x10 | (mantissa >> 6)


def _srcb_bits(mantissa, phase):  # 7 bits of SrcB, 6 of them fractional
    return (mantissa & 0xF) << 3 if phase & 2 else 0x40 | (mantissa >> 4)


def _shift_round(value, shift):
    """value / 2^shift to the nearest integer, ties towards positive infinity."""
    shift = np.minimum(shift, 62)
    half = np.where(shift > 0, np.left_shift(np.int64(1), np.maximum(shift - 1, 0)), 0)
    return np.where(shift == 0, value, (value + half) >> shift)


def lane(x_terms, w_terms, phase):
    """One lane's (exponent, sum): its 8 products aligned to their largest exponent
    with 10 fractional bits. x_terms, w_terms: (sign, exponent, mantissa) [..., 8]."""
    bs, be, bm = x_terms
    as_, ae, am = w_terms
    zero = (ae == 0) | (be == 0)
    sums = np.where(zero, 0, ae + be)
    top = sums.max(-1)
    products = np.where(zero, 0, _srca_bits(am, phase) * _srcb_bits(bm, phase))
    shift = top[..., None] - sums
    aligned = (((products << 1) >> np.minimum(shift, 40)) + 1) >> 1
    aligned = np.where(shift >= 16, 0, aligned)
    total = np.where(as_ != bs, -aligned, aligned).sum(-1)
    exponent = (top - 127 - (5 if phase & 1 else 0) - (7 if phase & 2 else 0)) & 0x3FF
    under = (exponent == 0) | (exponent >= 512)
    return np.where(under, 0, exponent), np.where(under, 0, total)


def instruction(r0, r1, dst, erratum=True, seen=None):
    """Dst (float32 values) after one instruction with lanes r0 and r1. `seen`, a dict,
    counts in "hits" the values whose accumulator was exactly -1 and marks them in
    "hit"; "negative" counts negative accumulators, "count" the values."""
    bits = dst.view(np.uint32).astype(np.int64)
    d_exp = (bits >> 23) & 0xFF
    d_sign = np.where(d_exp == 0, 0, bits >> 31)
    d_man = np.where(d_exp == 0, 0, 0x800000 | (bits & 0x7FFFFF))
    exponent = np.maximum(np.maximum(r0[0], r1[0]), d_exp)
    acc = _shift_round(r0[1] << 13, np.minimum(exponent - r0[0], 31))
    acc = acc + _shift_round(r1[1] << 13, np.minimum(exponent - r1[0], 31))
    d_term = _shift_round(d_man, np.minimum(exponent - d_exp, 31))
    acc = acc + np.where(d_sign == 1, -d_term, d_term)
    sign = (acc < 0).astype(np.int64)
    magnitude = np.abs(acc)
    live = (exponent != 0) & (magnitude != 0)
    msb = np.where(magnitude > 0, np.frexp(magnitude.astype(np.float64))[1] - 1, 0)
    result_exp = exponent + msb - 23
    normalized = np.where(live, magnitude << np.maximum(28 - msb, 0), 0)
    hit = (acc == -1) & live
    if seen is not None:
        seen["hits"] = seen.get("hits", 0) + int(hit.sum())
        seen["hit"] = seen.get("hit", np.zeros(acc.shape, bool)) | hit
        seen["negative"] = seen.get("negative", 0) + int((acc < 0).sum())
        seen["count"] = seen.get("count", 0) + acc.size
    if erratum:  # Wormhole: -1 normalizes as if it were -16, the implicit 1 lost
        result_exp = np.where(hit, exponent + 4, result_exp)
        normalized = np.where(hit, 0, normalized)
    rounded = (normalized >> 5) + ((normalized >> 4) & 1)
    carry = (rounded >> 24) != 0
    rounded = np.where(carry, rounded >> 1, rounded)
    result_exp = result_exp + carry
    out = (sign << 31) | (np.clip(result_exp, 0, 255) << 23) | (rounded & 0x7FFFFF)
    out = np.where(~live | (result_exp <= 0), 0, out)
    out = np.where(live & (result_exp >= 255), (sign << 31) | (255 << 23), out)
    return out.astype(np.uint32).view(np.float32)


def tile(x, w, dst, fidelity="HiFi4", erratum=True, seen=None):
    """Dst [M, N] after one K tile: x [M, 32] and w [32, N] bfloat16 values."""
    xs = [part[:, None, :] for part in operands(x)]
    ws = [part.T[None, :, :] for part in operands(w)]
    for phase in range(PHASES[fidelity]):
        for chunk in (0, 1):
            lanes = []
            for half in (0, 1):
                k = slice(16 * chunk + 8 * half, 16 * chunk + 8 * half + 8)
                terms = [p[..., k] for p in xs], [p[..., k] for p in ws]
                lanes.append(lane(*terms, phase))
            dst = instruction(lanes[0], lanes[1], dst, erratum, seen)
    return dst


def l1_add(l1, value):
    """The packer's float32 accumulation in L1: IEEE addition, denormals flushed."""

    def flush(v):
        bits = np.ascontiguousarray(v, np.float32).view(np.uint32)
        kept = np.where((bits & 0x7F800000) == 0, bits & 0x80000000, bits)
        return kept.astype(np.uint32).view(np.float32)

    return flush((flush(l1) + flush(value)).astype(np.float32))


def product(
    x, w, fidelity="HiFi4", block=2, form="field", start=0.0, erratum=True, seen=None
):
    """x [M, K] @ w [K, N] (bfloat16 values) in blocks of `block` K tiles: "field" sums
    each block from a fresh Dst into L1 (`self.sums += f(xb, wb)`), "dst" keeps one Dst
    (`acc += f(xb, wb)`). `start`: each fresh Dst's value (0, or a positive bias)."""
    rows, depth = x.shape
    fresh = np.full((rows, w.shape[1]), start, np.float32)
    l1, dst = None, fresh.copy()
    for first in range(0, depth // 32, block):
        if form == "field":
            dst = fresh.copy()
        for kt in range(first, first + block):
            ks = slice(32 * kt, 32 * kt + 32)
            dst = tile(x[:, ks], w[ks], dst, fidelity, erratum, seen)
        if form == "field":
            l1 = dst.copy() if l1 is None else l1_add(l1, dst)
    return l1 if form == "field" else dst
