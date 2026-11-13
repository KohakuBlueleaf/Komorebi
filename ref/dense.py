"""The matrix product in numpy: the reference the card's dense project is checked
against. Inputs are taken as the card holds them (bf16 values); products and
sums are float64, so the reference carries no rounding of its own.
"""

import numpy as np

TILE = 32


def bf16(x):
    """x rounded to the nearest bfloat16 (ties to even), returned as float32."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16 << 16
    return b.astype(np.uint32).view(np.float32)


def inputs(m, k, n, seed=1, kind="uniform"):
    """A [m, k] and W [k, n] as float32 arrays of bf16 values: uniform in [-1, 1)
    rounded to bf16, or integers in [-2, 2]."""
    rng = np.random.default_rng(seed)
    if kind == "int":
        return (rng.integers(-2, 3, s).astype(np.float32) for s in ((m, k), (k, n)))
    a = bf16(rng.uniform(-1, 1, (m, k)).astype(np.float32))
    w = bf16(rng.uniform(-1, 1, (k, n)).astype(np.float32))
    return a, w


def sampled_rows(m, count=256, seed=2):
    """The rows a check compares, sorted: all of them when there are at most `count`,
    else count / 32 whole row tiles drawn from a seed (so a card reads few tiles)."""
    if m <= count:
        return np.arange(m)
    tiles = np.random.default_rng(seed).permutation(-(-m // TILE))[: count // TILE]
    rows = (np.sort(tiles)[:, None] * TILE + np.arange(TILE)).reshape(-1)
    return rows[rows < m]


def matmul(a, w):
    """a @ w in float64."""
    return np.asarray(a, dtype=np.float64) @ np.asarray(w, dtype=np.float64)


def errors(got, ref):
    """Max error over the largest |ref| and RMS error over the RMS of ref."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    err = np.abs(got - ref)
    return {
        "max_rel": float(err.max() / np.abs(ref).max()),
        "rms_rel": float(np.sqrt((err**2).mean() / (ref**2).mean())),
    }
