"""Scaled dot-product attention in numpy: the reference the card's attention is
checked against. Inputs are taken as the card holds them (bf16 values); products,
the softmax and sums are float64, so the reference carries no rounding of its own.
"""

import numpy as np

from ref.dense import bf16, errors

TILE = 32
__all__ = ["attention", "errors", "inputs", "sample_rows"]


def inputs(b, h, hk, sq, sk, d, dv, seed=1):
    """q [b, h, sq, d], k [b, hk, sk, d], v [b, hk, sk, dv]: standard normal values
    rounded to bf16, as float32 arrays."""
    rng = np.random.default_rng(seed)
    return tuple(
        bf16(rng.standard_normal(shape, dtype=np.float32))
        for shape in ((b, h, sq, d), (b, hk, sk, d), (b, hk, sk, dv))
    )


def attention(q, k, v, rows, causal, scale=None):
    """softmax(q k^T scale) v of query rows `rows` (indices into q's third axis) of
    one head: q [sq, d], k [sk, d], v [sk, dv]; causal: row i sees keys <= i."""
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    scale = 1.0 / np.sqrt(q.shape[-1]) if scale is None else scale
    s = q[rows] @ k.T * scale
    if causal:
        s = np.where(
            np.arange(k.shape[0])[None, :] > np.asarray(rows)[:, None], -np.inf, s
        )
    p = np.exp(s - s.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)) @ v


def sample_rows(b, h, sqt, count, seed=2):
    """`count` query tile rows (batch, head, tile row) drawn without repetition."""
    total = b * h * sqt
    picks = np.sort(np.random.default_rng(seed).permutation(total)[:count])
    return [(int(p // (h * sqt)), int(p // sqt % h), int(p % sqt)) for p in picks]
