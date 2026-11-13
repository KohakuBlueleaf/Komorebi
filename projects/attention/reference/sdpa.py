"""Scaled dot-product attention in numpy: the reference the card's attention is
checked against. Inputs are taken as the card holds them (bf16 values); products,
the softmax and sums are float64, so the reference carries no rounding of its own.
"""

import numpy as np

from reference.dense import bf16, errors

TILE = 32
__all__ = ["attention", "errors", "inputs", "sample_rows"]


def inputs(
    batches, heads, kv_heads, query_length, key_length, head_dim, value_dim, seed=1
):
    """queries [batches, heads, query_length, head_dim], keys [batches, kv_heads,
    key_length, head_dim], values [batches, kv_heads, key_length, value_dim]:
    standard normal values rounded to bf16, as float32 arrays."""
    generator = np.random.default_rng(seed)
    return tuple(
        bf16(generator.standard_normal(shape, dtype=np.float32))
        for shape in (
            (batches, heads, query_length, head_dim),
            (batches, kv_heads, key_length, head_dim),
            (batches, kv_heads, key_length, value_dim),
        )
    )


def attention(queries, keys, values, rows, causal, scale=None):
    """softmax(queries keys^T scale) values of query rows `rows` (indices into the
    queries' first axis) of one head: queries [query_length, head_dim], keys
    [key_length, head_dim], values [key_length, value_dim]; causal: row i sees keys
    <= i."""
    queries, keys, values = (
        np.asarray(x, dtype=np.float64) for x in (queries, keys, values)
    )
    scale = 1.0 / np.sqrt(queries.shape[-1]) if scale is None else scale
    scores = queries[rows] @ keys.T * scale
    if causal:
        scores = np.where(
            np.arange(keys.shape[0])[None, :] > np.asarray(rows)[:, None],
            -np.inf,
            scores,
        )
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return (weights / weights.sum(axis=1, keepdims=True)) @ values


def sample_rows(batches, heads, sqt, count, seed=2):
    """`count` query tile rows (batch, head, tile row) drawn without repetition."""
    total = batches * heads * sqt
    picks = np.sort(np.random.default_rng(seed).permutation(total)[:count])
    return [
        (int(pick // (heads * sqt)), int(pick // sqt % heads), int(pick % sqt))
        for pick in picks
    ]
