"""Piece 16 on the host: the request's state, where it lives in the reference, and how
to clear it (README.md, `driver/state.py`)."""

import torch


def state_tensors(ref, i):
    """The state tensors of layer i, by name (only those the layer has)."""
    attn = ref.model.layers[i].attn
    out = {"window_kv_cache": attn.window_kv_cache}
    if attn.is_kv_source:
        out["compress_kv_cache"] = attn.compress_kv_cache
        if attn.compressor.compress_ratio > 1:
            out["kv_state"] = attn.compressor.kv_state
            out["score_state"] = attn.compressor.score_state
    if attn.indexer is not None and attn.indexer.owns_k:
        out["index_k_cache"] = attn.indexer.k_cache
    return out


def reset(ref):
    """Clear the host state for a new request."""
    for i in range(len(ref.model.layers)):
        for name, t in state_tensors(ref, i).items():
            t.fill_(-torch.inf if name == "score_state" else 0)
    ref.model.engram_hash.cache.zero_()
    sa = ref.mod.shared_attn
    sa.compress_kv = sa.index_k = sa.topk_idxs = sa.candidates = None
