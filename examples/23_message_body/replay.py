"""The message body example replayed in numpy: where each thread goes, its state word
and its body after every hop. Pure; no device."""

import numpy as np

MASK = 0xFFFFFFFF


def xorshift(state, key):
    """chain_state of kernels/chain.h on uint64 arrays of 32-bit words."""
    state = state ^ key
    state = state ^ ((state << 13) & MASK)
    state = state ^ (state >> 17)
    return state ^ ((state << 5) & MASK)


def replay(threads, hops, ranks, body_words, change=True):
    """Per rank: arrivals; per thread: its last rank, its state word and its body
    (threads x body_words) after `hops` arrivals, thread t starting at rank t."""
    thread = np.arange(threads, dtype=np.uint64)
    word = np.arange(body_words, dtype=np.uint64)
    body = ((thread[:, None] + 1) * 0x9E3779B1 & MASK) ^ (
        word[None, :] * 0x85EBCA6B & MASK
    )
    state = thread + 1
    arrivals = np.zeros(ranks, dtype=np.int64)
    rank = thread.astype(np.int64)
    for hop in range(hops):
        rank = (rank + 1) % ranks
        arrivals += np.bincount(rank, minlength=ranks)
        key = (hop * 0x9E3779B1 + rank.astype(np.uint64)) & MASK
        state = xorshift(state, key)
        if change:
            rotated = ((body << 1) & MASK) | (body >> 31)
            body = (rotated + (key[:, None] ^ word[None, :])) & MASK
    return arrivals, rank, state, body
