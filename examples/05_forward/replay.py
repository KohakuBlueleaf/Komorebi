"""The forward example replayed in numpy: where every message goes and what each core's
totals are at the end. Pure; no device."""

import numpy as np

MASK_32_BITS = 0xFFFFFFFF


def xorshift(state):
    """forward_next of kernels/forward.h on a uint64 array of 32-bit states."""
    state = state ^ ((state << 13) & MASK_32_BITS)
    state = state ^ (state >> 17)
    return state ^ ((state << 5) & MASK_32_BITS)


def first_rank(messages, count):
    """The core that sends each message: km::partition of the messages over the
    group."""
    base, extra = divmod(messages, count)
    sizes = np.full(count, base) + (np.arange(count) < extra)
    return np.repeat(np.arange(count), sizes)


def replay(messages, forwards, count, one_core=None):
    """Per core: messages consumed there, the 64-bit total of their sums, visits.

    one_core: every forward goes to that rank (else to state mod count)."""
    state = (np.arange(messages, dtype=np.uint64) + 1) * 0x9E3779B1 & MASK_32_BITS
    state[state == 0] = 1
    core = first_rank(messages, count)
    sums = np.zeros(messages, dtype=np.uint64)
    visits = np.zeros(count, dtype=np.int64)
    for hop in range(forwards + 1):
        visits += np.bincount(core, minlength=count)
        state = xorshift(state)
        sums = (sums + state) & MASK_32_BITS
        if hop < forwards:
            core = (
                np.full(messages, one_core) if one_core is not None else state % count
            )
    consumed = np.bincount(core, minlength=count)
    total = np.zeros(count, dtype=np.uint64)
    np.add.at(total, core.astype(np.int64), sums)
    return consumed, total, visits
