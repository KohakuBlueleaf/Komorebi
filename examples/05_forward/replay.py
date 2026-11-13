"""The forward example replayed in numpy: where every message goes and what each core's
totals are at the end. Pure; no device."""

import numpy as np

M32 = 0xFFFFFFFF


def xorshift(s):
    """forward_next of kernels/forward.h on a uint64 array of 32-bit states."""
    s = s ^ ((s << 13) & M32)
    s = s ^ (s >> 17)
    return s ^ ((s << 5) & M32)


def first_rank(messages, count):
    """The core that sends each message: km::partition of the messages over the
    group."""
    base, extra = divmod(messages, count)
    sizes = np.full(count, base) + (np.arange(count) < extra)
    return np.repeat(np.arange(count), sizes)


def replay(messages, forwards, count, one_core=None):
    """Per core: messages consumed there, the 64-bit total of their sums, visits.

    one_core: every forward goes to that rank (else to state mod count)."""
    state = (np.arange(messages, dtype=np.uint64) + 1) * 0x9E3779B1 & M32
    state[state == 0] = 1
    core = first_rank(messages, count)
    sums = np.zeros(messages, dtype=np.uint64)
    visits = np.zeros(count, dtype=np.int64)
    for h in range(forwards + 1):
        visits += np.bincount(core, minlength=count)
        state = xorshift(state)
        sums = (sums + state) & M32
        if h < forwards:
            core = (
                np.full(messages, one_core) if one_core is not None else state % count
            )
    consumed = np.bincount(core, minlength=count)
    total = np.zeros(count, dtype=np.uint64)
    np.add.at(total, core.astype(np.int64), sums)
    return consumed, total, visits
