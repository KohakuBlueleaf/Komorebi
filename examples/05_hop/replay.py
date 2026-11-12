"""The hop example replayed in numpy: where every item goes and what each core's
totals are at the end. Pure; no device."""

import numpy as np

M32 = 0xFFFFFFFF


def xorshift(s):
    """hop_next of kernels/hop.h on a uint64 array of 32-bit states."""
    s = s ^ ((s << 13) & M32)
    s = s ^ (s >> 17)
    return s ^ ((s << 5) & M32)


def first_member(items, count):
    """The member that spawns each item: km::share of the items over the team."""
    base, extra = divmod(items, count)
    sizes = np.full(count, base) + (np.arange(count) < extra)
    return np.repeat(np.arange(count), sizes)


def replay(items, hops, count, one_core=None):
    """Per member: items retired there, the 64-bit total of their sums, visits.

    one_core: every hop goes to that member (else to state mod count)."""
    state = (np.arange(items, dtype=np.uint64) + 1) * 0x9E3779B1 & M32
    state[state == 0] = 1
    core = first_member(items, count)
    sums = np.zeros(items, dtype=np.uint64)
    visits = np.zeros(count, dtype=np.int64)
    for h in range(hops + 1):
        visits += np.bincount(core, minlength=count)
        state = xorshift(state)
        sums = (sums + state) & M32
        if h < hops:
            core = np.full(items, one_core) if one_core is not None else state % count
    retired = np.bincount(core, minlength=count)
    total = np.zeros(count, dtype=np.uint64)
    np.add.at(total, core.astype(np.int64), sums)
    return retired, total, visits
