"""The message handlers example replayed in numpy: every thread's path through its three
handlers, and each core's counts and sums at the end. Pure; no device."""

import numpy as np

MASK = 0xFFFFFFFF
LOOKUP, VALUE, REPORT = 1, 2, 3


def table_mix(value):
    """table_mix of kernels/handlers.h on a uint64 array of 32-bit values."""
    value = value ^ (value >> 16)
    value = (value * 0x7FEB352D) & MASK
    value = value ^ (value >> 15)
    value = (value * 0x846CA68B) & MASK
    return value ^ (value >> 16)


def origins(threads, count):
    """The rank that seeds each thread: km::partition of the threads over the group."""
    base, extra = divmod(threads, count)
    sizes = np.full(count, base) + (np.arange(count) < extra)
    return np.repeat(np.arange(count), sizes)


def replay(threads, steps, count):
    """Per rank: {"lookups", "values", "reports", "own", "sum", "probes"} arrays."""
    origin = origins(threads, count).astype(np.uint64)
    key = table_mix(np.arange(threads, dtype=np.uint64) + 1)
    core = key % count
    handler = np.full(threads, LOOKUP)
    value = np.zeros(threads, np.uint64)
    left = np.full(threads, steps, np.int64)
    total = np.zeros(threads, np.uint64)
    probes = np.zeros(threads, np.int64)
    out = {name: np.zeros(count, np.int64) for name in ("lookups", "values", "reports")}
    out |= {"own": np.zeros(count, np.int64), "probes": np.zeros(count, np.int64)}
    out["sum"] = np.zeros(count, np.uint64)
    live = np.ones(threads, bool)
    while live.any():
        destination = core.copy()
        changed = handler.copy()
        here = live & (handler == LOOKUP)
        np.add.at(out["lookups"], core[here].astype(np.int64), 1)
        probes[here] += 1
        holds = ((table_mix(key ^ (core * 0x9E3779B1 & MASK)) & 3) != 0) | (
            core == (key % count + count - 1) % count
        )
        found, missed = here & holds, here & ~holds
        value[found] = table_mix((key[found] + 0x632BE5AB) & MASK)
        destination[found] = value[found] % count
        changed[found] = VALUE
        destination[missed] = (core[missed] + 1) % count
        here = live & (handler == VALUE)
        np.add.at(out["values"], core[here].astype(np.int64), 1)
        total[here] = (total[here] + value[here]) & MASK
        left[here] -= 1
        again, done = here & (left > 0), here & (left == 0)
        key[again] = table_mix(value[again] ^ 0x85EBCA6B)
        destination[again] = key[again] % count
        changed[again] = LOOKUP
        destination[done] = origin[done]
        changed[done] = REPORT
        here = live & (handler == REPORT)
        cores = core[here].astype(np.int64)
        np.add.at(out["reports"], cores, 1)
        np.add.at(out["sum"], cores, total[here])
        np.add.at(out["probes"], cores, probes[here])
        live[here] = False
        moving = live.copy()
        np.add.at(out["own"], core[moving & (destination == core)].astype(np.int64), 1)
        core, handler = destination, changed
    return out
