"""The bucket sort's first pass in numpy, written apart from the card's code: the share
of each core, the counts table, the plan and the output that the kernels and the thread
must produce. The host script checks the card against it. See README.md."""

import numpy as np


def shares(n, cores):
    """(start, length) of each core's run of keys: km::share's split, the first
    n % cores cores one key more."""
    lengths = np.full(cores, n // cores, np.int64)
    lengths[: n % cores] += 1
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return starts, lengths


def buckets_of(keys, log2_b):
    """A key's bucket: its top log2(B) bits."""
    return keys >> np.uint32(32 - log2_b)


def counts(keys, log2_b, cores):
    """Step 1: [cores, B], how many of core c's keys fall in bucket b."""
    _, lengths = shares(keys.size, cores)
    bucket = buckets_of(keys, log2_b).astype(np.int64)
    core = np.repeat(np.arange(cores), lengths)
    width = 1 << log2_b
    return np.bincount(core * width + bucket, minlength=cores * width).reshape(
        cores, width
    )


def plan(table):
    """Step 2: [cores, B], where core c's first key of bucket b goes: the keys of all
    earlier buckets, then those of bucket b on cores before c."""
    totals = table.sum(axis=0)
    starts = np.concatenate(([0], np.cumsum(totals)[:-1]))
    before = np.cumsum(table, axis=0) - table
    return starts[None, :] + before


def output(keys, log2_b):
    """Step 3: the keys grouped by bucket, buckets in order, keys of one bucket in
    input order (a stable sort by bucket)."""
    return keys[np.argsort(buckets_of(keys, log2_b), kind="stable")]
