"""The bucket sort's first pass in numpy, written apart from the card's code: the
partition of each core, the counts table, the plan and the output that the kernels and
the thread must produce. The host script checks the card against it. See README.md."""

import numpy as np


def partitions(key_count, cores):
    """(start, length) of each core's run of keys: km::partition's split, the first
    `key_count % cores` cores one key more."""
    lengths = np.full(cores, key_count // cores, np.int64)
    lengths[: key_count % cores] += 1
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return starts, lengths


def buckets_of(keys, log2_buckets):
    """A key's bucket: its top log2(buckets) bits."""
    return keys >> np.uint32(32 - log2_buckets)


def counts(keys, log2_buckets, cores):
    """Step 1: [cores, buckets], how many of each core's keys fall in each bucket."""
    _, lengths = partitions(keys.size, cores)
    bucket = buckets_of(keys, log2_buckets).astype(np.int64)
    core = np.repeat(np.arange(cores), lengths)
    width = 1 << log2_buckets
    return np.bincount(core * width + bucket, minlength=cores * width).reshape(
        cores, width
    )


def plan(table):
    """Step 2: [cores, buckets], where each core's first key of each bucket goes: the
    keys of all earlier buckets, then those of the bucket on the cores before it."""
    totals = table.sum(axis=0)
    starts = np.concatenate(([0], np.cumsum(totals)[:-1]))
    before = np.cumsum(table, axis=0) - table
    return starts[None, :] + before


def output(keys, log2_buckets):
    """Step 3: the keys grouped by bucket, buckets in order, keys of one bucket in
    input order (a stable sort by bucket)."""
    return keys[np.argsort(buckets_of(keys, log2_buckets), kind="stable")]
