"""The bucket-sort example's numpy reference (examples/09_bucket_sort/reference.py):
the plan it gives places every key where the stable sort by bucket puts it."""

import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[2] / "examples/09_bucket_sort/reference.py"
SPEC = importlib.util.spec_from_file_location("bucket_sort_reference", PATH)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def test_the_plan_places_each_core_s_keys_where_the_stable_sort_puts_them():
    keys = np.array([0xF0000000, 0x10000000, 0xF0000001, 0x00000005, 0x10000002] * 3)
    keys = keys.astype(np.uint32)
    log2_b, cores = 4, 4  # 15 keys on 4 cores: partitions of 4, 4, 4 and 3
    plan = reference.plan(reference.counts(keys, log2_b, cores))
    starts, lengths = reference.partitions(keys.size, cores)
    placed = np.zeros_like(keys)
    for core, (start, length) in enumerate(zip(starts, lengths)):
        cursor = plan[core].copy()
        for key in keys[start : start + length]:
            bucket = key >> (32 - log2_b)
            placed[cursor[bucket]] = key
            cursor[bucket] += 1
    assert list(lengths) == [4, 4, 4, 3]
    assert (placed == reference.output(keys, log2_b)).all()
    assert list(plan[:, 0xF]) == [9, 11, 13, 14]  # 3 + 6 keys before bucket 15
