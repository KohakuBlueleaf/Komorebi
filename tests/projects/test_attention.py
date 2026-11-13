"""Host side of attention: the plan (blocks, broadcasts, every core's jobs, the job
tables) and the float64 reference."""

import numpy as np
import pytest

from projects.attention import plan as attention_plan
from projects.attention.reference import sdpa

CONSTANTS = attention_plan.K
GRID = (12, 10)


@pytest.mark.parametrize("causal", [False, True])
def test_plan_takes_every_query_row_once(causal):
    dims = attention_plan.dims(2, 4, 2, 256, 256, 64, 64, causal)
    plan = attention_plan.make(dims, GRID, length=5)
    seen = []
    for core, jobs in enumerate(plan.jobs):
        for j in jobs:
            assert j.rows <= plan.r and j.k0 % (dims.skt * dims.dt) == 0
            head = j.q0 // dims.dt // dims.sqt
            seen += [(head, j.start + i * j.stride) for i in range(j.rows)]
        # the cores of a broadcast deliver the same blocks: same jobs, same key tiles
        top = (core // GRID[0]) // plan.length * plan.length * GRID[0] + core % GRID[0]
        assert [j.nk for j in jobs] == [j.nk for j in plan.jobs[top]]
    assert sorted(seen) == [
        (head_index, row)
        for head_index in range(dims.b * dims.h)
        for row in range(dims.sqt)
    ]


def test_causal_chunks_of_equal_area_take_every_row_once_and_even_the_cores():
    dims = attention_plan.dims(1, 8, 8, 1024, 1024, 64, 64, True)
    plain, split = attention_plan.make(dims, GRID, length=5), attention_plan.make(
        dims, GRID, length=5, chunks=6
    )
    seen = sorted(
        (j.q0 // dims.dt // dims.sqt, j.start + i * j.stride)
        for jobs in split.jobs
        for j in jobs
        for i in range(j.rows)
    )
    assert seen == [
        (head_index, row) for head_index in range(dims.h) for row in range(dims.sqt)
    ]
    most = [
        max(attention_plan.score_tiles(dims, j) for j in plan.jobs)
        for plan in (plain, split)
    ]
    assert most[1] < most[0]
    with pytest.raises(ValueError):
        attention_plan.make(
            dims, GRID, length=1, chunks=1
        )  # 32 rows on one core: over R


def test_blocks_fit_the_budget():
    dims = attention_plan.dims(1, 1, 1, 128, 128, 64, 64, False)
    assert attention_plan.blocks(dims) == (13, 12)
    query_rows, key_block = attention_plan.blocks(dims, budget=300_000)
    assert (query_rows, key_block) == (6, 6) and attention_plan.l1_bytes(
        dims, query_rows, key_block
    ) <= 300_000
    assert (
        attention_plan.l1_bytes(dims, 7, 6) > 300_000
    )  # KB first went down to 6, then R
    assert attention_plan.blocks(dims, budget=10_000) == (0, 0)


def test_tables_follow_the_header():
    dims = attention_plan.dims(1, 2, 2, 64, 64, 64, 64, False)
    plan = attention_plan.make(dims, GRID, length=10)
    words = attention_plan.tables(plan)
    assert words.shape == (120, CONSTANTS["kTableBytes"] // 4)
    core = next(core_index for core_index, jobs in enumerate(plan.jobs) if jobs)
    j, offset = plan.jobs[core][0], CONSTANTS["kTabFirst"]
    assert words[core, CONSTANTS["kTabJobs"]] == len(plan.jobs[core])
    assert list(words[core, offset : offset + CONSTANTS["kJobWords"]]) == [
        j.q0, j.k0, j.v0, j.o0, j.rows, j.start, j.nk, j.stride
    ]  # fmt: skip


def test_buffers_and_profile_words_do_not_collide():
    dims = attention_plan.dims(1, 2, 2, 64, 64, 128, 128, True)
    cbs = [buffer[0] for buffer in attention_plan.buffers(dims, 13, 12)]
    assert len(set(cbs)) == len(cbs) and max(cbs) < 29  # 29 to 31: the runtime's
    prof = [
        value
        for name, value in CONSTANTS.items()
        if name.startswith("kPr") and "Prof" not in name
    ]
    assert len(set(prof)) == len(prof) and max(prof) < CONSTANTS["kProfWords"]
    assert CONSTANTS["kProfWords"] * 4 <= CONSTANTS["kProfBytes"]
    assert len(attention_plan.call_args(1, 2, 3, 4, 5, passes=7)) == 9


def test_exp_constants():
    factor, thr = attention_plan.exp_constants(0.125)
    assert np.uint32(factor).view(np.float32) == np.float32(0.125 * 1.4426950408889634)
    assert np.uint32(thr).view(np.float32) == 655.0


def test_reference_is_the_softmax():
    queries, keys, values = sdpa.inputs(1, 1, 1, 64, 64, 32, 32)
    rows = np.arange(64)
    scores = queries[0, 0].astype(np.float64) @ keys[0, 0].T / np.sqrt(32)
    probabilities = np.exp(scores) / np.exp(scores).sum(1, keepdims=True)
    assert np.allclose(
        sdpa.attention(queries[0, 0], keys[0, 0], values[0, 0], rows, False),
        probabilities @ values[0, 0],
    )
    causal = sdpa.attention(queries[0, 0], keys[0, 0], values[0, 0], rows, True)
    assert np.array_equal(causal[0], values[0, 0, 0].astype(np.float64))
