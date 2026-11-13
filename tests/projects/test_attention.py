"""Host side of attention: the plan (blocks, lanes, every core's jobs, the job
tables) and the float64 reference."""

import numpy as np
import pytest

from projects.attention import plan as pl
from projects.attention.reference import sdpa

K = pl.K
GRID = (12, 10)


@pytest.mark.parametrize("causal", [False, True])
def test_plan_takes_every_query_row_once(causal):
    s = pl.dims(2, 4, 2, 256, 256, 64, 64, causal)
    p = pl.make(s, GRID, length=5)
    seen = []
    for core, jobs in enumerate(p.jobs):
        for j in jobs:
            assert j.rows <= p.r and j.k0 % (s.skt * s.dt) == 0
            head = j.q0 // s.dt // s.sqt
            seen += [(head, j.start + i * j.stride) for i in range(j.rows)]
        # the cores of a lane hand over the same blocks: same jobs, same key tiles
        top = (core // GRID[0]) // p.length * p.length * GRID[0] + core % GRID[0]
        assert [j.nk for j in jobs] == [j.nk for j in p.jobs[top]]
    assert sorted(seen) == [(hd, r) for hd in range(s.b * s.h) for r in range(s.sqt)]


def test_blocks_fit_the_budget():
    s = pl.dims(1, 1, 1, 128, 128, 64, 64, False)
    assert pl.blocks(s) == (13, 12)
    r, kb = pl.blocks(s, budget=300_000)
    assert (r, kb) == (6, 6) and pl.l1_bytes(s, r, kb) <= 300_000
    assert pl.l1_bytes(s, 7, 6) > 300_000  # KB first went down to 6, then R
    assert pl.blocks(s, budget=10_000) == (0, 0)


def test_tables_follow_the_header():
    s = pl.dims(1, 2, 2, 64, 64, 64, 64, False)
    p = pl.make(s, GRID, length=10)
    words = pl.tables(p)
    assert words.shape == (120, K["kTableBytes"] // 4)
    core = next(c for c, jobs in enumerate(p.jobs) if jobs)
    j, at = p.jobs[core][0], K["kTabFirst"]
    assert words[core, K["kTabJobs"]] == len(p.jobs[core])
    assert list(words[core, at : at + K["kJobWords"]]) == [
        j.q0, j.k0, j.v0, j.o0, j.rows, j.start, j.nk, j.stride
    ]  # fmt: skip


def test_buffers_and_profile_words_do_not_collide():
    s = pl.dims(1, 2, 2, 64, 64, 128, 128, True)
    cbs = [b[0] for b in pl.buffers(s, 13, 12)]
    assert len(set(cbs)) == len(cbs) and max(cbs) < 29  # 29 to 31: the runtime's
    prof = [v for name, v in K.items() if name.startswith("kPr") and "Prof" not in name]
    assert len(set(prof)) == len(prof) and max(prof) < K["kProfWords"]
    assert K["kProfWords"] * 4 <= K["kProfBytes"]
    assert len(pl.call_args(1, 2, 3, 4, 5, passes=7)) == 9


def test_exp_constants():
    a, thr = pl.exp_constants(0.125)
    assert np.uint32(a).view(np.float32) == np.float32(0.125 * 1.4426950408889634)
    assert np.uint32(thr).view(np.float32) == 655.0


def test_reference_is_the_softmax():
    q, k, v = sdpa.inputs(1, 1, 1, 64, 64, 32, 32)
    rows = np.arange(64)
    s = q[0, 0].astype(np.float64) @ k[0, 0].T / np.sqrt(32)
    p = np.exp(s) / np.exp(s).sum(1, keepdims=True)
    assert np.allclose(
        sdpa.attention(q[0, 0], k[0, 0], v[0, 0], rows, False), p @ v[0, 0]
    )
    causal = sdpa.attention(q[0, 0], k[0, 0], v[0, 0], rows, True)
    assert np.array_equal(causal[0], v[0, 0, 0].astype(np.float64))
