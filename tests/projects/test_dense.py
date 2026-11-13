"""Host side of the matrix product: the reference, the plan and the tile order in
which the kernels read A and W, emulated in numpy."""

import numpy as np
import pytest

from komorebi import layout
from projects.dense import plan as pl
from ref import dense as ref

K = pl.K
T = ref.TILE


def tiles(x):
    """x [r*32, c*32] as a list of 32 x 32 tiles in row-major tile order (the
    order of a tile-layout tensor's pages)."""
    r, c = x.shape[0] // T, x.shape[1] // T
    return list(x.reshape(r, T, c, T).transpose(0, 2, 1, 3).reshape(r * c, T, T))


def emulate(p, a, w):
    """Y as the kernels compute it: the reader's pages, the compute loop's tile
    indices into the A and W buffers, the writer's output pages."""
    pa, pw = tiles(a), tiles(w)
    y = [None] * (p.mt * p.nt)
    count = pl.team_size(p, 120)
    for i in range(count):
        for m in layout.share(range(p.mt), i, count):
            row = pa[m * p.kt : (m + 1) * p.kt] if p.row_item else None
            for j0 in range(0, p.nt, p.ct):
                dst = [np.zeros((T, T)) for _ in range(p.ct)]
                for k0 in range(0, p.kt, p.kc):
                    a_item = row if p.row_item else pa[m * p.kt + k0 :][: p.kc]
                    a0 = k0 if p.row_item else 0
                    if p.resident:
                        wbuf = pw
                    else:
                        first = k0 * p.nt + j0
                        wbuf = [
                            pw[first + i * p.nt + j]
                            for i in range(p.kc)
                            for j in range(p.ct)
                        ]
                    for k in range(p.kc):
                        base = k * p.ct if not p.resident else (k0 + k) * p.nt + j0
                        for j in range(p.ct):
                            dst[j] += a_item[a0 + k] @ wbuf[base + j]
                for j in range(p.ct):
                    y[m * p.nt + j0 + j] = dst[j]
    full = np.array(y).reshape(p.mt, p.nt, T, T).transpose(0, 2, 1, 3)
    return full.reshape(p.mt * T, p.nt * T)


def test_bf16_rounds_to_nearest_even():
    x = np.array([1.0, 1 + 2**-8, 1 + 3 * 2**-8, -2.5, 1e-40], np.float32)
    got = ref.bf16(x)
    assert got.tolist()[:4] == [1.0, 1.0, 1 + 2**-6, -2.5]
    assert (got.view(np.uint32) & 0xFFFF == 0).all()


def test_errors_are_relative_to_the_reference():
    y = np.ones((64, 96))
    assert ref.errors(y, y) == {"max_rel": 0.0, "rms_rel": 0.0}
    z = y.copy()
    z[40, 70] = 2.0
    assert ref.errors(z, y)["max_rel"] == 1.0


def test_chunk_and_block():
    assert [pl.chunk(k) for k in (1, 12, 28, 32, 64, 96, 37)] == [
        1,
        12,
        28,
        32,
        32,
        32,
        1,
    ]
    assert [pl.block(n) for n in (1, 2, 3, 4, 6, 12, 7)] == [1, 2, 3, 4, 3, 4, 1]


def test_plan_resident_until_w_overflows():
    p = pl.plan(512, 12, 12)
    assert p.resident and (p.kc, p.ct) == (12, 4) and p.row_item
    assert p.l1_bytes() == (24 + 144 + 8) * pl.TILE_BYTES + K["kProfBytes"]
    big = pl.plan(512, 32, 32, budget=1_000_000)
    assert not big.resident and big.ws_pages == 2 * 32 * 4 and big.a_pages == 64
    assert pl.plan(512, 32, 32, budget=100_000) is None
    assert pl.plan(4, 4, 4, form=K["kFormStream"]).form == K["kFormStream"]


def test_product_args_follow_the_header():
    p = pl.plan(8, 4, 8)
    words = pl.product_args(p, 100, 200, 300, K["kModeReadOnly"])
    assert len(words) == 10
    assert words[:2] == [100, 200] and words[9] == 300
    assert words[2:9] == [8, 4, 8, p.kc, p.ct, K["kModeReadOnly"], p.form]
    assert pl.load_args(p, 300)[0] == 300 and pl.load_args(p, 300)[3:] == [4, 8]
    assert pl.flops(p) == 2 * 8 * 4 * 8 * 32**3


@pytest.mark.parametrize(
    "shape,form",
    [
        ((3, 4, 8), "kFormResident"),
        ((3, 40, 6), "kFormResident"),
        ((3, 4, 8), "kFormStream"),
        ((2, 40, 6), "kFormStream"),
    ],
)
def test_kernel_tile_order_gives_the_product(shape, form):
    mt, kt, nt = shape
    p = pl.plan(mt, kt, nt, form=K[form])
    rng = np.random.default_rng(1)
    a = rng.standard_normal((mt * T, kt * T))
    w = rng.standard_normal((kt * T, nt * T))
    assert np.allclose(emulate(p, a, w), a @ w)
