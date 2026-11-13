"""Host side of the matrix product: the reference, the plan and the tile order in
which the kernels read A and W, emulated in numpy."""

import numpy as np
import pytest

from komorebi import layout
from projects.dense import plan as pl
from ref import dense as ref

CONSTANTS = pl.K
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
    count = pl.group_size(p, 120)
    for i in range(count):
        for m in layout.partition(range(p.mt), i, count):
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
    assert p.l1_bytes() == (24 + 144 + 8) * pl.TILE_BYTES + CONSTANTS["kProfBytes"]
    big = pl.plan(512, 32, 32, budget=1_000_000)
    assert not big.resident and big.ws_pages == 2 * 32 * 4 and big.a_pages == 64
    assert pl.plan(512, 32, 32, budget=100_000) is None
    assert (
        pl.plan(4, 4, 4, form=CONSTANTS["kFormStream"]).form == CONSTANTS["kFormStream"]
    )


def test_product_args_follow_the_header():
    p = pl.plan(8, 4, 8)
    words = pl.product_args(p, 100, 200, 300, CONSTANTS["kModeReadOnly"])
    assert len(words) == 10
    assert words[:2] == [100, 200] and words[9] == 300
    assert words[2:9] == [8, 4, 8, p.kc, p.ct, CONSTANTS["kModeReadOnly"], p.form]
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
    p = pl.plan(mt, kt, nt, form=CONSTANTS[form])
    rng = np.random.default_rng(1)
    a = rng.standard_normal((mt * T, kt * T))
    w = rng.standard_normal((kt * T, nt * T))
    assert np.allclose(emulate(p, a, w), a @ w)


def emulate_grid(g, a, w, grid=(12, 10)):
    """Y as the grid kernels compute it: the blocks the readers fill (A rows stride
    kb, W rows stride bn), compute's tile indices, the writer's dropped tiles."""
    pa, pw = tiles(a), tiles(w)
    y = np.full((g.mt, g.nt, T, T), np.nan)
    for m0, n0 in g.places(grid).values():
        m_end, n_end = min(m0 + g.pcm, g.mt), min(n0 + g.pcn, g.nt)
        for blk in range(g.nbh * g.nbw):
            r0, c0 = m0 + blk // g.nbw * g.bm, n0 + blk % g.nbw * g.bn
            acc = np.zeros((g.bm, g.bn, T, T))
            for kblk in range(g.nk):
                k0 = kblk * g.kb
                ab = [np.zeros((T, T))] * (g.bm * g.kb)
                wb = [np.zeros((T, T))] * (g.kb * g.bn)
                for i in range(g.bm):
                    for j in range(min(g.kb, g.kt - k0)):
                        if r0 + i < g.mt:
                            ab[i * g.kb + j] = pa[(r0 + i) * g.kt + k0 + j]
                for i in range(min(g.kb, g.kt - k0)):
                    for j in range(g.bn):
                        if c0 + j < g.nt:
                            wb[i * g.bn + j] = pw[(k0 + i) * g.nt + c0 + j]
                inner = g.kb_last if kblk == g.nk - 1 else g.kb
                for i in range(g.bm):
                    for j in range(g.bn):
                        for k in range(inner):
                            acc[i, j] += ab[i * g.kb + k] @ wb[k * g.bn + j]
            for i in range(g.bm):
                for j in range(g.bn):
                    if r0 + i < m_end and c0 + j < n_end:
                        y[r0 + i, c0 + j] = acc[i, j]
    return y.transpose(0, 2, 1, 3).reshape(g.mt * T, g.nt * T)


@pytest.mark.parametrize("layout", pl.LAYOUTS)
def test_grid_tile_order_gives_the_product(layout):
    g = pl.grid_plan(13, 5, 14, layout=layout)
    g = pl.Grid(**(g.__dict__ | {"bm": 2, "bn": 2, "kb": 2, "sh": 1, "sw": 2}))
    rng = np.random.default_rng(1)
    a = rng.integers(-2, 3, (g.mt * T, g.kt * T)).astype(np.float64)
    w = rng.integers(-2, 3, (g.kt * T, g.nt * T)).astype(np.float64)
    assert np.array_equal(emulate_grid(g, a, w), a @ w)


def test_grid_rule_layouts_fit():
    shapes = {(128, 128, 128): "2d", (512, 32, 32): "rows", (1, 128, 128): "cols"}
    for (mt, kt, nt), lay in shapes.items():
        g = pl.grid_plan(mt, kt, nt)
        assert g.layout == lay and g.l1_bytes() <= pl.L1_BUDGET
        g.check((12, 10))
    g = pl.grid_plan(128, 128, 128)
    assert (g.rows, g.cols, g.pcm, g.pcn, g.sh, g.sw) == (10, 11, 13, 12, 1, 4)
    assert g.group((12, 10)) == (0, 119) and g.rect((12, 10)) == (0, 0, 11, 10)


def emulate_split(sp, a, w, form):
    """Y as the K-split kernels compute it: each core's partition of K in chunks (A rows
    stride kc, W rows stride ow), then the parts summed by owner or by tree."""
    pa, pw = tiles(a), tiles(w)
    y = np.full((sp.mt, sp.nt, T, T), np.nan)
    p = sp.oh * sp.ow
    for q in range(sp.groups()):
        m0, n0 = q // sp.nbw() * sp.oh, q % sp.nbw() * sp.ow
        parts = []
        for d in range(sp.s):
            ks = layout.partition(range(sp.kt), d, sp.s)
            acc = np.zeros((p, T, T))
            for k0 in range(0, len(ks), sp.kc):
                n = min(sp.kc, len(ks) - k0)
                ab = [np.zeros((T, T))] * (sp.oh * sp.kc)
                wb = [np.zeros((T, T))] * (sp.kc * sp.ow)
                for j in range(n):
                    k = ks[k0 + j]
                    for i in range(sp.oh):
                        if m0 + i < sp.mt:
                            ab[i * sp.kc + j] = pa[(m0 + i) * sp.kt + k]
                    for c in range(sp.ow):
                        if n0 + c < sp.nt:
                            wb[j * sp.ow + c] = pw[k * sp.nt + n0 + c]
                for i in range(sp.oh):
                    for c in range(sp.ow):
                        for k in range(n):
                            acc[i * sp.ow + c] += ab[i * sp.kc + k] @ wb[k * sp.ow + c]
            parts.append(acc)
        if form == "owner":
            total = parts[0] + sum(parts[1:], np.zeros_like(parts[0]))
        else:
            r = 0
            while 1 << r < sp.s:
                for j in range(0, sp.s, 2 << r):
                    if j + (1 << r) < sp.s:
                        parts[j] = parts[j] + parts[j + (1 << r)]
                r += 1
            total = parts[0]
        for i in range(sp.oh):
            for c in range(sp.ow):
                if m0 + i < sp.mt and n0 + c < sp.nt:
                    y[m0 + i, n0 + c] = total[i * sp.ow + c]
    return y.transpose(0, 2, 1, 3).reshape(sp.mt * T, sp.nt * T)


@pytest.mark.parametrize("form", list(pl.REDUCE_FORMS))
@pytest.mark.parametrize("s, oh, ow, kc", [(3, 2, 2, 2), (5, 1, 3, 1), (1, 3, 1, 4)])
def test_split_tile_order_gives_the_product(form, s, oh, ow, kc):
    sp = pl.split_plan(5, 7, 4, s, oh, ow, kc)
    rng = np.random.default_rng(1)
    a = rng.integers(-2, 3, (sp.mt * T, sp.kt * T)).astype(np.float64)
    w = rng.integers(-2, 3, (sp.kt * T, sp.nt * T)).astype(np.float64)
    assert np.array_equal(emulate_split(sp, a, w, form), a @ w)


def test_split_plans_of_the_narrow_shapes_fit():
    for (m, k, n), (s, oh, ow) in {
        (128, 896, 128): (7, 1, 4),
        (32, 4096, 4096): (7, 1, 8),
    }.items():
        sp = pl.split_plan(m // T, k // T, n // T, s, oh, ow)
        assert sp.cores() <= 120 and sp.l1_bytes() <= pl.L1_BUDGET
        assert len(sp.words()) == pl.S["kSplitWords"]
    with pytest.raises(ValueError):
        pl.split_plan(1, 4, 128, 5, 1, 8)  # a group larger than K's tiles
