"""The arithmetic check's inputs on the CPU: edge cases, ties and the expected values
(the reference's own expressions)."""

import numpy as np
import pytest

from projects.raytrace import arithmetic, packets
from ref import bvh, raytrace, scenes

FLOAT32 = np.float32


@pytest.fixture(scope="module")
def rows():
    scene = scenes.make_preset("tiny")
    tree = bvh.build(scene)
    rng = np.random.default_rng(2)
    n = 2048
    o = rng.uniform(-5, 5, (n, 3)).astype(FLOAT32)
    d = rng.normal(size=(n, 3)).astype(FLOAT32)
    leaf = rng.integers(0, len(tree["tris"]), n)
    g = tree["tris"][leaf]
    lo, hi = bvh.node_boxes(tree["nodes"][tree["leaf_node"][leaf]])
    d = (g[:, 0:3] + 0.3 * g[:, 3:6] + 0.3 * g[:, 6:9] - o).astype(FLOAT32)
    out = {
        "o": o, "d": d, "bmin": lo.copy(), "bmax": hi.copy(),
        "v0": g[:, 0:3].copy(), "e1": g[:, 3:6].copy(), "e2": g[:, 6:9].copy(),
        "tri": leaf.astype(np.int64), "best": np.full(n, raytrace.TMAX, FLOAT32),
    }  # fmt: skip
    edge = arithmetic.edge_inputs(rng, out, tree)
    arithmetic.bests(rng, out)
    return out, edge


def test_edge_cases_cover_their_kinds(rows):
    r, edge = rows
    assert 0.15 < edge.mean() < 0.35
    assert (r["d"] == 0).any() and np.signbit(r["d"][r["d"] == 0]).any()
    in_plane = (r["bmax"] == r["bmin"]) & (r["o"] == r["bmin"])
    assert in_plane.any(axis=1)[edge].sum() > 10
    ok, _, _, _ = arithmetic.triangle(r)
    assert ok.any() and (~ok).any()


def test_bests_make_ties_and_neighbours(rows):
    r, _ = rows
    ok, t, _, _ = arithmetic.triangle(r)
    tie = ok & (r["best"] == t)
    assert tie.sum() > 50
    assert (ok & (r["best"] == np.nextafter(t, FLOAT32(np.inf)))).any()
    assert set(np.unique(r["cur"] - r["tri"])) >= {-1, 0, 1}


def test_expected_matches_the_reference_trace_update(rows):
    r, _ = rows
    want = arithmetic.expected(r)
    ok, t, _, _ = arithmetic.triangle(r)
    closer = ok & (t < r["best"])
    assert np.array_equal(want["new_best"][closer], t[closer])
    keep = ~ok
    assert np.array_equal(want["new_best"][keep], r["best"][keep])


def test_pages_hold_the_rows(rows):
    r, _ = rows
    pages = arithmetic.input_pages(r)
    k = packets.K
    assert pages.shape == (len(r["o"]) // 32, k["kArInVectors"] * 32)
    v = pages.reshape(-1, k["kArInVectors"], 32)
    assert np.array_equal(v[:, k["kArO"]].ravel().view(FLOAT32), r["o"][:, 0])


def test_math_inputs_stay_in_range_and_fill_pages():
    a, b, x = arithmetic.math_inputs(np.random.default_rng(3), 4096)
    q = a / b
    assert (b > 0).all() and (x > 0).all() and (a == 0).any() and (a < 0).any()
    normal = np.abs(q[a != 0]) >= np.finfo(FLOAT32).tiny
    assert normal.all() and np.isfinite(q).all()
    v = arithmetic.math_pages(a, b, x).reshape(-1, packets.K["kMaInVectors"], 32)
    assert np.array_equal(v[:, packets.K["kMaB"]].ravel().view(FLOAT32), b)
