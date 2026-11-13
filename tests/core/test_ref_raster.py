"""The numpy rasterizer: fill rule, depth, colour, culling, binning, checksums."""

import hashlib

import numpy as np
import pytest

from ref import raster, scenes

CAMERA = np.array(
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 512, 512, 512, 0.05], dtype=np.float32
)
TINY_SHA = {
    "tri": "cb5967b33fd29b20ad41208da3772e66beb94e0386111a2ae4d3c12c60125970",
    "colour": "29e0b437db693c1e7b19786bcadd26265199a28ac2f6a332d12562af6f732e62",
    "tile_counts": "c7689608337f498d54a9ec17a1cc77f778bc2b6dd8ce89585753167c11db8c02",
}


def screen_scene(triangles, colors=None):
    """A scene whose vertices project to the given (x, y, z) pixel positions."""
    verts = np.array(triangles, dtype=np.float64).reshape(-1, 3)
    pos = np.stack(
        [
            (verts[:, 0] - 512) * verts[:, 2] / 512,
            (512 - verts[:, 1]) * verts[:, 2] / 512,
            verts[:, 2],
        ],
        axis=1,
    )
    n = len(verts) // 3
    if colors is None:
        colors = np.full((len(verts), 3), 0.5)
    return {
        "positions": pos.astype(np.float32),
        "colors": np.asarray(colors, dtype=np.float32),
        "indices": np.arange(3 * n, dtype=np.int32).reshape(n, 3),
        "camera": CAMERA,
    }


def covered(triangles, **kwargs):
    """Boolean image of the pixels one rendering of the triangles covers."""
    return raster.rasterize(screen_scene(triangles, **kwargs))["tri"] >= 0


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_shared_diagonal_belongs_to_one_triangle():
    upper = [(10, 10, 1), (18, 10, 1), (18, 18, 1)]
    lower = [(10, 10, 1), (18, 18, 1), (10, 18, 1)]
    out = raster.rasterize(screen_scene(upper + lower))
    tri = out["tri"][10:18, 10:18]
    assert (tri >= 0).all() and (out["tri"] >= 0).sum() == 64
    assert (np.diag(tri) == 0).all()
    assert (tri == 0).sum() == 36 and (tri == 1).sum() == 28


def test_top_edge_is_filled_and_bottom_edge_is_not():
    below = [(10, 12.5, 1), (18, 12.5, 1), (14, 20.5, 1)]
    above = [(10, 12.5, 1), (14, 4.5, 1), (18, 12.5, 1)]
    assert covered(below)[12, 10:18].all()
    assert not covered(above)[12].any()
    assert covered(above)[11, 11:17].all()


def test_left_edge_is_filled_and_right_edge_is_not():
    right_of = [(12.5, 10, 1), (20.5, 14, 1), (12.5, 18, 1)]
    left_of = [(12.5, 10, 1), (4.5, 14, 1), (12.5, 18, 1)]
    assert covered(right_of)[11:17, 12].all()
    assert not covered(left_of)[:, 12].any()


def test_nearer_triangle_wins_in_either_order():
    near = [(20, 20, 1), (60, 20, 1), (20, 60, 1)]
    far = [(10, 10, 4), (100, 10, 4), (10, 100, 4)]
    for order, nearer in (((far, near), 1), ((near, far), 0)):
        out = raster.rasterize(screen_scene(order[0] + order[1]))
        assert out["tri"][30, 30] == nearer
        assert out["depth"][30, 30] == pytest.approx(1.0)
    assert out["tri"][50, 50] == 1 and out["depth"][50, 50] == pytest.approx(0.25)


def test_equal_depth_goes_to_the_lower_index():
    tri = [(20, 20, 2), (60, 20, 2), (20, 60, 2)]
    out = raster.rasterize(screen_scene(tri + tri + tri))
    assert (out["tri"][(out["tri"] >= 0)] == 0).all() and (out["tri"] >= 0).sum() > 500


def test_colour_is_interpolated_linearly_in_screen_space():
    tri = [(16, 16, 1), (80, 16, 1), (16, 80, 1)]
    out = raster.rasterize(screen_scene(tri, colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    for x, y in [(20, 20), (40, 20), (20, 50), (30, 30)]:
        lam1, lam2 = (x + 0.5 - 16) / 64, (y + 0.5 - 16) / 64
        expect = [1 - lam1 - lam2, lam1, lam2]
        assert out["colour"][y, x] == pytest.approx(expect, abs=1e-5)
    assert (out["colour"][600, 600] == raster.BACKGROUND).all()


def test_triangles_that_cover_nothing_are_dropped():
    cases = {
        "behind the camera": [(300, 300, -1), (400, 300, -1), (300, 400, -1)],
        "one vertex behind": [(300, 300, 1), (400, 300, 1), (300, 400, 0.01)],
        "off screen": [(-300, 300, 1), (-100, 300, 1), (-300, 400, 1)],
        "zero area": [(300, 300, 1), (400, 400, 1), (500, 500, 1)],
        "between pixel centres": [
            (300.6, 300.6, 1),
            (300.9, 300.6, 1),
            (300.6, 300.9, 1),
        ],
    }
    for name, tri in cases.items():
        out = raster.rasterize(screen_scene(tri))
        assert (out["tri"] < 0).all() and out["tile_counts"].sum() == 0, name


def test_winding_does_not_matter():
    tri = [(20, 20, 1), (90, 30, 1), (40, 95, 1)]
    a = raster.rasterize(screen_scene(tri))
    b = raster.rasterize(screen_scene([tri[0], tri[2], tri[1]]))
    assert np.array_equal(a["tri"], b["tri"]) and (a["tri"] >= 0).sum() > 1000


def test_tile_counts_follow_the_pixel_box():
    tri = [(33.2, 33.2, 1), (100.7, 40.2, 1), (40.2, 95.7, 1)]
    out = raster.rasterize(screen_scene(tri))
    expect = np.zeros((32, 32), dtype=np.int32)
    expect[1:3, 1:4] = 1
    assert np.array_equal(out["tile_counts"], expect)
    full = raster.rasterize(
        screen_scene([(-2000, -2000, 1), (6000, -2000, 1), (-2000, 6000, 1)])
    )
    assert (full["tile_counts"] == 1).all() and (full["tri"] == 0).all()


def test_tiny_scene_checksums():
    out = raster.rasterize(scenes.make_preset("tiny"))
    covered_share = float((out["tri"] >= 0).mean())
    assert 0.3 < covered_share < 0.9
    assert out["tile_counts"].sum() > 300
    assert {k: sha(out[k]) for k in TINY_SHA} == TINY_SHA
