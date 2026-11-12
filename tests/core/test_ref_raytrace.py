"""The numpy ray tracer: triangle and box tests, the walk, shading, checksums."""

import hashlib

import numpy as np
import pytest

from ref import bvh, raytrace, scenes

F = np.float32
SUBSET = (
    np.arange(64)[:, None] * 16 * 1024 + np.arange(64)[None, :] * 16 + 8 * 1025
).ravel()
PINNED = {
    "image": "5b797f568ed5092fed036f9ffa58f941337e16a61dae1c626e697741cfb6f61c",
    "t": "5c50f6d7676117f9a460d5581eb3a8733c3b3114fdd417df4e8ff2eff0601998",
    "tri": "fd672e27918135466263397952886aae6f2ee7703399fa58324b9a60ec773a75",
    "nodes": "0d616eaa0e29b17318b37fc74c29461cfc79bd4aea60c449f68548e180be65fb",
}


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def vec(*rows):
    return np.array(rows, dtype=np.float32)


def tri_record(a, b, c):
    a, b, c = (np.array(p, dtype=np.float32) for p in (a, b, c))
    return np.concatenate([a, b - a, c - a])[None]


@pytest.fixture(scope="module")
def tiny():
    scene = scenes.make_preset("tiny")
    return scene, bvh.build(scene)


def test_triangle_test_hit_miss_and_both_sides():
    tri = tri_record((0, 0, 5), (4, 0, 5), (0, 4, 5))
    o = vec((1, 1, 0), (1, 1, 10), (3, 3, 0), (1, 1, 0))
    d = vec((0, 0, 1), (0, 0, -1), (0, 0, 1), (0, 1, 0))
    ok, t, u, v = raytrace.triangle_test(tri, np.zeros(4, dtype=int), o, d)
    assert ok.tolist() == [True, True, False, False]
    assert t[:2].tolist() == [5.0, 5.0] and u[0] == 0.25 and v[0] == 0.25


def test_triangle_test_parallel_ray_and_edges():
    tri = tri_record((0, 0, 5), (4, 0, 5), (0, 4, 5))
    o = vec((1, 1, 0), (0, 0, 0), (4, 0, 0), (2, 2, 0), (2.5, 2, 0))
    d = vec((1, 0, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1))
    ok, _, _, _ = raytrace.triangle_test(tri, np.zeros(5, dtype=int), o, d)
    assert ok.tolist() == [False, True, True, True, False]


def test_box_test_inside_outside_and_axis_parallel():
    box = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.float32)
    ids = np.zeros(4, dtype=int)
    o = vec((0.5, 0.5, -2), (5, 5, -2), (0.5, 0.5, 0.5), (0.5, 0.5, 3))
    d = vec((0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1))
    hit, entry = raytrace.box_test(box, ids, o, raytrace.inverse_direction(d), F(1e30))
    assert hit.tolist() == [True, False, True, False]
    assert entry[0] == 2.0 and entry[2] == 0.0
    hit, _ = raytrace.box_test(
        box, ids[:1], o[:1], raytrace.inverse_direction(d[:1]), F(1.5)
    )
    assert not hit[0]


def test_zero_direction_components_do_not_poison_the_slab_test():
    inv = raytrace.inverse_direction(vec((0, 0, 1), (-0.0, 1e-20, 2)))
    assert np.isfinite(inv).all() and inv[0, 0] == F(1e12) and inv[1, 0] == F(-1e12)


def test_walk_equals_brute_force_over_all_triangles(tiny):
    scene, tree = tiny
    o, d = raytrace.camera_rays(scene["camera"], SUBSET)
    hit = raytrace.trace(tree, o, d, raytrace.TMAX)
    n = len(SUBSET)
    best, leaf = np.full(n, raytrace.TMAX, dtype=np.float32), np.full(n, -1)
    for k in range(len(tree["tris"])):
        ok, t, _, _ = raytrace.triangle_test(tree["tris"], np.full(n, k), o, d)
        tie = ok & (t == best) & (leaf >= 0)
        tie &= tree["leaf_tri"][k] < tree["leaf_tri"][np.maximum(leaf, 0)]
        take = (ok & (t < best)) | tie
        best, leaf = np.where(take, t, best), np.where(take, k, leaf)
    assert (leaf == hit["leaf"]).all() and (best == hit["t"]).all()
    assert 0.4 < (leaf >= 0).mean() < 0.9 and hit["nodes"].min() >= 1


def test_nearer_of_two_triangles_wins_and_ties_go_to_the_lower_index():
    scene = {
        "positions": vec(
            *[(0, 0, 5), (4, 0, 5), (0, 4, 5)] * 2 + [(0, 0, 3), (4, 0, 3), (0, 4, 3)]
        ),
        "indices": np.arange(9, dtype=np.int32).reshape(3, 3),
    }
    tree = bvh.build(scene)
    o, d = vec((1, 1, 0)), vec((0, 0, 1))
    hit = raytrace.trace(tree, o, d, raytrace.TMAX)
    assert tree["leaf_tri"][hit["leaf"][0]] == 2 and hit["t"][0] == 3.0
    scene["indices"] = scene["indices"][:2]
    tree = bvh.build(scene)
    hit = raytrace.trace(tree, o, d, raytrace.TMAX)
    assert tree["leaf_tri"][hit["leaf"][0]] == 0 and hit["t"][0] == 5.0


def test_any_hit_stops_at_a_blocker_between_origin_and_tmax():
    scene = {
        "positions": vec((0, 0, 5), (4, 0, 5), (0, 4, 5)),
        "indices": np.arange(3, dtype=np.int32).reshape(1, 3),
    }
    tree = bvh.build(scene)
    o, d = vec((1, 1, 0), (1, 1, 0), (1, 1, 6)), vec((0, 0, 10), (0, 0, 10), (0, 0, 10))
    for tmax, expect in (
        (F(1.0), [True, True, False]),
        (F(0.4), [False, False, False]),
    ):
        out = raytrace.trace(tree, o, d, tmax, any_hit=True)
        assert (out["leaf"] >= 0).tolist() == expect


def test_per_pixel_numbers_do_not_depend_on_which_pixels_are_rendered(tiny):
    scene, tree = tiny
    few = raytrace.render(scene, tree, SUBSET[:50])
    more = raytrace.render(scene, tree, SUBSET[:200])
    for key in ("image", "t", "tri", "nodes"):
        assert np.array_equal(few[key], more[key][:50])


def test_misses_show_the_sky_and_hits_are_lit(tiny):
    scene, tree = tiny
    out = raytrace.render(scene, tree, np.array([100 * 1024 + 500, 900 * 1024 + 500]))
    assert out["tri"][0] == -1 and out["t"][0] == raytrace.TMAX and out["nodes"][0] >= 1
    _, d = raytrace.camera_rays(scene["camera"], np.array([100 * 1024 + 500]))
    assert np.array_equal(out["image"][0], raytrace.sky(d)[0])
    assert (
        out["tri"][1] >= 0
        and (out["image"][1] > 0).all()
        and (out["image"][1] < 1).all()
    )


def test_pinned_checksums_of_a_subset_of_the_tiny_scene(tiny):
    scene, tree = tiny
    out = raytrace.render(scene, tree, SUBSET)
    assert {k: sha(out[k]) for k in PINNED} == PINNED
