"""The numpy ray tracer: triangle and box tests, the walk, shading, checksums."""

import hashlib

import numpy as np
import pytest

from reference import bvh, raytrace, scenes

FLOAT32 = np.float32
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


def tri_record(corner_a, corner_b, corner_c):
    corner_a, corner_b, corner_c = (
        np.array(corner, dtype=np.float32) for corner in (corner_a, corner_b, corner_c)
    )
    return np.concatenate([corner_a, corner_b - corner_a, corner_c - corner_a])[None]


@pytest.fixture(scope="module")
def tiny():
    scene = scenes.make_preset("tiny")
    return scene, bvh.build(scene)


def test_triangle_test_hit_miss_and_both_sides():
    tri = tri_record((0, 0, 5), (4, 0, 5), (0, 4, 5))
    origin = vec((1, 1, 0), (1, 1, 10), (3, 3, 0), (1, 1, 0))
    direction = vec((0, 0, 1), (0, 0, -1), (0, 0, 1), (0, 1, 0))
    hits, distance, u, v = raytrace.triangle_test(
        tri, np.zeros(4, dtype=int), origin, direction
    )
    assert hits.tolist() == [True, True, False, False]
    assert distance[:2].tolist() == [5.0, 5.0] and u[0] == 0.25 and v[0] == 0.25


def test_triangle_test_parallel_ray_and_edges():
    tri = tri_record((0, 0, 5), (4, 0, 5), (0, 4, 5))
    origin = vec((1, 1, 0), (0, 0, 0), (4, 0, 0), (2, 2, 0), (2.5, 2, 0))
    direction = vec((1, 0, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1))
    hits, _, _, _ = raytrace.triangle_test(
        tri, np.zeros(5, dtype=int), origin, direction
    )
    assert hits.tolist() == [False, True, True, True, False]


def test_box_test_inside_outside_and_axis_parallel():
    box = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.float32)
    ids = np.zeros(4, dtype=int)
    origin = vec((0.5, 0.5, -2), (5, 5, -2), (0.5, 0.5, 0.5), (0.5, 0.5, 3))
    direction = vec((0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1))
    hit, entry = raytrace.box_test(
        box, ids, origin, raytrace.inverse_direction(direction), FLOAT32(1e30)
    )
    assert hit.tolist() == [True, False, True, False]
    assert entry[0] == 2.0 and entry[2] == 0.0
    hit, _ = raytrace.box_test(
        box,
        ids[:1],
        origin[:1],
        raytrace.inverse_direction(direction[:1]),
        FLOAT32(1.5),
    )
    assert not hit[0]


def test_zero_direction_components_do_not_poison_the_slab_test():
    inv = raytrace.inverse_direction(vec((0, 0, 1), (-0.0, 1e-20, 2)))
    assert (
        np.isfinite(inv).all()
        and inv[0, 0] == FLOAT32(1e12)
        and inv[1, 0] == FLOAT32(-1e12)
    )


def test_walk_equals_brute_force_over_all_triangles(tiny):
    scene, tree = tiny
    origin, direction = raytrace.camera_rays(scene["camera"], SUBSET)
    hit = raytrace.trace(tree, origin, direction, raytrace.TMAX)
    ray_count = len(SUBSET)
    best, leaf = np.full(ray_count, raytrace.TMAX, dtype=np.float32), np.full(
        ray_count, -1
    )
    for triangle in range(len(tree["tris"])):
        hits, distance, _, _ = raytrace.triangle_test(
            tree["tris"], np.full(ray_count, triangle), origin, direction
        )
        tie = hits & (distance == best) & (leaf >= 0)
        tie &= tree["leaf_tri"][triangle] < tree["leaf_tri"][np.maximum(leaf, 0)]
        take = (hits & (distance < best)) | tie
        best, leaf = np.where(take, distance, best), np.where(take, triangle, leaf)
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
    origin, direction = vec((1, 1, 0)), vec((0, 0, 1))
    hit = raytrace.trace(tree, origin, direction, raytrace.TMAX)
    assert tree["leaf_tri"][hit["leaf"][0]] == 2 and hit["t"][0] == 3.0
    scene["indices"] = scene["indices"][:2]
    tree = bvh.build(scene)
    hit = raytrace.trace(tree, origin, direction, raytrace.TMAX)
    assert tree["leaf_tri"][hit["leaf"][0]] == 0 and hit["t"][0] == 5.0


def test_any_hit_stops_at_a_blocker_between_origin_and_tmax():
    scene = {
        "positions": vec((0, 0, 5), (4, 0, 5), (0, 4, 5)),
        "indices": np.arange(3, dtype=np.int32).reshape(1, 3),
    }
    tree = bvh.build(scene)
    origin, direction = vec((1, 1, 0), (1, 1, 0), (1, 1, 6)), vec(
        (0, 0, 10), (0, 0, 10), (0, 0, 10)
    )
    for tmax, expect in (
        (FLOAT32(1.0), [True, True, False]),
        (FLOAT32(0.4), [False, False, False]),
    ):
        out = raytrace.trace(tree, origin, direction, tmax, any_hit=True)
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
    _, direction = raytrace.camera_rays(scene["camera"], np.array([100 * 1024 + 500]))
    assert np.array_equal(out["image"][0], raytrace.sky(direction)[0])
    assert (
        out["tri"][1] >= 0
        and (out["image"][1] > 0).all()
        and (out["image"][1] < 1).all()
    )


def test_pinned_checksums_of_a_subset_of_the_tiny_scene(tiny):
    scene, tree = tiny
    out = raytrace.render(scene, tree, SUBSET)
    assert {name: sha(out[name]) for name in PINNED} == PINNED
