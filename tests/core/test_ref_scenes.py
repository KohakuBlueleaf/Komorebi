"""The scene generator: determinism, sizes, files, and the hash."""

import numpy as np
import pytest

from ref import rng, scenes

TINY_SHA = "be345bfaf5f8346fc171db374ed27d975750e261800f3f93e3d8a1b65f0b1663"
S10K_SHA = "0e1cee8beafb645161800b75603c76801b6fe6b6ec2336024e0fb19fcf1670fa"


def test_hash_ranges():
    hashes = rng.hash32(np.arange(1000), stream=3, seed=7)
    assert hashes.dtype == np.uint32 and len(set(hashes.tolist())) == 1000
    u = rng.uniform(np.arange(100000))
    assert u.dtype == np.float32 and u.min() >= 0 and u.max() < 1
    assert abs(float(u.mean()) - 0.5) < 0.01


def test_hash_matches_the_written_formula():
    def hash_py(index, stream, seed):
        mask = 0xFFFFFFFF
        hashed = (index * 0x9E3779B1 + (stream + 1) * 0x85EBCA6B + seed) & mask
        hashed ^= hashed >> 16
        hashed = (hashed * 0x7FEB352D) & mask
        hashed ^= hashed >> 15
        hashed = (hashed * 0x846CA68B) & mask
        return hashed ^ (hashed >> 16)

    for index, stream, seed in [(0, 0, 0), (1, 2, 3), (99999, 11, 5), (2**31, 0, 1)]:
        assert int(rng.hash32(index, stream, seed)) == hash_py(index, stream, seed)


def test_scene_is_deterministic():
    first, second = scenes.make_preset("tiny"), scenes.make_preset("tiny")
    assert scenes.checksum(first) == scenes.checksum(second)


def test_pinned_checksums():
    assert scenes.checksum(scenes.make_preset("tiny")) == TINY_SHA
    assert scenes.checksum(scenes.make_preset("s10k")) == S10K_SHA


@pytest.mark.parametrize("name", ["tiny", "s10k", "s100k"])
def test_scene_shape(name):
    scene = scenes.make_preset(name)
    count, verts = scene["meta"][:2]
    target = scenes.PRESETS[name]
    assert 0.9 * target < count < 1.4 * target
    assert scene["indices"].shape == (count, 3) and len(scene["positions"]) == verts
    assert scene["indices"].min() == 0 and scene["indices"].max() == verts - 1
    assert scene["positions"].dtype == np.float32 and scene["indices"].dtype == np.int32
    assert np.isfinite(scene["positions"]).all() and scene["colors"].min() >= 0
    assert scene["colors"].max() <= 1


def test_no_degenerate_triangles():
    scene = scenes.make_preset("s10k")
    corners = scene["positions"][scene["indices"]].astype(np.float64)
    area = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    assert area.min() > 0


def test_camera_basis_is_orthonormal():
    cam = scenes.make_preset("tiny")["camera"].astype(np.float64)
    right, up, forward = cam[3:6], cam[6:9], cam[9:12]
    for first, second in [(right, up), (right, forward), (up, forward)]:
        assert abs(first @ second) < 1e-6
    for first in (right, up, forward):
        assert abs(first @ first - 1) < 1e-6
    assert cam[12] == pytest.approx(512 / scenes.TAN_HALF_FOV, rel=1e-6)


def test_cube_sphere_is_closed_and_unit():
    verts, tris = scenes.cube_sphere(3)
    assert len(tris) == 12 * 9 and len(verts) == 6 * 16
    assert np.allclose(np.linalg.norm(verts, axis=1), 1)
    ids = {}
    welded = [ids.setdefault(tuple(np.round(vertex, 9)), len(ids)) for vertex in verts]
    assert len(ids) == 6 * 9 + 2
    edges = {}
    for triangle in tris:
        for start, end in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((welded[triangle[start]], welded[triangle[end]])))
            edges[key] = edges.get(key, 0) + 1
    assert set(edges.values()) == {2}


def test_save_load_and_export(tmp_path):
    scene = scenes.make_preset("tiny")
    scenes.save_scene(tmp_path / "a.npz", scene)
    again = scenes.load_scene(tmp_path / "a.npz")
    assert scenes.checksum(again) == scenes.checksum(scene)
    scenes.export_npy(scene, tmp_path / "npy")
    for key in scenes.KEYS:
        assert np.array_equal(np.load(tmp_path / "npy" / f"{key}.npy"), scene[key])
