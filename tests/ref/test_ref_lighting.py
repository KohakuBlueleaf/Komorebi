"""The lighting of reference/lights.py on hand-made data: the sky's irradiance, a point
light's reach and the sun's shadow map (no scene files)."""

import numpy as np

from reference import lights

F = np.float32


def test_constant_sky_gives_pi_times_its_radiance():
    radiance = np.array([0.7, 1.3, 2.0])
    d, solid = lights.texel_directions(64, 128)
    sh = np.einsum("hwk,hw->k", lights.sh_basis(d), solid)[:, None] * radiance
    rng = np.random.default_rng(1)
    n = rng.normal(size=(200, 3))
    n = (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(F)
    e = lights.sky_irradiance(sh.astype(F), n)
    np.testing.assert_allclose(e, np.pi * radiance[None, :] * np.ones((200, 1)), 2e-3)


def test_point_light_is_zero_at_and_beyond_its_radius():
    radius = F(10)
    position = np.array([0, 0, 0], dtype=F)
    d = np.array([0.5, 5.0, 9.99, 10.0, 10.5, 100.0], dtype=F)
    points = np.stack([np.zeros_like(d), -d, np.zeros_like(d)], 1)
    normals = np.tile(np.array([[0, 1, 0]], dtype=F), (len(d), 1))
    e, _ = lights.point_irradiance(F(100), F(0.5), radius, position, points, normals)
    assert (e[:3] > 0).all()
    assert (e[3:] == 0).all()


def test_floor_under_a_box_is_shadowed_and_lit_beside_it():
    floor = [[-10, 0, -10], [10, 0, -10], [10, 0, 10], [-10, 0, 10]]
    box = [[x, y, z] for x in (-2, 2) for y in (3, 5) for z in (-2, 2)]
    positions = np.array(floor + box, dtype=F)
    faces = [[0, 2, 1], [0, 3, 2]]
    for a, b, c, e in [(0, 1, 3, 2), (4, 6, 5, 7), (0, 4, 1, 5), (2, 3, 6, 7)]:
        faces += [[4 + a, 4 + b, 4 + c], [4 + b, 4 + e, 4 + c]]
    for a, b, c, e in [(0, 2, 4, 6), (1, 5, 3, 7)]:
        faces += [[4 + a, 4 + b, 4 + c], [4 + b, 4 + e, 4 + c]]
    scene = {"positions": positions, "indices": np.array(faces, dtype=np.int32)}
    sun = np.array([0.0, 1.0, 0.0], dtype=F)
    depth, camera, texel = lights.shadow_map(scene, sun)
    state = {
        "shadow_map": depth,
        "shadow_camera": camera,
        "shadow_texel": texel,
        "shadow_bias": F(texel / float(texel * camera[12]) ** 2),
    }
    points = np.array([[0, 0, 0], [1, 0, -1], [6, 0, 0], [-6, 0, 7]], dtype=F)
    up = np.tile(np.array([[0, 1, 0]], dtype=F), (4, 1))
    lit = lights.sun_shadow(state, points, up)
    np.testing.assert_array_equal(lit, [0, 0, 1, 1])
    one_tap = lights.sun_shadow(state, points, up, pcf=False)
    np.testing.assert_array_equal(one_tap, [0, 0, 1, 1])
    outside = lights.sun_shadow(state, np.array([[90, 0, 0]], dtype=F), up[:1])
    assert outside[0] == 1


def test_authored_lights_replace_the_emissive_groups():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=F)
    scene = {"positions": positions, "indices": np.array([[0, 1, 2]], np.int32)}
    materials = {
        "material": np.array([0], np.int32),
        "kind": np.array([4], np.int32),
        "base_colour": np.ones((1, 3), F),
        "textures": np.full((1, 4), -1, np.int32),
    }
    entry = {"merge_distance": 1.0, "lamp_scale": 7.0}
    position, _, intensity, _ = lights.scene_lamps(entry, scene, materials)
    assert len(position) == 1 and intensity[0] == F(7.0 * 0.5)
    entry["lights"] = [
        {"position": [5, 6, 7], "colour": [1, 0.5, 0.25]},
        {"position": [-1, 2, 3], "colour": [0.2, 0.2, 0.2]},
    ]
    position, colour, intensity, _ = lights.scene_lamps(entry, scene, materials)
    np.testing.assert_array_equal(position, [[5, 6, 7], [-1, 2, 3]])
    np.testing.assert_array_equal(colour[0], F([1, 0.5, 0.25]))
    np.testing.assert_array_equal(intensity, [7.0, 7.0])
