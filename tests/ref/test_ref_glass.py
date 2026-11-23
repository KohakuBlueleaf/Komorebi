"""The glass layer of reference/raster.py on hand-made data: a quad seen through a
glass quad, or hidden by an opaque one (no scene files)."""

import numpy as np

from reference import public, raster

F = np.float32
GLASS = 1  # material numbers: 0 the opaque wall, 1 the glass


def quad(z, half):
    """Two triangles of a square at depth z, half its side wide, facing -z."""
    corners = [[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]]
    return np.array(corners, dtype=F), np.array([[0, 2, 1], [0, 3, 2]])


def scene_of(quads):
    """A scene of quads (z, half) seen from z = -10 along +z."""
    positions, faces = [], []
    for z, half in quads:
        p, f = quad(z, half)
        faces.append(f + 4 * len(positions))
        positions.append(p)
    positions = np.concatenate(positions)
    return {
        "positions": positions,
        "indices": np.concatenate(faces).astype(np.int32),
        "colors": np.full((len(positions), 3), 0.5, dtype=F),
        "camera": public.camera([0, 0, -10], [0, 0, 0], 60, 0.1),
        "light": np.array([0, 0, -10], dtype=F),
    }


def materials_of(material, wall=(0.6, 0.4, 0.2), tau=(1, 1, 1)):
    """Materials: 0 a `textured` wall without a texture, 1 `transparent` glass."""
    count = len(material)
    return {
        "material": np.array(material, dtype=np.int32),
        "kind": np.array([raster.TEXTURED, raster.TRANSPARENT], dtype=np.int32),
        "base_colour": np.array([wall, (0.3, 0.3, 0.3)], dtype=F),
        "textures": np.full((2, 4), -1, dtype=np.int32),
        "exponent": np.array([1, 100], dtype=np.int32),
        "specular": np.zeros((2, 3), dtype=F),
        "transmission": np.array([(0, 0, 0), tau], dtype=F),
        "uv": np.zeros((count, 3, 2), dtype=F),
        "light": np.ones((count, 3), dtype=F),
        "texels": np.full(4, 255, dtype=np.uint8),
        "mip": np.array([[[0, 1, 1]] * public.MIP_LEVELS], dtype=np.int64),
        "levels": np.ones(1, dtype=np.int32),
    }


def lights_of(scene, reflection=0.0):
    """A sun toward the eye, a sky, an environment map; no lamps, all lit."""
    rng = np.random.default_rng(4)
    return {
        "visibility": np.ones(len(scene["positions"]), dtype=F),
        "sh": rng.uniform(0, 0.2, (9, 3)).astype(F),
        "sun_direction": np.array([0.3, 0.4, -0.866], dtype=F),
        "sun_irradiance": np.array([2, 1.8, 1.5], dtype=F),
        "octahedral": rng.uniform(0, 3, (16, 16, 3)).astype(F),
        "turn": F(0),
        "shadow_map": np.zeros((1, 1), dtype=F),
        "shadow_camera": scene["camera"],
        "shadow_texel": F(0),
        "shadow_bias": F(0),
        "lamp_position": np.zeros((0, 3), dtype=F),
        "lamp_colour": np.zeros((0, 3), dtype=F),
        "lamp_intensity": np.zeros(0, dtype=F),
        "ambient": [0.1, 0.1, 0.1],
        "emissive_scale": 1.0,
        "exposure": 1.0,
        "radius": 100.0,
        "size": 1.0,
        "reflection": reflection,
    }


def without_glass(terms=raster.TERMS):
    return tuple(t for t in terms if t != "glass")


def test_a_glass_that_transmits_all_leaves_the_colour_behind():
    scene = scene_of([(5, 6), (0, 2)])
    materials = materials_of([0, 0, GLASS, GLASS])
    lights = lights_of(scene)
    seen = raster.rasterize_lighted(scene, materials, lights)
    front = seen["parts"]["glass_tri"] >= 0
    assert front.sum() > 10000
    behind = raster.rasterize_lighted(scene, materials, lights, terms=without_glass())
    np.testing.assert_array_equal(seen["colour"], behind["colour"])
    shaded = raster.rasterize_shaded(scene, materials)
    keep = materials["material"] != GLASS
    bare = raster.rasterize_shaded(raster.layer_scene(scene, keep), materials)
    assert shaded["front"].sum() == front.sum()
    np.testing.assert_array_equal(shaded["colour"], bare["colour"])


def test_a_glass_that_transmits_nothing_hides_what_is_behind():
    scene = scene_of([(5, 6), (0, 2)])
    colours = []
    for wall in ((0.6, 0.4, 0.2), (0.1, 0.9, 0.5)):
        materials = materials_of([0, 0, GLASS, GLASS], wall, tau=(0, 0, 0))
        seen = raster.rasterize_lighted(scene, materials, lights_of(scene))
        front = seen["parts"]["glass_tri"] >= 0
        shaded = raster.rasterize_shaded(scene, materials)
        np.testing.assert_array_equal(shaded["front"], front)
        colours.append((seen["colour"], shaded["colour"], front))
    (lit_a, shade_a, front), (lit_b, shade_b, _) = colours
    np.testing.assert_array_equal(lit_a[front], lit_b[front])
    np.testing.assert_array_equal(shade_a[front], shade_b[front])
    assert not np.array_equal(lit_a[~front], lit_b[~front])


def test_the_glass_loses_to_a_nearer_opaque_surface():
    scene = scene_of([(5, 6), (2, 2), (0, 3)])
    materials = materials_of([0, 0, GLASS, GLASS, 0, 0], tau=(0.5, 0, 0))
    lights = lights_of(scene, reflection=1.0)
    seen = raster.rasterize_lighted(scene, materials, lights)
    assert (seen["parts"]["glass_tri"] < 0).all()
    behind = raster.rasterize_lighted(scene, materials, lights, terms=without_glass())
    np.testing.assert_array_equal(seen["colour"], behind["colour"])
    shaded = raster.rasterize_shaded(scene, materials)
    assert not shaded["front"].any()
    assert (shaded["glass"]["tri"] >= 0).sum() > 0
