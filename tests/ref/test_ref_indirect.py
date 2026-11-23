"""Indirect light (reference/indirect.py) on hand-made data: form factors in a closed
box, the field's blend and its directions, the hierarchical apply against the dense one
(no scene files)."""

import numpy as np
import pytest

from reference import bvh, indirect, native
from reference import lights as light_data

F = np.float32
needs_cxx = pytest.mark.skipif(native.find_cxx() is None, reason="no C++ compiler")


@pytest.fixture(scope="module")
def build_dir(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    patch.setenv("KR_BUILD_DIR", str(tmp_path_factory.mktemp("indirect") / "build"))
    patch.setenv("KR_THREADS", "2")
    yield
    patch.undo()


def closed_box(cells=8, side=10.0):
    """A cube's 12 triangles facing in and cells x cells elements on each face (cell
    centres, inward normals, the cell's area)."""
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4)]
    faces += [(1, 5, 7, 3)]
    indices = [[a, b, c] for a, b, c, d in faces] + [[a, c, d] for a, b, c, d in faces]
    mesh = {
        "positions": (corners * side).astype(F),
        "indices": np.array(indices, np.int32),
    }
    centre = (np.arange(cells) + 0.5) / cells * side
    u, v = (a.ravel() for a in np.meshgrid(centre, centre, indexing="ij"))
    position, normal = [], []
    for axis in range(3):
        for wall, inward in ((0.0, 1.0), (side, -1.0)):
            p = np.zeros((len(u), 3))
            p[:, axis] = wall
            p[:, (axis + 1) % 3], p[:, (axis + 2) % 3] = u, v
            n = np.zeros((len(u), 3))
            n[:, axis] = inward
            position.append(p)
            normal.append(n)
    count = 6 * len(u)
    elements = {
        "position": np.concatenate(position).astype(F),
        "normal": np.concatenate(normal).astype(F),
        "area": np.full(count, (side / cells) ** 2, F),
        "sample": np.arange(count, dtype=np.int32),
    }
    return mesh, elements


@needs_cxx
def test_closed_box_receives_what_one_light_sends(build_dir):
    mesh, elements = closed_box()
    tree = {k: np.ascontiguousarray(v) for k, v in bvh.build(mesh).items()}
    count = len(elements["position"])
    f, _ = indirect.form_factor_rows(tree, elements, np.arange(count), 1e-3)
    light = np.zeros((count, 3), F)
    light[count // 2 + 27] = [1.0, 0.5, 0.25]
    e = np.pi * f.astype(np.float64) @ light
    sent = np.pi * elements["area"][count // 2 + 27] * light[count // 2 + 27]
    received = (elements["area"][:, None] * e).sum(0)
    np.testing.assert_allclose(received, sent, rtol=0.02)
    np.testing.assert_allclose(f.sum(1).mean(), 1.0, rtol=0.03)
    assert (np.diag(f) == 0).all()


def two_voxel_model(first, second):
    """A field of two neighbouring voxels along x (side 1, origin 0) with data in +y."""
    dims = np.array([4, 4, 4])
    keys = indirect.voxel_keys(np.array([[1, 1, 1], [2, 1, 1]]), dims)
    has = np.zeros((2, 6), bool)
    has[:, 2] = True
    values = np.zeros((2, 6, 3), F)
    values[0, 2], values[1, 2] = first, second
    model = {"field_spacing": F(1), "field_origin": np.zeros(3, F), "field_dims": dims}
    return model | {"field_keys": keys, "field_has": has}, values


def test_a_pixel_between_two_voxels_blends_them():
    first, second = [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]
    model, values = two_voxel_model(first, second)
    up = np.array([[0, 1, 0]] * 3, F)
    point = np.array([[1.5, 1.0, 1.5], [2.0, 1.0, 1.5], [2.25, 1.0, 1.5]], F)
    e, found = indirect.lookup(model, values, point, up)
    assert found.all()
    np.testing.assert_allclose(e[0], first, rtol=1e-6)
    np.testing.assert_allclose(e[1], np.add(first, second) / 2, rtol=1e-6)
    np.testing.assert_allclose(e[2], 0.25 * np.array(first) + 0.75 * np.array(second))
    _, found = indirect.lookup(model, values, point + F([0, 5, 0]), up)
    assert not found.any()


def test_an_element_behind_a_wall_feeds_only_the_directions_it_faces():
    position = np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], F)
    normal = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], F)
    field = indirect.build_field(position, normal, 1.0)
    rows, columns = field["field_rows"], field["field_columns"]
    np.testing.assert_array_equal(np.unique(rows[columns == 0] % 6), [1])
    np.testing.assert_array_equal(np.unique(rows[columns == 1] % 6), [0])
    values = indirect.field_values(field, np.array([[0, 0, 0], [1, 1, 1]], F))
    assert (values[:, 1] == 0).all()
    np.testing.assert_allclose(values[field["field_has"][:, 0], 0], 1.0, rtol=1e-6)


def test_the_hierarchical_apply_equals_the_dense_one():
    rng = np.random.default_rng(3)
    position = rng.random((2048, 3)).astype(F) * F([4, 1, 1])
    tree = indirect.cluster_tree(position, leaf_units=2)
    p = position[tree["perm"]].astype(np.float64)
    r2 = ((p[:, None] - p[None]) ** 2).sum(-1)
    matrix = (1.0 / (1.0 + 10.0 * r2)).astype(F)
    blocks, weights = indirect.hierarchical(matrix, tree, log=lambda _: None)
    assert (blocks["kind"] == 1).any() and (blocks["kind"] == 0).any()
    assert len(weights) < matrix.size
    x = rng.random((2048, 3)).astype(F)
    y = indirect.apply(blocks, weights, x)
    exact = matrix.astype(np.float64) @ x
    assert np.linalg.norm(y - exact) / np.linalg.norm(exact) < 1e-2


def test_the_packed_field_keeps_has_and_the_values_to_their_codes():
    rng = np.random.default_rng(5)
    values = (rng.random((40, 6, 3)) ** 3 * 7.0).astype(F)
    values[0, 0] = [7.5, 0.0, -1e-3]  # the largest red; a negative clamped at 0
    has = rng.random((40, 6)) < 0.7
    has[0, 0] = True
    texels, scale = indirect.pack_field(values, has)
    assert texels.dtype == np.uint32 and scale == F(7.5)
    np.testing.assert_array_equal((texels >> 31) != 0, has)
    got, got_has = indirect.unpack_field(texels, scale)
    np.testing.assert_array_equal(got_has, has)
    assert (got[~has] == 0).all()
    assert got[0, 0, 0] == F(7.5) and got[0, 0, 1] == 0 and got[0, 0, 2] == 0
    # a code is the nearest of sqrt(v / scale) x (2^bits - 1): half a step of the root
    for c, bits in enumerate(indirect.PACK_BITS):
        most = (1 << bits) - 1
        root = np.sqrt(np.maximum(values[..., c][has], 0) / scale)
        back = np.sqrt(got[..., c][has] / scale)
        assert np.abs(root - back).max() <= 0.5 / most + 1e-6


def bounce_model_and_lights(path):
    """A hand-made indirect model (the closed box's elements in a cluster tree's order,
    a small transport, random sky transfer rows, a field) under `path` in indirect's
    models, and lights lighting it: a sun with nothing in its shadow map, a grey sky."""
    mesh, elements = closed_box(cells=4)
    tree = indirect.cluster_tree(elements["position"], leaf_units=1)
    order = tree["perm"]
    p = elements["position"][order].astype(np.float64)
    r2 = ((p[:, None] - p[None]) ** 2).sum(-1)
    matrix = (0.3 / len(p) * 100.0 / (100.0 + r2)).astype(F)
    np.fill_diagonal(matrix, 0)
    blocks, weights = indirect.hierarchical(matrix, tree, log=lambda _: None)
    rng = np.random.default_rng(7)
    count = len(p)
    model = {
        "position": elements["position"][order],
        "normal": elements["normal"][order],
        "real": np.ones(count, bool),
        "albedo": rng.uniform(0.2, 0.9, (count, 3)).astype(F),
        "emission": np.zeros((count, 3), F),
        "transfer": rng.uniform(0, 0.2, (count, indirect.SKY_TERMS)).astype(F),
        "blocks": blocks,
        "weights": weights,
    }
    model |= indirect.build_field(model["position"], model["normal"], 2.5)
    indirect._MODELS[path] = model
    sun = np.array([0.3, 0.8, 0.5], F) / np.linalg.norm([0.3, 0.8, 0.5]).astype(F)
    lights = {
        "indirect": path,
        "sun_direction": sun,
        "sun_irradiance": F([3, 2, 1]),
        "turn": F(0),
        "octahedral": np.full((128, 128, 3), 0.5, F),
        "lamp_position": np.zeros((0, 3), F),
        "lamp_colour": np.zeros((0, 3), F),
        "lamp_intensity": np.zeros(0, F),
        "size": 1.0,
        "radius": 1.0,
        "emissive_scale": 1.0,
        "shadow_map": np.zeros((1, 1), F),
        "shadow_camera": light_data.sun_camera(mesh["positions"], sun, 1),
        "shadow_texel": F(0.01),
        "shadow_bias": F(0),
    }
    return model, lights


def test_the_bounce_count_sums_that_many_passes(tmp_path):
    model, lights = bounce_model_and_lights(str(tmp_path / "box_indirect.npz"))
    light = indirect.element_light(lights, model, 0)
    sky, l0 = light["sky"], light["l0"]
    assert light["passes"] == [] and np.isfinite(sky).all() and sky.any()
    e1 = indirect.apply(model["blocks"], model["weights"], l0)
    a = model["albedo"] * F(1 / np.pi)
    e2 = indirect.apply(model["blocks"], model["weights"], a * e1)
    e3 = indirect.apply(model["blocks"], model["weights"], a * e2)
    assert e3.any()
    want = {
        0: indirect.field_values(model, sky),
        1: indirect.field_values(model, e1 + sky),
        2: indirect.field_values(model, (e1 + e2) + sky),  # today's sum
        3: indirect.field_values(model, ((e1 + e2) + e3) + sky),
    }
    for count, values in want.items():
        _, got = indirect.field_for({**lights, "bounces": count})
        np.testing.assert_array_equal(got, values)
    _, default = indirect.field_for(lights)  # no count: 2, bit for bit
    np.testing.assert_array_equal(default, want[2])
    assert not np.array_equal(want[3], want[2]) and not np.array_equal(want[1], want[0])


def test_a_bounce_count_outside_0_to_3_is_refused(tmp_path):
    _, lights = bounce_model_and_lights(str(tmp_path / "box_indirect.npz"))
    for bad in (4, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="bounces"):
            light_data.bounce_count(bad)
        with pytest.raises(ValueError, match="bounces"):
            indirect.field_for({**lights, "bounces": bad})
    assert [light_data.bounce_count(n) for n in (0, 1, 2, 3)] == [0, 1, 2, 3]
