"""The Tungsten scene reader of reference/tungsten.py on a hand-written scene: an OBJ
and a .wo3 mesh, a texture, a quad emitter, a tinted glass and the camera."""

import json
import struct
import zipfile

import numpy as np

from reference import public, tungsten

OBJ = (
    "v 0 0 0\nv 1 0 0\nv 1 0 1\nv 0 0 1\n"
    "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n"
)
SCENE = {
    "media": [{"name": "red", "sigma_a": [10.0, 400.0, 400.0], "density": 1.0}],
    "bsdfs": [
        {"name": "Paint", "albedo": [0.8, 0.5, 0.2], "type": "lambert"},
        {"name": "Tiles", "albedo": "textures/tiles.png", "type": "rough_plastic"},
        {"name": "Glass", "albedo": 1.0, "type": "dielectric", "ior": 1.5},
    ],
    "primitives": [
        {"type": "mesh", "file": "models/floor.obj", "bsdf": "Tiles"},
        {
            "type": "mesh",
            "file": "models/wall.wo3",
            "bsdf": "Paint",
            "transform": {"position": [0, 0, 5], "scale": 2.0},
        },
        {
            "type": "mesh",
            "file": "models/floor.obj",
            "bsdf": "Glass",
            "int_medium": "red",
            "transform": {"position": [0, 1, 0]},
        },
        {
            "type": "quad",
            "emission": [3.0, 1.5, 0.0],
            "bsdf": {"type": "null"},
            "transform": {"position": [0, 3, 0], "rotation": [180, 0, 0]},
        },
        {"type": "curves", "file": "models/rug.fiber", "bsdf": "Paint"},
    ],
    "camera": {
        "fov": 60.0,
        "transform": {"position": [0, 1, -4], "look_at": [0, 1, 0], "up": [0, 1, 0]},
    },
}


def wo3(positions, uv, faces, normal=(0, 0, 0)):
    """A .wo3 file's bytes: vertices (position, normal, uv), triangles."""
    out = struct.pack("<Q", len(positions))
    for p, t in zip(positions, uv):
        out += struct.pack("<8f", *p, *normal, *t)
    out += struct.pack("<Q", len(faces))
    for f in faces:
        out += struct.pack("<3Ii", *f, 0)
    return out


def scene_zip(tmp_path):
    path = tmp_path / "scene.zip"
    square = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("room/scene.json", json.dumps(SCENE))
        z.writestr("room/models/floor.obj", OBJ)
        z.writestr(
            "room/models/wall.wo3",
            wo3(
                square,
                [[0, 0], [1, 0], [1, 1], [0, 1]],
                [[0, 1, 2], [0, 2, 3]],
                (0, 0, -1),
            ),
        )
        z.writestr("room/textures/tiles.png", b"not read by the scene reader")
    return path


def test_read_wo3_rejects_a_wrong_length():
    data = wo3([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0]] * 3, [[0, 1, 2]])
    p, normals, uv, faces = tungsten.read_wo3(data)
    assert p.shape == normals.shape == (3, 3) and uv.shape == (3, 2)
    assert faces.tolist() == [[0, 1, 2]]
    try:
        tungsten.read_wo3(data + b"\0")
    except ValueError:
        return
    raise AssertionError("a .wo3 file of the wrong length was read")


def test_scene_geometry_counts_and_camera(tmp_path):
    path = scene_zip(tmp_path)
    mesh = tungsten.read_scene(path, "room/scene.json", with_uv=True)
    assert mesh.counts["skipped_curves"] == 1 and mesh.counts["emitters"] == 1
    assert mesh.counts["triangles"] == 8 and mesh.counts["unmapped_bsdfs"] == 0
    names = mesh.material_names
    assert names == ["Tiles", "Paint", "Glass+red", "emitter_3"]
    wall = mesh.positions[mesh.faces[mesh.material == names.index("Paint")]]
    assert np.allclose(wall.reshape(-1, 3).min(0), [0, 0, 5])
    assert np.allclose(wall.reshape(-1, 3).max(0), [2, 2, 5])
    turned = np.cross(wall[:, 1] - wall[:, 0], wall[:, 2] - wall[:, 0])
    assert (turned[:, 2] < 0).all()  # wound to face its normals, -z
    lamp = mesh.positions[mesh.faces[mesh.material == names.index("emitter_3")]]
    normal = np.cross(lamp[:, 1] - lamp[:, 0], lamp[:, 2] - lamp[:, 0])
    assert np.allclose(normal / np.linalg.norm(normal, axis=1)[:, None], [0, -1, 0])
    floor = mesh.uv[mesh.material == names.index("Tiles")]
    assert floor.min() == 0 and floor.max() == 1
    eye, target, fov = tungsten.camera(SCENE)
    assert eye.tolist() == [0, 1, -4] and target.tolist() == [0, 1, 0] and fov == 60


class Files:
    """Stands in for public.TextureFiles: every named file exists."""

    def find(self, text):
        return ("archive", text) if text else None


def record(statements, **options):
    return public.material_record(statements, Files(), lambda f, r: 0, options)


def test_materials_map_to_kinds(tmp_path):
    mesh = tungsten.read_scene(scene_zip(tmp_path), "room/scene.json", with_uv=True)
    s = mesh.statements
    kind, base, *_ = record(s["Paint"])
    assert kind == public.KINDS.index("flat") and np.allclose(base, [0.8, 0.5, 0.2])
    kind, base, slots, exponent, specular, _ = record(s["Tiles"])
    assert kind == public.KINDS.index("textured") and slots[0] == 0
    assert np.allclose(specular, tungsten.PLASTIC_SPECULAR)
    kind, base, _, _, _, tau = record(s["Glass+red"])
    assert kind == public.KINDS.index("transparent")
    expected = tungsten.GLASS_TRANSMISSION * np.exp(
        -np.array([10.0, 400.0, 400.0]) * tungsten.MEDIUM_DEPTH
    )
    assert np.allclose(tau, expected, atol=1e-5) and base[0] == 1 and base[1] < 0.2
    kind, base, *_ = record(s["emitter_3"], emission_scale=3.0)
    assert kind == public.KINDS.index("emissive") and np.allclose(base, [1, 0.5, 0])


def test_inside_turns_named_materials_to_face_the_point(tmp_path):
    path = scene_zip(tmp_path)
    inside = {"point": [0.5, 1.0, 0.5], "materials": ["Tiles"]}
    mesh = tungsten.read_scene(path, "room/scene.json", with_uv=True, inside=inside)
    floor = mesh.positions[mesh.faces[mesh.material == 0]]
    normal = np.cross(floor[:, 1] - floor[:, 0], floor[:, 2] - floor[:, 0])
    assert (normal[:, 1] > 0).all() and mesh.counts["turned_inside"] == 2
