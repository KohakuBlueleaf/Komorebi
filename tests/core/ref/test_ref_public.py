"""The public scene table, the fetch check and the conversion, on files written here."""

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from reference import native, public, scenes

README = Path(__file__).resolve().parents[3] / "reference" / "SCENES.md"

OBJ = """mtllib room.mtl
v 0 0 0
v 4 0 0
v 4 0 4
v 0 0 4
v 1 0 1
v 2 2 1
v 3 0 1
usemtl floor
f 1 2 3 4
usemtl wall
f 5 7 6
"""
MTL = "newmtl floor\nKd 0.2 0.4 0.6\nnewmtl wall\nKd 0.9 0.1 0.1\n"


@pytest.fixture(autouse=True)
def no_extra_table(monkeypatch):
    monkeypatch.delenv("KR_SCENES_EXTRA", raising=False)


def test_table_is_complete():
    table = public.table()
    assert table
    for name, entry in table.items():
        assert re.fullmatch(r"[a-z_]+", name)
        for key in ("title", "page", "licence", "attribution", "light", "views"):
            assert entry[key], (name, key)
        assert len(entry["light"]) == 3
        files = {file["url"].rsplit("/", 1)[1] for file in entry["files"]}
        for file in entry["files"]:
            assert file["url"].startswith("https://")
            assert re.fullmatch(r"[0-9a-f]{64}", file["sha256"])
            assert file["bytes"] > 0
        for mesh in entry["meshes"]:
            assert mesh["file"] in files
        assert set(entry["views"]) == {"overview", "close"}
        for view in entry["views"].values():
            assert len(view["eye"]) == 3 and len(view["target"]) == 3
            assert view["eye"] != view["target"] and 0 < view["fov"] < 180


def test_attribution_is_documented():
    text = " ".join(README.read_text().split())
    for name, entry in public.table().items():
        assert f"`{name}`" in text
        assert entry["attribution"] in text


def test_extra_table_is_added(tmp_path, monkeypatch):
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps(fake_table(tmp_path)))
    monkeypatch.setenv("KR_SCENES_EXTRA", str(extra))
    assert "room" in public.table() and "sponza" in public.table()
    assert public.is_view("room_close") and public.triangles("room_close") == 3


def test_view_names():
    views = public.views()
    for name, entry in public.table().items():
        for view in entry["views"]:
            assert views[f"{name}_{view}"] == (name, view)
    assert not public.is_view("s10k")


def test_camera_is_orthonormal():
    cam = public.camera([1.0, 2.0, 3.0], [0.0, 0.5, -1.0], 60.0, 0.01).astype(float)
    right, up, forward = cam[3:6], cam[6:9], cam[9:12]
    for axis in (right, up, forward):
        assert abs(np.dot(axis, axis) - 1) < 1e-6
    assert abs(np.dot(right, up)) < 1e-6 and abs(np.dot(up, forward)) < 1e-6
    assert abs(right[1]) < 1e-7 and up[1] > 0
    assert cam[12] == pytest.approx(512 / np.tan(np.radians(30)), rel=1e-6)


def fake_table(tmp_path):
    """A one-scene table whose file is a zip written here, fetched by file:// URL."""
    zpath = tmp_path / "src" / "room.zip"
    zpath.parent.mkdir()
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("room.obj", OBJ)
        z.writestr("room.mtl", MTL)
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    view = {"eye": [2.0, 3.0, 8.0], "target": [2.0, 0.0, 2.0], "fov": 50}
    return {
        "room": {
            "title": "Room",
            "page": "test",
            "licence": "test",
            "attribution": "test",
            "triangles": 3,
            "files": [{"url": zpath.as_uri(), "sha256": digest, "bytes": 1}],
            "meshes": [{"file": "room.zip", "member": "room.obj"}],
            "light": [2.0, 5.0, 2.0],
            "views": {"overview": view, "close": view},
        }
    }


def test_fetch_checks_the_checksum(tmp_path, monkeypatch):
    table = fake_table(tmp_path)
    monkeypatch.setattr(public, "table", lambda: table)
    (path,) = public.fetch("room", tmp_path / "data")
    assert path.exists()
    table["room"]["files"][0]["sha256"] = "0" * 64
    path.unlink()
    with pytest.raises(ValueError, match="SHA-256"):
        public.fetch("room", tmp_path / "data")
    assert not path.exists()


def test_convert_and_view(tmp_path, monkeypatch):
    table = fake_table(tmp_path)
    monkeypatch.setattr(public, "table", lambda: table)
    monkeypatch.setenv("KR_DATA", str(tmp_path / "data"))
    with pytest.raises(FileNotFoundError, match="reference.public fetch room"):
        native.ensure_scene("room_overview")
    public.convert("room", tmp_path / "data")
    folder = native.ensure_scene("room_overview")
    scene = scenes.load_scene(tmp_path / "data" / "room_overview.npz")
    assert (folder / "positions.npy").exists()
    assert scene["meta"][0] == 3 and native.scene_triangles("room_overview") == 3
    tri = scene["positions"][scene["indices"]]
    colour = scene["colors"][scene["indices"]]
    # flat colour per triangle (vertices split by material), shaded by the fixed term
    assert (colour == colour[:, :1]).all()
    floor = np.isclose(tri[:, :, 1], 0).all(axis=1)
    assert floor.sum() == 2
    shade = colour[:, 0, :] / np.where(floor[:, None], [0.2, 0.4, 0.6], [0.9, 0.1, 0.1])
    assert np.allclose(shade, shade[:, :1], rtol=1e-6)
    expected = public.SHADE_BASE + public.SHADE_GAIN * public.SHADE_DIR[1]
    assert np.allclose(shade[floor], expected, rtol=1e-6)
    assert np.allclose(scene["light"], [2, 5, 2])
    assert scene["eps"][0] > 0 and scene["camera"][15] > 0


def test_ground_under_single_objects():
    lower, upper = np.array([-1.0, 0.5, -2.0]), np.array([1.0, 3.0, 2.0])
    positions, faces, tint = public.ground(lower, upper, 1.5)
    assert len(faces) == 2 * public.GROUND_CELLS**2 == len(tint)
    assert (positions[:, 1] == 0.5).all()
    assert positions[:, 0].min() == pytest.approx(-6) and positions[
        :, 2
    ].max() == pytest.approx(6)
    corners = positions[faces]
    area = (
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 1] / 2
    )
    assert area.sum() == pytest.approx(144)


def test_shared_vertices_keep_triangles():
    positions = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    colour = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    vertex_positions, vertex_colours, indices = public.shared_vertices(
        positions, faces, colour
    )
    assert len(vertex_positions) == 6
    np.testing.assert_array_equal(vertex_positions[indices], positions[faces])
    np.testing.assert_array_equal(vertex_colours[indices][:, 0], colour)
