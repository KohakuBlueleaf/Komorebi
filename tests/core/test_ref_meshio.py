"""The OBJ, MTL and PLY readers on small files written here."""

import gzip
import io
import struct
import tarfile
import zipfile

import numpy as np
import pytest

from ref import meshio

MTL = """# two materials
newmtl red
Kd 1.0 0.0 0.0
newmtl blank material
Ns 10
"""

OBJ = """mtllib scene.mtl
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 2 0
vt 0 0
vn 0 0 1
f 1 2 3
usemtl red
f 1/1/1 2/1/1 3/1/1 4/1/1
usemtl blank material
f -5 -4 -1
f 1 2 2
f 1 3 5
f 3 3 3 4
usemtl unknown
f 2 5 3
"""


def faces_as_sets(mesh):
    """The triangles as sets of corner positions (order-free)."""
    corners = mesh.positions[mesh.faces]
    return [frozenset(map(tuple, triangle)) for triangle in corners.tolist()]


def test_mtl_colours():
    colours = meshio.parse_mtl(MTL)
    assert colours == {"red": (1.0, 0.0, 0.0), "blank material": None}


def test_obj_fans_materials_and_degenerates():
    mesh = meshio.parse_obj(OBJ, meshio.parse_mtl(MTL))
    assert mesh.counts["polygons"] == 7
    # quad -> 2 triangles; "3 3 3 4" -> 2 fans, both with a repeated vertex
    assert mesh.counts["dropped_repeated_vertex"] == 3
    assert mesh.counts["dropped_collinear"] == 1
    assert mesh.counts["triangles"] == 5
    assert mesh.counts["materials"] == 3
    assert mesh.counts["materials_without_colour"] == 2
    assert mesh.counts["triangles_without_material"] == 1
    grey, red = meshio.DEFAULT_COLOUR, (1.0, 0.0, 0.0)
    assert [tuple(row) for row in mesh.colour] == [grey, red, red, grey, grey]
    np.testing.assert_array_equal(mesh.faces[1:3], [[0, 1, 2], [0, 2, 3]])
    # negative indices count back from the last vertex read so far
    np.testing.assert_array_equal(mesh.faces[3], [0, 1, 4])


def test_equal_positions_are_dropped():
    text = "v 0 0 0\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n"
    mesh = meshio.parse_obj(text, {})
    assert mesh.counts["dropped_equal_positions"] == 1
    assert mesh.counts["triangles"] == 1


def ply_header(fmt, faces, extra_vertex=True, face_head=False):
    """Header of a 4-vertex PLY with optional extra properties."""
    lines = ["ply", f"format {fmt} 1.0", "comment test", "element vertex 4"]
    lines += ["property float x", "property float y", "property float z"]
    if extra_vertex:
        lines += ["property float confidence"]
    lines += [f"element face {faces}"]
    if face_head:
        lines += ["property uchar intensity"]
    lines += ["property list uchar int vertex_indices", "end_header"]
    return ("\n".join(lines) + "\n").encode()


VERTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]


def test_ply_ascii():
    body = "".join(f"{x} {y} {z} 0.5\n" for x, y, z in VERTS)
    body += "3 0 1 2\n4 0 1 2 3\n"
    mesh = meshio.read_ply(ply_header("ascii", 2) + body.encode())
    np.testing.assert_array_equal(mesh.positions, VERTS)
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 1, 2], [0, 2, 3]])
    assert mesh.counts["polygons"] == 2


@pytest.mark.parametrize("order", ["<", ">"])
@pytest.mark.parametrize("uniform", [True, False])
def test_ply_binary(order, uniform):
    fmt = "binary_little_endian" if order == "<" else "binary_big_endian"
    faces = [[0, 1, 2], [0, 2, 3]] if uniform else [[0, 1, 2], [0, 1, 2, 3]]
    body = b"".join(struct.pack(order + "4f", *vertex, 0.5) for vertex in VERTS)
    for face in faces:
        body += struct.pack(order + "BB", 7, len(face)) + struct.pack(
            order + f"{len(face)}i", *face
        )
    mesh = meshio.read_ply(ply_header(fmt, 2, face_head=True) + body)
    np.testing.assert_array_equal(mesh.positions, VERTS)
    expected = [[0, 1, 2], [0, 2, 3]] if uniform else [[0, 1, 2], [0, 1, 2], [0, 2, 3]]
    np.testing.assert_array_equal(mesh.faces, expected)


def test_archives(tmp_path):
    zpath = tmp_path / "scene.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("dir/scene.obj", OBJ)
        z.writestr("dir/scene.mtl", MTL)
    mesh = meshio.read_mesh(zpath, "dir/scene.obj")
    assert mesh.counts["triangles"] == 5
    assert tuple(mesh.colour[1]) == (1.0, 0.0, 0.0)
    ply = ply_header("ascii", 1, extra_vertex=False)
    ply += "".join(f"{x} {y} {z}\n" for x, y, z in VERTS).encode() + b"3 0 1 2\n"
    tpath = tmp_path / "scan.tar.gz"
    with tarfile.open(tpath, "w:gz") as archive:
        info = tarfile.TarInfo("scan/mesh.ply")
        info.size = len(ply)
        archive.addfile(info, io.BytesIO(ply))
    assert meshio.read_mesh(tpath, "scan/mesh.ply").counts["triangles"] == 1
    gpath = tmp_path / "mesh.ply.gz"
    gpath.write_bytes(gzip.compress(ply))
    assert meshio.read_mesh(gpath, "mesh.ply").counts["triangles"] == 1


def test_colours_replace_the_diffuse_colour(tmp_path):
    zpath = tmp_path / "scene.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("scene.obj", OBJ)
        z.writestr("scene.mtl", MTL)
    mesh = meshio.read_mesh(zpath, "scene.obj", {"red": [0.25, 0.5, 0.75]})
    assert tuple(mesh.colour[1]) == (0.25, 0.5, 0.75)
    assert mesh.counts["materials_without_colour"] == 2
