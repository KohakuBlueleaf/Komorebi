"""Materials of the public scenes and the rasterizer's SHADE, on scenes written here:
the conversion, each kind's pixel, a hole showing the layer behind, numpy against C++.
"""

import hashlib
import subprocess
import zipfile

import numpy as np
import pytest

from projects.raster.reference import harness
from reference import native, png, public, raster, scenes

OBJ = """mtllib quad.mtl
v -1 -1 2
v 1 -1 2
v 1 1 2
v -1 1 2
vt 0 0
vt 1 0
vt 1 1
vt 0 1
usemtl painted
f 1/1 2/2 3/3 4/4
"""
MTL = "newmtl painted\nKd 1 1 1\nNs 7.6\nmap_Kd tex.png\nmap_d ..\\Masks\\mask.png\n"
# A 4 x 4 texture: texel (row, column) has red 16 row + 60 column, green 200 - 40 row.
TEXTURE = np.stack(
    np.broadcast_arrays(
        16 * np.arange(4)[:, None] + 60 * np.arange(4)[None, :],
        200 - 40 * np.arange(4)[:, None],
        np.full((4, 4), 90),
    ),
    axis=-1,
).astype(np.uint8)
# The mask: the left half of the texture is a hole.
MASK = np.where(np.arange(4)[None, :] < 2, 20, 230).repeat(4, axis=0).astype(np.uint8)
CAMERA = np.array(
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 512, 512, 512, 0.01], dtype=np.float32
)


def decode_png(data, tmp_path):
    """png.read_png on bytes (the PNGs here are written by png.write_png)."""
    path = tmp_path / "decode.png"
    path.write_bytes(data)
    pixels = png.read_png(path)
    return pixels.reshape(pixels.shape[0], pixels.shape[1], -1)


def write_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return {
        "url": path.as_uri(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def test_conversion(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    png.write_png(source / "tex.png", TEXTURE)
    png.write_png(source / "mask.png", MASK)
    mesh_zip = write_zip(
        source / "quad.zip",
        {
            "scene/quad.obj": OBJ,
            "scene/quad.mtl": MTL,
            "scene/tex.png": (source / "tex.png").read_bytes(),
        },
    )
    masks_zip = write_zip(
        source / "Masks.zip", {"mask.png": (source / "mask.png").read_bytes()}
    )
    view = {"eye": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 2.0], "fov": 90}
    table = {
        "quad": {
            "title": "Quad",
            "page": "test",
            "licence": "test",
            "attribution": "test",
            "triangles": 2,
            "files": [mesh_zip],
            "texture_files": [masks_zip],
            "meshes": [{"file": "quad.zip", "member": "scene/quad.obj"}],
            "materials": {"archives": {"Masks": "Masks.zip"}},
            "light": [0.0, 3.0, 0.0],
            "views": {"overview": view, "close": view},
        }
    }
    monkeypatch.setattr(public, "table", lambda: table)
    monkeypatch.setattr(public, "decode_image", lambda data: decode_png(data, tmp_path))
    data = tmp_path / "data"
    public.convert("quad", data)
    public.convert_materials("quad", data, workers=1)
    materials, counts = public.load_materials("quad_overview", data)
    mesh, _ = public.load_mesh("quad", data)
    assert counts["triangles_by_kind"]["masked"] == 2
    assert list(materials["material"]) == [1, 1]
    assert materials["kind"][1] == raster.MASKED and materials["exponent"][1] == 8
    assert list(materials["textures"][1]) == [0, 1, -1, -1]
    # the corners' coordinates follow the corners' positions: u = (x + 1) / 2
    corners = mesh["positions"][mesh["indices"]]
    np.testing.assert_allclose(materials["uv"], (corners[:, :, :2] + 1) / 2)
    shade = public.vertex_shade(mesh["positions"].astype(np.float64), mesh["indices"])[
        mesh["indices"]
    ]
    np.testing.assert_allclose(materials["light"], shade, rtol=1e-6)
    # mip chains 4 x 4, 2 x 2, 1 x 1; level 0 is the image with alpha 255
    assert list(materials["levels"]) == [3, 3]
    np.testing.assert_array_equal(
        materials["mip"][0, :3], [[0, 4, 4], [64, 2, 2], [80, 1, 1]]
    )
    level0 = materials["texels"][:64].reshape(4, 4, 4)
    np.testing.assert_array_equal(level0[:, :, :3], TEXTURE)
    assert (level0[:, :, 3] == 255).all()
    level1 = materials["texels"][64:80].reshape(2, 2, 4)
    expected = (TEXTURE.reshape(2, 2, 2, 2, 3).astype(int).sum(axis=(1, 3)) + 2) // 4
    np.testing.assert_array_equal(level1[:, :, :3], expected)
    mask0 = materials["texels"][materials["mip"][1, 0, 0] :][:64].reshape(4, 4, 4)
    np.testing.assert_array_equal(mask0[:, :, 3], MASK)


def quad(z, x0=-1.0, x1=1.0):
    """Corners of a square at depth z facing the camera, and its coordinates."""
    positions = [[x0, -1, z], [x1, -1, z], [x1, 1, z], [x0, 1, z]]
    return np.array(positions, dtype=np.float32)


def scene_and_materials(front_kind, back_kind=raster.FLAT, normal=False):
    """A textured square at z = 2 in front of a square at z = 4, as a scene and its
    materials: square k's material is k + 1, texture 0 is TEXTURE, 1 the mask, 2 a
    normal map leaning to +u."""
    positions = np.concatenate([quad(2.0), quad(4.0, -4.0, 4.0)])
    indices = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)
    scene = {
        "positions": positions,
        "colors": np.repeat([[0.2, 0.4, 0.6], [0.7, 0.1, 0.3]], 4, axis=0).astype(
            np.float32
        ),
        "indices": indices,
        "camera": CAMERA,
        "light": np.array([0.0, 3.0, 0.0], dtype=np.float32),
        "eps": np.array([1e-4], dtype=np.float32),
        "meta": np.array([4, 8, 0, 0, 0], dtype=np.int32),
    }
    square_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    uv = square_uv[np.array([[0, 1, 2], [0, 2, 3]] * 2)]
    rgba = np.concatenate([TEXTURE, np.full((4, 4, 1), 255, np.uint8)], axis=2)
    mask = np.repeat(MASK[:, :, None], 4, axis=2)
    lean = np.array([200, 128, 200, 255], dtype=np.uint8)
    normal_map = np.broadcast_to(lean, (4, 4, 4))
    texels = np.concatenate([a.reshape(-1) for a in (rgba, mask, normal_map)])
    mip = np.zeros((3, 11, 3), dtype=np.int64)
    mip[:, 0] = [[0, 4, 4], [64, 4, 4], [128, 4, 4]]
    textures = [[0, 1, 2 if normal else -1, -1], [0, -1, -1, -1]]
    materials = {
        "uv": uv,
        "material": np.array([1, 1, 2, 2], dtype=np.int32),
        "light": np.full((4, 3), 0.8, dtype=np.float32),
        "kind": np.array([raster.FLAT, front_kind, back_kind], dtype=np.int32),
        "base_colour": np.array(
            [[1, 1, 1], [0.9, 1.0, 0.5], [1, 1, 1]], dtype=np.float32
        ),
        "textures": np.array([[-1] * 4, *textures], dtype=np.int32),
        "exponent": np.array([1, 8, 1], dtype=np.int32),
        "specular": np.array([[0, 0, 0], [0.3, 0.3, 0.3], [0, 0, 0]], np.float32),
        "texels": texels,
        "mip": mip,
        "levels": np.ones(3, dtype=np.int32),
    }
    return scene, materials


# Pixels in texel (1, 1) and (2, 3) of the front square (screen 256 .. 768, 128 pixels
# a texel), and on the back square only (x 0 .. 1024, y 384 .. 640).
LEFT, RIGHT, OUTSIDE = (420, 450), (700, 560), (900, 500)


def texel_colour(materials, row, column):
    """The front material's albedo at a texel, as linear colour times its base."""
    linear = raster.TEXEL_LINEAR[TEXTURE[row, column]]
    return linear * materials["base_colour"][1]


@pytest.mark.parametrize("kind", [raster.FLAT, raster.TEXTURED, raster.EMISSIVE])
def test_kind_pixels(kind):
    scene, materials = scene_and_materials(kind)
    shaded = raster.rasterize_shaded(scene, materials)
    plain = raster.rasterize(scene)
    x, y = RIGHT
    colour = shaded["colour"][y, x]
    assert shaded["tri"][y, x] in (0, 1)
    if kind == raster.FLAT:
        np.testing.assert_array_equal(colour, plain["colour"][y, x])
    elif kind == raster.TEXTURED:
        np.testing.assert_allclose(colour, texel_colour(materials, 2, 3) * 0.8, 1e-5)
    else:
        np.testing.assert_array_equal(colour, texel_colour(materials, 2, 3))
    # the back square stays flat
    np.testing.assert_array_equal(
        shaded["colour"][OUTSIDE[1], OUTSIDE[0]], plain["colour"][500, 900]
    )


def test_hole_shows_the_layer_behind():
    scene, materials = scene_and_materials(raster.MASKED)
    shaded = raster.rasterize_shaded(scene, materials)
    plain = raster.rasterize(scene)
    x, y = RIGHT
    assert shaded["tri"][y, x] in (0, 1)
    np.testing.assert_allclose(
        shaded["colour"][y, x], texel_colour(materials, 2, 3) * 0.8, 1e-5
    )
    # a hole: the back square's flat colour, depth and triangle
    x, y = LEFT
    assert shaded["tri"][y, x] in (2, 3) and plain["tri"][y, x] in (0, 1)
    np.testing.assert_allclose(shaded["colour"][y, x], [0.7, 0.1, 0.3], rtol=1e-5)
    assert shaded["depth"][y, x] == pytest.approx(0.25)
    # the masked layer alone holds the front square; the other the back one
    assert (shaded["masked"]["tri"][y, x], shaded["opaque"]["tri"][y, x]) in (
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    )
    # without the mask texture the albedo's alpha (opaque) decides: no hole
    materials["textures"][1, raster.MASK] = -1
    shaded = raster.rasterize_shaded(scene, materials)
    assert shaded["tri"][y, x] in (0, 1)


def test_lit_pixel():
    scene, materials = scene_and_materials(raster.LIT)
    x, y = RIGHT
    colour = raster.rasterize_shaded(scene, materials)["colour"][y, x]
    # in float64: the point on the square, light and eye, the normal toward the eye
    z = 2.0
    point = np.array([(x + 0.5 - 512) / 512 * z, (512 - y - 0.5) / 512 * z, z])
    to_light = scene["light"] - point
    to_light /= np.linalg.norm(to_light)
    to_eye = -point / np.linalg.norm(point)
    half = (to_light + to_eye) / np.linalg.norm(to_light + to_eye)
    normal = np.array([0.0, 0.0, -1.0])
    albedo = texel_colour(materials, 2, 3)
    gloss = max(normal @ half, 0) ** 8
    expected = albedo * (0.25 + 0.75 * max(normal @ to_light, 0)) + 0.3 * gloss
    np.testing.assert_allclose(colour, expected, rtol=1e-5)
    # a normal map leaning toward +u (+x) turns the normal: the light (above, at x
    # = 0) is seen at a smaller angle on the right half
    scene, materials = scene_and_materials(raster.LIT, normal=True)
    bent = raster.rasterize_shaded(scene, materials)["colour"][y, x]
    assert (bent != colour).any()


def test_mip_level():
    texels = np.array([1.0, 1.9, 2.1, 7.9, 8.1, 1e9], dtype=np.float32)
    area2 = np.ones(6, dtype=np.float32)
    levels = np.array([11, 11, 11, 11, 11, 3], dtype=np.int32)
    # one level per factor of 4 in area, rounded at 2 x 4^k; clamped to the chain
    assert list(raster.mip_level(texels, area2, levels)) == [0, 0, 1, 1, 2, 2]


needs_cxx = pytest.mark.skipif(native.find_cxx() is None, reason="no C++ compiler")


@needs_cxx
@pytest.mark.parametrize("kind", [raster.MASKED, raster.LIT])
def test_numpy_and_cpp_agree(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("KR_BUILD_DIR", str(tmp_path / "build"))
    scene, materials = scene_and_materials(kind, raster.TEXTURED, normal=True)
    # a near distance that cuts the back square (clipped pieces carry coordinates),
    # which now crosses the front square
    scene["positions"][:4, 2] = 3.0
    scene["positions"][4:, 2] = [2.0, 5.0, 5.0, 2.0]
    scene["camera"] = CAMERA.copy()
    scene["camera"][15] = 2.5
    folder, matter = tmp_path / "scene", tmp_path / "materials"
    scenes.export_npy(scene, folder)
    public.export_materials(materials, matter)
    np.save(matter / "texel_linear.npy", raster.TEXEL_LINEAR)
    out = tmp_path / "out"
    out.mkdir()
    exe = native.build(str(harness.HERE / "cpu" / "raster_cpu.cpp"), "raster_cpu")
    command = [exe, "--scene", folder, "--materials", matter, "--out", out]
    command += ["--threads", "3", "--reps", "1"]
    subprocess.run([str(c) for c in command], check=True, capture_output=True)
    expected = raster.rasterize_shaded(scene, materials)
    for key in ("colour", "depth", "tri", "tile_counts"):
        np.testing.assert_array_equal(np.load(out / f"{key}.npy"), expected[key])
    layers = ("opaque", "masked") if kind == raster.MASKED else ("opaque",)
    for layer in layers:
        assert set(np.unique(expected[layer]["tri"])) > {-1}
