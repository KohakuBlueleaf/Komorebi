"""The lights of a scene for `raster.rasterize_lighted`: sun and sky from an HDR map,
sky visibility baked per vertex, a sun shadow map, point lights at the lamps. Scene
coordinates, y up; the map is equirectangular, column x at azimuth atan2(-z, -x) =
2 pi ((x + 0.5) / W - 0.5), row 0 straight up. `prepare` writes, `load` reads."""

import concurrent.futures
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from reference import bvh, public, raster, raytrace, rng, scenes

F = np.float32
# Sun: the cone (degrees) around the brightest texel taken as the sun, and the outer
# edge of the ring around it whose mean radiance replaces it in the sky.
SUN_CONE_DEG = 4.0
SUN_RING_DEG = 8.0
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
# Spherical harmonics: the basis constants (bands 0 to 2) and the irradiance
# constants of Ramamoorthi and Hanrahan (2001).
SH_Y = (0.282095, 0.488603, 1.092548, 0.315392, 0.546274)
C1, C2, C3, C4, C5 = (F(c) for c in (0.429043, 0.511664, 0.743125, 0.886227, 0.247708))
# Visibility bake: rays per vertex and the hash seed of their directions.
VIS_RAYS = 64
VIS_SEED = 11
VIS_CHUNK = raytrace.CHUNK // VIS_RAYS
# Shadow map: its default side (a `lighting` entry's `shadow_side`), FILL's frame (a
# side beyond it is rendered in quadrants of this side) and the sun camera's distance
# in scene radii.
SHADOW_SIDE = 2048
FILL_SIDE = scenes.WIDTH
SUN_DISTANCE = 1000.0
OCTA_SIDE = 2048  # the background's octahedral map, side in texels (`sky_side`)
# Indirect light: the bounce passes summed (a `lighting` entry's `bounces`), at most.
BOUNCES = 2
MOST_BOUNCES = 3
_BAKE = {}


def read_hdr(path):
    """The radiance (H, W, 3) float32 of a Radiance .hdr file (RGBE, flat or with
    the run-length rows of the format), rows from the top."""
    data = Path(path).read_bytes()
    end = data.index(b"\n\n") + 2
    line_end = data.index(b"\n", end)
    words = data[end:line_end].split()
    if words[0] != b"-Y" or words[2] != b"+X":
        raise ValueError(f"{path}: orientation {words} is not -Y H +X W")
    height, width = int(words[1]), int(words[3])
    raw, pos = np.frombuffer(data, dtype=np.uint8), line_end + 1
    rgbe = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        head = raw[pos : pos + 4]
        if not (
            head[0] == 2
            and head[1] == 2
            and (int(head[2]) << 8 | int(head[3])) == width
        ):
            rgbe[y:] = raw[pos : pos + (height - y) * width * 4].reshape(-1, width, 4)
            break
        pos += 4
        for c in range(4):
            x = 0
            while x < width:
                count = int(raw[pos])
                if count > 128:
                    rgbe[y, x : x + count - 128, c] = raw[pos + 1]
                    x, pos = x + count - 128, pos + 2
                else:
                    rgbe[y, x : x + count, c] = raw[pos + 1 : pos + 1 + count]
                    x, pos = x + count, pos + 1 + count
    exponent = rgbe[:, :, 3].astype(np.int32)
    scale = np.where(exponent > 0, np.ldexp(1.0, exponent - 136), 0.0)
    return (rgbe[:, :, :3] * scale[:, :, None]).astype(np.float32)


def turned(d, turn_deg):
    """Directions (..., 3) turned about the y axis by turn_deg degrees (float64)."""
    a = np.radians(turn_deg)
    x, z = d[..., 0], d[..., 2]
    return np.stack(
        [np.cos(a) * x + np.sin(a) * z, d[..., 1], -np.sin(a) * x + np.cos(a) * z], -1
    )


def texel_directions(height, width):
    """Unit directions (H, W, 3) of the map's texel centres and their solid angles
    (H, W), float64."""
    azimuth = 2 * np.pi * ((np.arange(width) + 0.5) / width - 0.5)
    elevation = np.pi * (0.5 - (np.arange(height) + 0.5) / height)
    a, e = np.meshgrid(azimuth, elevation)
    d = np.stack([-np.cos(a) * np.cos(e), np.sin(e), -np.sin(a) * np.cos(e)], -1)
    solid = (2 * np.pi / width) * (np.pi / height) * np.cos(e)
    return d, solid


def map_texel(d, height, width):
    """Row and column of the texel nearest to world directions d (n, 3) float32."""
    a = np.arctan2(-d[:, 2], -d[:, 0])
    e = np.arctan2(d[:, 1], np.sqrt(d[:, 0] * d[:, 0] + d[:, 2] * d[:, 2]))
    x = np.floor((a / F(2 * np.pi) + F(0.5)) * F(width)).astype(np.int64) % width
    y = np.floor((F(0.5) - e / F(np.pi)) * F(height)).astype(np.int64)
    return np.clip(y, 0, height - 1), x


def octahedral_texel(d, side):
    """Row and column of the texel of an octahedral map (side x side, y up) nearest
    to directions d (n, 3) float32: (u, v) = (x, z) / (|x| + |y| + |z|), folded for
    y < 0 to ((1 - |v|) sign u, (1 - |u|) sign v); column (u + 1) / 2 side, row
    (v + 1) / 2 side, rounded down (sign 0 is +)."""
    norm = (np.abs(d[:, 0]) + np.abs(d[:, 1])) + np.abs(d[:, 2])
    u, v = d[:, 0] / norm, d[:, 2] / norm
    su = np.where(u < 0, F(-1), F(1))
    sv = np.where(v < 0, F(-1), F(1))
    below = d[:, 1] < 0
    u, v = (
        np.where(below, (F(1) - np.abs(v)) * su, u),
        np.where(below, (F(1) - np.abs(u)) * sv, v),
    )
    column = np.floor((u + F(1)) * F(0.5) * F(side)).astype(np.int64)
    row = np.floor((v + F(1)) * F(0.5) * F(side)).astype(np.int64)
    return np.clip(row, 0, side - 1), np.clip(column, 0, side - 1)


def octahedral_direction(u, v):
    """Unit directions (..., 3) float64 of octahedral coordinates u, v in [-1, 1]:
    y = 1 - |u| - |v|; for y < 0, (u, v) unfolded to ((1 - |v|) sign u, (1 - |u|)
    sign v); then (u, y, v) normalized."""
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    y = 1 - np.abs(u) - np.abs(v)
    su, sv = np.where(u < 0, -1.0, 1.0), np.where(v < 0, -1.0, 1.0)
    u, v = (
        np.where(y < 0, (1 - np.abs(v)) * su, u),
        np.where(y < 0, (1 - np.abs(u)) * sv, v),
    )
    d = np.stack([u, y, v], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def octahedral_map(radiance, turn, side=OCTA_SIDE):
    """The map's radiance (side, side, 3) float32 resampled to an octahedral map of
    world directions (texel centres; nearest texel of the map turned by turn)."""
    c = (np.arange(side) + 0.5) / side * 2 - 1
    v, u = np.meshgrid(c, c, indexing="ij")
    d = turned(octahedral_direction(u, v).reshape(-1, 3), -turn).astype(np.float32)
    row, column = map_texel(d, radiance.shape[0], radiance.shape[1])
    return radiance[row, column].reshape(side, side, 3).astype(np.float32)


def sh_basis(d):
    """The 9 real spherical harmonics (..., 9) at unit directions d (..., 3): L00,
    L1-1 (y), L10 (z), L11 (x), L2-2 (xy), L2-1 (yz), L20, L21 (xz), L22."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    a, b, c, e, g = SH_Y
    return np.stack(
        [a + 0 * x, b * y, b * z, b * x, c * x * y, c * y * z, e * (3 * z * z - 1)]
        + [c * x * z, g * (x * x - y * y)],
        -1,
    )


def sh_turned(sh, turn_deg):
    """Coefficients sh (9, C) of a sky turned about y by turn_deg degrees (as `turned`):
    M sh, M (9 x 9) the least-squares fit of the basis at turned directions to the basis
    (exact: each band is closed under rotation), float64 on 128 texel directions."""
    d = texel_directions(8, 16)[0].reshape(-1, 3)
    m = np.linalg.lstsq(sh_basis(d), sh_basis(turned(d, turn_deg)), rcond=None)[0]
    return (m.T @ np.asarray(sh, np.float64)).astype(np.float32)


def sky_irradiance(sh, n):
    """Irradiance (n, 3) float32 of the sky with coefficients sh (9, 3) on unit
    normals n (n, 3): the quadratic form of Ramamoorthi and Hanrahan, in this order."""
    s = sh.astype(np.float32)
    x, y, z = n[:, 0:1], n[:, 1:2], n[:, 2:3]
    e = C4 * s[0] - C5 * s[6]
    e = e + C3 * s[6] * (z * z)
    e = e + C1 * s[8] * (x * x - y * y)
    e = e + F(2) * C1 * ((s[4] * (x * y) + s[7] * (x * z)) + s[5] * (y * z))
    e = e + F(2) * C2 * ((s[3] * x + s[1] * y) + s[2] * z)
    return e


def environment(path, turn):
    """The sun and the sky of a map turned by `turn` degrees about y: the sun's unit
    direction and colour (luminance 1), the sky's coefficients sh (9, 3) with the
    sun's cone replaced by the mean of the ring around it, and the map itself."""
    radiance = read_hdr(path).astype(np.float64)
    height, width = radiance.shape[:2]
    d, solid = texel_directions(height, width)
    d = turned(d, turn)
    lum = radiance @ LUMINANCE
    peak = d[np.unravel_index(np.argmax(lum), lum.shape)]
    angle = np.degrees(np.arccos(np.clip(d @ peak, -1.0, 1.0)))
    cone, ring = angle < SUN_CONE_DEG, (angle >= SUN_CONE_DEG) & (angle < SUN_RING_DEG)
    around = (radiance[ring] * solid[ring, None]).sum(0) / solid[ring].sum()
    excess = np.maximum(radiance[cone] - around, 0.0) * solid[cone, None]
    weight = excess @ LUMINANCE
    direction = (d[cone] * weight[:, None]).sum(0)
    direction /= np.linalg.norm(direction)
    colour = excess.sum(0) / (excess.sum(0) @ LUMINANCE)
    sky = radiance.copy()
    sky[cone] = around
    sh = np.einsum("hwk,hwc,hw->kc", sh_basis(d), sky, solid)
    return {
        "sun_direction": direction.astype(np.float32),
        "sun_colour": colour.astype(np.float32),
        "sh": sh.astype(np.float32),
        "environment": radiance.astype(np.float32),
        "sun_share": float(weight.sum() / ((lum * solid).sum())),
    }


def vertex_normals(positions, indices):
    """Unit area-weighted normals (V, 3) float32 of the vertices (zero where the
    triangles around a vertex cancel)."""
    p = positions[indices].astype(np.float64)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    total = np.zeros((len(positions), 3))
    for k in range(3):
        np.add.at(total, indices[:, k], n)
    length = np.linalg.norm(total, axis=1)
    return np.where(
        (length > 0)[:, None], total / np.maximum(length, 1e-300)[:, None], 0
    ).astype(np.float32)


def bake_directions(vertex, normal):
    """VIS_RAYS cosine-weighted unit directions (n, VIS_RAYS, 3) about the normals
    (n, 3), from the hash of vertex * VIS_RAYS + ray."""
    index = vertex[:, None].astype(np.int64) * VIS_RAYS + np.arange(VIS_RAYS)
    u1 = rng.uniform(index, 0, VIS_SEED)
    u2 = rng.uniform(index, 1, VIS_SEED)
    r, phi = np.sqrt(u1), F(2 * np.pi) * u2
    helper = np.where(
        (np.abs(normal[:, 0]) < F(0.9))[:, None], F([1, 0, 0]), F([0, 1, 0])
    )
    t = raster.normalized(raster.cross3(helper, normal))
    b = raster.cross3(normal, t)
    up = np.sqrt(np.maximum(F(1) - u1, F(0)))
    return (
        t[:, None] * (r * np.cos(phi))[..., None]
        + b[:, None] * (r * np.sin(phi))[..., None]
    ) + normal[:, None] * up[..., None]


def _bake_chunk(span):
    """Visibility of vertices start..end (the arrays in _BAKE)."""
    start, end = span
    positions, normals, tree, eps = (
        _BAKE[k] for k in ("positions", "normals", "tree", "eps")
    )
    vertex = np.arange(start, end)
    n = normals[start:end]
    d = bake_directions(vertex, n).reshape(-1, 3).astype(np.float32)
    o = np.repeat(positions[start:end] + n * eps, VIS_RAYS, axis=0)
    hit = raytrace.trace(tree, o, d, raytrace.TMAX, any_hit=True)["leaf"] >= 0
    vis = F(1) - hit.reshape(-1, VIS_RAYS).sum(1).astype(np.float32) / F(VIS_RAYS)
    return np.where(raster.dot3(n, n) > 0, vis, F(1))


def bake_visibility(positions, indices, tree, eps, workers=4):
    """Per vertex the share of VIS_RAYS cosine-weighted rays from the vertex (moved
    eps along its normal) that leave the scene without a hit; 1 where the normal is
    zero. Traced with the reference ray tracer in `workers` processes."""
    _BAKE.update(
        positions=positions,
        normals=vertex_normals(positions, indices),
        tree=tree,
        eps=F(eps),
    )
    spans = [
        (s, min(s + VIS_CHUNK, len(positions)))
        for s in range(0, len(positions), VIS_CHUNK)
    ]
    if workers > 1:
        context = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(workers, context) as pool:
            parts = list(pool.map(_bake_chunk, spans))
    else:
        parts = [_bake_chunk(span) for span in spans]
    _BAKE.clear()
    return np.concatenate(parts).astype(np.float32)


def emission_colours(materials):
    """Per material its emission colour (M, 3): the base colour times the linear mean
    of its albedo texture (the chain's last, 1 x 1 level) when it has one, scaled to
    a largest channel of 1 (the hue; a lamp's brightness is its intensity)."""
    colour = materials["base_colour"].astype(np.float32).copy()
    for m, tex in enumerate(materials["textures"][:, raster.ALBEDO]):
        if tex >= 0:
            last = materials["levels"][tex] - 1
            start = materials["mip"][tex, last, 0]
            colour[m] *= raster.TEXEL_LINEAR[materials["texels"][start : start + 3]]
    top = colour.max(axis=1, keepdims=True)
    return np.where(top > 0, colour / np.where(top > 0, top, F(1)), F(1))


def find_lamps(scene, materials, merge_distance, lamp_scale):
    """Point lights at the lamps: emissive triangles connected through shared
    vertices form a group, groups whose centroids are nearer than merge_distance are
    merged (transitively); a lamp sits at its area-weighted centroid with the
    area-weighted emission colour and intensity lamp_scale x its area."""
    kind = materials["kind"][materials["material"]]
    tris = np.nonzero(kind == raster.EMISSIVE)[0]
    counts = {"emissive_triangles": len(tris), "groups": 0, "lamps": 0}
    empty = np.zeros((0, 3), dtype=np.float32)
    if len(tris) == 0:
        return empty, empty, np.zeros(0, dtype=np.float32), counts
    idx = scene["indices"][tris]
    label = np.arange(len(tris))
    while True:
        vertex = np.full(len(scene["positions"]), len(tris))
        for k in range(3):
            np.minimum.at(vertex, idx[:, k], label)
        new = vertex[idx].min(1)
        if np.array_equal(new, label):
            break
        label = new
    group = np.unique(label, return_inverse=True)[1]
    p = scene["positions"][idx].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    centre = p.mean(1)

    def weighted(values, by):
        total = np.zeros((by.max() + 1, values.shape[1]))
        np.add.at(total, by, values * area[:, None])
        mass = np.bincount(by, area, by.max() + 1)
        return total / np.maximum(mass, 1e-300)[:, None], mass

    centroid, _ = weighted(centre, group)
    counts["groups"] = len(centroid)
    merged = np.arange(len(centroid))
    for i in range(len(centroid)):
        near = np.nonzero(
            np.linalg.norm(centroid - centroid[i], axis=1) < merge_distance
        )[0]
        roots = np.unique(merged[near])
        merged[np.isin(merged, roots)] = roots.min()
    lamp = np.unique(merged, return_inverse=True)[1][group]
    position, mass = weighted(centre, lamp)
    colour, _ = weighted(emission_colours(materials)[materials["material"][tris]], lamp)
    counts["lamps"] = len(position)
    return (
        position.astype(np.float32),
        colour.astype(np.float32),
        (lamp_scale * mass).astype(np.float32),
        counts,
    )


def sun_camera(positions, sun, side=SHADOW_SIDE):
    """A camera (16,) float32 far along the sun's direction (SUN_DISTANCE radii of
    the scene's box) looking at its centre, whose square frame of side x side covers
    the box. Its axes (right = helper x forward) stay as the shadow maps were made with
    them, not the view cameras' forward x y."""
    lo, hi = positions.min(0).astype(np.float64), positions.max(0).astype(np.float64)
    centre, radius = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2
    distance = SUN_DISTANCE * radius
    fwd = -np.asarray(sun, np.float64)
    helper = np.array([0.0, 1.0, 0.0]) if abs(fwd[1]) < 0.99 else np.array([1, 0, 0])
    right = np.cross(helper, fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    box = np.stack(np.meshgrid(*zip(lo, hi), indexing="ij"), -1).reshape(-1, 3)
    half = np.abs(np.stack([(box - centre) @ right, (box - centre) @ up])).max()
    focal = (side / 2) * (distance - radius) / half
    eye = centre - fwd * distance
    tail = [focal, side / 2, side / 2, distance - 2 * radius]
    return np.concatenate([eye, right, up, fwd, tail]).astype(np.float32)


def quadrant_camera(camera, i, j):
    """The camera (16,) float32 of quadrant (i, j) of a view `camera` whose side is a
    multiple of FILL_SIDE: the principal point moved by (-FILL_SIDE i, -FILL_SIDE j),
    so its pixel (x, y) is the view's texel (x + FILL_SIDE i, y + FILL_SIDE j)."""
    out = np.array(camera, dtype=np.float32)
    out[13] = out[13] - F(FILL_SIDE * i)
    out[14] = out[14] - F(FILL_SIDE * j)
    return out


def shadow_map(scene, sun, materials=None, side=SHADOW_SIDE):
    """The 1/z (side, side) of the scene from the sun's camera, the camera, and a
    shadow texel's world size at the scene's centre. With materials, the two layers
    and holes of `rasterize_shaded` (leaves cast their own shadow); else one FILL. A
    side beyond FILL_SIDE: (side / FILL_SIDE)^2 renders, one a quadrant_camera."""
    if side % FILL_SIDE:
        raise ValueError(f"a shadow map's side {side} is not a multiple of {FILL_SIDE}")
    camera = sun_camera(scene["positions"], sun, side)
    depth = np.empty((side, side), dtype=np.float32)
    for j in range(side // FILL_SIDE):
        for i in range(side // FILL_SIDE):
            view = {
                "positions": scene["positions"],
                "indices": scene["indices"],
                "colors": np.zeros((len(scene["positions"]), 3), dtype=np.float32),
                "camera": quadrant_camera(camera, i, j),
                "light": np.zeros(3, dtype=np.float32),
            }
            if materials is None:
                part = raster.rasterize(view)["depth"]
            else:
                part = raster.rasterize_shaded(view, materials)["depth"]
            rows, columns = (slice(FILL_SIDE * k, FILL_SIDE * (k + 1)) for k in (j, i))
            depth[rows, columns] = part
    distance = np.linalg.norm(
        camera[0:3].astype(np.float64)
        - (scene["positions"].min(0) + scene["positions"].max(0)) / 2
    )
    return depth, camera, F(distance / camera[12])


def sun_shadow(lights, points, normals, pcf=True):
    """Sun light (n,) float32 in [0, 1] at points (n, 3) on surfaces with normals
    (n, 3). The point moved one shadow texel along its normal is in shadow at a texel
    when its 1/z in the sun's camera is below the map's by more than the bias; outside
    the map, lit. pcf: the four nearest texels, bilinear weights; else the nearest."""
    cam, depth = lights["shadow_camera"], lights["shadow_map"]
    q = points + normals * lights["shadow_texel"]
    d = q - cam[0:3]
    xc = (d[:, 0] * cam[3] + d[:, 1] * cam[4]) + d[:, 2] * cam[5]
    yc = (d[:, 0] * cam[6] + d[:, 1] * cam[7]) + d[:, 2] * cam[8]
    zc = (d[:, 0] * cam[9] + d[:, 1] * cam[10]) + d[:, 2] * cam[11]
    sx, sy, w = raster.to_screen(cam, xc, yc, zc)
    side = depth.shape[0]

    def lit_at(ix, iy):
        inside = (ix >= 0) & (ix < side) & (iy >= 0) & (iy < side)
        stored = depth[np.clip(iy, 0, side - 1), np.clip(ix, 0, side - 1)]
        return np.where(inside & (w < stored - lights["shadow_bias"]), F(0), F(1))

    if not pcf:
        return lit_at(np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64))
    gx, gy = sx - F(0.5), sy - F(0.5)
    x0, y0 = np.floor(gx), np.floor(gy)
    fx, fy = gx - x0, gy - y0
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    top = lit_at(x0, y0) * (F(1) - fx) + lit_at(x0 + 1, y0) * fx
    bottom = lit_at(x0, y0 + 1) * (F(1) - fx) + lit_at(x0 + 1, y0 + 1) * fx
    return top * (F(1) - fy) + bottom * fy


def point_irradiance(intensity, size, radius, position, points, normals):
    """Irradiance (n,) float32 of one point light of intensity I at position:
    I max(0, n.l) / (d^2 + s^2) (1 - (d / r)^4)^2 inside r, exactly 0 from r on."""
    to = position[None, :] - points
    d2 = raster.dot3(to, to)
    d = np.sqrt(d2)
    to_light = to / np.where(d > 0, d, F(1))[:, None]
    q = (d / F(radius)) * (d / F(radius))
    fall = (F(1) - q * q) * (F(1) - q * q)
    e = (
        F(intensity)
        * np.maximum(raster.dot3(normals, to_light), F(0))
        / (d2 + F(size) ** 2)
    )
    return np.where(d < F(radius), e * fall, F(0)), to_light


def lighting_path(name, data):
    """<public>/<scene>_lighting.npz."""
    return public.public_dir(data) / f"{name}_lighting.npz"


def scene_tree(name, data, mesh):
    """The ray tracer's tree of a scene: a view's saved tree when there is one with
    the scene's triangle count, else built here (median split)."""
    for view in public.table()[name]["views"]:
        path = Path(data) / f"{name}_{view}_tree.npz"
        if path.exists():
            tree = bvh.load_tree(path)
            if len(tree["leaf_tri"]) == len(mesh["indices"]):
                return tree
    return bvh.build(mesh, split="median")


def scene_lamps(entry, scene, materials):
    """The point lights of a `lighting` entry: its authored `lights` (position,
    colour), each of intensity `lamp_scale`, when it has them; else `find_lamps`
    (the emissive groups). Positions, colours, intensities and counts."""
    if "lights" not in entry:
        return find_lamps(
            scene, materials, entry["merge_distance"], entry["lamp_scale"]
        )
    lights = entry["lights"]
    position = np.array([light["position"] for light in lights], dtype=np.float32)
    colour = np.array([light["colour"] for light in lights], dtype=np.float32)
    intensity = np.full(len(lights), entry["lamp_scale"], dtype=np.float32)
    return position, colour, intensity, {"authored_lights": len(lights)}


def scene_sun(entry, env_path):
    """The `environment` of a `lighting` entry and its turn (degrees). With an
    authored `sun` (direction it shines along, colour), the map is turned so its
    sun's azimuth is the authored one, and the sun is the authored one."""
    if "sun" not in entry:
        return environment(env_path, entry["turn"]), float(entry["turn"])
    own = environment(env_path, 0.0)["sun_direction"].astype(np.float64)
    to_sun = -np.asarray(entry["sun"]["direction"], dtype=np.float64)
    to_sun /= np.linalg.norm(to_sun)
    turn = float(
        np.degrees(np.arctan2(own[2], own[0]) - np.arctan2(to_sun[2], to_sun[0]))
    )
    env = environment(env_path, turn)
    colour = np.asarray(entry["sun"]["colour"], dtype=np.float64)
    env["sun_direction"] = to_sun.astype(np.float32)
    env["sun_colour"] = (colour / (colour @ LUMINANCE)).astype(np.float32)
    return env, turn


def prepare(name, data, workers=4, rebake=False, shadow_side=None):
    """Fetches the scene's environment map, bakes visibility (kept in
    <scene>_visibility.npz; rebake to trace again), lamps and the shadow map (its side
    the entry's `shadow_side`, or `shadow_side` when given), and writes
    <scene>_lighting.npz. Returns its path and counts."""
    entry = public.table()[name]["lighting"]
    (env_path,) = public.fetch_environment(entry["environment"], data)
    mesh, _ = public.load_mesh(name, data)
    materials, _ = public.load_materials(name, data)
    counts = {"vertices": len(mesh["positions"])}
    vis_path = public.public_dir(data) / f"{name}_visibility.npz"
    if rebake or not vis_path.exists():
        tree = scene_tree(name, data, mesh)
        p = mesh["positions"].astype(np.float64)
        eps = public.EPS_SCALE * float(np.linalg.norm(p.max(0) - p.min(0)))
        started = time.perf_counter()
        visibility = bake_visibility(
            mesh["positions"], mesh["indices"], tree, eps, workers
        )
        bake = {
            "bake_seconds": time.perf_counter() - started,
            "rays": len(visibility) * VIS_RAYS,
            "workers": workers,
            "eps": eps,
        }
        np.savez(vis_path, visibility=visibility, counts=json.dumps(bake))
    with np.load(vis_path) as f:
        visibility, bake = f["visibility"], json.loads(str(f["counts"]))
    counts |= bake
    env, turn = scene_sun(entry, env_path)
    up = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    horizontal = sky_irradiance(env["sh"], up)[0]
    sun_irradiance = (
        F(entry["sun_to_sky"]) * (horizontal @ LUMINANCE.astype(np.float32))
    ) * env["sun_colour"]
    started = time.perf_counter()
    if shadow_side is None:
        shadow_side = entry.get("shadow_side", SHADOW_SIDE)
    counts["shadow_side"] = shadow_side = int(shadow_side)
    depth, camera, texel = shadow_map(
        mesh, env["sun_direction"], materials, shadow_side
    )
    counts["shadow_seconds"] = time.perf_counter() - started
    position, colour, intensity, lamps = scene_lamps(entry, mesh, materials)
    counts |= lamps
    counts["sun_share"], counts["turn"] = env["sun_share"], turn
    counts["sky_side"] = side = int(entry.get("sky_side", OCTA_SIDE))
    out = lighting_path(name, data)
    np.savez(
        out,
        visibility=visibility,
        sh=env["sh"],
        sun_direction=env["sun_direction"],
        sun_irradiance=sun_irradiance.astype(np.float32),
        octahedral=octahedral_map(env["environment"], 0.0, side),
        turn=np.float32(turn),
        shadow_map=depth,
        shadow_side=np.int32(shadow_side),
        shadow_camera=camera,
        shadow_texel=texel,
        lamp_position=position,
        lamp_colour=colour,
        lamp_intensity=intensity,
        counts=json.dumps(counts),
    )
    counts["bytes"] = os.path.getsize(out)
    return out, counts


def sun_angles(direction):
    """Azimuth and elevation (degrees) of a direction toward the sun: azimuth 0 at +x,
    90 at -z (a right-handed turn about +y, the sense of `turned`), elevation above
    the horizontal plane (y = 0)."""
    d = np.asarray(direction, np.float64)
    elevation = np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))
    return float(np.degrees(np.arctan2(-d[2], d[0]))), float(elevation)


def sun_direction(azimuth, elevation):
    """The unit direction (3,) float32 toward a sun at azimuth, elevation (degrees,
    as `sun_angles`)."""
    a, e = np.radians(azimuth), np.radians(elevation)
    return np.array(
        [np.cos(e) * np.cos(a), np.sin(e), -np.cos(e) * np.sin(a)], dtype=np.float32
    )


def moved(lights, scene, materials, azimuth, elevation):
    """The prepared lights (`load`) with the sun at azimuth, elevation (degrees): the
    map turned so its own sun has that azimuth (turn, sky by `sh_turned`), the shadow
    map rebuilt at its side (`shadow_side`); visibility, colour and irradiance kept.
    Elevation <= 0: irradiance 0 and a 1 x 1 shadow map of 0 (all lit). README "Moving
    the sun"."""
    own_azimuth, _ = sun_angles(lights["sun_direction"])
    change = float(azimuth) - own_azimuth
    sun = sun_direction(azimuth, elevation)
    out = dict(lights)
    out["turn"] = np.float32(float(lights["turn"]) + change)
    out["sh"] = sh_turned(lights["sh"], change)
    out["sun_direction"] = sun
    # The bias in shadow texels at the camera's distance, as `load` made it.
    distance = float(lights["shadow_texel"] * lights["shadow_camera"][12])
    texels = float(lights["shadow_bias"]) * distance**2 / float(lights["shadow_texel"])
    side = shadow_side(lights)
    out["shadow_side"] = np.int32(side)
    if elevation <= 0:
        camera = sun_camera(scene["positions"], sun, side)
        out["sun_irradiance"] = np.zeros(3, dtype=np.float32)
        out["shadow_map"] = np.zeros((1, 1), dtype=np.float32)
        out["shadow_camera"], out["shadow_texel"] = camera, lights["shadow_texel"]
        return out
    depth, camera, texel = shadow_map(scene, sun, materials, side)
    out["shadow_map"], out["shadow_camera"], out["shadow_texel"] = depth, camera, texel
    distance = float(texel * camera[12])
    out["shadow_bias"] = F(texels * float(texel) / distance**2)
    return out


def shadow_side(lights):
    """The shadow map's side of prepared lights: their `shadow_side`, else (lights of
    an older file, or made by hand) the map's."""
    if "shadow_side" in lights:
        return int(lights["shadow_side"])
    return int(lights["shadow_map"].shape[0])


LIGHT_KEYS = (
    "visibility",
    "sh",
    "sun_direction",
    "sun_irradiance",
    "octahedral",
    "turn",
    "shadow_map",
    "shadow_camera",
    "shadow_texel",
    "lamp_position",
    "lamp_colour",
    "lamp_intensity",
)


def bounce_count(value):
    """A number of bounce passes: a whole number 0 to MOST_BOUNCES; else ValueError
    with the reason."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"bounces is a whole number 0 to {MOST_BOUNCES}, not {value!r}"
        )
    if not 0 <= int(value) <= MOST_BOUNCES:
        raise ValueError(f"bounces is 0 to {MOST_BOUNCES}, not {int(value)}")
    return int(value)


def load(name, data):
    """The lights of a scene (or of a view's scene): the arrays of LIGHT_KEYS, the
    shadow map's side (`shadow_side`), the table's numbers (`reflection`: 0 without it;
    the tone step's `saturation` and `curve`: 1 and 0 without them; `bounces`: 2), the
    shadow bias as 1/z (shadow_bias texels at the sun camera's distance) and `indirect`,
    the path of <scene>_indirect.npz when there is one. Missing file: error."""
    if public.is_view(name):
        name = public.views()[name][0]
    path = lighting_path(name, data)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; write it with: python -m reference.public lighting "
            f"{name}"
        )
    with np.load(path) as f:
        lights = {k: f[k] for k in LIGHT_KEYS}
        lights["shadow_side"] = np.int32(
            f["shadow_side"] if "shadow_side" in f.files else len(f["shadow_map"])
        )
    entry = public.table()[name]["lighting"]
    distance = float(lights["shadow_texel"] * lights["shadow_camera"][12])
    bias = entry["shadow_bias"] * float(lights["shadow_texel"]) / distance**2
    numbers = {
        k: entry[k] for k in ("ambient", "emissive_scale", "exposure", "radius", "size")
    }
    numbers["reflection"] = entry.get("reflection", 0.0)
    numbers["saturation"] = entry.get("saturation", 1.0)
    numbers["curve"] = int(entry.get("curve", raster.CURVE_PLAIN))
    numbers["bounces"] = bounce_count(entry.get("bounces", BOUNCES))
    found = public.public_dir(data) / f"{name}_indirect.npz"
    if found.exists():
        numbers["indirect"] = str(found)
    return lights | numbers | {"shadow_bias": F(bias)}
