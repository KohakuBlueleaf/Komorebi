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
# Shadow map: its side (FILL's frame) and the sun camera's distance in scene radii.
SHADOW_SIDE = scenes.WIDTH
SUN_DISTANCE = 1000.0
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


def sun_camera(positions, sun):
    """A camera (16,) float32 far along the sun's direction (SUN_DISTANCE radii of
    the scene's box) looking at its centre, whose square frame covers the box."""
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
    focal = (SHADOW_SIDE / 2) * (distance - radius) / half
    eye = centre - fwd * distance
    tail = [focal, SHADOW_SIDE / 2, SHADOW_SIDE / 2, distance - 2 * radius]
    return np.concatenate([eye, right, up, fwd, tail]).astype(np.float32)


def shadow_map(scene, sun, materials=None):
    """The 1/z (side, side) of the scene from the sun's camera, the camera, and a
    shadow texel's world size at the scene's centre. With materials, the two layers
    and holes of `rasterize_shaded` (leaves cast their own shadow); else one FILL."""
    camera = sun_camera(scene["positions"], sun)
    view = {
        "positions": scene["positions"],
        "indices": scene["indices"],
        "colors": np.zeros((len(scene["positions"]), 3), dtype=np.float32),
        "camera": camera,
        "light": np.zeros(3, dtype=np.float32),
    }
    if materials is None:
        depth = raster.rasterize(view)["depth"]
    else:
        depth = raster.rasterize_shaded(view, materials)["depth"]
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


def prepare(name, data, workers=4, rebake=False):
    """Fetches the scene's environment map, bakes visibility (kept in
    <scene>_visibility.npz; rebake to trace again), lamps and the shadow map, and
    writes <scene>_lighting.npz. Returns its path and counts."""
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
    env = environment(env_path, entry["turn"])
    up = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    horizontal = sky_irradiance(env["sh"], up)[0]
    sun_irradiance = (
        F(entry["sun_to_sky"]) * (horizontal @ LUMINANCE.astype(np.float32))
    ) * env["sun_colour"]
    started = time.perf_counter()
    depth, camera, texel = shadow_map(mesh, env["sun_direction"], materials)
    counts["shadow_seconds"] = time.perf_counter() - started
    position, colour, intensity, lamps = find_lamps(
        mesh, materials, entry["merge_distance"], entry["lamp_scale"]
    )
    counts |= lamps
    counts["sun_share"] = env["sun_share"]
    out = lighting_path(name, data)
    np.savez(
        out,
        visibility=visibility,
        sh=env["sh"],
        sun_direction=env["sun_direction"],
        sun_irradiance=sun_irradiance.astype(np.float32),
        environment=env["environment"],
        turn=np.float32(entry["turn"]),
        shadow_map=depth,
        shadow_camera=camera,
        shadow_texel=texel,
        lamp_position=position,
        lamp_colour=colour,
        lamp_intensity=intensity,
        counts=json.dumps(counts),
    )
    counts["bytes"] = os.path.getsize(out)
    return out, counts


LIGHT_KEYS = (
    "visibility",
    "sh",
    "sun_direction",
    "sun_irradiance",
    "environment",
    "turn",
    "shadow_map",
    "shadow_camera",
    "shadow_texel",
    "lamp_position",
    "lamp_colour",
    "lamp_intensity",
)


def load(name, data):
    """The lights of a scene (or of a view's scene): the arrays of LIGHT_KEYS and the
    table's per-scene numbers; the shadow bias as 1/z (shadow_bias texels at the
    sun camera's distance). Missing file: an error naming the command."""
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
    entry = public.table()[name]["lighting"]
    distance = float(lights["shadow_texel"] * lights["shadow_camera"][12])
    bias = entry["shadow_bias"] * float(lights["shadow_texel"]) / distance**2
    numbers = {
        k: entry[k] for k in ("ambient", "emissive_scale", "exposure", "radius", "size")
    }
    return lights | numbers | {"shadow_bias": F(bias)}
