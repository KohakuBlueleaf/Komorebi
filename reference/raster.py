"""Rasterization in numpy, float32 in the order written, no fused multiply-add: the
definition the C++ and CUDA versions match (projects/raster/reference/README.md).
Triangles are clipped at the near distance, binned to 32 x 32 tiles by pixel box, filled
by the top-left rule; the larger 1/z wins, a tie goes to the lower triangle index.

SHADE (`rasterize_shaded`) colours each pixel from its winner's material: FILL runs
over the triangles neither `masked` nor `transparent`, over the masked ones (taken
where nearer and not a hole) and over the transparent ones (glass, blended over)."""

import importlib

import numpy as np

from reference import lights as light_data
from reference import scenes
from reference.scenes import HEIGHT, TILE, WIDTH

# The default frame's grid (a scene's `size` entry chooses another: `grid`).
TILES_X = WIDTH // TILE
TILES_Y = HEIGHT // TILE
BACKGROUND = np.array([0.05, 0.05, 0.08], dtype=np.float32)
CHUNK = 128
F = np.float32
# SHADE: the kinds' codes (reference/public.py KINDS), texture slots, the lit kind's
# ambient and diffuse weights, and the alpha byte under which a masked pixel is a hole.
FLAT, TEXTURED, MASKED, LIT, EMISSIVE, TRANSPARENT = range(6)
ALBEDO, MASK, NORMAL, SPECULAR = range(4)
AMBIENT, DIFFUSE = F(0.25), F(0.75)
OPAQUE_BYTE = 128
# LIGHTED: the terms of rasterize_lighted's colour (all by default), and the glass's
# reflectance facing the eye (Schlick's F0).
TERMS = ("sky", "ambient", "sun", "shadow", "lamps", "specular", "background", "glass")
OPTIONAL_TERMS = ("indirect",)  # off by default: bounced light (reference/indirect.py)
GLASS_F0 = F(0.04)
# The tone step (`tone`): the curves by number, the luminance weights of the saturation
# and the filmic curve's fit x (a x + b) / (x (c x + d) + e).
CURVE_PLAIN, CURVE_FILMIC = 0, 1
TONE_LUMA = (F(0.2126), F(0.7152), F(0.0722))
FILMIC = (F(2.51), F(0.03), F(2.43), F(0.59), F(0.14))
# Texel byte to value: albedo as linear colour (the pictures' 2.2 gamma undone), the
# other textures to [0, 1] and normals to [-1, 1].
TEXEL_LINEAR = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)
TEXEL_UNIT = (np.arange(256) / 255.0).astype(np.float32)
TEXEL_SIGNED = (np.arange(256) * 2.0 / 255.0 - 1.0).astype(np.float32)


def grid(scene):
    """The scene's frame grid (TX, TY) (scenes.frame_grid of its size): the frame
    rendered is TILE TX by TILE TY pixels."""
    return scenes.frame_grid(scenes.frame_size(scene))


def picture(image, scene):
    """The picture of a rendered (padded) frame's image: its top-left W by H."""
    width, height = scenes.frame_size(scene)
    return image[:height, :width]


def camera_space(scene):
    """Coordinates x, y, z (V,) float32 of the vertices along right, up, forward."""
    cam = scene["camera"]
    d = scene["positions"] - cam[0:3]
    xc = (d[:, 0] * cam[3] + d[:, 1] * cam[4]) + d[:, 2] * cam[5]
    yc = (d[:, 0] * cam[6] + d[:, 1] * cam[7]) + d[:, 2] * cam[8]
    zc = (d[:, 0] * cam[9] + d[:, 1] * cam[10]) + d[:, 2] * cam[11]
    return xc, yc, zc


def to_screen(cam, xc, yc, zc):
    """Screen x, y and 1/z of camera-space points with z beyond the near distance."""
    inv = F(1.0) / zc
    return cam[13] + cam[12] * (xc * inv), cam[14] - cam[12] * (yc * inv), inv


def clip_polygon(corners, near):
    """The polygon (list of 6-vectors x, y, z, colour) of a triangle cut at z = near.

    corners: (3, 6) float32; a crossing edge adds the point from its vertex beyond.
    """
    out = []
    for k in range(3):
        a, b = corners[k], corners[(k + 1) % 3]
        if a[2] > near:
            out.append(a)
        if (a[2] > near) != (b[2] > near):
            inside, outside = (a, b) if a[2] > near else (b, a)
            t = (near - inside[2]) / (outside[2] - inside[2])
            point = inside + t * (outside - inside)
            point[2] = near
            out.append(point.astype(np.float32))
    return out


def corners(scene, extra=None):
    """Screen x, y, 1/z and colour (R, 3) per corner of every piece, and its triangle.

    Whole triangles keep their vertices; crossing ones are clipped (1 or 2 pieces).
    extra: further values per triangle corner (T, 3, k) float32, cut with the colour
    and returned after its three channels.
    """
    cam, idx = scene["camera"], scene["indices"]
    xc, yc, zc = camera_space(scene)
    beyond = (zc > cam[15])[idx].sum(axis=1)
    whole = np.nonzero(beyond == 3)[0]
    pos = np.stack([xc, yc, zc], axis=1)
    pieces, src = [pos[idx[whole]]], [whole]
    values = scene["colors"][idx]
    if extra is not None:
        values = np.concatenate([values, extra.astype(np.float32)], axis=2)
    cols = [values[whole]]
    for t in np.nonzero((beyond == 1) | (beyond == 2))[0]:
        tri = np.concatenate([pos[idx[t]], values[t]], axis=1)
        poly = clip_polygon(tri, cam[15])
        for fan in ([0, 1, 2], [0, 2, 3])[: len(poly) - 2]:
            piece = np.stack([poly[i] for i in fan])
            pieces.append(piece[None, :, :3])
            cols.append(piece[None, :, 3:])
            src.append([t])
    pos, col, src = np.concatenate(pieces), np.concatenate(cols), np.concatenate(src)
    order = np.argsort(src, kind="stable")
    pos, col, src = pos[order], col[order], src[order]
    x, y, w = to_screen(cam, pos[..., 0], pos[..., 1], pos[..., 2])
    return x, y, w, col, src.astype(np.int64)


def planes(values, coef, inv_area):
    """The plane of per-vertex values (T, 3) over coefficients coef (T, 3)."""
    s = (coef[:, 0] * values[:, 0] + coef[:, 1] * values[:, 1]) + coef[:, 2] * values[
        :, 2
    ]
    return s * inv_area


def setup(scene, extra=None):
    """Per-triangle records of the triangles that can cover a pixel (dict of arrays).

    extra (T, 3, k): also each record's corner values ("extra", (R, 3, k), in the
    record's corner order), the planes of value times 1/z ("extra_planes", (R, 3, k),
    [x, y, 1] like the colour planes) and the doubled area ("area2").
    """
    x, y, ww, col, src = corners(scene, extra)
    cand = np.ones(len(src), dtype=bool)
    area2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
    area2 = area2 - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    cand &= area2 != 0
    flip = area2 < 0
    order = np.where(flip[:, None], [0, 2, 1], [0, 1, 2])
    rows = np.arange(len(src))[:, None]
    x, y, ww, col = x[rows, order], y[rows, order], ww[rows, order], col[rows, order]
    area2 = np.where(flip, -area2, area2)
    lo_x = np.minimum(np.minimum(x[:, 0], x[:, 1]), x[:, 2])
    hi_x = np.maximum(np.maximum(x[:, 0], x[:, 1]), x[:, 2])
    lo_y = np.minimum(np.minimum(y[:, 0], y[:, 1]), y[:, 2])
    hi_y = np.maximum(np.maximum(y[:, 0], y[:, 1]), y[:, 2])
    half = F(0.5)
    across, down = grid(scene)
    width, height = TILE * across, TILE * down  # the padded frame
    ix0 = np.minimum(np.maximum(np.ceil(lo_x - half), F(0)), F(width)).astype(np.int64)
    ix1 = np.maximum(np.minimum(np.floor(hi_x - half), F(width - 1)), F(-1)).astype(
        np.int64
    )
    iy0 = np.minimum(np.maximum(np.ceil(lo_y - half), F(0)), F(height)).astype(np.int64)
    iy1 = np.maximum(np.minimum(np.floor(hi_y - half), F(height - 1)), F(-1)).astype(
        np.int64
    )
    cand &= (ix0 <= ix1) & (iy0 <= iy1)
    keep = np.nonzero(cand)[0]
    x, y, ww, col = x[keep], y[keep], ww[keep], col[keep]
    col, values = col[:, :, :3], col[:, :, 3:]
    area2, ix0, ix1, iy0, iy1 = (a[keep] for a in (area2, ix0, ix1, iy0, iy1))
    a_coef, b_coef, c_coef = (
        np.empty((len(keep), 3), dtype=np.float32) for _ in range(3)
    )
    for i, (p, q) in enumerate([(1, 2), (2, 0), (0, 1)]):
        a_coef[:, i] = y[:, p] - y[:, q]
        b_coef[:, i] = x[:, q] - x[:, p]
        c_coef[:, i] = -(a_coef[:, i] * x[:, p] + b_coef[:, i] * y[:, p])
    top_left = (a_coef > 0) | ((a_coef == 0) & (b_coef > 0))
    inv_area = F(1.0) / area2
    wp = np.stack([planes(ww, c, inv_area) for c in (a_coef, b_coef, c_coef)], axis=1)
    cp = np.empty((len(keep), 3, 3), dtype=np.float32)
    for ch in range(3):
        for k, c in enumerate((a_coef, b_coef, c_coef)):
            cp[:, k, ch] = planes(col[:, :, ch], c, inv_area)
    records = {
        "tri": src[keep].astype(np.int32),
        "a": a_coef,
        "b": b_coef,
        "c": c_coef,
        "top_left": top_left,
        "w_planes": wp,
        "c_planes": cp,
        "box": np.stack([ix0, ix1, iy0, iy1], axis=1).astype(np.int32),
    }
    if extra is not None:
        weighted = values * ww[:, :, None]
        ep = np.empty((len(keep), 3, values.shape[2]), dtype=np.float32)
        for j in range(values.shape[2]):
            for k, c in enumerate((a_coef, b_coef, c_coef)):
                ep[:, k, j] = planes(weighted[:, :, j], c, inv_area)
        records |= {"extra": values, "extra_planes": ep, "area2": area2}
    return records


def bin_triangles(records, tiles_x=TILES_X, tiles_y=TILES_Y):
    """Tile lists: (counts per tile, offsets (tiles + 1), record numbers by tile);
    tile ty tiles_x + tx of a grid of tiles_x by tiles_y."""
    box = records["box"]
    tx0, tx1 = box[:, 0] // TILE, box[:, 1] // TILE
    ty0, ty1 = box[:, 2] // TILE, box[:, 3] // TILE
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    per_tri = nx * ny
    total = int(per_tri.sum())
    rec = np.repeat(np.arange(len(box)), per_tri)
    start = np.cumsum(per_tri) - per_tri
    local = np.arange(total) - np.repeat(start, per_tri)
    tile = (ty0[rec] + local // nx[rec]) * tiles_x + tx0[rec] + local % nx[rec]
    order = np.argsort(tile, kind="stable")
    counts = np.bincount(tile, minlength=tiles_x * tiles_y).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return counts, offsets, rec[order].astype(np.int32)


def fill_tile(records, members, tx, ty):
    """Colour (32, 32, 3), 1/z and record number (-1: none) of one tile."""
    px = (F(tx * TILE) + np.arange(TILE, dtype=np.float32) + F(0.5))[None, None, :]
    py = (F(ty * TILE) + np.arange(TILE, dtype=np.float32) + F(0.5))[None, :, None]
    cols = (np.arange(TILE) + tx * TILE)[None, None, :]
    rows = (np.arange(TILE) + ty * TILE)[None, :, None]
    depth = np.zeros((TILE, TILE), dtype=np.float32)
    winner = np.full((TILE, TILE), -1, dtype=np.int64)
    colour = np.empty((TILE, TILE, 3), dtype=np.float32)
    colour[:] = BACKGROUND
    if len(members) == 0:
        return colour, depth, winner
    for s in range(0, len(members), CHUNK):
        m = members[s : s + CHUNK]
        a, b, c = (records[k][m][:, :, None, None] for k in ("a", "b", "c"))
        tl = records["top_left"][m][:, :, None, None]
        e = (a * px[:, None] + b * py[:, None]) + c
        inside = ((e > 0) | ((e == 0) & tl)).all(axis=1)
        box = records["box"][m]
        inside &= (cols >= box[:, 0, None, None]) & (cols <= box[:, 1, None, None])
        inside &= (rows >= box[:, 2, None, None]) & (rows <= box[:, 3, None, None])
        wp = records["w_planes"][m]
        w = (wp[:, 0, None, None] * px + wp[:, 1, None, None] * py) + wp[
            :, 2, None, None
        ]
        w = np.where(inside, w, F(-np.inf))
        first = w.argmax(axis=0)
        best = np.take_along_axis(w, first[None], axis=0)[0]
        better = best > depth
        depth = np.where(better, best, depth)
        winner = np.where(better, m[first], winner)
    hit = winner >= 0
    cp = records["c_planes"][np.where(hit, winner, 0)]
    shade = (
        cp[:, :, 0] * px[0, 0, :, None] + cp[:, :, 1] * py[0, :, 0, None, None]
    ) + cp[:, :, 2]
    colour[hit] = shade[hit]
    return colour, depth, winner


def rasterize(scene):
    """Colour (H, W, 3), 1/z (H, W), triangle index (H, W, -1 = none), tile counts
    (TY, TX), of the padded frame (`grid`: H = TILE TY, W = TILE TX; `picture`)."""
    across, down = grid(scene)
    records = setup(scene)
    counts, offsets, members = bin_triangles(records, across, down)
    colour = np.empty((TILE * down, TILE * across, 3), dtype=np.float32)
    depth = np.empty((TILE * down, TILE * across), dtype=np.float32)
    tri = np.empty((TILE * down, TILE * across), dtype=np.int32)
    for t in range(across * down):
        ty, tx = divmod(t, across)
        c, d, win = fill_tile(records, members[offsets[t] : offsets[t + 1]], tx, ty)
        ys, xs = slice(ty * TILE, (ty + 1) * TILE), slice(tx * TILE, (tx + 1) * TILE)
        colour[ys, xs], depth[ys, xs] = c, d
        tri[ys, xs] = -1
        if len(records["tri"]):
            tri[ys, xs] = np.where(win >= 0, records["tri"][np.maximum(win, 0)], -1)
    return {
        "colour": colour,
        "depth": depth,
        "tri": tri,
        "tile_counts": counts.reshape(down, across),
    }


def layer_scene(scene, keep):
    """The scene with the triangles not in `keep` collapsed to their first corner
    (zero area: setup drops them), so the others keep their triangle numbers."""
    indices = scene["indices"].copy()
    indices[~keep] = indices[~keep][:, :1]
    return {**scene, "indices": indices}


def fill_layers(scene, materials, glass=True):
    """SHADE's FILLs: the layers of the triangles that are neither `masked` nor
    `transparent`, of the `masked` ones and of the `transparent` ones (glass); the
    glass layer is empty (no FILL) when there are none or glass is false."""
    kind = materials["kind"][materials["material"]]
    clear = kind == TRANSPARENT
    opaque = rasterize(layer_scene(scene, (kind != MASKED) & ~clear))
    masked = rasterize(layer_scene(scene, kind == MASKED))
    if glass and clear.any():
        return opaque, masked, rasterize(layer_scene(scene, clear))
    shape = opaque["depth"].shape
    empty = {
        "colour": np.broadcast_to(BACKGROUND, shape + (3,)).copy(),
        "depth": np.zeros(shape, dtype=np.float32),
        "tri": np.full(shape, -1, dtype=np.int32),
        "tile_counts": np.zeros_like(opaque["tile_counts"]),
    }
    return opaque, masked, empty


def glass_numbers(materials, layer, front):
    """Transmission tau (H, W, 3) of the glass layer's material where it is in front
    and its own surface's weight 1 - max(tau) (H, W, 1); both 0 elsewhere."""
    m = materials["material"][np.maximum(layer["tri"], 0)]
    tau = np.where(front[:, :, None], materials["transmission"][m], F(0))
    weight = np.where(front, F(1) - materials["transmission"][m].max(axis=2), F(0))
    return tau, weight[:, :, None]


def dot3(a, b):
    """(a0 b0 + a1 b1) + a2 b2 over the last axis (float32, in this order)."""
    return (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) + a[..., 2] * b[..., 2]


def normalized(v):
    """v / |v| over the last axis; zero where |v| is zero."""
    length = np.sqrt(dot3(v, v))
    safe = np.where(length > 0, length, F(1))
    return np.where((length > 0)[..., None], v / safe[..., None], F(0))


def cross3(a, b):
    """The cross product over the last axis (float32)."""
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def mip_level(texels, area2, levels):
    """The level (0 = largest) at which a texel is about a pixel: the number of
    k = 1 .. levels - 1 with texels > area2 2^(2k - 1), texels = the piece's
    doubled area in level-0 texels and area2 its doubled area in pixels."""
    level = np.zeros(texels.shape, dtype=np.int32)
    for k in range(1, int(levels.max(initial=1))):
        level += (texels > area2 * F(2.0 ** (2 * k - 1))) & (k < levels)
    return level


def shade_setup(scene, materials):
    """Per triangle what SHADE needs (T rows, valid where `first` >= 0): the planes of
    u, v and light term times 1/z of its first kept piece ("planes" (T, 3, 3),
    [x, y, 1][u, v, light]), a mip level per texture slot ("level" (T, 4)) and, for
    `lit`, the tangent frame ("frame" (T, 3, 3): tangent, bitangent, normal facing
    the eye; "frame_ok" false where the coordinates give no tangent)."""
    count = len(scene["indices"])
    extra = np.concatenate([materials["uv"], materials["light"][:, :, None]], axis=2)
    records = setup(scene, extra)
    first = np.full(count, -1, dtype=np.int64)
    first[records["tri"][::-1]] = np.arange(len(records["tri"]))[::-1]
    has = first >= 0
    pick = np.where(has, first, 0)
    planes_of = np.zeros((count, 3, 3), dtype=np.float32)
    level = np.zeros((count, 4), dtype=np.int32)
    if len(records["tri"]):
        planes_of = np.where(has[:, None, None], records["extra_planes"][pick], F(0))
        corner = records["extra"][pick]
        du1, dv1 = corner[:, 1, 0] - corner[:, 0, 0], corner[:, 1, 1] - corner[:, 0, 1]
        du2, dv2 = corner[:, 2, 0] - corner[:, 0, 0], corner[:, 2, 1] - corner[:, 0, 1]
        uv_area2 = np.abs(du1 * dv2 - du2 * dv1)
        texture = materials["textures"][materials["material"]]
        for slot in range(4):
            tex = np.maximum(texture[:, slot], 0)
            size = materials["mip"][tex, 0]
            texels = uv_area2 * (size[:, 1] * size[:, 2]).astype(np.float32)
            levels = np.where(texture[:, slot] >= 0, materials["levels"][tex], 1)
            level[:, slot] = mip_level(texels, records["area2"][pick], levels)
    p = scene["positions"][scene["indices"]]
    uv = materials["uv"]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    normal = normalized(cross3(e1, e2))
    to_eye = scene["camera"][None, 0:3] - p[:, 0]
    normal = np.where((dot3(normal, to_eye) < 0)[:, None], -normal, normal)
    du1, dv1 = uv[:, 1, 0] - uv[:, 0, 0], uv[:, 1, 1] - uv[:, 0, 1]
    du2, dv2 = uv[:, 2, 0] - uv[:, 0, 0], uv[:, 2, 1] - uv[:, 0, 1]
    det = du1 * dv2 - du2 * dv1
    sign = np.where(det < 0, F(-1), F(1))[:, None]
    tangent = (e1 * dv2[:, None] - e2 * dv1[:, None]) * sign
    bitangent = (e2 * du1[:, None] - e1 * du2[:, None]) * sign
    tangent = normalized(tangent - normal * dot3(normal, tangent)[:, None])
    side = cross3(normal, tangent)
    side = np.where((dot3(side, bitangent) < 0)[:, None], -side, side)
    frame_ok = (det != 0) & (dot3(tangent, tangent) > 0)
    return {
        "first": first,
        "planes": planes_of,
        "level": level,
        "frame": np.stack([tangent, side, normal], axis=1),
        "frame_ok": frame_ok,
    }


def at(plane, px, py):
    """A plane [x, y, 1] at pixel centres: (p0 px + p1 py) + p2."""
    return (plane[:, 0] * px + plane[:, 1] * py) + plane[:, 2]


def texel_xy(materials, tex, level, u, v):
    """Column and row of the texel nearest to (u, v) in a level, coordinates wrapping;
    v runs up the image (row 0 is v = 1). Also the level's (offset, width, height)."""
    entry = materials["mip"][tex, level]
    width, height = entry[:, 1], entry[:, 2]
    s = u - np.floor(u)
    t = v - np.floor(v)
    s = np.where(np.isfinite(s), s, F(0))
    t = np.where(np.isfinite(t), t, F(0))
    x = np.minimum((s * width.astype(np.float32)).astype(np.int64), width - 1)
    y = np.minimum(
        ((F(1) - t) * height.astype(np.float32)).astype(np.int64), height - 1
    )
    return x, y, entry


def fetch(materials, tex, level, u, v):
    """The texel (n, 4) uint8 nearest to (u, v) (texel_xy)."""
    x, y, entry = texel_xy(materials, tex, level, u, v)
    start = entry[:, 0] + (y * entry[:, 1] + x) * 4
    return materials["texels"][start[:, None] + np.arange(4)]


def power(x, exponent):
    """x ** exponent for integer exponents >= 1, by squaring (low bit first)."""
    result = np.ones_like(x)
    base = x.copy()
    e = exponent.copy()
    while (e > 0).any():
        odd = (e & 1) == 1
        result = np.where(odd, result * base, result)
        base = base * base
        e >>= 1
    return result


def pixel_values(layer, prepared, ys, xs):
    """Pixel centres, 1/z and the winner's u, v and light term at pixels (ys, xs)."""
    t = layer["tri"][ys, xs]
    px = xs.astype(np.float32) + F(0.5)
    py = ys.astype(np.float32) + F(0.5)
    w = layer["depth"][ys, xs]
    plane = prepared["planes"][t]
    u = at(plane[:, :, 0], px, py) / w
    v = at(plane[:, :, 1], px, py) / w
    light = at(plane[:, :, 2], px, py) / w
    return px, py, w, u, v, light


def shade_layer(scene, materials, prepared, layer):
    """Colour (H, W, 3) of one FILL layer by its winners' materials, and its holes
    (H, W) (masked pixels whose alpha byte is under OPAQUE_BYTE)."""
    colour = layer["colour"].copy()
    hole = np.zeros(layer["tri"].shape, dtype=bool)
    ys, xs = np.nonzero(layer["tri"] >= 0)
    t = layer["tri"][ys, xs]
    m = materials["material"][t]
    kind = materials["kind"][m]
    shaded = kind != FLAT
    ys, xs, t, m, kind = ys[shaded], xs[shaded], t[shaded], m[shaded], kind[shaded]
    px, py, w, u, v, light = pixel_values(layer, prepared, ys, xs)
    texture = materials["textures"][m]
    level = prepared["level"][t]
    base = materials["base_colour"][m]

    def texel(slot):
        if len(materials["mip"]) == 0:
            return np.zeros((len(u), 4), dtype=np.uint8)
        tex = np.maximum(texture[:, slot], 0)
        return fetch(materials, tex, level[:, slot], u, v)

    albedo_texel = texel(ALBEDO)
    albedo = np.where(
        (texture[:, ALBEDO] >= 0)[:, None],
        TEXEL_LINEAR[albedo_texel[:, :3]] * base,
        base,
    )
    out = albedo * light[:, None]
    lit = kind == LIT
    if lit.any():
        out[lit] = shade_lit(
            scene, materials, prepared, t, m, px, py, w, texel, albedo
        )[lit]
    out = np.where((kind == EMISSIVE)[:, None], albedo, out)
    masked = kind == MASKED
    albedo_alpha = np.where(texture[:, ALBEDO] >= 0, albedo_texel[:, 3], 255)
    alpha = np.where(texture[:, MASK] >= 0, texel(MASK)[:, 3], albedo_alpha)
    hole[ys, xs] = masked & (alpha < OPAQUE_BYTE)
    colour[ys, xs] = out
    return colour, hole


def shade_lit(scene, materials, prepared, t, m, px, py, w, texel, albedo):
    """The lit kind's colour: albedo (ambient + diffuse) + Blinn-Phong specular from
    the scene's point light, with the normal map's normal in the triangle's frame."""
    cam = scene["camera"]
    z = F(1) / w
    xc = ((px - cam[13]) / cam[12]) * z
    yc = ((cam[14] - py) / cam[12]) * z
    position = np.stack(
        [
            ((cam[k] + cam[3 + k] * xc) + cam[6 + k] * yc) + cam[9 + k] * z
            for k in range(3)
        ],
        axis=-1,
    )
    to_light = normalized(scene["light"][None, :] - position)
    to_eye = normalized(cam[None, 0:3] - position)
    half = normalized(to_light + to_eye)
    frame = prepared["frame"][t]
    texture = materials["textures"][m]
    mapped = (texture[:, NORMAL] >= 0) & prepared["frame_ok"][t]
    n_texel = TEXEL_SIGNED[texel(NORMAL)[:, :3]]
    bent = normalized(
        (frame[:, 0] * n_texel[:, 0:1] + frame[:, 1] * n_texel[:, 1:2])
        + frame[:, 2] * n_texel[:, 2:3]
    )
    normal = np.where(mapped[:, None], bent, frame[:, 2])
    diffuse = np.maximum(dot3(normal, to_light), F(0))
    gloss = power(np.maximum(dot3(normal, half), F(0)), materials["exponent"][m])
    specular = np.where(
        (texture[:, SPECULAR] >= 0)[:, None],
        materials["specular"][m] * TEXEL_UNIT[texel(SPECULAR)[:, :3]],
        materials["specular"][m],
    )
    return albedo * (AMBIENT + DIFFUSE * diffuse)[:, None] + specular * gloss[:, None]


def nearer_layer(first, second):
    """Where the first layer's fragment wins over the second's (larger 1/z, a tie to
    the lower triangle number; an empty second pixel always loses)."""
    return (first["tri"] >= 0) & (
        (first["depth"] > second["depth"])
        | ((first["depth"] == second["depth"]) & (first["tri"] < second["tri"]))
    )


def rasterize_shaded(scene, materials):
    """SHADE after three FILLs (fill_layers): the masked layer where nearer and not a
    hole, else the other; glass where nearer, tau C + (1 - max tau) a light. Colour,
    the surface behind the glass (1/z, triangle), the tile counts added, the layers
    ("opaque", "masked", "glass") and where the glass is in front ("front")."""
    opaque, masked, glass = fill_layers(scene, materials)
    prepared = shade_setup(scene, materials)
    opaque_colour, _ = shade_layer(scene, materials, prepared, opaque)
    masked_colour, hole = shade_layer(scene, materials, prepared, masked)
    take = nearer_layer(masked, opaque) & ~hole
    colour = np.where(take[:, :, None], masked_colour, opaque_colour)
    depth = np.where(take, masked["depth"], opaque["depth"])
    tri = np.where(take, masked["tri"], opaque["tri"])
    front = nearer_layer(glass, {"depth": depth, "tri": tri})
    if front.any():
        glass_colour, _ = shade_layer(scene, materials, prepared, glass)
        tau, weight = glass_numbers(materials, glass, front)
        mixed = tau * colour + weight * glass_colour
        colour = np.where(front[:, :, None], mixed, colour)
    counts = opaque["tile_counts"] + masked["tile_counts"] + glass["tile_counts"]
    return {
        "colour": colour,
        "depth": depth,
        "tri": tri,
        "tile_counts": counts,
        "opaque": opaque,
        "masked": masked,
        "glass": glass,
        "front": front,
    }


def world_points(cam, px, py, w):
    """Scene positions (n, 3) of pixel centres px, py at 1/z w."""
    z = F(1) / w
    xc = ((px - cam[13]) / cam[12]) * z
    yc = ((cam[14] - py) / cam[12]) * z
    return np.stack(
        [
            ((cam[k] + cam[3 + k] * xc) + cam[6 + k] * yc) + cam[9 + k] * z
            for k in range(3)
        ],
        axis=-1,
    )


def smooth_setup(scene):
    """Per triangle the planes (T, 3, 3) of its corners' area-weighted vertex normals
    times 1/z, of its first kept piece (as `shade_setup`): the smooth-normal variant."""
    normals = light_data.vertex_normals(scene["positions"], scene["indices"])
    records = setup(scene, normals[scene["indices"]])
    count = len(scene["indices"])
    first = np.full(count, -1, dtype=np.int64)
    first[records["tri"][::-1]] = np.arange(len(records["tri"]))[::-1]
    pick = np.maximum(first, 0)
    return np.where((first >= 0)[:, None, None], records["extra_planes"][pick], F(0))


def light_layer(scene, materials, prepared, layer, lights, variant):
    """Linear colour (H, W, 3) of one FILL layer under the lights (the terms in
    variant["terms"]), its holes, and the parts (README "Lighting")."""
    shape = layer["tri"].shape
    terms = variant["terms"]
    colour = np.zeros(shape + (3,), dtype=np.float32)
    parts = {
        k: np.zeros(shape + (3,), dtype=np.float32)
        for k in ("sky", "sun", "points", "albedo", "normal", "specular")
    }
    parts["visibility"] = np.zeros(shape, dtype=np.float32)
    parts["material"] = np.full(shape, -1, dtype=np.int32)
    hole = np.zeros(shape, dtype=bool)
    ys, xs = np.nonzero(layer["tri"] >= 0)
    t = layer["tri"][ys, xs]
    m = materials["material"][t]
    kind = materials["kind"][m]
    px, py, w, u, v, vis = pixel_values(layer, prepared, ys, xs)
    texture = materials["textures"][m]
    level = prepared["level"][t]
    base = materials["base_colour"][m]

    def texel(slot):
        if len(materials["mip"]) == 0:
            return np.zeros((len(u), 4), dtype=np.uint8)
        tex = np.maximum(texture[:, slot], 0)
        return fetch(materials, tex, level[:, slot], u, v)

    albedo_texel = texel(ALBEDO)
    albedo = np.where(
        (texture[:, ALBEDO] >= 0)[:, None],
        TEXEL_LINEAR[albedo_texel[:, :3]] * base,
        base,
    )
    cam = scene["camera"]
    position = world_points(cam, px, py, w)
    to_eye = normalized(cam[None, 0:3] - position)
    frame = prepared["frame"][t]
    if variant.get("smooth") is not None:
        smooth = normalized(
            np.stack(
                [at(variant["smooth"][t][:, :, k], px, py) / w for k in range(3)], -1
            )
        )
        smooth = np.where((dot3(smooth, to_eye) < 0)[:, None], -smooth, smooth)
        frame = np.stack([frame[:, 0], frame[:, 1], smooth], axis=1)
    mapped = (texture[:, NORMAL] >= 0) & prepared["frame_ok"][t]
    n_texel = TEXEL_SIGNED[texel(NORMAL)[:, :3]]
    bent = normalized(
        (frame[:, 0] * n_texel[:, 0:1] + frame[:, 1] * n_texel[:, 1:2])
        + frame[:, 2] * n_texel[:, 2:3]
    )
    normal = np.where(mapped[:, None], bent, frame[:, 2])
    glass = kind == TRANSPARENT
    normal = np.where(glass[:, None], prepared["frame"][t][:, 2], normal)
    has_specular = ((kind == LIT) | glass) & (materials["specular"][m].max(axis=1) > 0)
    specular = np.where(
        (texture[:, SPECULAR] >= 0)[:, None],
        materials["specular"][m] * TEXEL_UNIT[texel(SPECULAR)[:, :3]],
        materials["specular"][m],
    )
    specular = np.where(has_specular[:, None], specular, F(0))
    if "specular" not in terms:
        specular = np.zeros_like(specular)
    exponent = materials["exponent"][m]
    inv_pi = F(1 / np.pi)

    def gloss(to_light):
        half = normalized(to_light + to_eye)
        return power(np.maximum(dot3(normal, half), F(0)), exponent)

    zero = np.zeros_like(albedo)
    sky = zero
    if "sky" in terms and "indirect" not in terms:
        sky = np.maximum(light_data.sky_irradiance(lights["sh"], normal), F(0))
        sky = sky * vis[:, None]
    sun_dir = lights["sun_direction"]
    sun_e = zero
    if "sun" in terms:
        shadow = np.ones(len(position), dtype=np.float32)
        if "shadow" in terms:
            shadow = light_data.sun_shadow(lights, position, normal, variant["pcf"])
        sun_e = (
            lights["sun_irradiance"]
            * (np.maximum(dot3(normal, sun_dir[None, :]), F(0)) * shadow)[:, None]
        )
    shine = specular * gloss(sun_dir[None, :])[:, None] * (sun_e * inv_pi)
    points_e = np.zeros_like(albedo)
    for k in range(len(lights["lamp_intensity"]) if "lamps" in terms else 0):
        e, to_light = light_data.point_irradiance(
            lights["lamp_intensity"][k],
            lights["size"],
            lights["radius"],
            lights["lamp_position"][k],
            position,
            normal,
        )
        lamp_e = lights["lamp_colour"][k] * e[:, None]
        points_e = points_e + lamp_e
        shine = shine + specular * gloss(to_light)[:, None] * (lamp_e * inv_pi)
    ambient = np.asarray(lights["ambient"], dtype=np.float32)
    if "ambient" not in terms or "indirect" in terms:
        ambient = np.zeros(3, dtype=np.float32)
    light_sum = ((sky + ambient) + sun_e) + points_e
    if "indirect" in terms:
        bounce = variant["indirect"](position, normal)
        light_sum = (bounce + sun_e) + points_e
    out = albedo * inv_pi * light_sum
    if glass.any():
        own = F(1) - materials["transmission"][m].max(axis=1)
        out = out * np.where(glass, own, F(1))[:, None]
    out = out + shine
    emissive = kind == EMISSIVE
    out = np.where(emissive[:, None], albedo * F(lights["emissive_scale"]), out)
    lit_parts = {
        "sky": albedo * inv_pi * sky,
        "sun": albedo * inv_pi * sun_e,
        "points": albedo * inv_pi * points_e,
    }
    if "indirect" in terms:
        lit_parts["indirect"] = albedo * inv_pi * bounce
        parts.setdefault("indirect", np.zeros(shape + (3,), dtype=np.float32))
    for name, value in lit_parts.items():
        parts[name][ys, xs] = np.where(emissive[:, None], F(0), value)
    parts["visibility"][ys, xs] = vis
    parts["albedo"][ys, xs] = albedo
    parts["normal"][ys, xs] = normal
    parts["specular"][ys, xs] = specular
    parts["material"][ys, xs] = m
    masked = kind == MASKED
    albedo_alpha = np.where(texture[:, ALBEDO] >= 0, albedo_texel[:, 3], 255)
    alpha = np.where(texture[:, MASK] >= 0, texel(MASK)[:, 3], albedo_alpha)
    hole[ys, xs] = masked & (alpha < OPAQUE_BYTE)
    colour[ys, xs] = out
    return colour, hole, parts


def background(scene, lights):
    """The environment's radiance (H, W, 3) in each pixel's view direction
    (environment_radiance), over the padded frame (`grid`)."""
    cam = scene["camera"]
    across, down = grid(scene)
    height, width = TILE * down, TILE * across
    ys, xs = np.mgrid[0:height, 0:width]
    sx = (xs.ravel().astype(np.float32) + F(0.5) - cam[13]) / cam[12]
    sy = (ys.ravel().astype(np.float32) + F(0.5) - cam[14]) / cam[12]
    d = (cam[9:12] + cam[3:6] * sx[:, None]) - cam[6:9] * sy[:, None]
    return environment_radiance(lights, d).reshape(height, width, 3)


def environment_radiance(lights, d):
    """The environment's radiance (n, 3) along directions d (n, 3) turned by -turn
    about y: the nearest texel of the octahedral map (the map's frame, y up)."""
    angle = np.radians(-float(lights["turn"]))
    c, s = F(np.cos(angle)), F(np.sin(angle))
    x, z = d[:, 0], d[:, 2]
    d = np.stack([c * x + s * z, d[:, 1], c * z - s * x], -1)
    octa = lights["octahedral"]
    row, column = light_data.octahedral_texel(d, octa.shape[0])
    return octa[row, column]


def glass_over(scene, lights, glass, parts, front, behind, terms):
    """The colour where the glass is in front: tau C + light_layer's (1 - max tau)
    a / pi E + S (`behind`: C, that own colour, tau) + F x reflection x R, R the
    environment along the view mirrored about the glass's normal (with "background")."""
    ys, xs = np.nonzero(front)
    px, py = xs.astype(np.float32) + F(0.5), ys.astype(np.float32) + F(0.5)
    cam = scene["camera"]
    position = world_points(cam, px, py, glass["depth"][ys, xs])
    to_eye = normalized(cam[None, 0:3] - position)
    normal = parts["normal"][ys, xs]
    facing = np.maximum(dot3(normal, to_eye), F(0))
    q = F(1) - facing
    fresnel = GLASS_F0 + (F(1) - GLASS_F0) * (((q * q) * (q * q)) * q)
    mirror = normal * (F(2) * facing)[:, None] - to_eye
    reflection = F(lights.get("reflection", 0.0))
    shine = np.zeros((len(ys), 3), dtype=np.float32)
    if "background" in terms and reflection > 0:
        shine = (fresnel * reflection)[:, None] * environment_radiance(lights, mirror)
    colour, own, tau = behind
    out = colour.copy()
    seen = tau[ys, xs] * colour[ys, xs]
    out[ys, xs] = (seen + own[ys, xs]) + shine
    return out


def tone(colour, exposure, saturation=1.0, curve=CURVE_PLAIN):
    """The tone step of a linear colour (..., 3): x = exposure x colour; saturation s
    (s = 1: none) makes it max(0, lum + (x - lum) s), lum = 0.2126 R + 0.7152 G +
    0.0722 B; then the curve: CURVE_PLAIN x / (1 + x), CURVE_FILMIC the fit
    x (2.51 x + 0.03) / (x (2.43 x + 0.59) + 0.14) clamped to [0, 1]. Float32."""
    x = F(exposure) * colour
    s = F(saturation)
    if s != F(1):
        r, g, b = TONE_LUMA
        lum = (x[..., 0:1] * r + x[..., 1:2] * g) + x[..., 2:3] * b
        x = np.maximum(lum + (x - lum) * s, F(0))
    if curve == CURVE_PLAIN:
        return x / (F(1) + x)
    if curve == CURVE_FILMIC:
        a, b, c, d, e = FILMIC
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), F(0), F(1))
    raise ValueError(
        f"a tone curve is {CURVE_PLAIN} (plain) or {CURVE_FILMIC} (filmic)"
    )


def rasterize_lighted(scene, materials, lights, smooth=False, pcf=True, terms=TERMS):
    """SHADE's three FILLs and holes with direct light (raster reference README,
    "Lighting", "Glass", "Indirect light"); terms: what contributes, of TERMS and
    "indirect". "colour" linear, "display" its `tone` with the lights' exposure,
    saturation (default 1) and curve (default 0: x / (1 + x)), the parts.
    smooth (vertex normals): a comparison variant; pcf=False: one shadow tap."""
    opaque, masked, glass = fill_layers(scene, materials, "glass" in terms)
    corner_vis = lights["visibility"][scene["indices"]]
    prepared = shade_setup(scene, {**materials, "light": corner_vis})
    variant = {"smooth": smooth_setup(scene) if smooth else None, "pcf": pcf}
    variant["terms"] = set(terms)
    unknown = variant["terms"] - set(TERMS) - set(OPTIONAL_TERMS)
    if unknown:
        raise ValueError(f"unknown lighting terms {sorted(unknown)}")
    if "indirect" in variant["terms"]:
        indirect = importlib.import_module("reference.indirect")
        indirect.field_for(lights)
        variant["indirect"] = lambda p, n: indirect.irradiance(lights, p, n)
    opaque_colour, _, opaque_parts = light_layer(
        scene, materials, prepared, opaque, lights, variant
    )
    masked_colour, hole, masked_parts = light_layer(
        scene, materials, prepared, masked, lights, variant
    )
    take = nearer_layer(masked, opaque) & ~hole
    tri = np.where(take, masked["tri"], opaque["tri"])
    sky = (tri < 0)[:, :, None]
    colour = np.where(take[:, :, None], masked_colour, opaque_colour)
    behind = BACKGROUND
    if "background" in variant["terms"]:
        behind = background(scene, lights)
    colour = np.where(sky, behind, colour)
    parts = {}
    for name in opaque_parts:
        pick = take if opaque_parts[name].ndim == 2 else take[:, :, None]
        parts[name] = np.where(pick, masked_parts[name], opaque_parts[name])
    depth = np.where(take, masked["depth"], opaque["depth"])
    parts["depth"] = depth
    parts["layer"] = np.where(tri < 0, -1, take.astype(np.int32)).astype(np.int8)
    front = nearer_layer(glass, {"depth": depth, "tri": tri})
    parts["glass_depth"] = np.where(front, glass["depth"], F(0))
    parts["glass_tri"] = np.where(front, glass["tri"], -1).astype(np.int32)
    parts["glass_material"] = np.full(tri.shape, -1, dtype=np.int32)
    parts["glass_normal"] = np.zeros(tri.shape + (3,), dtype=np.float32)
    if front.any():
        own, _, glass_parts = light_layer(
            scene, materials, prepared, glass, lights, variant
        )
        tau, _ = glass_numbers(materials, glass, front)
        behind = (colour, own, tau)
        colour = glass_over(scene, lights, glass, glass_parts, front, behind, terms)
        parts["glass_material"] = np.where(front, glass_parts["material"], -1)
        parts["glass_normal"] = np.where(front[:, :, None], glass_parts["normal"], F(0))
    display = tone(
        colour,
        lights["exposure"],
        lights.get("saturation", 1.0),
        lights.get("curve", CURVE_PLAIN),
    )
    return {
        "colour": colour,
        "display": display,
        "depth": depth,
        "tri": tri,
        "parts": parts,
    }
