"""Rasterization in numpy, float32 in the order written, no fused multiply-add: the
definition the C++ and CUDA versions match (projects/raster/reference/README.md).
Triangles are clipped at the near distance, binned to 32 x 32 tiles by pixel box, filled
by the top-left rule; the larger 1/z wins, a tie goes to the lower triangle index.

SHADE (`rasterize_shaded`) colours each pixel from its winner's material: FILL runs
once over the triangles that are not `masked` and once over those that are, and the
masked layer's pixel is taken where it is nearer and not a hole."""

import numpy as np

from reference import lights as light_data
from reference.scenes import HEIGHT, WIDTH

TILE = 32
TILES_X = WIDTH // TILE
TILES_Y = HEIGHT // TILE
BACKGROUND = np.array([0.05, 0.05, 0.08], dtype=np.float32)
CHUNK = 128
F = np.float32
# SHADE: the kinds' codes (reference/public.py KINDS), texture slots, the lit kind's
# ambient and diffuse weights, and the alpha byte under which a masked pixel is a hole.
FLAT, TEXTURED, MASKED, LIT, EMISSIVE = range(5)
ALBEDO, MASK, NORMAL, SPECULAR = range(4)
AMBIENT, DIFFUSE = F(0.25), F(0.75)
OPAQUE_BYTE = 128
# LIGHTED: the terms of rasterize_lighted's colour (all by default).
TERMS = ("sky", "ambient", "sun", "shadow", "lamps", "specular", "background")
# Texel byte to value: albedo as linear colour (the pictures' 2.2 gamma undone), the
# other textures to [0, 1] and normals to [-1, 1].
TEXEL_LINEAR = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)
TEXEL_UNIT = (np.arange(256) / 255.0).astype(np.float32)
TEXEL_SIGNED = (np.arange(256) * 2.0 / 255.0 - 1.0).astype(np.float32)


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
    ix0 = np.minimum(np.maximum(np.ceil(lo_x - half), F(0)), F(WIDTH)).astype(np.int64)
    ix1 = np.maximum(np.minimum(np.floor(hi_x - half), F(WIDTH - 1)), F(-1)).astype(
        np.int64
    )
    iy0 = np.minimum(np.maximum(np.ceil(lo_y - half), F(0)), F(HEIGHT)).astype(np.int64)
    iy1 = np.maximum(np.minimum(np.floor(hi_y - half), F(HEIGHT - 1)), F(-1)).astype(
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


def bin_triangles(records):
    """Tile lists: (counts per tile, offsets (tiles + 1), record numbers by tile)."""
    box = records["box"]
    tx0, tx1 = box[:, 0] // TILE, box[:, 1] // TILE
    ty0, ty1 = box[:, 2] // TILE, box[:, 3] // TILE
    nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
    per_tri = nx * ny
    total = int(per_tri.sum())
    rec = np.repeat(np.arange(len(box)), per_tri)
    start = np.cumsum(per_tri) - per_tri
    local = np.arange(total) - np.repeat(start, per_tri)
    tile = (ty0[rec] + local // nx[rec]) * TILES_X + tx0[rec] + local % nx[rec]
    order = np.argsort(tile, kind="stable")
    counts = np.bincount(tile, minlength=TILES_X * TILES_Y).astype(np.int32)
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
    """Colour (H, W, 3), 1/z (H, W), triangle index (H, W, -1 = none), tile counts."""
    records = setup(scene)
    counts, offsets, members = bin_triangles(records)
    colour = np.empty((HEIGHT, WIDTH, 3), dtype=np.float32)
    depth = np.empty((HEIGHT, WIDTH), dtype=np.float32)
    tri = np.empty((HEIGHT, WIDTH), dtype=np.int32)
    for t in range(TILES_X * TILES_Y):
        ty, tx = divmod(t, TILES_X)
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
        "tile_counts": counts.reshape(TILES_Y, TILES_X),
    }


def layer_scene(scene, keep):
    """The scene with the triangles not in `keep` collapsed to their first corner
    (zero area: setup drops them), so the others keep their triangle numbers."""
    indices = scene["indices"].copy()
    indices[~keep] = indices[~keep][:, :1]
    return {**scene, "indices": indices}


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
    """SHADE after two FILLs: the layers of the triangles that are not `masked` and of
    those that are; a pixel takes the masked layer where it is nearer and not a hole.
    Colour (H, W, 3), the taken layer's 1/z and triangle, both layers' tile counts
    added, and the layers' own arrays ("opaque", "masked")."""
    kind = materials["kind"][materials["material"]]
    opaque = rasterize(layer_scene(scene, kind != MASKED))
    masked = rasterize(layer_scene(scene, kind == MASKED))
    prepared = shade_setup(scene, materials)
    opaque_colour, _ = shade_layer(scene, materials, prepared, opaque)
    masked_colour, hole = shade_layer(scene, materials, prepared, masked)
    take = nearer_layer(masked, opaque) & ~hole
    return {
        "colour": np.where(take[:, :, None], masked_colour, opaque_colour),
        "depth": np.where(take, masked["depth"], opaque["depth"]),
        "tri": np.where(take, masked["tri"], opaque["tri"]),
        "tile_counts": opaque["tile_counts"] + masked["tile_counts"],
        "opaque": opaque,
        "masked": masked,
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
    has_specular = (kind == LIT) & (materials["specular"][m].max(axis=1) > 0)
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
    if "sky" in terms:
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
    if "ambient" not in terms:
        ambient = np.zeros(3, dtype=np.float32)
    out = albedo * inv_pi * (((sky + ambient) + sun_e) + points_e) + shine
    emissive = kind == EMISSIVE
    out = np.where(emissive[:, None], albedo * F(lights["emissive_scale"]), out)
    lit_parts = {
        "sky": albedo * inv_pi * sky,
        "sun": albedo * inv_pi * sun_e,
        "points": albedo * inv_pi * points_e,
    }
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
    """The environment's radiance (H, W, 3) in each pixel's view direction: the
    nearest texel of the octahedral map (world directions, y up)."""
    cam = scene["camera"]
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    sx = (xs.ravel().astype(np.float32) + F(0.5) - cam[13]) / cam[12]
    sy = (ys.ravel().astype(np.float32) + F(0.5) - cam[14]) / cam[12]
    d = (cam[9:12] + cam[3:6] * sx[:, None]) - cam[6:9] * sy[:, None]
    octa = lights["octahedral"]
    row, column = light_data.octahedral_texel(d, octa.shape[0])
    return octa[row, column].reshape(HEIGHT, WIDTH, 3)


def rasterize_lighted(scene, materials, lights, smooth=False, pcf=True, terms=TERMS):
    """SHADE's two FILLs and holes with direct light (raster reference README,
    "Lighting"); terms: the subset of TERMS that contributes. "colour" linear,
    "display" x / (1 + x) of exposure x colour, the parts. smooth (vertex normals):
    a comparison variant; pcf=False: one shadow tap."""
    kind = materials["kind"][materials["material"]]
    opaque = rasterize(layer_scene(scene, kind != MASKED))
    masked = rasterize(layer_scene(scene, kind == MASKED))
    corner_vis = lights["visibility"][scene["indices"]]
    prepared = shade_setup(scene, {**materials, "light": corner_vis})
    variant = {"smooth": smooth_setup(scene) if smooth else None, "pcf": pcf}
    variant["terms"] = set(terms)
    unknown = variant["terms"] - set(TERMS)
    if unknown:
        raise ValueError(f"unknown lighting terms {sorted(unknown)}")
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
    x = F(lights["exposure"]) * colour
    parts = {}
    for name in opaque_parts:
        pick = take if opaque_parts[name].ndim == 2 else take[:, :, None]
        parts[name] = np.where(pick, masked_parts[name], opaque_parts[name])
    depth = np.where(take, masked["depth"], opaque["depth"])
    parts["depth"] = depth
    parts["layer"] = np.where(tri < 0, -1, take.astype(np.int32)).astype(np.int8)
    return {
        "colour": colour,
        "display": x / (F(1) + x),
        "depth": depth,
        "tri": tri,
        "parts": parts,
    }
