"""Rasterization in numpy, float32: the definition the C++ and CUDA versions match.

Per triangle: project the three vertices (a triangle with a vertex nearer than the
near distance is dropped), orient it, build edge and attribute planes and its pixel and
tile boxes. Binning: a triangle goes to every 32 x 32 tile its pixel box touches, in
triangle order. Fill: a pixel centre is inside when every edge value is > 0, or == 0
on a top or left edge; the fragment with the larger 1/z wins, a tie goes to the lower
triangle index. Colour is interpolated linearly in screen space. All arithmetic is
float32 in the order written here, with no fused multiply-add.
"""

import numpy as np

from ref.scenes import HEIGHT, WIDTH

TILE = 32
TILES_X = WIDTH // TILE
TILES_Y = HEIGHT // TILE
BACKGROUND = np.array([0.05, 0.05, 0.08], dtype=np.float32)
CHUNK = 128
F = np.float32


def project(scene):
    """Screen x, y, 1/z per vertex (float32) and the mask of vertices to drop."""
    cam = scene["camera"]
    d = scene["positions"] - cam[0:3]
    xc = (d[:, 0] * cam[3] + d[:, 1] * cam[4]) + d[:, 2] * cam[5]
    yc = (d[:, 0] * cam[6] + d[:, 1] * cam[7]) + d[:, 2] * cam[8]
    zc = (d[:, 0] * cam[9] + d[:, 1] * cam[10]) + d[:, 2] * cam[11]
    drop = zc <= cam[15]
    inv = F(1.0) / np.where(drop, F(1.0), zc)
    sx = cam[13] + cam[12] * (xc * inv)
    sy = cam[14] - cam[12] * (yc * inv)
    return sx, sy, inv, drop


def planes(values, coef, inv_area):
    """The plane of per-vertex values (T, 3) over coefficients coef (T, 3)."""
    s = (coef[:, 0] * values[:, 0] + coef[:, 1] * values[:, 1]) + coef[:, 2] * values[
        :, 2
    ]
    return s * inv_area


def setup(scene):
    """Per-triangle records of the triangles that can cover a pixel (dict of arrays)."""
    sx, sy, w, drop = project(scene)
    idx = scene["indices"]
    area2 = None
    x, y, ww = sx[idx], sy[idx], w[idx]
    col = scene["colors"][idx]
    cand = ~drop[idx].any(axis=1)
    area2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
    area2 = area2 - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    cand &= area2 != 0
    flip = area2 < 0
    order = np.where(flip[:, None], [0, 2, 1], [0, 1, 2])
    rows = np.arange(len(idx))[:, None]
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
    return {
        "tri": keep.astype(np.int32),
        "a": a_coef,
        "b": b_coef,
        "c": c_coef,
        "top_left": top_left,
        "w_planes": wp,
        "c_planes": cp,
        "box": np.stack([ix0, ix1, iy0, iy1], axis=1).astype(np.int32),
    }


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
