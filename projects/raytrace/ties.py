"""Pixels whose card colour differs from numpy's only by a tie: a primary or bounce ray
whose nearest hit has another triangle at the same distance (within REL), which the
card's walk may keep in its own order. Every triangle is tested against those rays."""

import itertools

import numpy as np

from ref import raytrace

F = np.float32
REL = 1e-5  # a tie: a triangle's distance within REL of the nearest
MOST = 4096  # differing pixels classified at most; the rest count as differing
CHOICES = 8  # tied triangles tried per ray at most


def tied_hits(tree, o, d):
    """Every hit of one ray within REL of its nearest, as dicts (leaf, t, u, v);
    empty on a miss."""
    leaves = np.arange(len(tree["tris"]))
    ok, t, u, v = raytrace.triangle_test(tree["tris"], leaves, o[None], d[None])
    ok &= t < raytrace.TMAX
    if not ok.any():
        return []
    near = t[ok].min()
    pick = np.nonzero(ok & (t <= near + abs(near) * F(REL)))[0][:CHOICES]
    return [
        {
            "leaf": np.array([k], np.int32),
            "t": t[k : k + 1],
            "u": u[k : k + 1],
            "v": v[k : k + 1],
        }
        for k in pick
    ]


def colours(scene, tree, pixel):
    """Every colour the pixel takes over the tied choices of its primary and bounce
    rays (the reference's arithmetic, render_pixels for one pixel)."""
    pix = np.array([pixel])
    o, d = raytrace.camera_rays(scene["camera"], pix)
    hits = tied_hits(tree, o[0], d[0])
    out = [] if hits else [raytrace.sky(d)[0]]
    for hit in hits:
        p = o + d * hit["t"][:, None]
        n, albedo = raytrace.surface(scene, tree, hit, d)
        o2, direct = raytrace.direct_light(scene, tree, p, n)
        db = raytrace.bounce_directions(pix, n)
        radiance = [raytrace.sky(db)]
        for bh in tied_hits(tree, o2[0], db[0]):
            p1 = o2 + db * bh["t"][:, None]
            n1, albedo1 = raytrace.surface(scene, tree, bh, db)
            _, direct1 = raytrace.direct_light(scene, tree, p1, n1)
            radiance.append(albedo1 * (raytrace.AMBIENT + direct1)[:, None])
        if len(radiance) > 1:
            radiance = radiance[1:]
        for r in radiance:
            light = (raytrace.AMBIENT + direct)[:, None] + raytrace.GI * r
            out.append((albedo * light)[0])
    return out


def classify(scene, tree, pixels, got):
    """(ties, differs) over pixels whose card colours `got` (R, 3) differ from numpy's:
    ties are those whose colour is one the tied choices give."""
    ties = 0
    for pixel, c in itertools.islice(zip(pixels, got), MOST):
        want = c.astype(F).view(np.uint32)
        if any(
            (x.astype(F).view(np.uint32) == want).all()
            for x in colours(scene, tree, pixel)
        ):
            ties += 1
    return ties, len(pixels) - ties
