"""Pixels whose card colour differs from numpy's only by a tie: a primary or bounce ray
whose nearest hit has another triangle at the same distance (within REL), which the
card's walk may keep in its own order. Every triangle is tested against those rays."""

import itertools

import numpy as np

from reference import raytrace

F = np.float32
REL = 1e-5  # a tie: a triangle's distance within REL of the nearest
MOST = 4096  # differing pixels classified at most; the rest count as differing
CHOICES = 8  # tied triangles tried per ray at most


def tied_hits(tree, origin, direction):
    """Every hit of one ray within REL of its nearest, as dicts (leaf, t, u, v);
    empty on a miss."""
    leaves = np.arange(len(tree["tris"]))
    valid, distance, u, v = raytrace.triangle_test(
        tree["tris"], leaves, origin[None], direction[None]
    )
    valid &= distance < raytrace.TMAX
    if not valid.any():
        return []
    near = distance[valid].min()
    pick = np.nonzero(valid & (distance <= near + abs(near) * F(REL)))[0][:CHOICES]
    return [
        {
            "leaf": np.array([leaf], np.int32),
            "t": distance[leaf : leaf + 1],
            "u": u[leaf : leaf + 1],
            "v": v[leaf : leaf + 1],
        }
        for leaf in pick
    ]


def colours(scene, tree, pixel):
    """Every colour the pixel takes over the tied choices of its primary and bounce
    rays (the reference's arithmetic, render_pixels for one pixel)."""
    pixels = np.array([pixel])
    origins, directions = raytrace.camera_rays(scene["camera"], pixels)
    hits = tied_hits(tree, origins[0], directions[0])
    found = [] if hits else [raytrace.sky(directions)[0]]
    for hit in hits:
        points = origins + directions * hit["t"][:, None]
        normals, albedo = raytrace.surface(scene, tree, hit, directions)
        shadow_origins, direct = raytrace.direct_light(scene, tree, points, normals)
        bounce_directions = raytrace.bounce_directions(pixels, normals)
        radiance = [raytrace.sky(bounce_directions)]
        for bounce_hit in tied_hits(tree, shadow_origins[0], bounce_directions[0]):
            bounce_points = (
                shadow_origins + bounce_directions * bounce_hit["t"][:, None]
            )
            bounce_normals, albedo1 = raytrace.surface(
                scene, tree, bounce_hit, bounce_directions
            )
            _, direct1 = raytrace.direct_light(
                scene, tree, bounce_points, bounce_normals
            )
            radiance.append(albedo1 * (raytrace.AMBIENT + direct1)[:, None])
        if len(radiance) > 1:
            radiance = radiance[1:]
        for bounce_radiance in radiance:
            light = (raytrace.AMBIENT + direct)[:, None] + raytrace.GI * bounce_radiance
            found.append((albedo * light)[0])
    return found


def classify(scene, tree, pixels, got):
    """(ties, differs) over pixels whose card colours `got` (R, 3) differ from numpy's:
    ties are those whose colour is one the tied choices give."""
    ties = 0
    for pixel, colour in itertools.islice(zip(pixels, got), MOST):
        want = colour.astype(F).view(np.uint32)
        if any(
            (candidate.astype(F).view(np.uint32) == want).all()
            for candidate in colours(scene, tree, pixel)
        ):
            ties += 1
    return ties, len(pixels) - ties
