"""Stress scenes in the reference's layout, built from screen positions: stress-big
(huge, off-screen, slivers, ties), stress-heavy (long lists in two tiles),
stress-degenerate (zero-area, off-screen, behind or across the near distance). Their
reference image is `raster.rasterize`."""

import numpy as np

NAMES = ("stress-big", "stress-heavy", "stress-degenerate")
FOCAL = 900.0
CENTRE = 512.0
NEAR = 0.05


def camera():
    """At the origin, looking along +z; y up, x right."""
    return np.array(
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, FOCAL, CENTRE, CENTRE, NEAR],
        dtype=np.float32,
    )


def world(sx, sy, z):
    """World positions (n, 3) whose projections are near screen (sx, sy) at depth z."""
    sx, sy, z = (np.asarray(v, dtype=np.float64) for v in (sx, sy, z))
    x = (sx - CENTRE) * z / FOCAL
    y = (CENTRE - sy) * z / FOCAL
    return np.stack([x, y, z], axis=-1)


def scene_of(tris, rng):
    """A scene from screen triangles (n, 3, 3): (sx, sy, z) per vertex."""
    tris = np.asarray(tris, dtype=np.float64)
    pos = world(tris[..., 0], tris[..., 1], tris[..., 2]).reshape(-1, 3)
    n = len(tris)
    return {
        "positions": pos.astype(np.float32),
        "colors": rng.random((3 * n, 3)).astype(np.float32),
        "indices": np.arange(3 * n, dtype=np.int32).reshape(n, 3),
        "camera": camera(),
        "light": np.array([0, 10, 0], dtype=np.float32),
        "eps": np.array([1e-4], dtype=np.float32),
        "meta": np.array([n, 3 * n, 0, 0, 0], dtype=np.int32),
    }


def small(rng, n, x0, y0, width, size, z):
    """n triangles of up to `size` pixels inside the square of `width` pixels at
    (x0, y0), vertex depths drawn from the range z."""
    base = np.stack(
        [
            rng.uniform(x0, x0 + width - size, n),
            rng.uniform(y0, y0 + width - size, n),
        ]
    )
    out = np.empty((n, 3, 3))
    for k in range(3):
        out[:, k, 0] = base[0] + rng.uniform(0, size, n)
        out[:, k, 1] = base[1] + rng.uniform(0, size, n)
        out[:, k, 2] = rng.uniform(*z, n)
    return out


def big(rng):
    t = [
        [(-20000, -20000, 9), (30000, -10, 9), (-10, 30000, 9)],  # covers everything
        [(-5000, 512, 5), (5000, 500, 5), (512, 6000, 5)],
        [(0, 0, 4), (1024, 0, 4), (0, 1024, 4)],  # two halves of the screen
        [(1024, 0, 4), (1024, 1024, 4), (0, 1024, 4)],
        [(0, 0, 3), (1024, 0, 3), (0, 1024, 3)],  # the same at another depth
        [(-300, 100.5, 2), (1400, 101.5, 2), (-300, 102.5, 2)],  # sliver across
        [(700.5, -900, 2.5), (701.5, 2000, 2.5), (702.5, -900, 2.5)],
        [(100, 900, 6), (900, 100, 2), (950, 950, 7)],  # slanted depth
        [(100, 900, 6), (900, 100, 2), (950, 950, 7)],  # the same again: tie
        [(-1e5, -1e5, 8), (1e5, -1e5, 8), (0, 1e5, 8)],
    ]
    return np.concatenate(
        [np.array(t, dtype=np.float64), small(rng, 200, 0, 0, 1024, 300, (1, 9))]
    )


def heavy(rng):
    a = small(rng, 40_000, 10 * 32, 12 * 32, 32, 3, (1, 5))
    b = small(rng, 20_000, 20 * 32, 5 * 32, 32, 2, (1, 5))
    c = small(rng, 300, 0, 0, 1024, 100, (2, 6))
    return np.concatenate([a, b, c])


def degenerate(rng):
    t = [
        [(100, 100, 2), (200, 200, 2), (300, 300, 2)],  # collinear
        [(400, 400, 2), (400, 400, 2), (500, 450, 2)],  # repeated vertex
        [(-500, 10, 2), (-100, 10, 2), (-300, 400, 2)],  # left of the screen
        [(1100, 10, 2), (1500, 10, 2), (1300, 400, 2)],  # right
        [(10, -500, 2), (400, -500, 2), (200, -100, 2)],  # above
        [(10, 1100, 2), (400, 1100, 2), (200, 1500, 2)],  # below
        [(300, 300, -2), (600, 300, -2), (450, 600, -2)],  # behind the camera
        [(300, 300, 0.01), (600, 300, 2), (450, 600, 2)],  # across the near distance
        [(10.1, 10.1, 2), (10.4, 10.1, 2), (10.1, 10.4, 2)],  # between pixel centres
        [(20.5, 20.5, 2), (30.5, 20.5, 2), (20.5, 30.5, 2)],  # edges on pixel centres
        [(30.5, 20.5, 2), (30.5, 30.5, 2), (20.5, 30.5, 2)],  # its shared diagonal
        [(-50, 1000, 3), (60, 1000, 3), (5, 1100, 3)],  # cut by two screen edges
        [(1000, -40, 3), (1100, 30, 3), (980, 40, 3)],
    ]
    return np.concatenate(
        [np.array(t, dtype=np.float64), small(rng, 300, 0, 0, 1024, 100, (1, 9))]
    )


def make(name):
    """The scene dict of a stress scene."""
    rng = np.random.default_rng(NAMES.index(name) + 1)
    tris = {"stress-big": big, "stress-heavy": heavy, "stress-degenerate": degenerate}
    return scene_of(tris[name](rng), rng)
