"""The numpy rasterizer covers a triangulated rectangle exactly once (slow: one
whole-screen rendering per triangle)."""

import numpy as np

from ref import raster
from tests.core.test_ref_raster import covered


def test_a_triangulated_rectangle_is_covered_once():
    rng = np.random.default_rng(3)
    xs = np.cumsum(rng.integers(8, 40, 7)) + 100.0
    ys = np.cumsum(rng.integers(8, 40, 6)) + 200.0
    grid = [[(x, y, 1.0) for x in xs] for y in ys]
    for row in grid[1:-1]:
        for k in range(1, len(row) - 1):
            row[k] = (row[k][0] + rng.integers(-3, 4) / 8, row[k][1] + 0.5, 1.0)
    tris = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            a, b, c, d = grid[j][i], grid[j][i + 1], grid[j + 1][i], grid[j + 1][i + 1]
            tris += [[a, b, d], [a, d, c]]
    total = np.zeros((raster.HEIGHT, raster.WIDTH), dtype=int)
    for t in tris:
        total += covered(t)
    inside = np.zeros_like(total, dtype=bool)
    inside[
        int(np.ceil(ys[0] - 0.5)) : int(np.floor(ys[-1] - 0.5)) + 1,
        int(np.ceil(xs[0] - 0.5)) : int(np.floor(xs[-1] - 0.5)) + 1,
    ] = True
    assert total.max() == 1 and (total[inside] == 1).mean() > 0.97
    assert total[~inside].max() <= 1
