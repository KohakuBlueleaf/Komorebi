"""The numpy rasterizer covers a triangulated rectangle exactly once (slow: one
whole-screen rendering per triangle)."""

import importlib.util
import os

import numpy as np

from reference import raster

# The shared fixture by its path: the name `tests` is another package on some paths.
_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "core", "ref", "ref_raster_fixture.py"
)
_SPEC = importlib.util.spec_from_file_location("ref_raster_fixture", _FIXTURE)
fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixture)
covered = fixture.covered


def test_a_triangulated_rectangle_is_covered_once():
    rng = np.random.default_rng(3)
    x_values = np.cumsum(rng.integers(8, 40, 7)) + 100.0
    y_values = np.cumsum(rng.integers(8, 40, 6)) + 200.0
    grid = [[(x, y, 1.0) for x in x_values] for y in y_values]
    for row in grid[1:-1]:
        for column in range(1, len(row) - 1):
            row[column] = (
                row[column][0] + rng.integers(-3, 4) / 8,
                row[column][1] + 0.5,
                1.0,
            )
    tris = []
    for j in range(len(y_values) - 1):
        for i in range(len(x_values) - 1):
            corner_a, corner_b, corner_c, corner_d = (
                grid[j][i],
                grid[j][i + 1],
                grid[j + 1][i],
                grid[j + 1][i + 1],
            )
            tris += [[corner_a, corner_b, corner_d], [corner_a, corner_d, corner_c]]
    total = np.zeros((raster.HEIGHT, raster.WIDTH), dtype=int)
    for triangle in tris:
        total += covered(triangle)
    inside = np.zeros_like(total, dtype=bool)
    inside[
        int(np.ceil(y_values[0] - 0.5)) : int(np.floor(y_values[-1] - 0.5)) + 1,
        int(np.ceil(x_values[0] - 0.5)) : int(np.floor(x_values[-1] - 0.5)) + 1,
    ] = True
    assert total.max() == 1 and (total[inside] == 1).mean() > 0.97
    assert total[~inside].max() <= 1
