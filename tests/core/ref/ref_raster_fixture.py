"""What test_ref_raster.py and other tests share: CAMERA, covered, crossing_scene,
screen_scene."""

import numpy as np

from reference import raster

CAMERA = np.array(
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 512, 512, 512, 0.05], dtype=np.float32
)


def screen_scene(triangles, colors=None):
    """A scene whose vertices project to the given (x, y, z) pixel positions."""
    verts = np.array(triangles, dtype=np.float64).reshape(-1, 3)
    positions = np.stack(
        [
            (verts[:, 0] - 512) * verts[:, 2] / 512,
            (512 - verts[:, 1]) * verts[:, 2] / 512,
            verts[:, 2],
        ],
        axis=1,
    )
    triangle_count = len(verts) // 3
    if colors is None:
        colors = np.full((len(verts), 3), 0.5)
    return {
        "positions": positions.astype(np.float32),
        "colors": np.asarray(colors, dtype=np.float32),
        "indices": np.arange(3 * triangle_count, dtype=np.int32).reshape(
            triangle_count, 3
        ),
        "camera": CAMERA,
    }


def covered(triangles, **kwargs):
    """Boolean image of the pixels one rendering of the triangles covers."""
    return raster.rasterize(screen_scene(triangles, **kwargs))["tri"] >= 0


def crossing_scene():
    """A floor quad from behind the camera to z = 100 and two triangles that cross
    the near plane (one and two vertices behind it), in camera space."""
    positions = [(-100, -1, -10), (100, -1, -10), (100, -1, 100), (-100, -1, 100)]
    positions += [(-0.5, 0.2, 0.01), (0.4, 0.3, 2), (-0.2, 0.6, 3)]
    positions += [(0.1, -0.2, 0.02), (0.6, -0.1, -1), (0.3, -0.5, 4)]
    colors = np.linspace(0.1, 0.9, 3 * len(positions)).reshape(-1, 3)
    return {
        "positions": np.array(positions, dtype=np.float32),
        "colors": colors.astype(np.float32),
        "indices": np.array(
            [[0, 1, 2], [0, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32
        ),
        "camera": CAMERA,
    }
