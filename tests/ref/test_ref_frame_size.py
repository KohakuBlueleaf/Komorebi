"""A frame of any size: the grid rule, the sized camera, the padded frame and its
picture (reference/scenes.py, reference/raster.py)."""

import numpy as np
import pytest

from reference import raster, scenes


def test_grid_rule():
    assert scenes.frame_grid((1024, 1024)) == (32, 32)
    assert scenes.frame_grid((1280, 720)) == (40, 23)
    assert scenes.frame_grid((1, 1)) == (1, 1)
    assert scenes.frame_grid((3840, 2160)) == (120, 68)  # no number of tiles bounds it
    assert scenes.frame_grid((20000, 3)) == (625, 1)
    for size in ((0, 5), (5, 0), (-1, 1)):
        with pytest.raises(ValueError, match=">= 1"):
            scenes.frame_grid(size)


def test_sized_camera():
    camera = scenes.make_preset("tiny")["camera"]
    same = scenes.sized_camera(camera, (1024, 1024))
    assert same.dtype == np.float32 and np.array_equal(
        same.view(np.uint32), camera.view(np.uint32)
    )
    wide = scenes.sized_camera(camera, (1280, 720))
    assert tuple(wide[13:15]) == (640.0, 360.0)
    assert wide[12] == np.float32(np.float64(camera[12]) * 1280 / 1024)
    assert np.array_equal(wide[:12], camera[:12]) and wide[15] == camera[15]


def test_padded_frame_and_picture():
    scene = scenes.make_preset("tiny")
    default = raster.rasterize(scene)
    assert raster.picture(default["colour"], scene).shape[:2] == (1024, 1024)
    view = scenes.with_size(scene, (1280, 720))
    out = raster.rasterize(view)
    assert out["colour"].shape[:2] == (736, 1280)
    assert out["tile_counts"].shape == (23, 40)
    assert raster.picture(out["colour"], view).shape[:2] == (720, 1280)
