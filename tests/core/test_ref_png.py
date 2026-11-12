"""PNG writing and colour maps."""

import numpy as np

from ref import png


def test_round_trip_grey_and_rgb(tmp_path):
    rng = np.random.default_rng(0)
    grey = rng.integers(0, 256, (17, 33), dtype=np.uint8)
    rgb = rng.integers(0, 256, (9, 5, 3), dtype=np.uint8)
    png.write_png(tmp_path / "g.png", grey)
    png.write_png(tmp_path / "c.png", rgb)
    assert np.array_equal(png.read_png(tmp_path / "g.png"), grey)
    assert np.array_equal(png.read_png(tmp_path / "c.png"), rgb)
    assert (tmp_path / "g.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_colormap_ends_and_order():
    rgb = png.colormap(np.array([0.0, 0.5, 1.0]))
    assert rgb[0].tolist() == [68, 1, 84] and rgb[2].tolist() == [253, 231, 37]
    assert rgb.dtype == np.uint8 and rgb[1].tolist() == [33, 145, 140]
    assert png.colormap(np.array([5.0, 5.0])).shape == (2, 3)


def test_srgb_and_downscale():
    assert png.to_srgb8(np.array([0.0, 1.0, 2.0, -1.0])).tolist() == [0, 255, 255, 0]
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert png.downscale(image, 2).tolist() == [[2, 4], [10, 12]]
