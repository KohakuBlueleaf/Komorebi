"""The image check's ties: a pixel whose colour is the reference's, or one a tied
triangle gives, counts as a tie; any other colour differs."""

import numpy as np
import pytest

from projects.raytrace import ties
from ref import bvh, raytrace, scenes


@pytest.fixture(scope="module")
def tiny():
    scene = scenes.make_preset("tiny")
    return scene, bvh.build(scene)


def test_reference_colours_are_ties_and_others_differ(tiny):
    scene, tree = tiny
    pixels = np.array([0, 512 * 1024 + 512, 700 * 1024 + 300])
    want = raytrace.render(scene, tree, pixels)["image"]
    assert ties.classify(scene, tree, pixels, want) == (3, 0)
    off = want.copy()
    off[1, 0] = np.nextafter(off[1, 0], np.float32(2))
    assert ties.classify(scene, tree, pixels, off) == (2, 1)
