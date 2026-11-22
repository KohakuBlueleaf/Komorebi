"""The surface-area tree: the structure of the median tree, the same nearest hits."""

import numpy as np
import pytest

from projects.raytrace.reference import harness
from reference import bvh, raytrace, scenes

STRIDE = 8


@pytest.fixture(scope="module", params=["tiny", "s10k"])
def trees(request):
    scene = scenes.make_preset(request.param)
    return scene, bvh.build(scene), bvh.build(scene, split="sah")


def test_sah_tree_structure(trees):
    scene, _, tree = trees
    nodes, triangle_count = tree["nodes"], len(scene["indices"])
    assert nodes.shape == (2 * triangle_count - 1, 8)
    assert tree["meta"][0] == 2 * triangle_count - 1
    leaf = bvh.is_leaf(nodes[:, 6])
    assert leaf.sum() == triangle_count and (nodes[leaf, 6] == nodes[leaf, 7]).all()
    assert sorted(nodes[leaf, 6] & ~bvh.LEAF) == list(range(triangle_count))
    assert sorted(tree["leaf_tri"]) == list(range(triangle_count))
    lower, upper = bvh.node_boxes(nodes)
    inner = np.nonzero(~leaf)[0]
    for col in (6, 7):
        assert (lower[inner] <= lower[nodes[inner, col]]).all()
        assert (upper[inner] >= upper[nodes[inner, col]]).all()
    levels = bvh.levels_of(tree)
    assert len(levels) == tree["meta"][2] + 1 and tree["meta"][2] < 64


def test_sah_tree_gives_the_median_trees_nearest_hits(trees):
    scene, median, sah = trees
    o, d = raytrace.camera_rays(scene["camera"], harness.lattice(STRIDE))
    first = raytrace.trace(median, o, d, raytrace.TMAX)
    second = raytrace.trace(sah, o, d, raytrace.TMAX)
    assert np.array_equal(first["t"], second["t"])
    tri_first = np.where(first["leaf"] >= 0, median["leaf_tri"][first["leaf"]], -1)
    tri_second = np.where(second["leaf"] >= 0, sah["leaf_tri"][second["leaf"]], -1)
    assert (tri_first >= 0).any() and (tri_first < 0).any()
    differ = np.nonzero(tri_first != tri_second)[0]
    # where the triangles differ, both lie at the nearest distance: a tie of the two
    # nearest distances
    for tree, tri in ((median, tri_first), (sah, tri_second)):
        leaf = np.argsort(tree["leaf_tri"])[tri[differ]]
        ok, t, _, _ = raytrace.triangle_test(tree["tris"], leaf, o[differ], d[differ])
        assert ok.all() and np.array_equal(t, first["t"][differ])
