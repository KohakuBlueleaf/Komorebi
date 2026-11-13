"""The box tree and its treelet cut: structure, boxes, balance, checksums."""

import hashlib

import numpy as np
import pytest

from reference import bvh, raytrace, scenes

CHECKSUMS = {
    "tiny": "d59506b2beec7012577a253b4de0b9101e0b0800df798b052f1dbafe403c9c21",
    "s10k": "326522e3cadf832b08e3a77ef559a07d20379efe2e25626f926b5b8eb94158dd",
}


def digest(tree):
    hasher = hashlib.sha256()
    for key in bvh.KEYS:
        hasher.update(np.ascontiguousarray(tree[key]).tobytes())
    return hasher.hexdigest()


def triangles_scene(vertices):
    positions = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    return {
        "positions": positions,
        "indices": np.arange(len(positions), dtype=np.int32).reshape(-1, 3),
    }


@pytest.fixture(scope="module")
def tiny():
    scene = scenes.make_preset("tiny")
    return scene, bvh.build(scene)


def test_pinned_checksums(tiny):
    assert digest(tiny[1]) == CHECKSUMS["tiny"]
    assert digest(bvh.build(scenes.make_preset("s10k"))) == CHECKSUMS["s10k"]


def test_structure(tiny):
    scene, tree = tiny
    nodes, triangle_count = tree["nodes"], len(scene["indices"])
    assert nodes.shape == (2 * triangle_count - 1, 8) and nodes.dtype == np.uint32
    leaf = bvh.is_leaf(nodes[:, 6])
    assert leaf.sum() == triangle_count and (leaf == bvh.is_leaf(nodes[:, 7])).all()
    assert (nodes[leaf, 6] == nodes[leaf, 7]).all()
    assert sorted(nodes[leaf, 6] & ~bvh.LEAF) == list(range(triangle_count))
    inner = np.nonzero(~leaf)[0]
    assert (nodes[inner, 6] > inner).all() and (
        nodes[inner, 7] == nodes[inner, 6] + 1
    ).all()
    children = np.concatenate([nodes[inner, 6], nodes[inner, 7]])
    assert sorted(children) == list(range(1, 2 * triangle_count - 1))
    assert sorted(tree["leaf_tri"]) == list(range(triangle_count))
    assert (
        tree["leaf_node"] == np.nonzero(leaf)[0][np.argsort(nodes[leaf, 6] & ~bvh.LEAF)]
    ).all()
    assert tree["meta"].tolist() == [
        2 * triangle_count - 1,
        triangle_count,
        int(np.ceil(np.log2(triangle_count))),
    ]


def test_boxes_contain_their_triangles_and_children(tiny):
    scene, tree = tiny
    lower, upper = bvh.node_boxes(tree["nodes"])
    nodes = tree["nodes"]
    inner = np.nonzero(~bvh.is_leaf(nodes[:, 6]))[0]
    for col in (6, 7):
        assert (lower[inner] <= lower[nodes[inner, col]]).all()
        assert (upper[inner] >= upper[nodes[inner, col]]).all()
    assert (lower[0] == scene["positions"].min(axis=0)).all()
    assert (upper[0] == scene["positions"].max(axis=0)).all()
    leaf = tree["leaf_node"]
    verts = scene["positions"][scene["indices"][tree["leaf_tri"]]]
    assert (lower[leaf] == verts.min(axis=1)).all() and (
        upper[leaf] == verts.max(axis=1)
    ).all()


def test_triangle_records_are_v0_e1_e2(tiny):
    scene, tree = tiny
    corners = scene["positions"][scene["indices"][tree["leaf_tri"]]]
    assert (tree["tris"][:, 0:3] == corners[:, 0]).all()
    assert (tree["tris"][:, 3:6] == corners[:, 1] - corners[:, 0]).all()
    assert (tree["tris"][:, 6:9] == corners[:, 2] - corners[:, 0]).all()


def test_median_split_on_a_hand_made_row_of_triangles():
    verts = []
    for x in (0, 10, 20, 30):
        verts += [(x, 0, 0), (x + 1, 0, 0), (x, 1, 0)]
    tree = bvh.build(triangles_scene(verts))
    nodes = tree["nodes"]
    assert nodes.shape == (7, 8) and tree["meta"][2] == 2
    assert tree["leaf_tri"].tolist() == [0, 1, 2, 3]
    assert nodes[0, 6:].tolist() == [1, 2] and nodes[1, 6:].tolist() == [3, 4]
    lower, upper = bvh.node_boxes(nodes)
    assert upper[1].tolist() == [11, 1, 0] and lower[2].tolist() == [20, 0, 0]


def test_levels_cover_every_node_once(tiny):
    levels = bvh.levels_of(tiny[1])
    assert sorted(np.concatenate(levels)) == list(range(len(tiny[1]["nodes"])))
    assert len(levels) == tiny[1]["meta"][2] + 1


def test_depth_is_balanced_on_a_large_scene():
    tree = bvh.build(scenes.make_preset("s10k"))
    assert tree["meta"][2] == int(np.ceil(np.log2(tree["meta"][1])))


def test_tree_files_round_trip(tmp_path, tiny):
    bvh.save_tree(tmp_path / "t.npz", tiny[1])
    assert digest(bvh.load_tree(tmp_path / "t.npz")) == digest(tiny[1])
    bvh.export_npy(tiny[1], tmp_path / "npy")
    assert np.array_equal(np.load(tmp_path / "npy" / "nodes.npy"), tiny[1]["nodes"])


def box_of(row):
    return np.ascontiguousarray(row[:6]).view(np.float32)[None]


def nearest(tris, tri_id, origin, direction, visit):
    """Closest hit of one ray by visiting every box it touches, in any order.

    `visit(entry)` returns (node row, children entries or the leaf as (index, id)).
    """
    best, found = raytrace.TMAX, -1
    inv = raytrace.inverse_direction(direction[None])
    stack = visit(None)
    while stack:
        row, kids = visit(stack.pop())
        hit, _ = raytrace.box_test(box_of(row), np.array([0]), origin[None], inv, best)
        if not hit[0]:
            continue
        if kids[0] == "leaf":
            intersects, distance, _, _ = raytrace.triangle_test(
                tris, np.array([kids[1]]), origin[None], direction[None]
            )
            if intersects[0] and (
                distance[0] < best
                or (distance[0] == best and tri_id[kids[1]] < tri_id[found])
            ):
                best, found = distance[0], kids[1]
        else:
            stack += kids[1]
    return best, found


def test_treelet_cut_is_a_partition_that_fits(tiny):
    _, tree = tiny
    for max_bytes in (4096, 20000, 10**9):
        cut = bvh.cut_treelets(tree, max_bytes)
        in_treelet = cut["node_treelet"] >= 0
        assert len(cut["top_nodes"]) == len(tree["nodes"]) - in_treelet.sum()
        assert (
            cut["treelet_node_off"][-1] == in_treelet.sum() == len(cut["treelet_nodes"])
        )
        assert cut["treelet_leaf_off"][-1] == len(cut["treelet_tris"])
        nodes, leaves = np.diff(cut["treelet_node_off"]), np.diff(
            cut["treelet_leaf_off"]
        )
        assert (nodes * 32 + leaves * 36).max() <= max_bytes
        assert (nodes == 2 * leaves - 1).all()
        if max_bytes == 10**9:
            assert len(cut["treelet_root"]) == 1 and len(cut["top_nodes"]) == 0
            assert cut["top_root"][0] == bvh.TREELET


def test_treelets_and_top_tree_walk_to_the_same_hits_as_the_flat_tree(tiny):
    scene, tree = tiny
    cut = bvh.cut_treelets(tree, 6000)
    assert len(cut["treelet_root"]) > 3 and len(cut["top_nodes"]) > 0
    flat_nodes = tree["nodes"]

    def flat(entry):
        if entry is None:
            return [0]
        row = flat_nodes[entry]
        if bvh.is_leaf(row[6]):
            return row, ("leaf", int(row[6] & ~bvh.LEAF))
        return row, ("inner", [int(row[6]), int(row[7])])

    top, top_word = cut["top_nodes"], int(cut["top_root"][0])
    node_off, leaf_off, first = (
        cut["treelet_node_off"],
        cut["treelet_leaf_off"],
        cut["treelet_leaf_first"],
    )

    def as_entry(word):
        if word & int(bvh.TREELET):
            return ("tl", word & ~int(bvh.TREELET), 0)
        return ("top", word)

    def cut_walk(entry):
        if entry is None:
            return [as_entry(top_word)]
        if entry[0] == "top":
            row = top[entry[1]]
            return row, ("inner", [as_entry(int(row[6])), as_entry(int(row[7]))])
        _, treelet, local = entry
        row = cut["treelet_nodes"][node_off[treelet] + local]
        if bvh.is_leaf(row[6]):
            return row, ("leaf", int(leaf_off[treelet] + (row[6] & ~bvh.LEAF)))
        return row, (
            "inner",
            [("tl", treelet, int(row[6])), ("tl", treelet, int(row[7]))],
        )

    pix = np.arange(0, 1024 * 1024, 9973)[:120]
    origin, direction = raytrace.camera_rays(scene["camera"], pix)
    leaf_global = np.concatenate(
        [np.arange(count) + start for count, start in zip(np.diff(leaf_off), first)]
    )
    assert (leaf_global == np.arange(len(tree["tris"]))).all()
    assert np.array_equal(cut["treelet_tris"], tree["tris"])
    hits = 0
    for i in range(len(pix)):
        flat_hit = nearest(
            tree["tris"], tree["leaf_tri"], origin[i], direction[i], flat
        )
        cut_hit = nearest(
            cut["treelet_tris"], tree["leaf_tri"], origin[i], direction[i], cut_walk
        )
        assert flat_hit == cut_hit
        hits += flat_hit[1] >= 0
    assert 10 < hits < 120
