"""A binary box tree over triangles (8-word nodes) and its cut into treelets.

Top-down, one triangle per leaf, median split on the longest axis, breadth first. Node:
box min, max (float32 bits), child 0, 1; a leaf: 0x80000000 | leaf in both child words.
"""

from pathlib import Path

import numpy as np

F = np.float32
LEAF = np.uint32(0x80000000)
TREELET = np.uint32(0x40000000)
NODE_BYTES = 32
TRIANGLE_BYTES = 36
KEYS = ("nodes", "tris", "leaf_tri", "leaf_node", "meta")


def triangle_arrays(scene):
    """Vertices (N, 3, 3), v0 / e1 / e2 (N, 9) and centroids (N, 3) of the triangles."""
    p = scene["positions"][scene["indices"]]
    tris = np.concatenate([p[:, 0], p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=1)
    centroid = ((p[:, 0] + p[:, 1]) + p[:, 2]) / F(3)
    return p, tris, centroid


def split_levels(centroid):
    """Levels of (first node, segment starts, counts) and the final triangle order."""
    n = len(centroid)
    perm = np.arange(n)
    start, count, base = np.array([0]), np.array([n]), 0
    levels = []
    while len(start):
        levels.append((base, start, count))
        first = np.cumsum(count) - count
        owner = np.repeat(np.arange(len(start)), count)
        pos = np.repeat(start - first, count) + np.arange(int(count.sum()))
        c = centroid[perm[pos]]
        lo = np.minimum.reduceat(c, first, axis=0)
        hi = np.maximum.reduceat(c, first, axis=0)
        axis = np.argmax(hi - lo, axis=1)
        key = c[np.arange(len(pos)), axis[owner]]
        perm[pos] = perm[pos][np.lexsort((key, owner))]
        inner = count > 1
        left = (count[inner] + 1) // 2
        start = np.stack([start[inner], start[inner] + left], axis=1).ravel()
        count = np.stack([left, count[inner] - left], axis=1).ravel()
        base += len(inner)
    return levels, perm


def build(scene):
    """The tree of a scene as a dict of the arrays named in KEYS."""
    p, tris, centroid = triangle_arrays(scene)
    levels, perm = split_levels(centroid)
    n_nodes = 2 * len(perm) - 1
    box = np.empty((n_nodes, 6), dtype=np.float32)
    nodes = np.empty((n_nodes, 8), dtype=np.uint32)
    leaf_node = np.empty(len(perm), dtype=np.int32)
    first_child = []
    for base, start, count in levels:
        ids = base + np.arange(len(start))
        leaf = count == 1
        t = perm[start[leaf]]
        box[ids[leaf], :3] = p[t].min(axis=1)
        box[ids[leaf], 3:] = p[t].max(axis=1)
        word = LEAF | start[leaf].astype(np.uint32)
        nodes[ids[leaf], 6] = nodes[ids[leaf], 7] = word
        leaf_node[start[leaf]] = ids[leaf]
        next_base = base + len(start)
        first_child.append(next_base + 2 * np.arange(int((~leaf).sum())))
        nodes[ids[~leaf], 6] = first_child[-1]
        nodes[ids[~leaf], 7] = first_child[-1] + 1
    for (base, start, count), child in zip(levels[::-1], first_child[::-1]):
        ids = (base + np.arange(len(start)))[count > 1]
        box[ids, :3] = np.minimum(box[child, :3], box[child + 1, :3])
        box[ids, 3:] = np.maximum(box[child, 3:], box[child + 1, 3:])
    nodes[:, :6] = box.view(np.uint32)
    meta = np.array([n_nodes, len(perm), len(levels) - 1], dtype=np.int32)
    return {
        "nodes": nodes,
        "tris": tris[perm],
        "leaf_tri": perm.astype(np.int32),
        "leaf_node": leaf_node,
        "meta": meta,
    }


def node_boxes(nodes):
    """Box min and max (N, 3) float32 of nodes (N, 8) uint32."""
    box = np.ascontiguousarray(nodes[:, :6]).view(np.float32)
    return box[:, :3], box[:, 3:]


def is_leaf(words):
    """True where a child word marks a leaf."""
    return (words & LEAF) != 0


def levels_of(tree):
    """Node ids by level (breadth first numbering makes each level a range)."""
    nodes, ids, out = tree["nodes"], np.array([0]), []
    while len(ids):
        out.append(ids)
        inner = ids[~is_leaf(nodes[ids, 6])]
        ids = np.stack([nodes[inner, 6], nodes[inner, 7]], axis=1).ravel()
    return out


def cut_treelets(tree, max_bytes):
    """Cuts the tree into treelets of at most max_bytes and a top tree over them: dict
    of treelet_root, node_treelet, treelet_node_off, treelet_leaf_first/off,
    treelet_nodes (local), treelet_tris, top_nodes (child: top node or 0x40000000 |
    treelet) and top_root (the word to start from)."""
    nodes, n_nodes = tree["nodes"], len(tree["nodes"])
    levels = levels_of(tree)
    size = np.zeros(n_nodes, dtype=np.int64)
    leaves = np.zeros(n_nodes, dtype=np.int64)
    for ids in levels[::-1]:
        leaf = is_leaf(nodes[ids, 6])
        size[ids[leaf]], leaves[ids[leaf]] = NODE_BYTES + TRIANGLE_BYTES, 1
        inner = ids[~leaf]
        c0, c1 = nodes[inner, 6], nodes[inner, 7]
        size[inner] = NODE_BYTES + size[c0] + size[c1]
        leaves[inner] = leaves[c0] + leaves[c1]
    top = size > max_bytes
    node_treelet = np.full(n_nodes, -1, dtype=np.int32)
    roots = [0] if not top[0] else []
    for ids in levels:
        inner = ids[~is_leaf(nodes[ids, 6])]
        for c in (nodes[inner, 6], nodes[inner, 7]):
            roots += c[top[inner] & ~top[c]].tolist()
    roots = np.array(sorted(roots), dtype=np.int32)
    node_treelet[roots] = np.arange(len(roots))
    for ids in levels:
        inner = ids[~is_leaf(nodes[ids, 6])]
        inner = inner[node_treelet[inner] >= 0]
        node_treelet[nodes[inner, 6]] = node_treelet[inner]
        node_treelet[nodes[inner, 7]] = node_treelet[inner]
    k = len(roots)
    members = np.nonzero(node_treelet >= 0)[0]
    members = members[np.argsort(node_treelet[members], kind="stable")]
    node_off = np.concatenate(
        [[0], np.cumsum(np.bincount(node_treelet[members], minlength=k))]
    )
    local = np.full(n_nodes, -1, dtype=np.int64)
    local[members] = np.arange(len(members)) - node_off[node_treelet[members]]
    leaf_first = np.array(
        [nodes[leaf_range_node(nodes, r)[0], 6] & ~LEAF for r in roots], dtype=np.int64
    )
    tl_nodes = nodes[members].copy()
    inner = ~is_leaf(tl_nodes[:, 6])
    tl_nodes[inner, 6] = local[tl_nodes[inner, 6]]
    tl_nodes[inner, 7] = local[tl_nodes[inner, 7]]
    word = tl_nodes[~inner, 6] & ~LEAF
    owner = node_treelet[members][~inner]
    tl_nodes[~inner, 6] = tl_nodes[~inner, 7] = LEAF | (
        word - leaf_first[owner]
    ).astype(np.uint32)
    leaf_off = np.concatenate([[0], np.cumsum(leaves[roots])])
    tl_tris = np.concatenate(
        [tree["tris"][a : a + n] for a, n in zip(leaf_first, leaves[roots])]
    )
    top_ids = np.nonzero(node_treelet < 0)[0]
    top_local = np.full(n_nodes, -1, dtype=np.int64)
    top_local[top_ids] = np.arange(len(top_ids))
    top_nodes = nodes[top_ids].copy()
    for col in (6, 7):
        child = top_nodes[:, col].astype(np.int64)
        to_top = node_treelet[child] < 0
        top_nodes[:, col] = np.where(
            to_top, top_local[child], TREELET | node_treelet[child]
        )
    top_root = np.array([0 if len(top_ids) else TREELET], dtype=np.uint32)
    return {
        "treelet_root": roots,
        "node_treelet": node_treelet,
        "treelet_node_off": node_off.astype(np.int32),
        "treelet_leaf_first": leaf_first.astype(np.int32),
        "treelet_leaf_off": leaf_off.astype(np.int32),
        "treelet_nodes": tl_nodes,
        "treelet_tris": tl_tris,
        "top_nodes": top_nodes,
        "top_root": top_root,
    }


def leaf_range_node(nodes, root):
    """The leftmost leaf node under `root` (its child word names the first leaf)."""
    node = int(root)
    while not is_leaf(nodes[node, 6]):
        node = int(nodes[node, 6])
    return node, node


def save_tree(path, tree):
    """Saves a tree as an uncompressed .npz."""
    np.savez(path, **tree)


def load_tree(path):
    """Loads a tree saved by `save_tree`."""
    with np.load(path) as data:
        return {key: data[key] for key in KEYS}


def export_npy(tree, folder):
    """Writes each array of a tree as <folder>/<key>.npy, the input of the C++ code."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for key in KEYS:
        np.save(folder / f"{key}.npy", tree[key])
