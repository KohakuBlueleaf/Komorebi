"""Host side of the ray tracer on the card: header constants, tree image, rays split
over cores by tile, packet pages and hit pages."""

import numpy as np
import pytest

from projects.raytrace import packets
from ref import bvh, raytrace, scenes

K = packets.K
VEC = packets.VEC
EVERY = np.arange(scenes.WIDTH * scenes.HEIGHT)


@pytest.fixture(scope="module")
def tiny():
    scene = scenes.make_preset("tiny")
    return scene, bvh.build(scene)


def test_page_sizes_follow_the_vectors():
    assert K["kPacketBytes"] == K["kPkVectors"] * VEC * 4
    assert VEC == 32 and K["kHitAnyVectors"] == len(packets.HIT_KEYS) + 1
    assert K["kHitVectors"] == K["kHitAnyVectors"] + K["kTriAttrWords"]
    assert K["kTriWords"] == K["kTriAttr"] + K["kTriAttrWords"] == 22


def test_constants_are_evaluated_in_order(tmp_path):
    header = tmp_path / "h.h"
    header.write_text(
        "constexpr uint32_t kA = 4;\nconstexpr uint32_t kB = kA * 3 + 1 - 2;\n"
    )
    assert packets.read_constants(header) == {"kA": 4, "kB": 11}
    header.write_text("constexpr uint32_t kA = 0xF | (1 << 4);\n")
    assert packets.read_constants(header) == {"kA": 31}
    header.write_text("constexpr uint32_t kA = 7 % 4;\n")
    with pytest.raises(ValueError):
        packets.read_constants(header)


def test_tree_image_holds_nodes_order_and_triangles(tiny):
    scene, tree = tiny
    image = packets.tree_image(tree, scene)
    assert image.shape[1] == K["kTreePageBytes"] // 4
    w = image.ravel()
    n = len(tree["nodes"])
    assert w[K["kTreeNodes"]] == n
    nodes = w[w[K["kTreeNodeOff"]] :][: n * K["kNodeWords"]]
    assert np.array_equal(nodes.reshape(n, -1), tree["nodes"])
    order = w[w[K["kTreeOrderOff"]] :][:n]
    assert np.array_equal(order, packets.child_order(tree["nodes"]))
    m = len(tree["tris"])
    tris = w[w[K["kTreeTriOff"]] :][: m * K["kTriWords"]].reshape(m, -1)
    assert np.array_equal(tris[:, :9].view(np.float32), tree["tris"])
    assert np.array_equal(tris[:, 9], tree["leaf_tri"])
    attrs = tris[:, K["kTriAttr"] :].view(np.float32)
    colours = scene["colors"][scene["indices"][tree["leaf_tri"]]].reshape(m, 9)
    assert np.array_equal(attrs[:, 3:], colours)
    hit = {"leaf": np.arange(m), "u": np.zeros(m, np.float32), "v": np.zeros(m)}
    n, _ = raytrace.surface(scene, tree, hit, -attrs[:, :3])
    assert np.array_equal(n, attrs[:, :3])  # facing -n: not flipped


def test_child_order_names_the_lower_child(tiny):
    _, tree = tiny
    nodes = tree["nodes"]
    order = packets.child_order(nodes)
    lo, hi = bvh.node_boxes(nodes)
    centre = (lo.astype(np.float64) + hi) / 2
    inner = np.nonzero(~bvh.is_leaf(nodes[:, 6]))[0]
    c0 = nodes[inner, 6].astype(np.int64)
    axis = order[inner] & 3
    flip = (order[inner] & K["kOrderFlip"]) != 0
    rows = np.arange(len(inner))
    lower = np.where(flip, c0 + 1, c0)
    upper = np.where(flip, c0, c0 + 1)
    assert np.all(centre[lower, axis] <= centre[upper, axis])
    assert np.all(order[bvh.is_leaf(nodes[:, 6])] == 0)
    assert len(rows) and np.all(axis < 3)


def check_split(pixels, cores):
    lanes, directory, tiles = packets.split(pixels, cores)
    used = lanes[lanes >= 0]
    assert np.array_equal(np.sort(used), np.arange(len(pixels)))
    first, count = directory[:, K["kDirFirst"]], directory[:, K["kDirCount"]]
    assert np.array_equal(first, np.concatenate([[0], np.cumsum(count)[:-1]]))
    assert count.sum() == len(lanes) == len(tiles)
    _, tile = packets.ray_order(pixels)
    for p in range(len(lanes)):
        rays = lanes[p][lanes[p] >= 0]
        assert len(rays) and np.all(tile[rays] == tiles[p])
    return lanes, count


def test_whole_frame_gives_one_8x4_block_per_packet():
    lanes, count = check_split(EVERY, 119)
    assert len(lanes) == len(EVERY) // VEC and np.all(lanes >= 0)
    x, y = lanes % scenes.WIDTH, lanes // scenes.WIDTH
    assert np.all(np.ptp(x // 8, axis=1) == 0) and np.all(np.ptp(y // 4, axis=1) == 0)
    assert count.max() - count.min() <= 32


def test_subset_pads_packets_and_balances_cores():
    rng = np.random.default_rng(3)
    pixels = np.sort(rng.choice(EVERY, 200_000, replace=False))
    lanes, count = check_split(pixels, 7)
    _, tile = packets.ray_order(pixels)
    per_tile = np.bincount(tile)
    biggest = ((per_tile + VEC - 1) // VEC).max()
    assert count.max() - count.min() <= biggest


def test_more_cores_than_tiles_leaves_cores_empty():
    lanes, count = check_split(EVERY[:64], 5)
    assert len(lanes) == 2 and sorted(count) == [0, 0, 0, 1, 1]  # two tiles


def test_coherence_key_orders_octant_then_morton():
    lo, hi = np.zeros(3), np.full(3, 1024.0)
    o = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2], [5, 5, 5]], float)
    d = np.array([[1, 1, 1]] * 4 + [[-1, 1, 1]], float)
    key = packets.coherence_key(o, d, lo, hi)
    assert key.tolist() == [0, 1, 2, 56, (1 << 30) | 0o707]


@pytest.mark.parametrize("within_tile", [True, False])
def test_sorted_split_keeps_every_ray_once_in_key_order(within_tile):
    rng = np.random.default_rng(5)
    pixels = np.sort(rng.choice(EVERY, 50_000, replace=False))
    key = rng.integers(0, 1 << 33, len(pixels))
    lanes, directory, tiles = packets.split(pixels, 7, None, key, within_tile)
    used = lanes[lanes >= 0]
    assert np.array_equal(np.sort(used), np.arange(len(pixels)))
    assert directory[:, K["kDirCount"]].sum() == len(lanes) == len(tiles)
    _, tile = packets.sorted_order(pixels, key, within_tile)
    for t in np.unique(tiles)[:20]:
        rays = lanes[tiles == t]
        rays = rays[rays >= 0]
        assert np.all(tile[rays] == t) and np.all(np.diff(key[rays]) >= 0)
    if not within_tile:
        assert np.bincount(tile).max() == 32 * 32


def plan_total(per, slots, sizes, tree_pages):
    used = sizes["state"] + sizes["lists"] + sizes["slots"]
    return used + packets.fixed_bytes(tree_pages)


def test_plan_keeps_slots_and_fits_the_budget():
    per, slots, sizes = packets.plan_commands(300, 900, 128 * 1024, 3)
    assert slots == 4 and 16 <= per < 300
    assert plan_total(per, slots, sizes, 3) <= packets.L1_BUDGET
    bigger = packets.treelet_buffers(per + 1, 64 * (per + 1), 900, 4, 128 * 1024)
    assert plan_total(per + 1, 4, bigger, 3) > packets.L1_BUDGET
    assert sizes["entries"] >= 900 and sizes["entries"] % 2 == 0


def test_plan_drops_slots_before_packets_and_lists_hold_every_treelet():
    per, slots, sizes = packets.plan_commands(50, 20_000, 390 * 1024, 2)
    assert slots == 2 and per >= 16 and sizes["entries"] >= 20_000
    assert packets.plan_commands(5, 10, 4096, 1, slots=12)[1] == packets.MAX_SLOTS
    with pytest.raises(ValueError, match="cut smaller"):
        packets.plan_commands(1, 10, 600 * 1024, 1)
    with pytest.raises(ValueError, match="treelets"):
        packets.plan_commands(1, packets.MAX_TREELETS, 4096, 1)


def test_chunks_cover_each_core_range_once():
    directory = np.zeros((4, K["kDirBytes"] // 4), dtype=np.uint32)
    directory[:, K["kDirFirst"]] = [0, 10, 13, 13]
    directory[:, K["kDirCount"]] = [10, 3, 0, 7]
    parts = packets.chunks(directory, 4)
    assert len(parts) == 3
    for core in range(4):
        got = []
        for d in parts:
            f, c = d[core, K["kDirFirst"]], d[core, K["kDirCount"]]
            got += list(range(f, f + c))
        first, count = directory[core, K["kDirFirst"]], directory[core, K["kDirCount"]]
        assert got == list(range(first, first + count))


def test_packet_pages_and_hits_round_trip():
    rng = np.random.default_rng(1)
    n = 100
    o = rng.normal(size=(n, 3)).astype(np.float32)
    d = rng.normal(size=(n, 3)).astype(np.float32)
    d[0] = (-1, 2, -3)
    lanes = np.full((5, VEC), -1, dtype=np.int64)
    lanes.ravel()[:n] = rng.permutation(n)
    pages = packets.packet_pages(lanes, o, d, raytrace.TMAX)
    assert pages.shape == (5, K["kPacketBytes"] // 4)
    pk = pages.reshape(5, K["kPkVectors"], VEC)
    valid = lanes >= 0
    inv = raytrace.inverse_direction(d)
    for k in range(3):
        assert np.array_equal(
            pk[:, K["kPkO"] + k][valid].view(np.float32), o[lanes[valid], k]
        )
        assert np.array_equal(
            pk[:, K["kPkInv"] + k][valid].view(np.float32), inv[lanes[valid], k]
        )
    best = pk[:, K["kPkBest"]].view(np.float32)
    assert np.all(best[valid] == raytrace.TMAX) and np.all(best[~valid] == -1)
    neg = np.where(valid[:, :, None], d[np.maximum(lanes, 0)] < 0, False).sum(axis=1)
    signs = ((2 * neg > valid.sum(axis=1)[:, None]) * [1, 2, 4]).sum(axis=1)
    assert np.array_equal(pk[:, K["kPkHead"], K["kHeadSigns"]], signs)
    pix = packets.packet_pages(lanes, o, d, raytrace.TMAX, np.arange(n) + 7)
    head = pix.reshape(5, K["kPkVectors"], VEC)[:, K["kPkHead"], K["kHeadPixel"]]
    assert np.array_equal(head[:3], lanes[:3, 0] + 7)
    rows = packets.group(pages)
    assert rows.shape == (1, K["kPagePackets"] * K["kPacketBytes"] // 4)
    assert np.array_equal(packets.ungroup(rows, K["kPkVectors"], 5), pages)
    hits = np.zeros((5, K["kHitVectors"], VEC), dtype=np.uint32)
    hits[:, 0] = np.where(valid, lanes, 0).astype(np.float32).view(np.uint32)
    hits[:, 1] = np.where(valid, lanes * 3, 0)
    hits[:, K["kHitN"] + 11] = (
        np.where(valid, lanes, 0).astype(np.float32).view(np.uint32)
    )
    out = packets.decode_hits(hits.reshape(5, -1), lanes, n)
    assert np.array_equal(out["t"], np.arange(n, dtype=np.float32))
    assert np.array_equal(out["tri"], np.arange(n) * 3)
    assert np.array_equal(out["packet_cycles"], np.zeros(5))
    assert np.array_equal(out["attrs"][:, 11], np.arange(n, dtype=np.float32))
    any_hits = hits[:, : K["kHitAnyVectors"]].reshape(5, -1)
    assert "attrs" not in packets.decode_hits(any_hits, lanes, n)


def test_treelet_images_hold_their_part_of_the_tree(tiny):
    _, tree = tiny
    cut = bvh.cut_treelets(tree, 8 * 1024)
    pages, table = packets.treelet_images(tree, cut)
    assert len(table) == len(cut["treelet_root"]) > 1
    assert np.array_equal(table[1:, 0], np.cumsum(table[:-1, 1]))
    members = np.argsort(cut["node_treelet"], kind="stable")
    for t, (first, count) in enumerate(table):
        w = pages[first : first + count].ravel()
        n = w[K["kTreeNodes"]]
        nodes = w[w[K["kTreeNodeOff"]] :][: n * K["kNodeWords"]].reshape(n, -1)
        mine = np.sort(np.nonzero(cut["node_treelet"] == t)[0])
        assert np.array_equal(nodes[:, :6], tree["nodes"][mine, :6])
        leaf = bvh.is_leaf(nodes[:, 6])
        local = nodes[leaf, 6] & ~bvh.LEAF
        recs = w[w[K["kTreeTriOff"]] :][: (local.max() + 1) * K["kTriWords"]]
        index = recs.reshape(-1, K["kTriWords"])[local, 9]
        global_leaf = tree["nodes"][mine[leaf], 6] & ~bvh.LEAF
        assert np.array_equal(index, tree["leaf_tri"][global_leaf])
    assert len(members) == len(tree["nodes"])


def test_top_image_ends_in_one_node_per_treelet(tiny):
    _, tree = tiny
    cut = bvh.cut_treelets(tree, 8 * 1024)
    _, table = packets.treelet_images(tree, cut)
    w = packets.top_image(cut, table).ravel()
    n, k = w[K["kTreeNodes"]], w[K["kTreeTreelets"]]
    nodes = w[w[K["kTreeNodeOff"]] :][: n * K["kNodeWords"]].reshape(n, -1)
    n_top = len(cut["top_nodes"])
    assert k == len(table) and n == n_top + k
    assert np.array_equal(nodes[0, :6], tree["nodes"][0, :6])
    refs = nodes[n_top:, 6]
    assert np.array_equal(refs & ~np.uint32(K["kTreeletBit"]), np.arange(k))
    assert np.array_equal(nodes[n_top:, :6], tree["nodes"][cut["treelet_root"], :6])
    children = nodes[:n_top, 6:8].ravel()
    assert np.array_equal(np.sort(children), np.arange(1, n))
    tab = w[w[K["kTreeTableOff"]] :][: 2 * k].reshape(k, 2)
    assert np.array_equal(tab, table)


def test_split_by_cost_balances_the_cost():
    rng = np.random.default_rng(5)
    cost = rng.gamma(0.5, 1000.0, packets.TILES)
    _, directory, tiles = packets.split(EVERY, 119, cost)
    count = directory[:, K["kDirCount"]]
    first = directory[:, K["kDirFirst"]]
    load = [cost[np.unique(tiles[f : f + c])].sum() for f, c in zip(first, count)]
    assert max(load) - min(load) <= cost.max()
    _, directory, _ = packets.split(EVERY, 119)
    assert directory[:, K["kDirCount"]].max() - directory[:, K["kDirCount"]].min() <= 32


def test_det_range_is_checked(tiny):
    _, tree = tiny
    d = np.array([[0.0, 0.0, 1.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    bound = packets.check_det_range(tree["tris"], d)
    e = np.linalg.norm(tree["tris"][:, 3:6], axis=1) * np.linalg.norm(
        tree["tris"][:, 6:9], axis=1
    )
    assert 5 * e.max() <= bound <= 5.01 * e.max()
    huge = np.array([[2.0**120, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="2\\^103"):
        packets.check_det_range(tree["tris"], huge)


def test_frame_layout_and_constants_describe_whole_blocks(tiny):
    scene, _ = tiny
    pix, directory, desc = packets.frame_layout([0, 33], 4)
    assert pix.shape == (64, VEC) and np.array_equal(desc[:, 0], pix[:, 0])
    assert directory[:2, K["kDirCount"]].tolist() == [32, 32]
    x, y = pix % 1024, pix // 1024
    assert np.array_equal(x - x[:, :1], np.tile(np.arange(VEC) % 8, (64, 1)))
    assert np.array_equal(y - y[:, :1], np.tile(np.arange(VEC) // 8, (64, 1)))
    page = packets.frame_constants(scene).reshape(K["kCnVectors"], VEC)
    assert np.array_equal(page[0, :15].view(np.float32), scene["camera"][:15])
    assert np.array_equal(page[K["kCnPixel"]], pix[0] - pix[0, 0])
    assert page[0, K["kCnWidth"]] == 1024
