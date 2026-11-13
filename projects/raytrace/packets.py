"""Host side of the ray tracer on the card: the tree image, rays split over cores and
packed 32 to a packet, and hit pages decoded into per-ray arrays. Pure numpy; the
layouts are those of `kernels/raytrace.h`, whose constants are read from the header.
"""

import ast
import operator
import re
from pathlib import Path

import numpy as np

from ref import bvh, raytrace
from ref.scenes import HEIGHT, WIDTH

HERE = Path(__file__).resolve().parent
HEADER = HERE / "kernels" / "raytrace.h"
CONSTEXPR = re.compile(r"constexpr\s+uint32_t\s+(k\w+)\s*=\s*([^;]+);")


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.floordiv,  # unsigned division in C++
}


def evaluate(node, names):
    """An integer expression of literals, earlier names, +, -, * and /."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        a, b = evaluate(node.left, names), evaluate(node.right, names)
        return OPS[type(node.op)](a, b)
    raise ValueError(f"unsupported constant expression {ast.dump(node)}")


def read_constants(path):
    """Every `constexpr uint32_t kName = <expression of earlier names>;` in a header."""
    out = {}
    for name, expr in CONSTEXPR.findall(path.read_text()):
        out[name] = evaluate(ast.parse(expr, mode="eval").body, out)
    return out


K = read_constants(HEADER)
VEC = K["kVec"]
TILE = 32
TILES_X = WIDTH // TILE
TILES = TILES_X * TILES_X
BLOCK_W, BLOCK_H = 8, 4  # a packet of primary rays covers 8 x 4 pixels
NEG_ONE = np.float32(-1.0).view(np.uint32)


def bits(x):
    """float32 values as uint32 bit patterns."""
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def inner_nodes(nodes):
    """True for nodes with two child nodes (not a leaf, not a treelet reference)."""
    words = nodes[:, 6]
    return ~bvh.is_leaf(words) & ((words & K["kTreeletBit"]) == 0)


def child_order(nodes):
    """Order word of every node: the axis along which its children's centres differ
    most, and whether child 1 is the lower one along it (0 for other nodes)."""
    lo, hi = bvh.node_boxes(nodes)
    centre = (lo.astype(np.float64) + hi) / 2
    inner = inner_nodes(nodes)
    c0 = nodes[inner, 6].astype(np.int64)
    c1 = nodes[inner, 7].astype(np.int64)
    gap = centre[c1] - centre[c0]
    axis = np.argmax(np.abs(gap), axis=1)
    flip = gap[np.arange(len(c0)), axis] < 0
    order = np.zeros(len(nodes), dtype=np.uint32)
    order[inner] = axis.astype(np.uint32) | np.where(flip, K["kOrderFlip"], 0)
    return order


def pages_of(words):
    """Words padded to whole tree pages: (pages, 1024) uint32."""
    page = K["kTreePageBytes"] // 4
    words = np.concatenate([words, np.zeros(-len(words) % page, dtype=np.uint32)])
    return words.reshape(-1, page)


def tri_attrs(scene, tree, leaves):
    """What shading needs of each leaf's triangle, as float32 (L, 12): the unit normal
    as the reference's surface() computes it, then the three vertex colours."""
    g = tree["tris"][leaves]
    n = raytrace.cross(g[:, 3:6], g[:, 6:9])
    n = n / np.sqrt(raytrace.dot(n, n))[:, None]
    tri = tree["leaf_tri"][leaves]
    colours = scene["colors"][scene["indices"][tri]].reshape(len(tri), 9)
    return np.concatenate([n, colours], axis=1).astype(np.float32)


def image_words(nodes, tris=None, index=None, table=None, attrs=None):
    """Header, nodes, order words, then triangle records (with their attributes, or
    zeros) or the treelet table."""
    nodes = np.ascontiguousarray(nodes, dtype=np.uint32)
    n = len(nodes)
    head = np.zeros(K["kTreeHeadWords"], dtype=np.uint32)
    head[K["kTreeNodes"]] = n
    head[K["kTreeNodeOff"]] = K["kTreeHeadWords"]
    head[K["kTreeOrderOff"]] = K["kTreeHeadWords"] + n * K["kNodeWords"]
    tail_off = head[K["kTreeOrderOff"]] + n
    parts = [head, nodes.ravel(), child_order(nodes)]
    if tris is not None:
        records = np.zeros((len(tris), K["kTriWords"]), dtype=np.uint32)
        records[:, :9] = bits(tris)
        records[:, K["kTriIndex"]] = np.asarray(index).astype(np.uint32)
        if attrs is not None:
            records[:, K["kTriAttr"] :] = bits(attrs)
        head[K["kTreeTriOff"]] = tail_off
        parts.append(records.ravel())
    if table is not None:
        head[K["kTreeTreelets"]] = len(table)
        head[K["kTreeTableOff"]] = tail_off
        parts.append(np.asarray(table, dtype=np.uint32).ravel())
    return np.concatenate(parts)


def tree_image(tree, scene=None):
    """The tree as the card's L1 image: (pages, 1024) uint32; with the scene, the
    triangles carry their shading attributes."""
    leaves = np.arange(len(tree["tris"]))
    attrs = None if scene is None else tri_attrs(scene, tree, leaves)
    return pages_of(
        image_words(tree["nodes"], tree["tris"], tree["leaf_tri"], None, attrs)
    )


def treelet_images(tree, cut, scene=None):
    """Every treelet as an L1 image, one after the other in whole pages: the pages
    (P, 1024) and a table of (first page, pages) per treelet."""
    off, leaf_off = cut["treelet_node_off"], cut["treelet_leaf_off"]
    images, table, first = [], [], 0
    for t in range(len(cut["treelet_root"])):
        nodes = cut["treelet_nodes"][off[t] : off[t + 1]]
        tris = cut["treelet_tris"][leaf_off[t] : leaf_off[t + 1]]
        lead = cut["treelet_leaf_first"][t]
        leaves = np.arange(lead, lead + len(tris))
        attrs = None if scene is None else tri_attrs(scene, tree, leaves)
        images.append(
            pages_of(image_words(nodes, tris, tree["leaf_tri"][leaves], None, attrs))
        )
        table.append((first, len(images[-1])))
        first += len(images[-1])
    return np.concatenate(images), np.array(table, dtype=np.uint32)


def top_image(cut, table):
    """The top tree as the card's L1 image: its nodes, one node per treelet (the
    treelet root's box), node 0 the root; then the treelet table."""
    top = cut["top_nodes"].astype(np.uint32).copy()
    n_top, k = len(top), len(cut["treelet_root"])
    ref = (top[:, 6:8] & bvh.TREELET) != 0
    top[:, 6:8] = np.where(ref, n_top + (top[:, 6:8] & ~bvh.TREELET), top[:, 6:8])
    roots = cut["treelet_nodes"][cut["treelet_node_off"][:-1]]
    leaves = np.zeros((k, K["kNodeWords"]), dtype=np.uint32)
    leaves[:, :6] = roots[:, :6]
    leaves[:, 6] = leaves[:, 7] = K["kTreeletBit"] | np.arange(k, dtype=np.uint32)
    return pages_of(image_words(np.concatenate([top, leaves]), table=table))


def ray_order(pixels):
    """Rays sorted by 32 x 32 tile, then by 8 x 4 block, then by pixel number."""
    x, y = pixels % WIDTH, pixels // WIDTH
    tile = (y // TILE) * TILES_X + x // TILE
    block = ((y % TILE) // BLOCK_H) * (TILE // BLOCK_W) + (x % TILE) // BLOCK_W
    return np.lexsort((pixels, block, tile)), tile


def assign(load, cores):
    """Items to cores, largest first to the least loaded core: a list per core."""
    total = np.zeros(cores)
    out = [[] for _ in range(cores)]
    for i in np.argsort(-np.asarray(load), kind="stable"):
        c = int(np.argmin(total))
        out[c].append(int(i))
        total[c] += load[i]
    return [sorted(items) for items in out]


def coherence_key(o, d, lo, hi):
    """Sort key of rays: direction octant, then the Morton code of the origin on a
    1024-cell grid per axis over the box (lo, hi)."""
    cell = (np.asarray(o, dtype=np.float64) - lo) / np.maximum(hi - lo, 1e-30)
    cell = np.clip((cell * 1024).astype(np.int64), 0, 1023)
    code = np.zeros(len(cell), dtype=np.int64)
    for b in range(10):
        for k in range(3):
            code |= ((cell[:, k] >> b) & 1) << (3 * b + k)
    octant = ((np.asarray(d) < 0) * np.array([1, 2, 4])).sum(axis=1)
    return (octant.astype(np.int64) << 30) | code


def sorted_order(pixels, key, within_tile):
    """Rays ordered by `key` inside each pixel tile, or over the whole pass with every
    TILE * TILE consecutive rays taken as one tile. Returns (order, tile)."""
    pixels = np.asarray(pixels)
    if within_tile:
        _, tile = ray_order(pixels)
        return np.lexsort((pixels, key, tile)), tile
    order = np.lexsort((pixels, key))
    tile = np.empty(len(pixels), dtype=np.int64)
    tile[order] = np.arange(len(pixels)) // (TILE * TILE)
    return order, tile


def split(pixels, cores, cost=None, key=None, within_tile=True):
    """Packets of 32 rays by tile, tiles to cores by packet count or measured cost.
    With `key`, rays are packed in key order (sorted_order). Returns lanes (P, 32) ray
    numbers (-1: empty lane), the directory (cores, 16) and the tile of each packet."""
    pixels = np.asarray(pixels)
    if key is None:
        order, tile = ray_order(pixels)
    else:
        order, tile = sorted_order(pixels, key, within_tile)
    tiles, first, count = np.unique(tile[order], return_index=True, return_counts=True)
    packets = (count + VEC - 1) // VEC
    load = packets.astype(np.float64)
    if cost is not None:
        known = np.asarray(cost, dtype=np.float64)[tiles]
        per = known[known > 0].sum() / max(packets[known > 0].sum(), 1)
        load = np.where(known > 0, known, packets * per)
    per_core = assign(load, cores)
    lanes, tile_of = [], []
    directory = np.zeros((cores, K["kDirBytes"] // 4), dtype=np.uint32)
    start = 0
    for core, items in enumerate(per_core):
        n = 0
        for i in items:
            rays = np.full(packets[i] * VEC, -1, dtype=np.int64)
            rays[: count[i]] = order[first[i] : first[i] + count[i]]
            lanes.append(rays.reshape(-1, VEC))
            tile_of += [tiles[i]] * int(packets[i])
            n += int(packets[i])
        directory[core, K["kDirFirst"]] = start
        directory[core, K["kDirCount"]] = n
        start += n
    lanes = np.concatenate(lanes) if lanes else np.zeros((0, VEC), dtype=np.int64)
    return lanes, directory, np.asarray(tile_of, dtype=np.int64)


def head_signs(d, valid):
    """Direction signs of packets (P, 32, 3) over their valid lanes: bit k set when
    more of them have the sign bit of d[k] set than not (the kernels' rule)."""
    neg = (np.signbit(d) & valid[:, :, None]).sum(axis=1)
    more = 2 * neg > valid.sum(axis=1)[:, None]
    return (more * np.array([1, 2, 4])).sum(axis=1).astype(np.uint32)


def packet_pages(lanes, o, d, tmax, pixels=None):
    """Packet pages (P, kPkVectors * 32) uint32 of rays (o, d) placed by `lanes`;
    `pixels` (per ray) gives each header the pixel of lane 0."""
    o = np.asarray(o, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    valid = lanes >= 0
    idx = np.maximum(lanes, 0)
    page = np.zeros((len(lanes), K["kPkVectors"], VEC), dtype=np.uint32)
    inv = raytrace.inverse_direction(d)
    for k in range(3):
        page[:, K["kPkO"] + k] = np.where(valid, bits(o[idx, k]), 0)
        page[:, K["kPkD"] + k] = np.where(valid, bits(d[idx, k]), 0)
        page[:, K["kPkInv"] + k] = np.where(valid, bits(inv[idx, k]), 0)
    page[:, K["kPkBest"]] = np.where(valid, bits(np.float32(tmax)), NEG_ONE)
    page[:, K["kPkHead"], K["kHeadSigns"]] = head_signs(d[idx], valid)
    if pixels is not None:
        page[:, K["kPkHead"], K["kHeadPixel"]] = np.asarray(pixels)[idx[:, 0]]
    return page.reshape(len(lanes), -1)


def group(pages):
    """Per-packet pages (P, W) as DRAM rows of kPagePackets packets (zero padded)."""
    per = K["kPagePackets"]
    pages = np.asarray(pages, dtype=np.uint32)
    pad = np.zeros((-len(pages) % per, pages.shape[1]), dtype=np.uint32)
    return np.concatenate([pages, pad]).reshape(-1, per * pages.shape[1])


def ungroup(rows, vectors, n):
    """The first n per-packet pages of `vectors` vectors from grouped DRAM rows."""
    return np.asarray(rows, dtype=np.uint32).reshape(-1, vectors * VEC)[:n]


def frame_constants(scene):
    """The frame's constant page (1, kCnVectors * 32): camera, light, offset, sky,
    width; the lane ramp of an 8 x 4 packet (dx, dy as floats, pixel offsets)."""
    page = np.zeros((K["kCnVectors"], VEC), dtype=np.uint32)
    cam = scene["camera"]
    if not cam[12] > 0:
        raise ValueError("the camera's scale (word 12) must be positive")
    page[0, K["kCnCamera"] : K["kCnCamera"] + 15] = bits(cam[:15])
    page[0, K["kCnLight"] : K["kCnLight"] + 3] = bits(scene["light"])
    page[0, K["kCnEps"]] = bits(scene["eps"][:1])[0]
    sky = np.concatenate([raytrace.HORIZON, raytrace.ZENITH - raytrace.HORIZON])
    page[0, K["kCnSky"] : K["kCnSky"] + 6] = bits(sky)
    page[0, K["kCnWidth"]] = WIDTH
    lane = np.arange(VEC)
    page[K["kCnDx"]] = bits((lane % BLOCK_W).astype(np.float32))
    page[K["kCnDy"]] = bits((lane // BLOCK_W).astype(np.float32))
    page[K["kCnPixel"]] = lane_ramp()
    return page.reshape(1, -1)


def frame_layout(tiles, workers):
    """Packets of whole tiles over the cores: the pixel of every lane (P, 32), the
    directory and the descriptor pages (P, 32)."""
    every = np.arange(WIDTH * HEIGHT)
    _, tile = ray_order(every)
    pixels = every if tiles is None else every[np.isin(tile, tiles)]
    lanes, directory, _ = split(pixels, workers)
    pix = pixels[lanes]
    if (lanes < 0).any() or not (pix == pix[:, :1] + lane_ramp()).all():
        raise ValueError("packets are not whole 8 x 4 blocks")
    desc = np.zeros((len(pix), K["kDescVectors"] * VEC), dtype=np.uint32)
    desc[:, 0] = pix[:, 0]
    return pix, directory, desc


def lane_ramp():
    """Pixel offset of each lane from lane 0."""
    lane = np.arange(VEC)
    return (lane // BLOCK_W) * WIDTH + lane % BLOCK_W


L1_BUDGET = 1100 * 1024  # bytes of circular buffers per core the planner may use
MAX_TREELETS = 0x8000  # treelet numbers are 15 bits in the reader's lists
MIN_SLOTS = 2  # a fetch can overlap the walk of the other slot
MAX_SLOTS = 8  # the reader's slot table


def treelet_buffers(packets, entries, n_treelets, slots, slot_bytes):
    """Bytes of the state, lists and slot buffers for a record of `packets` packets
    with room for `entries` list entries (the reader's layout after the lists)."""
    lists = 4 * (K["kLsHead"] + 2 * packets) + 2 * entries + 6 * n_treelets
    lists += 2 * max(entries, packets)
    return {
        "state": max(packets * K["kStWords"] * 4, 64),
        "lists": lists + -lists % 16,
        "slots": slots * slot_bytes,
        "entries": entries,
    }


def fixed_bytes(tree_pages):
    """Bytes of every buffer of a treelet record other than state, lists and slots."""
    total = K["kInPageBytes"] * K["kInPages"] + K["kOutPageBytes"] * K["kOutPages"]
    total += K["kWorkBytes"] * K["kWorkPages"] + K["kSignalBytes"]
    total += K["kDirReaderBytes"] + K["kDirBytes"]
    return total + tree_pages * K["kTreePageBytes"]


def plan_records(
    most,
    n_treelets,
    slot_bytes,
    tree_pages,
    slots=4,
    per_packet=64,
    entries=0,
    budget=L1_BUDGET,
    least=16,
):
    """Packets per record, slots and buffer sizes for a core's `most` packets: the most
    slots (down to MIN_SLOTS) that leave room for `least` packets per record, then the
    most packets that fit. List entries are at least one packet's worst case (every
    treelet), so a record always admits a packet. Raises when nothing fits."""
    if n_treelets >= MAX_TREELETS:
        raise ValueError(f"{n_treelets} treelets; at most {MAX_TREELETS - 1}")
    free = budget - fixed_bytes(tree_pages)

    def fits(b, s):
        e = max(entries or per_packet * b, n_treelets)
        sizes = treelet_buffers(b, e + e % 2, n_treelets, s, slot_bytes)
        return sum(v for k, v in sizes.items() if k != "entries") <= free, sizes

    most = max(most, 1)
    for s in range(min(max(slots, MIN_SLOTS), MAX_SLOTS), MIN_SLOTS - 1, -1):
        if not fits(1, s)[0]:
            continue
        lo, hi = 1, most
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if fits(mid, s)[0] else (lo, mid - 1)
        if lo >= min(least, most) or s == MIN_SLOTS:
            return lo, s, fits(lo, s)[1]
    raise ValueError(
        f"treelets of {slot_bytes} bytes x {MIN_SLOTS} slots and a top image of "
        f"{tree_pages} pages leave no room in {budget} bytes of L1: cut smaller ones"
    )


def chunks(directory, per_record):
    """The directory (cores, 16) split into records of at most per_record packets per
    core: a list of directories, the j-th giving each core its j-th chunk."""
    first = directory[:, K["kDirFirst"]].astype(np.int64)
    count = directory[:, K["kDirCount"]].astype(np.int64)
    out = []
    for j in range(max(1, -(-int(count.max()) // per_record))):
        d = directory.copy()
        d[:, K["kDirFirst"]] = first + j * per_record
        d[:, K["kDirCount"]] = np.clip(count - j * per_record, 0, per_record)
        out.append(d)
    return out


DET_LIMIT = 2.0**103  # the card's exact reciprocal holds for |det| below this


def det_bound(tris, d):
    """An upper bound on |det| = |e1 . (d x e2)| over every triangle and ray:
    max |e1| |e2| times max |d|, with a margin for float32 rounding."""
    tris = np.asarray(tris, dtype=np.float64)
    e1 = np.linalg.norm(tris[:, 3:6], axis=1)
    e2 = np.linalg.norm(tris[:, 6:9], axis=1)
    norm = np.linalg.norm(np.asarray(d, dtype=np.float64), axis=1)
    return float((e1 * e2).max(initial=0.0) * norm.max(initial=0.0) * 1.001)


def check_det_range(tris, d):
    """Refuses rays and triangles whose determinant could leave the reciprocal's
    exact range."""
    bound = det_bound(tris, d)
    if not bound < DET_LIMIT:
        raise ValueError(
            f"|det| may reach {bound:.3g} >= 2^103: the card's reciprocal is exact"
            " only below 2^103; scale the scene down"
        )
    return bound


HIT_KEYS = ("t", "tri", "u", "v", "nodes", "leaves")


def hit_vectors(any_hit):
    """Vectors of a hit page: an any-hit page has no attributes."""
    return K["kHitAnyVectors"] if any_hit else K["kHitVectors"]


def decode_hits(pages, lanes, n_rays):
    """Per-ray arrays (t, tri, u, v, nodes, leaves; with a nearest hit's pages also
    attrs, (n, 12)) from hit pages (P, vectors * 32), and the cycles of each packet
    (packet_cycles, per packet)."""
    pages = np.asarray(pages, dtype=np.uint32)
    pages = pages.reshape(len(lanes), pages.shape[1] // VEC, VEC)
    valid = lanes >= 0
    rays = lanes[valid]
    out = {}
    types = (np.float32, np.int32, np.float32, np.float32, np.int32, np.int32)
    for k, (key, kind) in enumerate(zip(HIT_KEYS, types)):
        values = np.zeros(n_rays, dtype=kind)
        values[rays] = pages[:, k][valid].view(kind)
        out[key] = values
    out["packet_cycles"] = pages[:, K["kHitCycles"], 0].astype(np.int64)
    if pages.shape[1] == K["kHitVectors"]:
        attrs = np.zeros((n_rays, K["kTriAttrWords"]), dtype=np.float32)
        a = pages[:, K["kHitN"] :].transpose(0, 2, 1)
        attrs[rays] = a[valid].view(np.float32)
        out["attrs"] = attrs
    return out
