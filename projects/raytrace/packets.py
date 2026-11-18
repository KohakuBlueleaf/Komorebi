"""Host side of the ray tracer on the card: the tree image, rays split over cores and
packed 32 to a packet, and hit pages decoded into per-ray arrays. Pure numpy; the
layouts are those of `kernels/raytrace.h`, whose constants are read from the header.
"""

import ast
import operator
import re
from pathlib import Path

import numpy as np

from reference import bvh, raytrace
from reference.scenes import HEIGHT, WIDTH

HERE = Path(__file__).resolve().parent
HEADER = HERE / "kernels" / "raytrace.h"
CONSTEXPR = re.compile(r"constexpr\s+uint32_t\s+(k\w+)\s*=\s*([^;]+);")


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.floordiv,  # unsigned division in C++
    ast.BitOr: operator.or_,
    ast.LShift: operator.lshift,
}


def evaluate(node, names):
    """An integer expression of literals, earlier names, +, -, *, /, | and <<."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = evaluate(node.left, names), evaluate(node.right, names)
        return OPS[type(node.op)](left, right)
    raise ValueError(f"unsupported constant expression {ast.dump(node)}")


def read_constants(path):
    """Every `constexpr uint32_t kName = <expression of earlier names>;` in a header."""
    constants = {}
    for name, expression in CONSTEXPR.findall(path.read_text()):
        constants[name] = evaluate(ast.parse(expression, mode="eval").body, constants)
    return constants


K = read_constants(HEADER)
VEC = K["kVec"]
TILE = 32
TILES_X = WIDTH // TILE
TILES = TILES_X * TILES_X
BLOCK_W, BLOCK_H = 8, 4  # a packet of primary rays covers 8 x 4 pixels
NEG_ONE = np.float32(-1.0).view(np.uint32)


def bits(values):
    """float32 values as uint32 bit patterns."""
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


def inner_nodes(nodes):
    """True for nodes with two child nodes (not a leaf, not a treelet reference)."""
    words = nodes[:, 6]
    return ~bvh.is_leaf(words) & ((words & K["kTreeletBit"]) == 0)


def child_order(nodes):
    """Order word of every node: the axis along which its children's centres differ
    most, and whether child 1 is the lower one along it (0 for other nodes)."""
    low, high = bvh.node_boxes(nodes)
    centre = (low.astype(np.float64) + high) / 2
    inner = inner_nodes(nodes)
    first_child = nodes[inner, 6].astype(np.int64)
    second_child = nodes[inner, 7].astype(np.int64)
    gap = centre[second_child] - centre[first_child]
    axis = np.argmax(np.abs(gap), axis=1)
    flip = gap[np.arange(len(first_child)), axis] < 0
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
    triangles = tree["tris"][leaves]
    normals = raytrace.cross(triangles[:, 3:6], triangles[:, 6:9])
    normals = normals / np.sqrt(raytrace.dot(normals, normals))[:, None]
    triangle_ids = tree["leaf_tri"][leaves]
    colours = scene["colors"][scene["indices"][triangle_ids]].reshape(
        len(triangle_ids), 9
    )
    return np.concatenate([normals, colours], axis=1).astype(np.float32)


def image_words(
    nodes, triangles=None, triangle_index=None, table=None, attributes=None
):
    """Header, nodes, order words, then triangle records (with their attributes, or
    zeros) or the treelet table."""
    nodes = np.ascontiguousarray(nodes, dtype=np.uint32)
    node_count = len(nodes)
    header = np.zeros(K["kTreeHeadWords"], dtype=np.uint32)
    header[K["kTreeNodes"]] = node_count
    header[K["kTreeNodeOff"]] = K["kTreeHeadWords"]
    header[K["kTreeOrderOff"]] = K["kTreeHeadWords"] + node_count * K["kNodeWords"]
    tail_offset = header[K["kTreeOrderOff"]] + node_count
    parts = [header, nodes.ravel(), child_order(nodes)]
    if triangles is not None:
        records = np.zeros((len(triangles), K["kTriWords"]), dtype=np.uint32)
        records[:, :9] = bits(triangles)
        records[:, K["kTriIndex"]] = np.asarray(triangle_index).astype(np.uint32)
        if attributes is not None:
            records[:, K["kTriAttr"] :] = bits(attributes)
        header[K["kTreeTriOff"]] = tail_offset
        parts.append(records.ravel())
    if table is not None:
        header[K["kTreeTreelets"]] = len(table)
        header[K["kTreeTableOff"]] = tail_offset
        parts.append(np.asarray(table, dtype=np.uint32).ravel())
    return np.concatenate(parts)


def tree_image(tree, scene=None):
    """The tree as the card's L1 image: (pages, 1024) uint32; with the scene, the
    triangles carry their shading attributes."""
    leaves = np.arange(len(tree["tris"]))
    attributes = None if scene is None else tri_attrs(scene, tree, leaves)
    return pages_of(
        image_words(tree["nodes"], tree["tris"], tree["leaf_tri"], None, attributes)
    )


def treelet_images(tree, cut, scene=None):
    """Every treelet as an L1 image, one after the other in whole pages: the pages
    (P, 1024) and a table of (first page, pages) per treelet."""
    node_offsets, leaf_offsets = cut["treelet_node_off"], cut["treelet_leaf_off"]
    images, table, first_page = [], [], 0
    for treelet in range(len(cut["treelet_root"])):
        nodes = cut["treelet_nodes"][node_offsets[treelet] : node_offsets[treelet + 1]]
        triangles = cut["treelet_tris"][
            leaf_offsets[treelet] : leaf_offsets[treelet + 1]
        ]
        first_leaf = cut["treelet_leaf_first"][treelet]
        leaves = np.arange(first_leaf, first_leaf + len(triangles))
        attributes = None if scene is None else tri_attrs(scene, tree, leaves)
        images.append(
            pages_of(
                image_words(
                    nodes, triangles, tree["leaf_tri"][leaves], None, attributes
                )
            )
        )
        table.append((first_page, len(images[-1])))
        first_page += len(images[-1])
    return np.concatenate(images), np.array(table, dtype=np.uint32)


def top_image(cut, table):
    """The top tree as the card's L1 image: its nodes, one node per treelet (the
    treelet root's box), node 0 the root; then the treelet table."""
    top = cut["top_nodes"].astype(np.uint32).copy()
    top_count, treelet_count = len(top), len(cut["treelet_root"])
    is_reference = (top[:, 6:8] & bvh.TREELET) != 0
    top[:, 6:8] = np.where(
        is_reference, top_count + (top[:, 6:8] & ~bvh.TREELET), top[:, 6:8]
    )
    roots = cut["treelet_nodes"][cut["treelet_node_off"][:-1]]
    leaves = np.zeros((treelet_count, K["kNodeWords"]), dtype=np.uint32)
    leaves[:, :6] = roots[:, :6]
    leaves[:, 6] = leaves[:, 7] = K["kTreeletBit"] | np.arange(
        treelet_count, dtype=np.uint32
    )
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
    per_core = [[] for _ in range(cores)]
    for item in np.argsort(-np.asarray(load), kind="stable"):
        core = int(np.argmin(total))
        per_core[core].append(int(item))
        total[core] += load[item]
    return [sorted(items) for items in per_core]


def coherence_key(origins, directions, low, high):
    """Sort key of rays: direction octant, then the Morton code of the origin on a
    1024-cell grid per axis over the box (low, high)."""
    cell = (np.asarray(origins, dtype=np.float64) - low) / np.maximum(high - low, 1e-30)
    cell = np.clip((cell * 1024).astype(np.int64), 0, 1023)
    code = np.zeros(len(cell), dtype=np.int64)
    for bit in range(10):
        for axis in range(3):
            code |= ((cell[:, axis] >> bit) & 1) << (3 * bit + axis)
    octant = ((np.asarray(directions) < 0) * np.array([1, 2, 4])).sum(axis=1)
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
    tiles, first_ray, ray_count = np.unique(
        tile[order], return_index=True, return_counts=True
    )
    tile_packets = (ray_count + VEC - 1) // VEC
    load = tile_packets.astype(np.float64)
    if cost is not None:
        known = np.asarray(cost, dtype=np.float64)[tiles]
        per_packet_cost = known[known > 0].sum() / max(tile_packets[known > 0].sum(), 1)
        load = np.where(known > 0, known, tile_packets * per_packet_cost)
    per_core = assign(load, cores)
    lanes, tile_of = [], []
    directory = np.zeros((cores, K["kDirBytes"] // 4), dtype=np.uint32)
    start = 0
    for core, items in enumerate(per_core):
        core_packets = 0
        for item in items:
            rays = np.full(tile_packets[item] * VEC, -1, dtype=np.int64)
            rays[: ray_count[item]] = order[
                first_ray[item] : first_ray[item] + ray_count[item]
            ]
            lanes.append(rays.reshape(-1, VEC))
            tile_of += [tiles[item]] * int(tile_packets[item])
            core_packets += int(tile_packets[item])
        directory[core, K["kDirFirst"]] = start
        directory[core, K["kDirCount"]] = core_packets
        start += core_packets
    lanes = np.concatenate(lanes) if lanes else np.zeros((0, VEC), dtype=np.int64)
    return lanes, directory, np.asarray(tile_of, dtype=np.int64)


def head_signs(directions, valid):
    """Direction signs of packets (P, 32, 3) over their valid lanes: bit `axis` set
    when more of them have the sign bit of directions[..., axis] set than not (the
    kernels' rule).
    """
    negative = (np.signbit(directions) & valid[:, :, None]).sum(axis=1)
    more = 2 * negative > valid.sum(axis=1)[:, None]
    return (more * np.array([1, 2, 4])).sum(axis=1).astype(np.uint32)


def packet_pages(lanes, origins, directions, tmax, pixels=None):
    """Packet pages (P, kPkVectors * 32) uint32 of rays (origins, directions) placed
    by `lanes`; `pixels` (per ray) gives each header the pixel of lane 0."""
    origins = np.asarray(origins, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    valid = lanes >= 0
    indices = np.maximum(lanes, 0)
    page = np.zeros((len(lanes), K["kPkVectors"], VEC), dtype=np.uint32)
    inverse = raytrace.inverse_direction(directions)
    for axis in range(3):
        page[:, K["kPkO"] + axis] = np.where(valid, bits(origins[indices, axis]), 0)
        page[:, K["kPkD"] + axis] = np.where(valid, bits(directions[indices, axis]), 0)
        page[:, K["kPkInv"] + axis] = np.where(valid, bits(inverse[indices, axis]), 0)
    page[:, K["kPkBest"]] = np.where(valid, bits(np.float32(tmax)), NEG_ONE)
    page[:, K["kPkHead"], K["kHeadSigns"]] = head_signs(directions[indices], valid)
    if pixels is not None:
        page[:, K["kPkHead"], K["kHeadPixel"]] = np.asarray(pixels)[indices[:, 0]]
    return page.reshape(len(lanes), -1)


def group(pages):
    """Per-packet pages (P, W) as DRAM rows of kPagePackets packets (zero padded)."""
    packets_per_row = K["kPagePackets"]
    pages = np.asarray(pages, dtype=np.uint32)
    padding = np.zeros((-len(pages) % packets_per_row, pages.shape[1]), dtype=np.uint32)
    return np.concatenate([pages, padding]).reshape(
        -1, packets_per_row * pages.shape[1]
    )


def ungroup(rows, vectors, count):
    """The first `count` per-packet pages of `vectors` vectors from grouped DRAM
    rows."""
    return np.asarray(rows, dtype=np.uint32).reshape(-1, vectors * VEC)[:count]


def frame_constants(scene):
    """The frame's constant page (1, kCnVectors * 32): camera, light, offset, sky,
    width; the lane ramp of an 8 x 4 packet (dx, dy as floats, pixel offsets)."""
    page = np.zeros((K["kCnVectors"], VEC), dtype=np.uint32)
    camera = scene["camera"]
    if not camera[12] > 0:
        raise ValueError("the camera's scale (word 12) must be positive")
    page[0, K["kCnCamera"] : K["kCnCamera"] + 15] = bits(camera[:15])
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


def frame_layout(tiles, workers, cost=None):
    """Packets of whole tiles over the cores (by packet count, or by `cost`, cycles a
    tile): the pixel of every lane (P, 32), the directory and the descriptor pages."""
    every = np.arange(WIDTH * HEIGHT)
    _, tile = ray_order(every)
    pixels = every if tiles is None else every[np.isin(tile, tiles)]
    lanes, directory, _ = split(pixels, workers, cost=cost)
    pixel_of_lane = pixels[lanes]
    if (lanes < 0).any() or not (
        pixel_of_lane == pixel_of_lane[:, :1] + lane_ramp()
    ).all():
        raise ValueError("packets are not whole 8 x 4 blocks")
    descriptors = np.zeros(
        (len(pixel_of_lane), K["kDescVectors"] * VEC), dtype=np.uint32
    )
    descriptors[:, 0] = pixel_of_lane[:, 0]
    return pixel_of_lane, directory, descriptors


def lane_ramp():
    """Pixel offset of each lane from lane 0."""
    lane = np.arange(VEC)
    return (lane // BLOCK_W) * WIDTH + lane % BLOCK_W


L1_BUDGET = 1100 * 1024  # bytes of circular buffers per core the planner may use
MAX_TREELETS = 0x8000  # treelet numbers are 15 bits in the reader's lists
MIN_SLOTS = 2  # a fetch can overlap the walk of the other slot
MAX_SLOTS = 8  # the reader's slot table


def state_bytes(packets):
    """Bytes of the state buffer: a Dst tile image per packet, then the headers, in
    whole tiles."""
    tile = K["kStTileBytes"]
    total = packets * (tile + 4 * K["kStHeadWords"])
    return max(total + -total % tile, tile)


def treelet_buffers(packets, entries, treelet_count, slots, slot_bytes):
    """Bytes of the state, lists and slot buffers for a command of `packets` packets
    with room for `entries` list entries (the reader's layout after the lists)."""
    lists = 4 * (K["kLsHead"] + 2 * packets) + 2 * entries + 6 * treelet_count
    lists += 2 * max(entries, packets)
    return {
        "state": state_bytes(packets),
        "lists": lists + -lists % 16,
        "slots": slots * slot_bytes,
        "entries": entries,
    }


def sort_bytes(packets):
    """Bytes of the slot buffer SORT and UNSORT use for a core's `packets` packets:
    keys and orders twice, two vectors, the radix counts and the per-packet signs."""
    return (6 * packets * VEC + (1 << K["kSortDigitBits"]) + packets) * 4


def fixed_bytes(tree_pages):
    """Bytes of every buffer of a treelet command other than state, lists and slots."""
    total = K["kInPageBytes"] * K["kInPages"] + K["kOutPageBytes"] * K["kOutPages"]
    total += K["kWorkBytes"] * K["kWorkPages"] + K["kSignalBytes"]
    total += K["kDirReaderBytes"] + K["kDirBytes"]
    return total + tree_pages * K["kTreePageBytes"]


def plan_commands(
    core_packets,
    treelet_count,
    slot_bytes,
    tree_pages,
    slots=4,
    entries_per_packet=64,
    entries=0,
    budget=L1_BUDGET,
    least=16,
):
    """Packets per command, slots and buffer sizes for a core's `core_packets`: the most
    slots (down to MIN_SLOTS) that leave room for `least` packets per command, then the
    most packets that fit. List entries hold one packet's worst case (every treelet), so
    a command admits a packet. Raises when nothing fits."""
    if treelet_count >= MAX_TREELETS:
        raise ValueError(f"{treelet_count} treelets; at most {MAX_TREELETS - 1}")
    free = budget - fixed_bytes(tree_pages)

    def fits(packets_in_command, slot_count):
        list_entries = max(
            entries or entries_per_packet * packets_in_command, treelet_count
        )
        sizes = treelet_buffers(
            packets_in_command,
            list_entries + list_entries % 2,
            treelet_count,
            slot_count,
            slot_bytes,
        )
        return (
            sum(size for name, size in sizes.items() if name != "entries") <= free,
            sizes,
        )

    core_packets = max(core_packets, 1)
    for slot_count in range(min(max(slots, MIN_SLOTS), MAX_SLOTS), MIN_SLOTS - 1, -1):
        if not fits(1, slot_count)[0]:
            continue
        low, high = 1, core_packets
        while low < high:
            middle = (low + high + 1) // 2
            low, high = (
                (middle, high) if fits(middle, slot_count)[0] else (low, middle - 1)
            )
        if low >= min(least, core_packets) or slot_count == MIN_SLOTS:
            return low, slot_count, fits(low, slot_count)[1]
    raise ValueError(
        f"treelets of {slot_bytes} bytes x {MIN_SLOTS} slots and a top image of "
        f"{tree_pages} pages leave no room in {budget} bytes of L1: cut smaller ones"
    )


def chunks(directory, per_command):
    """The directory (cores, 16) split into commands of at most per_command packets per
    core: a list of directories, each giving every core its next chunk."""
    first = directory[:, K["kDirFirst"]].astype(np.int64)
    count = directory[:, K["kDirCount"]].astype(np.int64)
    chunked = []
    for chunk in range(max(1, -(-int(count.max()) // per_command))):
        part = directory.copy()
        part[:, K["kDirFirst"]] = first + chunk * per_command
        part[:, K["kDirCount"]] = np.clip(count - chunk * per_command, 0, per_command)
        chunked.append(part)
    return chunked


DET_LIMIT = 2.0**103  # the card's exact reciprocal holds for |det| below this


def det_bound(triangles, directions):
    """An upper bound on |det| = |e1 . (direction x e2)| over every triangle and ray:
    max |e1| |e2| times max |direction|, with a margin for float32 rounding."""
    triangles = np.asarray(triangles, dtype=np.float64)
    e1 = np.linalg.norm(triangles[:, 3:6], axis=1)
    e2 = np.linalg.norm(triangles[:, 6:9], axis=1)
    norm = np.linalg.norm(np.asarray(directions, dtype=np.float64), axis=1)
    return float((e1 * e2).max(initial=0.0) * norm.max(initial=0.0) * 1.001)


def check_det_range(triangles, directions):
    """Rejects rays and triangles whose determinant could leave the reciprocal's
    exact range."""
    bound = det_bound(triangles, directions)
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


def decode_hits(pages, lanes, ray_count):
    """Per-ray arrays (t, tri, u, v, nodes, leaves; with a nearest hit's pages also
    attrs, (rays, 12)) from hit pages (P, vectors * 32), and the cycles of each packet
    (packet_cycles, per packet)."""
    pages = np.asarray(pages, dtype=np.uint32)
    pages = pages.reshape(len(lanes), pages.shape[1] // VEC, VEC)
    valid = lanes >= 0
    rays = lanes[valid]
    hits = {}
    types = (np.float32, np.int32, np.float32, np.float32, np.int32, np.int32)
    for vector, (key, kind) in enumerate(zip(HIT_KEYS, types)):
        values = np.zeros(ray_count, dtype=kind)
        values[rays] = pages[:, vector][valid].view(kind)
        hits[key] = values
    hits["packet_cycles"] = pages[:, K["kHitCycles"], 0].astype(np.int64)
    if pages.shape[1] == K["kHitVectors"]:
        attributes = np.zeros((ray_count, K["kTriAttrWords"]), dtype=np.float32)
        attribute_pages = pages[:, K["kHitN"] :].transpose(0, 2, 1)
        attributes[rays] = attribute_pages[valid].view(np.float32)
        hits["attrs"] = attributes
    return hits
