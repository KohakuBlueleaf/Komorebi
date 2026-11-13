"""One ray against many boxes: the reference's tree collapsed into wide nodes inside the
treelets, every ray of a trace walked alone through it, and the counts per ray. Host
only (numpy); one JSON file per run (the README has the commands)."""

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from experiments.ray_moves.paths import ray_sets
from projects.raytrace import packets
from reference import bvh, raytrace, scenes

FLOAT = np.float32
TRACES = ("primary", "shadow", "bounce", "shadow2")
ANY_HIT = ("shadow", "shadow2")
SCENES = ("sponza_overview", "bistro_interior_overview", "s1m")
WIDTHS = (8, 16, 32)
ORDERS = ("nearest", "stored")
KINDS = ("box", "triangle")  # entered children taken by entry distance, or as stored
LAYOUTS = {"1x32": 32, "2x16": 16, "4x8": 8}  # rays x children an operation; children
LANES = 32
BUDGET = 128  # KiB: the treelets as the frame cuts them today
CHUNK_RAYS = 1 << 14
STACK = 512
CHECK_RAYS = 4096
CHILD_WORDS = 7  # a child: box min and max, its child word
SEED = 5

SHARED = {}  # arrays the worker processes inherit when a pool forks


def log(message):
    """One progress line with the wall clock."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def collapse(tree, cut, width):
    """Wide nodes of up to `width` children: from each root, the child with the largest
    box surface in the same treelet (or the top tree) is replaced by its two children
    until there are `width`. Returns children (N, width) of binary node ids (-1: none),
    the binary node of each wide node, and the wide node of each binary inner node."""
    nodes = tree["nodes"]
    leaf = bvh.is_leaf(nodes[:, 6])
    first = nodes[:, 6].astype(np.int64)
    second = nodes[:, 7].astype(np.int64)
    low, high = bvh.node_boxes(nodes)
    extent = (high - low).astype(np.float64)
    area = extent[:, 0] * extent[:, 1] + extent[:, 1] * extent[:, 2]
    area += extent[:, 2] * extent[:, 0]
    region = cut["node_treelet"].astype(np.int64)
    roots, parts, binary = np.array([0]), [], []
    while len(roots):
        children = np.full((len(roots), width), -1, dtype=np.int64)
        children[:, 0], children[:, 1] = first[roots], second[roots]
        count = np.full(len(roots), 2)
        rows = np.arange(len(roots))
        for _ in range(width - 2):
            present = children >= 0
            safe = np.where(present, children, 0)
            open_ = present & ~leaf[safe] & (region[safe] == region[roots][:, None])
            score = np.where(open_, area[safe], -1.0)
            pick = np.argmax(score, axis=1)
            grow = open_[rows, pick]
            if not grow.any():
                break
            chosen = children[rows[grow], pick[grow]]
            children[rows[grow], pick[grow]] = first[chosen]
            children[rows[grow], count[grow]] = second[chosen]
            count[grow] += 1
        parts.append(children)
        binary.append(roots)
        inner = children[(children >= 0)]
        roots = inner[~leaf[inner]]
    children = np.concatenate(parts)
    binary = np.concatenate(binary)
    wide_of = np.full(len(nodes), -1, dtype=np.int64)
    wide_of[binary] = np.arange(len(binary))
    return children, binary, wide_of


def wide_tree(tree, cut, width):
    """The wide tree as the walk reads it: per child slot its box, whether it is a leaf,
    and its wide node or leaf number."""
    children, binary, wide_of = collapse(tree, cut, width)
    nodes = tree["nodes"]
    boxes = np.ascontiguousarray(nodes[:, :6]).view(np.float32)
    present = children >= 0
    safe = np.where(present, children, 0)
    is_leaf = bvh.is_leaf(nodes[safe, 6]) & present
    target = np.where(
        is_leaf, (nodes[safe, 6] & ~bvh.LEAF).astype(np.int64), wide_of[safe]
    )
    child_boxes = np.where(present[..., None], boxes[safe], np.nan).astype(np.float32)
    return {
        "children": children,
        "binary": binary,
        "present": present,
        "is_leaf": is_leaf,
        "target": np.where(present, target, -1),
        "boxes": child_boxes,
        "region": cut["node_treelet"][binary].astype(np.int64),
    }


def node_bytes(tree, cut, wide, width):
    """Node bytes of the wide tree against today's binary one, and each treelet's L1
    image (nodes and triangle records, whole pages) against its image today."""
    constants = packets.K
    today_nodes = len(tree["nodes"]) * (constants["kNodeWords"] + 1) * 4
    wide_nodes = len(wide["children"]) * width * CHILD_WORDS * 4
    treelets = len(cut["treelet_root"])
    region = wide["region"]
    per = np.bincount(region[region >= 0], minlength=treelets)
    triangles = np.diff(cut["treelet_leaf_off"]).astype(np.int64)
    binary = np.diff(cut["treelet_node_off"]).astype(np.int64)
    page = constants["kTreePageBytes"]

    def image(node_words):
        words = (
            constants["kTreeHeadWords"]
            + node_words
            + triangles * constants["kTriWords"]
        )
        return -(-words * 4 // page) * page

    wide_image = image(per * width * CHILD_WORDS)
    today_image = image(binary * (constants["kNodeWords"] + 1))
    return {
        "wide_nodes": int(len(wide["children"])),
        "children_per_node": float(wide["present"].sum(axis=1).mean()),
        "node_bytes": int(wide_nodes),
        "today_node_bytes": int(today_nodes),
        "treelets": int(treelets),
        "treelet_image_bytes": int(wide_image.sum()),
        "today_treelet_image_bytes": int(today_image.sum()),
        "largest_treelet_bytes": int(wide_image.max()),
        "today_largest_treelet_bytes": int(today_image.max()),
    }


def triangle_batch(rays, leaves, best, hit, any_hit):
    """Triangle tests of `leaves` (one per pair) for rays `rays`, in the reference's
    order of preference: a nearer t, then the lower triangle on a tie. Updates best and
    hit (leaf, u, v) in place; returns the rays that hit (for any hit)."""
    origins, directions = SHARED["origins"], SHARED["directions"]
    ok, t, u, v = raytrace.triangle_test(
        SHARED["triangles"], leaves, origins[rays], directions[rays]
    )
    leaf_triangle = SHARED["leaf_triangle"]
    if any_hit:
        better = ok & (t < best[rays])
        winners = rays[better]
        best[winners] = -1.0
        hit["leaf"][winners] = leaves[better]
        return winners
    # Pairs in order of t then triangle: the first of each ray is its best.
    order = np.lexsort((leaf_triangle[leaves], t, rays))
    rays, leaves, ok, t, u, v = (x[order] for x in (rays, leaves, ok, t, u, v))
    keep = ok & (t < best[rays])
    same = ok & (t == best[rays]) & (hit["leaf"][rays] >= 0)
    old = leaf_triangle[np.maximum(hit["leaf"][rays], 0)]
    keep |= same & (leaf_triangle[leaves] < old)
    rays, leaves, t, u, v = (x[keep] for x in (rays, leaves, t, u, v))
    _, first = np.unique(rays, return_index=True)
    rays, leaves, t, u, v = (x[first] for x in (rays, leaves, t, u, v))
    best[rays] = t
    hit["leaf"][rays], hit["u"][rays], hit["v"][rays] = leaves, u, v
    return rays


def walk_chunk(bounds):
    """One ray at a time (vectorized over the rays of a chunk) through the wide tree:
    leaves a node enters are tested there, inner children entered are pushed nearest
    on top (or in stored order). Returns the hits and per ray counts."""
    start, stop = bounds
    wide = SHARED["wide"]
    any_hit, order = SHARED["any_hit"], SHARED["order"]
    boxes, present, is_leaf, target = (
        wide["boxes"],
        wide["present"],
        wide["is_leaf"],
        wide["target"],
    )
    width = boxes.shape[1]
    origins = SHARED["origins"][start:stop]
    inverse = SHARED["inverse"][start:stop]
    count = stop - start
    best = SHARED["maximum"][start:stop].copy()
    names = (
        "pops",
        "visits",
        "boxes",
        "entered",
        "pushes",
        "leaf_visits",
        "triangles",
    ) + tuple(f"{kind}_{name}" for kind in ("box", "tri") for name in LAYOUTS)
    counts = {name: np.zeros(count, dtype=np.int64) for name in names}
    stack_node = np.zeros((count, STACK), dtype=np.int64)
    stack_entry = np.zeros((count, STACK), dtype=np.float32)
    depth = np.zeros(count, dtype=np.int64)
    root = SHARED["root_box"]
    entered, entry = box_entry(root[None, :], origins, inverse, best)
    first = np.nonzero(entered)[0]
    stack_entry[first, 0] = entry[first]
    depth[first] = 1
    active = first
    offset = SHARED["offset"]
    while len(active):
        depth[active] -= 1
        top = depth[active]
        counts["pops"][active] += 1
        live = stack_entry[active, top] <= best[active]
        rays = active[live]
        node = stack_node[rays, top[live]]
        children = present[node].sum(axis=1)
        counts["visits"][rays] += 1
        counts["boxes"][rays] += children
        for name, lanes in LAYOUTS.items():
            counts[f"box_{name}"][rays] += -(-children // lanes)
        hit_children, entries = box_entry(
            boxes[node], origins[rays, None, :], inverse[rays, None, :], best[rays]
        )
        hit_children &= present[node]
        counts["entered"][rays] += hit_children.sum(axis=1)
        leaves = hit_children & is_leaf[node]
        leaf_count = leaves.sum(axis=1)
        tested = leaf_count > 0
        counts["leaf_visits"][rays[tested]] += 1
        counts["triangles"][rays] += leaf_count
        for name, lanes in LAYOUTS.items():
            counts[f"tri_{name}"][rays] += -(-leaf_count // lanes)
        pair_row, pair_slot = np.nonzero(leaves)
        if len(pair_row):
            winners = triangle_batch(
                rays[pair_row] + offset,
                target[node[pair_row], pair_slot],
                SHARED["best_all"],
                SHARED["hit_all"],
                any_hit,
            )
            best[:] = SHARED["best_all"][start:stop]
            if any_hit and len(winners):
                depth[winners - offset] = 0
        inner = hit_children & ~is_leaf[node] & (entries <= best[rays][:, None])
        if any_hit:
            inner &= (best[rays] >= 0)[:, None]
        key = np.where(inner, entries, -np.inf)
        if order == "nearest":
            sequence = np.argsort(-key, axis=1, kind="stable")
        else:
            sequence = np.broadcast_to(np.arange(width)[::-1], key.shape)
        sorted_inner = np.take_along_axis(inner, sequence, axis=1)
        sorted_node = np.take_along_axis(target[node], sequence, axis=1)
        sorted_entry = np.take_along_axis(entries, sequence, axis=1)
        if order == "stored":
            # Pushed children in reverse stored order, packed to the front.
            packing = np.argsort(~sorted_inner, axis=1, kind="stable")
            sorted_inner = np.take_along_axis(sorted_inner, packing, axis=1)
            sorted_node = np.take_along_axis(sorted_node, packing, axis=1)
            sorted_entry = np.take_along_axis(sorted_entry, packing, axis=1)
        pushes = inner.sum(axis=1)
        counts["pushes"][rays] += pushes
        if (depth[rays] + pushes).max(initial=0) >= STACK:
            raise RuntimeError("stack bound reached")
        for slot in range(width):
            take = sorted_inner[:, slot]
            if not take.any():
                break
            which = rays[take]
            stack_node[which, depth[which]] = sorted_node[take, slot]
            stack_entry[which, depth[which]] = sorted_entry[take, slot]
            depth[which] += 1
        active = active[depth[active] > 0]
    return start, {name: values for name, values in counts.items()}


def box_entry(boxes, origins, inverse, best):
    """The reference's slab test of boxes (..., 6) against rays: hit flag and entry."""
    low, high = boxes[..., :3], boxes[..., 3:]
    near_low = (low - origins) * inverse
    near_high = (high - origins) * inverse
    near = np.minimum(near_low, near_high)
    far = np.maximum(near_low, near_high)
    entry = np.maximum(
        np.maximum(near[..., 0], near[..., 1]), np.maximum(near[..., 2], FLOAT(0))
    )
    exit_ = np.minimum(np.minimum(far[..., 0], far[..., 1]), far[..., 2])
    exit_ = exit_ * raytrace.SLAB_SCALE
    limit = np.minimum(exit_, np.asarray(best)[..., None] if entry.ndim > 1 else best)
    with np.errstate(invalid="ignore"):
        return entry <= limit, entry


def walk(rays, any_hit, order, workers):
    """Every ray of a trace through the wide tree in SHARED: hits and counts per ray."""
    _, origins, directions, maximum = rays
    SHARED.update(
        origins=np.ascontiguousarray(origins, dtype=np.float32),
        directions=np.ascontiguousarray(directions, dtype=np.float32),
        inverse=raytrace.inverse_direction(np.asarray(directions, dtype=np.float32)),
        maximum=np.asarray(maximum, dtype=np.float32),
        any_hit=any_hit,
        order=order,
    )
    count = len(origins)
    bounds = [(s, min(s + CHUNK_RAYS, count)) for s in range(0, count, CHUNK_RAYS)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        parts = pool.map(walk_single, bounds)
    counts = {
        name: np.concatenate([part[1][name] for part in parts]) for name in parts[0][1]
    }
    hits = {
        name: np.concatenate([part[2][name] for part in parts]) for name in parts[0][2]
    }
    return counts, hits


def walk_single(bounds):
    """walk_chunk with this chunk's own best and hit arrays."""
    start, stop = bounds
    best = SHARED["maximum"].copy()
    hit = {
        "leaf": np.full(len(best), -1, dtype=np.int64),
        "u": np.zeros(len(best), dtype=np.float32),
        "v": np.zeros(len(best), dtype=np.float32),
    }
    SHARED.update(best_all=best, hit_all=hit, offset=start)
    _, counts = walk_chunk(bounds)
    final = {
        "best": best[start:stop],
        "leaf": hit["leaf"][start:stop],
        "u": hit["u"][start:stop],
        "v": hit["v"][start:stop],
    }
    return start, counts, final


def summary(counts, rays):
    """Means per ray, the layouts' vector operations per ray and their lane fill."""
    mean = {name: float(values.sum() / rays) for name, values in counts.items()}
    result = {
        name: round(mean[name], 3)
        for name in (
            "pops",
            "visits",
            "boxes",
            "entered",
            "pushes",
            "leaf_visits",
            "triangles",
        )
    }
    result["pops_p99"] = float(np.percentile(counts["pops"], 99))
    layouts = {}
    for name, lanes in LAYOUTS.items():
        sharing = LANES // lanes
        box_ops = mean[f"box_{name}"] / sharing
        triangle_ops = mean[f"tri_{name}"] / sharing
        layouts[name] = {
            "box_operations": round(box_ops, 3),
            "triangle_operations": round(triangle_ops, 3),
            "box_fill": round(mean["boxes"] / (box_ops * LANES), 3),
            "triangle_fill": round(
                mean["triangles"] / max(triangle_ops * LANES, 1e-9), 3
            ),
        }
    result["layouts"] = layouts
    return result


def check(tree, rays, any_hit, counts_hits, generator):
    """The wide walk's hits against the reference's trace on a sample of rays: how many
    differ in leaf (and t for nearest hits)."""
    _, origins, directions, maximum = rays
    pick = np.sort(generator.choice(len(origins), min(CHECK_RAYS, len(origins)), False))
    tmax = maximum[pick]
    reference = raytrace.trace(tree, origins[pick], directions[pick], tmax[0], any_hit)
    hits = counts_hits[pick]
    if any_hit:
        return int(((reference["leaf"] >= 0) != (hits["leaf"] >= 0)).sum())
    differ = (reference["leaf"] != hits["leaf"]) | (reference["t"] != hits["best"])
    return int(differ.sum())


def tables(report, today):
    """Markdown tables per scene from a report of main(), today's counts beside."""
    lines = []
    for name, scene in report["scenes"].items():
        lines += [
            f"{name}",
            "",
            "| W | trace | pops | node visits | boxes tested | children entered |"
            " pushes | leaf visits | triangle tests | operations a ray, 1 x 32 / 2 x 16"
            " / 4 x 8 (boxes + triangles) | lanes full, 1 x 32 / 2 x 16 / 4 x 8 |"
            " stored order: visits, triangles |",
            "|" + " --- |" * 12,
        ]
        traces = today["scenes"].get(name, {}).get("traces", {})
        for trace, row in traces.items():
            counts = row["counts"]
            lanes = counts["packets"] * LANES
            lines.append(
                f"| today (2) | {trace} | {counts['pops'] / lanes:.2f} |"
                f" {counts['inner'] / lanes:.2f} | {2 * counts['inner'] / lanes:.2f} |"
                f" | | | {counts['leaves'] / lanes:.2f} |"
                f" {(2 * counts['inner'] + counts['leaves']) / lanes:.2f} | | |"
            )
        for width, entry in scene["widths"].items():
            for trace, orders in entry["traces"].items():
                row, stored = orders["nearest"], orders["stored"]
                layouts = row["layouts"]
                operations = " / ".join(
                    f"{sum(layouts[k][f'{kind}_operations'] for kind in KINDS):.1f}"
                    for k in LAYOUTS
                )
                fill = " / ".join(f"{layouts[k]['box_fill']:.2f}" for k in LAYOUTS)
                counts = " | ".join(
                    f"{row[key]:.2f}"
                    for key in (
                        "pops",
                        "visits",
                        "boxes",
                        "entered",
                        "pushes",
                        "leaf_visits",
                        "triangles",
                    )
                )
                lines.append(
                    f"| {width} | {trace} | {counts} | {operations} | {fill} |"
                    f" {stored['visits']:.2f}, {stored['triangles']:.2f} |"
                )
        lines += [
            "",
            "| W | wide nodes | children a node | node MiB, wide / today |"
            " treelet images MiB, wide / today | largest treelet KiB, wide / today |",
            "|" + " --- |" * 6,
        ]
        for width, entry in scene["widths"].items():
            nodes = entry["nodes"]
            mib = 2**20
            lines.append(
                f"| {width} | {nodes['wide_nodes']:,} |"
                f" {nodes['children_per_node']:.2f} |"
                f" {nodes['node_bytes'] / mib:.1f} /"
                f" {nodes['today_node_bytes'] / mib:.1f} |"
                f" {nodes['treelet_image_bytes'] / mib:.1f} /"
                f" {nodes['today_treelet_image_bytes'] / mib:.1f} |"
                f" {nodes['largest_treelet_bytes'] // 1024} /"
                f" {nodes['today_largest_treelet_bytes'] // 1024} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--widths", nargs="+", type=int, default=list(WIDTHS))
    parser.add_argument("--traces", nargs="+", default=list(TRACES))
    parser.add_argument("--orders", nargs="+", default=list(ORDERS))
    parser.add_argument("--data", default=os.environ.get("KR_DATA", "build/data"))
    parser.add_argument("--cache", default="build/ray_moves")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--json", default="wide.json")
    parser.add_argument("--tables", default=None, help="print the tables of a report")
    parser.add_argument("--today", default=None, help="today's counts for the tables")
    arguments = parser.parse_args()
    if arguments.tables:
        today = json.loads(Path(arguments.today).read_text()) if arguments.today else {}
        print(
            tables(
                json.loads(Path(arguments.tables).read_text()), today or {"scenes": {}}
            )
        )
        return
    data = Path(arguments.data)
    report = {
        "budget_kib": BUDGET,
        "load_before": Path("/proc/loadavg").read_text().split()[:3],
        "scenes": {},
    }
    for name in arguments.scenes:
        scene = scenes.load_scene(data / f"{name}.npz")
        tree = bvh.load_tree(data / f"{name}_tree.npz")
        cut = bvh.cut_treelets(tree, BUDGET * 1024)
        sets = ray_sets(name, scene, tree, arguments.cache, arguments.workers)
        entry = report["scenes"].setdefault(name, {"widths": {}})
        boxes = np.ascontiguousarray(tree["nodes"][:, :6]).view(np.float32)
        SHARED.update(
            triangles=tree["tris"],
            leaf_triangle=tree["leaf_tri"],
            root_box=boxes[0],
        )
        for width in arguments.widths:
            began = time.time()
            wide = wide_tree(tree, cut, width)
            SHARED["wide"] = wide
            row = {"nodes": node_bytes(tree, cut, wide, width), "traces": {}}
            log(f"{name} W={width}: {row['nodes']} ({time.time() - began:.0f} s)")
            for trace in arguments.traces:
                any_hit = trace in ANY_HIT
                row["traces"][trace] = {}
                for order in arguments.orders:
                    began = time.time()
                    counts, hits = walk(sets[trace], any_hit, order, arguments.workers)
                    result = summary(counts, len(sets[trace][1]))
                    generator = np.random.default_rng(SEED)
                    result["check_rays"] = min(CHECK_RAYS, len(sets[trace][1]))
                    result["check_differ"] = check(
                        tree, sets[trace], any_hit, HitView(hits), generator
                    )
                    result["seconds"] = round(time.time() - began, 1)
                    row["traces"][trace][order] = result
                    log(
                        f"{name} W={width} {trace} {order}: {result['visits']} visits"
                        f" {result['pops']} pops {result['triangles']} triangles a ray,"
                        f" {result['check_differ']} differ ({result['seconds']} s)"
                    )
                    entry["widths"][str(width)] = row
                    Path(arguments.json).write_text(json.dumps(report))
    report["load_after"] = Path("/proc/loadavg").read_text().split()[:3]
    Path(arguments.json).write_text(json.dumps(report))
    log(f"wrote {arguments.json}")


class HitView:
    """Hits of a trace indexed by a sample of rays."""

    def __init__(self, hits):
        self.hits = hits

    def __getitem__(self, pick):
        return {name: values[pick] for name, values in self.hits.items()}


if __name__ == "__main__":
    main()
