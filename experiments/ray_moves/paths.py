"""The rays' paths through resident treelets: every ray of the four traces through the
top tree and its treelets, the owners, the bundles and the lanes of the owners' walks.
Host only (numpy); one JSON file per run (the README has the commands)."""

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from projects.raytrace import packets
from reference import bvh, raytrace, scenes

FLOAT = np.float32
INFINITE = FLOAT(np.inf)
TRACES = ("primary", "shadow", "bounce", "shadow2")
ANY_HIT = ("shadow", "shadow2")
SCENES = ("s100k", "s1m", "sponza_overview", "bistro_interior_overview")
BUDGETS = (128, 64, 32)  # KiB; 128 first: its traces make the later traces' rays
BUNDLE_RAYS = (32, 64, 128)
RULES = ("full", "each")  # full bins until the input is done; every bin per input
CORES = 118  # the grid's 120 cores less the two counter tiles
LANES = 32
STACK = 64
RECORD_BYTES = 48  # origin, direction, best, pixel, triangle, u, v, progress
BUNDLE_HEADER_BYTES = 16
L1_ROOM_BYTES = 1 << 20  # L1 a core gives to resident treelets (assumption)
MOST_OWNERS = 8
CHUNK_RAYS = 1 << 15
SAMPLE_PACKETS = 16384  # packets walked per configuration, stratified by treelet
SAMPLE_PER_TREELET = 64
CALIBRATION_PACKETS = 1024
SORT_CELLS = 512  # the card's SORT: origin cells per axis
SEED = 11
# Cycles of the walk's steps on main (the card's profile of a Sponza frame): a pop,
# an inner node (two box tests), a leaf (nearest, any hit), a packet state in and out.
POP_CYCLES = 36
INNER_CYCLES = 362
LEAF_CYCLES = {False: 737, True: 485}
STATE_CYCLES = 250

SHARED = {}  # arrays the worker processes inherit when a pool forks


def log(message):
    """One progress line with the wall clock."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def pool(workers):
    """A pool of forked workers that see SHARED as it is now."""
    return multiprocessing.get_context("fork").Pool(workers)


def box_entry(low, high, origins, inverse, best):
    """Hit flag and entry distance of boxes against rays: the reference's slab test."""
    near_low = (low - origins) * inverse
    near_high = (high - origins) * inverse
    near = np.minimum(near_low, near_high)
    far = np.maximum(near_low, near_high)
    entry = np.maximum(
        np.maximum(near[..., 0], near[..., 1]), np.maximum(near[..., 2], FLOAT(0))
    )
    exit_distance = (
        np.minimum(np.minimum(far[..., 0], far[..., 1]), far[..., 2])
        * raytrace.SLAB_SCALE
    )
    return entry <= np.minimum(exit_distance, best), entry


def use_tree(tree, cut):
    """The tree and its cut as the walks below read them (SHARED)."""
    SHARED.update(
        nodes=tree["nodes"],
        boxes=np.ascontiguousarray(tree["nodes"][:, :6]).view(np.float32),
        triangles=tree["tris"],
        leaf_triangle=tree["leaf_tri"],
        order=packets.child_order(tree["nodes"]).astype(np.int64),
        node_treelet=cut["node_treelet"].astype(np.int64),
        treelet_root=cut["treelet_root"].astype(np.int64),
        treelets=len(cut["treelet_root"]),
    )


def use_rays(origins, directions, maximum, any_hit):
    """The rays of one trace as the walks below read them (SHARED)."""
    SHARED.update(
        origins=np.ascontiguousarray(origins, dtype=np.float32),
        directions=np.ascontiguousarray(directions, dtype=np.float32),
        inverse=raytrace.inverse_direction(np.asarray(directions, dtype=np.float32)),
        maximum=np.asarray(maximum, dtype=np.float32),
        any_hit=any_hit,
    )


def walk_from(rays, start, state, any_hit):
    """The reference's one-ray walk of the subtree under `start`, from each ray's
    state (best, leaf, u, v; changed in place). Returns whether a hit was found."""
    nodes, boxes, triangles = SHARED["nodes"], SHARED["boxes"], SHARED["triangles"]
    leaf_triangle = SHARED["leaf_triangle"]
    origins, directions = SHARED["origins"], SHARED["directions"]
    inverse = SHARED["inverse"]
    best, leaf, u, v = state
    count = len(rays)
    found = np.zeros(count, dtype=bool)
    stack_node = np.zeros((count, STACK), dtype=np.int64)
    stack_entry = np.zeros((count, STACK), dtype=np.float32)
    depth = np.zeros(count, dtype=np.int64)
    hit, entry = box_entry(
        boxes[start, :3], boxes[start, 3:], origins[rays], inverse[rays], best
    )
    stack_node[:, 0] = start
    stack_entry[:, 0] = entry
    depth[hit] = 1
    active = np.nonzero(hit)[0]
    while len(active):
        depth[active] -= 1
        top = depth[active]
        live = stack_entry[active, top] <= best[active]
        node = stack_node[active, top][live]
        lane = active[live]
        word = nodes[node, 6]
        is_leaf = bvh.is_leaf(word)
        tested = lane[is_leaf]
        if len(tested):
            index = (word[is_leaf] & ~bvh.LEAF).astype(np.int64)
            ray = rays[tested]
            ok, distance, hit_u, hit_v = raytrace.triangle_test(
                triangles, index, origins[ray], directions[ray]
            )
            better = ok & (distance < best[tested])
            if not any_hit:
                same = ok & (distance == best[tested]) & (leaf[tested] >= 0)
                old = leaf_triangle[np.maximum(leaf[tested], 0)]
                better |= same & (leaf_triangle[index] < old)
            winners = tested[better]
            if any_hit:
                depth[winners] = 0
                found[winners] = True
            else:
                best[winners] = distance[better]
            leaf[winners] = index[better]
            u[winners], v[winners] = hit_u[better], hit_v[better]
        inner, inner_node = lane[~is_leaf], node[~is_leaf]
        if len(inner):
            ray = rays[inner]
            children = [nodes[inner_node, column].astype(np.int64) for column in (6, 7)]
            tests = [
                box_entry(
                    boxes[child, :3],
                    boxes[child, 3:],
                    origins[ray],
                    inverse[ray],
                    best[inner],
                )
                for child in children
            ]
            (hit_first, entry_first), (hit_second, entry_second) = tests
            second_nearer = hit_second & (~hit_first | (entry_second < entry_first))
            far = np.where(second_nearer, children[0], children[1])
            far_entry = np.where(second_nearer, entry_first, entry_second)
            near = np.where(second_nearer, children[1], children[0])
            near_entry = np.where(second_nearer, entry_second, entry_first)
            both = hit_first & hit_second
            either = hit_first | hit_second
            for pushed, push_node, push_entry in (
                (both, far, far_entry),
                (either, near, near_entry),
            ):
                target = inner[pushed]
                stack_node[target, depth[target]] = push_node[pushed]
                stack_entry[target, depth[target]] = push_entry[pushed]
                depth[target] += 1
        active = active[depth[active] > 0]
    return found


def entered_treelets(rays):
    """Every treelet whose root box each ray enters through the top tree, tested
    against the ray's tmax: (ray position, treelet, entry distance)."""
    boxes, nodes = SHARED["boxes"], SHARED["nodes"]
    node_treelet = SHARED["node_treelet"]
    origins, inverse = SHARED["origins"][rays], SHARED["inverse"][rays]
    maximum = SHARED["maximum"][rays]
    position = np.arange(len(rays))
    node = np.zeros(len(rays), dtype=np.int64)
    found = []
    while len(position):
        hit, entry = box_entry(
            boxes[node, :3],
            boxes[node, 3:],
            origins[position],
            inverse[position],
            maximum[position],
        )
        position, node, entry = position[hit], node[hit], entry[hit]
        treelet = node_treelet[node]
        root = treelet >= 0
        found.append((position[root], treelet[root], entry[root]))
        position, node = np.repeat(position[~root], 2), node[~root]
        node = np.stack([nodes[node, 6], nodes[node, 7]], axis=1).ravel()
        node = node.astype(np.int64)
    return tuple(np.concatenate(part) for part in zip(*found))


def follow_chunk(bounds):
    """The design's walk of rays [first, last): the top tree at home, then the
    entered treelets nearest first while the entry is within the ray's best."""
    first, last = bounds
    any_hit = SHARED["any_hit"]
    rays = np.arange(first, last)
    count = len(rays)
    position, treelet, entry = entered_treelets(rays)
    order = np.lexsort((treelet, entry, position))
    position, treelet, entry = position[order], treelet[order], entry[order]
    entered = np.bincount(position, minlength=count)
    start = np.concatenate([[0], np.cumsum(entered)[:-1]])
    best = SHARED["maximum"][rays].copy()
    leaf = np.full(count, -1, dtype=np.int64)
    u = np.zeros(count, dtype=np.float32)
    v = np.zeros(count, dtype=np.float32)
    finished = np.zeros(count, dtype=bool)
    step = np.zeros(count, dtype=np.int64)
    record = {"ray": [], "treelet": [], "best": [], "leaf": []}
    active = np.nonzero(entered > 0)[0]
    while len(active):
        index = start[active] + step[active]
        go = entry[index] <= best[active]
        active, index = active[go], index[go]
        if not len(active):
            break
        record["ray"].append(rays[active])
        record["treelet"].append(treelet[index])
        record["best"].append(best[active].copy())
        record["leaf"].append(leaf[active].copy())
        state = (best[active], leaf[active], u[active], v[active])
        found = walk_from(
            rays[active], SHARED["treelet_root"][treelet[index]], state, any_hit
        )
        best[active], leaf[active], u[active], v[active] = state
        finished[active] |= found
        step[active] += 1
        active = active[(step[active] < entered[active]) & ~finished[active]]
    visits = {
        key: np.concatenate(value) if value else np.zeros(0, dtype=np.int64)
        for key, value in record.items()
    }
    return {
        "entered": entered,
        "best": best,
        "leaf": leaf,
        "u": u,
        "v": v,
        "found": finished,
        **{f"visit_{key}": value for key, value in visits.items()},
    }


def follow(rays, any_hit, workers):
    """follow_chunk over every ray in a pool: each ray's final state, and its visits
    ordered by ray, then by step."""
    use_rays(*rays, any_hit)
    total = len(rays[0])
    bounds = [
        (first, min(first + CHUNK_RAYS, total)) for first in range(0, total, CHUNK_RAYS)
    ]
    with pool(workers) as workers_pool:
        parts = workers_pool.map(follow_chunk, bounds, chunksize=1)
    joined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(joined["visit_ray"], kind="stable")
    final = {
        key: value for key, value in joined.items() if not key.startswith("visit_")
    }
    visits = {
        key[6:]: value[order]
        for key, value in joined.items()
        if key.startswith("visit_")
    }
    return final, visits


def next_rays(scene, tree, rays, result):
    """Shadow and bounce rays at the hits of `result`, as the reference makes them:
    (pixels, origins, directions, tmax) each."""
    pixels, origins, directions, _ = rays
    hit = np.nonzero(result["leaf"] >= 0)[0]
    sub = {"leaf": result["leaf"][hit], "u": result["u"][hit], "v": result["v"][hit]}
    points = origins[hit] + directions[hit] * result["best"][hit][:, None]
    normals, _ = raytrace.surface(scene, tree, sub, directions[hit])
    above = points + normals * scene["eps"][0]
    to_light = scene["light"] - above
    shadow = (pixels[hit], above, to_light, np.ones(len(hit), dtype=np.float32))
    bounce_directions = raytrace.bounce_directions(pixels[hit], normals)
    maximum = np.full(len(hit), raytrace.TMAX, dtype=np.float32)
    return shadow, (pixels[hit], above, bounce_directions, maximum)


def ray_sets(name, scene, tree, cache, workers):
    """The four traces' rays of a scene (cached as .npz in `cache`), made by following
    the nearest-hit traces through the treelets of the cut in use."""
    path = Path(cache) / f"{name}_rays.npz"
    if path.exists():
        with np.load(path) as data:
            return {
                trace: tuple(data[f"{trace}_{part}"] for part in range(4))
                for trace in TRACES
            }
    pixels = np.arange(scenes.WIDTH * scenes.HEIGHT)
    origins, directions = raytrace.camera_rays(scene["camera"], pixels)
    maximum = np.full(len(pixels), raytrace.TMAX, dtype=np.float32)
    sets = {"primary": (pixels, origins, directions, maximum)}
    log(f"{name}: primary rays, for the next traces' rays")
    primary, _ = follow(sets["primary"][1:], False, workers)
    sets["shadow"], sets["bounce"] = next_rays(scene, tree, sets["primary"], primary)
    log(f"{name}: bounce rays, for the second shadow rays")
    bounce, _ = follow(sets["bounce"][1:], False, workers)
    sets["shadow2"], _ = next_rays(scene, tree, sets["bounce"], bounce)
    Path(cache).mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{trace}_{part}": sets[trace][part] for trace in TRACES for part in range(4)
    }
    np.savez(path, **arrays)
    return sets


def treelet_bytes(cut):
    """L1 image bytes of every treelet as the card lays it out: header, nodes, order
    words, triangle records with their shading attributes, in whole tree pages."""
    constants = packets.K
    nodes = np.diff(cut["treelet_node_off"]).astype(np.int64)
    triangles = np.diff(cut["treelet_leaf_off"]).astype(np.int64)
    words = (
        constants["kTreeHeadWords"]
        + nodes * (constants["kNodeWords"] + 1)
        + triangles * constants["kTriWords"]
    )
    page = constants["kTreePageBytes"]
    return -(-words * 4 // page) * page


def summary(values):
    """Mean, median and 99th percentile."""
    if len(values) == 0:
        return [0.0, 0.0, 0.0]
    values = np.asarray(values, dtype=np.float64)
    return [
        float(values.mean()),
        float(np.median(values)),
        float(np.percentile(values, 99)),
    ]


def skew(load):
    """Shares of the whole load taken by the busiest 1 % and 10 % of the items."""
    ordered = np.sort(np.asarray(load, dtype=np.float64))[::-1]
    total = max(ordered.sum(), 1.0)
    shares = []
    for fraction in (0.01, 0.10):
        top = max(1, int(np.ceil(fraction * len(ordered))))
        shares.append(float(ordered[:top].sum() / total))
    return shares


def owner_copies(load):
    """Copies per treelet: one per half a core's mean load, 1 to MOST_OWNERS."""
    half_core = max(load.sum() / CORES / 2, 1e-9)
    return np.clip(np.ceil(load / half_core), 1, MOST_OWNERS).astype(np.int64)


def assign_owners(load, sizes, copies):
    """Treelet copies to cores, the largest piece of load first to the least loaded
    core with room in L1 and no copy of that treelet; a copy that finds no room is
    dropped (a first copy goes to the core with the fewest bytes). Returns owners
    (K, MOST_OWNERS) (-1: none), copies placed, and each core's load and bytes."""
    count = len(load)
    owners = np.full((count, MOST_OWNERS), -1, dtype=np.int64)
    core_load = np.zeros(CORES)
    core_bytes = np.zeros(CORES, dtype=np.int64)
    pieces = np.repeat(np.arange(count), copies)
    piece_load = load[pieces] / copies[pieces]
    placed = np.zeros(count, dtype=np.int64)
    for piece in np.argsort(-piece_load, kind="stable"):
        treelet = pieces[piece]
        candidates = np.argsort(core_load, kind="stable")
        candidates = candidates[
            ~np.isin(candidates, owners[treelet, : placed[treelet]])
        ]
        room = core_bytes[candidates] + sizes[treelet] <= L1_ROOM_BYTES
        if room.any():
            core = int(candidates[room][0])
        elif placed[treelet] == 0:
            core = int(candidates[np.argmin(core_bytes[candidates])])
        else:
            continue
        owners[treelet, placed[treelet]] = core
        placed[treelet] += 1
        core_load[core] += piece_load[piece]
        core_bytes[core] += sizes[treelet]
    core_load = np.zeros(CORES)
    for copy in range(MOST_OWNERS):
        held = owners[:, copy] >= 0
        np.add.at(core_load, owners[held, copy], load[held] / placed[held])
    return owners, placed, core_load, core_bytes


def home_cores(pixels):
    """The home core of each ray (1024 pixel tiles of 32 x 32 in equal contiguous
    ranges over the cores) and the rays in the order a core takes them."""
    order, tile = packets.ray_order(np.asarray(pixels))
    return tile * CORES // packets.TILES, order


def bundles(paths, owners, copies, home, seed_order, bundle_rays, rule):
    """Rays moved hop by hop in rounds, binned by next treelet copy or home and sent as
    bundles of at most `bundle_rays` under `rule` (README, part A). Returns the
    statistics, and per treelet visit its bundle and the core that walks it."""
    ray_start, visit_treelet, visits = paths
    ray_count = len(home)
    rays = np.nonzero(visits > 0)[0]
    keys = CORES + len(owners) * (MOST_OWNERS + 1)
    location = home[rays].copy()
    seed_rank = np.empty(ray_count, dtype=np.int64)
    seed_rank[seed_order] = np.arange(ray_count)
    previous = seed_rank[rays]
    if rule == "each":
        core_rank = np.empty(ray_count, dtype=np.int64)
        ordered_home = home[seed_order]
        for core in range(CORES):
            members = seed_order[ordered_home == core]
            core_rank[members] = np.arange(len(members))
        previous = home[rays] * ray_count + core_rank[rays] // bundle_rays
    visit_bundle = np.full(len(visit_treelet), -1, dtype=np.int64)
    visit_core = np.full(len(visit_treelet), -1, dtype=np.int64)
    sent = np.zeros(CORES, dtype=np.int64)
    received = np.zeros(CORES, dtype=np.int64)
    fills, local_bundles, local_rays, bundle_base = [], 0, 0, 0
    alive = np.ones(len(rays), dtype=bool)
    hop = 0
    while alive.any():
        index = np.nonzero(alive)[0]
        ray = rays[index]
        source = location[index]
        to_home = hop >= visits[ray]
        visit = ray_start[ray] + np.minimum(hop, visits[ray] - 1)
        treelet = visit_treelet[visit]
        holds = (owners[treelet] == source[:, None]).any(axis=1)
        copy = np.where(holds, MOST_OWNERS, source % copies[treelet])
        chosen = owners[treelet, np.minimum(copy, MOST_OWNERS - 1)]
        target = np.where(to_home, home[ray], np.where(holds, source, chosen))
        bin_key = CORES + treelet * (MOST_OWNERS + 1) + copy
        group = source * keys + np.where(to_home, home[ray], bin_key)
        if rule == "each":
            group = previous[index] * (keys * CORES) + group
        order = np.lexsort((ray, previous[index], group))
        group_sorted = group[order]
        starts = np.r_[0, np.nonzero(np.diff(group_sorted))[0] + 1]
        sizes = np.diff(np.r_[starts, len(order)])
        rank = np.arange(len(order)) - np.repeat(starts, sizes)
        chunks = -(-sizes // bundle_rays)
        chunk_base = np.r_[0, np.cumsum(chunks)[:-1]]
        local_bundle = np.repeat(chunk_base, sizes) + rank // bundle_rays
        bundle_of = np.empty(len(order), dtype=np.int64)
        bundle_of[order] = bundle_base + local_bundle
        filled = np.bincount(local_bundle, minlength=int(chunks.sum()))
        chunk_local = np.repeat((target == source)[order][starts], chunks)
        chunk_source = np.repeat(source[order][starts], chunks)
        chunk_target = np.repeat(target[order][starts], chunks)
        sent += np.bincount(chunk_source[~chunk_local], minlength=CORES)
        received += np.bincount(chunk_target[~chunk_local], minlength=CORES)
        fills.append(filled[~chunk_local])
        local_bundles += int(chunk_local.sum())
        local_rays += int(filled[chunk_local].sum())
        bundle_base += int(chunks.sum())
        walking = ~to_home
        visit_bundle[visit[walking]] = bundle_of[walking]
        visit_core[visit[walking]] = target[walking]
        location[index] = target
        previous[index] = bundle_of
        alive[index[to_home]] = False
        hop += 1
    fill = np.concatenate(fills) if fills else np.zeros(0, dtype=np.int64)
    remote = int(len(fill))
    stats = {
        "rounds": hop,
        "bundles": remote,
        "bundles_local": local_bundles,
        "ray_moves": int(fill.sum()),
        "ray_moves_local": local_rays,
        "fill_mean": float(fill.mean() / bundle_rays) if remote else 0.0,
        "fill_median": float(np.median(fill) / bundle_rays) if remote else 0.0,
        "full_share": float((fill == bundle_rays).mean()) if remote else 0.0,
        "bytes_fixed": remote * (BUNDLE_HEADER_BYTES + bundle_rays * RECORD_BYTES),
        "bytes_filled": remote * BUNDLE_HEADER_BYTES + int(fill.sum()) * RECORD_BYTES,
        "sent_per_core_max_mean": [int(sent.max()), float(sent.mean())],
        "received_per_core_max_mean": [int(received.max()), float(received.mean())],
        "received_per_core": received.tolist(),
    }
    return stats, visit_bundle, visit_core


def packet_walk(roots, lanes, best, leaf, any_hit, stop=None):
    """The card's packet walk (README, part A) of packets (P, 32) of rays from `roots`;
    best and leaf (P, 32) change in place. With `stop` (node -> treelet or -1) treelet
    roots are emitted, not walked. Returns counts per packet and the emitted pairs."""
    nodes, boxes, triangles = SHARED["nodes"], SHARED["boxes"], SHARED["triangles"]
    leaf_triangle, order_words = SHARED["leaf_triangle"], SHARED["order"]
    count = len(roots)
    valid = lanes >= 0
    ray = np.maximum(lanes, 0)
    origins, directions = SHARED["origins"][ray], SHARED["directions"][ray]
    inverse = SHARED["inverse"][ray]
    signs = packets.head_signs(directions, valid).astype(np.int64)
    names = ("pops", "empty", "inner", "leaves", "inner_lanes", "leaf_lanes", "emits")
    counts = {name: np.zeros(count, dtype=np.int64) for name in names}
    stack_node = np.zeros((count, STACK), dtype=np.int64)
    stack_entry = np.zeros((count, STACK, LANES), dtype=np.float32)
    depth = np.ones(count, dtype=np.int64)
    hit, entry = box_entry(
        boxes[roots, None, :3], boxes[roots, None, 3:], origins, inverse, best
    )
    stack_node[:, 0] = roots
    stack_entry[:, 0] = np.where(hit & valid, entry, INFINITE)
    emitted = []
    active = np.arange(count)
    while len(active):
        depth[active] -= 1
        top = depth[active]
        node = stack_node[active, top]
        alive = stack_entry[active, top] <= best[active]
        counts["pops"][active] += 1
        live = alive.any(axis=1)
        counts["empty"][active[~live]] += 1
        packet, node, alive = active[live], node[live], alive[live]
        if stop is not None:
            treelet = stop[node]
            emit = treelet >= 0
            emitted.append((packet[emit], treelet[emit]))
            counts["emits"][packet[emit]] += 1
            packet, node, alive = packet[~emit], node[~emit], alive[~emit]
        word = nodes[node, 6]
        is_leaf = bvh.is_leaf(word)
        tested = packet[is_leaf]
        if len(tested):
            mask = alive[is_leaf]
            counts["leaves"][tested] += 1
            counts["leaf_lanes"][tested] += mask.sum(axis=1)
            row, lane = np.nonzero(mask)
            target = tested[row]
            index = (word[is_leaf][row] & ~bvh.LEAF).astype(np.int64)
            ok, distance, _, _ = raytrace.triangle_test(
                triangles, index, origins[target, lane], directions[target, lane]
            )
            current = best[target, lane]
            better = ok & (distance < current)
            if any_hit:
                best[target[better], lane[better]] = FLOAT(-1)  # the lane leaves
            else:
                old = leaf[target, lane]
                same = ok & (distance == current) & (old >= 0)
                better |= same & (
                    leaf_triangle[index] < leaf_triangle[np.maximum(old, 0)]
                )
                best[target[better], lane[better]] = distance[better]
            leaf[target[better], lane[better]] = index[better]
        walked = packet[~is_leaf]
        if len(walked):
            mask = alive[~is_leaf]
            inner = node[~is_leaf]
            counts["inner"][walked] += 1
            counts["inner_lanes"][walked] += mask.sum(axis=1)
            order = order_words[inner]
            second = ((signs[walked] >> (order & 3)) ^ (order >> 2)) & 1
            first_child = nodes[inner, 6].astype(np.int64)
            second_child = nodes[inner, 7].astype(np.int64)
            near = np.where(second == 1, second_child, first_child)
            far = np.where(second == 1, first_child, second_child)
            for child in (far, near):
                hit, entry = box_entry(
                    boxes[child, None, :3],
                    boxes[child, None, 3:],
                    origins[walked],
                    inverse[walked],
                    best[walked],
                )
                slot = depth[walked]
                stack_node[walked, slot] = child
                stack_entry[walked, slot] = np.where(hit & mask, entry, INFINITE)
                depth[walked] += 1
        active = active[depth[active] > 0]
    return counts, emitted


def walk_job(job):
    """packet_walk of packets that each visit one treelet (a pool job)."""
    roots, lanes, best, leaf, any_hit = job
    counts, _ = packet_walk(roots, lanes, best, leaf, any_hit)
    return counts


def bundle_packets(visit_bundle, visit_treelet, visit_core):
    """Packets of 32 lanes made from the bundles' treelet visits (a bundle of R rays
    is R / 32 packets): each visit's packet and lane, each packet's treelet and core."""
    order = np.lexsort((np.arange(len(visit_bundle)), visit_bundle))
    order = order[visit_bundle[order] >= 0]
    bundle = visit_bundle[order]
    starts = np.r_[0, np.nonzero(np.diff(bundle))[0] + 1] if len(bundle) else []
    sizes = np.diff(np.r_[starts, len(order)])
    rank = np.arange(len(order)) - np.repeat(starts, sizes)
    first_packet = np.r_[0, np.cumsum(-(-sizes // LANES))[:-1]]
    packet = np.repeat(first_packet, sizes) + rank // LANES
    count = int(packet.max()) + 1 if len(packet) else 0
    treelet = np.zeros(count, dtype=np.int64)
    core = np.zeros(count, dtype=np.int64)
    treelet[packet] = visit_treelet[order]
    core[packet] = visit_core[order]
    return order, packet, rank % LANES, treelet, core


def owner_walks(visits, visit_bundle, visit_core, any_hit, generator, workers):
    """The owners' packet walks of a sample of the bundles' packets (stratified by
    treelet): alive lanes per inner node and leaf visit, counts per packet; each
    core's estimated counts and cycles, and each treelet's cycles, from the
    treelets' mean packets."""
    order, packet, lane, packet_treelet, packet_core = bundle_packets(
        visit_bundle, visits["treelet"], visit_core
    )
    treelets = SHARED["treelets"]
    per_treelet = np.bincount(packet_treelet, minlength=treelets)
    quota = int(
        np.clip(
            SAMPLE_PACKETS // max((per_treelet > 0).sum(), 1), 1, SAMPLE_PER_TREELET
        )
    )
    chosen = []
    for treelet in np.nonzero(per_treelet)[0]:
        members = np.nonzero(packet_treelet == treelet)[0]
        chosen.append(
            generator.choice(members, size=min(quota, len(members)), replace=False)
        )
    chosen = np.sort(np.concatenate(chosen))
    weight = (
        per_treelet[packet_treelet[chosen]]
        / np.bincount(packet_treelet[chosen], minlength=treelets)[
            packet_treelet[chosen]
        ]
    )
    row_of = np.full(len(packet_treelet), -1, dtype=np.int64)
    row_of[chosen] = np.arange(len(chosen))
    row = row_of[packet]
    keep = row >= 0
    lane_visit = np.full((len(chosen), LANES), -1, dtype=np.int64)
    lane_visit[row[keep], lane[keep]] = order[keep]
    valid = lane_visit >= 0
    visit = np.maximum(lane_visit, 0)
    lanes = np.where(valid, visits["ray"][visit], -1)
    best = np.where(valid, visits["best"][visit], FLOAT(-1)).astype(np.float32)
    leaf = np.where(valid, visits["leaf"][visit], -1).astype(np.int64)
    roots = SHARED["treelet_root"][packet_treelet[chosen]]
    parts = np.array_split(np.arange(len(chosen)), max(1, min(workers, len(chosen))))
    jobs = [(roots[p], lanes[p], best[p], leaf[p], any_hit) for p in parts]
    with pool(len(jobs)) as workers_pool:
        results = workers_pool.map(walk_job, jobs, chunksize=1)
    counts = {name: np.concatenate([r[name] for r in results]) for name in results[0]}
    total = weight.sum()
    result = {
        name: float((counts[name] * weight).sum() / total)
        for name in ("pops", "empty", "inner", "leaves")
    }
    result["lanes_per_inner"] = float(
        (counts["inner_lanes"] * weight).sum()
        / max((counts["inner"] * weight).sum(), 1)
    )
    result["lanes_per_leaf"] = float(
        (counts["leaf_lanes"] * weight).sum()
        / max((counts["leaves"] * weight).sum(), 1)
    )
    result["lanes_per_packet"] = float((valid.sum(axis=1) * weight).sum() / total)
    result["packets"] = int(len(packet_treelet))
    result["sampled"] = int(len(chosen))
    sampled_treelet = packet_treelet[chosen]
    number = np.maximum(np.bincount(sampled_treelet, minlength=treelets), 1)
    per_core = {"packets": np.bincount(packet_core, minlength=CORES)}
    packet_cycles = np.zeros(treelets)
    for name, cycles in (
        ("pops", POP_CYCLES),
        ("inner", INNER_CYCLES),
        ("leaves", LEAF_CYCLES[any_hit]),
    ):
        mean = np.bincount(sampled_treelet, counts[name], minlength=treelets) / number
        per_core[name] = np.bincount(packet_core, mean[packet_treelet], minlength=CORES)
        packet_cycles += mean * cycles
    packet_cycles += STATE_CYCLES
    per_core["cycles"] = np.bincount(
        packet_core, packet_cycles[packet_treelet], minlength=CORES
    )
    result["cycles_per_core_max_mean"] = [
        float(per_core["cycles"].max()),
        float(per_core["cycles"].mean()),
    ]
    treelet_cycles = per_treelet * packet_cycles
    per_core = {name: value.tolist() for name, value in per_core.items()}
    return result, per_core, treelet_cycles


def calibration(rays, home, any_hit, generator):
    """Today's walk on a sample of today's packets (each core's rays sorted by
    direction octant and origin cell, 32 to a packet): the top walk emits treelets,
    the packet then walks them in that order. Counts per packet and alive lanes."""
    origins, directions = SHARED["origins"], SHARED["directions"]
    low, high = (
        box.astype(np.float64)[0] for box in bvh.node_boxes(SHARED["nodes"][:1])
    )
    cell = np.clip(
        ((origins - low) / np.maximum(high - low, 1e-30) * SORT_CELLS).astype(np.int64),
        0,
        SORT_CELLS - 1,
    )
    code = np.zeros(len(origins), dtype=np.int64)
    for bit in range(9):
        for axis in range(3):
            code |= ((cell[:, axis] >> bit) & 1) << (3 * bit + axis)
    octant = ((directions < 0) * np.array([1, 2, 4])).sum(axis=1)
    order = np.lexsort((np.arange(len(origins)), code, octant, home))
    starts = np.r_[0, np.nonzero(np.diff(home[order]))[0] + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    rank = np.arange(len(order)) - np.repeat(starts, sizes)
    first_packet = np.r_[0, np.cumsum(-(-sizes // LANES))[:-1]]
    packet = np.repeat(first_packet, sizes) + rank // LANES
    count = int(packet.max()) + 1
    chosen = np.sort(
        generator.choice(count, size=min(CALIBRATION_PACKETS, count), replace=False)
    )
    row_of = np.full(count, -1, dtype=np.int64)
    row_of[chosen] = np.arange(len(chosen))
    lanes = np.full((len(chosen), LANES), -1, dtype=np.int64)
    keep = row_of[packet] >= 0
    lanes[row_of[packet][keep], (rank % LANES)[keep]] = order[keep]
    valid = lanes >= 0
    best = np.where(valid, SHARED["maximum"][np.maximum(lanes, 0)], FLOAT(-1))
    best = best.astype(np.float32)
    leaf = np.full(lanes.shape, -1, dtype=np.int64)
    roots = np.zeros(len(chosen), dtype=np.int64)
    if SHARED["node_treelet"][0] >= 0:
        emitted = [(np.arange(len(chosen)), np.zeros(len(chosen), dtype=np.int64))]
        totals = {}
    else:
        totals, emitted = packet_walk(
            roots, lanes, best, leaf, any_hit, stop=SHARED["node_treelet"]
        )
    packet_of = np.concatenate([e[0] for e in emitted])
    treelet_of = np.concatenate([e[1] for e in emitted])
    sequence = np.argsort(packet_of, kind="stable")
    packet_of, treelet_of = packet_of[sequence], treelet_of[sequence]
    first = np.searchsorted(packet_of, np.arange(len(chosen)))
    emits = np.bincount(packet_of, minlength=len(chosen))
    names = ("pops", "empty", "inner", "leaves", "inner_lanes", "leaf_lanes")
    sums = {name: int(totals.get(name, np.zeros(1)).sum()) for name in names}
    step = 0
    while True:
        busy = np.nonzero(emits > step)[0]
        if not len(busy):
            break
        visit_roots = SHARED["treelet_root"][treelet_of[first[busy] + step]]
        part_best, part_leaf = best[busy], leaf[busy]
        counts, _ = packet_walk(visit_roots, lanes[busy], part_best, part_leaf, any_hit)
        best[busy], leaf[busy] = part_best, part_leaf
        for name in names:
            sums[name] += int(counts[name].sum())
        step += 1
    packets_walked = len(chosen)
    return {
        "packets": packets_walked,
        "visits_per_packet": float(emits.mean()),
        "pops_per_packet": sums["pops"] / packets_walked,
        "inner_per_packet": sums["inner"] / packets_walked,
        "leaves_per_packet": sums["leaves"] / packets_walked,
        "lanes_per_inner": sums["inner_lanes"] / max(sums["inner"], 1),
        "lanes_per_leaf": sums["leaf_lanes"] / max(sums["leaves"], 1),
    }


def placements(load, sizes):
    """Owners by `load` with one copy of each treelet and with copies of the busy
    ones: {kind: (owners, copies)} and each placement's balance and bytes."""
    found, report = {}, {}
    for kind, copies in (
        ("single", np.ones(len(load), dtype=np.int64)),
        ("replicated", owner_copies(load)),
    ):
        owners, copies, core_load, core_bytes = assign_owners(load, sizes, copies)
        found[kind] = (owners, copies)
        report[kind] = {
            "copies": int(copies.sum()),
            "copies_max": int(copies.max()),
            "treelets_copied": int((copies > 1).sum()),
            "load_per_core_max_over_mean": float(core_load.max() / core_load.mean()),
            "treelet_bytes_per_core_max": int(core_bytes.max()),
            "fits": bool(core_bytes.max() <= L1_ROOM_BYTES),
        }
    return found, report


def run_trace(name, budget, trace, rays, sizes, arguments, generator):
    """Every number of part A for one scene, budget and trace. Owners are placed
    twice: by the trace's treelet visits, then by the cycles a first pass of
    bundles predicts per treelet (the frame before, for the same camera)."""
    any_hit = trace in ANY_HIT
    pixels = rays[0]
    began = time.time()
    final, visits = follow(rays[1:], any_hit, arguments.workers)
    visits_per_ray = np.bincount(visits["ray"], minlength=len(pixels))
    entered = final["entered"]
    treelets = SHARED["treelets"]
    load = np.bincount(visits["treelet"], minlength=treelets).astype(np.float64)
    report = {
        "rays": int(len(pixels)),
        "visits": int(visits_per_ray.sum()),
        "visits_per_ray": summary(visits_per_ray),
        "entered_per_ray": summary(entered),
        "saved_share": float(1 - visits_per_ray.sum() / max(entered.sum(), 1)),
        "skew_1_10": skew(load),
        "follow_seconds": round(time.time() - began, 1),
    }
    log(
        f"{name} {budget} KiB {trace}: {report['visits']} visits, "
        f"{report['visits_per_ray'][0]:.2f} a ray ({report['follow_seconds']} s)"
    )
    home, seed_order = home_cores(pixels)
    ray_start = np.searchsorted(visits["ray"], np.arange(len(pixels)))
    paths = (ray_start, visits["treelet"], visits_per_ray)
    by_visits, report["by_visits"] = placements(load, sizes)
    first = "replicated" if report["by_visits"]["replicated"]["fits"] else "single"
    _, visit_bundle, visit_core = bundles(
        paths, *by_visits[first], home, seed_order, BUNDLE_RAYS[0], RULES[0]
    )
    _, _, cycles = owner_walks(
        visits, visit_bundle, visit_core, any_hit, generator, arguments.workers
    )
    by_cycles, report["by_cycles"] = placements(cycles, sizes)
    chosen = "replicated" if report["by_cycles"]["replicated"]["fits"] else "single"
    report["placement"] = chosen
    for kind, (owners, copies) in by_cycles.items():
        report["by_cycles"][kind]["bundles"] = {}
        for bundle_rays in BUNDLE_RAYS:
            for rule in RULES:
                stats, visit_bundle, visit_core = bundles(
                    paths, owners, copies, home, seed_order, bundle_rays, rule
                )
                if kind == chosen:
                    stats["walk"], stats["walk_per_core"], _ = owner_walks(
                        visits,
                        visit_bundle,
                        visit_core,
                        any_hit,
                        generator,
                        arguments.workers,
                    )
                report["by_cycles"][kind]["bundles"][f"{bundle_rays}/{rule}"] = stats
        log(f"{name} {budget} KiB {trace}: bundles of the {kind} placement done")
    if (
        budget == 128
        and name in arguments.calibrate
        and trace in arguments.calibrate_traces
    ):
        report["today"] = calibration(rays, home, any_hit, generator)
        log(f"{name} {trace}: today's walk on {report['today']['packets']} packets")
    return report


def tables(report):
    """One markdown table per scene from a report of main()."""
    lines = []
    for name, scene in report["scenes"].items():
        lines += [
            f"{name} ({scene['triangles']:,} triangles)",
            "",
            "| budget KiB | trace | visits a ray, mean / median / p99 | saved by"
            " the early exit | busiest 1 % / 10 % of treelets | visits per core max /"
            " mean, one owner / several | L1 fits | bundles, R = 32 / 64 / 128 (fill)"
            " | fill under the each rule | lanes per inner node | MB moved |",
            "|" + " --- |" * 11,
        ]
        for budget, entry in scene["budgets"].items():
            for trace, row in entry["traces"].items():
                placement = row["by_cycles"][row["placement"]]
                runs = placement["bundles"]
                full = " / ".join(
                    f"{runs[f'{r}/full']['bundles'] / 1e3:.0f}k"
                    f" ({runs[f'{r}/full']['fill_mean']:.2f})"
                    for r in BUNDLE_RAYS
                )
                each = " / ".join(
                    f"{runs[f'{r}/each']['fill_mean']:.3f}" for r in BUNDLE_RAYS
                )
                walk = runs[f"{BUNDLE_RAYS[0]}/full"].get("walk", {})
                lanes = walk.get("lanes_per_inner")
                mean, median, high = row["visits_per_ray"]
                share = row["skew_1_10"]

                def balance(kind, row=row):
                    return row["by_visits"][kind]["load_per_core_max_over_mean"]

                lines.append(
                    f"| {budget} | {trace} | {mean:.2f} / {median:.0f} / {high:.0f}"
                    f" | {row['saved_share'] * 100:.0f} %"
                    f" | {share[0] * 100:.0f} % / {share[1] * 100:.0f} %"
                    f" | {balance('single'):.2f} / {balance('replicated'):.2f}"
                    f" | {'yes' if placement['fits'] else 'no'} | {full} | {each}"
                    f" | {'' if lanes is None else f'{lanes:.2f}'}"
                    f" | {runs[f'{BUNDLE_RAYS[0]}/full']['bytes_filled'] / 1e6:.0f} |"
                )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--traces", nargs="+", default=list(TRACES))
    parser.add_argument("--data", default=os.environ.get("KR_DATA", "build/data"))
    parser.add_argument("--cache", default="build/ray_moves")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--calibrate", nargs="*", default=["sponza_overview"])
    parser.add_argument("--calibrate-traces", nargs="*", default=["bounce"])
    parser.add_argument("--json", default="paths.json")
    parser.add_argument("--tables", default=None, help="print the tables of a report")
    arguments = parser.parse_args()
    if arguments.tables:
        print(tables(json.loads(Path(arguments.tables).read_text())))
        return
    data = Path(arguments.data)
    report = {
        "cores": CORES,
        "record_bytes": RECORD_BYTES,
        "l1_room_bytes": L1_ROOM_BYTES,
        "load_before": Path("/proc/loadavg").read_text().split()[:3],
        "scenes": {},
    }
    for name in arguments.scenes:
        scene = scenes.load_scene(data / f"{name}.npz")
        tree = bvh.load_tree(data / f"{name}_tree.npz")
        report["scenes"][name] = {"triangles": int(len(tree["tris"])), "budgets": {}}
        sets = None
        for budget in sorted(set(arguments.budgets), reverse=True):
            began = time.time()
            cut = bvh.cut_treelets(tree, budget * 1024)
            use_tree(tree, cut)
            sizes = treelet_bytes(cut)
            seconds = time.time() - began
            log(f"{name} {budget} KiB: {len(sizes)} treelets ({seconds:.0f} s)")
            if sets is None:
                sets = ray_sets(name, scene, tree, arguments.cache, arguments.workers)
            entry = {
                "treelets": int(len(sizes)),
                "top_nodes": int(len(cut["top_nodes"])),
                "scene_bytes": int(sizes.sum()),
                "traces": {},
            }
            for trace in arguments.traces:
                generator = np.random.default_rng(SEED)
                entry["traces"][trace] = run_trace(
                    name, budget, trace, sets[trace], sizes, arguments, generator
                )
                report["scenes"][name]["budgets"][str(budget)] = entry
                Path(arguments.json).write_text(json.dumps(report))
    report["load_after"] = Path("/proc/loadavg").read_text().split()[:3]
    Path(arguments.json).write_text(json.dumps(report))
    log(f"wrote {arguments.json}")


if __name__ == "__main__":
    main()
