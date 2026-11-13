"""The arithmetic check's host side: inputs from a real scene and edge cases, the
expected outputs with the reference's own expressions, and the comparison.

Pure numpy, no card (the driver is `arith.py`).
"""

import numpy as np

from projects.raytrace import packets
from projects.raytrace.reference import harness
from reference import bvh, native, raytrace, scenes

K = packets.K
F = np.float32
VEC = packets.VEC
IN_NAMES = (
    ("o", 3), ("d", 3), ("inv", 3), ("bmin", 3), ("bmax", 3), ("best", 1),
    ("v0", 3), ("e1", 3), ("e2", 3), ("tri", 1), ("cur", 1),
)  # fmt: skip
OUT_NAMES = (
    "entry", "exit", "stack", "det", "det_inv", "u", "v", "t",
    "new_best", "new_tri", "new_u", "new_v",
)  # fmt: skip


def parents(nodes):
    """Parent of every node (-1 for the root)."""
    parent_of = np.full(len(nodes), -1, dtype=np.int64)
    inner = np.nonzero(~bvh.is_leaf(nodes[:, 6]))[0]
    parent_of[nodes[inner, 6].astype(np.int64)] = inner
    parent_of[nodes[inner, 7].astype(np.int64)] = inner
    return parent_of


def ray_pool(scene_name, generator, n_pixels):
    """(origins, directions, tmax, hit leaf) of primary, shadow and bounce rays of
    random pixels."""
    data = native.data_dir()
    tree = harness.ensure_tree(scene_name)
    scene = scenes.load_scene(data / f"{scene_name}.npz")
    pixels = generator.choice(scenes.WIDTH * scenes.HEIGHT, n_pixels, replace=False)
    origins, directions = raytrace.camera_rays(scene["camera"], pixels)
    hit = raytrace.trace(tree, origins, directions, raytrace.TMAX)
    hit_rows = np.nonzero(hit["leaf"] >= 0)[0]
    hits = {key: values[hit_rows] for key, values in hit.items()}
    points = origins[hit_rows] + directions[hit_rows] * hits["t"][:, None]
    normals, _ = raytrace.surface(scene, tree, hits, directions[hit_rows])
    shadow_origins = points + normals * scene["eps"][0]
    to_light = scene["light"] - shadow_origins
    bounce_directions = raytrace.bounce_directions(pixels[hit_rows], normals)
    shadow = raytrace.trace(tree, shadow_origins, to_light, F(1.0), any_hit=True)
    bounce = raytrace.trace(tree, shadow_origins, bounce_directions, raytrace.TMAX)
    all_origins = np.concatenate([origins, shadow_origins, shadow_origins])
    all_directions = np.concatenate([directions, to_light, bounce_directions])
    tmax = np.concatenate(
        [
            np.full(len(origins), raytrace.TMAX),
            np.ones(len(hit_rows), F),
            np.full(len(hit_rows), raytrace.TMAX),
        ]
    ).astype(F)
    leaf = np.concatenate([hit["leaf"], shadow["leaf"], bounce["leaf"]])
    return scene, tree, all_origins.astype(F), all_directions.astype(F), tmax, leaf


def scene_inputs(generator, count, scene_name):
    """`count` input rows from rays of a scene, the nodes and triangles near their
    hits."""
    _, tree, origins, directions, tmax, leaf = ray_pool(scene_name, generator, 4096)
    nodes, tris = tree["nodes"], tree["tris"]
    leaf_count = len(tris)
    parent_of = parents(nodes)
    picks = generator.integers(0, len(origins), count)
    hit_leaf = leaf[picks]
    # Triangles: the hit one, its neighbours in leaf order, or a random one.
    kind = generator.integers(0, 4, count)
    candidate = np.where(
        hit_leaf >= 0, hit_leaf, generator.integers(0, leaf_count, count)
    )
    triangle_ids = np.where(
        kind == 0, candidate, candidate + generator.choice([-1, 1], count)
    )
    triangle_ids = np.where(
        kind == 3, generator.integers(0, leaf_count, count), triangle_ids
    )
    triangle_ids = np.clip(triangle_ids, 0, leaf_count - 1)
    # Boxes: the hit leaf's node or one of its ancestors, or a random node.
    node = tree["leaf_node"][np.maximum(hit_leaf, 0)].astype(np.int64)
    for _ in range(int(generator.integers(0, 6))):
        ancestor = parent_of[node]
        node = np.where(
            (ancestor >= 0) & (generator.random(count) < 0.6), ancestor, node
        )
    node = np.where(
        (hit_leaf < 0) | (generator.random(count) < 0.3),
        generator.integers(0, len(nodes), count),
        node,
    )
    box_min, box_max = bvh.node_boxes(nodes)
    rows = {
        "o": origins[picks],
        "d": directions[picks],
        "bmin": box_min[node],
        "bmax": box_max[node],
        "v0": tris[triangle_ids, 0:3],
        "e1": tris[triangle_ids, 3:6],
        "e2": tris[triangle_ids, 6:9],
        "tri": tree["leaf_tri"][triangle_ids].astype(np.int64),
        "best": tmax[picks],
    }
    return rows, tree


def edge_inputs(generator, rows, tree):
    """Edge cases written over a partition of the rows (in place); returns their
    mask."""
    count = len(rows["o"])
    kind = generator.integers(0, 10, count)
    edge = generator.random(count) < 0.25
    origins, directions = rows["o"].copy(), rows["d"].copy()
    axis = generator.integers(0, 3, count)
    row_index = np.arange(count)
    # 0: a direction component zero (+-0) or below the clamp.
    selected = edge & (kind == 0)
    tiny = np.array([0.0, -0.0, 1e-13, -1e-13, 1e-12, -1e-12, 1e-30], F)
    directions[row_index[selected], axis[selected]] = tiny[
        generator.integers(0, len(tiny), selected.sum())
    ]
    # 1: origin on a box face; 2: a flat box with the origin in its plane.
    selected = edge & (kind == 1)
    face = np.where(
        generator.random(count) < 0.5,
        rows["bmin"][row_index, axis],
        rows["bmax"][row_index, axis],
    )
    origins[row_index[selected], axis[selected]] = face[selected]
    selected = edge & (kind == 2)
    rows["bmax"][row_index[selected], axis[selected]] = rows["bmin"][
        row_index[selected], axis[selected]
    ]
    origins[row_index[selected], axis[selected]] = rows["bmin"][
        row_index[selected], axis[selected]
    ]
    # 3, 4: aimed at a vertex or at an edge point of its triangle (grazing hits).
    vertex, edge1, edge2 = rows["v0"], rows["e1"], rows["e2"]
    selected = edge & (kind == 3)
    corner = np.stack([vertex, vertex + edge1, vertex + edge2])[
        generator.integers(0, 3, count), row_index
    ]
    directions[selected] = (corner - origins)[selected]
    selected = edge & (kind == 4)
    along = generator.random(count).astype(F)[:, None]
    point = np.where(
        generator.random(count)[:, None] < 0.5,
        vertex + edge1 * along,
        vertex + edge2 * along,
    )
    directions[selected] = (point - origins)[selected]
    # 5: a direction in the triangle's plane (det 0 or nearly).
    selected = edge & (kind == 5)
    directions[selected] = np.where(
        generator.random(count)[:, None] < 0.5, edge1, edge2
    )[selected]
    # 6: signed zeros in the origin.
    selected = edge & (kind == 6)
    origins[row_index[selected], axis[selected]] = np.where(
        generator.random(selected.sum()) < 0.5, F(0.0), F(-0.0)
    )
    rows["o"], rows["d"] = origins.astype(F), directions.astype(F)
    return edge


def bests(generator, rows):
    """best: tmax, the pair's own t (a tie), its neighbours, or random; cur: -1, the
    triangle index or one off."""
    count = len(rows["o"])
    hit_ok, distance, _, _ = triangle(rows)
    kind = generator.integers(0, 6, count)
    best = rows["best"].copy()
    distance_or_one = np.where(hit_ok, distance, F(1.0)).astype(F)
    best = np.where(kind == 1, distance_or_one, best)
    best = np.where(kind == 2, np.nextafter(distance_or_one, F(np.inf)), best)
    best = np.where(kind == 3, np.nextafter(distance_or_one, F(0)), best)
    best = np.where(kind == 4, generator.uniform(0, 50, count).astype(F), best)
    rows["best"] = best.astype(F)
    index_kind = generator.integers(0, 4, count)
    triangles = rows["tri"]
    rows["cur"] = np.where(
        index_kind == 0,
        -1,
        np.where(
            index_kind == 1,
            triangles - 1,
            np.where(index_kind == 2, triangles + 1, triangles),
        ),
    )


def triangle(rows):
    """The reference's triangle test on the rows (one triangle per row)."""
    geometry = np.concatenate([rows["v0"], rows["e1"], rows["e2"]], axis=1)
    with np.errstate(all="ignore"):
        return raytrace.triangle_test(
            geometry, np.arange(len(geometry)), rows["o"], rows["d"]
        )


def expected(rows):
    """The outputs in numpy float32, in OUT_NAMES order (with the reference's code)."""
    origins, directions = rows["o"], rows["d"]
    inverse = raytrace.inverse_direction(directions)
    box = np.concatenate([rows["bmin"], rows["bmax"]], axis=1)
    best = rows["best"]
    with np.errstate(all="ignore"):
        hit, entry = raytrace.box_test(
            box, np.arange(len(origins)), origins, inverse, best
        )
        to_min = (box[:, :3] - origins) * inverse
        to_max = (box[:, 3:] - origins) * inverse
        far = np.maximum(to_min, to_max)
        exit_ = (
            np.minimum(np.minimum(far[:, 0], far[:, 1]), far[:, 2])
            * raytrace.SLAB_SCALE
        )
        exit_limit = np.minimum(exit_, best)
        stack = np.where(hit, entry, F(np.inf))
        perpendicular = raytrace.cross(directions, rows["e2"])
        determinant = raytrace.dot(rows["e1"], perpendicular)
        determinant_inverse = F(1.0) / determinant
        valid, distance, u, v = triangle(rows)
    current, triangle_index = rows["cur"], rows["tri"]
    better = valid & (distance < best)
    better |= valid & (distance == best) & (current >= 0) & (triangle_index < current)
    return {
        "entry": entry,
        "exit": exit_limit,
        "stack": stack,
        "det": determinant,
        "det_inv": determinant_inverse,
        "u": u,
        "v": v,
        "t": distance,
        "new_best": np.where(better, distance, best),
        "new_tri": np.where(better, triangle_index, current).astype(np.int32),
        "new_u": np.where(better, u, F(0)),
        "new_v": np.where(better, v, F(0)),
        "hit": hit,
        "ok": valid,
        "inv": inverse,
    }


def input_pages(rows):
    """Input pages (P, kArInVectors * 32) uint32; rows padded to a multiple of 32."""
    columns = []
    inverse = raytrace.inverse_direction(rows["d"])
    for name, width in IN_NAMES:
        values = inverse if name == "inv" else rows[name]
        if name in ("tri", "cur"):
            values = np.asarray(values, dtype=np.int64).astype(np.int32).view(np.uint32)
        else:
            values = packets.bits(values)
        columns.append(np.asarray(values).reshape(len(values), width))
    lanes = np.concatenate(columns, axis=1)  # (count, 27)
    count = len(lanes)
    return (
        lanes.reshape(count // VEC, VEC, -1)
        .transpose(0, 2, 1)
        .reshape(count // VEC, -1)
    )


def partition(items, index, count):
    """(start, length) of rank `index` of `count` in `items` items, as the runtime's."""
    base, extra = divmod(items, count)
    return index * base + min(index, extra), base + (index < extra)


def contiguous(pages, cores):
    """Directory (cores, 16): a contiguous partition of the pages for each core."""
    directory = np.zeros((cores, K["kDirBytes"] // 4), dtype=np.uint32)
    for core in range(cores):
        start, count = partition(pages, core, cores)
        directory[core, K["kDirFirst"]] = start
        directory[core, K["kDirCount"]] = count
    return directory


def denormal(x):
    x = np.asarray(x, dtype=np.float32)
    return (x != 0) & (np.abs(x) < np.finfo(np.float32).tiny)


def compare(rows, got, edge):
    """Mismatch counts per output (bits; float compare and zero-sign aside)."""
    want = expected(rows)
    det_ok = (want["det"] != 0) & ~denormal(want["det"])
    det_ok &= np.isfinite(want["det_inv"]) & ~denormal(want["det_inv"])
    report = {}
    for name in OUT_NAMES:
        found = got[name]
        wanted = np.asarray(want[name])
        wanted_bits = (
            wanted.astype(np.int32).view(np.uint32)
            if name == "new_tri"
            else packets.bits(wanted)
        )
        found_bits = found.view(np.uint32)
        mask = np.ones(len(found), bool)
        if name in ("det_inv", "u", "v", "t"):
            mask = det_ok  # 1/det is not defined on the card for det 0
        bad = (found_bits != wanted_bits) & mask
        found_float, wanted_float = found.view(np.float32), (
            wanted.astype(np.float32) if name != "new_tri" else wanted
        )
        zero_sign = (
            bad & (found_float == 0) & (wanted_float == 0)
            if name != "new_tri"
            else bad & False
        )
        entry = {
            "compared": int(mask.sum()),
            "differ_bits": int(bad.sum()),
            "differ_only_zero_sign": int(zero_sign.sum()),
            "differ_edge_rows": int((bad & edge).sum()),
        }
        first_bad = np.nonzero(bad & ~zero_sign)[0][:5]
        entry["examples"] = [
            {
                "row": int(row),
                "got": int(found_bits[row]),
                "want": int(wanted_bits[row]),
            }
            for row in first_bad
        ]
        report[name] = entry
    hit_card = got["stack"].view(np.float32) != np.inf
    report["box_hit_flag_differs"] = int((hit_card != want["hit"]).sum())
    report["box_hits"] = int(want["hit"].sum())
    report["tri_hits"] = int(want["ok"].sum())
    report["det_zero_or_out_of_range"] = int((~det_ok).sum())
    intermediate = [
        want["det"],
        want["det_inv"],
        want["u"],
        want["v"],
        want["t"],
        want["entry"],
    ]
    report["rows_with_denormal_values"] = int(
        np.any([denormal(values) for values in intermediate], axis=0).sum()
    )
    report["ties_tested"] = int((want["ok"] & (want["t"] == rows["best"])).sum())
    return report


def math_inputs(generator, count):
    """Rows of the division and square root check: dividend (any sign, zeros among
    them), divisor positive and radicand positive, mantissas random, near one or
    short, exponents within the kernels' ranges; quotients stay normal. Returns
    (dividend, divisor, radicand) float32 arrays."""
    exponents = generator.integers(-60, 61, size=(3, count))
    mantissas = generator.integers(0, 1 << 23, size=(3, count))
    kind = generator.integers(0, 4, size=(3, count))
    mantissas = np.where(kind == 1, mantissas & 0xF, mantissas)  # short mantissas
    mantissas = np.where(
        kind == 2, (1 << 23) - 1 - (mantissas & 0xF), mantissas
    )  # just below a power
    word = ((exponents + 127) << 23 | mantissas).astype(np.uint32)
    dividend, divisor, radicand = word.view(np.float32)
    dividend = np.where(generator.random(count) < 0.5, -dividend, dividend)
    dividend[generator.random(count) < 0.01] = F(0.0)
    radicand = np.where(
        generator.random(count) < 0.3, np.abs(dividend) + F(1e-30), radicand
    ).astype(np.float32)
    return dividend.astype(np.float32), divisor.astype(np.float32), radicand


def math_pages(dividend, divisor, radicand):
    """Input pages (P, kMaInVectors * 32) of the check; the count a multiple of 32."""
    cols = np.stack(
        [packets.bits(dividend), packets.bits(divisor), packets.bits(radicand)], axis=1
    )
    count = len(dividend)
    return (
        cols.reshape(count // VEC, VEC, 3).transpose(0, 2, 1).reshape(count // VEC, -1)
    )


def math_compare(dividend, divisor, radicand, out):
    """Bits that differ from numpy's float32 dividend / divisor and sqrt(radicand)."""
    out = out.reshape(-1, K["kMaOutVectors"], VEC).transpose(1, 0, 2)
    quotient, root = out[K["kMaQ"]].reshape(-1), out[K["kMaS"]].reshape(-1)
    want_q, want_s = packets.bits(dividend / divisor), packets.bits(np.sqrt(radicand))
    report = {}
    for name, got, want in (("divide", quotient, want_q), ("sqrt", root, want_s)):
        bad = got != want
        first_bad = np.nonzero(bad)[0][:5]
        report[name] = {
            "compared": int(got.size),
            "differ_bits": int(bad.sum()),
            "examples": [
                {"row": int(row), "got": int(got[row]), "want": int(want[row])}
                for row in first_bad
            ],
        }
    return report
