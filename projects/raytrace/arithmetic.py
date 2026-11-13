"""The arithmetic check's host side: inputs from a real scene and edge cases, the
expected outputs with the reference's own expressions, and the comparison.

Pure numpy, no card (the driver is `arith.py`).
"""

import numpy as np

from projects.raytrace import packets
from projects.raytrace.reference import harness
from ref import bvh, native, raytrace, scenes

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
    par = np.full(len(nodes), -1, dtype=np.int64)
    inner = np.nonzero(~bvh.is_leaf(nodes[:, 6]))[0]
    par[nodes[inner, 6].astype(np.int64)] = inner
    par[nodes[inner, 7].astype(np.int64)] = inner
    return par


def ray_pool(scene_name, rng, n_pixels):
    """(o, d, tmax, hit leaf) of primary, shadow and bounce rays of random pixels."""
    data = native.data_dir()
    tree = harness.ensure_tree(scene_name)
    scene = scenes.load_scene(data / f"{scene_name}.npz")
    pix = rng.choice(scenes.WIDTH * scenes.HEIGHT, n_pixels, replace=False)
    o, d = raytrace.camera_rays(scene["camera"], pix)
    hit = raytrace.trace(tree, o, d, raytrace.TMAX)
    h = np.nonzero(hit["leaf"] >= 0)[0]
    sub = {k: v[h] for k, v in hit.items()}
    p = o[h] + d[h] * sub["t"][:, None]
    n, _ = raytrace.surface(scene, tree, sub, d[h])
    o2 = p + n * scene["eps"][0]
    to_light = scene["light"] - o2
    db = raytrace.bounce_directions(pix[h], n)
    shadow = raytrace.trace(tree, o2, to_light, F(1.0), any_hit=True)
    bounce = raytrace.trace(tree, o2, db, raytrace.TMAX)
    oo = np.concatenate([o, o2, o2])
    dd = np.concatenate([d, to_light, db])
    tmax = np.concatenate(
        [
            np.full(len(o), raytrace.TMAX),
            np.ones(len(h), F),
            np.full(len(h), raytrace.TMAX),
        ]
    ).astype(F)
    leaf = np.concatenate([hit["leaf"], shadow["leaf"], bounce["leaf"]])
    return scene, tree, oo.astype(F), dd.astype(F), tmax, leaf


def scene_inputs(rng, n, scene_name):
    """n input rows from rays of a scene, the nodes and triangles near their hits."""
    _, tree, o, d, tmax, leaf = ray_pool(scene_name, rng, 4096)
    nodes, tris = tree["nodes"], tree["tris"]
    n_leaves = len(tris)
    par = parents(nodes)
    r = rng.integers(0, len(o), n)
    hit_leaf = leaf[r]
    # Triangles: the hit one, its neighbours in leaf order, or a random one.
    kind = rng.integers(0, 4, n)
    base = np.where(hit_leaf >= 0, hit_leaf, rng.integers(0, n_leaves, n))
    tl = np.where(kind == 0, base, base + rng.choice([-1, 1], n))
    tl = np.where(kind == 3, rng.integers(0, n_leaves, n), tl)
    tl = np.clip(tl, 0, n_leaves - 1)
    # Boxes: the hit leaf's node or one of its ancestors, or a random node.
    node = tree["leaf_node"][np.maximum(hit_leaf, 0)].astype(np.int64)
    for _ in range(int(rng.integers(0, 6))):
        up = par[node]
        node = np.where((up >= 0) & (rng.random(n) < 0.6), up, node)
    node = np.where(
        (hit_leaf < 0) | (rng.random(n) < 0.3), rng.integers(0, len(nodes), n), node
    )
    lo, hi = bvh.node_boxes(nodes)
    rows = {
        "o": o[r], "d": d[r], "bmin": lo[node], "bmax": hi[node],
        "v0": tris[tl, 0:3], "e1": tris[tl, 3:6], "e2": tris[tl, 6:9],
        "tri": tree["leaf_tri"][tl].astype(np.int64),
        "best": tmax[r],
    }  # fmt: skip
    return rows, tree


def edge_inputs(rng, rows, tree):
    """Edge cases written over a share of the rows (in place); returns their mask."""
    n = len(rows["o"])
    kind = rng.integers(0, 10, n)
    edge = rng.random(n) < 0.25
    o, d = rows["o"].copy(), rows["d"].copy()
    axis = rng.integers(0, 3, n)
    at = np.arange(n)
    # 0: a direction component zero (+-0) or below the clamp.
    m = edge & (kind == 0)
    tiny = np.array([0.0, -0.0, 1e-13, -1e-13, 1e-12, -1e-12, 1e-30], F)
    d[at[m], axis[m]] = tiny[rng.integers(0, len(tiny), m.sum())]
    # 1: origin on a box face; 2: a flat box with the origin in its plane.
    m = edge & (kind == 1)
    face = np.where(rng.random(n) < 0.5, rows["bmin"][at, axis], rows["bmax"][at, axis])
    o[at[m], axis[m]] = face[m]
    m = edge & (kind == 2)
    rows["bmax"][at[m], axis[m]] = rows["bmin"][at[m], axis[m]]
    o[at[m], axis[m]] = rows["bmin"][at[m], axis[m]]
    # 3, 4: aimed at a vertex or at an edge point of its triangle (grazing hits).
    v0, e1, e2 = rows["v0"], rows["e1"], rows["e2"]
    m = edge & (kind == 3)
    corner = np.stack([v0, v0 + e1, v0 + e2])[rng.integers(0, 3, n), at]
    d[m] = (corner - o)[m]
    m = edge & (kind == 4)
    s = rng.random(n).astype(F)[:, None]
    point = np.where(rng.random(n)[:, None] < 0.5, v0 + e1 * s, v0 + e2 * s)
    d[m] = (point - o)[m]
    # 5: a direction in the triangle's plane (det 0 or nearly).
    m = edge & (kind == 5)
    d[m] = np.where(rng.random(n)[:, None] < 0.5, e1, e2)[m]
    # 6: signed zeros in the origin.
    m = edge & (kind == 6)
    o[at[m], axis[m]] = np.where(rng.random(m.sum()) < 0.5, F(0.0), F(-0.0))
    rows["o"], rows["d"] = o.astype(F), d.astype(F)
    return edge


def bests(rng, rows):
    """best: tmax, the pair's own t (a tie), its neighbours, or random; cur: -1, the
    triangle index or one off."""
    n = len(rows["o"])
    ok, t, _, _ = triangle(rows)
    kind = rng.integers(0, 6, n)
    best = rows["best"].copy()
    tf = np.where(ok, t, F(1.0)).astype(F)
    best = np.where(kind == 1, tf, best)
    best = np.where(kind == 2, np.nextafter(tf, F(np.inf)), best)
    best = np.where(kind == 3, np.nextafter(tf, F(0)), best)
    best = np.where(kind == 4, rng.uniform(0, 50, n).astype(F), best)
    rows["best"] = best.astype(F)
    ck = rng.integers(0, 4, n)
    tri = rows["tri"]
    rows["cur"] = np.where(
        ck == 0, -1, np.where(ck == 1, tri - 1, np.where(ck == 2, tri + 1, tri))
    )


def triangle(rows):
    """The reference's triangle test on the rows (one triangle per row)."""
    g = np.concatenate([rows["v0"], rows["e1"], rows["e2"]], axis=1)
    with np.errstate(all="ignore"):
        return raytrace.triangle_test(g, np.arange(len(g)), rows["o"], rows["d"])


def expected(rows):
    """The outputs in numpy float32, in OUT_NAMES order (with the reference's code)."""
    o, d = rows["o"], rows["d"]
    inv = raytrace.inverse_direction(d)
    box = np.concatenate([rows["bmin"], rows["bmax"]], axis=1)
    best = rows["best"]
    with np.errstate(all="ignore"):
        hit, entry = raytrace.box_test(box, np.arange(len(o)), o, inv, best)
        t0 = (box[:, :3] - o) * inv
        t1 = (box[:, 3:] - o) * inv
        far = np.maximum(t0, t1)
        exit_ = (
            np.minimum(np.minimum(far[:, 0], far[:, 1]), far[:, 2])
            * raytrace.SLAB_SCALE
        )
        m = np.minimum(exit_, best)
        stack = np.where(hit, entry, F(np.inf))
        pv = raytrace.cross(d, rows["e2"])
        det = raytrace.dot(rows["e1"], pv)
        det_inv = F(1.0) / det
        ok, t, u, v = triangle(rows)
    cur, tri = rows["cur"], rows["tri"]
    better = ok & (t < best)
    better |= ok & (t == best) & (cur >= 0) & (tri < cur)
    return {
        "entry": entry, "exit": m, "stack": stack, "det": det, "det_inv": det_inv,
        "u": u, "v": v, "t": t,
        "new_best": np.where(better, t, best),
        "new_tri": np.where(better, tri, cur).astype(np.int32),
        "new_u": np.where(better, u, F(0)), "new_v": np.where(better, v, F(0)),
        "hit": hit, "ok": ok, "inv": inv,
    }  # fmt: skip


def input_pages(rows):
    """Input pages (P, kArInVectors * 32) uint32; rows padded to a multiple of 32."""
    cols = []
    inv = raytrace.inverse_direction(rows["d"])
    for name, width in IN_NAMES:
        x = inv if name == "inv" else rows[name]
        if name in ("tri", "cur"):
            x = np.asarray(x, dtype=np.int64).astype(np.int32).view(np.uint32)
        else:
            x = packets.bits(x)
        cols.append(np.asarray(x).reshape(len(x), width))
    a = np.concatenate(cols, axis=1)  # (n, 27)
    n = len(a)
    return a.reshape(n // VEC, VEC, -1).transpose(0, 2, 1).reshape(n // VEC, -1)


def share(n, index, count):
    """(start, length) of member `index` of `count` in n items, as the runtime's."""
    base, extra = divmod(n, count)
    return index * base + min(index, extra), base + (index < extra)


def contiguous(pages, cores):
    """Directory (cores, 16): a contiguous share of the pages for each core."""
    directory = np.zeros((cores, K["kDirBytes"] // 4), dtype=np.uint32)
    for c in range(cores):
        start, count = share(pages, c, cores)
        directory[c, K["kDirFirst"]] = start
        directory[c, K["kDirCount"]] = count
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
        g = got[name]
        w = np.asarray(want[name])
        w_bits = (
            w.astype(np.int32).view(np.uint32) if name == "new_tri" else packets.bits(w)
        )
        g_bits = g.view(np.uint32)
        mask = np.ones(len(g), bool)
        if name in ("det_inv", "u", "v", "t"):
            mask = det_ok  # 1/det is not defined on the card for det 0
        bad = (g_bits != w_bits) & mask
        gf, wf = g.view(np.float32), w.astype(np.float32) if name != "new_tri" else w
        zero_sign = bad & (gf == 0) & (wf == 0) if name != "new_tri" else bad & False
        entry = {
            "compared": int(mask.sum()),
            "differ_bits": int(bad.sum()),
            "differ_only_zero_sign": int(zero_sign.sum()),
            "differ_edge_rows": int((bad & edge).sum()),
        }
        idx = np.nonzero(bad & ~zero_sign)[0][:5]
        entry["examples"] = [
            {"row": int(i), "got": int(g_bits[i]), "want": int(w_bits[i])} for i in idx
        ]
        report[name] = entry
    hit_card = got["stack"].view(np.float32) != np.inf
    report["box_hit_flag_differs"] = int((hit_card != want["hit"]).sum())
    report["box_hits"] = int(want["hit"].sum())
    report["tri_hits"] = int(want["ok"].sum())
    report["det_zero_or_out_of_range"] = int((~det_ok).sum())
    inter = [
        want["det"],
        want["det_inv"],
        want["u"],
        want["v"],
        want["t"],
        want["entry"],
    ]
    report["rows_with_denormal_values"] = int(
        np.any([denormal(x) for x in inter], axis=0).sum()
    )
    report["ties_tested"] = int((want["ok"] & (want["t"] == rows["best"])).sum())
    return report


def math_inputs(rng, n):
    """Rows of the division and square root check: a (any sign, zeros among them), b
    positive and x positive, mantissas random, near one or short, exponents within the
    kernels' ranges; quotients stay normal. Returns (a, b, x) float32 arrays."""
    exp = rng.integers(-60, 61, size=(3, n))
    man = rng.integers(0, 1 << 23, size=(3, n))
    kind = rng.integers(0, 4, size=(3, n))
    man = np.where(kind == 1, man & 0xF, man)  # short mantissas
    man = np.where(kind == 2, (1 << 23) - 1 - (man & 0xF), man)  # just below a power
    word = ((exp + 127) << 23 | man).astype(np.uint32)
    a, b, x = word.view(np.float32)
    a = np.where(rng.random(n) < 0.5, -a, a)
    a[rng.random(n) < 0.01] = F(0.0)
    x = np.where(rng.random(n) < 0.3, np.abs(a) + F(1e-30), x).astype(np.float32)
    return a.astype(np.float32), b.astype(np.float32), x


def math_pages(a, b, x):
    """Input pages (P, kMaInVectors * 32) of the check; n a multiple of 32."""
    cols = np.stack([packets.bits(a), packets.bits(b), packets.bits(x)], axis=1)
    n = len(a)
    return cols.reshape(n // VEC, VEC, 3).transpose(0, 2, 1).reshape(n // VEC, -1)


def math_compare(a, b, x, out):
    """Bits that differ from numpy's float32 a / b and sqrt(x)."""
    out = out.reshape(-1, K["kMaOutVectors"], VEC).transpose(1, 0, 2)
    q, s = out[K["kMaQ"]].reshape(-1), out[K["kMaS"]].reshape(-1)
    want_q, want_s = packets.bits(a / b), packets.bits(np.sqrt(x))
    report = {}
    for name, got, want in (("divide", q, want_q), ("sqrt", s, want_s)):
        bad = got != want
        idx = np.nonzero(bad)[0][:5]
        report[name] = {
            "compared": int(got.size),
            "differ_bits": int(bad.sum()),
            "examples": [
                {"row": int(i), "got": int(got[i]), "want": int(want[i])} for i in idx
            ],
        }
    return report
