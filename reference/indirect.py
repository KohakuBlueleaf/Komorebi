"""Indirect light, the term `indirect` of `raster.rasterize_lighted`: elements, their
form-factor matrix in the card's files, the sky by 25 harmonics, a sparse voxel field.
`build` writes (`python -m reference.public indirect <scene>`), `irradiance` reads;
projects/raster/reference/README.md, "Indirect light"."""

import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from reference import bvh, native, public, raster, rng
from reference import lights as light_data

F = np.float32
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
RAYTRACE = native.ROOT / "projects" / "raytrace" / "reference"
# Elements: the default count, the sampling seed, and the probe (rays, seed, dead share,
# near distance in eps) that drops elements seeing only the void or buried.
ELEMENTS = 50_000
SAMPLE_SEED = 31
PROBE_RAYS, PROBE_SEED, PROBE_DEAD, PROBE_NEAR = 32, 41, 0.97, 10.0
# Transport: 32-element units, leaves of at most LEAF_UNITS units, admissibility
# (separation at least ETA), the low-rank tolerance, rows of F a strip of the build.
UNIT = 32
LEAF_UNITS = 8
ETA = 1.0
TOLERANCE = 1e-2
STRIP = 128
# Sky: rays a transfer row and their seed; harmonics of bands 0 to 4, fitted to the map
# on a SKY_GRID x 2 SKY_GRID grid (32 patches measured worse: README).
SKY_RAYS, SKY_SEED = 256, 51
SKY_TERMS = 25
SKY_GRID = 256
# Field: radius and lookup shift in voxels, reach of the fill of empty directions.
FIELD_RADIUS = 1.5
FIELD_SHIFT = 0.5
FILL_REACH = 2.0
# The field as the card holds it (pack_field): a texel a (voxel, direction), bit 31 has
# data, then red, green, blue codes of these bits at these shifts.
PACK_HAS = 1 << 31
PACK_BITS = (11, 11, 9)
PACK_SHIFTS = (20, 9, 0)
DIRECTIONS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=F
)
# Truth: rays a pixel and the seeds of the three path segments.
TRUTH_RAYS = 256
TRUTH_SEEDS = (21, 22, 23)
_LIBRARY = {}
_MODELS = {}


def threads():
    """Threads of the batch ray routine: $KR_THREADS, else the CPUs this process may
    use, at most 8."""
    if "KR_THREADS" in os.environ:
        return int(os.environ["KR_THREADS"])
    return max(1, min(8, len(os.sched_getaffinity(0))))


def library():
    """The batch ray routine (native/trace_batch.cpp), built once as a library."""
    if "lib" not in _LIBRARY:
        source = native.NATIVE / "trace_batch.cpp"
        path = native.build(str(source), "libtrace_batch.so", (), [RAYTRACE], True)
        if path is None:
            raise RuntimeError("indirect light traces rays in C++: no compiler found")
        lib = ctypes.CDLL(str(path))
        pointer = ctypes.c_void_p
        lib.trace_batch.argtypes = [pointer] * 6 + [ctypes.c_long, ctypes.c_int]
        lib.trace_batch.argtypes += [pointer] * 4 + [ctypes.c_int]
        _LIBRARY["lib"] = lib
    return _LIBRARY["lib"]


def trace(tree, origin, direction, tmax, any_hit=False):
    """t, triangle (-1: none), u, v (n,) of rays origin + t direction with t < tmax (a
    number or (n,)); the reference tree walk and triangle test, threaded."""
    o = np.ascontiguousarray(origin, F)
    d = np.ascontiguousarray(direction, F)
    n = len(o)
    tm = np.ascontiguousarray(np.broadcast_to(np.asarray(tmax, F), (n,)), F)
    t, leaf, u, v = (
        np.empty(n, F),
        np.empty(n, np.int32),
        np.empty(n, F),
        np.empty(n, F),
    )
    library().trace_batch(
        *(tree[k].ctypes.data for k in ("nodes", "tris", "leaf_tri")),
        o.ctypes.data,
        d.ctypes.data,
        tm.ctypes.data,
        n,
        int(any_hit),
        *(a.ctypes.data for a in (t, leaf, u, v)),
        threads(),
    )
    tri = np.where(leaf >= 0, tree["leaf_tri"][np.maximum(leaf, 0)], -1)
    return t, tri, u, v


def open_tree(mesh, materials, tree=None):
    """The ray tree with glass open (over the triangles that are not `transparent`,
    leaf_tri naming the scene's triangles); `tree` (all triangles) is kept when there
    is no glass. Masked triangles are opaque."""
    glass = materials["kind"][materials["material"]] == raster.TRANSPARENT
    if tree is None or glass.any():
        keep = np.nonzero(~glass)[0]
        tree = bvh.build(
            {"positions": mesh["positions"], "indices": mesh["indices"][keep]}
        )
        tree["leaf_tri"] = keep[tree["leaf_tri"]].astype(np.int32)
    return {k: np.ascontiguousarray(v) for k, v in tree.items()}


def triangle_geometry(mesh):
    """Corners (T, 3, 3), areas (T,) and unit normals (T, 3) by the winding, float64."""
    p = mesh["positions"][mesh["indices"]].astype(np.float64)
    c = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = 0.5 * np.linalg.norm(c, axis=1)
    return p, area, c / np.maximum(2 * area, 1e-30)[:, None]


def triangle_colour(materials, coarse=4):
    """Per triangle a linear colour (T, 3): the albedo texture at the centroid's (u, v)
    in the level `coarse` above the last (16 x 16) times the base colour; else the base.
    """
    m = materials["material"]
    tex = materials["textures"][m, raster.ALBEDO]
    base = materials["base_colour"][m]
    out = base.astype(F).copy()
    has = tex >= 0
    if has.any() and len(materials["mip"]):
        tx = tex[has]
        level = np.maximum(materials["levels"][tx] - 1 - coarse, 0)
        uv = materials["uv"][has].mean(1).astype(F)
        texel = raster.fetch(materials, tx, level, uv[:, 0], uv[:, 1])
        out[has] = raster.TEXEL_LINEAR[texel[:, :3]] * base[has]
    return out


def cosine_directions(index, normal, seed):
    """Cosine-weighted unit directions (n, 3) float32 about normals (n, 3), one per hash
    index (n,), in the frame of `lights.bake_directions`."""
    u1 = rng.uniform(index, 0, seed)
    u2 = rng.uniform(index, 1, seed)
    r, phi = np.sqrt(u1), F(2 * np.pi) * u2
    normal = normal.astype(F)
    helper = np.where(
        (np.abs(normal[:, 0]) < F(0.9))[:, None], F([1, 0, 0]), F([0, 1, 0])
    )
    t = raster.normalized(raster.cross3(helper, normal))
    b = raster.cross3(normal, t)
    up = np.sqrt(np.maximum(F(1) - u1, F(0)))
    return (t * (r * np.cos(phi))[:, None] + b * (r * np.sin(phi))[:, None]) + (
        normal * up[:, None]
    )


def sample_elements(mesh, materials, count):
    """`count` points by area: element k at area (k + 0.5) A / count of the cumulative
    triangle area (glass has none), uniform in its triangle from the hash of k.
    Position, normal, area A / count, albedo, emission (emissive: colour, albedo 0)."""
    p, area, normal = triangle_geometry(mesh)
    kind = materials["kind"][materials["material"]]
    area = np.where(kind == raster.TRANSPARENT, 0.0, area)
    cumulative = np.cumsum(area)
    total = cumulative[-1]
    k = np.arange(count)
    tri = np.minimum(
        np.searchsorted(cumulative, (k + 0.5) * (total / count)), len(area) - 1
    )
    su = np.sqrt(rng.uniform(k, 0, SAMPLE_SEED).astype(np.float64))
    u2 = rng.uniform(k, 1, SAMPLE_SEED).astype(np.float64)
    b0, b1 = 1 - su, su * (1 - u2)
    q = p[tri]
    pos = (
        q[:, 0] * b0[:, None] + q[:, 1] * b1[:, None] + q[:, 2] * (1 - b0 - b1)[:, None]
    )
    colour = triangle_colour(materials)[tri]
    glow = (kind[tri] == raster.EMISSIVE)[:, None]
    elements = {
        "position": pos.astype(F),
        "normal": normal[tri].astype(F),
        "area": np.full(count, total / count, F),
        "albedo": np.where(glow, F(0), colour).astype(F),
        "emission": np.where(glow, colour, F(0)).astype(F),
        "triangle": tri.astype(np.int32),
        "sample": k.astype(np.int32),
    }
    return elements, float(np.sqrt(total / count))


def probe(tree, elements, eps):
    """Per element the shares of PROBE_RAYS cosine rays from p + eps n that escape and
    that hit within PROBE_NEAR eps; dead when the two add up to more than PROBE_DEAD."""
    p, n = elements["position"], elements["normal"]
    escape, near = np.zeros(len(p)), np.zeros(len(p))
    index = elements["sample"].astype(np.int64) * PROBE_RAYS
    for k in range(PROBE_RAYS):
        d = cosine_directions(index + k, n, PROBE_SEED)
        t, tri, _, _ = trace(tree, p + n * F(eps), d, light_data.raytrace.TMAX)
        escape += tri < 0
        near += (tri >= 0) & (t < F(PROBE_NEAR * eps))
    return escape / PROBE_RAYS, near / PROBE_RAYS


def subset(elements, keep):
    """The elements at the indices or mask `keep`."""
    return {k: v[keep] for k, v in elements.items()}


def form_factor_rows(tree, elements, rows, eps):
    """F (len(rows), N) float32 with visibility: max(0, cos_i) max(0, cos_j) A_j /
    (pi r^2 + A_j) (the near field clamped as a disc of area A_j seen on its axis), 0 on
    the diagonal or where an any-hit ray between the points moved eps along their
    normals hits. Also the rays traced."""
    p = elements["position"].astype(np.float64)
    n = elements["normal"].astype(np.float64)
    area = elements["area"].astype(np.float64)
    count = len(p)
    i = np.repeat(rows, count)
    j = np.tile(np.arange(count), len(rows))
    d = p[j] - p[i]
    r2 = np.einsum("ij,ij->i", d, d)
    r = np.sqrt(np.maximum(r2, 1e-30))
    ci = np.einsum("ij,ij->i", n[i], d) / r
    cj = -np.einsum("ij,ij->i", n[j], d) / r
    value = np.maximum(ci, 0) * np.maximum(cj, 0) * area[j] / (np.pi * r2 + area[j])
    value[i == j] = 0
    live = np.nonzero(value > 0)[0]
    if len(live):
        o = p[i[live]] + n[i[live]] * eps
        e = p[j[live]] + n[j[live]] * eps
        _, tri, _, _ = trace(tree, o, e - o, 1.0, any_hit=True)
        value[live[tri >= 0]] = 0
    return value.reshape(len(rows), count).astype(F), len(live)


def dense_form_factors(tree, elements, eps, folder, log=print):
    """The whole F (N, N) float32 with visibility as a memmap <folder>/F.npy, built by
    strips of STRIP rows; a build stopped part way goes on from <folder>/F_rows.npy (the
    strips done) when <folder>/F_elements.npz holds the same elements."""
    folder.mkdir(parents=True, exist_ok=True)
    count = len(elements["position"])
    path, done_path = folder / "F.npy", folder / "F_rows.npy"
    same = (folder / "F_elements.npz").exists() and path.exists() and done_path.exists()
    if same:
        with np.load(folder / "F_elements.npz") as f:
            same = all(np.array_equal(f[k], elements[k]) for k in elements)
    if same:
        matrix, done = open_memmap(path, mode="r+"), np.load(done_path)
    else:
        np.savez(folder / "F_elements.npz", **elements)
        matrix = open_memmap(path, mode="w+", dtype=F, shape=(count, count))
        done = np.zeros(-(-count // STRIP), dtype=bool)
    started, rays = time.perf_counter(), 0
    for s in np.nonzero(~done)[0]:
        rows = np.arange(s * STRIP, min(count, (s + 1) * STRIP))
        matrix[rows], traced = form_factor_rows(tree, elements, rows, eps)
        rays += traced
        done[s] = True
        if s % 20 == 19 or s == len(done) - 1:
            matrix.flush()
            np.save(done_path, done)
            seconds = time.perf_counter() - started
            log(f"F strip {s + 1}/{len(done)}: {rays} rays, {seconds:.0f} s")
    matrix.flush()
    np.save(done_path, done)
    return matrix, rays


def separation(tree, a, b):
    """Distance between the boxes of nodes a and b over the larger box diagonal."""
    lo, hi = tree["lo"], tree["hi"]
    gap = np.maximum(0, np.maximum(lo[a] - hi[b], lo[b] - hi[a]))
    diagonal = np.maximum(
        np.linalg.norm(hi[a] - lo[a], axis=-1), np.linalg.norm(hi[b] - lo[b], axis=-1)
    )
    return np.linalg.norm(gap, axis=-1) / np.maximum(diagonal, 1e-9)


def cluster_tree(position, leaf_units=LEAF_UNITS):
    """A binary tree of whole UNIT-element units (count a multiple of UNIT): a node of
    u units cuts its longest box axis at u // 2 units (sorted, stable); u <= leaf_units
    is a leaf. Order `perm` and per node start, end, children (-1), box lo, hi."""
    count = len(position)
    perm = np.arange(count)
    nodes = []

    def make(start, end):
        k = len(nodes)
        nodes.append([start, end, -1, -1])
        units = (end - start) // UNIT
        if units > leaf_units:
            idx = perm[start:end]
            p = position[idx]
            axis = int(np.argmax(p.max(0) - p.min(0)))
            perm[start:end] = idx[np.argsort(p[:, axis], kind="stable")]
            middle = start + (units // 2) * UNIT
            nodes[k][2] = make(start, middle)
            nodes[k][3] = make(middle, end)
        return k

    make(0, count)
    nodes = np.array(nodes, dtype=np.int64)
    p = position.astype(np.float64)
    lo = np.array([p[perm[s:e]].min(0) for s, e in nodes[:, :2]])
    hi = np.array([p[perm[s:e]].max(0) for s, e in nodes[:, :2]])
    return {"perm": perm, "start": nodes[:, 0], "end": nodes[:, 1]} | {
        "child": nodes[:, 2:4],
        "lo": lo,
        "hi": hi,
    }


def partition(tree, eta=ETA):
    """Node pairs (a, b) from (root, root): admissible when separation >= eta; two
    leaves a near block; one leaf: the other's two children; else the four pairs."""
    admissible, near = [], []
    a, b = np.array([0]), np.array([0])
    child = tree["child"]
    while len(a):
        ok = separation(tree, a, b) >= eta
        admissible.append(np.stack([a[ok], b[ok]], 1))
        a, b = a[~ok], b[~ok]
        la, lb = child[a, 0] < 0, child[b, 0] < 0
        near.append(np.stack([a[la & lb], b[la & lb]], 1))
        pairs = []
        m = la & ~lb
        pairs += [(a[m], child[b[m], 0]), (a[m], child[b[m], 1])]
        m = ~la & lb
        pairs += [(child[a[m], 0], b[m]), (child[a[m], 1], b[m])]
        m = ~la & ~lb
        pairs += [(child[a[m], i], child[b[m], j]) for i in (0, 1) for j in (0, 1)]
        a = np.concatenate([p[0] for p in pairs])
        b = np.concatenate([p[1] for p in pairs])
    return np.concatenate(admissible), np.concatenate(near)


def hierarchical(matrix, tree, tolerance=TOLERANCE, eta=ETA, log=print):
    """The matrix as blocks in the card's layout (y_row = x_row @ W), zero blocks
    dropped: dense n x m (near, or a rank-k pair not smaller), else V (n x kp) then U^T
    (kp x m) with sigma^(1/2) each, k the singular values above tolerance x the largest,
    kp = 32 ceil(k / 32). Blocks dict, weights."""
    admissible, near = partition(tree, eta)
    start, end = tree["start"], tree["end"]
    rows = []
    for a, b in near:
        block = matrix[start[a] : end[a], start[b] : end[b]]
        if block.any():
            rows.append(
                (start[a], end[a], start[b], end[b], 0, 0, block.T.copy(), None)
            )
    near_count = len(rows)
    r0, r1 = start[admissible[:, 0]], end[admissible[:, 0]]
    c0, c1 = start[admissible[:, 1]], end[admissible[:, 1]]
    shapes = sorted(set(zip((r1 - r0).tolist(), (c1 - c0).tolist())), key=np.prod)
    started = time.perf_counter()
    found = {}
    for m, n in shapes:
        group = np.nonzero((r1 - r0 == m) & (c1 - c0 == n))[0]
        per = max(1, 40_000_000 // (m * n))
        for s in range(0, len(group), per):
            ids = group[s : s + per]
            stack = np.stack([matrix[r0[i] : r1[i], c0[i] : c1[i]] for i in ids])
            live = stack.reshape(len(ids), -1).any(axis=1)
            if not live.any():
                continue
            stack = stack[live]
            u, sigma, vt = np.linalg.svd(stack, full_matrices=False)
            for q, i in enumerate(ids[live]):
                k = int((sigma[q] > tolerance * sigma[q][0]).sum())
                if m * n < k * (m + n):
                    found[i] = (0, 0, stack[q].T.copy(), None)
                    continue
                root = np.sqrt(sigma[q][:k])
                kp = -(-k // UNIT) * UNIT
                v = np.zeros((n, kp), F)
                v[:, :k] = vt[q][:k].T * root
                ut = np.zeros((kp, m), F)
                ut[:k] = (u[q][:, :k] * root).T
                found[i] = (1, k, v, ut)
        if m * n >= 256 * 256:
            seconds = time.perf_counter() - started
            log(f"SVD {m} x {n}: {len(group)} blocks, {seconds:.0f} s")
    rows += [(r0[i], r1[i], c0[i], c1[i], *found[i]) for i in sorted(found)]
    kind = np.array([row[4] for row in rows], dtype=np.int8)
    rank = np.array([row[5] for row in rows], dtype=np.int32)
    sizes = [row[6].size + (0 if row[7] is None else row[7].size) for row in rows]
    offset_a = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    weights = np.empty(int(np.sum(sizes)), F)
    for row, a in zip(rows, offset_a):
        weights[a : a + row[6].size] = row[6].ravel()
        if row[7] is not None:
            weights[a + row[6].size : a + row[6].size + row[7].size] = row[7].ravel()
    m_of = np.array([row[1] - row[0] for row in rows], dtype=np.int64)
    n_of = np.array([row[3] - row[2] for row in rows], dtype=np.int64)
    padded = (-(-rank // UNIT) * UNIT).astype(np.int32)
    blocks = {
        "row0": np.array([row[0] for row in rows], dtype=np.int32),
        "row1": np.array([row[1] for row in rows], dtype=np.int32),
        "col0": np.array([row[2] for row in rows], dtype=np.int32),
        "col1": np.array([row[3] for row in rows], dtype=np.int32),
        "kind": kind,
        "rank": rank,
        "rank_padded": padded,
        "offset_a": offset_a,
        "offset_b": offset_a + np.where(kind == 1, n_of * padded, m_of * n_of),
        "scale": np.ones(len(rows), F),
        "near": np.arange(len(rows)) < near_count,
    }
    return blocks, weights


BLOCK_KEYS = ("row0", "row1", "col0", "col1", "kind", "rank_padded", "offset_a")
BLOCK_KEYS += ("offset_b", "scale")


def apply(blocks, weights, x, left=None):
    """y = M x (N, C) float32 from the blocks: the loop of the card files' README,
    block after block; `left` rounds the left operand (x's and t's rows) as the card."""
    y = np.zeros(x.shape, dtype=F)
    xq = (left(x) if left else x).astype(F)
    for r0, r1, c0, c1, kd, kp, oa, ob, s in zip(*(blocks[k] for k in BLOCK_KEYS)):
        m, n, xr = r1 - r0, c1 - c0, xq[c0:c1].T
        if kd == 0:
            y[r0:r1] += ((xr @ weights[oa : oa + n * m].reshape(n, m)) * s).T
            continue
        t = xr @ weights[oa : oa + n * kp].reshape(n, kp)
        if left:
            t = left(t)
        y[r0:r1] += ((t @ weights[ob : ob + kp * m].reshape(kp, m)) * s).T
    return y


def round_bits(a, bits):
    """a rounded to `bits` significant bits, half to even (bfloat16: 8)."""
    m, e = np.frexp(np.asarray(a, np.float64))
    return np.ldexp(np.rint(np.ldexp(m, bits)), e - bits).astype(F)


def bfp8_b(a):
    """a (size a multiple of 16) in bfp8_b by the host encode: each 16 consecutive
    numbers share their largest exponent, magnitudes rounded half to even to 7 bits."""
    g = np.asarray(a, F).reshape(-1, 16).astype(np.float64)
    top = np.abs(g).max(1, keepdims=True)
    _, e = np.frexp(np.where(top > 0, top, 1.0))
    unit = np.ldexp(1.0, e.astype(np.int64) - 1 - 6)
    q = np.minimum(np.rint(np.abs(g) / unit), 127)
    return np.where(top > 0, np.sign(g) * q * unit, 0.0).astype(F).reshape(np.shape(a))


def card_apply(blocks, weights, x, chunk=1 << 24):
    """The card's numbers emulated: every weight bfp8_b, x's and t's rows bfloat16 (the
    left operand at HiFi4 keeps 8 bits), sums float32."""
    quantised = np.empty_like(weights)
    for s in range(0, len(weights), chunk):
        quantised[s : s + chunk] = bfp8_b(weights[s : s + chunk])
    return apply(blocks, quantised, x, lambda v: round_bits(v, 8))


def sh_basis(d):
    """The 25 real spherical harmonics of bands 0 to 4 (..., 25) at unit directions d
    (..., 3), float64; the first 9 are `lights.sh_basis` (z the polar axis)."""
    d = np.asarray(d, np.float64)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    band3 = [0.590044 * y * (3 * x2 - y2), 2.890611 * x * y * z]
    band3 += [0.457046 * y * (5 * z2 - 1), 0.373176 * z * (5 * z2 - 3)]
    band3 += [0.457046 * x * (5 * z2 - 1), 1.445306 * z * (x2 - y2)]
    band3 += [0.590044 * x * (x2 - 3 * y2)]
    band4 = [2.503343 * x * y * (x2 - y2), 1.770131 * y * z * (3 * x2 - y2)]
    band4 += [0.946175 * x * y * (7 * z2 - 1), 0.669047 * y * z * (7 * z2 - 3)]
    band4 += [0.105786 * (35 * z2 * z2 - 30 * z2 + 3), 0.669047 * x * z * (7 * z2 - 3)]
    band4 += [0.473087 * (x2 - y2) * (7 * z2 - 1), 1.770131 * x * z * (x2 - 3 * y2)]
    band4 += [0.625836 * (x2 * (x2 - 3 * y2) - y2 * (3 * x2 - y2))]
    return np.concatenate([light_data.sh_basis(d), np.stack(band3 + band4, -1)], -1)


def sky_map(lights):
    """The octahedral map (map frame) with its sun removed as the sky's coefficients
    have it: texels within SUN_CONE_DEG of the brightest texel take the mean of those
    from there to SUN_RING_DEG. Kept per map."""
    octa = lights["octahedral"]
    key = ("sky", id(octa))
    if key in _MODELS and _MODELS[key][0] is octa:
        return _MODELS[key][1]
    side = octa.shape[0]
    c = (np.arange(side) + 0.5) / side * 2 - 1
    v, u = np.meshgrid(c, c, indexing="ij")
    d = light_data.octahedral_direction(u, v)
    peak = d.reshape(-1, 3)[np.argmax(octa.reshape(-1, 3) @ LUMINANCE)]
    angle = np.degrees(np.arccos(np.clip(d @ peak, -1.0, 1.0)))
    cone = angle < light_data.SUN_CONE_DEG
    ring = (angle >= light_data.SUN_CONE_DEG) & (angle < light_data.SUN_RING_DEG)
    clean = octa.astype(F).copy()
    clean[cone] = octa[ring].mean(0)
    _MODELS[key] = (octa, clean)
    return clean


def sky_radiance(lights, d):
    """The sky's radiance (n, 3) along world directions d (n, 3): the nearest texel of
    `sky_map`, d turned by -turn about y (as `raster.environment_radiance`)."""
    return raster.environment_radiance({**lights, "octahedral": sky_map(lights)}, d)


def sky_coefficients(lights):
    """The sky's 25 coefficients (SKY_TERMS, 3) float32: the least-squares fit of
    `sh_basis` to `sky_radiance` (the turned map, the sun removed) at the texel centres
    of a SKY_GRID x 2 SKY_GRID equirectangular grid, weighted by their solid angle."""
    d, solid = light_data.texel_directions(SKY_GRID, 2 * SKY_GRID)
    d, root = d.reshape(-1, 3), np.sqrt(solid.ravel())[:, None]
    radiance = sky_radiance(lights, d.astype(F)).astype(np.float64)
    fit = np.linalg.lstsq(sh_basis(d) * root, radiance * root, rcond=None)[0]
    return fit.astype(F)


def transfer(tree, elements, eps, rays=SKY_RAYS):
    """Each element's transfer row (N, SKY_TERMS) float32: pi / rays x the sum of
    `sh_basis` over the escaping rays of `rays` cosine rays (hash index sample x rays +
    k, SKY_SEED) from p + eps n, any hit. Its sky irradiance: the row times the sky's
    coefficients."""
    p, n = elements["position"], elements["normal"]
    out = np.zeros((len(p), SKY_TERMS))
    index = elements["sample"].astype(np.int64) * rays
    origin = p + n * F(eps)
    for k in range(rays):
        d = cosine_directions(index + k, n, SKY_SEED)
        _, tri, _, _ = trace(tree, origin, d, light_data.raytrace.TMAX, any_hit=True)
        free = np.nonzero(tri < 0)[0]
        out[free] += sh_basis(d[free])
    return (out * (np.pi / rays)).astype(F)


def voxel_keys(ijk, dims):
    """Keys (...,) of voxel indices (..., 3) in a grid of dims (3,); -1 outside it."""
    inside = ((ijk >= 0) & (ijk < dims)).all(-1)
    key = (ijk[..., 0] * dims[1] + ijk[..., 1]) * dims[2] + ijk[..., 2]
    return np.where(inside, key, -1)


def build_field(position, normal, spacing, chunk=2000):
    """The sparse field: voxels of side h = spacing; (voxel, d) = sum w E / sum w over
    the elements within R = 1.5 h of its centre, w = max(0, n.d) (1 - (r/R)^2)^2; an
    empty (voxel, d) takes the row of the nearest voxel within 2 h whose d has elements.
    Rows (voxel x 6 + d), columns (elements), weights; keys, has (nv, 6), grid."""
    h = float(spacing)
    radius = FIELD_RADIUS * h
    p, nrm = position.astype(np.float64), normal.astype(np.float64)
    origin = np.floor(p.min(0) / h) * h - 2 * h
    dims = (np.ceil((p.max(0) - origin) / h) + 3).astype(np.int64)
    reach = int(np.ceil(2 * radius / h)) + 2
    offsets = np.stack(np.meshgrid(*(np.arange(reach),) * 3, indexing="ij"), -1)
    offsets = offsets.reshape(-1, 3)
    keys, columns, weights = [], [], []
    for s in range(0, len(p), chunk):
        q, n = p[s : s + chunk], nrm[s : s + chunk]
        corner = np.floor((q - radius - origin) / h - 0.5).astype(np.int64)
        ijk = corner[:, None, :] + offsets[None]
        r = np.linalg.norm(origin + (ijk + 0.5) * h - q[:, None, :], axis=-1) / radius
        fall = np.where(r < 1, (1 - r * r) ** 2, 0.0)
        w = (
            fall[:, :, None]
            * np.maximum(n @ DIRECTIONS.T.astype(np.float64), 0)[:, None]
        )
        e, o = np.nonzero(fall > 0)
        keys.append(voxel_keys(ijk[e, o], dims))
        columns.append(e + s)
        weights.append(w[e, o])
    key, column, w = (
        np.concatenate(keys),
        np.concatenate(columns),
        np.concatenate(weights),
    )
    some = (w.sum(1) > 0) & (key >= 0)
    voxels, voxel = np.unique(key[some], return_inverse=True)
    rows = (voxel[:, None] * 6 + np.arange(6)).ravel()
    column, w = np.repeat(column[some], 6), w[some].ravel()
    rows, column, w = rows[w > 0], column[w > 0], w[w > 0]
    total = np.bincount(rows, w, len(voxels) * 6)
    own = (total > 0).reshape(-1, 6)
    w = w / total[rows]
    rows, column, w, source = fill_rows(voxels, own, dims, rows, column, w)
    order = np.argsort(rows, kind="stable")
    has = np.zeros(len(voxels) * 6, dtype=bool)
    has[rows] = True
    return {
        "field_origin": origin.astype(F),
        "field_spacing": np.float32(h),
        "field_dims": dims,
        "field_keys": voxels.astype(np.int64),
        "field_has": has.reshape(-1, 6),
        "field_own": own,
        "field_rows": rows[order].astype(np.int64),
        "field_columns": column[order].astype(np.int32),
        "field_weights": w[order].astype(F),
        "field_filled_from": source,
    }


def fill_rows(voxels, own, dims, rows, column, w):
    """The rows of `build_field` with each empty (voxel, d) given the row of the nearest
    voxel (centre distance up to FILL_REACH voxels; ties: the first offset in (distance,
    x, y, z) order) whose d has elements. Also each filled (voxel, d)'s source voxel."""
    span = int(FILL_REACH)
    grid = np.arange(-span, span + 1)
    offsets = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    d2 = (offsets * offsets).sum(1)
    keep = (d2 > 0) & (d2 <= FILL_REACH**2)
    offsets = offsets[keep][np.lexsort((*offsets[keep].T[::-1], d2[keep]))]
    ijk = np.stack(np.unravel_index(voxels, tuple(dims)), -1)
    source = np.full(own.shape, -1, dtype=np.int64)
    for offset in offsets:
        missing = (~own) & (source < 0)
        if not missing.any():
            break
        key = voxel_keys(ijk + offset, dims)
        at = np.minimum(np.searchsorted(voxels, key), len(voxels) - 1)
        found = (voxels[at] == key) & (key >= 0)
        take = missing & found[:, None] & own[at]
        source[take] = np.broadcast_to(at[:, None], own.shape)[take]
    order = np.argsort(rows, kind="stable")
    start = np.searchsorted(rows[order], np.arange(own.size + 1))
    v, d = np.nonzero(source >= 0)
    src_row = source[v, d] * 6 + d
    count = start[src_row + 1] - start[src_row]
    pick = np.repeat(start[src_row], count) + (
        np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    )
    new_rows = np.repeat(v * 6 + d, count)
    rows = np.concatenate([rows, new_rows])
    column = np.concatenate([column, column[order][pick]])
    w = np.concatenate([w, w[order][pick]])
    return rows, column, w, source.astype(np.int32)


def field_values(model, e):
    """The field's values (nv, 6, 3) float32 for element irradiance e (N, 3)."""
    rows, cols, w = (model[f"field_{k}"] for k in ("rows", "columns", "weights"))
    size = len(model["field_keys"]) * 6
    out = np.stack(
        [np.bincount(rows, w * e[cols, c], size) for c in range(e.shape[1])], -1
    )
    return out.reshape(-1, 6, e.shape[1]).astype(F)


def lookup(model, values, position, normal):
    """I (n, 3) float32 at points P with normals n, and where any voxel had data: at
    Q = P + n h / 2 the 8 voxel centres around Q, trilinear; per axis a the direction
    d(a, sign n_a) over the voxels with data, weights renormalised; I = sum n_a^2 V_a
    over the axes that found data, over the sum of their n_a^2."""
    h, origin, dims = model["field_spacing"], model["field_origin"], model["field_dims"]
    keys, has = model["field_keys"], model["field_has"]
    n = normal.astype(F)
    g = ((position.astype(F) + n * (F(FIELD_SHIFT) * h)) - origin) / h - F(0.5)
    base = np.floor(g).astype(np.int64)
    t = g - base.astype(F)
    total = np.zeros((len(n), values.shape[2]), F)
    axes = np.zeros(len(n), F)
    for a in range(3):
        d = 2 * a + (n[:, a] < 0)
        num = np.zeros_like(total)
        den = np.zeros(len(n), F)
        for corner in np.ndindex(2, 2, 2):
            key = voxel_keys(base + np.array(corner), dims)
            at = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
            ok = (keys[at] == key) & (key >= 0) & has[at, d]
            wc = np.prod(np.where(np.array(corner) == 1, t, F(1) - t), 1) * ok
            num += wc[:, None] * values[at, d]
            den += wc
        got = den > 0
        na2 = n[:, a] * n[:, a]
        total += np.where(got, na2 / np.where(got, den, F(1)), F(0))[:, None] * num
        axes += np.where(got, na2, F(0))
    found = axes > 0
    return total / np.where(found, axes, F(1))[:, None], found


def lamp_irradiance(lights, position, normal):
    """The point lights' irradiance (n, 3) float32, without shadows (as a pixel's)."""
    out = np.zeros((len(position), 3), F)
    for k in range(len(lights["lamp_intensity"])):
        e, _ = light_data.point_irradiance(
            lights["lamp_intensity"][k],
            lights["size"],
            lights["radius"],
            lights["lamp_position"][k],
            position,
            normal,
        )
        out = out + lights["lamp_colour"][k] * e[:, None]
    return out


def element_light(lights, model, bounces=light_data.BOUNCES):
    """For the current lights each element's sky irradiance (transfer row x sky),
    L0 = emission x emissive_scale + albedo / pi (sun with the shadow map's lookup + the
    lamps + sky) and the `bounces` passes E1 = M L0, E_k = M (albedo / pi E_(k-1))
    (`passes`); the hierarchical apply, float32.
    """
    p, n = model["position"], model["normal"]
    sun = light_data.sun_shadow(lights, p, n) * np.maximum(
        raster.dot3(n, lights["sun_direction"][None, :]), F(0)
    )
    sky = np.maximum(model["transfer"] @ sky_coefficients(lights), F(0)).astype(F)
    direct = (
        sun[:, None] * lights["sun_irradiance"] + lamp_irradiance(lights, p, n)
    ) + sky
    scale = F(lights["emissive_scale"])
    l0 = model["emission"] * scale + model["albedo"] * F(1 / np.pi) * direct
    l0 = np.where(model["real"][:, None], l0, F(0)).astype(F)
    passes = []
    for _ in range(light_data.bounce_count(bounces)):
        x = l0 if not passes else model["albedo"] * F(1 / np.pi) * passes[-1]
        passes.append(apply(model["blocks"], model["weights"], x))
    return {"sky": sky, "direct": direct, "l0": l0, "passes": passes}


def field_source(light):
    """What the field holds per element for `element_light`'s result: E1 + ... +
    E_bounces + the element's own sky irradiance (with no pass, the sky alone)."""
    if not light["passes"]:
        return light["sky"]
    total = light["passes"][0]
    for e in light["passes"][1:]:
        total = total + e
    return total + light["sky"]


def model_path(name, data):
    """<public>/<scene>_indirect.npz (a view's: its scene's)."""
    if public.is_view(name):
        name = public.views()[name][0]
    return public.public_dir(data) / f"{name}_indirect.npz"


def transport_dir(name, data):
    """<public>/<scene>_transport/, the card's matrix files."""
    return public.public_dir(data) / f"{name}_transport"


def load_model(path):
    """The elements, transfer rows and field of an indirect file and its matrix files
    (weights mapped from the file), kept per path."""
    path = str(path)
    if path not in _MODELS:
        with np.load(path) as f:
            model = {k: f[k] for k in f.files}
        folder = Path(path).with_name(
            Path(path).name.replace("_indirect.npz", "_transport")
        )
        with np.load(folder / "blocks.npz") as f:
            model["blocks"] = {k: f[k] for k in f.files}
        model["weights"] = np.fromfile(folder / "weights.f32", dtype=F)
        _MODELS[path] = model
    return _MODELS[path]


def signature(lights):
    """What the element light depends on: the shadow map (by identity), the sun, the
    turn, the lamps and the emissive scale."""
    small = [lights[k] for k in ("sun_direction", "sun_irradiance", "turn")]
    small += [lights[k] for k in ("lamp_position", "lamp_colour", "lamp_intensity")]
    small += [np.float32(lights["emissive_scale"]), lights["shadow_camera"]]
    return b"".join(np.asarray(a, F).tobytes() for a in small)


def pack_field(values, has):
    """The field as 32-bit texels (nv, 6) uint32 and its scale: bit 31 where the
    (voxel, d) has data, then per channel (PACK_SHIFTS, PACK_BITS) the code
    round(sqrt(v / scale) x (2^bits - 1)), v clamped at 0; the scale is the field's
    largest value (float32; 1 when none is above 0)."""
    v = np.maximum(np.asarray(values, np.float64), 0.0)
    top = float(np.float32(v.max(initial=0.0)))
    scale = np.float32(top if top > 0 else 1.0)
    texels = np.where(has, np.uint32(PACK_HAS), np.uint32(0)).astype(np.uint32)
    for c, (bits, shift) in enumerate(zip(PACK_BITS, PACK_SHIFTS)):
        most = (1 << bits) - 1
        code = np.minimum(np.rint(np.sqrt(v[..., c] / float(scale)) * most), most)
        texels |= np.where(has, code.astype(np.uint32) << np.uint32(shift), 0).astype(
            np.uint32
        )
    return texels, scale


def unpack_field(texels, scale):
    """The values (nv, 6, 3) float32 and has (nv, 6) of pack_field's texels, as the
    card decodes them: c = code x F(1 / (2^bits - 1)), v = (c c) scale; 0 without data.
    """
    texels = np.asarray(texels, np.uint32)
    has = (texels >> np.uint32(31)) != 0
    out = np.zeros(texels.shape + (3,), F)
    for c, (bits, shift) in enumerate(zip(PACK_BITS, PACK_SHIFTS)):
        most = (1 << bits) - 1
        code = ((texels >> np.uint32(shift)) & np.uint32(most)).astype(F)
        unit = code * F(1.0 / most)
        out[..., c] = np.where(has, (unit * unit) * F(scale), F(0))
    return out, has


def field_for(lights):
    """The model and the field's values (nv, 6, 3) for the lights (`lights["indirect"]`
    names the file): E1 + ... + E_bounces (`lights["bounces"]`, default 2) + each
    element's own sky irradiance, through the field's weights; remade when the lights or
    the count change, never the files. `lights["indirect_packed"]`: as the card holds
    them (pack_field, then unpack_field)."""
    if "indirect" not in lights:
        raise ValueError(
            "the term `indirect` needs the scene's indirect file: write it with "
            "python -m reference.public indirect <scene> (lights.load finds it)"
        )
    bounces = light_data.bounce_count(lights.get("bounces", light_data.BOUNCES))
    model = load_model(lights["indirect"])
    key = (signature(lights), id(lights["shadow_map"]), id(lights["octahedral"]))
    key += (bounces,)
    cache = model.setdefault("cache", [])
    values = None
    for entry in cache:
        if entry[0] == key and entry[1] is lights["shadow_map"]:
            values = entry[2]
    if values is None:
        light = element_light(lights, model, bounces)
        values = field_values(model, field_source(light))
        cache[:] = cache[-3:] + [(key, lights["shadow_map"], values)]
    if lights.get("indirect_packed"):
        values, _ = unpack_field(*pack_field(values, model["field_has"]))
    return model, values


def irradiance(lights, position, normal):
    """The indirect irradiance (n, 3) float32 a pixel receives (`lookup` of
    `field_for`); where no voxel has data, the lights' `ambient`."""
    model, values = field_for(lights)
    e, found = lookup(model, values, position, normal)
    ambient = np.asarray(lights["ambient"], dtype=F)
    return np.where(found[:, None], e, ambient).astype(F)


def padded(elements, count):
    """The elements and zero-area copies of the first up to `count` (a multiple of
    UNIT), and which are real."""
    extra = count - len(elements["position"])
    out = {
        k: np.concatenate([v, np.repeat(v[:1], extra, 0)]) for k, v in elements.items()
    }
    out["area"][-extra or len(out["area"]) :] = 0
    out["sample"][-extra or len(out["sample"]) :] = -1
    return out, np.arange(count) < count - extra


def build(name, data, log=print):
    """Writes <public>/<scene>_indirect.npz and the card's files in <public>/<scene>
    _transport/ (README "Indirect light"); the dense F is kept in <public>/<scene>
    _indirect_work/ and a stopped build goes on from it. Returns the counts."""
    entry = public.table()[name]["lighting"]
    mesh, _ = public.load_mesh(name, data)
    materials, _ = public.load_materials(name, data)
    lights = light_data.load(name, data)
    times, counts, clock = {}, {}, [time.perf_counter()]

    def step(label):
        now = time.perf_counter()
        times[label] = round(now - clock[0], 1)
        clock[0] = now
        log(f"{name}: {label} {times[label]} s")

    tree = open_tree(mesh, materials, light_data.scene_tree(name, data, mesh))
    p = mesh["positions"].astype(np.float64)
    eps = public.EPS_SCALE * float(np.linalg.norm(p.max(0) - p.min(0)))
    step("tree")
    sampled, spacing = sample_elements(mesh, materials, entry.get("elements", ELEMENTS))
    step("sample")
    escape, near = probe(tree, sampled, eps)
    live = subset(sampled, escape + near <= PROBE_DEAD)
    counts |= {
        "sampled": len(escape),
        "probe_dead": int((escape + near > PROBE_DEAD).sum()),
    }
    step("probe")
    work = public.public_dir(data) / f"{name}_indirect_work"
    dense, rays = dense_form_factors(tree, live, eps, work, log)
    counts["form_factor_rays"] = rays
    step("form_factors")
    row_sum = np.concatenate(
        [dense[s : s + 4096].sum(1) for s in range(0, len(dense), 4096)]
    )
    keep = np.nonzero(row_sum > 0)[0]
    counts["empty_rows"] = int(len(row_sum) - len(keep))
    count = -(-len(keep) // UNIT) * UNIT
    elements, real = padded(subset(live, keep), count)
    cluster = cluster_tree(elements["position"])
    perm = cluster["perm"]
    elements, real = subset(elements, perm), real[perm]
    source = np.where(real, keep[np.minimum(perm, len(keep) - 1)], -1)
    matrix = np.zeros((count, count), F)
    columns = np.nonzero(real)[0]
    for s in range(0, count, 2048):
        rows = np.arange(s, min(count, s + 2048))
        rows = rows[real[rows]]
        part = np.asarray(dense[np.sort(source[rows])])
        part = part[np.argsort(np.argsort(source[rows]))]
        matrix[rows[:, None], columns[None, :]] = part[:, source[columns]] * F(np.pi)
    del dense
    counts |= {
        "kept": int(real.sum()),
        "elements": count,
        "spacing": spacing,
        "eps": eps,
    }
    step("order")
    blocks, weights = hierarchical(matrix, cluster, log=log)
    step("hierarchical")
    model = elements | {"real": real, "blocks": blocks, "weights": weights}
    model["transfer"] = np.where(real[:, None], transfer(tree, elements, eps), F(0))
    step("transfer")
    model |= build_field(elements["position"][real], elements["normal"][real], spacing)
    model["field_columns"] = np.nonzero(real)[0][model["field_columns"]].astype(
        np.int32
    )
    step("field")
    x = element_light(lights, model)["l0"]
    exact = np.concatenate(
        [matrix[s : s + 2048].astype(np.float64) @ x for s in range(0, count, 2048)]
    )
    del matrix
    result = {
        "x": x,
        "y_exact": exact.astype(F),
        "y_float": apply(blocks, weights, x),
        "y_q8": card_apply(blocks, weights, x),
    }
    step("checks")
    counts |= matrix_counts(blocks) | field_counts(model)
    counts["seconds"] = times
    write(name, data, model, result, source, counts, live["sample"])
    step("write")
    counts["seconds"] = times
    return counts


def matrix_counts(blocks):
    """Blocks, tiles and numbers of the stored matrix."""
    m = (blocks["row1"] - blocks["row0"]).astype(np.int64)
    n = (blocks["col1"] - blocks["col0"]).astype(np.int64)
    lr = blocks["kind"] == 1
    kp = blocks["rank_padded"].astype(np.int64)
    tiles = np.where(
        lr, (kp // UNIT) * (m // UNIT + n // UNIT), (m // UNIT) * (n // UNIT)
    )
    return {
        "blocks": int(len(lr)),
        "blocks_low_rank": int(lr.sum()),
        "blocks_dense_near": int(blocks["near"].sum()),
        "blocks_dense_admissible": int((~lr & ~blocks["near"]).sum()),
        "rank_median": float(np.median(blocks["rank"][lr])) if lr.any() else 0.0,
        "tiles": int(tiles.sum()),
        "tiles_dense": int(tiles[~lr].sum()),
        "tiles_low_rank": int(tiles[lr].sum()),
    }


def field_counts(model):
    """Voxels and weights of the field."""
    return {
        "voxels": int(len(model["field_keys"])),
        "voxel_directions_own": int(model["field_own"].sum()),
        "voxel_directions_filled": int((model["field_filled_from"] >= 0).sum()),
        "field_weights": int(len(model["field_weights"])),
    }


def write(name, data, model, result, source, counts, samples):
    """The files: the indirect file (elements in the matrix's order, transfer rows,
    field) and the card's folder (blocks.npz, weights.f32, x, y_*, perm.npy, README)."""
    folder = transport_dir(name, data)
    folder.mkdir(parents=True, exist_ok=True)
    blocks, weights = model["blocks"], model["weights"]
    np.savez(folder / "blocks.npz", **blocks)
    weights.tofile(folder / "weights.f32")
    for key, value in result.items():
        np.save(folder / f"{key}.npy", value.astype(F))
    perm = np.where(source >= 0, samples[np.maximum(source, 0)], -1).astype(np.int32)
    np.save(folder / "perm.npy", perm)
    (folder / "README.md").write_text(transport_readme(name, counts, len(weights)))
    keep = {k: v for k, v in model.items() if k not in ("blocks", "weights", "cache")}
    out = model_path(name, data)
    np.savez(out, **keep, counts=json.dumps(counts))
    sizes = {p.name: p.stat().st_size for p in sorted(folder.iterdir())}
    counts["bytes"] = {out.name: out.stat().st_size} | sizes
    np.savez(out, **keep, counts=json.dumps(counts))


def transport_readme(name, counts, numbers):
    """The card folder's README: what the files hold and the apply that reproduces
    y_float.npy."""
    return TRANSPORT_README.format(
        name=name,
        elements=counts["elements"],
        units=counts["elements"] // UNIT,
        kept=counts["kept"],
        blocks=counts["blocks"],
        low_rank=counts["blocks_low_rank"],
        dense=counts["blocks"] - counts["blocks_low_rank"],
        numbers=numbers,
        tiles=counts["tiles"],
        tiles_dense=counts["tiles_dense"],
        tiles_low_rank=counts["tiles_low_rank"],
    )


TRANSPORT_README = """# {name}: the light-transport matrix in the card's layout

Written by `python -m reference.public indirect {name}` (reference/indirect.py; the
definition: projects/raster/reference/README.md, "Indirect light"). The card computes
`y = M x`: `M = pi F`, F the elements' form factor with visibility and the near-field
clamp, so `y` is each element's bounced irradiance for the outgoing light `x`.
Tolerance 1e-2 (of each block's largest singular value), admissibility separation >= 1,
no scaling (every `scale` is 1.0).

## Elements

- N = {elements:,} = {units:,} units of 32: the {kept:,} kept elements and zero-area
  copies (their rows and columns of M are 0, their x is 0). Order: the cluster tree,
  every cluster whole 32-element units, leaves of at most 256 elements; the same order
  as the elements of `{name}_indirect.npz`.
- `perm.npy` (N,) int32: each position's sample number k (element k of the area
  sampling); -1 for a copy.
- `x.npy` (N, 3) float32: L0 of the scene's own lights, RGB. `y_exact.npy`: the dense
  apply (every entry of M, float64 sums). `y_float.npy`: the apply below from these
  files, float32, bit for bit. `y_q8.npy`: the card's numbers emulated: every stored
  number bfp8_b (host encode: one exponent for each 16 consecutive numbers of a stored
  row), x's and t's rows bfloat16 as the LEFT operand at HiFi4, float32 sums.

## Stored TRANSPOSED: the card's product is y_row = x_row @ W

- dense block of M rows `row0:row1` (receivers, m = row1 - row0) and columns `col0:col1`
  (sources, n = col1 - col0): stored as `Wt` = n x m, row-major, at `offset_a`.
- low-rank block `M_block = U V^T`: `V` (n x kp) row-major at `offset_a`, then `U^T`
  (kp x m) at `offset_b = offset_a + n * kp`; kp = 32 ceil(k / 32), zero padded; each
  carries sigma^(1/2) of the block's SVD.
- m, n and kp are multiples of 32: every stored matrix is whole 32 x 32 tiles, plain
  row-major in the file; every block starts at a multiple of 32 numbers.

## blocks.npz ({blocks:,} blocks: {low_rank:,} low rank, {dense:,} dense)

| key | type | meaning |
| --- | --- | --- |
| `row0`, `row1` | int32 | rows of M (receivers), multiples of 32 |
| `col0`, `col1` | int32 | columns of M (sources), multiples of 32 |
| `kind` | int8 | 0 dense, 1 low rank |
| `rank` | int32 | low rank: k (0 for dense) |
| `rank_padded` | int32 | low rank: kp (0 for dense) |
| `offset_a` | int64 | in float32 numbers into `weights.f32`: the dense `Wt`, or `V` |
| `offset_b` | int64 | low rank: `U^T` (dense: the end of its block, unused) |
| `scale` | float32 | multiply the block's product by it (all 1.0) |
| `near` | bool | a leaf x leaf near block (dense) |

## weights.f32 ({numbers:,} float32, little endian, no header)

Blocks in the order of `blocks.npz`. Tiles in all: {tiles:,} (dense {tiles_dense:,}; low
rank {tiles_low_rank:,}).

## Apply (reproduces y_float.npy exactly)

```python
import numpy as np
b = np.load("blocks.npz")
W = np.fromfile("weights.f32", np.float32)
x = np.load("x.npy")
y = np.zeros_like(x)
keys = ("row0", "row1", "col0", "col1", "kind", "rank_padded", "offset_a", "offset_b")
for r0, r1, c0, c1, kd, kp, oa, ob, s in zip(*(b[k] for k in keys + ("scale",))):
    m, n, xr = r1 - r0, c1 - c0, x[c0:c1].T
    if kd == 0:
        y[r0:r1] += ((xr @ W[oa : oa + n * m].reshape(n, m)) * s).T
    else:
        t = xr @ W[oa : oa + n * kp].reshape(n, kp)
        y[r0:r1] += ((t @ W[ob : ob + kp * m].reshape(kp, m)) * s).T
print("max |y - y_float| =", np.abs(y - np.load("y_float.npy")).max())
```
"""


def truth(scene, materials, lights, frame, tree, rays=TRUTH_RAYS, step=2, log=print):
    """The gathered truth, kept out of `rasterize_lighted`: at the frame's visible
    non-emissive pixels (every `step`-th) `rays` cosine rays about the pixel's normal
    from its point moved eps along the triangle's normal facing the eye; returns per
    pixel the sky with visibility and one and two bounces of the sources (`path`)."""
    cam, eps = scene["camera"], F(scene["eps"][0])
    tri = frame["tri"][::step, ::step]
    kind = materials["kind"][materials["material"]]
    iy, ix = np.nonzero((tri >= 0) & (kind[np.maximum(tri, 0)] != raster.EMISSIVE))
    t = tri[iy, ix]
    px, py = (step * ix).astype(F) + F(0.5), (step * iy).astype(F) + F(0.5)
    position = raster.world_points(cam, px, py, frame["depth"][::step, ::step][iy, ix])
    normal = frame["parts"]["normal"][::step, ::step][iy, ix].astype(F)
    _, _, flat = triangle_geometry(scene)
    g = flat[t].astype(F)
    g = np.where((raster.dot3(g, cam[None, 0:3] - position) < 0)[:, None], -g, g)
    surfaces = {
        "flat": flat.astype(F),
        "colour": triangle_colour(materials),
        "kind": kind,
    }
    origin = position + g * eps
    pixel = (iy * (tri.shape[1]) + ix).astype(np.int64)
    sums = {k: np.zeros((len(iy), 3)) for k in ("sky", "bounce1", "bounce2")}
    escape = np.zeros(len(iy))
    started = time.perf_counter()
    for k0 in range(0, rays, 4):
        ks = np.arange(k0, min(rays, k0 + 4))
        ii = np.repeat(np.arange(len(iy)), len(ks))
        index = np.repeat(pixel * rays, len(ks)) + np.tile(ks, len(iy))
        path(
            tree, lights, surfaces, origin[ii], normal[ii], index, eps, ii, sums, escape
        )
        if k0 % 64 == 60:
            log(
                f"truth: {k0 + 4} of {rays} rays, {time.perf_counter() - started:.0f} s"
            )
    scale = np.pi / rays
    out = {k: v * scale for k, v in sums.items()}
    return (
        out
        | {"iy": iy, "ix": ix, "position": position, "normal": normal}
        | {"escape": escape / rays}
    )


def hit_direct(tree, lights, surfaces, origin, d, t, tri, eps):
    """At ray hits: the point, its normal facing the ray, the triangle's albedo (0 when
    emissive) and its outgoing light from the sources but the sky: emission x scale +
    albedo / pi (sun with a shadow ray + the lamps)."""
    point = origin + d * t[:, None]
    n = surfaces["flat"][tri]
    n = np.where((raster.dot3(n, d) > 0)[:, None], -n, n)
    glow = (surfaces["kind"][tri] == raster.EMISSIVE)[:, None]
    colour = surfaces["colour"][tri]
    albedo = np.where(glow, F(0), colour)
    s = lights["sun_direction"].astype(F)
    cos = np.maximum(raster.dot3(n, s[None, :]), F(0))
    lit = np.nonzero(cos > 0)[0]
    sun = np.zeros(len(point), F)
    if len(lit):
        o = point[lit] + n[lit] * eps
        _, block, _, _ = trace(tree, o, np.broadcast_to(s, o.shape), 1e30, any_hit=True)
        sun[lit] = cos[lit] * (block < 0)
    e = sun[:, None] * lights["sun_irradiance"] + lamp_irradiance(lights, point, n)
    out = np.where(
        glow, colour * F(lights["emissive_scale"]), albedo * F(1 / np.pi) * e
    )
    return point + n * eps, n, albedo, out


def path(tree, lights, surfaces, origin, normal, index, eps, owner, sums, escape):
    """One path per ray, added to `sums` by `owner`: escaping, the sky (`sky`); at the
    first hit its direct light, then one cosine ray from it: the sky x albedo (both
    `bounce1`); at its hit albedo1 x direct, then one more ray: the sky x both albedos.
    """
    seed1, seed2, seed3 = TRUTH_SEEDS
    d = cosine_directions(index, normal, seed1)
    t, tri, _, _ = trace(tree, origin, d, 1e30)
    out = tri < 0
    np.add.at(sums["sky"], owner[out], sky_radiance(lights, d[out]))
    escape += np.bincount(owner[out], minlength=len(escape))
    h = np.nonzero(~out)[0]
    o1, n1, a1, l1 = hit_direct(
        tree, lights, surfaces, origin[h], d[h], t[h], tri[h], eps
    )
    np.add.at(sums["bounce1"], owner[h], l1)
    d2 = cosine_directions(index[h], n1, seed2)
    t2, tri2, _, _ = trace(tree, o1, d2, 1e30)
    out2 = tri2 < 0
    np.add.at(
        sums["bounce1"], owner[h[out2]], a1[out2] * sky_radiance(lights, d2[out2])
    )
    g = np.nonzero(~out2)[0]
    hit2 = hit_direct(tree, lights, surfaces, o1[g], d2[g], t2[g], tri2[g], eps)
    o2, n2, a2, l2 = hit2
    np.add.at(sums["bounce2"], owner[h[g]], a1[g] * l2)
    d3 = cosine_directions(index[h[g]], n2, seed3)
    _, tri3, _, _ = trace(tree, o2, d3, 1e30, any_hit=True)
    out3 = np.nonzero(tri3 < 0)[0]
    gain = a1[g[out3]] * a2[out3]
    np.add.at(sums["bounce2"], owner[h[g[out3]]], gain * sky_radiance(lights, d3[out3]))
