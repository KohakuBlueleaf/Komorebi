"""Arithmetic check: the fill's per-pixel expressions (a px + b py + c, each step
rounded to float32, compared in the total order) on the card against numpy, bit for bit,
on a real scene's records and edge cases. See the README's "Arithmetic check".
`python -m projects.raster.arith --n 1000000 [--scene s10k] [--json FILE]`"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from projects.raster import card, lists
from ref import native, raster, scenes

K = lists.K
F = np.float32
LANES = K["kArithLanes"]
NAMES = ("a px", "b py", "sum", "e", "e > T", "e > d")


def total_key(x):
    """Signed keys that order float32 bit patterns as the vector unit's SFPGT does."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    return np.where(u >> 31, -(u & 0x7FFFFFFF) - 1, u)


def ftz(x):
    """Denormals to zero of the same sign."""
    x = np.asarray(x, dtype=np.float32)
    tiny = (np.abs(x) < np.finfo(np.float32).tiny) & (x != 0)
    return np.where(tiny, np.copysign(F(0), x), x).astype(np.float32)


def expected(a, b, c, px, py, t, d, flush=False):
    """The six outputs in numpy float32 (IEEE; with flush, denormals flushed)."""
    f = ftz if flush else (lambda v: np.asarray(v, dtype=np.float32))
    with np.errstate(all="ignore"):
        p1 = f(f(a) * f(px))
        p2 = f(f(b) * f(py))
        s = f(p1 + p2)
        e = f(s + f(c))
    gt_t = total_key(e) > total_key(t)
    gt_d = total_key(e) > total_key(d)
    return [p1, p2, s, e, gt_t.astype(np.uint32), gt_d.astype(np.uint32)]


def scene_inputs(rng, n, scene):
    """n inputs from the planes of a scene's records near their boxes."""
    native.ensure_scene(scene)
    recs = raster.setup(scenes.load_scene(native.data_dir() / f"{scene}.npz"))
    m = len(recs["tri"])
    i = rng.integers(0, m, n)
    kind = rng.integers(0, 7, n)  # 0-2 edges, 3 the 1/z plane, 4-6 colour channels
    coef = np.empty((n, 3), dtype=np.float32)
    for k in range(3):
        sel = kind == k
        coef[sel] = np.stack(
            [recs["a"][i[sel], k], recs["b"][i[sel], k], recs["c"][i[sel], k]], axis=1
        )
    sel = kind == 3
    coef[sel] = recs["w_planes"][i[sel]]
    for ch in range(3):
        sel = kind == 4 + ch
        coef[sel] = recs["c_planes"][i[sel], :, ch]
    box = recs["box"][i].astype(np.int64)
    col = np.clip(rng.integers(box[:, 0] - 2, box[:, 1] + 3), 0, raster.WIDTH - 1)
    row = np.clip(rng.integers(box[:, 2] - 2, box[:, 3] + 3), 0, raster.HEIGHT - 1)
    px = (col.astype(np.float32) + F(0.5)).astype(np.float32)
    py = (row.astype(np.float32) + F(0.5)).astype(np.float32)
    tl = np.where(
        kind < 3, recs["top_left"][i, np.minimum(kind, 2)], rng.random(n) < 0.5
    )
    t = np.where(tl, lists.NEG_ZERO_NEIGHBOUR, 0).astype(np.uint32).view(np.float32)
    return coef[:, 0], coef[:, 1], coef[:, 2], px, py, t, recs


def edge_inputs(rng, n):
    """n edge cases: signed zeros, tiny, denormal and large values, cancellation."""
    special = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            1.2e-38,
            -1.2e-38,
            1e-40,
            -1e-40,
            3e-39,
            1e30,
            -1e30,
            3.4e38,
            1e-20,
            2.0**-63,
        ],
        dtype=np.float32,
    )
    pick = rng.integers(0, len(special), (n, 3))
    ab = rng.uniform(-1e3, 1e3, (n, 3)).astype(np.float32)
    use = rng.random((n, 3)) < 0.5
    a, b, c = (np.where(use[:, k], special[pick[:, k]], ab[:, k]) for k in range(3))
    px = (rng.integers(0, 1024, n) + F(0.5)).astype(np.float32)
    py = (rng.integers(0, 1024, n) + F(0.5)).astype(np.float32)
    # A third: c cancels a finite sum exactly, so e is +-0.
    with np.errstate(all="ignore"):
        s = (a * px).astype(np.float32) + (b * py).astype(np.float32)
    cancel = (rng.random(n) < 1 / 3) & np.isfinite(s)
    c = np.where(cancel, -s, c).astype(np.float32)
    t = np.where(rng.random(n) < 0.5, lists.NEG_ZERO_NEIGHBOUR, 0).astype(np.uint32)
    return a, b, c, px, py, t.view(np.float32)


def depths(rng, e, n):
    """d: e itself, its neighbours, zeros, or random depths."""
    kind = rng.integers(0, 5, n)
    up = np.nextafter(e, np.float32(np.inf))
    down = np.nextafter(e, np.float32(-np.inf))
    rnd = rng.uniform(0, 1, n).astype(np.float32)
    zero = np.where(rng.random(n) < 0.5, F(0), F(-0.0))
    return np.choose(kind, [e, up, down, zero, rnd]).astype(np.float32)


def chunk_pages(inputs):
    """Pages of the inputs: per chunk of 512, one page per input array."""
    n = len(inputs[0])
    chunks = n // LANES
    arr = np.stack([np.asarray(x, dtype=np.float32).view(np.uint32) for x in inputs])
    arr = arr.reshape(len(inputs), chunks, LANES).transpose(1, 0, 2)
    return arr.reshape(chunks * len(inputs), LANES)


def plan(chunks, cores):
    """(list pages per core order, directory): chunks dealt round-robin to cores."""
    owner = [list(range(c, chunks, cores)) for c in range(cores)]
    return owner


def build(n, scene, edge_fraction, seed):
    """(inputs, source): n inputs (a multiple of a chunk), a fraction of them edge
    cases (source True), in a random order."""
    rng = np.random.default_rng(seed)
    n = (n + LANES - 1) // LANES * LANES
    n_edge = int(n * edge_fraction)
    a, b, c, px, py, t, _ = scene_inputs(rng, n - n_edge, scene)
    ea, eb, ec, epx, epy, et = edge_inputs(rng, n_edge)
    a, b, c = np.concatenate([a, ea]), np.concatenate([b, eb]), np.concatenate([c, ec])
    px, py, t = (
        np.concatenate([px, epx]),
        np.concatenate([py, epy]),
        np.concatenate([t, et]),
    )
    e_ref = expected(a, b, c, px, py, t, t)[3]
    d = depths(rng, e_ref, n)
    perm = rng.permutation(n)
    inputs = [x[perm] for x in (a, b, c, px, py, t, d)]
    source = np.concatenate([np.zeros(n - n_edge, bool), np.ones(n_edge, bool)])[perm]
    return inputs, source


def lay_out(inputs, workers):
    """(list pages, directory) of the chunks dealt round-robin to the workers."""
    chunks = len(inputs[0]) // LANES
    per_chunk = chunk_pages(inputs)  # (chunks * 7, 512)
    owner = plan(chunks, workers)
    pages, directory, first = [], np.zeros((workers, 1024), np.uint32), 0
    end = np.zeros((1, LANES), np.uint32)
    end[0, 0] = K["kMarkEnd"]
    for core, ids in enumerate(owner):
        mine = [per_chunk[j * 7 : j * 7 + 7] for j in ids] + [end]
        block = np.concatenate(mine)
        directory[core, K["kDirFirst"]] = first
        directory[core, K["kDirPages"]] = len(block)
        directory[core, K["kDirItems"]] = len(ids)
        directory[core, K["kDirIds"] : K["kDirIds"] + len(ids)] = ids
        pages.append(block)
        first += len(block)
    return np.concatenate(pages), directory


def compare(out, inputs, source):
    """Per output and per rounding (IEEE, denormals flushed): how many differ."""
    n = len(inputs[0])
    chunks = n // LANES
    got = out.reshape(chunks, 6, LANES).transpose(1, 0, 2).reshape(6, n)
    report = {}
    for flush in (False, True):
        want = expected(*inputs, flush=flush)
        key = "ftz" if flush else "ieee"
        report[key] = {}
        for j, name in enumerate(NAMES):
            w = np.asarray(want[j]).view(np.uint32) if j < 4 else want[j]
            bad = got[j] != w
            entry = {
                "differ": int(bad.sum()),
                "differ_scene": int((bad & ~source).sum()),
                "differ_edge": int((bad & source).sum()),
            }
            if j < 4:
                gv, wv = got[j].view(np.float32), np.asarray(want[j], np.float32)
                both_zero = (gv == 0) & (wv == 0)
                entry["differ_only_zero_sign"] = int((bad & both_zero).sum())
            idx = np.nonzero(bad)[0][:5]
            entry["examples"] = [
                {
                    "in": [f"{inputs[k][i]:.9g}" for k in range(7)],
                    "got": int(got[j][i]),
                    "want": int(w[i]),
                }
                for i in idx
            ]
            report[key][name] = entry
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--scene", default="s10k")
    p.add_argument("--edge-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    inputs, source = build(args.n, args.scene, args.edge_fraction, args.seed)
    n = len(inputs[0])
    chunks = n // LANES
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        pages, directory = lay_out(inputs, workers)
        out, _, status, _, times = card.run_lists(
            session, K["kHandlerArith"], pages, directory, chunks * 3, K["kArithOut"]
        )
    report = {"n": n, "edge_cases": int(source.sum()), "scene": args.scene}
    report["run"] = status
    report["times"] = times
    report.update(compare(out, inputs, source))
    # Exact: no difference from the flushing model anywhere, none from IEEE on the
    # scene's own inputs (only the edge cases hold denormals).
    report["ok"] = bool(
        status["ok"]
        and all(e["differ"] == 0 for e in report["ftz"].values())
        and all(e["differ_scene"] == 0 for e in report["ieee"].values())
    )
    print(json.dumps(report), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"wall {time.time() - t0:.1f}s", flush=True)
