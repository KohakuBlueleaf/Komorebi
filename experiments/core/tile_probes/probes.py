"""Tile probes: single-core Metalium kernels that time the vector unit and the compute
RISC-V cores of one Blackhole tile (see README.md); prints one JSON line per run.

Usage: python probes.py <probe>[,<probe>...] [options]   (probe names: PROBES below)
"""

import argparse
import json
import os
import secrets
import statistics
import sys
import time

os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")

import numpy as np

import komorebi as km

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "kernels")
PROBES = {
    "smoke": 0,
    "pred": 1,
    "anycc": 2,
    "gather": 3,
    "slab": 4,
    "riscv": 5,
    "walk": 6,
    "nocread": 7,
    "push": 8,
    "macro": 9,
    "sfpcfg": 10,
    "fpu": 11,
    "icache": 12,
    "sfpi": 13,
    "ties": 14,
    "packer": 15,
}
# probe.hpp layout
NONCE, ERROR, NUM_TIMES, DONE0, TIMES, DATA = 0, 1, 2, 4, 16, 2048
IN_WORDS = OUT_WORDS = 8192
CB_IN, CB_OUT, CB_SCRATCH, SCRATCH_WORDS = 0, 16, 24, 4096


def log(msg):
    print(msg, flush=True)


def aiclk_mhz():
    """AICLK of the card in MHz from the driver's sysfs (TT_VISIBLE_DEVICES card)."""
    card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        with open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def to_dev(session, words):
    """A uint32 word array as a one-page row-major DRAM tensor."""
    return session.tensor(np.asarray(words, dtype=np.uint32).reshape(1, -1), "uint32")


def run_kernel(
    session,
    probe,
    in_words,
    params,
    defines=(),
    rvv=False,
    peer=None,
    extra_cbs=None,
    in_format="float32",
):
    """Runs one probe on core (0, 0); returns (out words as uint32, nonce). With peer
    (x, y), the reader and the CBs also run on that core; extra_cbs maps more CB
    indices to (format, page bytes), one page each."""
    in_buf = np.zeros(IN_WORDS, dtype=np.uint32)
    in_buf[: len(in_words)] = in_words
    nonce = secrets.randbits(32) | 1
    t_in = to_dev(session, in_buf)
    t_out = to_dev(session, np.zeros(OUT_WORDS, dtype=np.uint32))
    core = [(0, 0, 0, 0)]
    both = core if peer is None else core + [(*peer, *peer)]
    launch = session.launch()
    sizes = {CB_IN: IN_WORDS, CB_OUT: OUT_WORDS, CB_SCRATCH: SCRATCH_WORDS}
    for index, words in sizes.items():
        fmt = in_format if index == CB_IN else "float32"
        launch.cb(both, index, fmt, words * 4, words * 4)
    for index, (fmt, page) in (extra_cbs or {}).items():
        launch.cb(core, index, fmt, page, page)

    def args(words, ranges=core):
        return [(x, y, list(words)) for x, y, _, _ in ranges]

    in_ct, out_ct = t_in.accessor_args(), t_out.accessor_args()
    launch.kernel(
        os.path.join(KDIR, "reader.cpp"), both, 1, in_ct, [], [],
        args([t_in.address], both),
    )  # fmt: skip
    launch.kernel(
        os.path.join(KDIR, "writer.cpp"), core, 0, out_ct, [], [], args([t_out.address])
    )
    p = list(params) + [0] * (7 - len(params))
    launch.kernel(
        os.path.join(KDIR, "probe.cpp"), core, 2, [],
        [("PROBE", str(PROBES[probe]))] + list(defines), [], args([nonce] + p),
        math_fidelity="HiFi4",
        flags=[("fp32_dest_acc_en", True), ("enable_trisc2_rvv", bool(rvv))],
    )  # fmt: skip
    t0 = time.perf_counter()
    launch.run()
    session.device.finish()
    log(f"kernel done {time.perf_counter() - t0:.2f}s")
    out = t_out.read().reshape(-1).copy()
    return out, nonce


def times_of(out):
    n = int(out[NUM_TIMES])
    return [int(v) for v in out[TIMES : TIMES + n]]


def stats(v):
    """min / median / max of a list."""
    return {"min": min(v), "med": statistics.median(v), "max": max(v)}


def lane_words(out, base, n=32):
    return out[DATA + base : DATA + base + n]


# ---- probe 0 ------------------------------------------------------------------------


def smoke(session, args):
    rng = np.random.default_rng(args.seed)
    words = rng.integers(0, 2**32, 64, dtype=np.uint64).astype(np.uint32)
    # Include float edge patterns: denormals, -0, inf, NaN, max sign-magnitude.
    words[:8] = [0x00000001, 0x007FFFFF, 0x80000000, 0x7F800000, 0x7FC00001,
                 0xFF800000, 0x80000001, 0xFFFFFFFF]  # fmt: skip
    out, nonce = run_kernel(session, "smoke", words, [])
    got = lane_words(out, 0, 64)
    lane_ids = lane_words(out, 64, 32)
    t = times_of(out)
    bad = [i for i in range(64) if got[i] != words[i]]
    return (
        {
            "roundtrip_bad": len(bad),
            "roundtrip_bad_first": [(i, hex(words[i]), hex(got[i])) for i in bad[:8]],
            "lane_ids_ok": bool(all(lane_ids[i] == 2 * i for i in range(32))),
            "lane_ids_first": [int(v) for v in lane_ids[:8]],
            "wall_cycles_spin": t[0],
            "mcycle_spin": t[1],
            "tensix_sync_idle": stats(t[2:10]),
            "wall_read_pair": stats(t[10:18]),
        },
        out,
        nonce,
    )


# ---- probe 1 ------------------------------------------------------------------------


def pred(session, args):
    reps, k = args.reps, args.k
    out, nonce = run_kernel(session, "pred", [], [reps, k])
    t = times_of(out)
    names = ["empty", "all", "one", "none", "nest8", "chain", "nestcost"]
    per = {n: t[i * k : (i + 1) * k] for i, n in enumerate(names)}
    base = statistics.median(per["empty"])
    lane = np.arange(32, dtype=np.float32) * 2
    inc = np.float32(8 * reps)
    expect = {  # per data variant: L0..L3 per lane
        "all": [lane + inc] * 4,
        "one": [np.where(np.arange(32) == 13, lane + inc, lane)] * 4,
        "none": [lane] * 4,
        "nest8": [np.where(np.arange(32) >= 8, lane + inc, lane)] * 4,
        "chain": [lane + np.float32(32 * reps), lane, lane, lane],
        "nestcost": [lane] * 4,
    }
    bad = {}
    for v, name in enumerate(["all", "one", "none", "nest8", "chain", "nestcost"]):
        got = out[DATA + 128 * v : DATA + 128 * v + 128].view(np.float32).reshape(4, 32)
        want = np.stack(expect[name])
        nb = int((got != want).sum())
        if nb:
            bad[name] = {
                "n": nb,
                "got0": got[0, :4].tolist(),
                "want0": want[0, :4].tolist(),
            }
    n_instr = 32 * reps
    rec = {"reps": reps, "k": k, "empty_cycles": stats(per["empty"]), "values_bad": bad}
    for name in names[1:]:
        cyc = [c - base for c in per[name]]
        per_instr = [c / (24 * reps if name == "nestcost" else n_instr) for c in cyc]
        rec[name] = {"cycles": stats(per[name]), "per_instr": stats(per_instr)}
    rec["nestcost"]["per_sequence"] = stats(
        [(c - base) / reps for c in per["nestcost"]]
    )
    return rec, out, nonce


# ---- probe 2 ------------------------------------------------------------------------


DELAYS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]  # vec.hpp kDelays
CHECKS = ["tensix_sync", "busy_bit", "fence", "or_window", "fence_busy_bit"]


def anycc(session, args):
    k = min(args.k, 6)
    out, nonce = run_kernel(session, "anycc", [], [k])
    t = times_of(out)
    d = out[DATA:]
    truth = [1 if body <= 31 else 0 for body in range(1, 41)]
    rec = {"k": k}
    stale = {}
    for name, bit in (
        ("no_wait", 0),
        ("fence", 1),
        ("busy_bit", 2),
        ("tensix_sync", 3),
    ):
        wrong = []
        for pas in range(3):
            for body in range(40):
                v = (int(d[40 * pas + body]) >> bit) & 1
                if v != truth[body]:
                    wrong.append((pas, body + 1, v))
        stale[name] = {"wrong": len(wrong), "first": wrong[:6]}
    rec["recording"] = stale
    rec["bit24_equals_bit11"] = all(
        ((int(d[i]) >> 4) & 1) == ((int(d[i]) >> 3) & 1) for i in range(120)
    )
    rec["busy_polls_after_fence"] = stats([int(d[i]) >> 8 for i in range(120)])
    rec["delay_sweep_wrong"] = {
        str(delay): [
            body + 1 for body in range(40) if int(d[512 + 40 * i + body]) != truth[body]
        ]
        for i, delay in enumerate(DELAYS)
    }
    for m, name in enumerate(CHECKS):
        trips = [int(v) for v in d[128 + m * k : 128 + (m + 1) * k]]
        rec[name] = {"trips": trips, "cycles": stats(t[m * k : (m + 1) * k])}
    nc = len(CHECKS)
    fixed = t[nc * k : (nc + 1) * k]
    rec["fixed32"] = {"cycles": stats(fixed)}
    fixed_med = statistics.median(fixed)
    alone = t[(nc + 1) * k : (2 * nc + 1) * k]
    for m, name in enumerate(CHECKS):
        loop = t[m * k : (m + 1) * k]
        rec[name]["per_trip_minus_fixed"] = stats([(c - fixed_med) / 32 for c in loop])
        rec[name]["check_alone"] = stats(alone[m * k : (m + 1) * k])
    rec["check_alone_any"] = [int(v) for v in d[160 : 160 + nc]]
    lane_iters = np.arange(32) + 1
    bad = {}
    for v, name in enumerate([*CHECKS, "fixed32"]):
        got = d[256 + 32 * v : 256 + 32 * v + 32].astype(np.int64)
        nb = int((got != lane_iters).sum())
        if nb:
            bad[name] = {"n": nb, "got": got[:8].tolist()}
    rec["lane_iterations_bad"] = bad
    return rec, out, nonce


# ---- probe 3 ------------------------------------------------------------------------


def gather(session, args):
    rng = np.random.default_rng(args.seed)
    table = rng.integers(0, 2**32, (256, 4), dtype=np.uint64).astype(np.uint32)
    table[0, :4] = [0x00000001, 0x80000000, 0x7FC00001, 0xFFFFFFFF]  # float edge bits
    n_sets = 8
    sets = np.stack([rng.permutation(256)[:32] for _ in range(n_sets)]).astype(
        np.uint32
    )
    words = np.concatenate([table.reshape(-1), sets.reshape(-1)])
    w, reps = args.words, args.reps
    out, nonce = run_kernel(
        session, "gather", words, [w, reps, n_sets, args.variant], rvv=args.variant == 2
    )
    t = times_of(out)
    idx_out, copy, load = t[0::3], t[1::3], t[2::3]
    total = [a + b + c for a, b, c in zip(idx_out, copy, load)]
    bad = 0
    first = []
    for r in range(max(0, reps - n_sets), reps):
        s = r % n_sets
        for k in range(w):
            got = out[DATA + 128 * s + 32 * k : DATA + 128 * s + 32 * k + 32]
            want = table[sets[s], k]
            nb = int((got != want).sum())
            bad += nb
            if nb and len(first) < 4:
                first.append({"set": s, "word": k, "got": got[:4].tolist(),
                              "want": want[:4].tolist()})  # fmt: skip
    return (
        {
            "words": w,
            "reps": reps,
            "variant": ["plain", "scheduled", "rvv"][args.variant],
            "indices_out": stats(idx_out),
            "copy": stats(copy),
            "sfpload_sync": stats(load),
            "total": stats(total),
            "copy_per_lane": statistics.median(copy) / 32,
            "values_bad": bad,
            "values_bad_first": first,
        },
        out,
        nonce,
    )


# ---- probe 4 ------------------------------------------------------------------------

EXACT_INV = np.array([0.25, 0.5, 1.0, 2.0, 4.0, -0.25, -0.5, -1.0, -2.0, -4.0])


def bf16_bits(x):
    """bf16 bit patterns (truncation) of float32 values that are bf16-exact."""
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint32)


def slab_data(rng, node_count, exact):
    """Rays (inv, ob = -o inv, tmax) and boxes; exact: all values multiples of 1/4."""
    # Rays start in [-6, 6]^3 and head towards the origin on every axis; boxes lie
    # around the origin, so about a fifth of the ray-box pairs hit.
    if exact:
        o = rng.integers(-24, 25, (32, 3)).astype(np.float32) / 4
        inv = rng.choice(EXACT_INV, (32, 3)).astype(np.float32)
        tmax = rng.integers(16, 129, 32).astype(np.float32) / 4
        lo = rng.integers(-16, 9, (node_count, 3)).astype(np.float32) / 4
        hi = lo + rng.integers(4, 25, (node_count, 3)).astype(np.float32) / 4
    else:
        o = rng.uniform(-6, 6, (32, 3)).astype(np.float32)
        d = (-o + rng.normal(scale=2.0, size=(32, 3))).astype(np.float32)
        inv = (1 / d).astype(np.float32)
        tmax = rng.uniform(4, 32, 32).astype(np.float32)
        lo = rng.uniform(-4, 2, (node_count, 3)).astype(np.float32)
        hi = (lo + rng.uniform(1, 6, (node_count, 3))).astype(np.float32)
    inv = np.where(o * inv > 0, -inv, inv).astype(np.float32)
    ob = (-o * inv).astype(np.float32)
    return inv, ob, tmax, lo, hi


def slab_ref(inv, ob, tmax, lo, hi, fp64):
    """Hit mask per node (bit l = ray l) of the slab test the kernel computes."""
    dt = np.float64 if fp64 else np.float32
    inv, ob, tmax = inv.astype(dt), ob.astype(dt), tmax.astype(dt)
    masks, gaps = [], []
    for n in range(len(lo)):
        t0 = lo[n].astype(dt)[None, :] * inv + ob
        t1 = hi[n].astype(dt)[None, :] * inv + ob
        tnear = np.maximum(np.minimum(t0, t1).max(1), 0)
        tfar = np.minimum(np.maximum(t0, t1).min(1), tmax)
        hit = tfar - tnear >= 0
        masks.append(int(sum(1 << i for i in range(32) if hit[i])))
        gaps.append(np.abs(tfar - tnear))
    return masks, gaps


def slab(session, args):
    rng = np.random.default_rng(args.seed)
    node_count, exact = args.nodes, not args.random_data
    inv, ob, tmax, lo, hi = slab_data(rng, node_count, exact)
    mode = {"f32": 0, "bf16": 1, "const": 2}[args.slab_mode]
    if mode == 2:  # every node is node 0, fixed at compile time
        lo[:] = lo[0]
        hi[:] = hi[0]
    rays = np.concatenate([inv.T, ob.T, tmax[None, :]]).astype(np.float32)
    nodes = np.concatenate([lo, hi], 1).astype(np.float32)
    if mode == 0:
        node_words = nodes.view(np.uint32).reshape(-1)
    else:
        if not exact:
            raise SystemExit("bf16 and const modes need exact (bf16) boxes")
        node_words = bf16_bits(nodes).reshape(-1)
    words = np.zeros(512 + 6 * node_count, dtype=np.uint32)
    words[: 7 * 32] = rays.view(np.uint32).reshape(-1)
    words[512:] = node_words
    defines = []
    if mode == 2:
        box = bf16_bits(nodes[0])
        defines = [("SLAB_BOX", ",".join(str(int(v)) for v in box))]
    readout = args.variant
    out, nonce = run_kernel(
        session,
        "slab",
        words,
        [node_count, args.reps, mode, readout],
        defines,
        readout == 2,
    )
    t = times_of(out)
    sweep, read_t = t[0::2], t[1::2]
    got = [int(v) for v in out[DATA : DATA + node_count]]
    want, gaps = slab_ref(inv, ob, tmax, lo, hi, fp64=not exact)
    bad_bits = 0
    near = 0
    for n in range(node_count):
        diff = got[n] ^ want[n]
        for i in range(32):
            if (diff >> i) & 1:
                bad_bits += 1
                near += int(gaps[n][i] < 1e-5 * max(1.0, float(tmax[i])))
    raw = (
        out[DATA + 64 : DATA + 64 + 32 * node_count]
        .view(np.float32)
        .reshape(node_count, 32)
    )
    return (
        {
            "nodes": node_count,
            "reps": args.reps,
            "mode": args.slab_mode,
            "readout": ["plain", "scheduled", "rvv"][readout],
            "exact_data": exact,
            "sweep_cycles": stats(sweep),
            "per_node": stats([c / node_count for c in sweep]),
            "readout_cycles": stats(read_t),
            "readout_per_node": stats([c / node_count for c in read_t]),
            "hit_fraction": float(np.mean([m.bit_count() / 32 for m in want])),
            "flags_bad": bad_bits,
            "flags_bad_near_boundary": near,
            "first_gap_raw": raw[0, :4].tolist(),
        },
        out,
        nonce,
    )


# ---- probe 5 ------------------------------------------------------------------------

LOOP_NAMES = ["add", "mul", "load", "store", "copy", "float"]


def riscv_ref(t, loops, n, n_float, a, b):
    """Checksums of rv::run on thread t (uint32 arithmetic)."""
    m32 = 0xFFFFFFFF
    mine = [(j * 3 + t) & m32 for j in range(1024)]
    res = {}
    if loops & 1:
        x, y = 1, 2
        for i in range(n):
            x = (x + (i ^ y)) & m32
            y = (y + x) & m32
        res[0] = x ^ y
    if loops & 2:
        x = 1
        for i in range(n):
            x = (x * 0x9E3779B1 + i) & m32
        res[1] = x
    if loops & 4:
        res[2] = sum(mine[i & 1023] for i in range(n)) & m32
    if loops & 8:
        for i in range(n):
            mine[i & 1023] = i & m32
        res[3] = mine[(n - 1) & 1023]
    if loops & 16:
        for i in range(n):
            j = i & 511
            mine[512 + j] = (mine[j] + 1) & m32
        res[4] = sum(mine[512:1024]) & m32
    if loops & 32:
        x = np.float32(1.0)
        fa, fb = np.uint32(a).view(np.float32), np.uint32(b).view(np.float32)
        for _ in range(n_float):
            x = np.float32(x * fa)
            x = np.float32(x + fb)
        res[5] = int(np.float32(x).view(np.uint32))
    return res


def riscv(session, args):
    mask, loops, n, n_float = args.mask, args.loops, args.n, args.n_float
    a = int(np.float32(0.999).view(np.uint32))
    b = int(np.float32(0.001).view(np.uint32))
    out, nonce = run_kernel(session, "riscv", [], [mask, loops, n, n_float, a, b])
    rec = {"mask": mask, "loops": loops, "n": n, "n_float": n_float, "threads": {}}
    bad = []
    for t in range(3):
        if not (mask >> t) & 1:
            continue
        r = out[DATA + 256 * t : DATA + 256 * t + 256]
        ref = riscv_ref(t, loops, n, n_float, a, b)
        th = {"start": int(r[0]), "end": int(r[1])}
        for i, name in enumerate(LOOP_NAMES):
            if not (loops >> i) & 1:
                continue
            iters = n_float if i == 5 else n
            th[name] = round(int(r[8 + i]) / iters, 3)
            if int(r[32 + i]) != ref[i]:
                bad.append((t, name, int(r[32 + i]), ref[i]))
        rec["threads"][t] = th
        if t < 2 and int(out[DONE0 + t]) != nonce:
            bad.append((t, "done word", int(out[DONE0 + t]), nonce))
    starts = [th["start"] for th in rec["threads"].values()]
    ends = [th["end"] for th in rec["threads"].values()]
    rec["overlap_cycles"] = (min(ends) - max(starts)) % 2**32 if starts else None
    rec["checks_bad"] = bad
    return rec, out, nonce


# ---- probe 6 ------------------------------------------------------------------------

LEAF = 0x80000000
WALK_STATS = ["total", "setup", "stack", "copy", "vector", "readout", "check", "steps",
              "lane_tests", "check_mismatch"]  # fmt: skip
BEST0 = np.float32(1e30)


def build_tree(rng):
    """128 leaf boxes on a 1/4 grid and a median-split box tree over them, nodes in
    breadth-first order (children after parents): (records [255, 8] uint32, boxes)."""
    centre = rng.integers(-28, 29, (128, 3)) / 4
    half = rng.integers(2, 7, (128, 3)) / 4
    lo, hi = (centre - half).astype(np.float32), (centre + half).astype(np.float32)
    rec = np.zeros((255, 8), dtype=np.uint32)
    queue, next_id = [(0, np.arange(128))], 1
    while queue:
        node, leaves = queue.pop(0)
        box = np.concatenate([lo[leaves].min(0), hi[leaves].max(0)]).astype(np.float32)
        rec[node, :6] = box.view(np.uint32)
        if len(leaves) == 1:
            rec[node, 6:] = LEAF | int(leaves[0])
            continue
        c = (lo[leaves] + hi[leaves])[:, :]
        axis = int(np.argmax(c.max(0) - c.min(0)))
        order = leaves[np.argsort(c[:, axis], kind="stable")]
        half_n = len(order) // 2
        rec[node, 6:] = [next_id, next_id + 1]
        queue += [(next_id, order[:half_n]), (next_id + 1, order[half_n:])]
        next_id += 2
    assert next_id == 255
    return rec, lo, hi


def walk_rays(rng, n, kind):
    """(inv, ob) float32 [n, 3]. random: origins on a 1/4 grid, inverse directions
    powers of two heading towards the scene (every product exact); camera: a pinhole
    at z = -24 looking at the boxes (general float values)."""
    if kind == "random":
        o = rng.integers(-48, 49, (n, 3)).astype(np.float32) / 4
        inv = rng.choice(EXACT_INV, (n, 3)).astype(np.float32)
        inv = np.where(o * inv > 0, -inv, inv).astype(np.float32)
    else:
        w = 32 if n >= 1024 else (16 if n >= 256 else 8)
        h = n // w
        u = (np.arange(w) + 0.5) / w * 2 - 1
        v = (np.arange(h) + 0.5) / h * 2 - 1
        uu, vv = np.meshgrid(u, v)
        d = np.stack([uu.ravel() * 0.4, vv.ravel() * 0.4, np.ones(n)], 1)
        o = np.tile([0.3, 0.2, -24.0], (n, 1))
        inv = (1 / d.astype(np.float32)).astype(np.float32)
    ob = (-o.astype(np.float32) * inv).astype(np.float32)
    return inv, ob


def walk_ref(inv, ob, lo, hi):
    """Per ray: closest entry distance over the leaf boxes hit (each plane distance
    rounded once to float32, as the vector unit's multiply-add), and the leaves at it.
    """
    inv64, ob64 = inv.astype(np.float64), ob.astype(np.float64)
    t0 = (lo[None].astype(np.float64) * inv64[:, None] + ob64[:, None]).astype(
        np.float32
    )
    t1 = (hi[None].astype(np.float64) * inv64[:, None] + ob64[:, None]).astype(
        np.float32
    )
    near = np.maximum(np.minimum(t0, t1).max(2), np.float32(0))
    far = np.minimum(np.maximum(t0, t1).min(2), BEST0)
    hit = far >= near
    dist = np.where(hit, near, np.float32(np.inf))
    best = dist.min(1)
    ties = [np.flatnonzero(dist[r] == best[r]) for r in range(len(best))]
    return best, ties


def walk(session, args):
    rng = np.random.default_rng(args.seed)
    n = args.rays
    rec, lo, hi = build_tree(rng)
    inv, ob = walk_rays(rng, n, args.ray_set)
    words = np.zeros(8192, dtype=np.uint32)
    words[: 255 * 8] = rec.reshape(-1)
    fields = np.concatenate([inv, ob], 1).reshape(n // 32, 32, 6).transpose(0, 2, 1)
    words[2048 : 2048 + 6 * n] = fields.astype(np.float32).view(np.uint32).reshape(-1)
    form = {"a": 0, "b": 1}[args.form]
    copy = {"rvv": 0, "scalar": 1}[args.copy]
    best0 = int(BEST0.view(np.uint32))
    out, nonce = run_kernel(session, "walk", words, [n, form, copy, best0], rvv=True)
    st = dict(zip(WALK_STATS, (int(v) for v in out[TIMES : TIMES + 10])))
    got_best = out[DATA : DATA + n].view(np.float32)
    got_leaf = out[DATA + 1024 : DATA + 1024 + n]
    ref_best, ties = walk_ref(inv, ob, lo, hi)
    ref_hit = np.isfinite(ref_best)
    dev_hit = got_leaf != 0xFFFFFFFF
    both = ref_hit & dev_hit
    diff = np.abs(got_best[both].astype(np.float64) - ref_best[both])
    leaf_bad, leaf_tie = 0, 0
    for r in np.flatnonzero(both):
        leaf = int(got_leaf[r]) & 0x7FFFFFFF
        if leaf not in ties[r]:
            leaf_bad += 1
        elif len(ties[r]) > 1:
            leaf_tie += 1
    total = st["total"]
    return (
        {
            "form": args.form,
            "copy": args.copy if args.form == "a" else None,
            "rays": n,
            "ray_set": args.ray_set,
            "stats": st,
            "cycles_per_ray": round(total / n, 2),
            "cycles_per_lane_test": round(total / max(1, st["lane_tests"]), 2),
            "cycles_per_step": round(total / max(1, st["steps"]), 2),
            "ref_hits": int(ref_hit.sum()),
            "hit_disagree": int((ref_hit != dev_hit).sum()),
            "best_exact": int((diff == 0).sum()),
            "best_max_abs_error": float(diff.max()) if diff.size else 0.0,
            "leaf_wrong": leaf_bad,
            "leaf_tie_ok": leaf_tie,
            "ref_rays_with_ties": int(sum(len(t) > 1 for t in ties)),
        },
        out,
        nonce,
    )


# ---- probe 7 ------------------------------------------------------------------------

NOC_WHERE = ["", "started", "waited for the peer", "command buffer ready", "submitting",
             "responded or gave up", "registers restored"]  # fmt: skip


def nocread(session, args):
    rng = np.random.default_rng(args.seed)
    words = rng.integers(0, 2**32, 64, dtype=np.uint64).astype(np.uint32)
    peer = (1, 0)
    px, py = session.device.worker_core(*peer)
    coord = int(px) | int(py) << 6
    log(f"peer {peer} at NoC-0 ({px}, {py})")
    out, nonce = run_kernel(session, "nocread", words, [coord, 200000], peer=peer)
    d = out[DATA:]
    got = d[16:32]
    return (
        {
            "peer_logical": list(peer),
            "peer_noc0": [int(px), int(py)],
            "my_noc_id": hex(int(d[4])),
            "where": NOC_WHERE[int(d[0])] if int(d[0]) < len(NOC_WHERE) else int(d[0]),
            "code": ["ok", "command buffer busy", "no response"][int(d[1])],
            "polls": int(d[2]),
            "cycles": int(d[3]),
            "counter_before": int(d[5]),
            "counter_after": int(d[6]),
            "data_ok": bool((got == words[:16]).all()),
            "data_first": [hex(int(v)) for v in got[:4]],
        },
        out,
        nonce,
    )


# ---- probe 8 ------------------------------------------------------------------------

PUSH_FORMS = ["empty", "tti_x16", "tti_x64", "stored_x16", "stored_x64", "built",
              "table"]  # fmt: skip
LOOP_FORMS = ["pushed", "replay16", "replay4", "mop_replay16", "mop_replay4", "mop_two"]


def f32_bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def push(session, args):
    r, k = args.push_reps, args.k
    if r % 32 or not 32 <= r <= 512:
        raise SystemExit("push: --push-reps must be a multiple of 32 in 32..512")
    out, nonce = run_kernel(session, "push", [], [r, k])
    t = times_of(out)
    n_push = len(PUSH_FORMS) * k
    per = {n: t[i * k : (i + 1) * k] for i, n in enumerate(PUSH_FORMS)}
    for i, n in enumerate(LOOP_FORMS):
        per[n] = t[n_push + i * k : n_push + (i + 1) * k]
    base = statistics.median(per["empty"]) if per["empty"] else 0
    rec = {
        "reps": r,
        "k": k,
        "empty_cycles": stats(per["empty"]) if per["empty"] else None,
    }
    bad = {}
    for f, name in enumerate(PUSH_FORMS):
        want = int(out[DATA + 2048 + f])
        got = lane_words(out, 128 * f)
        nb = int((got != want).sum())
        if nb:
            bad[name] = {"n": nb, "want": hex(want), "got0": hex(int(got[0]))}
        if f and per[name]:
            rec[name] = {
                "cycles": stats(per[name]),
                "per_push": stats([(c - base) / (16 * r) for c in per[name]]),
            }
    lane = np.arange(32, dtype=np.float32) * 2
    for f, name in enumerate(LOOP_FORMS):
        inc = [4 * r] * 4 if name != "mop_two" else [8 * r, 8 * r, 0, 0]
        want = np.stack([f32_bits(lane + np.float32(i)) for i in inc])
        got = out[DATA + 128 * (8 + f) : DATA + 128 * (9 + f)].reshape(4, 32)
        nb = int((got != want).sum())
        if nb:
            bad[name] = {
                "n": nb,
                "want0": hex(int(want[0, 0])),
                "got0": hex(int(got[0, 0])),
            }
        if per[name]:
            rec[name] = {
                "cycles": stats(per[name]),
                "per_instr": stats([(c - base) / (16 * r) for c in per[name]]),
            }
    rec["values_bad"] = bad
    return rec, out, nonce


# ---- probe 9 ------------------------------------------------------------------------

MACRO_FORMS = ["one_by_one", "interleaved4", "load_macro"]


def macro(session, args):
    p, k = args.macro_passes, args.k
    rng = np.random.default_rng(args.seed)
    x = rng.integers(-1000, 1000, 64 * 32).astype(np.float32)
    out, nonce = run_kernel(session, "macro", f32_bits(x), [p, k])
    t = times_of(out)
    d = out[DATA:]
    rec = {"passes": p, "k": k, "rows_per_pass": 64}
    for f, name in enumerate(MACRO_FORMS):
        span = t[f * k : (f + 1) * k]
        rec[name] = {
            "cycles": stats(span) if span else None,
            "per_row_gross": stats([c / (64 * p) for c in span]) if span else None,
            "kernel_bad_lanes": int(d[3 * f]),
            "kernel_first_bad": int(d[3 * f + 1]) if int(d[3 * f]) else None,
            "n": int(d[3 * f + 2]),
        }
    n = 3 * k * p
    want = f32_bits(x + np.float32(n))
    got = d[64 : 64 + 64 * 32]
    rec["host_bad_lanes"] = int((got != want).sum())
    return rec, out, nonce


# ---- probe 10 -----------------------------------------------------------------------


def sfpcfg(session, args):
    out, nonce = run_kernel(session, "sfpcfg", [], [])
    slot = [lane_words(out, 32 * s) for s in range(10)]
    first = [int(v[0]) for v in slot]
    uniform = [bool((v == v[0]).all()) for v in slot]
    misc = [v & 0xFFF for v in first[:6] + first[8:9]]
    readings = {  # value each source predicts for slots 3, 4 (Misc), 6, 7 (LaneConfig)
        "isa_or2_and4": {"m3": 0x0FC, "m4": 0x0C0, "l6": 0x0101, "l7": 0x0001},
        "sfpi_and2_or4": {"m3": 0x0C0, "m4": 0x0FC, "l6": 0x0000, "l7": 0x0101},
    }
    seen = {"m3": first[3] & 0xFFF, "m4": first[4] & 0xFFF,
            "l6": first[6] & 0xFFFF, "l7": first[7] & 0xFFFF}  # fmt: skip
    match = [k for k, v in readings.items() if v == seen]
    return (
        {
            "slots_hex": [hex(v) for v in first],
            "slots_uniform": uniform,
            "misc_12bit": [hex(v) for v in misc],
            "after_replace_0x0f0": hex(first[2] & 0xFFF),
            "xor_mode6": hex(first[5] & 0xFFF),
            "seen": {k: hex(v) for k, v in seen.items()},
            "matches": match,
            "restored": first[8] == first[0] and first[9] == first[1],
        },
        out,
        nonce,
    )


# ---- probe 11 -----------------------------------------------------------------------

# (op, K): K Dst blocks in rotation; ("mvmul", 0) is one MVMUL pushed once.
FPU_CONFIGS = (
    [("mvmul", 0)]
    + [("mvmul", k) for k in (1, 2, 3, 4, 5, 6, 8)]
    + [("elwmul", 1), ("elwmul", 8), ("elwadd", 1), ("elwadd", 8)]
    + [("mova2d", 1), ("mova2d", 8), ("movb2d", 8)]
)  # fmt: skip  (fpu.hpp run_config)
FPU_A = 1 + 2**-6  # every SrcA and SrcB value
FPU_REPS, FPU_PLAYS = 3, 64  # kReps spans of kReplays replays of the body (fpu.hpp)
# Rows a block covers, and what a word adds per instruction when SrcA = SrcB = FPU_A:
# MVMUL phase 0 16 products of 1.0 and FPU_A, ELWMUL one such product, ELWADD A + B.
FPU_ROWS = {"mvmul": 8, "elwmul": 8, "elwadd": 8, "mova2d": 8, "movb2d": 4}
FPU_ADDS = {"mvmul": 16 * FPU_A, "elwmul": FPU_A, "elwadd": 1 + FPU_A}


def fpu_rows(out, first, count):
    """One row per config run: spans, cycles per instruction, Dst check."""
    d = out[DATA:]
    empty = [int(v) for v in d[32 * len(FPU_CONFIGS) : 32 * len(FPU_CONFIGS) + 3]]
    base = statistics.median(empty)
    rows = []
    for c in range(first, first + count):
        op, k = FPU_CONFIGS[c]
        w = d[32 * c : 32 * c + 32]
        single = k == 0
        length = 1 if single else (20 if k == 5 else 24)
        blocks = 1 if single else k
        n = 1 if single else FPU_REPS * FPU_PLAYS * length // blocks  # a block gets
        spans = [int(w[0])] if single else [int(v) for v in w[0:3]]
        hist = [(float(np.uint32(w[5 + 2 * q]).view(np.float32)), int(w[6 + 2 * q]))
                for q in range(4) if int(w[6 + 2 * q])]  # fmt: skip
        want = n * FPU_ADDS[op] if op in FPU_ADDS else FPU_A
        want_words = FPU_ROWS[op] * 16 * blocks
        rows.append({
            "config": c, "op": op, "k": k, "spans": spans,
            "instr_a_span": 1 if single else FPU_PLAYS * length,
            "per_instr": None if single else
                round((statistics.median(spans) - base) / (FPU_PLAYS * length), 3),
            "len_ok": int(w[3]) == length,
            "nonzero": int(w[4]), "nonzero_expected": want_words,
            "values": hist, "value_expected": want,
            "dst_ok": int(w[4]) == want_words and hist == [(want, want_words)],
        })  # fmt: skip
    return {"empty_cycles": empty, "rows": rows}


def fpu(session, args):
    pair = int(f32_bits(FPU_A) >> 16) * 0x10001  # two bf16 values a word
    words = np.full(512, pair, dtype=np.uint32)  # one bf16 tile, as the dense kernel's
    first, count = args.first, args.count
    if not (first >= 0 and count >= 1 and first + count <= len(FPU_CONFIGS)):
        raise SystemExit(f"fpu: configs 0..{len(FPU_CONFIGS) - 1}")
    out, nonce = run_kernel(session, "fpu", words, [first, count], in_format="bfloat16")
    return fpu_rows(out, first, count), out, nonce


# ---- probe 12 -----------------------------------------------------------------------

ICACHE_SIZES = [64, 120, 124, 126, 128, 136, 192, 256, 480, 520]  # icache.hpp kSizes


def icache(session, args):
    mask, n = args.mask, args.icache_passes
    out, nonce = run_kernel(session, "icache", [], [mask, n])
    rec = {"mask": mask, "passes": n}
    for t in range(3):
        if not (mask >> t) & 1:
            continue
        r = out[DATA + 256 * t :]
        rows = []
        for i, s in enumerate(ICACHE_SIZES):
            cyc, x = int(r[2 * i]), int(r[2 * i + 1])
            rows.append(
                {
                    "adds": s,
                    "body_bytes": 4 * s + 8,
                    "cycles": cyc,
                    "per_pass": round(cyc / n, 2),
                    "per_instr": round(cyc / (n * (s + 2)), 3),
                    "count_ok": x == (s * n) & 0xFFFFFFFF,
                }
            )
        rec[f"trisc{t}"] = rows
    return rec, out, nonce


# ---- probe 13 -----------------------------------------------------------------------


def sfpi(session, args):
    rng = np.random.default_rng(args.seed)
    a, b, c = (rng.integers(-64, 64, 32).astype(np.float32) / 4 for _ in range(3))
    out, nonce = run_kernel(
        session, "sfpi", np.concatenate([f32_bits(v) for v in (a, b, c)]), []
    )
    got = [lane_words(out, 32 * i) for i in range(7)]
    want = {
        "mad": f32_bits(a * b + c),
        "sub": f32_bits(a - b),
        "const": f32_bits(a * np.float32(1.5) + np.float32(0.25)),
        "if": f32_bits(np.where(a < b, c, a)),
    }
    bad = {k: int((got[3 + i] != v).sum()) for i, (k, v) in enumerate(want.items())}
    inputs_ok = all((got[i] == f32_bits(v)).all() for i, v in enumerate((a, b, c)))
    return {"values_bad": bad, "inputs_kept": bool(inputs_ok)}, out, nonce


# ---- probes 14 and 15 ---------------------------------------------------------------

TIE_CBS = {17: ("bfloat16", 2048), 18: ("bfloat16", 2048)}  # probe.hpp kCbRne, kCbSr
TIE_A = 2.0**-12  # products 1 and A * A = 2^-24: sums at half an FP32 unit of 1


def tie_tile():
    """Face 0 of the input tile (row-major 16 x 16): a few rows of 1 and A whose
    product with the same tile gives 1 + 2^-24 and -(1 + 2^-24), the two halves in one
    multiplier lane (k = 4, 5) or in two (k = 6, 9 and 7, 10)."""
    t = np.zeros((16, 16), dtype=np.float64)
    a = TIE_A
    t[0, 4], t[4, 0], t[0, 5], t[5, 0] = 1, 1, a, a
    t[1, 4], t[4, 1], t[1, 5], t[5, 1] = 1, -1, a, -a
    t[2, 6], t[6, 2], t[2, 9], t[9, 2] = 1, 1, a, a
    t[3, 7], t[7, 3], t[3, 10], t[10, 3] = 1, -1, a, -a
    return t


def pack_values():
    """1024 FP32 values for the packer, element e: kind e >> 9 (0 a tie of BF16, 1
    exact), sign bit 7, BF16 mantissa e & 127. Every value has no bit below 2^-10, so
    SrcA's TF32 keeps it whole on its way to Dst."""
    e = np.arange(1024)
    m, sign, kind = e & 127, (e >> 7) & 1, e >> 9
    v = 1 + m / 128 + np.where(kind == 0, 2**-8, 0.0)
    return np.where(sign == 1, -v, v), kind


def round_f32(x, mode):
    """x (float64, exact) to FP32 under 'rne' or 'away' (ties away from zero)."""
    f = np.float32(x)
    if mode == "rne":
        return f
    out = []
    for xi in np.atleast_1d(x):
        lo = np.float32(xi)
        if float(lo) == xi:
            out.append(lo)
            continue
        hi = np.nextafter(lo, np.float32(np.inf if xi > float(lo) else -np.inf))
        a, b = sorted([lo, hi], key=float)
        if xi - float(a) == float(b) - xi:  # a tie: away from zero
            out.append(b if xi > 0 else a)
        else:
            out.append(lo)
    return np.array(out, dtype=np.float32)


def ties(session, args):
    t = tie_tile()
    tile = np.zeros(1024, dtype=np.float64)
    tile[:256] = t.reshape(-1)
    out, nonce = run_kernel(session, "ties", f32_bits(tile), [])
    d = out[DATA:]
    dst = d[0:1024].view(np.float32).reshape(64, 16)
    exact = t[:8, :] @ t  # Dst[i][j] = sum_k B[i][k] A[k][j], B = rows 0..7
    exact_t = t.T[:8, :] @ t.T  # the transposed reading, in case
    rec = {}
    for name, ex in [("b_at_a", exact), ("transposed", exact_t)]:
        r = {m: int((dst[:8] != round_f32(ex.reshape(-1), m).reshape(8, 16)).sum())
             for m in ("rne", "away")}  # fmt: skip
        rec[f"mvmul_mismatch_{name}"] = r
    ties = [(0, 0), (1, 1), (2, 2), (3, 3)]
    rec["mvmul_ties"] = {
        f"{i},{j}": [float(dst[i, j]), float(exact[i, j])] for i, j in ties
    }
    sr = {f"{i},{j}": [float(dst[8 * b + i, j]) for b in range(1, 8)] for i, j in ties}
    rec["mvmul_ties_srnd"] = sr
    rec["srnd_rows_nontie_equal"] = int(
        sum((dst[8 * b : 8 * b + 8] == dst[:8]).sum() for b in range(1, 8))
    )
    return rec, out, nonce


def packer(session, args):
    pv, kind = pack_values()
    out, nonce = run_kernel(session, "packer", f32_bits(pv), [], extra_cbs=TIE_CBS)
    d = out[DATA:]
    rec = {}

    def bf(words16):
        return (words16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    p_rne = bf(d[1024:1536].view(np.uint16))
    p_sr = bf(d[1536:2048].view(np.uint16))
    lo = np.sign(pv) * (1 + (np.arange(1024) & 127) / 128)
    hi = np.sign(pv) * (1 + ((np.arange(1024) & 127) + 1) / 128)
    pk = {}
    for name, got in [("rne", p_rne), ("srnd", p_sr)]:
        res = {}
        for kd, label in enumerate(["tie", "exact"]):
            sel = kind == kd
            g = got[sel]
            res[label] = {
                "to_larger_magnitude": int((g == hi[sel]).sum()),
                "to_smaller_magnitude": int((g == lo[sel]).sum()),
                "other": int(((g != hi[sel]) & (g != lo[sel])).sum()),
            }
            if kd == 0:
                even = (np.arange(1024)[sel] & 1) == 0  # BF16 mantissa m even: lo even
                res[label]["to_even"] = int(
                    ((g == lo[sel]) & even).sum() + ((g == hi[sel]) & ~even).sum()
                )
        pk[name] = res
    rec["packer_bf16"] = pk
    return rec, out, nonce


def main():
    p = argparse.ArgumentParser()
    p.add_argument("probe", help="a probe name or a comma list run in one session")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--words", type=int, default=4, choices=[1, 2, 4])
    p.add_argument(
        "--variant",
        type=int,
        default=0,
        help="gather copy / slab readout: 0 plain, 1 scheduled, 2 RVV",
    )
    p.add_argument("--nodes", type=int, default=64)
    p.add_argument("--slab-mode", choices=["f32", "bf16", "const"], default="f32")
    p.add_argument("--random-data", action="store_true")
    p.add_argument("--mask", type=int, default=4, help="probe riscv: TRISCs (bit t)")
    p.add_argument("--loops", type=int, default=63, help="probe riscv: loops (bit i)")
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--push-reps", type=int, default=256, help="push: loops of 16")
    p.add_argument(
        "--macro-passes", type=int, default=16, help="macro: passes of 64 rows"
    )
    p.add_argument(
        "--icache-passes", type=int, default=1000, help="icache: loop passes"
    )
    p.add_argument("--n-float", type=int, default=10000)
    p.add_argument("--rays", type=int, default=32, help="walk: multiple of 32, <= 1024")
    p.add_argument("--ray-set", choices=["camera", "random"], default="random")
    p.add_argument("--form", choices=["a", "b"], default="a")
    p.add_argument("--first", type=int, default=0, help="fpu: first config")
    p.add_argument("--count", type=int, default=1, help="fpu: number of configs")
    p.add_argument("--copy", choices=["rvv", "scalar"], default="rvv")
    args = p.parse_args()
    names = args.probe.split(",")
    for name in names:
        if name not in PROBES:
            p.error(f"unknown probe {name!r}; one of {', '.join(PROBES)}")
    if "slab" in names:
        args.nodes = min(args.nodes, 64)

    t_start = time.time()
    with open("/proc/loadavg") as f:
        load = f.read().split()[:3]
    status = 0
    with km.Session() as session:
        log(f"device open {time.time() - t_start:.1f}s")
        for name in names:
            mhz = [aiclk_mhz()]
            log(f"probe {name}")
            rec, out, nonce = globals()[name](session, args)
            mhz.append(aiclk_mhz())
            rec = {"probe": name, **rec}
            rec["nonce_ok"] = int(out[NONCE]) == nonce
            rec["kernel_error"] = int(out[ERROR])
            rec["aiclk_mhz"] = mhz
            rec["load"] = load
            rec["wall_s"] = round(time.time() - t_start, 1)
            print(json.dumps(rec), flush=True)
            if not (rec["nonce_ok"] and rec["kernel_error"] == 0):
                status = 1
                break
    return status


if __name__ == "__main__":
    sys.exit(main())
