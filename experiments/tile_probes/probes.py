"""Tile probes: small single-core Metalium kernels that measure the vector unit and the
compute RISC-V cores of one Blackhole Tensix tile (see README.md).

Every run launches one program on core (0, 0) through ttnn's generic operation: a
reader copies the input page into L1, the compute kernel runs one probe and fills the
output page, a writer copies it to DRAM. The host checks every value against a CPU
reference and prints one JSON line with the timings (wall-clock cycles).

Usage: python probes.py <probe> [options]   (probe: smoke pred anycc gather slab riscv)
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
import torch
import ttnn

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


def to_dev(device, words):
    """A uint32 word array as a one-page int32 row-major DRAM tensor."""
    t = torch.from_numpy(np.asarray(words, dtype=np.uint32).view(np.int32).copy())
    return ttnn.from_torch(
        t.reshape(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def run_kernel(device, probe, in_words, params, defines=(), rvv=False, peer=None):
    """Runs one probe on core (0, 0); returns (out words as uint32, nonce). With peer
    (x, y), the reader and the CBs also run on that core."""
    in_buf = np.zeros(IN_WORDS, dtype=np.uint32)
    in_buf[: len(in_words)] = in_words
    nonce = secrets.randbits(32) | 1
    t_in = to_dev(device, in_buf)
    t_out = to_dev(device, np.zeros(OUT_WORDS, dtype=np.uint32))
    core = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
    )
    both = core
    if peer is not None:
        both = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(*peer), ttnn.CoreCoord(*peer)),
            ]
        )

    def cb(index, words):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=index, data_format=ttnn.float32, page_size=words * 4
        )
        return ttnn.CBDescriptor(
            total_size=words * 4, core_ranges=both, format_descriptors=[fmt]
        )

    def kernel(name, config, args, compile_args=(), extra_defines=(), ranges=core):
        rt = ttnn.RuntimeArgs()
        rt[0][0] = list(args)
        if ranges is not core:
            rt[peer[0]][peer[1]] = list(args)
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ranges,
            compile_time_args=list(compile_args),
            defines=list(extra_defines),
            runtime_args=rt,
            config=config,
        )

    p = list(params) + [0] * (7 - len(params))
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    if rvv:
        compute_cfg.enable_trisc2_rvv = True
    desc = ttnn.ProgramDescriptor(
        kernels=[
            kernel(
                "reader.cpp",
                ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.RISCV_1_default,
                ),
                [t_in.buffer_address()],
                ttnn.TensorAccessorArgs(t_in).get_compile_time_args(),
                ranges=both,
            ),
            kernel(
                "writer.cpp",
                ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.RISCV_0_default,
                ),
                [t_out.buffer_address()],
                ttnn.TensorAccessorArgs(t_out).get_compile_time_args(),
            ),
            kernel(
                "probe.cpp",
                compute_cfg,
                [nonce] + p,
                extra_defines=[("PROBE", str(PROBES[probe]))] + list(defines),
            ),
        ],
        semaphores=[],
        cbs=[cb(CB_IN, IN_WORDS), cb(CB_OUT, OUT_WORDS), cb(CB_SCRATCH, SCRATCH_WORDS)],
    )
    t0 = time.perf_counter()
    ttnn.generic_op([t_in, t_out], desc)
    ttnn.synchronize_device(device)
    log(f"kernel done {time.perf_counter() - t0:.2f}s")
    out = ttnn.to_torch(t_out).numpy().reshape(-1).view(np.uint32).copy()
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


def smoke(device, args):
    rng = np.random.default_rng(args.seed)
    words = rng.integers(0, 2**32, 64, dtype=np.uint64).astype(np.uint32)
    # Include float edge patterns: denormals, -0, inf, NaN, max sign-magnitude.
    words[:8] = [0x00000001, 0x007FFFFF, 0x80000000, 0x7F800000, 0x7FC00001,
                 0xFF800000, 0x80000001, 0xFFFFFFFF]  # fmt: skip
    out, nonce = run_kernel(device, "smoke", words, [])
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


def pred(device, args):
    reps, k = args.reps, args.k
    out, nonce = run_kernel(device, "pred", [], [reps, k])
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


def anycc(device, args):
    k = min(args.k, 6)
    out, nonce = run_kernel(device, "anycc", [], [k])
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


def gather(device, args):
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
        device, "gather", words, [w, reps, n_sets, args.variant], rvv=args.variant == 2
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


def slab_data(rng, n_nodes, exact):
    """Rays (inv, ob = -o inv, tmax) and boxes; exact: all values multiples of 1/4."""
    # Rays start in [-6, 6]^3 and head towards the origin on every axis; boxes lie
    # around the origin, so about a fifth of the ray-box pairs hit.
    if exact:
        o = rng.integers(-24, 25, (32, 3)).astype(np.float32) / 4
        inv = rng.choice(EXACT_INV, (32, 3)).astype(np.float32)
        tmax = rng.integers(16, 129, 32).astype(np.float32) / 4
        lo = rng.integers(-16, 9, (n_nodes, 3)).astype(np.float32) / 4
        hi = lo + rng.integers(4, 25, (n_nodes, 3)).astype(np.float32) / 4
    else:
        o = rng.uniform(-6, 6, (32, 3)).astype(np.float32)
        d = (-o + rng.normal(scale=2.0, size=(32, 3))).astype(np.float32)
        inv = (1 / d).astype(np.float32)
        tmax = rng.uniform(4, 32, 32).astype(np.float32)
        lo = rng.uniform(-4, 2, (n_nodes, 3)).astype(np.float32)
        hi = (lo + rng.uniform(1, 6, (n_nodes, 3))).astype(np.float32)
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


def slab(device, args):
    rng = np.random.default_rng(args.seed)
    n_nodes, exact = args.nodes, not args.random_data
    inv, ob, tmax, lo, hi = slab_data(rng, n_nodes, exact)
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
    words = np.zeros(512 + 6 * n_nodes, dtype=np.uint32)
    words[: 7 * 32] = rays.view(np.uint32).reshape(-1)
    words[512:] = node_words
    defines = []
    if mode == 2:
        box = bf16_bits(nodes[0])
        defines = [("SLAB_BOX", ",".join(str(int(v)) for v in box))]
    readout = args.variant
    out, nonce = run_kernel(
        device,
        "slab",
        words,
        [n_nodes, args.reps, mode, readout],
        defines,
        readout == 2,
    )
    t = times_of(out)
    sweep, read_t = t[0::2], t[1::2]
    got = [int(v) for v in out[DATA : DATA + n_nodes]]
    want, gaps = slab_ref(inv, ob, tmax, lo, hi, fp64=not exact)
    bad_bits = 0
    near = 0
    for n in range(n_nodes):
        diff = got[n] ^ want[n]
        for i in range(32):
            if (diff >> i) & 1:
                bad_bits += 1
                near += int(gaps[n][i] < 1e-5 * max(1.0, float(tmax[i])))
    raw = (
        out[DATA + 64 : DATA + 64 + 32 * n_nodes].view(np.float32).reshape(n_nodes, 32)
    )
    return (
        {
            "nodes": n_nodes,
            "reps": args.reps,
            "mode": args.slab_mode,
            "readout": ["plain", "scheduled", "rvv"][readout],
            "exact_data": exact,
            "sweep_cycles": stats(sweep),
            "per_node": stats([c / n_nodes for c in sweep]),
            "readout_cycles": stats(read_t),
            "readout_per_node": stats([c / n_nodes for c in read_t]),
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


def riscv(device, args):
    mask, loops, n, n_float = args.mask, args.loops, args.n, args.n_float
    a = int(np.float32(0.999).view(np.uint32))
    b = int(np.float32(0.001).view(np.uint32))
    out, nonce = run_kernel(device, "riscv", [], [mask, loops, n, n_float, a, b])
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


def walk(device, args):
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
    out, nonce = run_kernel(device, "walk", words, [n, form, copy, best0], rvv=True)
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
            "best_max_abs_err": float(diff.max()) if diff.size else 0.0,
            "leaf_wrong": leaf_bad,
            "leaf_tie_ok": leaf_tie,
            "ref_rays_with_ties": int(sum(len(t) > 1 for t in ties)),
        },
        out,
        nonce,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("probe", choices=list(PROBES))
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
    p.add_argument("--n-float", type=int, default=10000)
    p.add_argument("--rays", type=int, default=32, help="walk: multiple of 32, <= 1024")
    p.add_argument("--ray-set", choices=["camera", "random"], default="random")
    p.add_argument("--form", choices=["a", "b"], default="a")
    p.add_argument("--copy", choices=["rvv", "scalar"], default="rvv")
    args = p.parse_args()
    if args.probe == "slab":
        args.nodes = min(args.nodes, 64)

    t_start = time.time()
    with open("/proc/loadavg") as f:
        load = f.read().split()[:3]
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    device.disable_and_clear_program_cache()
    log(f"device open {time.time() - t_start:.1f}s")
    try:
        mhz = [aiclk_mhz()]
        rec, out, nonce = globals()[args.probe](device, args)
        mhz.append(aiclk_mhz())
    finally:
        ttnn.close_mesh_device(device)
    rec = {"probe": args.probe, **rec}
    rec["nonce_ok"] = int(out[NONCE]) == nonce
    rec["kernel_error"] = int(out[ERROR])
    rec["aiclk_mhz"] = mhz
    rec["load"] = load
    rec["wall_s"] = round(time.time() - t_start, 1)
    print(json.dumps(rec), flush=True)
    return 0 if rec["nonce_ok"] and rec["kernel_error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
