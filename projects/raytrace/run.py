"""The ray tracer with the card as the intersector, on one scene, checked against the
reference pass by pass and, for the whole frame, image against image; --save writes
the result files (the reference's results layout) into DIR/card/.
"""

import argparse
import json
import sys
import time

import numpy as np

from projects.raytrace import card, packets, trace
from projects.raytrace.reference import harness
from ref import bvh, native, raytrace, results, scenes

STAGES = ("primary", "shadow", "bounce", "shadow2")


def choose_pixels(args):
    """Pixel numbers to trace, in packet order."""
    every = np.arange(scenes.WIDTH * scenes.HEIGHT)
    order, tile = packets.ray_order(every)
    if args.rays:
        return np.sort(every[order][: args.rays])
    if args.tiles:
        return every[np.isin(tile, args.tiles)]
    return every


def summary(info, mhz):
    """One pass: card time, rate, profile sums and the check, for the JSON report."""
    prof = info.pop("profile")
    info.pop("lanes")
    info.pop("tiles")
    busy = prof["total"]
    used = busy > 0
    us = info["trace_us"]
    out = dict(info)
    out["trace_us_median"] = float(np.median(us))
    out["mrays_per_s"] = info["rays"] / float(np.median(us))
    out["busy_cycles"] = {
        "max": int(busy.max()),
        "mean": float(busy[used].mean()) if used.any() else 0.0,
        "min": int(busy[used].min()) if used.any() else 0,
    }
    out["busy_max_over_mean"] = out["busy_cycles"]["max"] / max(
        out["busy_cycles"]["mean"], 1
    )
    out["profile_sum"] = {k: int(v.sum()) for k, v in prof.items()}
    out["profile_per_core"] = {k: v.tolist() for k, v in prof.items()}
    out["mhz"] = mhz
    return out


def passes_ok(passes, n_workers):
    """Every pass ran cleanly, with no walk error, and agrees with the reference
    where it was checked (no hit, distance, (u, v) or carried attribute differs)."""
    ok = True
    for info in passes:
        ok &= card.ran_cleanly(info["run"], n_workers) and info["walk_errors"] == 0
        check = info.get("check")
        if check is None:
            continue
        if info["any_hit"]:
            ok &= check["differ"] == 0
        else:
            ok &= check["tri_differs"] == check["t_differs"] == 0
            ok &= check["uv_differs"] == check.get("attr_differs", 0) == 0
    return bool(ok)


def reference_frame(scene):
    """The stored reference result of a scene (numpy where it exists, else C++); without
    one, numpy renders the frame first and stores it (seconds to a minute)."""
    found = harness.collect(scene)
    if not found:
        harness.run_numpy(scene)
        found = harness.collect(scene)
    name = harness.REFERENCE if harness.REFERENCE in found else sorted(found)[0]
    return name, found[name][0]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument("--rays", type=int, default=0, help="first n rays of tile 0")
    p.add_argument("--tiles", type=int, nargs="*", default=None, help="32 x 32 tiles")
    p.add_argument("--cores", type=int, default=0, help="0: every worker")
    p.add_argument("--primary-only", action="store_true")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--save", default=None, help="results folder")
    p.add_argument("--json", default=None)
    p.add_argument("--treelets", type=int, default=0, help="KiB: treelets from DRAM")
    p.add_argument("--slots", type=int, default=4, help="treelet slots per core")
    p.add_argument("--entries", type=int, default=0, help="list entries per core")
    p.add_argument("--rounds", action="store_true", help="r-th list entries per round")
    p.add_argument("--per-record", type=int, default=0, help="packets per core at most")
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--balance", action="store_true", help="tiles by measured cost")
    p.add_argument(
        "--sort",
        choices=("none", "tile", "pass"),
        default="none",
        help="bounce rays by octant and origin: inside each tile or over the pass",
    )
    args = p.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree(args.scene)
    scene = scenes.load_scene(data / f"{args.scene}.npz")
    cut = None
    if args.treelets:
        path = data / f"{args.scene}_treelets_{args.treelets}k.npz"
        if path.exists():
            with np.load(path) as f:
                cut = {k: f[k] for k in f.files}
        else:
            cut = bvh.cut_treelets(tree, args.treelets * 1024)
    pixels = choose_pixels(args)
    report = {"scene": args.scene, "pixels": len(pixels), "cores": args.cores}
    report.update(treelets_kib=args.treelets, slots=args.slots, rounds=args.rounds)
    report.update(frames=args.frames, balance=args.balance, sort=args.sort)
    with card.open_session() as session:
        tracer = trace.CardTracer(
            session,
            tree,
            args.cores,
            check=not args.no_check,
            reps=args.reps,
            cut=cut,
            slots=args.slots,
            entries=args.entries,
            policy=0 if args.rounds else packets.K["kPolicyOnce"],
            per_record=args.per_record,
            balance=args.balance,
            sort=args.sort,
            scene=scene,
        )
        mhz = card.aiclk_mhz()
        t0 = time.perf_counter()
        for frame in range(args.frames):
            tracer.frame = frame
            if args.primary_only:
                o, d = raytrace.camera_rays(scene["camera"], pixels)
                tracer.trace("primary", o, d, raytrace.TMAX, False, pixels)
            else:
                colour, hit = trace.render_pixels(scene, tree, pixels, tracer)
        report["host_s"] = time.perf_counter() - t0
    report["passes"] = [summary(info, mhz) for info in tracer.passes]
    if not args.primary_only and len(pixels) == scenes.WIDTH * scenes.HEIGHT:
        tri = np.where(
            hit["leaf"] >= 0, tree["leaf_tri"][np.maximum(hit["leaf"], 0)], -1
        )
        shape = (scenes.HEIGHT, scenes.WIDTH)
        got = {
            "image": colour.reshape(shape + (3,)),
            "t": hit["t"].reshape(shape),
            "tri": tri.astype(np.int32).reshape(shape),
            "nodes": hit["nodes"].reshape(shape),
        }
        name, ref = reference_frame(args.scene)
        stats = harness.compare(ref, got)
        mask = stats.pop("mask")
        report["image_check"] = {"against": name, **stats}
        if args.save:
            stage_ms = {
                f"{s}_ms": p["trace_us_median"] / 1000
                for s, p in zip(STAGES, report["passes"][-len(STAGES) :])
            }
            stage_ms["total_ms"] = sum(stage_ms.values())
            last = report["passes"][-len(STAGES) :]
            rays = {f"rays_{s}": p["rays"] for s, p in zip(STAGES, last)}
            info = {
                "name": "Blackhole card",
                "scene": args.scene,
                "timings_ms": stage_ms,
                "rays": rays,
            }
            results.save_result(f"{args.save}/card", args.scene, "raytrace", got, info)
            np.save(f"{args.save}/card/{args.scene}_differs.npy", mask)
    report["ok"] = (
        passes_ok(report["passes"], tracer.workers)
        and report.get("image_check", {}).get("any_differs", 0) == 0
    )
    for pinfo in report["passes"]:
        keys = ("frame", "split", "sort", "pass", "rays", "packets", "trace_us_median")
        keys += ("walk_errors", "busy_max_over_mean")
        brief = {k: pinfo[k] for k in keys}
        print(json.dumps({**brief, "check": pinfo.get("check"), "run": pinfo["run"]}))
    last = {k: v for k, v in report.items() if k != "passes"}
    last["passes"] = [pinfo["pass"] for pinfo in report["passes"]]
    print(json.dumps(last), flush=True)  # the last line: the run's summary and ok
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=int)
    return report["ok"]


if __name__ == "__main__":
    t0 = time.time()
    ok = main()
    print(f"wall {time.time() - t0:.1f}s", flush=True)
    sys.exit(0 if ok else 1)
