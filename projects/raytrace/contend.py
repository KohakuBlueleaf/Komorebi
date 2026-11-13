"""Contention between cores: the heaviest pixel tile of one pass traced by one core
alone, then by every core at once (each its own copy), checked against the reference.

    python -m projects.raytrace.contend --scene s100k --treelets 128 --pass bounce
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from projects.raytrace import card, packets, trace
from projects.raytrace.reference import harness
from ref import native, raytrace, scenes


class HostTracer:
    """The reference's trace with every pass's rays and costs kept."""

    def __init__(self, tree):
        self.tree = tree
        self.passes = {}

    def trace(self, name, o, d, tmax, any_hit, pixels):
        hit = raytrace.trace(self.tree, o, d, tmax, any_hit)
        cost = hit["nodes"].astype(np.int64) + hit["leaves"]
        self.passes[name] = dict(o=o, d=d, tmax=tmax, any_hit=any_hit)
        self.passes[name].update(pixels=pixels, cost=cost)
        return hit


def heaviest(scene, tree, name):
    """Rays of the pixel tile with the most box and triangle tests in pass `name`."""
    host = HostTracer(tree)
    trace.render_pixels(scene, tree, np.arange(scenes.WIDTH * scenes.HEIGHT), host)
    p = host.passes[name]
    _, tile = packets.ray_order(p["pixels"])
    per_tile = np.bincount(tile, p["cost"], minlength=packets.TILES)
    best = int(np.argmax(per_tile))
    pick = tile == best
    out = {k: p[k][pick] for k in ("o", "d", "pixels", "cost")}
    out.update(tile=best, tmax=p["tmax"], any_hit=p["any_hit"])
    out["tile_cost_over_mean"] = float(per_tile[best] / per_tile[per_tile > 0].mean())
    return out


def copies(lanes, n, cores):
    """`cores` copies of the packets `lanes` of n rays, copy c on core c."""
    rows = [np.where(lanes >= 0, lanes + c * n, -1) for c in range(cores)]
    directory = np.zeros((cores, packets.K["kDirBytes"] // 4), dtype=np.uint32)
    directory[:, packets.K["kDirFirst"]] = np.arange(cores) * len(lanes)
    directory[:, packets.K["kDirCount"]] = len(lanes)
    tiles = np.zeros(cores * len(lanes), dtype=np.int64)
    return np.concatenate(rows), directory, tiles


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument("--pass", dest="name", default="bounce")
    p.add_argument("--treelets", type=int, default=0, help="treelet cut in KiB")
    p.add_argument("--rounds", action="store_true", help="r-th list entries per round")
    p.add_argument("--cores", default="1,0", help="core counts; 0 is every core")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--rays", default=None, help="npz cache of the picked rays")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree(args.scene)
    scene = scenes.load_scene(data / f"{args.scene}.npz")
    cut = None
    if args.treelets:
        with np.load(data / f"{args.scene}_treelets_{args.treelets}k.npz") as f:
            cut = {k: f[k] for k in f.files}
    if args.rays and Path(args.rays).exists():
        with np.load(args.rays) as f:
            rays = {k: f[k] for k in f.files}
    else:
        t0 = time.perf_counter()
        rays = heaviest(scene, tree, args.name)
        card.log(f"picked tile {rays['tile']} in {time.perf_counter() - t0:.1f} s")
        if args.rays:
            np.savez(args.rays, **rays)
    n = len(rays["o"])
    lanes, _, _ = packets.split(rays["pixels"], 1)
    report = {"scene": args.scene, "pass": args.name, "tile": int(rays["tile"])}
    report.update(rays=n, packets=len(lanes), treelets_kib=args.treelets)
    report["rounds"] = args.rounds
    report["tile_cost_over_mean"] = float(rays["tile_cost_over_mean"])
    report["runs"] = []
    with card.open_session() as session:
        policy = 0 if args.rounds else packets.K["kPolicyOnce"]
        tracer = trace.CardTracer(session, tree, reps=args.reps, cut=cut, policy=policy)
        for count in (int(c) or tracer.workers for c in args.cores.split(",")):
            tracer.cores = count
            layout = copies(lanes, n, count)
            o, d = (np.tile(rays[k], (count, 1)) for k in ("o", "d"))
            pixels = np.tile(rays["pixels"], count)
            any_hit = bool(rays["any_hit"])
            card.log(f"{count} cores: {count * n} rays")
            tracer.trace(args.name, o, d, rays["tmax"], any_hit, pixels, layout)
            info = tracer.passes[-1]
            busy = info["profile"]["total"][:count].astype(np.float64)
            run = {
                "cores": count,
                "trace_microseconds": float(np.median(info["trace_microseconds"])),
            }
            run.update(busy_first=float(busy[0]), busy_mean=float(busy.mean()))
            run.update(busy_max=float(busy.max()), busy_min=float(busy.min()))
            run.update(check=info["check"], run=info["run"])
            run["walk_errors"] = info["walk_errors"]
            if cut is not None:
                for k in (
                    "reader_fetch",
                    "reader_treelet_reads",
                    "reader_treelet_pages",
                ):
                    run[k] = float(info["profile"][k][:count].mean())
            report["runs"].append(run)
            print(json.dumps({k: v for k, v in run.items() if k != "run"}), flush=True)
    report["ok"] = all(
        card.ran_cleanly(r["run"], tracer.workers)
        and r["walk_errors"] == 0
        and not r["check"].get("tri_differs", r["check"].get("differ"))
        for r in report["runs"]
    )
    print(json.dumps({"ok": report["ok"]}), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=int)
    return report["ok"]


if __name__ == "__main__":
    t0 = time.time()
    ok = main()
    print(f"wall {time.time() - t0:.1f}s", flush=True)
    sys.exit(0 if ok else 1)
