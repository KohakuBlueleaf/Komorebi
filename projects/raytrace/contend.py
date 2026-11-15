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

from komorebi import power
from projects.raytrace import card, packets, trace
from projects.raytrace.reference import harness
from reference import native, raytrace, scenes


class HostTracer:
    """The reference's trace with every pass's rays and costs kept."""

    def __init__(self, tree):
        self.tree = tree
        self.passes = {}

    def trace(self, name, origins, directions, tmax, any_hit, pixels):
        hit = raytrace.trace(self.tree, origins, directions, tmax, any_hit)
        cost = hit["nodes"].astype(np.int64) + hit["leaves"]
        self.passes[name] = dict(o=origins, d=directions, tmax=tmax, any_hit=any_hit)
        self.passes[name].update(pixels=pixels, cost=cost)
        return hit


def heaviest(scene, tree, name):
    """Rays of the pixel tile with the most box and triangle tests in pass `name`."""
    host = HostTracer(tree)
    trace.render_pixels(scene, tree, np.arange(scenes.WIDTH * scenes.HEIGHT), host)
    recorded = host.passes[name]
    _, tile = packets.ray_order(recorded["pixels"])
    per_tile = np.bincount(tile, recorded["cost"], minlength=packets.TILES)
    best = int(np.argmax(per_tile))
    pick = tile == best
    picked = {key: recorded[key][pick] for key in ("o", "d", "pixels", "cost")}
    picked.update(tile=best, tmax=recorded["tmax"], any_hit=recorded["any_hit"])
    picked["tile_cost_over_mean"] = float(
        per_tile[best] / per_tile[per_tile > 0].mean()
    )
    return picked


def copies(lanes, rays, cores):
    """`cores` copies of the packets `lanes` of `rays` rays, copy number `core` on core
    `core`."""
    rows = [np.where(lanes >= 0, lanes + core * rays, -1) for core in range(cores)]
    directory = np.zeros((cores, packets.K["kDirBytes"] // 4), dtype=np.uint32)
    directory[:, packets.K["kDirFirst"]] = np.arange(cores) * len(lanes)
    directory[:, packets.K["kDirCount"]] = len(lanes)
    tiles = np.zeros(cores * len(lanes), dtype=np.int64)
    return np.concatenate(rows), directory, tiles


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--pass", dest="name", default="bounce")
    parser.add_argument("--treelets", type=int, default=0, help="treelet cut in KiB")
    parser.add_argument(
        "--rounds", action="store_true", help="r-th list entries per round"
    )
    parser.add_argument("--cores", default="1,0", help="core counts; 0 is every core")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--rays", default=None, help="npz cache of the picked rays")
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree(arguments.scene)
    scene = scenes.load_scene(data / f"{arguments.scene}.npz")
    cut = None
    if arguments.treelets:
        with np.load(
            data / f"{arguments.scene}_treelets_{arguments.treelets}k.npz"
        ) as archive:
            cut = {key: archive[key] for key in archive.files}
    if arguments.rays and Path(arguments.rays).exists():
        with np.load(arguments.rays) as archive:
            rays = {key: archive[key] for key in archive.files}
    else:
        started = time.perf_counter()
        rays = heaviest(scene, tree, arguments.name)
        card.log(f"picked tile {rays['tile']} in {time.perf_counter() - started:.1f} s")
        if arguments.rays:
            np.savez(arguments.rays, **rays)
    ray_count = len(rays["o"])
    lanes, _, _ = packets.split(rays["pixels"], 1)
    report = {
        "scene": arguments.scene,
        "pass": arguments.name,
        "tile": int(rays["tile"]),
    }
    report.update(rays=ray_count, packets=len(lanes), treelets_kib=arguments.treelets)
    report["rounds"] = arguments.rounds
    report["tile_cost_over_mean"] = float(rays["tile_cost_over_mean"])
    report["runs"] = []
    with card.open_session() as session:
        policy = 0 if arguments.rounds else packets.K["kPolicyOnce"]
        tracer = trace.CardTracer(
            session, tree, repetitions=arguments.reps, cut=cut, policy=policy
        )
        for count in (
            int(text) or tracer.workers for text in arguments.cores.split(",")
        ):
            tracer.cores = count
            layout = copies(lanes, ray_count, count)
            origins, directions = (np.tile(rays[key], (count, 1)) for key in ("o", "d"))
            pixels = np.tile(rays["pixels"], count)
            any_hit = bool(rays["any_hit"])
            card.log(f"{count} cores: {count * ray_count} rays")
            tracer.trace(
                arguments.name,
                origins,
                directions,
                rays["tmax"],
                any_hit,
                pixels,
                layout,
            )
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
                for key in (
                    "reader_fetch",
                    "reader_treelet_reads",
                    "reader_treelet_pages",
                ):
                    run[key] = float(info["profile"][key][:count].mean())
            report["runs"].append(run)
            print(
                json.dumps(
                    power.clocked(
                        {key: value for key, value in run.items() if key != "run"}
                    )
                ),
                flush=True,
            )
    report["ok"] = all(
        card.ran_cleanly(entry["run"], tracer.workers)
        and entry["walk_errors"] == 0
        and not entry["check"].get("tri_differs", entry["check"].get("differ"))
        for entry in report["runs"]
    )
    print(json.dumps(power.clocked({"ok": report["ok"]})), flush=True)
    if arguments.json:
        with open(arguments.json, "w") as archive:
            json.dump(report, archive, indent=1, default=int)
    return report["ok"]


if __name__ == "__main__":
    started = time.time()
    ok = main()
    print(f"wall {time.time() - started:.1f}s", flush=True)
    sys.exit(0 if ok else 1)
