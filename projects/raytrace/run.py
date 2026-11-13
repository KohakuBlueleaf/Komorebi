"""The ray tracer with the card as the intersector, on one scene, checked against the
reference pass by pass and, for the whole frame, image against image; --save writes
the result files (the reference's results layout) into DIR/card/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from projects.raytrace import card, packets, trace  # noqa: E402
from projects.raytrace.reference import harness  # noqa: E402
from ref import bvh, native, raytrace, results, scenes  # noqa: E402

STAGES = ("primary", "shadow", "bounce", "shadow2")


def choose_pixels(arguments):
    """Pixel numbers to trace, in packet order."""
    every = np.arange(scenes.WIDTH * scenes.HEIGHT)
    order, tile = packets.ray_order(every)
    if arguments.rays:
        return np.sort(every[order][: arguments.rays])
    if arguments.tiles:
        return every[np.isin(tile, arguments.tiles)]
    return every


def summary(info, mhz):
    """One pass: card time, rate, profile sums and the check, for the JSON report."""
    profile = info.pop("profile")
    info.pop("lanes")
    info.pop("tiles")
    busy = profile["total"]
    used = busy > 0
    microseconds = info["trace_microseconds"]
    result = dict(info)
    result["trace_microseconds_median"] = float(np.median(microseconds))
    result["mrays_per_s"] = info["rays"] / float(np.median(microseconds))
    result["busy_cycles"] = {
        "max": int(busy.max()),
        "mean": float(busy[used].mean()) if used.any() else 0.0,
        "min": int(busy[used].min()) if used.any() else 0,
    }
    result["busy_max_over_mean"] = result["busy_cycles"]["max"] / max(
        result["busy_cycles"]["mean"], 1
    )
    result["profile_sum"] = {
        name: int(values.sum()) for name, values in profile.items()
    }
    result["profile_per_core"] = {
        name: values.tolist() for name, values in profile.items()
    }
    result["mhz"] = mhz
    return result


def passes_ok(passes, worker_count):
    """Every pass ran cleanly, with no walk error, and agrees with the reference
    where it was checked (no hit, distance, (u, v) or carried attribute differs)."""
    ok = True
    for info in passes:
        ok &= card.ran_cleanly(info["run"], worker_count) and info["walk_errors"] == 0
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--rays", type=int, default=0, help="first n rays of tile 0")
    parser.add_argument(
        "--tiles", type=int, nargs="*", default=None, help="32 x 32 tiles"
    )
    parser.add_argument("--cores", type=int, default=0, help="0: every worker")
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--save", default=None, help="results folder")
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--treelets", type=int, default=0, help="KiB: treelets from DRAM"
    )
    parser.add_argument("--slots", type=int, default=4, help="treelet slots per core")
    parser.add_argument("--entries", type=int, default=0, help="list entries per core")
    parser.add_argument(
        "--rounds", action="store_true", help="r-th list entries per round"
    )
    parser.add_argument(
        "--per-command", type=int, default=0, help="packets per core at most"
    )
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--balance", action="store_true", help="tiles by measured cost")
    parser.add_argument(
        "--sort",
        choices=("none", "tile", "pass"),
        default="none",
        help="bounce rays by octant and origin: inside each tile or over the pass",
    )
    arguments = parser.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree(arguments.scene)
    scene = scenes.load_scene(data / f"{arguments.scene}.npz")
    cut = None
    if arguments.treelets:
        path = data / f"{arguments.scene}_treelets_{arguments.treelets}k.npz"
        if path.exists():
            with np.load(path) as archive:
                cut = {key: archive[key] for key in archive.files}
        else:
            cut = bvh.cut_treelets(tree, arguments.treelets * 1024)
    pixels = choose_pixels(arguments)
    report = {"scene": arguments.scene, "pixels": len(pixels), "cores": arguments.cores}
    report.update(
        treelets_kib=arguments.treelets, slots=arguments.slots, rounds=arguments.rounds
    )
    report.update(
        frames=arguments.frames, balance=arguments.balance, sort=arguments.sort
    )
    with card.open_session() as session:
        tracer = trace.CardTracer(
            session,
            tree,
            arguments.cores,
            check=not arguments.no_check,
            repetitions=arguments.reps,
            cut=cut,
            slots=arguments.slots,
            entries=arguments.entries,
            policy=0 if arguments.rounds else packets.K["kPolicyOnce"],
            per_command=arguments.per_command,
            balance=arguments.balance,
            sort=arguments.sort,
            scene=scene,
        )
        mhz = card.aiclk_mhz()
        started = time.perf_counter()
        for frame in range(arguments.frames):
            tracer.frame = frame
            if arguments.primary_only:
                origins, directions = raytrace.camera_rays(scene["camera"], pixels)
                tracer.trace(
                    "primary", origins, directions, raytrace.TMAX, False, pixels
                )
            else:
                colour, hit = trace.render_pixels(scene, tree, pixels, tracer)
        report["host_s"] = time.perf_counter() - started
    report["passes"] = [summary(info, mhz) for info in tracer.passes]
    if not arguments.primary_only and len(pixels) == scenes.WIDTH * scenes.HEIGHT:
        triangles = np.where(
            hit["leaf"] >= 0, tree["leaf_tri"][np.maximum(hit["leaf"], 0)], -1
        )
        shape = (scenes.HEIGHT, scenes.WIDTH)
        got = {
            "image": colour.reshape(shape + (3,)),
            "t": hit["t"].reshape(shape),
            "tri": triangles.astype(np.int32).reshape(shape),
            "nodes": hit["nodes"].reshape(shape),
        }
        name, reference = reference_frame(arguments.scene)
        stats = harness.compare(reference, got)
        mask = stats.pop("mask")
        report["image_check"] = {"against": name, **stats}
        if arguments.save:
            stage_ms = {
                f"{stage}_ms": entry["trace_microseconds_median"] / 1000
                for stage, entry in zip(STAGES, report["passes"][-len(STAGES) :])
            }
            stage_ms["total_ms"] = sum(stage_ms.values())
            last = report["passes"][-len(STAGES) :]
            rays = {
                f"rays_{stage}": entry["rays"] for stage, entry in zip(STAGES, last)
            }
            info = {
                "name": "Blackhole card",
                "scene": arguments.scene,
                "timings_ms": stage_ms,
                "rays": rays,
            }
            results.save_result(
                f"{arguments.save}/card", arguments.scene, "raytrace", got, info
            )
            np.save(f"{arguments.save}/card/{arguments.scene}_differs.npy", mask)
    report["ok"] = (
        passes_ok(report["passes"], tracer.workers)
        and report.get("image_check", {}).get("any_differs", 0) == 0
    )
    for pass_info in report["passes"]:
        keys = (
            "frame",
            "split",
            "sort",
            "pass",
            "rays",
            "packets",
            "trace_microseconds_median",
        )
        keys += ("walk_errors", "busy_max_over_mean")
        brief = {key: pass_info[key] for key in keys}
        print(
            json.dumps(
                {**brief, "check": pass_info.get("check"), "run": pass_info["run"]}
            )
        )
    last = {key: value for key, value in report.items() if key != "passes"}
    last["passes"] = [pass_info["pass"] for pass_info in report["passes"]]
    print(json.dumps(last), flush=True)  # the last line: the run's summary and ok
    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(report, handle, indent=1, default=int)
    return report["ok"]


if __name__ == "__main__":
    started = time.time()
    ok = main()
    print(f"wall {time.time() - started:.1f}s", flush=True)
    sys.exit(0 if ok else 1)
