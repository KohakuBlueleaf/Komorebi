"""Context row: the same rays through Embree (Python package `embreex`).

Not part of the reference: Embree has its own tree and triangle test, so it agrees with
ours up to ties and rounding. Rays and timing: this folder's README.
"""

import argparse
import importlib.metadata
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from embreex import rtcore_scene
from embreex.mesh_construction import TriangleMesh

from projects.raytrace.reference import harness
from ref import native, raytrace, runner, scenes

F = np.float32
GROUPS = ("primary", "shadow", "bounce", "shadow2")


def chunks(o, d, threads):
    """Views of the rays in pieces for a thread pool (one piece for 1 thread)."""
    edges = np.linspace(0, len(o), (4 * threads if threads > 1 else 1) + 1).astype(int)
    return [(o[i:j], d[i:j]) for i, j in zip(edges[:-1], edges[1:])]


def trace(embree, kind, pieces, threads):
    """Primitive ids (-1: none) of one group of rays: kind is "closest" or "any"."""

    def one(piece):
        o, d = piece
        if kind == "closest":
            return embree.run(o, d)
        return embree.run(o, d, dists=np.ones(len(o), dtype=F), query="OCCLUDED")

    if threads == 1:
        return np.concatenate([one(p) for p in pieces])
    with ThreadPoolExecutor(threads) as pool:
        return np.concatenate(list(pool.map(one, pieces)))


def hit_points(scene, tree, leaf_of, tri, o, d, t):
    """Surface points, ray origins just above them and the normals of the hits."""
    zero = np.zeros(len(tri), dtype=F)
    hit = {"leaf": leaf_of[tri], "u": zero, "v": zero}
    n, _ = raytrace.surface(scene, tree, hit, d)
    p = o + d * t[:, None]
    return p + n * scene["eps"][0], n


def frame_rays(scene, tree, embree, numpy_result):
    """The four groups of rays as {name: (kind, origins, directions)}."""
    pixels = np.arange(scenes.WIDTH * scenes.HEIGHT)
    o, d = raytrace.camera_rays(scene["camera"], pixels)
    tri, t = numpy_result["tri"].ravel(), numpy_result["t"].ravel()
    leaf_of = np.empty(len(tree["leaf_tri"]), dtype=np.int64)
    leaf_of[tree["leaf_tri"]] = np.arange(len(leaf_of))
    h = np.nonzero(tri >= 0)[0]
    o2, n = hit_points(scene, tree, leaf_of, tri[h], o[h], d[h], t[h])
    db = raytrace.bounce_directions(pixels[h], n)
    bounce_ids = trace(embree, "closest", chunks(o2.astype(F), db.astype(F), 1), 1)
    b = np.nonzero(bounce_ids >= 0)[0]
    t1 = embree.run(o2[b].astype(F), db[b].astype(F), query="DISTANCE")
    q, _ = hit_points(scene, tree, leaf_of, bounce_ids[b], o2[b], db[b], t1)
    to_light = scene["light"] - o2
    groups = {
        "primary": ("closest", o, d),
        "shadow": ("any", o2, to_light),
        "bounce": ("closest", o2, db),
        "shadow2": ("any", q, scene["light"] - q),
    }
    return {k: (kind, x.astype(F), y.astype(F)) for k, (kind, x, y) in groups.items()}


def measure(embree, groups, threads, runs):
    """Median seconds per group over runs (after one untimed pass) and the loads."""
    times = {k: [] for k in GROUPS}
    loads = []
    for _ in range(runs + 1):
        loads.append(native.loadavg())
        for k in GROUPS:
            kind, o, d = groups[k]
            pieces = chunks(o, d, threads)
            start = time.perf_counter()
            trace(embree, kind, pieces, threads)
            times[k].append(time.perf_counter() - start)
    return {k: statistics.median(v[1:]) for k, v in times.items()}, loads


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    version = importlib.metadata.version("embreex")
    all_threads = runner.threads_all()
    modes = [(n, f"{n} thread{'s' * (n > 1)}") for n in (1, 24, all_threads)]
    report = {}
    print(
        "| scene | implementation | primary | shadow | bounce | together | "
        "ms per frame | host load |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name in args.scenes:
        scene = scenes.load_scene(native.data_dir() / f"{name}.npz")
        tree = harness.ensure_tree(name)
        embree = rtcore_scene.EmbreeScene()
        positions = scene["positions"].astype(F)
        TriangleMesh(embree, positions, scene["indices"].astype(np.int32))
        harness.ensure_results(name)
        reference = harness.collect(name)["numpy"][0]
        groups = frame_rays(scene, tree, embree, reference)
        counts = {k: len(groups[k][1]) for k in GROUPS}
        same = trace(embree, "closest", chunks(*groups["primary"][1:], 1), 1)
        same = float((same == reference["tri"].ravel()).mean())
        report[name] = {"embreex": version, "rays": counts, "primary_same": same}
        for threads, label in modes:
            seconds, loads = measure(embree, groups, threads, args.runs)
            rate = {k: counts[k] / seconds[k] / 1e6 for k in GROUPS}
            rate["together"] = sum(counts.values()) / sum(seconds.values()) / 1e6
            frame_ms = sum(seconds.values()) * 1000
            report[name][label] = {"mray_per_s": rate, "frame_ms": frame_ms}
            print(
                f"| {name} | Embree via embreex {version}, {label} (a different "
                f"implementation, own tree) | {rate['primary']:.2f} | "
                f"{rate['shadow']:.2f} | {rate['bounce']:.2f} | "
                f"{rate['together']:.2f} | {frame_ms:.1f} | {loads[-1]} |"
            )
        print(f"<!-- {name}: rays {counts}, primary same triangle {same:.6f} -->")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
