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


def chunks(origins, directions, threads):
    """Views of the rays in pieces for a thread pool (one piece for 1 thread)."""
    edges = np.linspace(
        0, len(origins), (4 * threads if threads > 1 else 1) + 1
    ).astype(int)
    return [
        (origins[start:end], directions[start:end])
        for start, end in zip(edges[:-1], edges[1:])
    ]


def trace(embree, kind, pieces, threads):
    """Primitive ids (-1: none) of one group of rays: kind is "closest" or "any"."""

    def one(piece):
        origins, directions = piece
        if kind == "closest":
            return embree.run(origins, directions)
        return embree.run(
            origins, directions, dists=np.ones(len(origins), dtype=F), query="OCCLUDED"
        )

    if threads == 1:
        return np.concatenate([one(piece) for piece in pieces])
    with ThreadPoolExecutor(threads) as pool:
        return np.concatenate(list(pool.map(one, pieces)))


def hit_points(scene, tree, leaf_of, triangle_ids, origins, directions, distances):
    """Surface points, ray origins just above them and the normals of the hits."""
    zero = np.zeros(len(triangle_ids), dtype=F)
    hit = {"leaf": leaf_of[triangle_ids], "u": zero, "v": zero}
    normals, _ = raytrace.surface(scene, tree, hit, directions)
    points = origins + directions * distances[:, None]
    return points + normals * scene["eps"][0], normals


def frame_rays(scene, tree, embree, numpy_result):
    """The four groups of rays as {name: (kind, origins, directions)}."""
    pixels = np.arange(scenes.WIDTH * scenes.HEIGHT)
    origins, directions = raytrace.camera_rays(scene["camera"], pixels)
    triangle_ids, distances = numpy_result["tri"].ravel(), numpy_result["t"].ravel()
    leaf_of = np.empty(len(tree["leaf_tri"]), dtype=np.int64)
    leaf_of[tree["leaf_tri"]] = np.arange(len(leaf_of))
    hit_rows = np.nonzero(triangle_ids >= 0)[0]
    shadow_origins, normals = hit_points(
        scene,
        tree,
        leaf_of,
        triangle_ids[hit_rows],
        origins[hit_rows],
        directions[hit_rows],
        distances[hit_rows],
    )
    bounce_directions = raytrace.bounce_directions(pixels[hit_rows], normals)
    bounce_ids = trace(
        embree,
        "closest",
        chunks(shadow_origins.astype(F), bounce_directions.astype(F), 1),
        1,
    )
    bounce_rows = np.nonzero(bounce_ids >= 0)[0]
    bounce_distances = embree.run(
        shadow_origins[bounce_rows].astype(F),
        bounce_directions[bounce_rows].astype(F),
        query="DISTANCE",
    )
    bounce_points, _ = hit_points(
        scene,
        tree,
        leaf_of,
        bounce_ids[bounce_rows],
        shadow_origins[bounce_rows],
        bounce_directions[bounce_rows],
        bounce_distances,
    )
    to_light = scene["light"] - shadow_origins
    groups = {
        "primary": ("closest", origins, directions),
        "shadow": ("any", shadow_origins, to_light),
        "bounce": ("closest", shadow_origins, bounce_directions),
        "shadow2": ("any", bounce_points, scene["light"] - bounce_points),
    }
    return {
        group: (kind, group_origins.astype(F), group_directions.astype(F))
        for group, (kind, group_origins, group_directions) in groups.items()
    }


def measure(embree, groups, threads, runs):
    """Median seconds per group over runs (after one untimed pass) and the loads."""
    times = {group: [] for group in GROUPS}
    loads = []
    for _ in range(runs + 1):
        loads.append(native.loadavg())
        for group in GROUPS:
            kind, origins, directions = groups[group]
            pieces = chunks(origins, directions, threads)
            start = time.perf_counter()
            trace(embree, kind, pieces, threads)
            times[group].append(time.perf_counter() - start)
    return {
        group: statistics.median(durations[1:]) for group, durations in times.items()
    }, loads


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    version = importlib.metadata.version("embreex")
    all_threads = runner.threads_all()
    modes = [
        (thread_count, f"{thread_count} thread{'s' * (thread_count > 1)}")
        for thread_count in (1, 24, all_threads)
    ]
    report = {}
    print(
        "| scene | implementation | primary | shadow | bounce | together | "
        "ms per frame | host load |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name in arguments.scenes:
        scene = scenes.load_scene(native.data_dir() / f"{name}.npz")
        tree = harness.ensure_tree(name)
        embree = rtcore_scene.EmbreeScene()
        positions = scene["positions"].astype(F)
        TriangleMesh(embree, positions, scene["indices"].astype(np.int32))
        harness.ensure_results(name)
        reference = harness.collect(name)["numpy"][0]
        groups = frame_rays(scene, tree, embree, reference)
        counts = {group: len(groups[group][1]) for group in GROUPS}
        same = trace(embree, "closest", chunks(*groups["primary"][1:], 1), 1)
        same = float((same == reference["tri"].ravel()).mean())
        report[name] = {"embreex": version, "rays": counts, "primary_same": same}
        for threads, label in modes:
            seconds, loads = measure(embree, groups, threads, arguments.runs)
            rate = {group: counts[group] / seconds[group] / 1e6 for group in GROUPS}
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
    if arguments.json:
        with open(arguments.json, "w") as file:
            json.dump(report, file, indent=1)


if __name__ == "__main__":
    main()
