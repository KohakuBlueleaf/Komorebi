"""Compares rasterizers: differing pixels, where, error statistics. Runs numpy, C++ and
CUDA on each scene and compares every implementation found (also in --external folders,
see reference/results.py) with numpy, and C++ with CUDA; stored results are reused.
`python -m projects.raster.reference.agree [--scenes ...] [--fresh]`"""

import argparse
import json

from projects.raster.reference import harness


def agree(scene, external=(), numpy_max=None, fresh=False):
    """{"A vs B": statistics} for a scene over every implementation found."""
    harness.ensure_results(scene, numpy_max, fresh)
    found = harness.collect(scene, external)
    names = list(found)
    reference = harness.REFERENCE if harness.REFERENCE in found else names[0]
    pairs = [(reference, name) for name in names if name != reference]
    cpu, gpu = "C++ all threads", "V100"
    if cpu in found and gpu in found and reference not in (cpu, gpu):
        pairs.append((cpu, gpu))
    statistics = {}
    for first, second in pairs:
        result = harness.compare(found[first][0], found[second][0])
        result.pop("mask")
        statistics[f"{first} vs {second}"] = result
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--external", nargs="*", default=[])
    parser.add_argument("--numpy-max", type=int, default=None, help="largest scene")
    parser.add_argument("--fresh", action="store_true", help="run everything again")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = {
        scene: agree(scene, args.external, args.numpy_max, args.fresh)
        for scene in args.scenes
    }
    for scene, pairs in report.items():
        for pair, statistics in pairs.items():
            print(
                f"{scene:6s} {pair:34s} differing pixels (tri/depth/colour/any) "
                f"{statistics['tri_differs']}/{statistics['depth_differs']}/"
                f"{statistics['colour_differs']}/{statistics['any_differs']} of "
                f"{statistics['pixels']}, on edge {statistics['differs_on_edge']}, "
                f"max colour {statistics['colour_max_abs']:.3g}, "
                f"tile counts equal {statistics['tile_counts_equal']}"
            )
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)


if __name__ == "__main__":
    main()
