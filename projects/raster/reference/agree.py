"""Compares rasterizers: differing pixels, where, error statistics. Runs numpy, C++ and
CUDA on each scene and compares every implementation found (also in --external folders,
see ref/results.py) with numpy, and C++ with CUDA; stored results are reused (--fresh:
run again). `python -m projects.raster.reference.agree [--scenes ...] [--fresh]`"""

import argparse
import json

from projects.raster.reference import harness


def agree(scene, external=(), numpy_max=None, fresh=False):
    """{"A vs B": statistics} for a scene over every implementation found."""
    harness.ensure_results(scene, numpy_max, fresh)
    found = harness.collect(scene, external)
    names = list(found)
    ref = harness.REFERENCE if harness.REFERENCE in found else names[0]
    pairs = [(ref, n) for n in names if n != ref]
    cpu, gpu = "C++ all threads", "V100"
    if cpu in found and gpu in found and ref not in (cpu, gpu):
        pairs.append((cpu, gpu))
    stats = {}
    for first, second in pairs:
        result = harness.compare(found[first][0], found[second][0])
        result.pop("mask")
        stats[f"{first} vs {second}"] = result
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--external", nargs="*", default=[])
    parser.add_argument("--numpy-max", type=int, default=None, help="largest scene")
    parser.add_argument("--fresh", action="store_true", help="run everything again")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = {
        s: agree(s, args.external, args.numpy_max, args.fresh) for s in args.scenes
    }
    for scene, pairs in report.items():
        for pair, s in pairs.items():
            print(
                f"{scene:6s} {pair:34s} differing pixels (tri/depth/colour/any) "
                f"{s['tri_differs']}/{s['depth_differs']}/{s['colour_differs']}/"
                f"{s['any_differs']} of {s['pixels']}, on edge {s['differs_on_edge']}, "
                f"max colour {s['colour_max_abs']:.3g}, "
                f"tile counts equal {s['tile_counts_equal']}"
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
