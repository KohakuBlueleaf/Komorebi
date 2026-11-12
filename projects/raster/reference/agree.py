"""Compares numpy, C++ and CUDA rasterizers: differing pixels, where, error statistics.

python -m projects.raster.reference.agree [--scenes tiny s10k ...] [--json FILE]
"""

import argparse
import json

from projects.raster.reference import harness

PAIRS = [("numpy", "cpu"), ("cpu", "gpu")]


def agree(scene, numpy_max=None):
    """Runs what is available on a scene; returns {pair name: statistics}."""
    outputs = {}
    if numpy_max is None or harness.scenes.PRESETS[scene] <= numpy_max:
        outputs["numpy"] = harness.run_numpy(scene)
    if harness.run_cpu(scene, harness.threads_all(), reps=1) is not None:
        outputs["cpu"] = harness.load_outputs(
            harness.out_dir(scene, f"cpu{harness.threads_all()}")
        )
    if harness.run_gpu(scene, reps=1) is not None:
        outputs["gpu"] = harness.load_outputs(harness.out_dir(scene, "gpu"))
    stats = {}
    for first, second in PAIRS:
        if first in outputs and second in outputs:
            result = harness.compare(outputs[first], outputs[second])
            result.pop("mask")
            stats[f"{first} vs {second}"] = result
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--numpy-max", type=int, default=None, help="largest scene")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = {scene: agree(scene, args.numpy_max) for scene in args.scenes}
    print("scene  pair  differing pixels (tri/depth/colour/any) on edge  max colour")
    for scene, pairs in report.items():
        for pair, s in pairs.items():
            print(
                f"{scene:6s} {pair:14s} {s['tri_differs']}/{s['depth_differs']}/"
                f"{s['colour_differs']}/{s['any_differs']} of {s['pixels']}, "
                f"on edge {s['differs_on_edge']}, "
                f"max colour {s['colour_max_abs']:.3g}, "
                f"tile counts equal {s['tile_counts_equal']}"
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
