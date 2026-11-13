"""Compares ray tracers: differing pixels, nearest-hit mismatches and ties, image error.

Every implementation found (also in --external folders) against numpy, and C++ against
CUDA; method in this folder's README. Stored results are reused unless --fresh.
"""

import argparse
import json

from projects.raytrace.reference import harness
from ref import native

STRIDE = 4


def restrict(arrays, pixels):
    """The values of the given pixel numbers, as flat arrays."""
    restricted = {}
    for key, value in arrays.items():
        if key == "image" or (key == "colour" and value.ndim == 3):
            restricted[key] = value.reshape(-1, 3)[pixels]
        elif value.ndim == 2:
            restricted[key] = value.reshape(-1)[pixels]
    return restricted


def prepare(scene, numpy_max, fresh=False):
    """Runs the implementations of this repository; returns numpy's subset or None."""
    subset = None
    harness.ensure_results(scene, numpy_max, fresh)
    if numpy_max is not None and native.scene_triangles(scene) > numpy_max:
        pixels = harness.lattice(STRIDE)
        subset = (pixels, harness.run_numpy(scene, pixels))
    return subset


def agree(scene, external=(), numpy_max=None, fresh=False):
    """{"A vs B": statistics} for a scene over every implementation found."""
    subset = prepare(scene, numpy_max, fresh)
    found = harness.collect(scene, external)
    names = list(found)
    pairs, stats = [], {}
    if harness.REFERENCE in found:
        reference = harness.REFERENCE
        pairs = [(reference, name) for name in names if name != reference]
    elif subset is not None:
        pixels, reference_arrays = subset
        for name in names:
            label = f"numpy (every {STRIDE}th pixel) vs {name}"
            result = harness.compare(reference_arrays, restrict(found[name][0], pixels))
            result.pop("mask")
            stats[label] = result
    cpu_name, gpu_name = "C++ all threads", "V100"
    if cpu_name in found and gpu_name in found:
        pairs.append((cpu_name, gpu_name))
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
    arguments = parser.parse_args()
    report = {
        scene: agree(scene, arguments.external, arguments.numpy_max, arguments.fresh)
        for scene in arguments.scenes
    }
    for scene, pairs in report.items():
        for pair, stats in pairs.items():
            print(f"{scene:6s} {pair}")
            print(
                f"    differing pixels {stats['any_differs']} of {stats['pixels']} "
                f"(on edge {stats.get('differs_on_edge', 'n/a')}); nearest triangle "
                f"differs {stats['tri_differs']}: exact ties "
                f"{stats['tri_exact_ties']}, near ties {stats['tri_near_ties']}, "
                f"hit vs miss "
                f"{stats['tri_hit_vs_miss']}, other {stats['tri_other']}"
            )
            print(
                f"    image max abs {stats['image_max_abs']:.3g}, rms "
                f"{stats['image_rms']:.3g}, pixels over 1/255: "
                f"{stats['image_over_1_255']}"
            )
    if arguments.json:
        with open(arguments.json, "w") as file:
            json.dump(report, file, indent=1)


if __name__ == "__main__":
    main()
