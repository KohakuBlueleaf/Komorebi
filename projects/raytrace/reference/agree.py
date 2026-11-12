"""Compares ray tracers: differing pixels, nearest-hit mismatches and ties, image error.

Runs numpy, C++ and CUDA on each scene and compares every implementation found (also
those in --external folders, see ref/results.py) with numpy, and C++ with CUDA. Where
numpy is too slow for the whole image (scenes above --numpy-max triangles) it traces
every STRIDE-th pixel in both directions and the comparison covers those pixels.

    python -m projects.raytrace.reference.agree [--scenes ...] [--external DIR ...]
        [--numpy-max N] [--fresh]

Results already stored in the data folder are reused (with their timings and power);
--fresh runs everything again.
"""

import argparse
import json

from projects.raytrace.reference import harness
from ref import native

STRIDE = 4


def restrict(arrays, pixels):
    """The values of the given pixel numbers, as flat arrays."""
    out = {}
    for key, value in arrays.items():
        if key == "image" or (key == "colour" and value.ndim == 3):
            out[key] = value.reshape(-1, 3)[pixels]
        elif value.ndim == 2:
            out[key] = value.reshape(-1)[pixels]
    return out


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
        ref = harness.REFERENCE
        pairs = [(ref, n) for n in names if n != ref]
    elif subset is not None:
        pixels, ref_arrays = subset
        for name in names:
            label = f"numpy (every {STRIDE}th pixel) vs {name}"
            result = harness.compare(ref_arrays, restrict(found[name][0], pixels))
            result.pop("mask")
            stats[label] = result
    cpu, gpu = "C++ all threads", "V100"
    if cpu in found and gpu in found:
        pairs.append((cpu, gpu))
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
            print(f"{scene:6s} {pair}")
            print(
                f"    differing pixels {s['any_differs']} of {s['pixels']} "
                f"(on edge {s.get('differs_on_edge', 'n/a')}); nearest triangle "
                f"differs {s['tri_differs']}: exact ties {s['tri_exact_ties']}, "
                f"near ties {s['tri_near_ties']}, hit vs miss {s['tri_hit_vs_miss']}, "
                f"other {s['tri_other']}"
            )
            print(
                f"    image max abs {s['image_max_abs']:.3g}, rms "
                f"{s['image_rms']:.3g}, pixels over 1/255: {s['image_over_1_255']}"
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
