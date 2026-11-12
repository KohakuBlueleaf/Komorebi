"""Pictures of the rasterization: colour, depth, binning load, where outputs differ.

    python -m projects.raster.reference.pictures --out DIR [--scenes ...] [--small DIR]
        [--external DIR ...]

Every implementation found (also in --external folders, see ref/results.py) gets its
colour image and a map of where it differs from numpy. --small writes the 256 x 256
colour images of the tiny and 10^4 scenes for the README.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from projects.raster.reference import agree, harness
from ref import native, png, raster, scenes

SMALL_SCENES = ("tiny", "s10k")


def depth_picture(depth):
    """Log of the view distance 1/w on the viridis map; empty pixels black."""
    hit = depth > 0
    z = np.log(1.0 / np.where(hit, depth, 1.0))
    rgb = png.colormap(z, z[hit].min(), z[hit].max())
    rgb[~hit] = 0
    return rgb, {
        "z_min": float(np.exp(z[hit].min())),
        "z_max": float(np.exp(z[hit].max())),
    }


def tiles_picture(counts):
    """Triangles per tile on the viridis map of log(1 + count), one block per tile."""
    rgb = png.colormap(np.log1p(counts.astype(np.float64)), 0.0, None)
    big = np.repeat(np.repeat(rgb, raster.TILE, axis=0), raster.TILE, axis=1)
    return big, {
        "tile_max": int(counts.max()),
        "tile_nonempty": int((counts > 0).sum()),
    }


def diff_picture(colour, mask):
    """The colour image dimmed, with differing pixels (grown by 2) in red."""
    base = (png.to_srgb8(colour) * 0.35).astype(np.uint8)
    grown = mask.copy()
    for _ in range(2):
        grown[1:] |= grown[:-1]
        grown[:-1] |= grown[1:]
        grown[:, 1:] |= grown[:, :-1]
        grown[:, :-1] |= grown[:, 1:]
    base[grown] = [255, 40, 40]
    return base


def slug(name):
    """A file-name form of an implementation name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("+", "p")).strip("_")


def pictures(scene, out, external):
    """Writes the pictures of one scene into out; returns the numbers on them."""
    agree.prepare(scene, None)
    found = harness.collect(scene, external)
    ref_name = harness.REFERENCE if harness.REFERENCE in found else next(iter(found))
    ref = found[ref_name][0]
    meta = scenes.load_scene(native.data_dir() / f"{scene}.npz")["meta"]
    info = {"triangles": int(meta[0]), "reference": ref_name, "implementations": {}}
    rgb, info["depth"] = depth_picture(ref["depth"])
    png.write_png(out / f"{scene}_depth.png", rgb)
    rgb, info["tiles"] = tiles_picture(ref["tile_counts"])
    png.write_png(out / f"{scene}_tiles.png", rgb)
    for name, (arrays, run) in found.items():
        key = slug(name)
        png.write_png(out / f"{scene}_{key}_colour.png", png.to_srgb8(arrays["colour"]))
        stats = harness.compare(ref, arrays)
        png.write_png(
            out / f"{scene}_{key}_diff.png", diff_picture(ref["colour"], stats["mask"])
        )
        info["implementations"][key] = {
            "name": name,
            "differing_pixels": stats["any_differs"],
            "seconds_per_frame": run.get("seconds_per_frame"),
            "timings_ms": run.get("timings_ms"),
            "power": run.get("power"),
        }
    return info, ref


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--small", type=Path, default=None)
    parser.add_argument("--external", nargs="*", default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = {}
    for scene in args.scenes:
        index[scene], cpu = pictures(scene, args.out, args.external)
        if args.small and scene in SMALL_SCENES:
            args.small.mkdir(parents=True, exist_ok=True)
            small = png.downscale(png.to_srgb8(cpu["colour"]), 4)
            png.write_png(args.small / f"{scene}_colour_256.png", small)
    (args.out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
