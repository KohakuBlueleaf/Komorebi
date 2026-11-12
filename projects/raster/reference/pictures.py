"""Pictures of the rasterization: colour, depth, binning load, where outputs differ.

    python -m projects.raster.reference.pictures --out DIR [--scenes ...] [--small DIR]

--small writes the 256 x 256 colour images of the tiny and 10^4 scenes for the README.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from projects.raster.reference import harness
from ref import png, raster

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


def pictures(scene, out):
    """Writes the pictures of one scene into out; returns the numbers on them."""
    harness.run_cpu(scene, harness.threads_all(), reps=1)
    cpu = harness.load_outputs(harness.out_dir(scene, f"cpu{harness.threads_all()}"))
    meta = harness.scenes.load_scene(harness.native.data_dir() / f"{scene}.npz")["meta"]
    info = {"triangles": int(meta[0])}
    png.write_png(out / f"{scene}_colour.png", png.to_srgb8(cpu["colour"]))
    rgb, info["depth"] = depth_picture(cpu["depth"])
    png.write_png(out / f"{scene}_depth.png", rgb)
    rgb, info["tiles"] = tiles_picture(cpu["tile_counts"])
    png.write_png(out / f"{scene}_tiles.png", rgb)
    others = {"numpy": harness.run_numpy(scene)}
    if harness.run_gpu(scene, reps=1) is not None:
        others["gpu"] = harness.load_outputs(harness.out_dir(scene, "gpu"))
    mask = np.zeros(cpu["tri"].shape, dtype=bool)
    for other in others.values():
        mask |= harness.compare(cpu, other)["mask"]
    png.write_png(out / f"{scene}_diff.png", diff_picture(cpu["colour"], mask))
    info["differing_pixels"] = {"compared": sorted(others), "count": int(mask.sum())}
    return info, cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--small", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = {}
    for scene in args.scenes:
        index[scene], cpu = pictures(scene, args.out)
        if args.small and scene in SMALL_SCENES:
            args.small.mkdir(parents=True, exist_ok=True)
            small = png.downscale(png.to_srgb8(cpu["colour"]), 4)
            png.write_png(args.small / f"{scene}_colour_256.png", small)
    (args.out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
