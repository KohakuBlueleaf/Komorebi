"""Pictures of the rasterization: colour, depth, binning load, where outputs differ, for
every implementation found (also in --external folders); --small writes 256 x 256 colour
images of the tiny and 10^4 scenes for the README.
`python -m projects.raster.reference.pictures --out DIR [--scenes ...] [--small DIR]`"""

import argparse
import json
from pathlib import Path

import numpy as np

from projects.raster.reference import harness
from reference import native, png, raster, scenes

SMALL_SCENES = ("tiny", "s10k")


def tiles_picture(counts):
    """Triangles per tile on the viridis map of log(1 + count), one block per tile."""
    rgb = png.colormap(np.log1p(counts.astype(np.float64)), 0.0, None)
    big = np.repeat(np.repeat(rgb, raster.TILE, axis=0), raster.TILE, axis=1)
    return big, {
        "tile_max": int(counts.max()),
        "tile_nonempty": int((counts > 0).sum()),
    }


def pictures(scene, folder, external):
    """Writes the pictures of one scene into `folder`; returns the numbers on them."""
    harness.ensure_results(scene)
    found = harness.collect(scene, external)
    reference_name = (
        harness.REFERENCE if harness.REFERENCE in found else next(iter(found))
    )
    reference = found[reference_name][0]
    meta = scenes.load_scene(native.data_dir() / f"{scene}.npz")["meta"]
    info = {
        "triangles": int(meta[0]),
        "reference": reference_name,
        "implementations": {},
    }
    rgb, info["depth"] = png.depth_picture(reference["depth"])
    png.write_png(folder / f"{scene}_depth.png", rgb)
    rgb, info["tiles"] = tiles_picture(reference["tile_counts"])
    png.write_png(folder / f"{scene}_tiles.png", rgb)
    for name, (arrays, record) in found.items():
        key = png.slug(name)
        png.write_png(
            folder / f"{scene}_{key}_colour.png", png.to_srgb8(arrays["colour"])
        )
        statistics = harness.compare(reference, arrays)
        png.write_png(
            folder / f"{scene}_{key}_diff.png",
            png.diff_picture(reference["colour"], statistics["mask"]),
        )
        info["implementations"][key] = {
            "name": name,
            "differing_pixels": statistics["any_differs"],
            "seconds_per_frame": record.get("seconds_per_frame"),
            "timings_ms": record.get("timings_ms"),
            "power": record.get("power"),
        }
    return info, reference


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
        index[scene], reference_arrays = pictures(scene, args.out, args.external)
        if args.small and scene in SMALL_SCENES:
            args.small.mkdir(parents=True, exist_ok=True)
            small = png.downscale(png.to_srgb8(reference_arrays["colour"]), 4)
            png.write_png(args.small / f"{scene}_colour_256.png", small)
    (args.out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
