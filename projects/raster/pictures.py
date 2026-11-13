"""Pictures of a card run (`run.py --out` and its profile) against the reference's
stored result: colour, where it differs, busy time.
`python -m projects.raster.pictures --scene tiny --card DIR --prof P.npz --out DIR`"""

import argparse
from pathlib import Path

import numpy as np

from projects.raster import lists, stress
from projects.raster.reference import harness
from ref import png, raster, results, runner


def reference(scene):
    """The reference outputs: numpy for tiny and the stress scenes, else C++ on all
    threads (equal to it)."""
    if scene in stress.NAMES:
        return raster.rasterize(stress.make(scene))
    folder = runner.results_root("raster") / ("numpy" if scene == "tiny" else "cpu_all")
    if not (folder / f"{scene}.npz").exists():
        if scene == "tiny":
            harness.run_numpy(scene)
        else:
            harness.run_cpu(scene, runner.threads_all(), reps=1)
    return results.load_result(folder, scene, "raster")[0]


def busy_picture(prof_file):
    """Each tile coloured by the busy cycles of the core that filled it (viridis)."""
    data = np.load(prof_file)
    total = data["prof"][:, 10].astype(np.float64)
    per_tile = np.zeros(lists.TILES)
    per_tile[data["tiles_of"]] = total[data["owner"]]
    grid = np.repeat(np.repeat(per_tile.reshape(32, 32), 32, axis=0), 32, axis=1)
    return png.colormap(grid, 0.0, None), {
        "busy_max_cycles": int(total.max()),
        "busy_min_cycles": int(total[total > 0].min()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument("--card", type=Path, required=True, help="the card's result folder")
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s = args.scene
    got = results.load_result(args.card, s, "raster")[0]
    mask = harness.compare(reference(s), got)["mask"]
    png.write_png(args.out / f"{s}_card_colour.png", png.to_srgb8(got["colour"]))
    png.write_png(
        args.out / f"{s}_card_diff.png", png.diff_picture(got["colour"], mask)
    )
    rgb, info = busy_picture(args.profile)
    png.write_png(args.out / f"{s}_card_busy.png", rgb)
    info["differing_pixels"] = int(mask.sum())
    print(info)


if __name__ == "__main__":
    main()
