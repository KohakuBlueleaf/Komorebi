"""The trace pass's inputs off the card, as the project's run.py makes them: the tiny
scene's primary rays over every pixel and every core, their packet pages, the
directory and the tree image; written to build/inputs.npz beside this file.
Example: python prepare.py --scene tiny"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from projects.raytrace import packets  # noqa: E402
from projects.raytrace.reference import harness  # noqa: E402
from reference import native, raytrace, scenes  # noqa: E402

K = packets.K
CORES = 120  # the grid's cores (12 x 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--out", default=str(HERE / "build" / "inputs.npz"))
    arguments = parser.parse_args()
    started = time.perf_counter()
    tree = harness.ensure_tree(arguments.scene)
    scene = scenes.load_scene(native.data_dir() / f"{arguments.scene}.npz")
    pixels = np.arange(scenes.WIDTH * scenes.HEIGHT)
    origins, directions = raytrace.camera_rays(scene["camera"], pixels)
    lanes, directory, _ = packets.split(pixels, CORES)
    pages = packets.packet_pages(lanes, origins, directions, raytrace.TMAX)
    image = packets.tree_image(tree, scene)
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.out,
        image=image,
        directory=directory,
        packets=packets.group(pages),
        lanes=lanes,
        origins=origins,
        directions=directions,
        count=np.array([len(pages)]),
    )
    print(
        f"{arguments.scene}: {len(pixels)} rays, {len(pages)} packets on {CORES} "
        f"cores, tree {len(image)} pages; {arguments.out} in "
        f"{time.perf_counter() - started:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
