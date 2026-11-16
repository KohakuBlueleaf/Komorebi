"""Builds the probe images into build/ (out of tree).

Usage: python build.py [--tree TT_METAL_TREE] [names...]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from komorebi import _env  # noqa: E402
from komorebi.builder import flags as build  # noqa: E402

IMAGES = ["beat", "dram", "rates"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, default=_env.metal_home())
    parser.add_argument("--out", type=Path, default=HERE / "build")
    parser.add_argument("names", nargs="*", default=IMAGES)
    args = parser.parse_args()
    if args.tree is None:
        sys.exit(f"build.py: pass --tree or set {_env.HOME_VARIABLE}")
    for name in args.names:
        image = args.out / f"{name}.bin"
        source = HERE / "x280" / f"{name}.c"
        build.build_image(args.tree, image, [source], includes=[HERE / "x280"])
        print(f"{image}: {image.stat().st_size} bytes")


if __name__ == "__main__":
    main()
