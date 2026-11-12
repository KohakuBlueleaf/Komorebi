"""Builds the dispatch kernel image and this harness's host module, out of tree.

    python build.py [--tree TT_METAL_TREE] [--out DIR] [--image-only] [--define N=V]

The image is linked with the X280 runtime of the tt-metal tree (TT_METAL_HOME by
default); the module borrows the flags of the tree's X280 Python module target.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from komorebi import _env, build  # noqa: E402

NATIVE = ROOT / "src" / "komorebi_native"
KERNEL = NATIVE / "x280" / "kernel"
INCLUDES = [NATIVE / "include", KERNEL]
IMAGE = "km-dispatch.bin"
VERIFY = "km-dispatch-verify.bin"
MEMCOST = "km-memcost.bin"
MODULE = "_km_dispatch_test.abi3.so"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, default=_env.metal_home())
    parser.add_argument("--out", type=Path, default=HERE / "build")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--define", action="append", default=[], help="image define")
    args = parser.parse_args()
    if args.tree is None:
        sys.exit(f"build.py: pass --tree or set {_env.HOME_VARIABLE}")
    args.out.mkdir(parents=True, exist_ok=True)
    image = args.out / IMAGE
    build.build_image(
        args.tree, image, [KERNEL / "main.c"], includes=INCLUDES, defines=args.define
    )
    print(f"{image}: {image.stat().st_size} bytes")
    defines = [*args.define, "KMD_VERIFY_LINES=1"]
    sources = [KERNEL / "main.c"]
    build.build_image(
        args.tree, args.out / VERIFY, sources, includes=INCLUDES, defines=defines
    )
    build.build_image(args.tree, args.out / MEMCOST, [HERE / "x280" / "memcost.c"])
    if not args.image_only:
        module = args.out / MODULE
        build.build_cxx(
            "pymodule",
            args.tree,
            module,
            HERE / "host" / "dispatch_test.cpp",
            includes=[str(path) for path in INCLUDES],
        )
        print(module)


if __name__ == "__main__":
    main()
