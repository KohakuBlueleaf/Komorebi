"""Builds the kernel images (also the KMD_VERIFY_LINES one and the measuring image) and
the host module against a tt-metal tree (--tree, default TT_METAL_HOME):

    python build.py [--tree TT_METAL_TREE] [--out DIR] [--image-only] [--define N=V]
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
    defines = [*args.define, "KMD_VERIFY_LINES=1", "KMD_TRACE=1"]
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
