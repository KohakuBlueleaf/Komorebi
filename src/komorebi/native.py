"""The dispatch kernel image and the native host module: found prebuilt beside the
package, else built once against the tt-metal tree into a cache directory.

    python -m komorebi.native [--tree TREE]    # build both now, print their paths
"""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

from komorebi import _env, build

IMAGE = "km-dispatch.bin"
MODULE = "_km_host"
MODULE_FILE = f"{MODULE}.abi3.so"
CACHE_VARIABLE = "KMR_CACHE"


def native_dir():
    """The native layer's source directory beside the package in a checkout, or None."""
    path = _env.package_dir().parent / "komorebi_native"
    return path if path.is_dir() else None


def kernel_dir():
    """The dispatch kernel's sources (and its host interface, km_dispatch.h)."""
    return _require_native() / "x280" / "kernel"


def module_source():
    """The native host module's source."""
    return _require_native() / "csrc" / "host" / "km_host.cpp"


def includes():
    """The include directories of the image and the module: the ABI, the kernel."""
    return [_require_native() / "include", kernel_dir()]


def cache_dir(environ=None):
    """KMR_CACHE if set, else build/komorebi in a checkout, else ~/.cache/komorebi."""
    value = (os.environ if environ is None else environ).get(CACHE_VARIABLE)
    if value:
        return Path(value).expanduser()
    checkout = _env.package_dir().parents[1]
    if (checkout / "pyproject.toml").is_file():
        return checkout / "build" / "komorebi"
    return Path.home() / ".cache" / "komorebi"


def source_key(paths, tree):
    """A short hash of the files under `paths` (sorted) and of the tree's path."""
    digest = hashlib.sha256(str(Path(tree).resolve()).encode())
    files = []
    for path in map(Path, paths):
        files += sorted(path.rglob("*")) if path.is_dir() else [path]
    for path in files:
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _require_native():
    found = native_dir()
    if found is None:
        raise build.BuildError("the native sources are not beside the package")
    return found


def _tree(tree):
    tree = tree or _env.metal_home()
    if tree is None:
        raise build.BuildError(f"set {_env.HOME_VARIABLE} or pass a tt-metal tree")
    return Path(tree)


def image_path(tree=None, defines=()):
    """The dispatch kernel image: prebuilt beside the package, else built in the cache.

    `defines` always builds (a variant image is never shipped prebuilt).
    """
    prebuilt = _env.package_dir() / IMAGE
    if prebuilt.is_file() and not defines:
        return prebuilt
    tree = _tree(tree)
    key = source_key(includes(), tree)
    if defines:
        key += "-" + hashlib.sha256("\0".join(defines).encode()).hexdigest()[:8]
    out = cache_dir() / f"image-{key}" / IMAGE
    if not out.is_file():
        build.build_image(
            tree,
            out,
            [kernel_dir() / "main.c"],
            includes=includes(),
            defines=list(defines),
        )
    return out


def module_path(tree=None):
    """The native host module: prebuilt beside the package, else built in the cache."""
    prebuilt = _env.package_dir() / MODULE_FILE
    if prebuilt.is_file():
        return prebuilt
    tree = _tree(tree)
    key = source_key([*includes(), module_source()], tree)
    out = cache_dir() / f"module-{key}" / MODULE_FILE
    if not out.is_file():
        build.build_cxx(
            "pymodule",
            tree,
            out,
            module_source(),
            includes=[str(p) for p in includes()],
        )
    return out


def load_module(tree=None):
    """Imports the native host module (building it first if needed)."""
    if MODULE in sys.modules:
        return sys.modules[MODULE]
    path = module_path(tree)
    loader = importlib.machinery.ExtensionFileLoader(MODULE, str(path))
    spec = importlib.util.spec_from_file_location(MODULE, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[MODULE] = module
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, help="tt-metal tree (default: env)")
    args = parser.parse_args(argv)
    try:
        print(image_path(args.tree))
        print(module_path(args.tree))
    except build.BuildError as err:
        print(f"komorebi.native: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
