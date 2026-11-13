"""The dispatcher image and the native host module: found prebuilt beside the
package, else built once against the tt-metal tree into a cache directory.

    python -m komorebi.native [--tree TREE]    # build both now, print their paths
"""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

from komorebi import _env, build

IMAGE = "km-dispatch.bin"
MODULE = "_km_host"
MODULE_FILE = f"{MODULE}.abi3.so"
CACHE_VARIABLE = "KMR_CACHE"
PLACEHOLDER = Path("/OUT")  # the output directory in the commands that are hashed


def native_directory():
    """The native layer's source directory beside the package in a checkout, or None."""
    path = _env.package_directory().parent / "komorebi_native"
    return path if path.is_dir() else None


def kernel_directory():
    """The dispatcher's sources (and its host interface, km_dispatch.h)."""
    return _require_native() / "x280" / "kernel"


def module_source():
    """The native host module's source."""
    return _require_native() / "csrc" / "host" / "km_host.cpp"


def engines_directory():
    """The engines' plain-C sources, compiled into the dispatcher image."""
    return _require_native() / "x280" / "engines"


def kmx_directory():
    """The kmx library: start-up, kmx.h and the layout it shares with the host."""
    return _require_native() / "x280" / "kmx"


def includes():
    """The include directories of the image and the module: the ABI, the kernel, the
    engines."""
    return [_require_native() / "include", kernel_directory(), engines_directory()]


def cache_directory(environ=None):
    """KMR_CACHE if set, else build/komorebi in a checkout, else ~/.cache/komorebi."""
    value = (os.environ if environ is None else environ).get(CACHE_VARIABLE)
    if value:
        return Path(value).expanduser()
    checkout = _env.package_directory().parents[1]
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


def inputs_key(commands, sources):
    """A short hash of a build's inputs: its commands (compilers, flags, libraries),
    the size and time of every file they name, and the sources' contents."""
    digest = hashlib.sha256()
    for cmd in commands:
        for arg in cmd:
            digest.update(arg.encode() + b"\0")
            named = shutil.which(arg) if "/" not in arg else arg
            if named and Path(named).is_file():
                info = Path(named).stat()
                digest.update(f"{info.st_size}:{info.st_mtime_ns}".encode())
    digest.update(source_key(sources, "/").encode())
    return digest.hexdigest()[:16]


def build_whole(out, make):
    """Builds `out` with make(path) into a temporary directory beside it, then moves
    it into place: an interrupted or concurrent build never leaves a partial file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{out.parent.name}-", dir=out.parent.parent))
    try:
        make(work / out.name)
        os.replace(work / out.name, out)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _require_native():
    found = native_directory()
    if found is None:
        raise build.BuildError("the native sources are not beside the package")
    return found


def _tree(tree):
    tree = tree or _env.find_tree()[0]
    if tree is None:
        raise build.BuildError(
            f"set {_env.HOME_VARIABLE}, install {_env.RUNTIME_SOURCE} or pass a tree"
        )
    return Path(tree)


def metal_tree(tree=None):
    """`tree`, else TT_METAL_HOME's tree, else the runtime package's root (a
    BuildError if none)."""
    return _tree(tree)


def image_path(tree=None, defines=()):
    """The dispatcher image: prebuilt beside the package, else built in the cache.

    `defines` always builds (a variant image is never shipped prebuilt).
    """
    prebuilt = _env.package_directory() / IMAGE
    if prebuilt.is_file() and not defines:
        return prebuilt
    tree = _tree(tree)
    sources = [kernel_directory() / "main.c", engines_directory() / "km_graph_engine.c"]
    defines = list(defines)
    commands = build.image_commands(
        tree, PLACEHOLDER / IMAGE, sources, includes(), defines
    )
    out = cache_directory() / f"image-{inputs_key(commands, includes())}" / IMAGE
    if not out.is_file():
        build_whole(
            out,
            lambda path: build.build_image(
                tree, path, sources, includes=includes(), defines=defines
            ),
        )
    return out


def thread_image(thread, tree=None):
    """The image of a program's X280 thread (`komorebi.thread.Thread`): its sources
    linked with the kmx library, built once into the cache."""
    tree = _tree(tree)
    sources = [*thread.source_paths(), kmx_directory() / "kmx.c"]
    extra = [Path(i) for i in thread.includes]
    defines = [f"{k}={v}" for k, v in sorted(thread.defines.items())]
    key = source_key([*sources, kmx_directory(), *includes(), *extra], tree)
    key += "-" + hashlib.sha256("\0".join(defines).encode()).hexdigest()[:8]
    out = cache_directory() / f"thread-{key}" / "thread.bin"
    if not out.is_file():
        build.build_image(
            tree,
            out,
            sources,
            includes=[kmx_directory(), *includes(), *extra],
            defines=defines,
        )
    return out


def module_path(tree=None):
    """The native host module: prebuilt beside the package when the tree is the
    runtime package's (it links that runtime), else built in the cache."""
    prebuilt = _env.package_directory() / MODULE_FILE
    runtime = tree is None and _env.find_tree()[1] == _env.RUNTIME_SOURCE
    if prebuilt.is_file() and runtime:
        return prebuilt
    tree = _tree(tree)
    paths = [str(p) for p in includes()]
    commands = build.cxx_commands(
        "pymodule", tree, PLACEHOLDER / MODULE_FILE, module_source(), paths
    )
    key = inputs_key(commands, [*includes(), module_source()])
    out = cache_directory() / f"module-{key}" / MODULE_FILE
    if not out.is_file():
        build_whole(
            out,
            lambda path: build.build_cxx(
                "pymodule", tree, path, module_source(), includes=paths
            ),
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
    module.set_idle_image(str(_env.x280_idle_image(_tree(tree))))
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
