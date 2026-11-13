"""The tt-metal build that Komorebi and Kohaku-TT share: where its libraries, its kernel
runtime tree, its X280 runtime pieces and its RISC-V compiler are, and its pin.

Nothing runs at import; `activate()` is the only call that changes the process.
"""

import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
CACHE_VARIABLE = "KMR_CACHE"
ROOT_VARIABLE = "TT_METAL_RUNTIME_ROOT"
MANIFEST = "manifest.json"

# Inside the package: the libraries, the kernel runtime tree (laid out as in a tt-metal
# tree), and, for now, tt-metal's ttnn Python package.
LIB = "lib"
TREE = "tree"
PYTHON = "python"

# The X280 pieces, at their paths inside a tt-metal tree and its build.
X280_SOURCE = Path("tt_metal/fabric/experimental/x280")
X280_OBJECTS = ("entry_active.o", "x280_active_start.o")
X280_OBJECT_DIR = Path("build/x280/obj")
SFPI_VERSION = Path("tt_metal/sfpi-version")
SFPI_BIN = Path("compiler/bin")


class RuntimeMissing(RuntimeError):
    """The package holds no built runtime (a source checkout, or a damaged install)."""


class X280(NamedTuple):
    """The pieces every X280 image is built with."""

    objects: list
    linker_script: Path
    include: Path


def package_dir(base=None):
    """The directory of the installed package (or `base`, for a layout elsewhere)."""
    return Path(base) if base is not None else HERE


def manifest(base=None):
    """The build's manifest: pin, patches, build number, version, file count."""
    path = package_dir(base) / MANIFEST
    if not path.is_file():
        raise RuntimeMissing(f"{path} not found: this package holds no built runtime")
    return json.loads(path.read_text())


def installed(base=None):
    """True when the package holds a built runtime."""
    return (package_dir(base) / MANIFEST).is_file()


def pin(base=None):
    """The tt-metal release this runtime is built from, e.g. "v0.81.1"."""
    return manifest(base)["pin"]


def version(base=None):
    """The package version: the pin plus our build number, e.g. "0.81.1.1"."""
    return manifest(base)["version"]


def tree(base=None):
    """The kernel runtime tree inside the package (read only)."""
    return package_dir(base) / TREE


def lib_dir(base=None):
    """The directory of the tt-metal libraries."""
    return package_dir(base) / LIB


def python_dir(base=None):
    """The directory holding tt-metal's ttnn package (interim, see `activate`)."""
    return package_dir(base) / PYTHON


def x280(base=None):
    """The X280 runtime objects (in link order), the linker script and the include
    directory."""
    source = tree(base) / X280_SOURCE
    objects = [tree(base) / X280_OBJECT_DIR / name for name in X280_OBJECTS]
    return X280(objects, source / "ld" / "x280.ld", source / "include")


def cache_dir(environ=None):
    """KMR_CACHE if set, else the user's cache directory (XDG) under komorebi."""
    environ = os.environ if environ is None else environ
    if environ.get(CACHE_VARIABLE):
        return Path(environ[CACHE_VARIABLE]).expanduser()
    base = environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "komorebi"


def sfpi_release(base=None):
    """The sfpi release the tree names, as a dict of its sfpi-version file."""
    text = (tree(base) / SFPI_VERSION).read_text()
    pairs = (line.partition("=") for line in text.splitlines())
    return {k.removeprefix("sfpi_"): v.strip("'") for k, _, v in pairs if k and v}


def sfpi_dir(environ=None, base=None):
    """Where `python -m komorebi_runtime sfpi` puts the compiler (it may be absent)."""
    return cache_dir(environ) / f"sfpi-{sfpi_release(base)['version']}" / "sfpi"


def sfpi_bin(environ=None, base=None):
    """The fetched RISC-V compiler's bin directory, or None before the fetch."""
    path = sfpi_dir(environ, base) / SFPI_BIN
    return path if (path / "riscv-tt-elf-gcc").is_file() else None


def root(environ=None, base=None):
    """The runtime root to run and build against: a directory in the cache whose
    entries link to the package's tree, plus runtime/sfpi once the compiler is fetched.

    It has a tt-metal tree's layout, so it serves as TT_METAL_RUNTIME_ROOT.
    """
    out = cache_dir(environ) / f"runtime-{version(base)}"
    source = tree(base)
    for entry in source.iterdir():
        if entry.name != "runtime":
            _link(out / entry.name, entry)
    for entry in (source / "runtime").iterdir():
        _link(out / "runtime" / entry.name, entry)
    if sfpi_bin(environ, base) is not None:
        _link(out / "runtime" / "sfpi", sfpi_dir(environ, base))
    return out


def activate(environ=None, base=None):
    """Prepares this process to run tt-metal from the package: TT_METAL_RUNTIME_ROOT
    names `root()` unless already set, and ttnn (interim) becomes importable."""
    environ = os.environ if environ is None else environ
    out = root(environ, base)
    environ.setdefault(ROOT_VARIABLE, str(out))
    path = str(python_dir(base))
    if path not in sys.path:
        sys.path.insert(0, path)
    return out


def _link(link, target):
    """Makes `link` a symlink to `target`, replacing a link that points elsewhere."""
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except FileExistsError:
        # Another process made it at the same moment; it must point at the same place.
        if link.resolve() != target.resolve():
            raise
