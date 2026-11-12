"""Where the tt-metal tree, its build and the X280 toolchain are.

Every function only looks at the file system; nothing runs at import.
"""

import os
from pathlib import Path

HOME_VARIABLE = "TT_METAL_HOME"

_X280_SOURCE = Path("tt_metal/fabric/experimental/x280")
_SFPI_BIN = Path("runtime/sfpi/compiler/bin")
_X280_OBJECTS = ("entry_active.o", "x280_active_start.o")


def package_dir():
    """The directory of the installed `komorebi` package."""
    return Path(__file__).resolve().parent


def include_dir():
    """The native include directory beside the package in a checkout, else None."""
    path = package_dir().parent / "komorebi_native" / "include"
    return path if path.is_dir() else None


def metal_home(environ=None):
    """The tt-metal source tree named by TT_METAL_HOME, or None if unset or absent."""
    value = (os.environ if environ is None else environ).get(HOME_VARIABLE)
    if not value:
        return None
    tree = Path(value).expanduser()
    return tree if tree.is_dir() else None


def build_dir(tree):
    """`tree`'s build directory (the one holding build.ninja), or None."""
    path = Path(tree) / "build"
    return path if (path / "build.ninja").is_file() else None


def x280_source_dir(tree):
    """The directory of tt-metal's X280 runtime sources inside `tree`."""
    return Path(tree) / _X280_SOURCE


def x280_linker_script(tree):
    """The linker script of X280 images."""
    return x280_source_dir(tree) / "ld" / "x280.ld"


def x280_objects(tree):
    """The runtime objects every X280 image is linked with, in link order."""
    obj_dir = Path(tree) / "build" / "x280" / "obj"
    return [obj_dir / name for name in _X280_OBJECTS]


def sfpi_bin(tree):
    """The directory of the RISC-V compiler tt-metal installed under `tree`."""
    return Path(tree) / _SFPI_BIN


def x280_compiler(tree):
    """The compiler that builds X280 images, or None if it is not installed."""
    path = sfpi_bin(tree) / "riscv-tt-elf-gcc"
    return path if path.is_file() else None


def has_x280_runtime(tree):
    """True when `tree` was built with the X280 runtime (sources, objects, script)."""
    needed = [x280_linker_script(tree), *x280_objects(tree)]
    return all(path.is_file() for path in needed)
