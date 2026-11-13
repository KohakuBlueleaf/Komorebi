"""Where the tt-metal tree, its build and the X280 toolchain are.

The tree is TT_METAL_HOME's when set, else the root of the komorebi-runtime package when
it is installed. Nothing runs at import.
"""

import os
from pathlib import Path

try:
    import komorebi_runtime
except ImportError:  # the runtime package is optional (the `runtime` extra)
    komorebi_runtime = None

HOME_VARIABLE = "TT_METAL_HOME"
RUNTIME_SOURCE = "komorebi-runtime"

_X280_SOURCE = Path("tt_metal/fabric/experimental/x280")
_SFPI_BIN = Path("runtime/sfpi/compiler/bin")
_X280_OBJECTS = ("entry_active.o", "x280_active_start.o")
_X280_IDLE_IMAGE = Path("build/x280/firmware-lim-idle.bin")


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


def find_tree(environ=None):
    """(tree, source): TT_METAL_HOME's tree when the variable is set (None if it names
    no directory), else the runtime package's root, else (None, None)."""
    if (os.environ if environ is None else environ).get(HOME_VARIABLE):
        return metal_home(environ), HOME_VARIABLE
    if komorebi_runtime is not None and komorebi_runtime.installed():
        return komorebi_runtime.root(environ), RUNTIME_SOURCE
    return None, None


def activate(environ=None):
    """When the tree is the runtime package's, prepares this process to run on it
    (TT_METAL_RUNTIME_ROOT, ttnn on the path); returns the tree's source."""
    tree, source = find_tree(environ)
    if source == RUNTIME_SOURCE:
        komorebi_runtime.activate(environ)
    return source


def runtime_version():
    """The installed runtime package's version, or None."""
    if komorebi_runtime is None or not komorebi_runtime.installed():
        return None
    return komorebi_runtime.version()


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


def x280_idle_image(tree):
    """The image the X280 tiles are booted to before an active image runs."""
    return Path(tree) / _X280_IDLE_IMAGE


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
