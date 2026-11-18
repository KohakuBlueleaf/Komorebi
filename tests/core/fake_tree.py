"""A small fake tt-metal tree on disk, for the doctor and build tests."""

from komorebi import _env
from komorebi.builder import flags as build

NINJA_COMPILE = """{head}a.o: CXX_COMPILER: /src/a.cpp
  DEFINES = -DFOO=1 -DBAR="two words"
  FLAGS = -O2 -std=c++20 -Wunused-parameter -fPIC
  INCLUDES = -I/a/include -isystem /b/include
  OBJECT_FILE_DIR = ignored
"""
NINJA_LINK = """{head} CXX_LINKER: a.o
  LINK_LIBRARIES = lib/libone.so -lpthread /usr/lib/libz.so  lib/libtwo.a
"""
NINJA_NANOBIND = """{head}nb.o: CXX_COMPILER: /nb/src/nb.cpp
  INCLUDES = -I/nb/include -I/py/include -isystem /nb/ext/robin_map/include
"""


def build_ninja(x280=True, nanobind=True):
    """The text of a build.ninja with the statements `cxx_commands` reads."""
    parts = ["rule x\n  command = true\n"]
    if x280:
        parts.append(NINJA_COMPILE.format(head=build.X280_LIBRARY))
        parts.append(NINJA_LINK.format(head=build.X280_LINK))
    if nanobind:
        parts.append(NINJA_NANOBIND.format(head=build.NANOBIND[0]))
    return "\n".join(parts)


def make_tree(root, x280=True, compiler=True, ninja=True):
    """A tree under `root`; the flags choose which parts of it exist."""
    root.mkdir(parents=True, exist_ok=True)
    if ninja:
        (root / "build").mkdir()
        (root / "build" / "build.ninja").write_text(build_ninja())
    if x280:
        files = [_env.x280_linker_script(root), *_env.x280_objects(root)]
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
    if compiler:
        gcc = _env.sfpi_bin(root) / "riscv-tt-elf-gcc"
        gcc.parent.mkdir(parents=True)
        gcc.write_text("")
    return root
