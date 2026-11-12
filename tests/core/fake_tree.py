"""A small fake tt-metal tree on disk, for the doctor and build tests."""

from komorebi import _env, build

NINJA_COMPILE = """{head} CXX_COMPILER: /src/{kind}.cpp
  DEFINES = -DFOO=1 -DTTX_KERNEL_DIR=/somewhere/kernels -DBAR="two words"
  FLAGS = -O2 -std=c++20 -fPIC
  INCLUDES = -I/a/include -isystem /b/include
  OBJECT_FILE_DIR = ignored
"""
NINJA_LINK = """{head} CXX_LINKER: {kind}.o
  LINK_LIBRARIES = lib/libone.so -lpthread /usr/lib/libz.so  lib/libtwo.a
"""


def build_ninja(include_kinds=("host", "pymodule")):
    """The text of a build.ninja with the statements `build_cxx` borrows from."""
    parts = ["rule x\n  command = true\n"]
    for kind in include_kinds:
        compile_head, link_head, _ = build.TEMPLATES[kind]
        parts.append(NINJA_COMPILE.format(head=compile_head + "a.o:", kind=kind))
        parts.append(NINJA_LINK.format(head=link_head, kind=kind))
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
