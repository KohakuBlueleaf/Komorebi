"""Out-of-tree builds against a configured tt-metal tree.

Host and Python-module builds borrow flags from targets that tt-x280-dispatch adds to
a tt-metal build (ttx_bench, _ttx280); this ends with Komorebi's own build files.
"""

import re
import shlex
import subprocess
from pathlib import Path

from komorebi import _env


class BuildError(RuntimeError):
    """A build could not be set up or a build step failed."""


# Two targets of the tree's build.ninja (a test host, a nanobind module) supply the
# flags. Each entry: compile statement prefix, link statement prefix, link flags.
TEMPLATES = {
    "host": (
        "build tt_metal/fabric/experimental/x280/CMakeFiles/ttx_bench.dir/",
        "build test/tt_metal/x280/ttx_bench:",
        ["-O3", "-rdynamic"],
    ),
    "pymodule": (
        "build CMakeFiles/_ttx280.dir/",
        "build ttx280/_ttx280.abi3.so:",
        ["-O3", "-shared", "-Wl,--gc-sections", "-rdynamic"],
    ),
}

# A define of the borrowed target that points into a source directory: not carried over.
_DROPPED_DEFINE = re.compile(r"-DTTX_KERNEL_DIR=\S+")

_X280_TARGET = ["-mcpu=sifive-x280", "-march=rv64gc", "-mabi=lp64d"]


def ninja_block(build_dir, header_prefix):
    """The variables of the first build.ninja statement starting with header_prefix."""
    path = Path(build_dir) / "build.ninja"
    if not path.is_file():
        raise BuildError(f"{path} not found: the tree is not built")
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(header_prefix):
            block = {}
            for var in lines[i + 1 :]:
                if not var.startswith("  "):
                    break
                key, _, value = var.partition("=")
                block[key.strip()] = value.strip()
            return block
    raise BuildError(f"no build statement starting with {header_prefix!r}")


def required(block, name, header_prefix):
    """The variable `name` of a build statement; a BuildError names the statement."""
    if name not in block:
        raise BuildError(f"build statement {header_prefix!r} has no {name}")
    return block[name]


def cxx_commands(kind, tree, out, source, includes=(), defines=()):
    """The compile and link commands of one C++ host or Python module, as lists."""
    build_dir = Path(tree) / "build"
    compile_prefix, link_prefix, link_flags = TEMPLATES[kind]
    compile_vars = ninja_block(build_dir, compile_prefix)
    link_vars = ninja_block(build_dir, link_prefix)
    out = Path(out)
    obj = out.with_suffix(".o")
    compile_cmd = (
        ["clang++-20"]
        + shlex.split(_DROPPED_DEFINE.sub("", compile_vars.get("DEFINES", "")))
        + [f"-D{d}" for d in defines]
        + [f"-I{i}" for i in includes]
        + shlex.split(compile_vars.get("INCLUDES", ""))
        + shlex.split(required(compile_vars, "FLAGS", compile_prefix))
        + ["-c", str(source), "-o", str(obj)]
    )

    # Library names in the link statement are relative to the build directory.
    libs = [
        str(build_dir / tok) if not tok.startswith(("-", "/")) else tok
        for tok in shlex.split(required(link_vars, "LINK_LIBRARIES", link_prefix))
    ]
    link_cmd = ["clang++-20", *link_flags, str(obj), "-o", str(out), *libs]
    return compile_cmd, link_cmd


def image_commands(tree, out, sources):
    """The commands that build one X280 image: compile each source, link, objcopy."""
    gcc = _env.sfpi_bin(tree) / "riscv-tt-elf-gcc"
    objcopy = _env.sfpi_bin(tree) / "riscv-tt-elf-objcopy"
    include = _env.x280_source_dir(tree) / "include"
    cflags = _X280_TARGET + [
        "-nostdlib", "-nostartfiles", "-ffreestanding", "-O2", "-g", "-Wall",
        "-Werror", "-mcmodel=medany", f"-I{include}",
    ]  # fmt: skip
    out = Path(out)
    commands, objects = [], []
    for src in sources:
        obj = out.parent / (Path(src).stem + ".x280.o")
        commands.append([str(gcc), *cflags, "-c", str(src), "-o", str(obj)])
        objects.append(str(obj))
    elf = out.with_suffix(".elf")
    commands.append(
        [
            str(gcc),
            *_X280_TARGET,
            "-nostdlib",
            "-Wl,--defsym=X280_REQUIRE_NO_BSS=1",
            "-Wl,--no-warn-rwx-segments",
            "-T",
            str(_env.x280_linker_script(tree)),
            "-o",
            str(elf),
            *map(str, _env.x280_objects(tree)),
            *objects,
        ]
    )
    commands.append([str(objcopy), "-O", "binary", str(elf), str(out)])
    return commands


def run(commands):
    """Runs each command in order; a failing one raises BuildError."""
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True)
        except (OSError, subprocess.CalledProcessError) as err:
            raise BuildError(f"{shlex.join(cmd)}: {err}") from err


def build_cxx(kind, tree, out, source, includes=(), defines=()):
    """Compiles and links one C++ host ("host") or nanobind module ("pymodule")."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run(cxx_commands(kind, tree, out, source, includes, defines))


def build_image(tree, out, sources):
    """Compiles and links one X280 image into `out` (a .bin; the .elf is beside it)."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run(image_commands(tree, out, sources))
