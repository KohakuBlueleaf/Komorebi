"""Out-of-tree builds against a configured tt-metal tree.

Host and Python-module builds take their flags from targets that tt-metal itself defines
in a tree configured with -DTT_METAL_BUILD_X280=ON; images use the X280 runtime's files.
"""

import shlex
import subprocess
from pathlib import Path

from komorebi import _env


class BuildError(RuntimeError):
    """A build could not be set up or a build step failed."""


# tt-metal's own build.ninja statements: the X280 runtime's library (compile flags, with
# tt_metal's), its priming test (link libraries), ttnn's nanobind library and archive.
X280_LIBRARY = "build tt_metal/fabric/experimental/x280/CMakeFiles/x280_device.dir/"
X280_LINK = "build test/tt_metal/x280/x280_prime_test:"
NANOBIND = (
    "build ttnn/CMakeFiles/nanobind-static-abi3.dir/",
    "ttnn/libnanobind-static-abi3.a",
)
# The option that adds the X280 runtime; tt-metal accepts it only with its tests on.
X280_OPTION = "-DTT_METAL_BUILD_X280=ON (with -DTT_METAL_BUILD_TESTS=ON)"
# The warning tt_metal/CMakeLists.txt adds for tt-metal's own sources: not carried over.
_OWN_WARNINGS = {"-Wunused-parameter"}

# Each kind: compile flags after the tree's, link flags, whether it links nanobind.
TEMPLATES = {
    "host": ([], ["-O3", "-rdynamic"], False),
    "pymodule": (
        ["-Os", "-ffunction-sections", "-fdata-sections"],
        ["-O3", "-shared", "-Wl,--gc-sections", "-rdynamic"],
        True,
    ),
}

_X280_TARGET = ["-mcpu=sifive-x280", "-march=rv64gc", "-mabi=lp64d"]


def ninja_block(build_directory, header_prefix, missing=None):
    """The variables of the first build.ninja statement starting with header_prefix;
    `missing` is the error when there is none."""
    path = Path(build_directory) / "build.ninja"
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
    raise BuildError(missing or f"no build statement starting with {header_prefix!r}")


def required(block, name, header_prefix):
    """The variable `name` of a build statement; a BuildError names the statement."""
    if name not in block:
        raise BuildError(f"build statement {header_prefix!r} has no {name}")
    return block[name]


def nanobind_flags(tree):
    """nanobind's include directories, as system ones (as ttnn/CMakeLists.txt gives
    them to its users), and its archive."""
    build_directory = Path(tree) / "build"
    missing = f"{tree} has no nanobind library: the tree is configured without ttnn"
    block = ninja_block(build_directory, NANOBIND[0], missing)
    includes = []
    for token in shlex.split(block.get("INCLUDES", "")):
        if token.startswith("-I"):
            includes += ["-isystem", token[2:]]
    return includes, [str(build_directory / NANOBIND[1])]


def cxx_commands(kind, tree, out, source, includes=(), defines=()):
    """The compile and link commands of one C++ host or Python module, as lists."""
    build_directory = Path(tree) / "build"
    compile_flags, link_flags, nanobind = TEMPLATES[kind]
    missing = f"{tree} has no X280 runtime: configure it with {X280_OPTION}"
    compile_vars = ninja_block(build_directory, X280_LIBRARY, missing)
    link_vars = ninja_block(build_directory, X280_LINK, missing)
    system, archives = nanobind_flags(tree) if nanobind else ([], [])
    tree_flags = shlex.split(required(compile_vars, "FLAGS", X280_LIBRARY))
    out = Path(out)
    object_file = out.with_suffix(".o")
    compile_command = (
        ["clang++-20"]
        + shlex.split(compile_vars.get("DEFINES", ""))
        # Where the tree's X280 images are, as tt-metal gives its own X280 tests.
        + [f'-DX280_IMAGE_DIR="{_env.x280_image_directory(tree)}"']
        + [f"-D{define}" for define in defines]
        + [f"-I{include_path}" for include_path in includes]
        + shlex.split(compile_vars.get("INCLUDES", ""))
        + system
        + [flag for flag in tree_flags if flag not in _OWN_WARNINGS]
        + compile_flags
        + ["-c", str(source), "-o", str(object_file)]
    )

    # Library names in the link statement are relative to the build directory.
    libraries = [
        str(build_directory / token) if not token.startswith(("-", "/")) else token
        for token in shlex.split(required(link_vars, "LINK_LIBRARIES", X280_LINK))
    ]
    link_command = [
        "clang++-20",
        *link_flags,
        str(object_file),
        "-o",
        str(out),
        *archives,
        *libraries,
    ]
    return compile_command, link_command


def image_commands(tree, out, sources, includes=(), defines=()):
    """The commands that build one X280 image: compile each source, link, objcopy.

    `includes` and `defines` go to every compile, ahead of the runtime's include
    directory.
    """
    gcc = _env.sfpi_bin(tree) / "riscv-tt-elf-gcc"
    objcopy = _env.sfpi_bin(tree) / "riscv-tt-elf-objcopy"
    include = _env.x280_source_directory(tree) / "include"
    cflags = _X280_TARGET + [
        "-nostdlib", "-nostartfiles", "-ffreestanding", "-O2", "-g", "-Wall",
        "-Werror", "-mcmodel=medany",
    ]  # fmt: skip
    cflags += [f"-D{define}" for define in defines] + [
        f"-I{include_path}" for include_path in includes
    ]
    cflags.append(f"-I{include}")
    out = Path(out)
    commands, objects = [], []
    for source in sources:
        object_file = out.parent / (Path(source).stem + ".x280.o")
        commands.append([str(gcc), *cflags, "-c", str(source), "-o", str(object_file)])
        objects.append(str(object_file))
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
    for command in commands:
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise BuildError(f"{shlex.join(command)}: {error}") from error


def build_cxx(kind, tree, out, source, includes=(), defines=()):
    """Compiles and links one C++ host ("host") or nanobind module ("pymodule")."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run(cxx_commands(kind, tree, out, source, includes, defines))


def build_image(tree, out, sources, includes=(), defines=()):
    """Compiles and links one X280 image into `out` (a .bin; the .elf is beside it)."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run(image_commands(tree, out, sources, includes, defines))
