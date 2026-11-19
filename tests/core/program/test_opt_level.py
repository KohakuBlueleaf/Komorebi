"""A program's optimisation level per kernel: checked, passed to the kernel's build."""

import re
import subprocess

import pytest
from fake_session import load_session

from komorebi import _env
from komorebi.device import native
from komorebi.errors import ProgramError
from komorebi.program.description import OPT_LEVELS, Program
from komorebi.session import kernels as card_kernels

# Three callers of one static function: inlined at the default level, called at Os.
SOURCE = """
static unsigned step(unsigned x, unsigned k) {
    for (unsigned i = 0; i < 4; i++) { x ^= x << 7; x += k * i; x ^= x >> 9; }
    return x;
}
unsigned a(unsigned x) { return step(x, 3); }
unsigned b(unsigned x) { return step(x, 5) + step(x + 1, 7); }
unsigned c(unsigned x) { return step(step(x, 11), 13); }
"""


def kernels(tmp_path):
    for kernel in ("reader", "compute", "writer"):
        (tmp_path / f"{kernel}.cpp").write_text("")
    return tmp_path


def test_the_level_is_checked_by_kernel_and_name(tmp_path):
    for level in OPT_LEVELS:
        Program("p", kernels=kernels(tmp_path), opt={"reader": level})
    with pytest.raises(ProgramError, match="unknown kernel 'trisc0'"):
        Program("p", kernels=tmp_path, opt={"trisc0": "Os"})
    with pytest.raises(ProgramError, match="'-Os'"):
        Program("p", kernels=tmp_path, opt={"writer": "-Os"})


def test_a_set_level_reaches_the_descriptor_and_the_default_leaves_it(
    monkeypatch, tmp_path
):
    load_session(monkeypatch, tmp_path)
    program = Program("p", kernels=kernels(tmp_path), opt={"reader": "Os"})
    defines = [("N", "1")]
    build = card_kernels.kernel_build
    assert build(program, "writer", defines, "blackhole") == {"defines": defines}
    assert build(program, "reader", defines, "blackhole") == {
        "defines": [("N", "1"), (card_kernels.OPT_DEFINE, "Os")],  # a build key apart
        "opt_level": "Os",
    }


def test_wormhole_builds_the_writer_for_size_unless_the_program_says(tmp_path):
    """Wormhole's second data-movement CPU holds 16 KiB of code: its writer is built
    at Os with its waits out of line (chips.Chip.writer_opt, writer_defines); at a
    writer level above 0 both movers take KM_WRITER_COLD (the reader loads the part
    the writer calls); a program's own level still wins."""
    plain = Program("p", kernels=kernels(tmp_path))
    build = card_kernels.kernel_build
    assert build(plain, "writer", [], "wormhole_b0") == {
        "defines": [("KM_WAIT_SHARED", "1"), (card_kernels.OPT_DEFINE, "Os")],
        "opt_level": "Os",
    }
    assert build(plain, "reader", [], "wormhole_b0") == {"defines": []}
    plain.writer_level = 2
    cold = ("KM_WRITER_COLD", "2")
    assert build(plain, "writer", [], "wormhole_b0") == {
        "defines": [cold, ("KM_WAIT_SHARED", "1"), (card_kernels.OPT_DEFINE, "Os")],
        "opt_level": "Os",
    }
    assert build(plain, "reader", [], "wormhole_b0") == {"defines": [cold]}
    assert build(plain, "compute", [], "wormhole_b0") == {
        "defines": card_kernels.compute_defines("wormhole_b0")
    }  # no writer level
    own = Program("q", kernels=kernels(tmp_path), opt={"movement": "O2"})
    assert build(own, "writer", [], "wormhole_b0")["opt_level"] == "O2"


def test_the_feeder_harness_builds_with_the_sessions_level_and_island_defines():
    """The feeder's kernels (komorebi.tools.feeder) take these two, as a load does."""
    assert card_kernels.level_build([], None) == {"defines": []}
    assert card_kernels.level_build([], "O3") == {
        "defines": [(card_kernels.OPT_DEFINE, "O3")],
        "opt_level": "O3",
    }
    assert card_kernels.one_island(None, None) == [("KM_ONE_ISLAND", "1")]
    assert card_kernels.one_island(("graph", 0), None) == []
    assert card_kernels.one_island(None, ("split",)) == []


@pytest.mark.tree
def test_the_names_are_the_runtimes():
    try:
        tree = native.metal_tree()
    except Exception:  # noqa: BLE001 (no tree: the test is skipped)
        pytest.skip("needs TT_METAL_HOME")
    header = tree / "tt_metal" / "api" / "tt-metalium" / "kernel_types.hpp"
    body = re.search(
        r"enum class KernelBuildOptLevel[^{]*{([^}]*)}", header.read_text()
    )
    names = re.findall(r"^\s*(\w+),", body.group(1), re.M)
    assert sorted(names) == sorted(OPT_LEVELS)


def text_bytes(tmp_path, level):
    try:
        tree = native.metal_tree()
    except Exception:  # noqa: BLE001 (no tree: the test is skipped)
        pytest.skip("needs TT_METAL_HOME")
    gcc = _env.sfpi_bin(tree) / "riscv-tt-elf-gcc"
    size = _env.sfpi_bin(tree) / "riscv-tt-elf-size"
    if not (gcc.is_file() and size.is_file()):
        pytest.skip(f"needs {gcc.name} and {size.name}")
    source_file, object_file = tmp_path / "kernel.c", tmp_path / f"kernel-{level}.o"
    source_file.write_text(SOURCE)
    flags = ["-mcpu=tt-bh", "-mabi=ilp32", f"-{level}"]
    subprocess.run(
        [str(gcc), *flags, "-c", str(source_file), "-o", str(object_file)], check=True
    )
    out = subprocess.run(
        [str(size), str(object_file)], capture_output=True, text=True, check=True
    )
    return int(out.stdout.splitlines()[1].split()[0])


@pytest.mark.tree
def test_os_gives_a_smaller_text_than_the_data_movement_default(tmp_path):
    assert text_bytes(tmp_path, "Os") < text_bytes(tmp_path, "O2")
