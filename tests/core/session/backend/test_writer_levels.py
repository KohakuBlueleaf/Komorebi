"""Wormhole's writer levels: a load builds at the lowest level whose writer fits the
writer CPU's code memory (NCRISC: 16 KiB, chips.Chip.writer_code), from the level kept
for the program, keeps the level found, states the writer's margin and warns at the
highest level under load.WRITER_MARGIN."""

import contextlib
import re
import time
import types
import warnings

import pytest

from komorebi.device import remembered
from komorebi.errors import ProgramError
from komorebi.program.description import KernelSizes
from komorebi.session import kernels
from komorebi.session.parts import load

OVER = (
    "TT_THROW: /c/kernels/movement/1//ncrisc/ncrisc.elf: segment[0] [0xffc00000,"
    "+0x4648) overflows region:0 limit of 0x4000 bytes, reduce the size of code"
)


def logged(arch, ncrisc, level=0):
    """The lines _log_kernel_buffer gives a load whose writer's text is `ncrisc`."""
    lines = []
    session = types.SimpleNamespace(arch=arch, log=lines.append)
    text = {"BRISC": 9000, "NCRISC": ncrisc, "TRISC0": 5000}
    sizes = KernelSizes(text, 30000, 70656)
    program = types.SimpleNamespace(name="p", sizes=sizes, writer_level=level)
    load.LoadMixin._log_kernel_buffer(session, program)
    return lines


def test_wormhole_states_the_writer_margin_and_level():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lines = logged("wormhole_b0", 12540, 1)
    assert (
        lines[-1] == "writer code: 12,540 of 16,384 B, margin 3,844 B, writer level 1"
    )


def test_only_the_highest_level_warns_under_one_kib_of_margin():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        logged("wormhole_b0", 16140, 3)  # a program that grows moves up a level
    with pytest.warns(UserWarning, match=r"p: writer code: 16,140 of 16,384 B, margin"):
        lines = logged("wormhole_b0", 16140, 4)
    assert lines[-1].endswith("margin 244 B, writer level 4")


def test_blackhole_has_no_writer_code_limit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lines = logged("blackhole", 30000)
    assert not any(line.startswith("writer code") for line in lines)


def test_only_a_writer_elf_over_its_code_memory_moves_a_level():
    assert kernels.writer_over(RuntimeError(OVER))
    assert not kernels.writer_over(RuntimeError(OVER.replace("/ncrisc/", "/brisc/")))
    assert not kernels.writer_over(RuntimeError("error: 'x' was not declared"))


class Built:
    """A backend whose writer fits from level `fits` on; `errors` raised otherwise."""

    def __init__(self, fits, error=OVER):
        self.fits, self.error, self.levels = fits, error, []

    def describe(self, program, *_):
        level = kernels.writer_level(program)
        self.levels.append(level)
        fits, error = level >= self.fits, self.error

        def build():
            if not fits:
                raise RuntimeError(error)
            return ["elves", "sizes"]

        return types.SimpleNamespace(build=build)


def build_levels(monkeypatch, backend, kept=0, arch="wormhole_b0"):
    """_build_levels of a program on `arch`, the level kept for it `kept`: (the levels
    described, the level kept after, the log)."""
    saved, lines = {}, []
    monkeypatch.setattr(remembered, "writer_level", lambda program, arch: kept)
    monkeypatch.setattr(
        remembered,
        "keep_writer_level",
        lambda program, arch, level: saved.setdefault("level", level),
    )
    session = types.SimpleNamespace(arch=arch, log=lines.append, _dispatcher=None)
    program = types.SimpleNamespace(name="p")
    load.LoadMixin._build_levels(session, backend, program, None, (program,))
    return backend.levels, saved.get("level"), lines


def test_a_writer_over_its_code_memory_is_built_again_a_level_up(monkeypatch):
    levels, kept, lines = build_levels(monkeypatch, Built(fits=2))
    assert levels == [0, 1, 2] and kept == 2
    assert "built again at writer level 2, from L1: the missed page bound, " in lines[1]


def test_the_seconds_of_a_build_again_leave_out_the_wait_for_the_lock(monkeypatch):
    @contextlib.contextmanager
    def slow_lock(log=print):
        time.sleep(0.3)  # another process's build holds the lock
        yield

    monkeypatch.setattr(kernels, "build_lock", slow_lock)
    _, _, lines = build_levels(monkeypatch, Built(fits=1))
    seconds = float(re.search(r"\(([\d.]+) s\)", lines[0])[1])
    assert seconds < 0.25, lines[0]


def test_the_level_kept_is_the_first_one_built(monkeypatch):
    levels, kept, _ = build_levels(monkeypatch, Built(fits=2), kept=2)
    assert levels == [2] and kept == 2


def test_over_at_the_highest_level_or_another_error_is_the_build_s(monkeypatch):
    with pytest.raises(ProgramError, match="overflows region:0"):
        build_levels(monkeypatch, Built(fits=5))
    with pytest.raises(ProgramError, match="not declared"):
        build_levels(monkeypatch, Built(fits=1, error="'x' was not declared"))


def test_blackhole_builds_once_at_level_0(monkeypatch):
    levels, kept, _ = build_levels(monkeypatch, Built(fits=0), arch="blackhole")
    assert levels == [0] and kept is None


def test_a_level_is_kept_by_program_and_chip(tmp_path, monkeypatch):
    monkeypatch.setattr(remembered, "_path", lambda: tmp_path / "kernel_buffer.json")
    source = tmp_path / "writer.cpp"
    source.write_text("// a writer\n")
    program = types.SimpleNamespace(
        name="p", sources=lambda: {"writer": source}, defines={}, compile_args=None
    )
    assert remembered.writer_level(program, "wormhole_b0") == 0
    remembered.keep_writer_level(program, "wormhole_b0", 2)
    assert remembered.writer_level(program, "wormhole_b0") == 2
    assert remembered.writer_level(program, "blackhole") == 0
    source.write_text("// another writer\n")  # other code: no level kept for it
    assert remembered.writer_level(program, "wormhole_b0") == 0
