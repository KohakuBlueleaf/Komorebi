"""A failed build fails the prebuild: in_load() finds a load's frames by module, a real
failed build exits as a failure, and the card check counts a prebuild line only when
its kernels were built (a stop on the stand-in's zeros after its loads still counts)."""

import os
import subprocess
import sys

import pytest

from komorebi.builder import prebuild
from komorebi.device import mock
from komorebi.session import session
from komorebi.tools import card_check

PROGRAM = """
import numpy as np

import komorebi as km

BODY = "copy_init(tile_buffer){end}\\ncopy_tile(tile_buffer, tile_index, 0);"


class Scaled(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 1), write=True)
        for tile in km.Stream(a.pages(0, 1)):
            scaled = km.cxx(BODY, takes=[tile], gives=km.Tile)
            results.push(scaled)


with km.Session() as session:
    source = session.tensor(np.zeros((32, 32), np.float32), "bfloat16", "tile")
    target = session.tensor(np.zeros((32, 32), np.float32), "bfloat16", "tile")
    session.load(Scaled(a=source, out=target))
    raise ValueError("a check of the script's on the stand-in's zeros")
"""
LAUNCHED = """
import komorebi as km

with km.Session() as session:
    session.launch()  # a raw launch counts as one load
    raise ValueError("a check of the script's on the stand-in's zeros")
"""
# a child that runs the script as prebuild's does and says whether its error is a load's
DRIVER = (
    "import runpy, sys\n"
    "import komorebi.builder.prebuild as p\n"
    "try:\n"
    "    runpy.run_path(sys.argv[1], run_name='__main__')\n"
    "except Exception as error:\n"
    "    print('IN_LOAD', p.in_load(error))\n"
)


def mock_environment(folder):
    """The environment `prebuild.run` gives its child: a mock of card 0."""
    env = dict(os.environ, **{mock.ACTIVE: "1"})
    env[mock.MOCK] = str(mock.descriptor((0,), "blackhole"))
    env[mock.MEMORY] = str(folder / "kernel_buffer.json")
    env["TT_VISIBLE_DEVICES"] = "0"
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    return env


def test_the_load_frames_are_the_session_s_where_it_defines_them():
    for name in ("_load", "load_islands"):
        function = getattr(session.Session, name)
        assert (function.__module__, function.__code__.co_name) in prebuild.LOAD_FRAMES


def test_an_error_of_the_script_is_not_a_load_s():
    try:
        raise ValueError("the script's own")
    except ValueError as error:
        assert not prebuild.in_load(error)


@pytest.mark.tree
def test_a_failed_build_is_in_a_load_and_the_prebuild_exits_as_a_failure(tmp_path):
    script = tmp_path / "broken.py"
    script.write_text(PROGRAM.replace("{end}", ""))  # a ';' missing: a syntax error
    done = subprocess.run(
        [sys.executable, "-c", DRIVER, str(script)],
        env=mock_environment(tmp_path),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "IN_LOAD True" in done.stdout, done.stderr[-2000:]
    code = prebuild.run(script)
    assert code not in (0, prebuild.REFUSED, prebuild.STOPPED, prebuild.EXITED)


def test_a_script_that_stops_on_zeros_after_a_load_is_a_stop_the_check_accepts(
    tmp_path, capfd
):
    script = tmp_path / "launched.py"
    script.write_text(LAUNCHED)
    assert prebuild.run(script) == prebuild.STOPPED
    said = capfd.readouterr().err
    assert "the script stopped at line 6 after 1 load(s) (ValueError" in said
    assert card_check.prebuild_failure(prebuild.STOPPED, 0, said) == ""


# kmr prebuild's last words, as printed (paths and compiler noise trimmed)
BUILD_FAILED = """\
src/prog.py:15:51: error: expected ';' before 'copy_tile'
   15 |             scaled = km.cxx(BODY, takes=[tile], gives=km.Tile)
kmr prebuild: the script stopped at line 22 after 0 load(s) (ProgramError: prog: build \
failed: TT_THROW @ program.cpp:419: tt::exception ... src/prog.py:15:51: error: \
expected ';' before 'copy_tile')
"""
LATER_FAILED = BUILD_FAILED.replace("after 0 load(s)", "after 2 load(s)")
ZEROS_AFTER_LOADS = """\
Scaled: built into build/scaled-1da73ffe7e58d9eb
built: 3 found, 0 compiled
kmr prebuild: the script stopped at line 24 after 1 load(s) (ValueError: a check of \
the script's on the stand-in's zeros)
"""
ASSERT_ON_ZEROS = """\
built: 3 found, 0 compiled
kmr prebuild: the script's assert at line 30 failed after 2 load(s) (AssertionError: )
"""
ENDED = (
    "built: 3 found, 0 compiled\nkmr prebuild: 2 load(s) built; a prebuild follows\n"
)
CAUGHT_REFUSAL = "kmr prebuild: the script exited 1\nkmr prebuild: 0 load(s) built\n"


def test_a_stop_with_no_load_built_fails_with_the_log_s_first_error_line():
    why = card_check.prebuild_failure(prebuild.STOPPED, 0, BUILD_FAILED)
    assert why.startswith("no load built; first error: ")
    assert why.endswith("error: expected ';' before 'copy_tile'")


def test_a_stop_after_loads_that_names_a_failed_build_fails():
    why = card_check.prebuild_failure(prebuild.STOPPED, 0, LATER_FAILED)
    assert why.startswith("a load's build failed; first error: ")


def test_a_stop_after_every_load_was_built_counts():
    assert card_check.prebuild_failure(prebuild.STOPPED, 0, ZEROS_AFTER_LOADS) == ""
    assert card_check.prebuild_failure(0, 0, ASSERT_ON_ZEROS) == ""
    assert card_check.prebuild_failure(0, 0, ENDED) == ""


def test_an_end_or_a_refusal_with_no_load_built_fails():
    for code, text in ((0, "kmr prebuild: 0 load(s) built\n"), (77, CAUGHT_REFUSAL)):
        assert "no load built" in card_check.prebuild_failure(code, 0, text)


def test_a_second_pass_that_compiled_or_a_foreign_exit_fails():
    assert "2 compiled" in card_check.prebuild_failure(0, 2, ENDED)
    why = card_check.prebuild_failure(
        1, 0, "Traceback\nkmr prebuild: 2 load(s) built\n"
    )
    assert why.startswith("exit 1 is not a prebuild's own")


def test_the_loads_said_are_the_last_words_and_the_error_line_is_cut():
    assert card_check.built_loads(ZEROS_AFTER_LOADS + ENDED) == 2
    assert card_check.built_loads("built: 3 found, 0 compiled\n") == 0
    long = "x.cpp:1: error: " + "y" * 500
    assert len(card_check.first_error(long)) == card_check.ERROR + 4
    assert card_check.first_error("a\nkmr prebuild: the script stopped\nb") == (
        "kmr prebuild: the script stopped"
    )
