"""The C and C++ formatter: clang-format plus the house rules it cannot express."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from komorebi.tools import cxx_format

ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / ".clang-format"
BINARY = cxx_format.clang_format_binary()
needs_clang_format = pytest.mark.skipif(
    BINARY is None, reason="clang-format not installed"
)


@pytest.fixture(autouse=True)
def repository_style(monkeypatch):
    monkeypatch.setattr(cxx_format, "CONFIG", str(STYLE))


MESSY = (
    "int f(int a){\n"
    "  if(a) return 1;\n"
    "    int  long_name_of_a_value = some_function_name(argument_one, argument_two,"
    " argument_three) + 1;\n"
    "  return 0;}\n"
)
CLEAN = """int f(int a) {
    if (a) {
        return 1;
    }
    int long_name_of_a_value =
        (some_function_name(argument_one, argument_two, argument_three) + 1);
    return 0;
}
"""


@needs_clang_format
def test_format_text_applies_style_braces_and_value_wrap():
    assert cxx_format.format_text(MESSY, BINARY) == CLEAN


@needs_clang_format
def test_format_text_keeps_formatted_text():
    assert cxx_format.format_text(CLEAN, BINARY) == CLEAN


@needs_clang_format
def test_argument_comment_on_its_own_line_becomes_a_trailing_comment():
    text = "void g() {\n    f(\n        /*count=*/3\n    );\n}\n"
    expected = "void g() {\n    f(\n        3  // count\n    );\n}\n"
    assert cxx_format.format_text(text, BINARY) == expected


def run(monkeypatch, *args):
    config = [] if "--config" in args else ["--config", str(STYLE)]
    monkeypatch.setattr(sys, "argv", ["cxx_format", *config, *args])
    return cxx_format.main()


@needs_clang_format
def test_check_lists_a_file_without_changing_it(monkeypatch, tmp_path, capsys):
    path = tmp_path / "a.cpp"
    path.write_text(MESSY)
    assert run(monkeypatch, "--check", str(path)) == 1
    assert f"would reformat {path}" in capsys.readouterr().out
    assert path.read_text() == MESSY


@needs_clang_format
def test_format_rewrites_then_check_passes(monkeypatch, tmp_path):
    path = tmp_path / "a.cpp"
    path.write_text(MESSY)
    assert run(monkeypatch, str(path)) == 0
    assert path.read_text() == CLEAN
    assert run(monkeypatch, "--check", str(path)) == 0


@needs_clang_format
def test_a_directory_is_searched_for_sources_only(monkeypatch, tmp_path):
    (tmp_path / "a.hpp").write_text(MESSY)
    (tmp_path / "notes.txt").write_text(MESSY)
    assert run(monkeypatch, "--check", str(tmp_path)) == 1
    assert (tmp_path / "notes.txt").read_text() == MESSY


def test_a_missing_style_file_is_a_one_line_error(monkeypatch, tmp_path, capsys):
    assert run(monkeypatch, "--config", str(tmp_path / "none"), "a.cpp") == 2
    err = capsys.readouterr().err
    assert str(tmp_path / "none") in err and err.count("\n") == 1


def test_without_a_style_file_in_the_current_directory_it_is_a_one_line_error(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["cxx_format", "--check"])
    assert cxx_format.main() == 2
    err = capsys.readouterr().err
    assert ".clang-format" in err and "Traceback" not in err and err.count("\n") == 1


def test_the_style_file_is_used_in_place_at_its_own_column_limit(tmp_path):
    assert cxx_format.style_file(STYLE, 88) == STYLE
    other = cxx_format.style_file(STYLE, 60)
    assert other != STYLE and "ColumnLimit: 60" in other.read_text()
    assert STYLE.read_text().count("ColumnLimit: 88") == 1


def test_a_style_file_without_a_column_limit_gets_one(tmp_path):
    config = tmp_path / "style.yaml"
    config.write_text("IndentWidth: 4\n")
    copy = cxx_format.style_file(config, 70)
    assert copy.read_text().splitlines() == ["IndentWidth: 4", "", "ColumnLimit: 70"]


@needs_clang_format
def test_no_temporary_directory_is_left_behind():
    code = (
        "from komorebi.tools import cxx_format as c; "
        f"c.CONFIG = {str(STYLE)!r}; c.clang_format('int x;', {BINARY!r}, 60); "
        "print(c.style_file(c.config_file(), 60))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert not Path(done.stdout.strip()).parent.exists()


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_default_listing_splits_on_null_and_skips_deleted_files(monkeypatch, tmp_path):
    repo = tmp_path.resolve()
    git("init", cwd=repo)
    for name in ["with space.cpp", "gone.cpp", "notes.txt", "out/skip.cpp"]:
        (repo / name).parent.mkdir(exist_ok=True)
        (repo / name).write_text("int x;\n")
    git("add", ".", cwd=repo)
    (repo / "gone.cpp").unlink()
    (repo / "new.hpp").write_text("int y;\n")
    (repo / "sub").mkdir()
    monkeypatch.chdir(repo / "sub")
    assert sorted(cxx_format.sources([])) == [repo / "new.hpp", repo / "with space.cpp"]


def test_files_matching_the_ignore_file_are_skipped_unless_named(monkeypatch, tmp_path):
    repo = tmp_path.resolve()
    git("init", cwd=repo)
    (repo / ".clang-format-ignore").write_text("# copies\nvendor/\nkeep.hpp\n*_gen.c\n")
    names = ["a.cpp", "keep.hpp", "x_gen.c", "vendor/v.h", "vendor/sub/w.h", "b/c.cc"]
    for name in names:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("int x;\n")
    monkeypatch.chdir(repo)
    assert sorted(cxx_format.sources([])) == [repo / "a.cpp", repo / "b/c.cc"]
    assert sorted(cxx_format.sources(["."])) == [Path("a.cpp"), Path("b/c.cc")]
    assert cxx_format.sources(["keep.hpp"]) == [Path("keep.hpp")]
