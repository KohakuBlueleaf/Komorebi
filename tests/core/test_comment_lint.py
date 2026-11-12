"""The comment-length check: each rule with a passing and a failing input."""

import subprocess
import sys
from pathlib import Path

import pytest

from komorebi.tools import comment_lint


def lines_of(count, prefix):
    return "".join(f"{prefix} line {n}\n" for n in range(count))


def py(text):
    return comment_lint.check_text(text, ".py")


def c(text, suffix=".cpp"):
    return comment_lint.check_text(text, suffix)


def test_python_comment_run_of_two_passes_and_three_fails():
    assert py(lines_of(2, "#") + "x = 1\n") == []
    assert py(lines_of(3, "#") + "x = 1\n") == [
        (1, "comment block of 3 lines (limit 2)")
    ]


def test_python_comment_run_is_reported_at_its_first_line():
    text = "x = 1\n\n" + lines_of(5, "    #") + "y = 2\n"
    assert py(text) == [(3, "comment block of 5 lines (limit 2)")]


def test_python_blank_line_ends_a_comment_run():
    assert py(lines_of(2, "#") + "\n" + lines_of(2, "#") + "x = 1\n") == []


def test_python_empty_comment_lines_count():
    assert py("# a\n#\n# b\nx = 1\n") == [(1, "comment block of 3 lines (limit 2)")]


def test_python_trailing_comments_are_blocks_of_their_own():
    assert py("a = 1  # x\nb = 2  # y\nc = 3  # z\n") == []


def test_python_comment_aligned_under_a_trailing_comment_continues_it():
    text = "a = 1  # x\n       # y\n       # z\n"
    assert py(text) == [(1, "comment block of 3 lines (limit 2)")]
    assert py("a = 1  # x\n# y\n# z\n") == []


def test_python_hash_inside_a_string_is_not_a_comment():
    text = 'x = """\n# a\n# b\n# c\n"""\n'
    assert py(text) == []


def test_python_shebang_is_not_a_comment():
    assert py("#!/usr/bin/env python\n# a\n# b\nx = 1\n") == []
    assert py("#!/usr/bin/env python\n# a\n# b\n# c\nx = 1\n") == [
        (2, "comment block of 3 lines (limit 2)")
    ]


def doc(count, blank=False):
    body = ["text"] * count
    if blank:
        body[1] = ""
    return '"""' + "\n".join(body) + '\n"""\n'


def test_docstring_of_four_lines_passes_and_five_fails():
    assert py(doc(4)) == []
    assert py(doc(5)) == [(1, "docstring of 5 lines (limit 4)")]


def test_docstring_blank_lines_count():
    assert py(doc(4, blank=True)) == []
    assert py(doc(5, blank=True)) == [(1, "docstring of 5 lines (limit 4)")]


def test_docstring_delimiter_only_lines_do_not_count():
    four = '"""\nA\nB\nC\nD\n"""\n'
    five = '"""\nA\nB\nC\nD\nE\n"""\n'
    assert py(four) == []
    assert py(five) == [(1, "docstring of 5 lines (limit 4)")]


def test_docstring_closing_quotes_after_text_count_as_text():
    assert py('"""A\nB\nC\nD"""\n') == []
    assert py('"""A\nB\nC\nD\nE"""\n') == [(1, "docstring of 5 lines (limit 4)")]


def test_docstring_of_function_class_and_async_function_are_checked():
    long = "\n".join(["    x"] * 5)
    for head in ["def f():", "class C:", "async def f():"]:
        text = f'{head}\n    """\n{long}\n    """\n'
        assert py(text) == [(2, "docstring of 5 lines (limit 4)")]


def test_a_string_that_is_not_the_first_statement_is_not_a_docstring():
    assert py('x = 1\n"""\na\nb\nc\nd\ne\n"""\n') == []


def test_a_syntax_error_is_raised_for_the_caller_to_report():
    with pytest.raises(SyntaxError):
        py("def (:\n")


def test_c_line_comment_run_of_two_passes_and_three_fails():
    assert c(lines_of(2, "//") + "int x;\n") == []
    assert c(lines_of(3, "//") + "int x;\n") == [
        (1, "comment block of 3 lines (limit 2)")
    ]


def test_c_block_comment_of_two_lines_passes_and_three_fails():
    assert c("/* a\n * b */\nint x;\n") == []
    assert c("/* a\n * b\n * c */\nint x;\n") == [
        (1, "comment block of 3 lines (limit 2)")
    ]


def test_c_block_comment_continuation_lines_without_stars_count():
    assert c("/* a\n   b\n   c */\nint x;\n") == [
        (1, "comment block of 3 lines (limit 2)")
    ]


def test_c_block_comments_on_adjacent_lines_are_one_block():
    text = "/* a */\n/* b */\n// c\nint x;\n"
    assert c(text) == [(1, "comment block of 3 lines (limit 2)")]


def test_c_blank_line_ends_a_comment_run():
    assert c(lines_of(2, "//") + "\n" + lines_of(2, "//") + "int x;\n") == []


def test_c_line_comment_continued_by_a_backslash_counts_its_lines():
    assert c("// a \\\nb \\\nc\nint x;\n") == [
        (1, "comment block of 3 lines (limit 2)")
    ]


def test_c_trailing_comments_are_blocks_of_their_own():
    assert c("int a; // x\nint b; // y\nint c; // z\n") == []
    assert c("f(/*a=*/1, /*b=*/2);\n") == []


def test_c_comment_aligned_under_a_trailing_comment_continues_it():
    text = "int a;  // x\n        // y\n        // z\n"
    assert c(text) == [(1, "comment block of 3 lines (limit 2)")]


def test_c_comment_markers_inside_literals_are_not_comments():
    text = (
        'const char* a = "// x\\n// y\\n// z";\n'
        "char b = '/'; char d = '\"';\n"
        'const char* e = R"x(\n// 1\n// 2\n// 3\n)x";\n'
        "int n = 1'000'000;\n"
    )
    assert c(text) == []


def test_c_comment_after_a_quote_character_literal_is_found():
    text = "int d = '\"';  // 1\n" + " " * 14 + "// 2\n" + " " * 14 + "// 3\n"
    assert c(text) == [(1, "comment block of 3 lines (limit 2)")]


def test_c_comment_after_a_literal_with_an_escaped_quote_is_found():
    text = 'const char* a = "q\\"";\n// 1\n// 2\n// 3\n'
    assert c(text) == [(2, "comment block of 3 lines (limit 2)")]


def test_assembly_is_checked():
    assert c("// a\n// b\n// c\nnop\n", ".S") == [
        (1, "comment block of 3 lines (limit 2)")
    ]
    assert c("/* a */\nnop\n", ".S") == []


HEADER = "// SPDX-License-Identifier: Apache-2.0\n// Copyright the authors\n// more\n"


def test_licence_header_at_the_top_is_exempt():
    assert c(HEADER + "int x;\n") == []
    assert c("/* Copyright 2026\n * the authors\n * Apache-2.0 */\nint x;\n") == []
    assert c("\n\n" + HEADER + "int x;\n") == []


def test_licence_exemption_covers_only_the_header_block():
    text = HEADER + "\n" + lines_of(3, "//") + "int x;\n"
    assert c(text) == [(5, "comment block of 3 lines (limit 2)")]


def test_licence_exemption_needs_the_marker():
    assert c(lines_of(3, "//") + "int x;\n") != []


def test_licence_text_below_code_is_not_exempt():
    assert c("int x;\n" + HEADER) == [(2, "comment block of 3 lines (limit 2)")]


def test_licence_exemption_does_not_apply_to_python():
    assert py("# SPDX-License-Identifier: Apache-2.0\n# a\n# b\nx = 1\n") != []


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    git("init", cwd=root)
    monkeypatch.chdir(root)
    return root


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def put(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["comment_lint", *args])
    return comment_lint.main()


LONG_C = lines_of(3, "//") + "int x;\n"
LONG_PY = lines_of(3, "#") + "x = 1\n"
MESSAGE = "comment block of 3 lines (limit 2)"


def test_main_reports_path_line_and_count_and_exits_1(monkeypatch, tmp_path, capsys):
    path = tmp_path / "a.cpp"
    path.write_text("int y;\n" + LONG_C)
    assert run(monkeypatch, "--check", str(path)) == 1
    assert capsys.readouterr().out == f"{path}:2: {MESSAGE}\n"


def test_main_exits_0_and_prints_nothing_for_clean_files(monkeypatch, tmp_path, capsys):
    (tmp_path / "a.py").write_text("# a\nx = 1\n")
    (tmp_path / "b.h").write_text("// a\nint x;\n")
    assert run(monkeypatch, str(tmp_path)) == 0
    assert capsys.readouterr().out == ""


def test_main_without_check_behaves_the_same(monkeypatch, tmp_path, capsys):
    path = tmp_path / "a.py"
    path.write_text(LONG_PY)
    assert run(monkeypatch, str(path)) == 1
    assert capsys.readouterr().out == f"{path}:1: {MESSAGE}\n"


def test_main_reports_a_docstring(monkeypatch, tmp_path, capsys):
    path = tmp_path / "a.py"
    path.write_text(doc(5))
    assert run(monkeypatch, str(path)) == 1
    assert capsys.readouterr().out == f"{path}:1: docstring of 5 lines (limit 4)\n"


def test_main_searches_a_directory_for_sources_only(monkeypatch, tmp_path, capsys):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.hpp").write_text(LONG_C)
    (tmp_path / "notes.txt").write_text(LONG_C)
    (tmp_path / "x.md").write_text(LONG_PY)
    assert run(monkeypatch, str(tmp_path)) == 1
    assert capsys.readouterr().out == f"{tmp_path / 'sub' / 'a.hpp'}:1: {MESSAGE}\n"


def test_main_default_checks_tracked_files_only(monkeypatch, repo, capsys):
    put(repo, "tracked.py", LONG_PY)
    put(repo, "sub/tracked.cc", LONG_C)
    put(repo, "untracked.py", LONG_PY)
    put(repo, "notes.txt", LONG_PY)
    git("add", "tracked.py", "sub/tracked.cc", cwd=repo)
    (repo / "sub").joinpath("deleted.c").write_text(LONG_C)
    git("add", "sub/deleted.c", cwd=repo)
    (repo / "sub" / "deleted.c").unlink()
    assert run(monkeypatch) == 1
    out = capsys.readouterr().out.splitlines()
    assert sorted(out) == [f"sub/tracked.cc:1: {MESSAGE}", f"tracked.py:1: {MESSAGE}"]


def test_main_clang_format_ignore_skips_c_files_but_not_python(
    monkeypatch, repo, capsys
):
    put(repo, ".clang-format-ignore", "# copies\nvendor/\nkeep.hpp\n")
    for name in ["vendor/v.h", "keep.hpp", "a.cpp", "vendor/v.py"]:
        put(repo, name, LONG_PY if name.endswith(".py") else LONG_C)
    git("add", ".", cwd=repo)
    assert run(monkeypatch) == 1
    out = capsys.readouterr().out.splitlines()
    assert sorted(out) == [f"a.cpp:1: {MESSAGE}", f"vendor/v.py:1: {MESSAGE}"]


def test_main_a_named_file_is_checked_even_when_ignored(monkeypatch, repo, capsys):
    put(repo, ".clang-format-ignore", "keep.hpp\n")
    put(repo, "keep.hpp", LONG_C)
    git("add", ".", cwd=repo)
    assert run(monkeypatch, "keep.hpp") == 1
    assert capsys.readouterr().out == f"keep.hpp:1: {MESSAGE}\n"


def test_main_a_file_that_does_not_parse_is_an_error_not_a_pass(
    monkeypatch, tmp_path, capsys
):
    path = tmp_path / "a.py"
    path.write_text("def (:\n")
    assert run(monkeypatch, str(path)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and str(path) in captured.err


def test_the_repository_itself_is_clean_of_the_checks_own_files():
    root = Path(comment_lint.__file__).resolve().parents[3]
    mine = [Path(comment_lint.__file__), root / "tests/core/test_comment_lint.py"]
    for path in mine:
        found = comment_lint.check_text(path.read_text(), path.suffix)
        assert found == [], f"{path}: {found}"
