"""The comment-length check: each rule with a passing and a failing input."""

import subprocess
import sys

import pytest

from komorebi.tools import comment_lint

MESSAGE = "comment block of 3 lines (limit 2)"
DOC5 = "docstring of 5 lines (limit 4)"
LONG_C = "// a\n// b\n// c\nint x;\n"
LONG_PY = "# a\n# b\n# c\nx = 1\n"


def py(text):
    return comment_lint.check_text(text, ".py")


def c(text, suffix=".cpp"):
    return comment_lint.check_text(text, suffix)


def test_python_comment_run_of_two_passes_and_three_fails_at_its_first_line():
    assert py("# a\n# b\nx = 1\n") == []
    assert py("x = 1\n\n    # a\n    # b\n    # c\ny = 2\n") == [(3, MESSAGE)]
    assert py("# a\n#\n# b\nx = 1\n") == [(1, MESSAGE)]
    assert py("# a\n# b\n\n# c\n# d\nx = 1\n") == []


def test_python_trailing_comments_are_blocks_of_their_own():
    assert py("a = 1  # x\nb = 2  # y\nc = 3  # z\n") == []
    assert py("a = 1  # x\n       # y\n       # z\n") == [(1, MESSAGE)]
    assert py("a = 1  # x\n# y\n# z\n") == []


def test_python_hash_in_a_string_and_a_shebang_are_not_comments():
    assert py('x = """\n# a\n# b\n# c\n"""\n') == []
    assert py("#!/usr/bin/env python\n# a\n# b\nx = 1\n") == []


def test_docstring_of_four_lines_passes_and_five_fails():
    assert py('"""A\nB\n\nD"""\n') == []
    assert py('"""\nA\nB\nC\nD\n"""\n') == []
    assert py('"""\nA\nB\nC\nD\nE\n"""\n') == [(1, DOC5)]
    assert py('x = 1\n"""\na\nb\nc\nd\ne\n"""\n') == []


def test_docstrings_of_functions_classes_and_async_functions_are_checked():
    long = "\n".join(["    x"] * 5)
    for head in ["def f():", "class C:", "async def f():"]:
        assert py(f'{head}\n    """\n{long}\n    """\n') == [(2, DOC5)]


def test_a_syntax_error_is_raised_for_the_caller_to_report():
    with pytest.raises(SyntaxError):
        py("def (:\n")


def test_c_line_and_block_comments_are_checked():
    assert c("// a\n// b\nint x;\n") == []
    assert c(LONG_C) == [(1, MESSAGE)]
    assert c("/* a\n * b */\nint x;\n") == []
    assert c("/* a\n   b\n   c */\nint x;\n") == [(1, MESSAGE)]
    assert c("/* a */\n/* b */\n// c\nint x;\n") == [(1, MESSAGE)]
    assert c("// a \\\nb \\\nc\nint x;\n") == [(1, MESSAGE)]
    assert c(LONG_C, ".S") == [(1, MESSAGE)]


def test_c_trailing_comments_are_blocks_of_their_own():
    assert c("int a; // x\nint b; // y\nint c; // z\n") == []
    assert c("int a;  // x\n        // y\n        // z\n") == [(1, MESSAGE)]


def test_c_comment_markers_inside_literals_are_not_comments():
    text = (
        'const char* a = "// x\\n// y\\n// z";\n'
        "char b = '/'; char d = '\"';\n"
        'const char* e = R"x(\n// 1\n// 2\n// 3\n)x";\n'
        "int n = 1'000'000;\n"
    )
    assert c(text) == []
    assert c('const char* a = "q\\"";\n// 1\n// 2\n// 3\n') == [(2, MESSAGE)]


def test_the_licence_header_at_the_top_is_exempt_in_c_only():
    header = (
        "// SPDX-License-Identifier: Apache-2.0\n// Copyright the authors\n// more\n"
    )
    assert c(header + "int x;\n") == []
    assert c(header + "\n" + LONG_C) == [(5, MESSAGE)]
    assert c("int x;\n" + header) == [(2, MESSAGE)]
    assert py("# SPDX-License-Identifier: Apache-2.0\n# a\n# b\nx = 1\n") != []


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["comment_lint", *args])
    return comment_lint.main()


def test_main_reports_path_line_and_count_and_exits_1(monkeypatch, tmp_path, capsys):
    path = tmp_path / "a.cpp"
    path.write_text("int y;\n" + LONG_C)
    (tmp_path / "ok.py").write_text("# a\nx = 1\n")
    (tmp_path / "notes.txt").write_text(LONG_C)
    assert run(monkeypatch, "--check", str(path)) == 1
    assert capsys.readouterr().out == f"{path}:2: {MESSAGE}\n"
    assert run(monkeypatch, str(tmp_path)) == 1
    assert capsys.readouterr().out == f"{path}:2: {MESSAGE}\n"
    path.unlink()
    assert run(monkeypatch, str(tmp_path)) == 0


def test_main_a_file_that_does_not_parse_is_an_error_not_a_pass(
    monkeypatch, tmp_path, capsys
):
    path = tmp_path / "a.py"
    path.write_text("def (:\n")
    assert run(monkeypatch, str(path)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and str(path) in captured.err


def test_main_default_checks_tracked_files_minus_the_clang_format_ignore(
    monkeypatch, tmp_path, capsys
):
    names = {"a.py": LONG_PY, "sub/b.cc": LONG_C, "vendor/v.h": LONG_C, "n.txt": LONG_C}
    names |= {".clang-format-ignore": "# copies\nvendor/\n", "untracked.py": LONG_PY}
    for name, text in names.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = [file_name for file_name in names if file_name != "untracked.py"]
    subprocess.run(["git", "add", *tracked], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    assert run(monkeypatch) == 1
    out = capsys.readouterr().out.splitlines()
    assert sorted(out) == [f"a.py:1: {MESSAGE}", f"sub/b.cc:1: {MESSAGE}"]
    assert run(monkeypatch, "vendor/v.h") == 1
