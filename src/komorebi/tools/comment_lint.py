"""Comment-length check: runs of `#` comment lines (Python), `//` lines and `/* */`
blocks (C, C++) are at most 2 lines; a Python docstring is at most 4 lines, counted
between its quotes (blank lines included, delimiter-only lines not).
Usage: python -m komorebi.tools.comment_lint [--check] [paths...]
"""

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

from komorebi.tools.cxx_format import is_ignored, repo_root

COMMENT_LINES = 2
DOCSTRING_LINES = 4
C_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".S"}
LICENCE_MARKERS = ("SPDX", "Copyright")
# A character literal; anything else with a quote (a digit separator) is code.
CHAR_LITERAL = re.compile(r"'(?:\\[^\n]*?|[^'\\\n])'")
RAW_PREFIXES = {"", "u8", "u", "U", "L"}
DOC_OPEN = re.compile(r"""[rRuU]?(?:\"\"\"|''')""")
DOC_CLOSE = ('"""', "'''")


@dataclass(frozen=True)
class Comment:
    """One comment: its lines, its column and whether its lines hold no code."""

    first: int
    last: int
    col: int
    pure: bool
    text: str


def joins(previous, item):
    """True when `item` continues the comment block that ends with `previous`: it is
    alone on its lines and starts on or right after `previous`'s last line; a comment
    after code is continued only by comments at its own column."""
    return (
        item.pure
        and item.first <= previous.last + 1
        and (previous.pure or previous.col == item.col)
    )


def blocks(comments):
    """The comments grouped into blocks of adjacent comment lines."""
    runs = []
    for item in comments:
        if runs and joins(runs[-1][-1], item):
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def span(run):
    """(first line, line count) of a block."""
    first = run[0].first
    return first, max(comment.last for comment in run) - first + 1


def comment_findings(comments, skip=None):
    """(line, message) for each block over the limit; `skip` is a block to leave."""
    found = []
    for run in blocks(comments):
        first, count = span(run)
        if run != skip and count > COMMENT_LINES:
            found.append((first, f"comment block of {count} lines (limit 2)"))
    return found


def python_comments(text):
    """The `#` comments of Python source (a shebang line is not one)."""
    lines = text.split("\n")
    out = []
    readline = io.StringIO(text).readline
    for tok in tokenize.generate_tokens(readline):
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        if row == 1 and tok.string.startswith("#!"):
            continue
        pure = not lines[row - 1][:col].strip()
        out.append(Comment(row, row, col, pure, tok.string))
    return out


def docstring_lines(lines, node):
    """Lines between the quotes of a docstring statement: a first line holding only
    the opening quotes and a last line holding only the closing quotes do not count."""
    count = node.end_lineno - node.lineno + 1
    if count == 1:
        return 1
    first = lines[node.lineno - 1].encode()[node.col_offset :].decode().strip()
    last = lines[node.end_lineno - 1].encode()[: node.end_col_offset].decode().strip()
    return count - bool(DOC_OPEN.fullmatch(first)) - (last in DOC_CLOSE)


def docstring_findings(tree, text):
    """(line, message) for each docstring over the limit."""
    lines = text.split("\n")
    found = []
    for scope in ast.walk(tree):
        if not isinstance(
            scope, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        body = scope.body[0] if scope.body else None
        value = body.value if isinstance(body, ast.Expr) else None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            count = docstring_lines(lines, body)
            if count > DOCSTRING_LINES:
                found.append((body.lineno, f"docstring of {count} lines (limit 4)"))
    return found


def check_python(text):
    """(line, message) for the findings in Python source `text`."""
    tree = ast.parse(text)
    found = comment_findings(python_comments(text))
    return sorted(found + docstring_findings(tree, text))


def raw_string_start(text, i):
    """True when the quote at `i` opens a C++ raw string (R, u8R, uR, UR, LR)."""
    if i == 0 or text[i - 1] != "R":
        return False
    start = i - 1
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
        start -= 1
    return text[start : i - 1] in RAW_PREFIXES


def skip_string(text, i, raw):
    """Index after the string literal whose opening quote is at `i`."""
    if raw:
        paren = text.find("(", i)
        end_mark = ")" + text[i + 1 : paren] + '"'
        end = text.find(end_mark, paren) if paren > 0 else -1
        return len(text) if end < 0 else end + len(end_mark)
    i += 1
    while i < len(text) and text[i] != '"':
        if text[i] == "\n":
            return i
        i += 2 if text[i] == "\\" else 1
    return i + 1


def skip_line_comment(text, i):
    """Index of the newline ending the `//` comment at `i` (a backslash continues)."""
    end = text.find("\n", i)
    while end > 0 and text[end - 2 : end].rstrip("\r").endswith("\\"):
        end = text.find("\n", end + 1)
    return len(text) if end < 0 else end


def c_comments(text):
    """(comments, code_lines): the comments of C / C++ source and the lines that
    hold code (string and character literals count as code)."""
    comments, code_lines = [], set()
    i, line, start = 0, 1, 0
    while i < len(text):
        character = text[i]
        end = i + 1
        comment = False
        if character.isspace():
            pass
        elif text.startswith("//", i):
            end, comment = skip_line_comment(text, i), True
        elif text.startswith("/*", i):
            close = text.find("*/", i + 2)
            end, comment = (len(text) if close < 0 else close + 2), True
        elif character == '"':
            end = skip_string(text, i, raw_string_start(text, i))
        else:
            literal = CHAR_LITERAL.match(text, i) if character == "'" else None
            end = literal.end() if literal else end
        lines = text.count("\n", i, end)
        if comment:
            comments.append(Comment(line, line + lines, i - start, True, text[i:end]))
        elif not character.isspace():
            code_lines.update(range(line, line + lines + 1))
        if lines:
            line, start = line + lines, text.rfind("\n", i, end) + 1
        i = end
    return comments, code_lines


def mark_code(comments, code_lines):
    """The comments with `pure` cleared when code shares a line with them."""
    return [
        Comment(
            comment.first,
            comment.last,
            comment.col,
            not any(
                number in code_lines
                for number in range(comment.first, comment.last + 1)
            ),
            comment.text,
        )
        for comment in comments
    ]


def licence_block(comments, code_lines):
    """The block at the top of the file when it is a licence header, else None."""
    if not comments or comments[0].first > min(code_lines, default=comments[0].first):
        return None
    run = blocks(comments)[0]
    if comments[0].pure and any(
        marker in comment.text for comment in run for marker in LICENCE_MARKERS
    ):
        return run
    return None


def check_c(text):
    """(line, message) for the findings in C / C++ source or assembly `text`."""
    raw, code_lines = c_comments(text)
    comments = mark_code(raw, code_lines)
    return sorted(comment_findings(comments, licence_block(comments, code_lines)))


def check_text(text, suffix):
    """(line, message) for the findings in source `text` of a file with `suffix`."""
    if suffix == ".py":
        return check_python(text)
    return check_c(text)


def read_source(path):
    """The text of a source file (Python files in their declared encoding)."""
    data = Path(path).read_bytes()
    encoding = tokenize.detect_encoding(io.BytesIO(data).readline)[0]
    return data.decode(encoding)


def wanted(path):
    """True for a file the check reads."""
    return path.suffix == ".py" or path.suffix in C_SUFFIXES


def under(directory, root):
    """The files below `directory`: those git tracks, else every file."""
    done = subprocess.run(
        ["git", "ls-files", "-z", "--", "."],
        capture_output=True,
        text=True,
        cwd=directory,
    )
    if done.returncode == 0:
        return sorted(directory / name for name in done.stdout.split("\0") if name)
    skipped = {".git", ".venv", "__pycache__"}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not skipped & set(path.relative_to(directory).parts)
    )


def sources(paths):
    """Python and C / C++ files named by `paths` (directories: their tracked files)
    or, by default, those git tracks; `.clang-format-ignore` applies to C and C++
    files found in a directory or the default listing, not to files named."""
    root = repo_root()
    if paths:
        found = []
        for path in map(Path, paths):
            if path.is_dir():
                found += [
                    file
                    for file in under(path, root)
                    if wanted(file) and listed(file, root)
                ]
            elif wanted(path):
                found.append(path)
        return found
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    ).stdout.split("\0")
    files = [root / name for name in tracked if name]
    return [
        file for file in files if wanted(file) and file.is_file() and listed(file, root)
    ]


def listed(path, root):
    """False for a C / C++ file that `.clang-format-ignore` names."""
    return path.suffix == ".py" or not is_ignored(path, root)


def shown_path(path, root):
    """`path` relative to the repository root when it is inside it."""
    try:
        return path.resolve().relative_to(root)
    except ValueError:
        return path


def main():
    """Prints each finding as path:line: message; exit status 1 when any."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="the default; accepted")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    missing = [path for path in args.paths if not Path(path).exists()]
    if missing:
        print(f"comment_lint: no such path: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        files = sources(args.paths)
    except (OSError, subprocess.CalledProcessError):
        print(f"comment_lint: cannot list files in {repo_root()}", file=sys.stderr)
        return 2
    root = repo_root().resolve()
    status = 0
    for path in files:
        try:
            findings = check_text(read_source(path), path.suffix)
        except (SyntaxError, tokenize.TokenError, UnicodeDecodeError) as error:
            print(f"comment_lint: {path}: cannot read: {error}", file=sys.stderr)
            status = max(status, 2)
            continue
        shown = shown_path(path, root)
        for line, message in findings:
            print(f"{shown}:{line}: {message}")
        status = status or int(bool(findings))
    return status


if __name__ == "__main__":
    sys.exit(main())
