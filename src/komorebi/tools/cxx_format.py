"""C / C++ formatter: clang-format, then rules it cannot express (wrapped brackets
indented one level from the line that opens them, overlong values as ``= (`` / value /
``);``, ``;`` in empty loops, ``value,  // name`` arguments).
Usage: python -m komorebi.tools.cxx_format [--check] [--config FILE] [paths...]
"""

import argparse
import atexit
import contextlib
import fnmatch
import functools
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

COLUMNS = 88
INDENT = 4
SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}

# The style file to use instead of <root>/.clang-format (set by --config).
CONFIG = None
# Temporary directories of style files with another ColumnLimit; removed at exit.
_TEMP = contextlib.ExitStack()
atexit.register(_TEMP.close)

# A statement assigning (or returning) a value: indent, head (target and operator).
ASSIGN = re.compile(
    r"^(\s*)(return|(?!(?:if|for|while|switch|case)\b)[^=;\"'({]*?[^=!<>\s(]\s*"
    r"(?:<<|>>|[-+*/%&|^])?=)(?:\s+(\S.*))?$"
)
# A declaration clang-format split after its type, and the line naming it.
TYPE_ONLY = re.compile(
    r"^\s*(?:(?:const|constexpr|static|unsigned|auto)\s+)*[\w:<>]+[&*]?$"
)
NAMED_VALUE = re.compile(r"^\s*\w+ = \S")
OWN_LINE_ARG = re.compile(r"^(\s*)/\*(\w+)=\*/(.*?)(,?)\s*$")
EMPTY_LOOP = re.compile(r"^(\s*)(?:while|for) \(.*\) \{$")


def clang_format_binary():
    """clang-format beside the running Python (the dev venv), else on PATH."""
    local = Path(sys.executable).with_name("clang-format")
    return str(local) if local.exists() else shutil.which("clang-format")


def scan(line):
    """(code, has_comment): the line with literals and a // comment blanked."""
    out, quote, i = [], None, 0
    while i < len(line):
        character = line[i]
        if quote:
            out.append(" ")
            if character == "\\" and i + 1 < len(line):
                out.append(" ")
                i += 1
            elif character == quote:
                quote = None
        elif character in "\"'":
            quote = character
            out.append(" ")
        elif line.startswith("//", i):
            # The comment blanked; a trailing newline kept (positions line up).
            tail = line[i:]
            newline = "\n" if tail.endswith("\n") else ""
            out.append(" " * (len(tail) - len(newline)) + newline)
            return "".join(out), True
        else:
            out.append(character)
        i += 1
    return "".join(out), False


def statement_end(lines, start):
    """Index of the line ending the statement starting at `start` (its ; at bracket
    depth 0); None for anything else (braces anywhere, macros, comments, no end)."""
    depth = 0
    for i in range(start, min(start + 200, len(lines))):
        code, comment = scan(lines[i].rstrip("\n"))
        code = code.rstrip()
        # A /* */ comment too: after a ; it hides that ; from the test below.
        if comment or "/*" in code or code.endswith("\\"):
            return None
        for character in code:
            if character in "([":
                depth += 1
            elif character in ")]":
                depth -= 1
            elif character in "{}":
                return None
        if depth == 0 and code.endswith(";"):
            return i
    return None


def join(parts):
    """The lines of one expression as one line."""
    out = ""
    for part in parts:
        part = part.strip()
        glued = part.startswith((")", "]", ".", "->", "::")) or out.endswith(
            ("(", "[", "::")
        )
        if out and not glued:
            out += " "
        out += part
    return out


def top_level_comma(text):
    """True when `text` has a comma outside brackets (several declarators)."""
    depth = 0
    for character in scan(text)[0]:
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            return True
    return False


def strip_outer_parens(text):
    """`text` without one pair of parentheses enclosing all of it."""
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for i, character in enumerate(scan(text)[0]):
        depth += character == "("
        depth -= character == ")"
        if depth == 0 and i < len(text) - 1:
            return text
    return text[1:-1].strip()


def repo_root():
    """The git top-level of the current directory, else the current directory."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return Path.cwd()
    return Path(done.stdout.strip())


def config_file():
    """The style file: CONFIG if set, else .clang-format in the repository root."""
    return Path(CONFIG) if CONFIG else repo_root() / ".clang-format"


@functools.cache
def style_file(config, columns):
    """`config` itself when its ColumnLimit is `columns`, else a copy that has it."""
    base = config.read_text()
    limit = f"ColumnLimit: {columns}"
    if re.search(rf"(?m)^{limit}\s*$", base):
        return config
    text, found = re.subn(r"(?m)^ColumnLimit:.*$", limit, base)
    path = Path(_TEMP.enter_context(tempfile.TemporaryDirectory(prefix="kmr-cxx-")))
    (path / "style").write_text(text if found else f"{base}\n{limit}\n")
    return path / "style"


def clang_format(text, binary, columns=COLUMNS):
    """clang-format's output for C++ `text` at `columns`."""
    style = style_file(config_file(), columns)
    return subprocess.run(
        [binary, f"--style=file:{style}", "--assume-filename=x.cpp"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def format_value(value, indent, binary):
    """clang-format's layout of expression `value` starting at column `indent`, as the
    sole argument of a call (operands of a broken expression stay at one indent)."""
    snippet = f"void f() {{\n    kmr_value_({value});\n}}\n"
    columns = COLUMNS - indent + 2 * INDENT
    lines = clang_format(snippet, binary, columns).splitlines()[1:-1]
    first, last = lines[0].strip(), lines[-1].strip()
    arg = " " * (2 * INDENT)
    if len(lines) == 1:
        # Everything on one line: kmr_value_(value);
        body = [arg + first[len("kmr_value_(") : -2]]
    elif first == "kmr_value_(":
        # The value on its own lines between kmr_value_( and );.
        body = lines[1:-1]
    else:
        # The value hugged: kmr_value_(call( ... )); its inner lines one indent deeper.
        inner = [" " * INDENT + line for line in lines[1:-1]]
        body = [arg + first[len("kmr_value_(") :], *inner, arg + last[:-2]]
    pad = " " * (indent - 2 * INDENT)
    return [pad + line + "\n" for line in body]


def unwrap_braced_values(lines):
    """Parentheses around an assigned / returned value containing braces (a lambda,
    an initializer) removed: such values keep clang-format's own layout."""
    text = "".join(lines)
    code = "".join(scan(line)[0] for line in text.splitlines(keepends=True))
    out, last = [], 0
    for matched in re.finditer(r"(?:\breturn|[^=!<>]=)\s*\(", code):
        open_at = matched.end() - 1
        depth, close_at, braced = 0, None, False
        for position in range(open_at, len(code)):
            character = code[position]
            depth += character == "("
            depth -= character == ")"
            braced |= character == "{"
            if depth == 0:
                close_at = position
                break
        after = code[close_at + 1 :].lstrip() if close_at else ""
        if close_at and braced and after.startswith(";") and open_at >= last:
            out += [text[last:open_at], text[open_at + 1 : close_at]]
            last = close_at + 1
    out.append(text[last:])
    return "".join(out).splitlines(keepends=True)


def merge_split_declarations(lines):
    """A type-only line joined with the ``name = value`` line after it."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if TYPE_ONLY.match(line) and NAMED_VALUE.match(next_line):
            out.append(f"{line} {next_line.lstrip()}")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return out


def split_declarators(indent, head, value):
    """``type a = x, b = y;`` that wraps as one declaration per line (the lines, or None
    when the statement is not a plain declaration of values)."""
    declaration = re.fullmatch(r"((?:[\w:<>]+\s+)+)(\w+)\s*=", head)
    if not declaration or any(character in declaration[1] for character in "*&[("):
        return None
    items = split_items(value)
    if len(items) < 2 or not all(
        re.fullmatch(r"\w+(\s*=\s*\S.*)?", x) for x in items[1:]
    ):
        return None
    prefix = declaration[1].strip()
    return [
        f"{indent}{head} {items[0]};\n",
        *(f"{indent}{prefix} {item};\n" for item in items[1:]),
    ]


def wrap_values(lines, binary):
    """Assigned / returned values that span lines or overflow the line as ``= (`` /
    value / ``);``."""
    lines = merge_split_declarations(unwrap_braced_values(lines))
    out, i = [], 0
    while i < len(lines):
        assignment = ASSIGN.match(lines[i].rstrip("\n"))
        if assignment and re.match(
            r"(using|typedef|template|namespace)\b", assignment[2]
        ):
            assignment = None  # a type alias or declaration, not a value
        end = statement_end(lines, i) if assignment else None
        fits = len(lines[i].rstrip("\n")) <= COLUMNS
        if end is None or (end == i and fits):
            out.append(lines[i])
            i += 1
            continue
        indent, head, first = assignment.groups()
        first = first or ""
        rest = (lines[number].rstrip("\n") for number in range(i + 1, end + 1))
        value = strip_outer_parens(join([first, *rest])[:-1].rstrip())
        if value.startswith(("[", "{")) or top_level_comma(value):
            out.extend(split_declarators(indent, head, value) or lines[i : end + 1])
        elif len(indent) + len(head) + len(value) + 2 <= COLUMNS:
            out.append(f"{indent}{head} {value};\n")
        else:
            inner = len(indent) + INDENT
            if inner + len(value) <= COLUMNS:
                body = [" " * inner + value + "\n"]
            else:
                body = format_value(value, inner, binary)
            out += [f"{indent}{head} (\n", *body, f"{indent});\n"]
        i = end + 1
    return out


def split_items(text):
    """`text` split at commas outside brackets, items stripped."""
    items, depth, start = [], 0, 0
    for i, character in enumerate(scan(text)[0]):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            items.append(text[start:i].strip())
            start = i + 1
    items.append(text[start:].strip())
    return [item for item in items if item]


def hug_braced_lists(lines):
    """A braced list that is a call's only argument opens on the call's line:
    ``f({`` / one element per line / ``});``."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not (
            line.endswith("(")
            and not line.endswith("= (")
            and next_line.startswith("{")
        ):
            out.append(lines[i])
            i += 1
            continue

        # The list: from the next line to its matching brace, then the call's close.
        depth, j, text = 0, i + 1, ""
        while j < len(lines):
            code, comment = scan(lines[j].rstrip("\n"))
            if comment:
                break
            depth += code.count("{") - code.count("}")
            text = join([text, lines[j]]) if text else lines[j].strip()
            if depth == 0:
                break
            j += 1
        close = lines[j + 1].strip() if j + 1 < len(lines) else ""
        indent = re.match(r"\s*", line).group(0)
        inner = len(indent) + INDENT
        items = split_items(text[1:-1]) if text.endswith("}") and depth == 0 else []
        fits = all(inner + len(item) + 1 <= COLUMNS for item in items)
        if not items or not fits or not close.startswith(")"):
            out.append(lines[i])
            i += 1
            continue
        out.append(line + "{\n")
        out += [" " * inner + item + ",\n" for item in items[:-1]]
        out.append(" " * inner + items[-1] + "\n")
        out.append(indent + "}" + close + "\n")
        i = j + 2
    return out


def matching_brace(text):
    """Index of the brace closing `text`'s opening brace (text[0] == "{"), or -1."""
    depth = 0
    for i, character in enumerate(scan(text)[0]):
        depth += character == "{"
        depth -= character == "}"
        if depth == 0:
            return i
    return -1


def block_braced_lists(lines):
    """A braced list clang-format aligns after its brace (``{a,`` / `` b}``) laid out
    as ``{`` / one element per line / ``}`` (then whatever followed the list)."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.lstrip()
        code, comment = scan(line)
        opens = stripped.startswith("{") and stripped != "{"
        if not opens or comment or code.count("{") <= code.count("}"):
            out.append(lines[i])
            i += 1
            continue

        # The list's lines up to its matching brace; `rest` follows it on that line.
        depth, j, text = 0, i, ""
        while j < len(lines):
            code, comment = scan(lines[j].rstrip("\n"))
            if comment:
                break
            depth += code.count("{") - code.count("}")
            text = join([text, lines[j]]) if text else lines[j].strip()
            if depth == 0:
                break
            j += 1
        close = matching_brace(text) if depth == 0 else -1
        indent = line[: len(line) - len(stripped)]
        inner = len(indent) + INDENT
        rest_ok = close > 0 and re.fullmatch(r"[)\]};,\s]*", text[close + 1 :])
        items = split_items(text[1:close]) if rest_ok else []
        if not items or any(inner + len(item) + 1 > COLUMNS for item in items):
            out.append(lines[i])
            i += 1
            continue
        out.append(indent + "{\n")
        out += [" " * inner + item + ",\n" for item in items[:-1]]
        out.append(" " * inner + items[-1] + "\n")
        out.append(indent + text[close:] + "\n")
        i = j + 1
    return out


def blank_code(lines):
    """Per line (code, kind): the line with literals and comments blanked, and its kind:
    "blank", "preprocessor", "comment" (only a comment) or "code"."""
    out, in_block, continued = [], False, False
    for line in lines:
        text = line.rstrip("\n")
        if not in_block and (continued or text.lstrip().startswith("#")):
            continued = text.rstrip().endswith("\\")
            out.append(("", "preprocessor"))
            continue
        code, quote, i = [], None, 0
        while i < len(text):
            character = text[i]
            if in_block:
                end = text.find("*/", i)
                stop = len(text) if end < 0 else end + 2
                code.append(" " * (stop - i))
                in_block = end < 0
                i = stop
                continue
            if quote:
                code.append(" ")
                if character == "\\" and i + 1 < len(text):
                    code.append(" ")
                    i += 1
                elif character == quote:
                    quote = None
            elif character in "\"'":
                quote = character
                code.append(" ")
            elif text.startswith("//", i):
                code.append(" " * (len(text) - i))
                break
            elif text.startswith("/*", i):
                in_block = True
                code.append("  ")
                i += 1
            else:
                code.append(character)
            i += 1
        code = "".join(code)
        kind = "code" if code.strip() else ("comment" if text.strip() else "blank")
        out.append((code, kind))
    return out


def reindent_brackets(lines):
    """Wrapped bracket contents one level in from the line that opens the bracket, the
    closing bracket at that line's indent: nothing aligned under an operand, under
    ``return`` or under the bracket's column. A line continuing an item keeps its indent
    relative to it; contents on the bracket's own line move with it."""
    code = blank_code(lines)
    new = [0] * len(lines)  # the indent each line ends up with
    moved = [0] * len(lines)  # by how much it moved
    anchor = [0] * len(lines)  # by how much the line starting its statement moved
    # A bracket: [line, character, wraps, contents' move, after ',', inside another]
    out, stack = [], []
    for number, line in enumerate(lines):
        text = line.rstrip("\n")
        blanked, kind = code[number]
        indent = len(text) - len(text.lstrip(" "))
        new[number] = indent
        if kind in ("blank", "preprocessor"):
            out.append(line)
            continue
        first = blanked.lstrip()[:1]
        if stack:
            entry = stack[-1]
            opener_line, bracket, wraps, shift, comma, inside = entry
            if bracket == "{" and not inside:
                shift = anchor[opener_line]
            elif first in (")", "]", "}"):
                shift = new[opener_line] - indent
            elif not wraps:
                shift = moved[opener_line]
            elif shift is None or (comma and bracket != "{"):
                shift = new[opener_line] + INDENT - indent
                entry[3] = shift
            moved[number] = shift
            new[number] = max(indent + shift, 0)
        # A line of closing brackets (``)) {``) starts nothing: its statement does
        closes = first in (")", "]", "}")
        anchor[number] = anchor[stack[-1][0]] if stack and closes else moved[number]
        out.append(" " * new[number] + text.lstrip(" ") + "\n")
        for j, character in enumerate(blanked):
            rest = not blanked[j + 1 :].strip()
            template = (
                character == "<" and rest and j and re.match(r"[\w:>]", blanked[j - 1])
            )
            if character in "([{" or template:
                inside = bool(stack) and stack[-1][1] in "([<"
                stack.append([number, character, rest, None, False, inside])
            elif character in ")]}" or (
                character == ">" and stack and stack[-1][1] == "<"
            ):
                while stack and character != ">" and stack[-1][1] == "<":
                    stack.pop()
                if stack:
                    stack.pop()
        if stack and kind == "code":
            stack[-1][4] = blanked.rstrip().endswith(",")
    return out


def join_member_calls(lines):
    """``)`` and the member call clang-format put on the line after it (``.count();``)
    joined, when they fit on one line."""
    out = []
    for line in lines:
        text = line.strip()
        joined = out[-1].rstrip("\n") + text if out else ""
        if (
            out
            and text.startswith((".", "->"))
            and out[-1].strip().startswith(")")
            and len(joined) <= COLUMNS
        ):
            out[-1] = joined + "\n"
        else:
            out.append(line)
    return out


def hugged(code, lines, i):
    """(column of the bracket, line, column of its closing bracket) for the first
    bracket left open at the end of line `i` that has more text after it; None when
    there is none or it cannot be laid out safely."""
    here = code[i][0].rstrip()
    stack = []
    for j, character in enumerate(here):
        if character in "([{":
            stack.append(j)
        elif character in ")]}" and stack:
            stack.pop()
    if not stack or here[-1] in "([{" or any(here[j] == "{" for j in stack):
        return None
    open_column, depth = stack[0], 1
    for number in range(i, min(i + 60, len(lines))):
        part, kind = code[number]
        raw = lines[number].rstrip("\n")
        if kind != "code" or "/*" in raw:
            return None
        for column in range(open_column + 1 if number == i else 0, len(part)):
            if (
                part[column] in "{};"
                and part[column : column + 2] != "{}"
                and part[column - 1 : column + 1] != "{}"
            ):
                return None
            depth += part[column] in "(["
            depth -= part[column] in ")]"
            if depth == 0:
                return open_column, number, column
        if scan(raw)[1]:
            return None
    return None


def block_hugged_brackets(lines, binary):
    """A bracket clang-format opened without breaking after it and whose contents wrap
    (``f(a,`` / ``  b)``, ``if (a &&`` / ``    b) {``) laid out as ``f(`` / the
    contents, indented / ``)``. A bracket with a brace, a comment or a ``;`` inside
    stays as it is."""
    code = blank_code(lines)
    out, i = [], 0
    while i < len(lines):
        text = lines[i].rstrip("\n")
        found = hugged(code, lines, i) if code[i][1] == "code" else None
        if found is None:
            out.append(lines[i])
            i += 1
            continue
        open_column, end, close = found
        last = lines[end].rstrip("\n")
        pieces = [
            text[open_column + 1 :],
            *(line.rstrip("\n") for line in lines[i + 1 : end]),
        ]
        contents = join([*pieces, last[:close]])
        indent = len(text) - len(text.lstrip(" "))
        inner = indent + INDENT
        if inner + len(contents) <= COLUMNS and not top_level_comma(contents):
            body = [" " * inner + contents + "\n"]
        else:
            body = format_value(contents, inner, binary)
        out += [
            text[: open_column + 1] + "\n",
            *body,
            " " * indent + last[close:] + "\n",
        ]
        i = end + 1
    return out


def ternary_parts(text):
    """(condition, then, else) of `text` when it is one conditional expression (a ``?``
    outside brackets with its ``:``), else None."""
    code, depth, nested, found = scan(text)[0], 0, 0, None
    for i, character in enumerate(code):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif depth or character not in "?:" or code[i - 1 : i + 2].count(":") == 2:
            continue
        elif character == "?" and found is None:
            found = i
        elif character == "?":
            nested += 1
        elif found is not None and nested == 0:
            return (
                text[:found].strip(),
                text[found + 1 : i].strip(),
                text[i + 1 :].strip(),
            )
        elif found is not None:
            nested -= 1
    return None


def ternary_lines(text, indent, binary, prefix=""):
    """A conditional expression, one part per line: the condition, then ``? value`` and
    ``: value`` one level in; a chain of them one level in again for each link."""
    condition, then, other = ternary_parts(text)
    inner = indent + INDENT

    def piece(value, column, lead):
        if column + len(lead) + len(value) <= COLUMNS:
            return [" " * column + lead + value]
        lines = format_value(value, column + len(lead), binary)
        shift = column + INDENT - (len(lines[0]) - len(lines[0].lstrip(" ")))
        rest = [" " * shift + line.rstrip("\n") for line in lines[1:]]
        return [" " * column + lead + lines[0].strip(), *rest]

    out = piece(condition, indent, prefix) + piece(then, inner, "? ")
    if ternary_parts(other):
        return out + ternary_lines(other, inner, binary, ": ")
    return out + piece(other, inner, ": ")


def wrap_ternaries(lines, binary):
    """A conditional expression among the contents of a wrapped bracket laid out one
    part per line (see ternary_lines) instead of ``?`` and ``:`` under operands."""
    code = blank_code(lines)
    stack, context = [], []  # per line: the line of the wrapped bracket that holds it
    for number, (blanked, _) in enumerate(code):
        context.append(stack[-1][0] if stack and stack[-1][1] else -1)
        for j, character in enumerate(blanked):
            if character in "([{":
                stack.append(
                    (number, character != "{" and not blanked[j + 1 :].strip())
                )
            elif character in ")]}" and stack:
                stack.pop()

    def last_char(number):
        return code[number][0].rstrip()[-1:]

    out, i = [], 0
    while i < len(lines):
        owner = context[i]
        if owner < 0 or code[i][1] != "code" or last_char(i - 1) not in (",", "(", "["):
            out.append(lines[i])
            i += 1
            continue
        # An item: from here to the line before the next one in the same bracket.
        end = i
        while end + 1 < len(lines) and (
            context[end + 1] > owner
            or (
                context[end + 1] == owner
                and code[end + 1][1] == "code"
                and last_char(end) != ","
                and code[end + 1][0].lstrip()[:1] not in (")", "]")
            )
        ):
            end += 1
        item = lines[i : end + 1]
        text = join([line.rstrip("\n") for line in item])
        trailing = text.endswith(",")
        body = text[:-1].rstrip() if trailing else text
        own = [number for number in range(i, end + 1) if context[number] == owner]
        marked = any(code[number][0].lstrip()[:2] in ("? ", ": ") for number in own[1:])
        plain = all(
            code[number][1] == "code" and not set("{};") & set(code[number][0])
            for number in range(i, end + 1)
        )
        if not (marked and plain and ternary_parts(body)):
            out.append(lines[i])  # an item inside it may still be one
            i += 1
            continue
        i = end + 1
        indent = len(item[0]) - len(item[0].lstrip(" "))
        laid = [line + "\n" for line in ternary_lines(body, indent, binary)]
        out += laid[:-1] + [laid[-1].rstrip("\n") + (",\n" if trailing else "\n")]
    return out


def wrap_lambda_values(lines):
    """A lambda assigned on the line after ``=`` laid out on that line, its parameters
    wrapped: ``f = [&](`` / the parameters / ``) {`` / the body / ``};``."""
    out, i = [], 0
    while i < len(lines):
        head = lines[i].rstrip("\n")
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        assignment = re.fullmatch(r"(\s*)(.*[^=!<>\s])\s*=", head)
        lambda_match = re.fullmatch(r"(\[[^\]]*\])\((.*)\)((?:\s+\w+)*) \{", next_line)
        if not (
            assignment
            and lambda_match
            and scan(next_line)[0].count("(") == scan(next_line)[0].count(")")
        ):
            out.append(lines[i])
            i += 1
            continue
        indent = assignment.group(1)
        lambda_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip(" "))
        end = i + 2
        while end < len(lines) and not (
            lines[end].startswith(" " * lambda_indent + "}")
            and len(lines[end]) - len(lines[end].lstrip(" ")) == lambda_indent
        ):
            end += 1
        if end >= len(lines):
            out.append(lines[i])
            i += 1
            continue
        params = lambda_match.group(2)
        inner = len(indent) + INDENT
        if inner + len(params) <= COLUMNS:
            body = [" " * inner + params + "\n"]
        else:
            items = split_items(params)
            body = [" " * inner + item + ",\n" for item in items[:-1]]
            body.append(" " * inner + items[-1] + "\n")
        out += [f"{head} {lambda_match.group(1)}(\n", *body]
        out.append(f"{indent}){lambda_match.group(3)} {{\n")
        out += [
            line[INDENT:] if line.strip() else line for line in lines[i + 2 : end + 1]
        ]
        i = end + 1
    return out


DEFINE_VALUE = re.compile(r"(?P<value>[^;{}/]*?)\s*/\*(?P<note>.*?)\*/")


def hoist_define_comments(lines):
    """A constant whose value and trailing comment clang-format had to wrap
    (``#define NAME \\`` / ``value /* note */``) written as the comment, then the
    ``#define NAME value`` on one line: the line stays whole for tools that read it."""
    out, i = [], 0
    while i < len(lines):
        head = re.fullmatch(r"#define\s+(\w+(?:\([^)]*\))?)\s*\\\n", lines[i])
        end = i
        while head and lines[end].rstrip("\n").endswith("\\") and end + 1 < len(lines):
            end += 1
        if not head or end == i:
            out.append(lines[i])
            i += 1
            continue
        body = " ".join(
            part.rstrip("\n").rstrip("\\").strip() for part in lines[i + 1 : end + 1]
        )
        found = DEFINE_VALUE.fullmatch(body)
        if not found or not found["value"].strip():
            out.append(lines[i])
            i += 1
            continue
        note = " ".join(found["note"].split())
        wrapped = textwrap.wrap(
            note, COLUMNS - 6, break_long_words=False, break_on_hyphens=False
        )
        wrapped[0] = "/* " + wrapped[0]
        out += [
            f"{line}\n" if number == 0 else f"   {line}\n"
            for number, line in enumerate(wrapped)
        ]
        out[-1] = out[-1].rstrip("\n") + " */\n"
        out.append(f"#define {head.group(1)} {found['value'].strip()}\n")
        i = end + 1
    return out


def house_rules(lines, binary):
    """Argument comments, empty loops, braced lists and value wraps on clang-format's
    output."""
    fixed = []
    lines = hoist_define_comments(lines)
    for i, line in enumerate(lines):
        argument_line = OWN_LINE_ARG.match(line)
        if argument_line:
            indent, name, value, comma = argument_line.groups()
            line = f"{indent}{value}{comma}  // {name}\n"
        fixed.append(line)
        loop = EMPTY_LOOP.match(line)
        if loop and i + 1 < len(lines) and lines[i + 1].strip() == "}":
            fixed.append(f"{loop.group(1)}{' ' * INDENT};\n")
    lines = wrap_lambda_values(reindent_brackets(join_member_calls(fixed)))
    lines = block_hugged_brackets(lines, binary)
    lines = wrap_values(block_braced_lists(hug_braced_lists(lines)), binary)
    for _ in range(8):
        lines = reindent_brackets(lines)
        laid = wrap_ternaries(block_hugged_brackets(lines, binary), binary)
        if laid == lines:
            break
        lines = laid
    return reindent_brackets(lines)


def format_text(text, binary, passes=4):
    """The formatted text of one source file: the passes repeated to a fixed point."""
    for _ in range(passes):
        lines = clang_format(text, binary).splitlines(keepends=True)
        formatted = "".join(house_rules(lines, binary))
        if formatted == text:
            break
        text = formatted
    return text


def is_ignored(path, root):
    """True when `path` matches a pattern of <root>/.clang-format-ignore."""
    try:
        lines = (root / ".clang-format-ignore").read_text().splitlines()
    except OSError:
        return False
    try:
        rel = Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    patterns = [x.strip() for x in lines if x.strip() and not x.startswith("#")]
    return any(
        fnmatch.fnmatch(rel, pat) or rel.startswith(pat.rstrip("/") + "/")
        for pat in patterns
    )


def sources(paths):
    """C / C++ files under `paths` or, by default, the repository's, outside out/.

    Files matching .clang-format-ignore are skipped, unless named on the command line.
    """
    root = repo_root()
    if paths:
        found = []
        for path in map(Path, paths):
            if path.is_dir():
                found += sorted(
                    file
                    for file in path.rglob("*")
                    if file.suffix in SUFFIXES and not is_ignored(file, root)
                )
            else:
                found.append(path)
        return found
    listed = subprocess.run(
        ["git", "ls-files", "-z", "-co", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    ).stdout.split("\0")
    return [
        root / name
        for name in listed
        if Path(name).suffix in SUFFIXES
        and not name.startswith("out/")
        and (root / name).is_file()
        and not is_ignored(root / name, root)
    ]


def main():
    """Formats files in place; --check lists files that would change (exit 1)."""
    global CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--config", help="clang-format style file (default: root's)")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    CONFIG = args.config
    style = config_file()
    if not style.is_file():
        print(
            f"cxx_format: style file {style} not found (use --config)", file=sys.stderr
        )
        return 2
    binary = clang_format_binary()
    try:
        files = sources(args.paths)
    except (OSError, subprocess.CalledProcessError):
        print(
            f"cxx_format: cannot list files in {repo_root()}; pass paths",
            file=sys.stderr,
        )
        return 2
    if files and binary is None:
        print("clang-format not found (pip install clang-format)", file=sys.stderr)
        return 2
    changed = []
    for path in files:
        text = path.read_text()
        formatted = format_text(text, binary)
        if formatted != text:
            changed.append(path)
            if not args.check:
                path.write_text(formatted)
    for path in changed:
        print(("would reformat " if args.check else "reformatted ") + str(path))
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
