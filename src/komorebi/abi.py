"""The definitions the host shares with the kernels, read from their C headers: record
and event layout, start-up argument order, state words, the dispatch kernel's layout
and codes. Pure functions; reading happens on first use.
"""

import ast
import functools
import operator
import re
from pathlib import Path

from komorebi import _env

# a comment, or a string or character literal (kept, so that "/*" in one is not one)
LEXEME = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S
)
DIRECTIVE = re.compile(r"^\s*#\s*(\w+)\s*(.*?)\s*$")
DEFINE = re.compile(r"^\s*#\s*define\s+(KMD?_\w+)(\(?)(.*)$", re.M)
CONSTEXPR = re.compile(r"\bconstexpr\s+uint32_t\s+(k\w+)\s*=\s*([^;]*);")
ANY_CONSTEXPR = re.compile(r"\bconstexpr\b[^;{]*")
ENUM = re.compile(r"\benum\s*\{([^{}]*)\}\s*;")
ANY_ENUM = re.compile(r"\benum\b")
NUMBER = re.compile(r"\b\d\w*")
C_NUMBER = re.compile(r"(0[xX][0-9A-Fa-f]+|0|[1-9][0-9]*)")
SUFFIX = re.compile(
    r"\b(0[xX][0-9A-Fa-f]+|\d+)(?:[uU](?:ll|LL|l|L)?|(?:ll|LL|l|L)[uU]?)\b"
)
SIZEOF = re.compile(r"\bsizeof\s*\(\s*(\w+)\s*\)")
FIELD = re.compile(r"^uint32_t\s+(\w+(?:\s*,\s*\w+)*)$")
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
}


class HeaderError(ValueError):
    """A header holds a definition the host cannot read, or two that disagree."""


ARG_WORDS = [f"a{i}" for i in range(10)]
M32 = 0xFFFFFFFF


class Headers:
    """The headers of one native tree: include/ and the dispatch kernel's directory."""

    def __init__(self, include, kernel):
        self.abi = Path(include) / "komorebi" / "abi" / "km_abi.h"
        self.core = Path(include) / "komorebi" / "device" / "km_core.hpp"
        self.dispatch = Path(kernel) / "km_dispatch.h"

    @functools.cached_property
    def constants(self):
        """Every KM_* and KMD_* define, constexpr k*, and enum value of the three."""
        sizes = {"Args": 4 * len(self.arg_fields)}
        return constants(self.abi, self.core, self.dispatch, sizes=sizes)

    @functools.cached_property
    def record_fields(self):
        """The 16 words of km_record_t, in order."""
        return struct_fields(self.abi, "km_record_t")

    @functools.cached_property
    def event_fields(self):
        return struct_fields(self.abi, "km_event_t")

    @functools.cached_property
    def arg_fields(self):
        """The start-up arguments of km::Args, in order."""
        return struct_fields(self.core, "Args")

    @functools.cached_property
    def positions(self):
        """Word index in a record of hdr, team and a0 to a9."""
        index = {name: i for i, name in enumerate(self.record_fields)}
        return index["hdr"], index["team"], [index[a] for a in ARG_WORDS]

    def __getitem__(self, name):
        return self.constants[name]


def evaluate(expr, known, sizes=None, where=""):
    """The value of a C integer expression of C literals, known names, sizeof(struct)
    and + - * / << >> | &, where every value stays in 0..2**32 - 1 (so C's unsigned
    arithmetic gives the same); anything else raises HeaderError."""
    what = f"{where}: cannot read {expr.strip()!r}"
    text = SIZEOF.sub(
        lambda m: str((sizes or {}).get(m[1], f"sizeof_{m[1]}")), expr.strip()
    )
    text = SUFFIX.sub(r"\1", text)
    if any(not C_NUMBER.fullmatch(n) for n in NUMBER.findall(text)):
        raise HeaderError(f"{what}: not a C integer literal")
    try:
        tree = ast.parse(text.replace("/", "//"), mode="eval")
    except SyntaxError as err:
        raise HeaderError(what) from err

    def checked(v):
        if not 0 <= v <= M32:
            raise HeaderError(f"{what}: {v} is outside 0..2**32 - 1")
        return v

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return checked(node.value)
        if isinstance(node, ast.Name) and node.id in known:
            return checked(known[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            a, b = value(node.left), value(node.right)
            if type(node.op) in (ast.LShift, ast.RShift) and b > 31:
                raise HeaderError(f"{what}: shift by {b}")
            if type(node.op) is ast.FloorDiv and b == 0:
                raise HeaderError(f"{what}: division by zero")
            return checked(OPERATORS[type(node.op)](a, b))
        raise HeaderError(what)

    return value(tree.body)


def _add(found, name, value, where):
    if name in found and found[name] != value:
        raise HeaderError(f"{where}: {name} = {value} disagrees with {found[name]}")
    found[name] = value


def _strip(text):
    """The text with comments removed (string literals kept) and continued lines
    joined."""

    def keep(m):
        return m[0] if m[0][0] in "\"'" else " " * m[0].count("\n") or " "

    return LEXEME.sub(keep, text.replace("\\\n", " "))


def _condition(kind, rest, defined, after):
    """True or False for a conditional the reader can decide, else None: 0, 1, a
    name defined above, and an include guard (#ifndef X followed by #define X)."""
    name = re.fullmatch(r"!?\s*defined\s*\(?\s*(\w+)\s*\)?", rest)
    if kind in ("ifdef", "ifndef"):
        name, negate = re.fullmatch(r"(\w+)", rest), kind == "ifndef"
    else:
        negate = rest.startswith("!")
    if kind == "if" and rest in ("0", "1"):
        return rest == "1"
    if name is None:
        return None
    if name[1] in defined:
        return not negate
    if negate and re.match(rf"\s*#\s*define\s+{name[1]}\b", after):
        return True  # an include guard
    return None


def _clean(text, where="", known=()):
    """The active text of a header (`known`: names an earlier header defined), and
    the text of blocks whose condition it cannot decide (a constant there raises)."""
    lines = _strip(text).split("\n")
    active, unsure, stack, defined = [], [], [], set(known)
    for i, line in enumerate(lines):
        d = DIRECTIVE.match(line)
        kind, rest = (d[1], d[2]) if d else ("", "")
        if kind in ("if", "ifdef", "ifndef"):
            after = "\n".join(lines[i + 1 : i + 2])
            cond = _condition(kind, rest, defined, after)
            stack.append([cond, False])
            continue
        if kind in ("else", "elif", "endif"):
            if not stack:
                raise HeaderError(f"{where}: #{kind} without #if")
            if kind == "endif":
                stack.pop()
            elif kind == "else":
                cond = stack[-1][0]
                stack[-1][0] = None if cond is None else not cond
            else:
                stack[-1][0] = None  # #elif: not decided here
            continue
        if any(c is False for c, _ in stack):
            continue  # a disabled block
        target = unsure if any(c is None for c, _ in stack) else active
        target.append(line)
        if target is unsure and line.strip():
            active.append("@undecided;")  # a struct around it cannot be read
        if kind == "define" and target is active:
            defined.add(rest.split("(")[0].split()[0] if rest else "")
        elif kind == "undef" and target is active:
            defined.discard(rest)
    if stack:
        raise HeaderError(f"{where}: #if without #endif")
    return "\n".join(active), "\n".join(unsure)


def constants(*paths, sizes=None):
    """The named integer constants of the given headers, as a dict: every object-like
    `#define KM_*` / `KMD_*`, `constexpr uint32_t k*` and enum value. A definition that
    cannot be read, or a name defined twice with different values, raises HeaderError.
    """
    found = {}
    for path in paths:
        text, unsure = _clean(Path(path).read_text(), path, found)
        objects = [m for m in DEFINE.finditer(unsure) if not m[2]]
        if objects or ANY_CONSTEXPR.search(unsure) or "enum" in unsure:
            raise HeaderError(
                f"{path}: a constant inside an #if the host cannot decide"
            )
        for line in text.split("\n"):
            d = DIRECTIVE.match(line)
            if d and d[1] == "undef":
                found.pop(d[2], None)
            elif d and d[1] == "define":
                m = DEFINE.match(line)
                if m and not m[2]:  # function-like macros are not constants
                    _add(found, m[1], evaluate(m[3], found, sizes, f"{path}"), path)
        typed = CONSTEXPR.findall(text)
        if len(ANY_CONSTEXPR.findall(text)) != len(typed):
            other = [
                c for c in ANY_CONSTEXPR.findall(text) if not CONSTEXPR.match(c + ";")
            ]
            raise HeaderError(
                f"{path}: a constexpr that is not uint32_t k*: {other[:1]}"
            )
        for name, expr in typed:
            _add(found, name, evaluate(expr, found, sizes, f"{path}"), path)
        if len(ANY_ENUM.findall(text)) != len(ENUM.findall(text)):
            raise HeaderError(f"{path}: an enum other than `enum {{...}};`")
        for body in ENUM.findall(text):
            at = 0
            for item in filter(None, (i.strip() for i in body.split(","))):
                name, _, expr = item.partition("=")
                name = name.strip()
                at = evaluate(expr, found, sizes, f"{path}") if expr else at
                _add(found, name, at, path)
                at += 1
    return found


def struct_fields(path, name):
    """Field names in order of `typedef struct {...} name;` or `struct name {...};`.

    Every member must be a list of plain `uint32_t` names; anything else raises.
    """
    text, _ = _clean(Path(path).read_text(), path)
    found = re.findall(
        rf"typedef\s+struct\s*\{{([^{{}}]*)\}}\s*{name}\s*;", text
    ) + re.findall(rf"\bstruct\s+{name}\s*\{{([^{{}}]*)\}}\s*;", text)
    if not found:
        raise HeaderError(f"no struct {name} in {path}")
    if len(found) > 1:
        raise HeaderError(f"{path}: struct {name} defined {len(found)} times")
    body = [None, found[0]]
    fields = []
    for decl in filter(None, (d.strip() for d in body[1].split(";"))):
        member = FIELD.match(" ".join(decl.split()))
        if member is None:
            raise HeaderError(f"{path}: struct {name}: cannot read member {decl!r}")
        fields += [f.strip() for f in member[1].split(",")]
    if len(set(fields)) != len(fields):
        raise HeaderError(f"{path}: struct {name}: a member name is used twice")
    return fields


@functools.cache
def headers():
    """The headers beside the package (a checkout)."""
    include = _env.include_dir()
    if include is None:
        raise FileNotFoundError("the native headers are not beside the package")
    kernel = include.parent / "x280" / "kernel"
    return Headers(include, kernel)


def record(seg, first, count, args=(), flags=0, h=None):
    """The 16 words of a record; the dispatcher fills seq, slot and landing words."""
    h = h or headers()
    hdr, team, arg_at = h.positions
    if len(args) > len(arg_at):
        raise ValueError(f"{len(args)} arguments; a record has {len(arg_at)}")
    words = [0] * 16
    words[hdr] = (seg & 0xFF) | (flags & 0xFF) << 8
    words[team] = (first & 0xFFFF) | (count & 0xFFFF) << 16
    for at, value in zip(arg_at, args):
        words[at] = int(value) & M32
    return words


def start_args(values, h=None):
    """The start-up arguments in the order of km::Args, from a dict of every field."""
    h = h or headers()
    missing = set(h.arg_fields) - set(values)
    if missing:
        raise ValueError(f"missing start-up arguments: {sorted(missing)}")
    extra = set(values) - set(h.arg_fields)
    if extra:
        raise ValueError(f"unknown start-up arguments: {sorted(extra)}")
    return [int(values[f]) & M32 for f in h.arg_fields]


def column_groups(grid, cores):
    """Start-up groups of cores 0 to cores - 1: {core: (leader, rank, members, slot)}.

    The lowest core of each column leads and reports READY in the column's slot.
    """
    gx = grid[0]
    out = {}
    for x in range(gx):
        column = list(range(x, cores, gx))
        if column:
            out[column[0]] = (column[0], 0, len(column) - 1, x)
            for rank, core in enumerate(column[1:]):
                out[core] = (column[0], rank, 0, x)
    return out


def limit_units(seconds, mhz):
    """A bound in the kernels' units of 1024 cycles at `mhz`, at least 1."""
    return max(1, min(M32, int(seconds * mhz * 1e6) >> 10))
