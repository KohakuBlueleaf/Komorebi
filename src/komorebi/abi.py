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

COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
DEFINE = re.compile(r"^\s*#\s*define\s+(KMD?_\w+)(\(?)(.*)$", re.M)
CONSTEXPR = re.compile(r"\bconstexpr\s+uint32_t\s+(k\w+)\s*=\s*([^;]*);")
ENUM = re.compile(r"\benum\s*\{([^{}]*)\}\s*;")
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
    """The value of a C integer expression of numbers, known names, sizeof(struct)
    and + - * / << >> | &; anything else raises HeaderError."""
    text = SIZEOF.sub(
        lambda m: str((sizes or {}).get(m[1], f"sizeof_{m[1]}")), expr.strip()
    )
    text = SUFFIX.sub(r"\1", text).replace("/", "//")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as err:
        raise HeaderError(f"{where}: cannot read {expr.strip()!r}") from err

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in known:
            return known[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](value(node.left), value(node.right))
        raise HeaderError(f"{where}: cannot read {expr.strip()!r}")

    return value(tree.body)


def _add(found, name, value, where):
    if name in found and found[name] != value:
        raise HeaderError(f"{where}: {name} = {value} disagrees with {found[name]}")
    found[name] = value


def _clean(text):
    """The text with comments removed and continued lines joined."""
    return COMMENT.sub(" ", text.replace("\\\n", " "))


def constants(*paths, sizes=None):
    """The named integer constants of the given headers, as a dict: every object-like
    `#define KM_*` / `KMD_*`, `constexpr uint32_t k*` and enum value. A definition that
    cannot be read, or a name defined twice with different values, raises HeaderError.
    """
    found = {}
    for path in paths:
        text = _clean(Path(path).read_text())
        for name, paren, rest in DEFINE.findall(text):
            if paren:
                continue  # a function-like macro: not a constant
            _add(found, name, evaluate(rest, found, sizes, f"{path}"), path)
        for name, expr in CONSTEXPR.findall(text):
            _add(found, name, evaluate(expr, found, sizes, f"{path}"), path)
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
    text = _clean(Path(path).read_text())
    body = re.search(
        rf"typedef\s+struct\s*\{{([^{{}}]*)\}}\s*{name}\s*;", text
    ) or re.search(rf"\bstruct\s+{name}\s*\{{([^{{}}]*)\}}\s*;", text)
    if body is None:
        raise HeaderError(f"no struct {name} in {path}")
    fields = []
    for decl in filter(None, (d.strip() for d in body[1].split(";"))):
        member = FIELD.match(" ".join(decl.split()))
        if member is None:
            raise HeaderError(f"{path}: struct {name}: cannot read member {decl!r}")
        fields += [f.strip() for f in member[1].split(",")]
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
