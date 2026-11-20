"""Komorebi IR from Python, a thin face over the Rust core (``komorebi_ir._rs``): a
module's text (``str``) or byte-code file (``bytes``) checked, formatted, compiled,
hashed, run on the interpreter and emitted by a backend. A refusal raises ``IrError``
with its diagnostics. A frontend's text is compiled with its ``Origin`` beside it.
"""

from typing import NamedTuple

from komorebi_ir import _rs

IrError = _rs.IrError


class Place(NamedTuple):
    """Where a frontend's source holds a line of its text: the file, and its first and
    last lines (from 1)."""

    file: str
    first: int
    last: int


class Diagnostic(NamedTuple):
    """Why a text or a byte-code file is refused: a text's diagnostic has a line and a
    column (from 1, the column in characters); a byte-code file's refusal has the byte;
    a frontend's text's diagnostic, the place of its source too.
    """

    code: str
    message: str
    line: int | None
    column: int | None
    severity: str
    byte: int | None
    place: Place | None = None


class Origin(NamedTuple):
    """Where a frontend's text came from (docs/ir/frontends.md): its files; the lines
    its places cover, ``(file, number, line as written)``; ranges of the text's lines,
    each ``(text first, text last, file, first, last)``; and the renames of values and
    parameters, ``(safe name, source name)``."""

    files: list[str]
    lines: list[tuple[int, int, str]]
    map: list[tuple[int, int, int, int, int]]
    renames: list[tuple[str, str]] = []


class Hashes(NamedTuple):
    """A module's three hashes, as hex: the program; the program and its hints (a
    backend's cache key); this exact file."""

    semantic: str
    build: str
    full: str


class Requirement(NamedTuple):
    """One requirement, as a ``requires`` block writes it, and the constructs that
    cause it (``requires`` when the block states it)."""

    ability: str
    causes: list[str]


def _diagnostic(row) -> Diagnostic:
    """A row of the core as a Diagnostic."""
    *head, place = row
    return Diagnostic(*head, Place(*place) if place is not None else None)


def _given(origin: Origin | None):
    """An Origin as the core takes it, a tuple of lists."""
    if origin is None:
        return None
    return tuple(list(part) for part in Origin(*origin))


def diagnostics(error: IrError) -> list[Diagnostic]:
    """The diagnostics an ``IrError`` carries."""
    return [_diagnostic(row) for row in getattr(error, "diagnostics", [])]


def format(source: str) -> str:
    """The canonical text of a module's text."""
    return _rs.format(source)


def check(source: str, origin: Origin | None = None) -> list[Diagnostic]:
    """Every diagnostic of a module's text; empty when it checks. With an ``origin``,
    each is placed in the frontend's source, and a refused origin raises."""
    return [_diagnostic(row) for row in _rs.check(source, _given(origin))]


def compile(source: str, origin: Origin | None = None) -> bytes:
    """The byte-code program of a module's text; with an ``origin``, the origin in its
    notes and the spans left out."""
    return _rs.compile(source, _given(origin))


def hashes(program: str | bytes) -> Hashes:
    """The three hashes of a module's text or of a byte-code file."""
    return Hashes(*_rs.hashes(program))


def requirements(program: str | bytes) -> list[Requirement]:
    """The requirement set of a module's text or of a byte-code file."""
    return [Requirement(a, list(c)) for a, c in _rs.requirements(program)]


def run(program: str | bytes, script: str) -> str:
    """Runs a run script on the reference interpreter and gives the report's text."""
    return _rs.run(program, script)


class EntryPoint(NamedTuple):
    """What a host invokes: ``Program.handler``, its kind, number, symbol, parameters'
    kinds, argument words (``("address" | "tiles" | "word", parameter)``) and the
    ranges a host checks before sending it, ``(tensor, first, count)``, each end
    ``("word", i)``, ``("const", n)`` or ``("add", a, b)``."""

    name: str
    kind: str
    number: int
    symbol: str
    params: list[str]
    words: list[tuple[str, int]]
    bounds: list[tuple]


class Kernel(NamedTuple):
    """A kernel the machine's runtime compiles at load: its source (a file of the
    output), its processor, defines, compile-time arguments (``("word", n)``, or
    ``("layout", parameter)`` filled at load) and compile options."""

    source: str
    processor: str
    defines: list[tuple[str, str]]
    args: list[tuple[str, int]]
    options: list[tuple[str, str]]


class Buffer(NamedTuple):
    """A buffer of each core's memory the kernels pass tiles through."""

    index: int
    format: str
    page_bytes: int
    pages: int


class Step(NamedTuple):
    """A program run to build the output, in the folder its files are written to."""

    program: str
    args: list[str]


class Emitted(NamedTuple):
    """A backend's output: its cache key (hex), its files ``(name, bytes)``, its entry
    points, and how it is built: kernels and buffers for the machine's runtime, or
    steps run here."""

    key: str
    files: list[tuple[str, bytes]]
    entries: list[EntryPoint]
    kernels: list[Kernel]
    buffers: list[Buffer]
    steps: list[Step]


def emit(
    program: str | bytes, backend: str = "tensix", target: str | None = None
) -> Emitted:
    """The output of a module on a backend (``"tensix"`` or ``"cpu"``); ``target`` its
    architecture, by default the backend's (no chip for ``"tensix"``, the host's for
    ``"cpu"``). Nothing is written or run."""
    key, files, entries, kernels, buffers, steps = _rs.emit(program, backend, target)
    return Emitted(
        key,
        [(name, data) for name, data in files],
        [EntryPoint(*e[:4], list(e[4]), list(e[5]), list(e[6])) for e in entries],
        [Kernel(s, p, list(d), list(a), list(o)) for s, p, d, a, o in kernels],
        [Buffer(*b) for b in buffers],
        [Step(p, list(a)) for p, a in steps],
    )


__all__ = [
    "Buffer",
    "Diagnostic",
    "Emitted",
    "EntryPoint",
    "Hashes",
    "IrError",
    "Kernel",
    "Origin",
    "Place",
    "Requirement",
    "Step",
    "check",
    "compile",
    "diagnostics",
    "emit",
    "format",
    "hashes",
    "requirements",
    "run",
]
