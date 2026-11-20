"""Komorebi IR from Python, a thin face over the Rust core (``komorebi_ir._rs``): a
module's text (``str``) or byte-code file (``bytes``) checked, formatted, compiled,
hashed and run on the interpreter. A refusal raises ``IrError`` with its diagnostics.
"""

from typing import NamedTuple

from komorebi_ir import _rs

IrError = _rs.IrError


class Diagnostic(NamedTuple):
    """Why a text or a byte-code file is refused: a text's diagnostic has a line and a
    column (from 1, the column in characters); a byte-code file's refusal has the byte.
    """

    code: str
    message: str
    line: int | None
    column: int | None
    severity: str
    byte: int | None


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


def diagnostics(error: IrError) -> list[Diagnostic]:
    """The diagnostics an ``IrError`` carries."""
    return [Diagnostic(*row) for row in getattr(error, "diagnostics", [])]


def format(source: str) -> str:
    """The canonical text of a module's text."""
    return _rs.format(source)


def check(source: str) -> list[Diagnostic]:
    """Every diagnostic of a module's text; empty when it checks."""
    return [Diagnostic(*row) for row in _rs.check(source)]


def compile(source: str) -> bytes:
    """The byte-code program of a module's text."""
    return _rs.compile(source)


def hashes(program: str | bytes) -> Hashes:
    """The three hashes of a module's text or of a byte-code file."""
    return Hashes(*_rs.hashes(program))


def requirements(program: str | bytes) -> list[Requirement]:
    """The requirement set of a module's text or of a byte-code file."""
    return [Requirement(a, list(c)) for a, c in _rs.requirements(program)]


def run(program: str | bytes, script: str) -> str:
    """Runs a run script on the reference interpreter and gives the report's text."""
    return _rs.run(program, script)


__all__ = [
    "Diagnostic",
    "Hashes",
    "IrError",
    "Requirement",
    "check",
    "compile",
    "diagnostics",
    "format",
    "hashes",
    "requirements",
    "run",
]
