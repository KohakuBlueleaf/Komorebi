"""The extension module: the Rust core's Python face (src/komorebi-ir/bindings/)."""

# One diagnostic: code, message, line, column, severity, byte, place.
Row = tuple[
    str, str, int | None, int | None, str, int | None, tuple[str, int, int] | None
]
# A frontend's origin: files, lines, map, renames.
Given = tuple[
    list[str],
    list[tuple[int, int, str]],
    list[tuple[int, int, int, int, int]],
    list[tuple[str, str]],
]

class IrError(Exception):
    diagnostics: list[Row]

def format(source: str) -> str: ...
def check(source: str, origin: Given | None = None) -> list[Row]: ...
def compile(source: str, origin: Given | None = None) -> bytes: ...
def hashes(program: str | bytes) -> tuple[str, str, str]: ...
def requirements(program: str | bytes) -> list[tuple[str, list[str]]]: ...
def run(program: str | bytes, script: str) -> str: ...
def emit(
    program: str | bytes, backend: str = "tensix", target: str | None = None
) -> tuple[
    str,
    list[tuple[str, bytes]],
    list[tuple],
    list[tuple],
    list[tuple[int, str, int, int, int | None]],
    list[tuple[str, list[str]]],
]: ...
def load(
    program: str | bytes,
    tensors: list[tuple[str, tuple[int, list[int], int | None]]],
    chip: str | None = None,
    machine: (
        tuple[int, int, tuple[list[int], list[tuple[int, int]]], list[tuple[str, int]]]
        | None
    ) = None,
) -> tuple[
    str,
    str,
    list[tuple],
    list[tuple[int, str, int, int, int | None]],
]: ...
