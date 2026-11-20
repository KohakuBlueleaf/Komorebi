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
    its places cover, ``(file, number, line as written)``; ranges of the text, each
    ``(first line, its column, last line, its column, file, first, last)``, columns of
    characters from 1, both ends held; and the renames, ``(safe name, source name)``."""

    files: list[str]
    lines: list[tuple[int, int, str]]
    map: list[tuple[int, int, int, int, int, int, int]]
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


def layout(source: str, origin: Origin) -> tuple[str, Origin]:
    """A frontend's text in the canonical layout, its ``origin``'s ranges moved to the
    lines of the text as returned; a refused origin raises."""
    text, ranges = _rs.layout(source, _given(origin))
    return text, Origin(*origin)._replace(map=ranges)


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
    """What a host invokes: ``Program.handler``, kind, number, symbol, parameter kinds,
    words (``("address" | "tiles" | "word", p)``), the ranges a host checks, ``(tensor,
    first, count, cols)`` (a rectangle's rows, ``cols`` its columns, else None); an end
    ``("word" | "const", n)``, ``("rank" | "size",)`` or ``(op, a, b)``, ``+ - * / %``.
    """

    name: str
    kind: str
    number: int
    symbol: str
    params: list[str]
    words: list[tuple[str, int]]
    bounds: list[tuple]
    patterns: list[int]  # the parameters whose word is a pattern's, sent only valid


class Kernel(NamedTuple):
    """A kernel the machine's runtime compiles at load: its source (a file of the
    output), its processor, defines, compile-time arguments (``("word", n)``, or
    ``("layout", parameter)`` filled at load) and compile options (a bool or a text)."""

    source: str
    processor: str
    defines: list[tuple[str, str]]
    args: list[tuple[str, int]]
    options: list[tuple[str, bool | str]]


class Buffer(NamedTuple):
    """A buffer of each core's memory the kernels pass tiles through; ``pair`` the
    other half's index when it is one half of a stream's pair, one for each mover;
    ``unpack`` how compute unpacks it to Dst when not the default (tt-metal's name)."""

    index: int
    format: str
    page_bytes: int
    pages: int
    pair: int | None
    unpack: str | None


class Step(NamedTuple):
    """A program run to build the output, in the folder its files are written to."""

    program: str
    args: list[str]


class Emitted(NamedTuple):
    """A backend's output: its cache key (hex), its files ``(name, bytes)``, its entry
    points, and how it is built: kernels and buffers for the machine's runtime, or
    steps run here; ``at_load`` the kernels' files the load makes, by name (their
    text needs the load's tensors: ``Loaded.files``), which are not among ``files``."""

    key: str
    files: list[tuple[str, bytes]]
    entries: list[EntryPoint]
    kernels: list[Kernel]
    buffers: list[Buffer]
    steps: list[Step]
    at_load: list[str]


def emit(
    program: str | bytes, backend: str = "tensix", target: str | None = None
) -> Emitted:
    """The output of a module on a backend (``"tensix"`` or ``"cpu"``); ``target`` its
    architecture, by default the backend's (no chip for ``"tensix"``, the host's for
    ``"cpu"``). Nothing is written or run."""
    key, files, entries, kernels, buffers, steps, at_load = _rs.emit(
        program, backend, target
    )
    return Emitted(
        key,
        [(name, data) for name, data in files],
        [
            EntryPoint(*e[:4], list(e[4]), list(e[5]), list(e[6]), list(e[7]))
            for e in entries
        ],
        [Kernel(s, p, list(d), list(a), list(o)) for s, p, d, a, o in kernels],
        [Buffer(*b) for b in buffers],
        [Step(p, list(a)) for p, a in steps],
        list(at_load),
    )


class LoadTensor(NamedTuple):
    """A tensor a load is given for a tensor parameter: its pages, its layout as the
    kernels' compile arguments take it (tensors of one layout share a kind), the
    bytes of a page in DRAM when a thread may fetch and store it, and its shape in
    tiles ``(rows, columns)``, which a handler that streams a rectangle of it needs."""

    pages: int
    layout: list[int]
    dram_page: int | None = None
    shape: tuple[int, int] | None = None


class LoadedKernel(NamedTuple):
    """A kernel as a load hands it to the runtime: its compile arguments are words,
    each tensor's layout in its place."""

    source: str
    processor: str
    defines: list[tuple[str, str]]
    args: list[int]
    options: list[tuple[str, bool | str]]


class LoadBuffer(NamedTuple):
    """A buffer as a load makes it: its format by the loader's name (``bfloat16``),
    ``unpack`` as in ``Buffer``."""

    index: int
    dtype: str
    page_bytes: int
    pages: int
    pair: int | None
    unpack: str | None


class Islands(NamedTuple):
    """A machine's islands as a graph: the workers each holds, in order (island 0 the
    first ``sizes[0]``), and the connections between them, by number, each both ways;
    None: every pair of islands connected (an empty list of several: not connected)."""

    sizes: list[int]
    links: list[tuple[int, int]] | None = None


class Machine(NamedTuple):
    """The machine a load is on: its workers, a worker's memory in bytes, its islands
    and the load's named options (``depth``, ``max_live``)."""

    workers: int
    worker_memory: int
    islands: Islands
    options: list[tuple[str, int]]


class Loaded(NamedTuple):
    """What a load on Tensix makes from its tensors: the thread header (its name and
    text), the kernels, the buffers, the program's header as the machine sizes it
    (each reduce's arrival from its islands), in place of the compile's, and the
    files the output names ``at_load`` ``(name, text)``."""

    thread_name: str
    thread: str
    kernels: list[LoadedKernel]
    buffers: list[LoadBuffer]
    header_name: str
    header: str
    files: list[tuple[str, str]]


def load(
    program: str | bytes,
    tensors: dict[str, LoadTensor],
    chip: str | None = None,
    machine: Machine | None = None,
) -> Loaded:
    """A load of a module on Tensix given a tensor for each tensor parameter, by name;
    ``chip`` (``blackhole``, ``wormhole_b0``) checked when named; on ``machine`` (None:
    one island holding every worker). Nothing is written or run."""
    rows = [
        (name, (t.pages, list(t.layout), t.dram_page, t.shape))
        for name, t in tensors.items()
    ]
    given = None
    if machine is not None:
        links = machine.islands.links
        links = None if links is None else [tuple(link) for link in links]
        islands = (list(machine.islands.sizes), links)
        given = (machine.workers, machine.worker_memory, islands, list(machine.options))
    name, text, kernels, buffers, header_name, header, files = _rs.load(
        program, rows, chip, given
    )
    return Loaded(
        name,
        text,
        [LoadedKernel(s, p, list(d), list(a), list(o)) for s, p, d, a, o in kernels],
        [LoadBuffer(*b) for b in buffers],
        header_name,
        header,
        [(file, data) for file, data in files],
    )


__all__ = [
    "Buffer",
    "Diagnostic",
    "Emitted",
    "EntryPoint",
    "Hashes",
    "IrError",
    "Islands",
    "Kernel",
    "LoadBuffer",
    "LoadTensor",
    "Loaded",
    "LoadedKernel",
    "Machine",
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
    "layout",
    "load",
    "requirements",
    "run",
]
