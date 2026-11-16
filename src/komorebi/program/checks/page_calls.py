"""tt-metal's own page calls in an island build's kernel reach this island's DRAM only:
a load with the tensor table refuses a mover's source line naming one (comments and
strings aside) unless it says `// km: own island` (this core's island's bank) or the
program says all do (OWN); a marker on a line with no such call is refused too."""

import re
from pathlib import Path

from komorebi import _env
from komorebi.errors import ProgramError

MARKER = "// km: own island"  # the way out, on the call's own line
OWN = "own_island_pages"  # its variant: Program(own_island_pages=True), every call
MARKED = re.compile(r"//[ \t]*km:[ \t]*own island[ \t]*$", re.M)
NAMES = re.compile(
    r"\b(noc_async_(?:read|write)_(?:page|tile)|Interleaved(?:Pow2)?AddrGen\w*)\b"
)
COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
STRINGS = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
LOCAL = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.M)
MOVERS = ("reader", "writer", "movement")


def _blank(match):
    return "\n" * match.group(0).count("\n")


def _ours(path):
    """Whether `path` is Komorebi's own (its headers and kernels): not scanned."""
    return _env.include_directory().parent in path.parents


def _lines(text, pattern):
    """The 1-based lines of `text` where `pattern` matches: {line: first match}."""
    out = {}
    for hit in pattern.finditer(text):
        out.setdefault(text.count("\n", 0, hit.start()) + 1, hit)
    return out


def scan(source):
    """(calls, refused) of `source` and the files it includes by a quoted path beside
    it, recursively (Komorebi's own files aside): calls [(file, line, name)] marked
    with MARKER; refused [(file, line, name)], an unmarked call or (name None) a
    marker on a line with no call."""
    seen, todo, calls, refused = set(), [Path(source).resolve()], [], []
    while todo:
        path = todo.pop(0)
        if path in seen or _ours(path) or not path.is_file():
            continue
        seen.add(path)
        raw = path.read_text(errors="replace")
        marked = set(_lines(raw, MARKED))
        text = COMMENTS.sub(_blank, raw)
        todo += [(path.parent / name).resolve() for name in LOCAL.findall(text)]
        named = _lines(STRINGS.sub(_blank, text), NAMES)
        for line, hit in sorted(named.items()):
            (calls if line in marked else refused).append((path, line, hit.group(1)))
        refused += [(path, line, None) for line in sorted(marked - set(named))]
    return calls, refused


def found(source):
    """The first refused (file, line, name) of scan(source); None: none."""
    refused = scan(source)[1]
    return refused[0] if refused else None


def sources(program):
    """The mover sources of `program` a user wrote: its kernels' and its overlays'."""
    out = []
    if program.messages is None or not program.messages.parts:
        paths = program.sources()
        out += [paths[kernel] for kernel in ("reader", "writer")]
    for part in program.overlays.values():
        out += [path for kernel, path in part.sources.items() if kernel in MOVERS]
    return list(dict.fromkeys(Path(path) for path in out))


def check(programs, log=None):
    """Refuses a program of an island build (KM_TENSOR_TABLE) whose movers call
    tt-metal's page calls on a line without MARKER (unless the program says OWN: all
    taken as marked), or carry a stale MARKER; marked calls logged once a program."""
    for program in programs:
        if "KM_TENSOR_TABLE" not in program.defines:
            continue
        meant, own = [], getattr(program, OWN, False)
        for source in sources(program):
            calls, refused = scan(source)
            meant += calls + [hit for hit in refused if own and hit[2] is not None]
            refused = [hit for hit in refused if not own or hit[2] is None]
            if not refused:
                continue
            path, line, name = refused[0]
            if name is None:
                raise ProgramError(
                    f"{program.name}: {path}:{line}: `{MARKER}` on a line that names "
                    "none of tt-metal's page calls: move it to the call's line or "
                    "drop it"
                )
            raise ProgramError(
                f"{program.name}: {path}:{line}: {name} reaches this island's DRAM "
                f"only; say `{MARKER}` on the line when the call means this core's "
                f"own island's bank (Program({OWN}=True): every call of the "
                "program), or use km::read_pages / km::write_pages with a km::Tensor "
                "(or km::BankRange / km::SizedTensor)"
            )
        if meant and log is not None:
            path, line, name = meant[0]
            said = f" or taken so by {OWN}=True" if own else ""
            log(
                f"{program.name}: {len(meant)} page call(s) marked `{MARKER}`{said} "
                f"(the first {path.name}:{line}: {name}): this island's bank"
            )
