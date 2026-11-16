"""tt-metal's own page calls in a kernel of an island build: they read and write this
island's DRAM only, so a load with the tensor table refuses a mover's source that names
one (comments and strings aside), unless its program says it means its own island's
bank (OWN, a Program keyword)."""

import re
from pathlib import Path

from komorebi import _env
from komorebi.errors import ProgramError

OWN = "own_island_pages"  # the way out: Program(own_island_pages=True)
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


def found(source):
    """(file, line, name) of the first of NAMES in `source` or a file it includes by a
    quoted path beside it, recursively (Komorebi's own files aside); None: none."""
    seen, todo = set(), [Path(source).resolve()]
    while todo:
        path = todo.pop(0)
        if path in seen or _ours(path) or not path.is_file():
            continue
        seen.add(path)
        text = COMMENTS.sub(_blank, path.read_text(errors="replace"))
        todo += [(path.parent / name).resolve() for name in LOCAL.findall(text)]
        hit = NAMES.search(STRINGS.sub(_blank, text))
        if hit is not None:
            return path, text.count("\n", 0, hit.start()) + 1, hit.group(1)
    return None


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
    tt-metal's page calls; one with OWN set is logged once and built."""
    for program in programs:
        if "KM_TENSOR_TABLE" not in program.defines:
            continue
        for source in sources(program):
            hit = found(source)
            if hit is None:
                continue
            path, line, name = hit
            if getattr(program, OWN, False):
                if log is not None:
                    log(
                        f"{program.name}: {path.name}:{line}: {name} reaches this "
                        f"island's DRAM only ({OWN}=True: meant)"
                    )
                break
            raise ProgramError(
                f"{program.name}: {path}:{line}: {name} reaches this island's DRAM "
                "only; on islands use km::read_pages / km::write_pages with a "
                "km::Tensor (or km::BankRange / km::SizedTensor); a kernel that means "
                f"its own island's bank: Program({OWN}=True)"
            )
