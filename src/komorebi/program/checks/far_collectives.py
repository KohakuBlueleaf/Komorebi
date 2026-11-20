"""The collectives over islands in a mover's source (km::all_reduce, km::reduce_to,
km::all_gather, km::broadcast_from): whether a load's movers call one, its source lines
looked at (comments and strings aside, quoted includes beside it too)."""

import re

from komorebi.program.checks import page_calls

NAMES = re.compile(r"\bkm::(all_reduce|reduce_to|all_gather|broadcast_from)\s*\(")


def scan(source, names=NAMES):
    """[(file, line, name)] of the collective calls (or of `names`') in `source` and
    the files it includes by a quoted path beside it, recursively (Komorebi's own files
    aside)."""
    seen, todo, out = set(), [source.resolve()], []
    while todo:
        path = todo.pop(0)
        if path in seen or page_calls._ours(path) or not path.is_file():
            continue
        seen.add(path)
        text = page_calls.COMMENTS.sub(
            page_calls._blank, path.read_text(errors="replace")
        )
        todo += [
            (path.parent / name).resolve() for name in page_calls.LOCAL.findall(text)
        ]
        named = page_calls._lines(
            page_calls.STRINGS.sub(page_calls._blank, text), names
        )
        out += [
            (path, line, f"km::{hit.group(1)}") for line, hit in sorted(named.items())
        ]
    return out


def calls(programs):
    """Whether a mover of `programs` calls a collective, on any chip (per load: the
    host does not know which handler does)."""
    return any(
        scan(source) for program in programs for source in page_calls.sources(program)
    )
