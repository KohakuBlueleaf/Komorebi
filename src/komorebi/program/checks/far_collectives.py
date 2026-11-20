"""The collectives over islands (km::all_reduce, reduce_to, all_gather, broadcast_from)
in a mover's source (comments and strings aside, quoted includes too): whether a load's
movers call one, for the refusal after a give-up and, on KM_LINK_NO_FAR_DRAM chips, the
far leader's L1 landing (km_ask.hpp land_run) or a host collective's refusal."""

import re

from komorebi.program.checks import page_calls

HAZARD = "docs/reference/hazards.md#a-link-packet-for-a-far-wormhole-chips-dram"
RULE = "On Wormhole no link packet targets a DRAM address of the far chip"
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


def refusal(what):
    """The refusal of `what`, a collective over islands on such a chip."""
    return (
        f"{what} writes a copy's pages into the other islands' DRAM: {RULE} ({HAZARD})"
    )


def calls(programs):
    """Whether a mover of `programs` calls a collective, on any chip (per load: the
    host does not know which handler does)."""
    return any(
        scan(source) for program in programs for source in page_calls.sources(program)
    )


def wanted(programs):
    """Whether a mover of `programs` (an island build with KM_LINK_NO_FAR_DRAM) calls a
    collective: the load then makes the collectives' landing."""
    return any(
        scan(source)
        for program in programs
        if "KM_LINK_NO_FAR_DRAM" in program.defines
        for source in page_calls.sources(program)
    )
