"""The near build (KM_ISLAND_NEAR): an island program whose writer is over the chip's
code memory at the highest writer level is built again without the far paths when the
load proves it never needs them (`refusal` None); a far page, put, ask, collective or
chain gather gives up there by name, unreachable by the host's checks (`near`)."""

import re

from komorebi.program.checks import far_collectives

DEFINE = "KM_ISLAND_NEAR"
# the island defines its movers are built without (the link cores keep them): the
# page links' client side (km_far.hpp's lines, a command's settle of its puts)
DROPPED = ("KM_PAGE_LINKS", "KM_LINK_NO_FAR_DRAM", "KM_LINK_THIN_SECOND")
# the calls a near build gives up: the collectives and the lines that name an island
NAMES = re.compile(
    r"\bkm::(all_reduce|reduce_to|all_gather|broadcast_from|put_pages|put_piece"
    r"|ask_pages|ask_line|ask_line_at|ask_bytes_line|put_batch_send)\s*\("
)


def applies(programs):
    """Whether a load of `programs` is an island load (its tensor table)."""
    return any("KM_TENSOR_TABLE" in program.defines for program in programs)


def refusal(programs):
    """Why `programs` (one load's) are not built without the far paths: None when
    they may be (each a class's program of an island load of whole cards, its link
    cores apart, its make's island tensors copies, no X280 thread, its chain without
    a gather, its movers' sources naming none of NAMES)."""
    for program in programs:
        if "KM_TENSOR_TABLE" not in program.defines:
            return f"{program.name} is not a program of several islands"
        if "KM_WIRE" in program.defines:
            return f"{program.name} runs on a card split into islands"
        if program.messages is not None:
            return f"{program.name} is dynamic: its link cores carry its messages"
        if getattr(program, "threads", None):
            return f"{program.name}'s X280 threads send commands the host never checks"
        given = getattr(program, "make_given", None)
        if given is None:
            return (
                f"{program.name}'s kernels are given by hand: the host cannot tell "
                "which of its command words are tensors"
            )
        if "KM_CHAIN_GATHER" in program.defines:
            return f"{program.name}'s chain gathers (km.all_gather in a chain)"
        for name, tensor in (given.get("tensors") or {}).items():
            placement = getattr(tensor, "placement", None)
            if placement is not None and not placement.copy:
                return f"{program.name}'s tensor {name} at its make is not a copy"
        for source in far_collectives.page_calls.sources(program):
            calls = far_collectives.scan(source, NAMES)
            if calls:
                path, line, name = calls[0]
                return f"{program.name}: {path}:{line} calls {name}"
    return None


def line(far, most):
    """The words of a near build whose far build's writer was `far` bytes of `most`:
    "built <line>" in the load's listing and its refusals."""
    return (
        f"without the far paths (with them the writer is {far:,} B of {most:,} B at "
        "the highest writer level)"
    )
