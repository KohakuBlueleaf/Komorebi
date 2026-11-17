"""km.all_gather in a class's chain: its record's description and checks, and a
chain method's line that states it."""

import ast

from komorebi.chain import calls as kc
from komorebi.dsl.chain.described import Described, writes_key
from komorebi.dsl.front import front
from komorebi.dsl.front.names import spelling


def gathered(run, calls, index, described, group, starts):
    """km.all_gather's description: one range of its copy, the record's cores one
    rank of each island (island 0's last to the last island's first; one island: the
    chain's first rank), its block the copy's pages 0 to first + pages - 1."""
    call = calls[index]
    label = f"call {index} (km.{spelling('all_gather')})"
    program = call.program
    for other in calls if program is None else ():
        if isinstance(other, front.HandlerCall):
            program = getattr(other.handler, "program", None)
            program = program or run.programs[run._handler(other.handler)[1]]
            break
    tensor, first, pages = call.tensor, call.first, call.pages
    if not front.is_tensor(tensor) or pages < 1 or first + pages > tensor.pages:
        raise kc.ChainRefused(
            f"{label}: pages {first} to {first + pages - 1} of a tensor's pages"
        )
    if len(starts) > 1 and not getattr(getattr(tensor, "placement", None), "copy", 0):
        raise kc.ChainRefused(f"{label}: its tensor is a copy (copy=True)")
    most = kc.words()["kChainWidthMask"]
    if first + pages > most:
        raise kc.ChainRefused(
            f"{label}: pages to {first + pages - 1}; a gather's record holds pages 0 "
            f"to {most - 1} (its block's width)"
        )
    for at, other in enumerate(described[: index - 1]):
        if writes_key(other, id(tensor)):
            raise kc.ChainRefused(
                f"{label}: call {at} ({other.name}) writes its copy; a gather's pages "
                "are written by the call just before it, or before the chain"
            )
    cores = (group[0], 1)
    if len(starts) > 1:
        cores = (starts[1] - 1, starts[-1] - starts[1] + 2)
        if not group[0] <= cores[0] < cores[0] + cores[1] <= group[0] + group[1]:
            raise kc.ChainRefused(
                f"{label}: the chain's cores {group[0]} to {group[0] + group[1] - 1} "
                f"reach not every island (their first ranks {list(starts)}); a gather "
                f"takes ranks {cores[0]} to {cores[0] + cores[1] - 1}"
            )
    if cores[1] > kc.words()["kChainSizeMask"]:
        raise kc.ChainRefused(
            f"{label}: its record spans ranks {cores[0]} to {cores[0] + cores[1] - 1}; "
            f"a record's group is at most {kc.words()['kChainSizeMask']} ranks"
        )
    one = Described(program, [tensor.address, first, pages], cores, cores[1], 0)
    one.width, one.output, one.program = first + pages, tensor, program
    one.name = f"km.{spelling('all_gather')}"
    one.reads = {id(tensor): (tensor, None, [kc.Range(first, 0, pages)], [])}
    one.gather = tuple(starts)
    return one


def is_gather(call):
    """Whether a chain method's line `call` is km.all_gather(...)."""
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id != "self"
        and func.attr == spelling("all_gather")
    )
