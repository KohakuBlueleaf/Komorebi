"""km.all_gather in a class's chain: its record's description and checks, and a
chain method's line that states it."""

import ast

from komorebi.chain import calls as kc
from komorebi.dsl.chain.described import Described, writes_key
from komorebi.dsl.front import front
from komorebi.dsl.front.names import spelling


def gathered(run, calls, index, described, group, starts):
    """km.all_gather's description: one range of its copy, the record's cores one
    rank of each island (island 0's last to the last island's first), its block the
    copy's pages 0 to first + pages - 1; one island: one_island's ranks."""
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
    if len(starts) > 1 and (first > kc.RANGE_BASE or pages > kc.RANGE_WIDTH):
        raise kc.ChainRefused(
            f"{label}: pages {first} to {first + pages - 1}; a gather over islands "
            f"starts at page {kc.RANGE_BASE} at most and takes {kc.RANGE_WIDTH} pages "
            "at most (its record's range word)"
        )
    for at, other in enumerate(described[: index - 1]):
        if writes_key(other, id(tensor)):
            raise kc.ChainRefused(
                f"{label}: call {at} ({other.name}) writes its copy; a gather's pages "
                "are written by the call just before it, or before the chain"
            )
    if len(starts) == 1:
        return one_island(program, tensor, (first, pages), group, starts, label)
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


def one_island(program, tensor, gathered_pages, group, starts, label):
    """One island's gather, a plain record: n = ceil((first + pages) / 255) ranks of
    size 1 from the chain's first, rank r a root of pages r W to r W + W - 1."""
    first, pages = gathered_pages
    end, most = first + pages, kc.words()["kChainWidthMask"]
    n = -(-end // most)
    width = -(-end // n)
    if n > min(group[1], kc.words()["kChainSizeMask"]):
        raise kc.ChainRefused(
            f"{label}: pages to {end - 1}; on one island a gather's record takes "
            f"{most} pages a rank, so {n} ranks, and the chain has {group[1]} "
            f"(a record at most {kc.words()['kChainSizeMask']})"
        )
    one = Described(program, [tensor.address, first, pages], (group[0], n), 1, 0)
    one.width, one.output, one.program = width, tensor, program
    one.name = f"km.{spelling('all_gather')}"
    ranges = [kc.Range(first, 0, pages) if n == 1 else kc.Range(0, width, width)]
    one.reads = {id(tensor): (tensor, None, ranges, [])}
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
