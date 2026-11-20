"""A local=True reduce over islands, and a direct input on islands: a command whose
groups cross an island's edge, or whose direct input another island holds, is refused
on the host (IslandRun's submit, and a prebuild's mock run). It imports no
DSL module: the mock run imports it, and the DSL front reaches the mock's tensors."""

import numpy as np

from komorebi.errors import KomorebiError
from komorebi.program.checks import near as near_module


def check_local(runs, graph, handler, group):
    """Refuses a command of a class's handler whose reduce is local=True when its
    group, or one of the reduce's groups of S ranks, holds ranks of two islands
    (`runs` each island's Run, `graph` the islands' ranks)."""
    # a ClassHandler's method; a HandlerMethod is the one with a function (a number or
    # a name has none)
    method = getattr(handler, "method", handler)
    if (
        getattr(method, "function", None) is None
        or len(runs) < 2
        or not any(getattr(p, "reduce_local", None) for p in runs[0].programs)
    ):
        return
    # asked with the handler as named: a ClassHandler keeps the program it was bound to
    program = runs[0].programs[runs[0]._handler(handler)[1]]
    line = getattr(program, "reduce_local", {}).get(method.name)
    if line is None:
        return
    first, count = int(group[0]), int(group[1])
    size = getattr(program, "reduce_groups", {}).get(method.name, (count,))[0]
    for island in range(1, len(runs)):
        edge = graph.first(island)
        if first < edge < first + count and (edge - first) % size:
            groups = "its group" if size == count else f"a group of {size} ranks"
            raise ValueError(
                f"{type(program).__name__}.{method.name}, {line}: local=True, and "
                f"{groups} of the command ({first}, {count}) holds ranks of islands "
                f"{island - 1} and {island} (island {island} starts at rank {edge}): "
                "submit groups that lie on one island, or leave local= out"
            )


def one_island(tensor):
    """The island that holds every page of island tensor `tensor`; None: a copy (each
    island its own) or not an island tensor; -1: split over islands."""
    placement = getattr(tensor, "placement", None)
    if placement is None or getattr(placement, "copy", True):
        return None
    holders = [i for i, (_, steps) in enumerate(placement.parts) if steps]
    return holders[0] if len(holders) == 1 else -1


def check_direct(graph, handler, group, values):
    """Refuses a command whose direct input (km.direct) is split over islands, or held
    by one island while the command's group holds ranks of another: a direct input
    is read through this island's own part (km::direct_or)."""
    method = getattr(handler, "method", handler)
    direct = getattr(method, "direct", ())
    if not direct or getattr(method, "function", None) is None:
        return
    given = dict(zip(method.places(), values))
    first, count = int(group[0]), int(group[1])
    for place in direct:
        holder = one_island(given.get(place))
        if holder is None:
            continue
        if holder < 0:
            raise ValueError(
                f"{method.name}: its direct input {place} is split over islands: a "
                "direct input is read on one island; place it on one island (a group "
                "inside it) or as a copy, or leave km.direct out on two cards"
            )
        start = graph.first(holder)
        if first < start or first + count > start + graph.count(holder):
            raise ValueError(
                f"{method.name}: its direct input {place} is held by island {holder} "
                f"(ranks {start} to {start + graph.count(holder) - 1}), and the "
                f"command's group ({first}, {count}) holds ranks of another island: "
                "submit a group inside that island, place the tensor as a copy, or "
                "leave km.direct out on two cards"
            )


def check_near(program, method, values, island, group=None, ranks=None):
    """Refuses a command of a class's handler on island `island` (flat ranks `ranks`) of
    a near build (program.near) when a tensor it is given is not whole on that island,
    or its handler reduces, uses ports or broadcasts over a group with ranks of that
    island and another: the near build reads this island's pages alone."""
    far = getattr(program, "near", None)
    if not far or island is None or getattr(method, "function", None) is None:
        return
    for place, value in zip(method.places(), values):
        holder = one_island(value)
        if holder is None and hasattr(value, "entry") and hasattr(value, "bank"):
            holder = value.island  # a bank range (BankRange): its island's
        if holder is None or holder == island:
            continue
        held = "split over islands" if holder < 0 else f"held by island {holder}"
        raise KomorebiError(
            f"{type(program).__name__}.{method.name}: its tensor {place} is {held}, "
            f"and the command runs on island {island}, but the program was built "
            f"{near_module.line(*far)}: such a load reads and writes the pages of the "
            f"island that runs the command alone; give {place} as a copy, or as a "
            "tensor of the island whose ranks the command takes"
        )
    crosses = method.name in getattr(program, "port_handlers", ()) or (
        method.name in getattr(program, "broadcast_lines", {})
    )
    if group is None or ranks is None or not crosses:
        return
    first, count = int(group[0]), int(group[1])
    size = getattr(program, "reduce_groups", {}).get(method.name, (count,))[0]
    low, high = ranks
    for start in range(first, first + count, max(size, 1)):
        end = start + size
        if start < high and end > low and (start < low or end > high):
            groups = "its group" if size == count else f"a group of {size} ranks"
            raise KomorebiError(
                f"{type(program).__name__}.{method.name}: {groups} of the command "
                f"({first}, {count}) holds ranks of island {island} (ranks {low} to "
                f"{high - 1}) and of another, and the handler reduces, uses ports or "
                f"broadcasts, but the program was built {near_module.line(*far)}: "
                "such a load's movers have no page links; submit groups that lie on "
                'one island, or make the class with form="runtime" (its handlers\' '
                "code in parts in L1, its movers built with the page links)"
            )


def near_words(programs, what):
    """Refuses `what`, commands given as words (submit_many, or a handler named by its
    name or number), on a load built without the far paths (a program's `near`): the
    host checks a near load's commands against the island that runs them from the
    tensors a class's handler is given (check_near), and words hold none."""
    for program in programs:
        far = getattr(program, "near", None)
        if far:
            raise KomorebiError(
                f"{what}: {program.name} was built {near_module.line(*far)}, and a "
                "command given as words cannot be checked against the island that "
                "runs it: submit each command as the class's handler with its "
                "tensors (run.submit(program.handler, group, args=(...)))"
            )


def near_collective(programs, what):
    """Refuses `what`, a collective of the host, on a load built without the far
    paths (a program's `near`): its service overlay was built so too."""
    for program in programs:
        far = getattr(program, "near", None)
        if far:
            raise KomorebiError(
                f"{what}: {program.name} was built {near_module.line(*far)}, and "
                "a collective crosses islands"
            )


def one_command(handler, firsts, counts, args):
    """(handler, group, words) when submit_many's arguments are one command (one
    handler, one first and count, at most one row of integer words), else None: those
    submit takes as they are (any other form, a wrong one too, goes submit_many's way).
    """
    if not isinstance(handler, str | int):  # a Handler is an int
        number = np.asarray(handler)
        if number.ndim or number.dtype.kind not in "iu":
            return None
        handler = int(number)
    first, count = np.asarray(firsts), np.asarray(counts)
    if first.size != 1 or count.size != 1 or first.ndim > 1 or count.ndim > 1:
        return None
    if first.dtype.kind not in "iu" or count.dtype.kind not in "iu":
        return None
    words = ()
    if args is not None:
        args = np.asarray(args)
        if args.ndim != 2 or args.shape[0] != 1 or args.dtype.kind not in "iu":
            return None
        words = args[0].tolist()
    elif first.ndim + count.ndim == 0:
        return None  # no row at all (submit_many refuses it)
    return handler, (int(first.reshape(-1)[0]), int(count.reshape(-1)[0])), words
