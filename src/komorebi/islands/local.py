"""A local=True reduce over islands, and a direct input on islands: a command whose
groups cross an island's edge, or whose direct input another island holds, is refused
on the host (IslandRun's submit, and a prebuild's mock run). It imports no
DSL module: the mock run imports it, and the DSL front reaches the mock's tensors."""


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
    program = runs[0].programs[runs[0]._handler(method)[1]]
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
