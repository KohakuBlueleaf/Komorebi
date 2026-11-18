"""A local=True reduce over islands: a command whose groups cross an island's edge is
refused on the host (IslandRun's submit, and a prebuild's mock run)."""

from komorebi.dsl.front import front


def check_local(runs, graph, handler, group):
    """Refuses a command of a class's handler whose reduce is local=True when its
    group, or one of the reduce's groups of S ranks, holds ranks of two islands
    (`runs` each island's Run, `graph` the islands' ranks)."""
    method = handler.method if isinstance(handler, front.ClassHandler) else handler
    if (
        not isinstance(method, front.HandlerMethod)
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
