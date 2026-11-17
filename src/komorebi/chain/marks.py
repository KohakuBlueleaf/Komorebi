"""A chain's marks on the host as the walk counts them (km_chain.hpp's chain_wanted
and chain_mark): what each core of a record waits for and which roots of the record
before mark it; a gather's leaders mark their own island's cores only."""

from komorebi.chain.calls import (
    _cores,
    blocks_met,
    gather_leaders,
    input_tiles,
    is_gather,
    island_of,
    words,
    written_groups,
)


def roots(call):
    """(root, its block's first tile, the tile after it, the island it marks; None:
    every island) of each root of `call`: a gather's leaders, each the whole copy."""
    if is_gather(call):
        return [
            (leader, 0, call.width, island) for island, leader in gather_leaders(call)
        ]
    first, count = call.group
    out = []
    for group in range(count // call.size):
        root, island, block = first + group * call.size, None, group
        if call.by_island is not None:  # written by island: block g of its island's
            island = island_of(call.by_island, root)
            block -= written_groups(call, root)[0]
        out.append((root, block * call.width, (block + 1) * call.width, island))
    return out


def marks(calls):
    """{record: {core: (marks it waits for, [the roots that mark it, once a mark])}}
    of each marked record (its input the call before's output, not a command's
    first); a core whose count differs from its marks would hang or be given up."""
    out = {}
    for index, call in enumerate(calls):
        if call.input is None or index % words()["kChainRecordsMax"] == 0:
            continue
        before = calls[index - 1]
        senders, starts = roots(before), before.gather or before.by_island
        record = {}
        for core, rank, group in _cores(call):
            groups = 1 if is_gather(before) else written_groups(before, core)[1]
            tiles = input_tiles(call, core, rank, group)
            wanted = sum(len(blocks_met(f, n, before.width, groups)) for f, n in tiles)
            marked = [
                root
                for root, low, high, island in senders
                if island is None or island_of(starts, core) == island
                for first, count in tiles
                if count and first < high and low < first + count
            ]
            record[core] = (wanted, marked)
        out[index] = record
    return out
