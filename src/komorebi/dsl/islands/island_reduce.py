"""A class reduce's arrival slots for groups of S ranks on islands: a group may touch
several islands, each part folding as its own group and its first rank sending once to
rank 0 (km::reduce_slots: the root part's rounds, then a slot each other part)."""


def rounds(ranks):
    """A flat tree's rounds over `ranks` (km::reduce_slots_flat)."""
    return max(0, ranks - 1).bit_length()


def group_slots(size, islands):
    """A tree reduce's arrival slots over groups of `size` ranks for a class built for
    `islands` (compiler Compiled.islands: count, most ranks): the most km::reduce_slots
    gives a group on one island or across (its root part at most size - 1 ranks and a
    slot each further island); on two islands max(rounds(S), rounds(S - 1) + 1)."""
    count, most = islands
    if count <= 1:
        return max(1, rounds(size))
    alone = rounds(size) if size <= most else 0
    across = [
        rounds(root) + min(count - 1, size - root)
        for root in range(1, min(size - 1, most) + 1)
    ]
    return max(1, alone, *across)
