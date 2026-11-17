"""A class reduce's arrival slots for groups of S ranks on islands: a group may touch
several islands, each part folding as its own group and its first rank sending once to
rank 0 (km::reduce_slots: the root part's rounds, then a slot each other part)."""


def group_slots(size, islands):
    """A tree reduce's arrival slots over groups of `size` ranks for a class built for
    `islands` (compiler Compiled.islands: count, most ranks): the rounds of the largest
    part, plus one each further island the group can touch; one island's unchanged."""
    count, most = islands
    if count <= 1:
        return max(1, (size - 1).bit_length())
    return max(1, (min(size, most) - 1).bit_length() + min(count - 1, size - 1))
