"""A core's place in its card's grid as the handler's C++ names it: the flat rank
(Core::me()) on one island, where they are the same; the card's core index on islands,
where the flat rank counts from the island's first rank instead (km::rect_place and
km::rect_broadcast take a place in the card's grid)."""


def card_core(me, islands):
    """The C++ of `me`'s (a Core or Compute's name) place in the card's grid for a
    class built for `islands` (compiler Compiled.islands: count, most ranks): one
    island's text unchanged (its kernels byte-equal)."""
    return f"{me}.me()" if islands[0] <= 1 else f"{me}.card_core()"
