"""What differs between chip architectures that the host needs, in one table keyed by
the open device's name for its architecture (Device.arch): the place for every constant
that depends on the chip."""

from typing import NamedTuple

from komorebi.errors import KomorebiError


class Chip(NamedTuple):
    """One architecture's facts."""

    busy_mhz: int  # the Tensix clock the driver raises a chip to while it is held
    # one multicast rectangle covers the whole worker grid, its NoC0 gaps included
    # (measured); else a rectangle stays in a panel
    whole_grid: bool
    full_grid: tuple | None  # the unharvested worker grid whole_grid was measured on
    tensix_noc0_x: tuple | None  # Tensix columns' NoC0 x by harvest bit (None: rows)


CHIPS = {
    "blackhole": Chip(
        busy_mhz=1350,
        whole_grid=False,
        full_grid=None,
        tensix_noc0_x=(1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16),
    ),
    # a multicast rectangle from any core, loopback_src, NoC 1 given high corner to
    # low: the 8 x 10 grid in one (round 3, a Galaxy's chip in slow dispatch)
    "wormhole_b0": Chip(
        busy_mhz=1000, whole_grid=True, full_grid=(8, 10), tensix_noc0_x=None
    ),
}


# A card's remedy when it is not ready, by the driver's card type (tt_card_type): a
# Galaxy's chips are reset together; any other card is reset alone.
RESETS = {
    "galaxy-wormhole": (
        "reset the whole box with tt-smi -glx_reset (a Galaxy's chips are reset "
        "together: every process holding one of them exits first), then open it again"
    ),
}
RESET = (
    "reset it with tt-smi -r {card} (the recovery guide, when a card reset is still "
    "the only way), then open it again"
)


def reset(card, card_type):
    """What to do for card `card` (tt-metal's number) of the driver's card type
    `card_type` (None: not known) when it is not ready."""
    return RESETS.get(card_type, RESET).format(card=card)


def noc0_columns(arch, harvest_mask):
    """NoC0 x of each logical column of a chip of `arch` whose harvest mask is
    `harvest_mask`: the Tensix columns whose mask bit is clear."""
    columns = chip(arch).tensix_noc0_x
    if columns is None:
        raise KomorebiError(f"a {arch} chip harvests rows, not columns")
    return [x for i, x in enumerate(columns) if not harvest_mask >> i & 1]


def grid_panels(arch, grid):
    """Whether a multicast rectangle may cover the whole `grid` of a chip of `arch`
    (whole_grid); KomorebiError for a grid other than the one that was measured."""
    facts = chip(arch)
    if facts.whole_grid and facts.full_grid and tuple(grid) != facts.full_grid:
        raise KomorebiError(
            f"a {arch} chip with a {grid[0]} x {grid[1]} worker grid (harvested; "
            f"{facts.full_grid[0]} x {facts.full_grid[1]} unharvested) is not handled: "
            "its multicast rectangles were measured on the whole grid only"
        )
    return facts.whole_grid


def chip(arch):
    """The facts of architecture `arch` (Device.arch's name); KomorebiError names the
    ones known for any other."""
    try:
        return CHIPS[arch]
    except KeyError:
        raise KomorebiError(
            f"no chip facts for architecture {arch!r} (known: {', '.join(CHIPS)})"
        ) from None
