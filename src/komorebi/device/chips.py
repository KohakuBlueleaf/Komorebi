"""What differs between chip architectures that the host needs, in one table keyed by
the open device's name for its architecture (Device.arch): the place for every constant
that depends on the chip."""

from typing import NamedTuple

from komorebi.errors import KomorebiError


class Chip(NamedTuple):
    """One architecture's facts."""

    busy_mhz: (
        int  # the Tensix clock the driver raises a chip to while a process holds it
    )


CHIPS = {
    "blackhole": Chip(busy_mhz=1350),
    "wormhole_b0": Chip(busy_mhz=1000),
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


def chip(arch):
    """The facts of architecture `arch` (Device.arch's name); KomorebiError names the
    ones known for any other."""
    try:
        return CHIPS[arch]
    except KeyError:
        raise KomorebiError(
            f"no chip facts for architecture {arch!r} (known: {', '.join(CHIPS)})"
        ) from None
