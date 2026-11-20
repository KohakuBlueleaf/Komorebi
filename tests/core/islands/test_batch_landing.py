"""The landing of km::put_bytes' batches on the host: a page for each source island and
put line owner of what is put to a card (so two neighbours of one holder never share
one), the far pages after them, every holder's on a split card's one DRAM, and a
session whose DRAM cannot hold it refused by name."""

from types import SimpleNamespace

import pytest

from komorebi import abi
from komorebi.errors import KomorebiError
from komorebi.islands.parts.link_plan import LinkPlanMixin


class Landing:
    def __init__(self, shape, address=0x10000):
        self.shape, self.address, self.freed = shape, address, False
        self.aligned_page_bytes = shape[1] * 4

    def free(self):
        self.freed = True


class Dram:
    def __init__(self, fits=True):
        self.fits, self.made = fits, []

    def empty(self, shape, dtype):
        if not self.fits:
            raise RuntimeError("Out of Memory: not enough space to allocate")
        self.made.append(Landing(shape))
        return self.made[-1]


def landings(sessions, split=1, arch="blackhole"):
    plan = SimpleNamespace(sessions=sessions, split=split, arch=arch)
    return LinkPlanMixin._landings(plan, abi.headers())


def test_a_page_for_each_source_and_owner_then_the_far_places():
    """A card's landing holds what is put to it: n sources' lines, not n x n (32
    cards: 32 x 520 pages of 2,048 B, 32.5 MiB a card)."""
    h = abi.headers()
    two = landings([Dram(), Dram()])
    assert two[0].shape == (2 * h["KM_QUEUE_LINES"], h["KM_PUT_BATCH_BYTES"] // 4)
    many = landings([Dram() for _ in range(32)])
    assert many[0].shape[0] == 32 * (h["KM_QUEUE_LINES"] + h["KM_FAR_LINES"])
    four = landings([Dram()] * 4, split=4)  # a split card: island 0's for every one
    pages = 4 * 4 * h["KM_QUEUE_LINES"] + 4 * 4 * h["KM_FAR_LINES"]
    assert four[0].shape[0] == pages and all(one is four[0] for one in four)


def test_a_landing_the_dram_cannot_hold_is_refused_by_name():
    first = Dram()
    with pytest.raises(KomorebiError, match="the put batches' landing: .* MiB"):
        landings([first, Dram(fits=False)])
    assert first.made[0].freed


def test_a_chip_whose_links_write_no_far_dram_allocates_no_landing():
    """Wormhole refuses every batch to another island (KM_LINK_NO_FAR_DRAM): nothing
    lands, so nothing is allocated; the link words then carry 0."""
    cards = [Dram() for _ in range(4)]
    assert landings(cards, arch="wormhole_b0") == [None] * 4
    assert all(card.made == [] for card in cards)
