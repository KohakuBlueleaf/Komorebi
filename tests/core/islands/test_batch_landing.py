"""The landing of km::put_bytes' batches on the host: a page for each holder, source
island and put line owner (so two neighbours of one holder never share one), the far
pages after them, and a session whose DRAM cannot hold it refused by name."""

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


def landings(sessions, split=1):
    plan = SimpleNamespace(sessions=sessions, split=split)
    return LinkPlanMixin._landings(plan, abi.headers())


def test_a_page_for_each_holder_source_and_owner_then_the_far_places():
    h = abi.headers()
    two = landings([Dram(), Dram()])
    assert two[0].shape == (2 * 2 * h["KM_QUEUE_LINES"], h["KM_PUT_BATCH_BYTES"] // 4)
    four = landings([Dram()] * 4, split=4)  # a split card: island 0's for every one
    pages = 4 * 4 * h["KM_QUEUE_LINES"] + 4 * 4 * h["KM_FAR_LINES"]
    assert four[0].shape[0] == pages and all(one is four[0] for one in four)


def test_a_landing_the_dram_cannot_hold_is_refused_by_name():
    first = Dram()
    with pytest.raises(KomorebiError, match="the put batches' landing: .* MiB"):
        landings([first, Dram(fits=False)])
    assert first.made[0].freed
