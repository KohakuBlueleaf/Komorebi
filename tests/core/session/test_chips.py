"""The table of what differs between chip architectures and card types."""

import pytest

from komorebi.device import chips
from komorebi.errors import KomorebiError


def test_each_architecture_has_its_busy_clock_and_another_is_named():
    assert chips.chip("blackhole").busy_mhz == 1350
    assert chips.chip("wormhole_b0").busy_mhz == 1000
    with pytest.raises(KomorebiError, match="'quasar'.*blackhole, wormhole_b0"):
        chips.chip("quasar")


def test_a_galaxy_chip_is_reset_with_its_whole_box_and_another_card_alone():
    assert "tt-smi -glx_reset" in chips.reset(3, "galaxy-wormhole")
    assert "tt-smi -r 3 " in chips.reset(3, "p150a")
    assert "tt-smi -r 3 " in chips.reset(3, None)
