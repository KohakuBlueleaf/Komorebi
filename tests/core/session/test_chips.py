"""The table of what differs between chip architectures and card types."""

import pytest

from komorebi.device import chips, remembered
from komorebi.errors import KomorebiError, ProgramError
from komorebi.islands.session import PACKET_BYTES
from komorebi.session import kernels, sizes


def test_each_architecture_has_its_busy_clock_and_another_is_named():
    assert chips.chip("blackhole").busy_mhz == 1350
    assert chips.chip("wormhole_b0").busy_mhz == 1000
    with pytest.raises(KomorebiError, match="'quasar'.*blackhole, wormhole_b0"):
        chips.chip("quasar")


def test_a_galaxy_chip_is_reset_with_its_whole_box_and_another_card_alone():
    assert "tt-smi -glx_reset" in chips.reset(3, "galaxy-wormhole")
    assert "tt-smi -r 3 " in chips.reset(3, "p150a")
    assert "tt-smi -r 3 " in chips.reset(3, None)


def test_a_blackhole_chip_s_columns_come_from_its_harvest_mask():
    # columns 7 and 10 harvested: logical 0-5 at NoC0 x 1-6, 6-11 at 11-16
    assert chips.noc0_columns("blackhole", 0b11000000) == [
        1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16
    ]  # fmt: skip
    with pytest.raises(KomorebiError, match="harvests rows"):
        chips.noc0_columns("wormhole_b0", 0)


def test_a_wormhole_grid_is_one_panel_and_a_harvested_one_is_refused_by_name():
    assert chips.grid_panels("wormhole_b0", (8, 10)) is True
    assert chips.grid_panels("blackhole", (12, 10)) is False
    with pytest.raises(KomorebiError, match="8 x 9 worker grid .* not handled"):
        chips.grid_panels("wormhole_b0", (8, 9))


def test_the_kernel_buffer_s_bounds_are_the_chip_s_and_a_card_type_names_it():
    assert sizes.kernel_buffer_max("blackhole") == 1536 * 1024 - 41088 - 64
    assert sizes.kernel_buffer_max("wormhole_b0") == 1464 * 1024 - 35168 - 64
    # the runtime's own buffer on a Wormhole chip leaves 1,393,312 B (measured)
    assert sizes.worker_l1_size(69 * 1024, "wormhole_b0") == 1393312
    assert chips.card_arch("p150a") == "blackhole"
    assert chips.card_arch("galaxy-wormhole") == "wormhole_b0"
    with pytest.raises(KomorebiError, match="card type 'e75' is not known"):
        chips.card_arch("e75")


def test_a_buffer_past_the_chip_s_indices_is_refused_by_name():
    """A Wormhole core has 32 buffer indices (Device.buffer_indices), Blackhole 64."""
    cbs = [(40, "bfloat16", 2048, 2)]
    assert kernels.within_indices(cbs, 64) == cbs
    with pytest.raises(ProgramError, match="index 40 is past the chip's 32 buffer"):
        kernels.within_indices(cbs, 32)
    with pytest.raises(ProgramError, match="index 32 .the chain's words, kChainBuffer"):
        kernels.within_indices([(32, "uint32", 2816, 1)], 32)


def test_ticks_a_microsecond_are_the_reported_clock_by_the_chip_s_ratio():
    assert chips.tick_mhz("blackhole", 1350) == 1350
    assert chips.tick_mhz("wormhole_b0", 1000) == pytest.approx(985.5)
    assert chips.tick_mhz("wormhole_b0", None) is None


def test_wormhole_s_fabric_payload_is_below_an_island_s_packet():
    assert chips.chip("wormhole_b0").fabric_payload_bytes == 7616 < PACKET_BYTES
    assert chips.chip("blackhole").fabric_payload_bytes == 15232


def test_a_script_s_remembered_size_is_its_chip_s():
    assert remembered.chip_shape("card", "blackhole") == "card"
    assert remembered.chip_shape("card", "wormhole_b0") == "card wormhole_b0"


def test_only_blackhole_has_the_x280_a_load_needs():
    assert chips.chip("blackhole").x280 and not chips.chip("wormhole_b0").x280
