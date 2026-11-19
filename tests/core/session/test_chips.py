"""The table of what differs between chip architectures and card types."""

import re
from types import SimpleNamespace

import pytest
from fake_port import FakeBackend, FakePort

from komorebi import chain
from komorebi.device import chips, remembered
from komorebi.errors import KomorebiError, ProgramError
from komorebi.islands.session import PACKET_BYTES
from komorebi.session import kernels, sizes
from komorebi.session.parts import load
from komorebi.session.run import Run


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
    words = [(chain.words()["kChainBuffer"], "uint32", 2816, 1)]  # a program's 0 to 28
    assert kernels.within_indices(words, 32) == words
    link = "the link cores' put queue (KM_LINK_CB): buffer index 33 is past"
    with pytest.raises(ProgramError, match=re.escape(link)):
        kernels.within_indices([(33,)], 32, "the link cores' put queue (KM_LINK_CB): ")


def test_the_slots_go_above_the_highest_index_only_when_none_below_is_free():
    """A program that takes every index below its highest (decode's chain program:
    0 to 28, the runtime's 29 and 30) has its slots above, on the chip's indices."""
    assert kernels.slot_index({0, 1, 2, 30}, most=32) == 29
    full = set(range(31))
    assert kernels.slot_index(full, most=64) == 31
    assert kernels.slot_index(full, (33,), 32) == 31
    assert kernels.slot_index(full | {31}, most=64) == 32
    assert kernels.slot_index(full | {31, 32}, (33,), 64) == 34
    with pytest.raises(ProgramError, match=r"decode: no circular buffer index for the"):
        kernels.slot_index(full | {31}, (), 32, "decode: ")
    with pytest.raises(ProgramError, match="every index from 0 to 31 .the chip's 32"):
        kernels.slot_index(full | {31}, most=32)
    with pytest.raises(ProgramError, match="every index from 0 to 29$"):
        kernels.slot_index(full)


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


# A program part loaded into a core's L1 after the launch (the program form, overlays,
# add-ons, a switch): refused by name where it fails (chips.late_code_fails).
LATE = "refused on a wormhole_b0 chip: a program part loaded into a core's L1"


def test_only_a_wormhole_chip_fails_code_loaded_after_the_launch():
    assert chips.chip("wormhole_b0").late_code_fails
    assert not chips.chip("blackhole").late_code_fails
    assert chips.late_code_refusal("blackhole", "overlays") is None
    assert chips.late_code_refusal(None, "overlays") is None
    text = chips.late_code_refusal("wormhole_b0", "overlays")
    assert text.startswith(f"overlays: {LATE}") and "hazards.md" in text


def program(name="p", **fields):
    return SimpleNamespace(**({"name": name, "overlays": {}, "defines": {}} | fields))


@pytest.mark.parametrize(
    "given, placed, what",
    [
        ({}, True, "the program form"),
        ({"overlays": {"o": 1}}, False, "overlays"),
        ({"defines": {"KM_ADDED_HANDLERS": 1}}, False, "add-ons"),
    ],
)
def test_a_wormhole_load_refuses_late_code_by_name(given, placed, what):
    one = program(**given)
    plan = (
        SimpleNamespace(programs=[one, program("q")], switch=None) if placed else None
    )
    owner = SimpleNamespace(arch="wormhole_b0")
    with pytest.raises(ProgramError, match=f"{what}.*\\(load of p\\): {LATE}"):
        load.LoadMixin._refuse_late_code(owner, one, plan)
    owner.arch = "blackhole"
    load.LoadMixin._refuse_late_code(owner, one, plan)


def test_the_kernel_form_stays_served_on_wormhole():
    owner = SimpleNamespace(arch="wormhole_b0")
    load.LoadMixin._refuse_late_code(owner, program(), None)


def test_a_wormhole_run_refuses_an_add_on_and_a_switch_by_name():
    run = Run(FakeBackend(FakePort()), (8, 10))
    run.arch = "wormhole_b0"
    with pytest.raises(ProgramError, match=f"add-ons \\(run.addon of x\\): {LATE}"):
        run.addon("x", {})
    with pytest.raises(ProgramError, match=f"a program switch \\(run.load\\): {LATE}"):
        run.load(None, (0, 1))
