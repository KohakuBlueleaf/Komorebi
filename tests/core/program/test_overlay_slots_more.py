"""A part's slots past 4: the default follows the program's overlays, the records of
slots 4 and up go after the overlay area, and a load of up to 4 keeps its words and
its kernels' text (no card)."""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi.program import overlay
from komorebi.program.description import Program
from komorebi.session.session import slots_most

C = overlay.constants()


def handlers(count, kernels=("reader", "writer", "compute")):
    return [SimpleNamespace(sources={kernel: "x.cpp" for kernel in kernels})] * count


def built(count):
    slots = {part: (count, 256) for part in overlay.PARTS}
    return overlay.Built({(1, "reader"): b"\0" * 64}, slots)


def test_default_follows_the_overlays_and_a_declared_count_wins():
    assert overlay.Slots().of("reader", handlers(5)) == (5, None)
    assert overlay.Slots().of("reader", handlers(5), most=4) == (4, None)
    assert overlay.Slots(count=7).of("reader", handlers(2)) == (7, None)
    with pytest.raises(ValueError, match="slots: 0"):
        overlay.Slots(count=0).of("reader", handlers(2))
    program = Program("five", kernels={"compute": "c.cpp"})
    program.overlays = {f"h{n}": each for n, each in enumerate(handlers(5))}
    assert slots_most(program) == 5
    program.overlays = dict(list(program.overlays.items())[:3])
    assert slots_most(program) == C["kOverlaySlotsFixed"]  # no define: same text
    assert overlay.slots_define(4) == []
    assert overlay.slots_define(5) == [("KM_OVERLAY_SLOTS", "5")]


def test_records_past_four_follow_the_area():
    four = overlay.area_words(built(4), {}, 0, 64, 0x1000, 4)
    assert len(four) * 4 == C["kOverlayAreaBytes"]
    assert np.array_equal(four, overlay.area_words(built(4), {}, 0, 64, 0x1000))
    five = overlay.area_words(built(5), {}, 0, 64, 0x1000, 5)
    assert len(five) * 4 == C["kOverlayAreaBytes"] + 16 * C["kOverlayParts"]
    for index in range(C["kOverlayParts"]):
        base = five[C["kOverlayBase"] + index]
        fifth = C["kOverlayMoreRecords"] + 16 * index
        assert five[fifth // 4 + C["kOverlayAddress"]] == base + 4 * 256


def test_lowered_default_says_so_in_one_line():
    lines = []
    room = 5 * 5 * 256 - 256  # one slot less
    made = overlay.fitted(built(5), overlay.Slots(), room, lines.append)
    assert made.slots["reader"] == (4, 256) and made.slots["pack"] == (5, 256)
    assert any(
        "slots: 4 a reader where the program's overlays would take 5" in line
        for line in lines
    )
