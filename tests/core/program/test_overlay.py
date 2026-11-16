"""The overlay builder's checks (each refusal once, on made-up link facts), the host
declaration of overlays and slots, and the overlay area and budget the session writes.
No card and no compiler."""

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from komorebi import OverlayError, Program, SlowCodeError
from komorebi.program import overlay
from komorebi.session import session

BASE = overlay.LINK_BASE
PERSISTENT = overlay.Persistent(
    text=(0x4000, 0x6000),
    symbols={
        "helper": (0x4100, "FUNC", "GLOBAL"),
        "noc_reads_num_issued": (0x9000, "OBJECT", "GLOBAL"),
        overlay.RESETS[0]: (0x9100, "OBJECT", "LOCAL"),
    },
)


def linked(base=BASE, **changes):
    """A part's link that passes every check, with `changes`."""
    made = overlay.Linked(
        base=base,
        entry=base,
        relocations=(),
        symbols={"km_overlay_entry": (base, "FUNC", "GLOBAL")},
        undefined=("noc_reads_num_issued",),
        extent=192,
        image=bytes(range(128)),
    )
    return dataclasses.replace(made, **changes)


def check(first=None, second=None, slot_bytes=256):
    first = first or linked()
    second = second or linked(BASE + overlay.ALIGN)
    overlay.check("mix", "reader", first, second, PERSISTENT, slot_bytes)


def test_a_good_part_passes():
    check()


def test_a_part_without_entry_is_refused():
    with pytest.raises(OverlayError, match="no entry km_overlay_entry"):
        check(linked(entry=None))


def test_a_part_with_relocations_is_refused():
    with pytest.raises(OverlayError, match="relocations left"):
        check(linked(relocations=(".rela.text",)))


def test_a_call_into_the_persistent_text_is_refused():
    with pytest.raises(OverlayError, match=r"calls \['helper'\]"):
        check(linked(undefined=("helper",)))


def test_a_second_copy_of_runtime_state_is_refused():
    copy = {"noc_reads_num_issued": (BASE + 64, "OBJECT", "LOCAL")}
    with pytest.raises(OverlayError, match="its own copy of the runtime's state"):
        check(linked(symbols=linked().symbols | copy))


def test_a_reset_copy_of_library_state_passes():
    reset = {overlay.RESETS[0]: (BASE + 64, "OBJECT", "LOCAL")}
    check(linked(symbols=linked().symbols | reset))


def test_a_part_larger_than_its_slot_is_refused():
    with pytest.raises(OverlayError, match="do not fit its slot of 128 bytes"):
        check(slot_bytes=128)


def test_two_links_that_differ_are_refused():
    moved = linked(BASE + overlay.ALIGN, image=bytes(range(1, 129)))
    with pytest.raises(OverlayError, match="not position-independent"):
        check(second=moved)


DEVICE = Path(__file__).parents[3] / "src/komorebi_native/include/komorebi/device"


def test_the_device_headers_clear_the_put_structs_a_field_at_a_time():
    """A block clear of PutRings or PutRun (`rings{}`, `= {}`) may compile to a
    memset call: a jal to the firmware's memset, whose offset differs at each link
    of an overlay part (a split card's page-link code took one: `not
    position-independent`). put_rings_clear and put_run_clear clear them instead."""
    block = re.compile(r"\bPut(Rings|Run)\s+\w+\s*(=\s*)?\{\s*\}")
    found = [
        f"{path.name}:{number}"
        for path in sorted(DEVICE.rglob("*.hpp"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if block.search(line.split("//")[0])
    ]
    assert found == []


@pytest.fixture
def kernels(tmp_path):
    """A program folder: three kernels of the kernel form, a header with handler and
    overlay numbers, and two overlays (one with reader and compute parts)."""
    for name in ("reader", "writer", "compute"):
        (tmp_path / f"{name}.cpp").write_text("")
    (tmp_path / "parts.h").write_text(
        "constexpr uint32_t kHandlerRun = 1;\n"
        "constexpr uint32_t kOverlayMix = 1;\n"
        "constexpr uint32_t kOverlayScaleTiles = 2;\n"
    )
    for name, parts in (("mix", ("reader",)), ("scale", ("reader", "compute"))):
        (tmp_path / name).mkdir()
        for part in parts:
            (tmp_path / name / f"{part}.cpp").write_text("")
    return tmp_path


def test_an_overlay_takes_its_number_from_the_header(kernels):
    program = Program("p", kernels=kernels, handlers=kernels / "parts.h")
    program.overlay("mix", kernels / "mix").overlay("scale_tiles", kernels / "scale")
    assert program.overlays["mix"].number == 1
    assert program.overlays["scale_tiles"].number == 2
    assert program.overlays["scale_tiles"].parts() == [
        ("reader", "reader"),
        ("compute", "unpack"),
        ("compute", "math"),
        ("compute", "pack"),
    ]


def test_an_overlay_without_number_or_with_a_taken_one_is_refused(kernels):
    program = Program("p", kernels=kernels, handlers=kernels / "parts.h")
    with pytest.raises(ValueError, match="no kOverlay constant"):
        program.overlay("other", kernels / "mix")
    program.overlay("mix", kernels / "mix")
    with pytest.raises(ValueError, match="number 1 is taken by mix"):
        program.overlay("again", kernels / "scale", number=1)
    with pytest.raises(ValueError, match="not in 1.."):
        program.overlay("big", kernels / "scale", number=64)
    with pytest.raises(FileNotFoundError, match="holds no reader.cpp"):
        program.overlay("none", kernels / "nowhere", number=3)


def test_slots_default_to_one_per_overlay_and_are_checked(kernels):
    overlays = [
        overlay.declare("mix", kernels / "mix", 1),
        overlay.declare("scale", kernels / "scale", 2),
    ]
    assert overlay.Slots().of("reader", overlays) == (2, None)
    assert overlay.Slots().of("compute", overlays) == (1, None)
    assert overlay.Slots(1, {"reader": 512}).of("reader", overlays) == (1, 512)
    assert overlay.Slots(5).of("reader", overlays) == (5, None)  # the L1 decides
    with pytest.raises(ValueError, match="slots: 0"):
        overlay.Slots(0).of("reader", overlays)
    with pytest.raises(ValueError, match="a multiple of 64"):
        overlay.Slots(1, 100).of("reader", overlays)


def test_the_area_holds_the_table_and_the_slots():
    c = overlay.constants()
    built = overlay.Built(
        images={(1, "reader"): bytes(128), (2, "reader"): bytes(256)},
        slots={"reader": (2, 256)},
    )
    rows, placed = overlay.dram_rows(built)
    assert rows.shape == (2 * c["kOverlayBanks"], 64)
    words = overlay.area_words(built, placed, 0x1000, 1024, 0x20000)
    assert words[c["kOverlayCount"]] == 3
    assert words[c["kOverlayBase"]] == 0x20000
    assert words[c["kOverlaySlots"]] == 2
    assert words[c["kOverlaySlotBytes"]] == 256
    entry = (c["kOverlayTable"] + 8 * c["kOverlayParts"] * 2) // 4
    # image 1's copies are pages 8 to 15: one page into each bank
    assert list(words[entry : entry + 2]) == [0x1000 + 1024, 256]
    assert np.count_nonzero(words[c["kOverlayBase"] + 1 : c["kOverlayBase"] + 5]) == 0
    second = (c["kOverlayRecords"] + 16) // 4  # the reader's slot 1: its address
    assert words[second + c["kOverlayAddress"]] == 0x20000 + 256


def test_the_budget_table_says_whether_it_fits():
    table, fits = overlay.budget(1000, [("slots: reader, 2 x 256 B", 512)])
    assert fits and "512 B of 1,000 B" in table
    assert not overlay.budget(500, [("slots", 512)])[1]


def test_slots_are_the_first_buffer_and_end_below_the_line():
    line = session.L1_FAST_CODE
    used = {0, 1, 2, 30}
    assert session.slot_area(0x1B480, 4801, used, False) == (0x1B480, 4864, 29)
    with pytest.raises(SlowCodeError):
        session.slot_area(line - 4800, 4801, used, False)
    assert session.slot_area(line - 4800, 4801, used, True)[1] == 4864
