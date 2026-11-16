"""A math part that defines matmul.h's throttled_mop_status is built again with its
copy set unknown at entry and the resident kernel's after the call; every other part
is built as before, and the copy passes only with that build (no card, no compiler)."""

from pathlib import Path

import pytest

from komorebi import OverlayError
from komorebi.program import overlay

BASE = overlay.LINK_BASE
ENTRY = Path(overlay.__file__).parent / (
    "../../komorebi_native/include/komorebi/device/km_overlay_entry.hpp"
)


def facts(defines_status, kernel_defines_status):
    symbols = {"km_overlay_entry": (BASE, "FUNC", "GLOBAL")}
    if defines_status:
        symbols[overlay.MOP_STATUS] = (BASE + 0x80, "OBJECT", "LOCAL")
    part = overlay.Linked(BASE, BASE, (), symbols, (), 192, bytes(128))
    kernel = {overlay.MOP_STATUS: (0xFFB00420, "OBJECT", "LOCAL")}
    persistent = overlay.Persistent(
        (0x4000, 0x6000), kernel if kernel_defines_status else {}
    )
    return part, persistent


def test_only_a_math_part_that_defines_it_is_built_with_the_resets():
    part, persistent = facts(True, True)
    assert overlay.mop_defines("math", part, persistent) == {
        "KM_OVERLAY_MOP_STATUS": 1,
        "KM_OVERLAY_RESIDENT_MOP_STATUS": "0xffb00420",
    }
    part, persistent = facts(True, False)
    assert overlay.mop_defines("math", part, persistent) == {"KM_OVERLAY_MOP_STATUS": 1}
    assert overlay.mop_defines("unpack", part, persistent) == {}
    assert overlay.mop_defines("math", *facts(False, True)) == {}


def test_the_entry_resets_both_copies_under_the_builder_s_defines_alone():
    text = ENTRY.read_text()
    guarded = text.index("#if defined(TRISC_MATH) && defined(KM_OVERLAY_MOP_STATUS)")
    assert text.index("ckernel::throttled_mop_status = 2;", guarded) > guarded
    resident = text.index("defined(KM_OVERLAY_RESIDENT_MOP_STATUS)")
    stored = text.index("(KM_OVERLAY_RESIDENT_MOP_STATUS) = 2;", resident)
    plain = text.index("#else\n    return km::overlay_invoke", stored)
    assert guarded < resident < stored < plain


def test_the_copy_passes_only_when_its_build_resets_it():
    part, persistent = facts(True, True)
    second = overlay.Linked(
        BASE + overlay.ALIGN,
        BASE + overlay.ALIGN,
        (),
        {
            "km_overlay_entry": (BASE + overlay.ALIGN, "FUNC", "GLOBAL"),
            overlay.MOP_STATUS: (BASE + overlay.ALIGN + 0x80, "OBJECT", "LOCAL"),
        },
        (),
        192,
        bytes(128),
    )
    with pytest.raises(OverlayError, match="throttled_mop_status"):
        overlay.check("qkv", "math", part, second, persistent, 256)
    overlay.check("qkv", "math", part, second, persistent, 256, (overlay.MOP_STATUS,))
