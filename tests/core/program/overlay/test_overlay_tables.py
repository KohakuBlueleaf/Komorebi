"""A part's copies of the build's 22 constant tables of the buffers (genfiles.cpp)
pass when their bytes equal the persistent kernel's, and are refused by name when
they differ (a part built against other buffers); made-up link facts, no card."""

import dataclasses

import pytest

from komorebi import OverlayError
from komorebi.program import overlay

BASE = overlay.LINK_BASE
TABLE = "_ZL22unpack_num_faces_c_dim"


def facts(part_bytes):
    persistent = overlay.Persistent(
        text=(0x4000, 0x6000),
        symbols={TABLE: (0x9000, "OBJECT", "LOCAL")},
        tables={TABLE: bytes(range(64))},
    )
    links = [
        overlay.Linked(
            base=base,
            entry=base,
            relocations=(),
            symbols={
                "km_overlay_entry": (base, "FUNC", "GLOBAL"),
                TABLE: (base + 0x80, "OBJECT", "LOCAL"),
            },
            undefined=(),
            extent=192,
            image=bytes(range(128)),
            tables={TABLE: part_bytes},
        )
        for base in (BASE, BASE + overlay.ALIGN)
    ]
    return links, persistent


def test_the_tables_are_the_22_genfiles_emits():
    assert len(overlay.TABLES) == len(set(overlay.TABLES)) == 22
    assert TABLE in overlay.TABLES and "_ZL20pack_tile_face_r_dim" in overlay.TABLES


def test_a_table_copy_passes_when_equal_and_is_refused_by_name_when_not():
    links, persistent = facts(bytes(range(64)))
    overlay.check("mix", "unpack", *links, persistent, 256)
    links, persistent = facts(bytes(64))
    with pytest.raises(OverlayError, match=f"its table {TABLE} differs"):
        overlay.check("mix", "unpack", *links, persistent, 256)
    links = [dataclasses.replace(link, tables={}) for link in links]
    with pytest.raises(OverlayError, match="differs"):  # bytes not read: not equal
        overlay.check("mix", "unpack", *links, persistent, 256)
