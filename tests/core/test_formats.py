"""Host-side buffer formats against bytes a card held: fixed vectors written through
the runtime's own tensor path once (data/formats.npz)."""

from pathlib import Path

import numpy as np

from komorebi import formats

V = np.load(Path(__file__).parent / "data" / "formats.npz")


def packed(name, dtype, layout, source=None):
    raw, shape = formats.to_bytes(V[source or f"{name}_in"], dtype, layout)
    return raw, shape, V[f"{name}_raw"].reshape(-1)


def test_bfloat16_tiles_round_to_nearest_even_and_read_back():
    raw, shape, want = packed("bf16_tile", "bfloat16", "tile")
    assert np.array_equal(raw, want) and shape == (64, 96)
    back = formats.from_bytes(want, shape, "bfloat16", "tile")
    assert np.array_equal(back, V["bf16_tile_back"])


def test_bfloat16_rows_float32_tiles_and_32_bit_rows():
    for name, dtype, layout, source in (
        ("bf16_row", "bfloat16", "row", None),
        ("f32_tile", "float32", "tile", None),
        ("i32_row", "int32", "row", None),
        ("u32_row", "uint32", "row", "i32_row_in"),
    ):
        raw, shape, want = packed(name, dtype, layout, source)
        assert np.array_equal(raw, want), name
    back = formats.from_bytes(want, shape, "uint32", "row")
    assert np.array_equal(back, V["u32_row_back"])


def test_a_tile_layout_pads_with_zeros_to_whole_tiles():
    raw, shape = formats.to_bytes(np.ones((33, 2), np.float32), "float32", "tile")
    back = formats.from_bytes(raw, shape, "float32", "tile")
    assert shape == (64, 32) and back.sum() == 66 and back[:33, :2].all()
