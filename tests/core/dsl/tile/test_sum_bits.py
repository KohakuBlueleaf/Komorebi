"""The bits a product multiplies on each side (the card's table), and the tile level's
row sum on the matrix unit refused only where its row's format loses bits."""

from komorebi import formats
from komorebi.dsl.tile.forms.steps import sum_bits_short


def test_the_side_table_is_the_cards():
    used = {
        (side, fidelity): formats.bits_used("float32", side, fidelity)
        for side in ("srca", "srcb")
        for fidelity in ("LoFi", "HiFi2", "HiFi3", "HiFi4")
    }
    assert used == {
        ("srca", "LoFi"): 5,
        ("srca", "HiFi2"): 10,
        ("srca", "HiFi3"): 10,
        ("srca", "HiFi4"): 10,
        ("srcb", "LoFi"): 7,
        ("srcb", "HiFi2"): 7,
        ("srcb", "HiFi3"): 11,
        ("srcb", "HiFi4"): 11,
    }
    assert formats.bits_used("bfloat16", "srca", "HiFi2") == 8
    assert formats.bits_used("bfp4_b", "srca", "LoFi") == 3
    assert formats.bits_used("float16", "srca", "HiFi4") == 10


def test_a_row_sum_is_refused_by_its_rows_format_and_fidelity():
    assert sum_bits_short("bfloat16", "HiFi3") is None
    assert "7 of a bfloat16 row's 8" in sum_bits_short("bfloat16", "HiFi2")
    assert "7 of a float32 row's 11" in sum_bits_short("float32", "LoFi")
    assert sum_bits_short("bfp8_b", "LoFi") is None  # 7 bits: SrcB's first phase
    assert sum_bits_short("bfp2_b", "HiFi2") is None
