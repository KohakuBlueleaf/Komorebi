"""komorebi.tools.wormhole_products pins the chip's erratum (-1 unit becomes -16 units)
with the numbers a Wormhole chip gave for these columns, in a float32 field and in Dst,
and for two random bfloat16 sums a chip got wrong (data/wormhole_erratum.npz)."""

from pathlib import Path

import numpy as np
import pytest

from komorebi.tools import wormhole_products as wp

E23, E22 = 2.0**-23, 2.0**-22
DATA = np.load(Path(__file__).parents[2] / "data" / "wormhole_erratum.npz")


def columns(terms, k=128):
    """x: one row of ones at every k a column uses; w: the columns' terms {k: value}."""
    used = sorted({k for column in terms for k in column})
    x = np.zeros((1, k), np.float32)
    x[0, used] = 1
    w = np.zeros((k, len(terms)), np.float32)
    for j, column in enumerate(terms):
        for at, value in column.items():
            w[at, j] = value
    return x, w


ONE_BLOCK = {0: 1.0, 8: E23, 16: -1.0, 24: -E22}  # the accumulator -1 at the 3rd term
TWO_BLOCKS = {0: 1.0, 8: E23, 64: -1.0, 72: -E22}  # the -1 falls in the second block


@pytest.mark.parametrize("fidelity", ["LoFi", "HiFi2", "HiFi3", "HiFi4"])
def test_an_accumulator_of_minus_one_unit_becomes_minus_sixteen(fidelity):
    mirror = {k: -v for k, v in ONE_BLOCK.items()}
    zero = {**ONE_BLOCK, 24: -E23}
    x, w = columns([ONE_BLOCK, mirror, zero, TWO_BLOCKS])
    field = wp.product(x, w, fidelity, 2, "field")[0]
    dst = wp.product(x, w, fidelity, 2, "dst")[0]
    assert field.tolist() == [-16.0, E23, 0.0, -E23]
    assert dst.tolist() == [-16.0, E23, 0.0, -16.0]
    right = wp.product(x, w, fidelity, 2, "dst", erratum=False)[0]
    assert right.tolist() == [-E23, E23, 0.0, -E23]


def test_the_error_scales_with_the_operands():
    scaled = [{k: v * 2.0**s for k, v in ONE_BLOCK.items()} for s in range(-8, 9)]
    x, w = columns(scaled)
    got = wp.product(x, w, "HiFi4", 2, "dst")[0]
    assert got.tolist() == [-16.0 * 2.0**s for s in range(-8, 9)]


def test_two_random_sums_a_chip_got_wrong():
    for case, bits, right in ((0, 0xBF802710, -0.0011620), (1, 0xBEEA2338, 0.0427305)):
        x = DATA["x"][case][None, :]
        w = DATA["w"][case][:, None]
        seen = {}
        got = wp.product(x, w, "HiFi4", 2, "dst", seen=seen)
        assert got.view(np.uint32)[0, 0] == bits and seen["hits"] == 1
        fixed = wp.product(x, w, "HiFi4", 2, "dst", erratum=False)[0, 0]
        assert abs(fixed - right) < 1e-4
        hifi3 = wp.product(x, w, "HiFi3", 2, "dst")[0, 0]
        assert hifi3 == pytest.approx(right, abs=1e-4)


def test_a_positive_start_keeps_every_accumulator_positive():
    x, w = columns([ONE_BLOCK, TWO_BLOCKS])
    for form, blocks in (("field", 2), ("dst", 1)):
        seen = {}
        got = wp.product(x, w, "HiFi4", 2, form, start=1024.0, seen=seen)[0]
        assert seen["negative"] == 0 and seen["hits"] == 0
        off = np.abs(got.astype(np.float64) - blocks * 1024.0 + E23)
        assert off.max() <= 4 * 2.0**-13


def test_operands_must_be_bfloat16_values():
    with pytest.raises(ValueError, match="bfloat16"):
        wp.operands(np.float32([1.0 + 2.0**-20]))
