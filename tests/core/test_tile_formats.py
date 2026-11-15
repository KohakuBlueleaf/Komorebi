"""The block-float and fp16 formats on the host: each encode and decode against a plain
per-datum model, and a tensor's bytes by its tiles."""

import math

import numpy as np
import pytest

from komorebi import formats


def plain_bfp(values, bits, family_a):
    """One datum at a time: a 16-datum row shares its largest exponent; each datum's
    magnitude is rounded half to even to the width's bits, at most all ones."""
    width = formats.BFP_WIDTH[bits]
    bias = 15 if family_a else 127
    out = []
    for row in np.asarray(values, np.float64).reshape(-1, 16):
        exps = [math.frexp(v)[1] - 1 + bias if v != 0 else 0 for v in row]
        if family_a:
            exps = [min(max(e, 0), 31) for e in exps]
        shared = max(exps)
        step = 2.0 ** (shared - bias - (width - 1))
        for v in row:
            m = min(round(abs(v) / step), (1 << width) - 1)
            out.append(math.copysign(m * step, v) if m else 0.0)
    return np.array(out, np.float32)


def values(seed):
    rng = np.random.default_rng(seed)
    rows = rng.uniform(0.1, 10.0, (64, 16)) * rng.choice([-1.0, 1.0], (64, 16))
    rows *= 10.0 ** rng.integers(-1, 3, (64, 1))  # inside the A family's exponents
    rows[3] = [1.0, 0.0, -0.0, 0.5, 0.25, 0.75, -0.75, 1.5, *([0.125] * 8)]
    rows[4] = [1.0, *(((np.arange(15) % 7) + 0.5) / 64)]  # bfp8 ties under 1
    return rows.astype(np.float32).reshape(1, 1024)


@pytest.mark.parametrize("bits", [8, 4, 2])
@pytest.mark.parametrize("family_a", [False, True])
def test_block_encode_and_decode_against_the_plain_model(bits, family_a):
    tiles = values(bits)
    raw = formats.bfp_encode(tiles, bits, family_a)
    assert raw.shape == (1, formats.bfp_tile_bytes(bits))
    back = formats.bfp_decode(raw, bits, family_a)
    assert np.array_equal(back[0], plain_bfp(tiles, bits, family_a))
    _, codes = formats.bfp_unpack_codes(raw, bits)
    assert not np.any(codes == 1 << formats.BFP_WIDTH[bits])  # no negative zero


def test_float16_is_ieee_half_and_saturates_to_infinity():
    data = np.array([[1 / 3, 65504.0, 70000.0, 2.0**-24, -0.0, 1.0 + 2**-11]])
    raw, shape = formats.to_bytes(data, "float16", "row")
    back = formats.from_bytes(raw, shape, "float16", "row")
    assert np.array_equal(back, data.astype(np.float16).astype(np.float32))
    assert back[0, 2] == np.inf and back[0, 5] == 1.0


def test_tile_bytes_and_a_tensor_of_tiles():
    sizes = {name: formats.tile_bytes(name) for name in formats.BLOCK}
    assert sizes == {
        "bfp8": 1088,
        "bfp4": 576,
        "bfp2": 320,
        "bfp8_b": 1088,
        "bfp4_b": 576,
        "bfp2_b": 320,
    }
    assert formats.tile_bytes("float16") == formats.tile_bytes("bfloat16") == 2048
    shape = formats.padded_shape((40, 96), "tile")
    assert formats.buffer_bytes(shape, "bfp4_b", "tile") == 6 * 576
    assert formats.buffer_bytes((3, 50), "float16", "row") == 300
    data = np.random.default_rng(1).standard_normal((64, 96)).astype(np.float32)
    raw, shape = formats.to_bytes(data, "bfp8_b", "tile")
    assert raw.size == formats.buffer_bytes(shape, "bfp8_b", "tile")
    back = formats.from_bytes(raw, shape, "bfp8_b", "tile")
    tiles = formats.tilize(data[:32, :32])
    assert np.array_equal(
        formats.tilize(back[:32, :32]), plain_bfp(tiles, 8, False)[None]
    )


def test_a_block_format_has_no_rows_and_no_element_size():
    with pytest.raises(ValueError, match="laid out in tiles"):
        formats.to_bytes(np.zeros((2, 32)), "bfp8_b", "row")
    with pytest.raises(ValueError, match="laid out in tiles"):
        formats.page_bytes((2, 32), "bfp4", "row")
    with pytest.raises(ValueError, match="only a tile's"):
        formats.itemsize("bfp2_b")
    assert formats.family("bfp8") == "a" and formats.family("bfp8_b") == "b"
    assert formats.family("float32") is None and formats.family("int32") is None
