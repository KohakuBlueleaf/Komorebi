"""The bytes of device buffers, made and read on the host in numpy: bfloat16, float16
and block-float values, 32 x 32 tiles of four 16 x 16 faces, and whole buffers of
rows or tiles.
"""

import math

import numpy as np

TILE = 32
FACE = 16
LAYOUTS = ("row", "tile")
PACKET_BYTES = (
    16384  # one NoC read's largest packet (NOC_MAX_BURST_SIZE): a page's most
)
# The data types a buffer holds, by name, and their numpy type on the host side.
DTYPES = {
    "bfloat16": np.dtype(np.float32),
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float32),
    "bfp8": np.dtype(np.float32),
    "bfp4": np.dtype(np.float32),
    "bfp2": np.dtype(np.float32),
    "bfp8_b": np.dtype(np.float32),
    "bfp4_b": np.dtype(np.float32),
    "bfp2_b": np.dtype(np.float32),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "uint16": np.dtype(np.uint16),
    "uint8": np.dtype(np.uint8),
}
BF16_NAN = 0x7FC0  # the quiet NaN a NaN becomes
# The exponent family of each float format: "a" a 5-bit exponent, "b" an 8-bit one;
# float32 and the integers are of neither and go with either.
FAMILY = {
    "bfloat16": "b",
    "bfp8_b": "b",
    "bfp4_b": "b",
    "bfp2_b": "b",
    "float16": "a",
    "bfp8": "a",
    "bfp4": "a",
    "bfp2": "a",
}
# Block-float formats by their bits a datum.
BLOCK = {"bfp8": 8, "bfp4": 4, "bfp2": 2, "bfp8_b": 8, "bfp4_b": 4, "bfp2_b": 2}
# Significand bits (the hidden one included) a format holds in the registers: fp32
# is held as tf32, its mantissa cut to 10 bits.
SIGNIFICAND = {
    "float32": 11,
    "float16": 11,
    "bfloat16": 8,
    "bfp8": 7,
    "bfp8_b": 7,
    "bfp4": 3,
    "bfp4_b": 3,
    "bfp2": 1,
    "bfp2_b": 1,
}

# Significand bits (of 11) the matrix unit multiplies a side at each fidelity, as
# the card gave them: a product's left operand is in SrcB, its right in SrcA.
SIDE_BITS = {
    "srca": {"LoFi": 5, "HiFi2": 10, "HiFi3": 10, "HiFi4": 10},
    "srcb": {"LoFi": 7, "HiFi2": 7, "HiFi3": 11, "HiFi4": 11},
}


def bits_used(dtype, side, fidelity):
    """Significand bits of a `dtype` operand the product multiplies on `side`
    ("srca" or "srcb") at `fidelity`: the smaller of the format's and the side's."""
    return min(SIGNIFICAND[dtype], SIDE_BITS[side][fidelity])


def itemsize(dtype):
    """Bytes of one element of `dtype` on the card (not of a block format: its tile
    is tile_bytes())."""
    if check_dtype(dtype) in BLOCK:
        raise ValueError(f"{dtype}: a block format has no element size, only a tile's")
    return 2 if dtype in ("bfloat16", "float16") else DTYPES[dtype].itemsize


def family(dtype):
    """ "a", "b", or None (float32, integers: either family)."""
    return FAMILY.get(check_dtype(dtype))


def tile_bytes(dtype):
    """Bytes of one 32 x 32 tile of `dtype`: 1088, 576, 320 for bfp8, bfp4, bfp2."""
    if check_dtype(dtype) in BLOCK:
        return bfp_tile_bytes(BLOCK[dtype])
    return TILE * TILE * itemsize(dtype)


def buffer_bytes(shape, dtype, layout, page_tiles=1):
    """Bytes of a buffer of `shape` (as on the card): its pages times a page."""
    if layout == "row":
        return math.prod(shape[:-1]) * page_bytes(shape, dtype, layout)
    pages = math.prod(shape) // TILE**2 // page_tiles
    return pages * page_bytes(shape, dtype, layout, page_tiles)


def check_dtype(dtype):
    if dtype not in DTYPES:
        raise ValueError(f"dtype {dtype!r}: one of {', '.join(DTYPES)}")
    return dtype


def bf16_bits(values):
    """float32 values as bfloat16 bits (uint16), rounded to nearest, ties to even."""
    bits = np.asarray(values, np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) >> 16
    out = rounded.astype(np.uint16)
    out[np.isnan(np.asarray(values, np.float32))] = BF16_NAN
    return out


def half_bits(values):
    """float32 values as IEEE half bits (uint16): nearest, ties to even; over 65504
    an infinity."""
    with np.errstate(over="ignore"):
        return np.asarray(values, np.float32).astype(np.float16).view(np.uint16)


def bf16_values(bits):
    """bfloat16 bits (uint16) as float32 values."""
    wide = np.asarray(bits, np.uint16).astype(np.uint32) << 16
    return wide.view(np.float32)


# A block-float tile: one exponent byte a 16-datum face row (64 bytes), then each
# datum's sign and magnitude (the hidden bit included) of 7, 3 or 1 bits.
BFP_WIDTH = {8: 7, 4: 3, 2: 1}
BFP_EXPONENTS = 64


def bfp_tile_bytes(bits):
    """Bytes of one 32 x 32 tile of bfp8, bfp4 or bfp2 (1088, 576, 320)."""
    return BFP_EXPONENTS + TILE * TILE * bits // 8


def _bfp_codes(tiles, bits, family_a):
    """[N, 1024] float32 tiles (face order) as (exponents [N, 64], codes [N, 1024]):
    tt-metal's convert_u32_to_bfp, nearest with ties to even straight to the width."""
    words = np.ascontiguousarray(tiles, np.float32).view(np.uint32)
    words = words.reshape(-1, BFP_EXPONENTS, FACE).astype(np.int64)
    exp = (words >> 23) & 0xFF
    man = words & 0x7FFFFF
    sign = words >> 31
    zero = exp == 0
    if family_a:
        exp = exp - 127 + 15
        man = np.where(exp > 31, 0x7FFFFF, np.where(exp < 0, 0, man))
        exp = np.clip(exp, 0, 31)
    shared = np.where(zero, 0, exp).max(axis=-1, keepdims=True)
    man = ((1 << 23) | man) >> np.minimum(shared - exp, 31)
    width = BFP_WIDTH[bits]
    shift = 24 - width
    rest = man & ((1 << shift) - 1)
    man >>= shift
    tie = 1 << (shift - 1)
    man += (rest > tie) | ((rest == tie) & ((man & 1) == 1))
    man = np.minimum(man, (1 << width) - 1)
    man = np.where(zero, 0, man)
    sign = np.where(man == 0, 0, sign)
    codes = (sign << width) | man
    return shared[..., 0].astype(np.uint8), codes.reshape(-1, TILE * TILE)


def bfp_pack_codes(exponents, codes, bits):
    """A tile's bytes from its exponents [N, 64] and datum codes [N, 1024]: datum 0
    of a byte in its low bits."""
    per = 8 // bits
    codes = codes.astype(np.uint8).reshape(len(codes), -1, per)
    shifts = (np.arange(per, dtype=np.uint8) * bits)[None, None]
    mantissas = np.bitwise_or.reduce(codes << shifts, axis=-1).astype(np.uint8)
    return np.concatenate([exponents.astype(np.uint8), mantissas], axis=1)


def bfp_encode(tiles, bits, family_a=False):
    """[N, 1024] float32 tiles (face order) as [N, tile bytes] uint8 of bfp8, bfp4 or
    bfp2 (the B family, or the A family's 5-bit exponent): tt-metal's host encode."""
    exponents, codes = _bfp_codes(tiles, bits, family_a)
    return bfp_pack_codes(exponents, codes, bits)


def bfp_unpack_codes(raw, bits):
    """(exponents [N, 64], codes [N, 1024]) of [N, tile bytes] uint8 block tiles."""
    raw = np.ascontiguousarray(raw, np.uint8).reshape(-1, bfp_tile_bytes(bits))
    exponents = raw[:, :BFP_EXPONENTS]
    per = 8 // bits
    shifts = (np.arange(per, dtype=np.uint8) * bits)[None, None]
    mantissas = raw[:, BFP_EXPONENTS:, None]
    codes = (mantissas >> shifts) & np.uint8((1 << bits) - 1)
    return exponents, codes.reshape(len(raw), TILE * TILE)


def bfp_decode(raw, bits, family_a=False):
    """[N, tile bytes] uint8 block tiles as [N, 1024] float32 (face order): each
    datum's magnitude times 2 ** (its row's exponent - bias - width + 1)."""
    exponents, codes = bfp_unpack_codes(raw, bits)
    width = BFP_WIDTH[bits]
    sign = codes.astype(np.int64) >> width
    man = codes.astype(np.int64) & ((1 << width) - 1)
    bias = 15 if family_a else 127
    exp = np.repeat(exponents.astype(np.int64), FACE, axis=1) - bias - (width - 1)
    values = np.ldexp(man.astype(np.float64), exp) * np.where(sign == 1, -1.0, 1.0)
    return values.astype(np.float32)


def tilize(array):
    """A [rows, cols] array (multiples of 32) as [tiles, 1024]: tiles in row-major
    order, each its four 16 x 16 faces in turn, each face row-major."""
    rows, cols = array.shape
    blocks = array.reshape(rows // TILE, 2, FACE, cols // TILE, 2, FACE)
    return blocks.transpose(0, 3, 1, 4, 2, 5).reshape(-1, TILE * TILE)


def untilize(tiles, rows, cols):
    """The [rows, cols] array of tiles laid out as tilize() lays them."""
    blocks = np.asarray(tiles).reshape(rows // TILE, cols // TILE, 2, 2, FACE, FACE)
    return blocks.transpose(0, 2, 4, 1, 3, 5).reshape(rows, cols)


def logical_shape(shape, layout):
    """`shape` when a tile layout pads it to whole tiles (the shape before the
    padding: docs/reference/tile.md, "Padding"), else None."""
    shape = tuple(int(size) for size in shape)
    return shape if padded_shape(shape, layout) != shape else None


def padded_shape(shape, layout):
    """The shape on the card: the last two dimensions of a tile layout rounded up to
    32."""
    shape = tuple(int(size) for size in shape)
    if layout == "row":
        return shape
    if len(shape) < 2:
        raise ValueError(f"shape {shape}: a tile layout needs two dimensions")
    return (*shape[:-2], *(-(-size // TILE) * TILE for size in shape[-2:]))


def page_bytes(shape, dtype, layout, page_tiles=1):
    """One page: a row (row layout), or `page_tiles` tiles that follow each other
    along the last dimension (tile layout)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout!r}: 'row' or 'tile'")
    if page_tiles != 1:
        check_page_tiles(shape, dtype, layout, page_tiles)
    if layout == "tile":
        return page_tiles * tile_bytes(dtype)
    if check_dtype(dtype) in BLOCK:
        raise ValueError(f"{dtype}: a block format is laid out in tiles, not rows")
    return shape[-1] * itemsize(dtype)


def check_page_tiles(shape, dtype, layout, page_tiles):
    """Refuses a page of `page_tiles` tiles that a tensor cannot have: not a tile
    layout, a width not whole pages, or a page past one NoC packet."""
    if not isinstance(page_tiles, int) or page_tiles < 1:
        raise ValueError(
            f"page_tiles={page_tiles!r}: a whole number of tiles, 1 or more"
        )
    if layout != "tile":
        raise ValueError(
            f"page_tiles={page_tiles} is for a tile layout; a {layout} tensor's page "
            "is a row"
        )
    columns = -(-int(shape[-1]) // TILE)
    if columns % page_tiles:
        raise ValueError(
            f"page_tiles={page_tiles}: a tile tensor's last dimension {shape[-1]} is "
            f"{columns} tiles, not a multiple of {page_tiles}: give a width of whole "
            f"pages of {page_tiles} tiles (multiples of {page_tiles * TILE})"
        )
    size = page_tiles * tile_bytes(dtype)
    if size > PACKET_BYTES:
        raise ValueError(
            f"page_tiles={page_tiles} of {dtype} is {size:,} B a page: one NoC packet "
            f"is at most {PACKET_BYTES:,} B (at most "
            f"{PACKET_BYTES // tile_bytes(dtype)} tiles of {dtype})"
        )


def transposed_tiles(array):
    """`array` ([..., rows, cols], multiples of 32) with each 32 x 32 tile transposed
    in its place: a turned product's SrcB operand as the card holds it."""
    *outer, rows, cols = array.shape
    tiles = array.reshape(*outer, rows // TILE, TILE, cols // TILE, TILE)
    return tiles.swapaxes(-1, -3).reshape(array.shape)


def to_bytes(data, dtype, layout, turned=False):
    """`data` as the buffer's bytes (uint8) and its shape on the card. A tile layout
    zero-pads the last two dimensions to 32 and tiles every [rows, cols] slice;
    `turned`, each tile transposed (transposed_tiles)."""
    array = np.asarray(data)
    shape = padded_shape(array.shape, layout)
    if layout == "tile" and shape != array.shape:
        pad = [(0, 0)] * (array.ndim - 2) + [
            (0, padded - size) for padded, size in zip(shape[-2:], array.shape[-2:])
        ]
        array = np.pad(array, pad)
    if check_dtype(dtype) in BLOCK and layout != "tile":
        raise ValueError(f"{dtype}: a block format is laid out in tiles, not rows")
    if turned and layout != "tile":
        raise ValueError("a tensor of transposed tiles is laid out in tiles")
    if turned:
        array = transposed_tiles(array)
    if dtype == "bfloat16":
        array = bf16_bits(array.astype(np.float32, copy=False))
    elif dtype == "float16":
        array = half_bits(array)
    elif dtype not in BLOCK:
        array = array.astype(DTYPES[dtype], copy=False)
    if layout == "tile":
        array = np.concatenate(
            [tilize(matrix) for matrix in array.reshape(-1, *shape[-2:])]
        )
    if dtype in BLOCK:
        array = bfp_encode(array, BLOCK[dtype], family_a=FAMILY[dtype] == "a")
    return np.ascontiguousarray(array).view(np.uint8).reshape(-1), shape


def from_bytes(raw, shape, dtype, layout, turned=False):
    """The values of a buffer's bytes (to_bytes' inverse); bfloat16 as float32."""
    raw = np.ascontiguousarray(raw, np.uint8)
    if dtype == "bfloat16":
        array = bf16_values(raw.view(np.uint16))
    elif dtype == "float16":
        array = raw.view(np.float16).astype(np.float32)
    elif check_dtype(dtype) in BLOCK:
        array = bfp_decode(raw, BLOCK[dtype], family_a=FAMILY[dtype] == "a")
    else:
        array = raw.view(DTYPES[dtype])
    if layout == "tile":
        rows, cols = shape[-2:]
        array = array.reshape(-1, rows * cols)
        array = np.stack([untilize(tile_slice, rows, cols) for tile_slice in array])
    array = array.reshape(shape)
    return transposed_tiles(array) if turned else array
