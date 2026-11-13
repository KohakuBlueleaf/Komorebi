"""The bytes of device buffers, made and read on the host in numpy: bfloat16 values,
32 x 32 tiles of four 16 x 16 faces, and whole buffers of rows or tiles.
"""

import numpy as np

TILE = 32
FACE = 16
LAYOUTS = ("row", "tile")
# The data types a buffer holds, by name, and their numpy type on the host side.
DTYPES = {
    "bfloat16": np.dtype(np.float32),
    "float32": np.dtype(np.float32),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "uint16": np.dtype(np.uint16),
    "uint8": np.dtype(np.uint8),
}
BF16_NAN = 0x7FC0  # the quiet NaN a NaN becomes


def itemsize(dtype):
    """Bytes of one element of `dtype` on the card."""
    return 2 if dtype == "bfloat16" else DTYPES[check_dtype(dtype)].itemsize


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


def bf16_values(bits):
    """bfloat16 bits (uint16) as float32 values."""
    wide = np.asarray(bits, np.uint16).astype(np.uint32) << 16
    return wide.view(np.float32)


def tilize(a):
    """A [rows, cols] array (multiples of 32) as [tiles, 1024]: tiles in row-major
    order, each its four 16 x 16 faces in turn, each face row-major."""
    rows, cols = a.shape
    t = a.reshape(rows // TILE, 2, FACE, cols // TILE, 2, FACE)
    return t.transpose(0, 3, 1, 4, 2, 5).reshape(-1, TILE * TILE)


def untilize(tiles, rows, cols):
    """The [rows, cols] array of tiles laid out as tilize() lays them."""
    t = np.asarray(tiles).reshape(rows // TILE, cols // TILE, 2, 2, FACE, FACE)
    return t.transpose(0, 2, 4, 1, 3, 5).reshape(rows, cols)


def padded_shape(shape, layout):
    """The shape on the card: the last two dimensions of a tile layout rounded up to
    32."""
    shape = tuple(int(n) for n in shape)
    if layout == "row":
        return shape
    if len(shape) < 2:
        raise ValueError(f"shape {shape}: a tile layout needs two dimensions")
    return (*shape[:-2], *(-(-n // TILE) * TILE for n in shape[-2:]))


def page_bytes(shape, dtype, layout):
    """One page: a row (row layout) or a tile."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout!r}: 'row' or 'tile'")
    per = TILE * TILE if layout == "tile" else shape[-1]
    return per * itemsize(dtype)


def to_bytes(data, dtype, layout):
    """`data` as the buffer's bytes (uint8) and its shape on the card. A tile layout
    zero-pads the last two dimensions to 32 and tiles every [rows, cols] slice."""
    a = np.asarray(data)
    shape = padded_shape(a.shape, layout)
    if layout == "tile" and shape != a.shape:
        pad = [(0, 0)] * (a.ndim - 2) + [
            (0, p - n) for p, n in zip(shape[-2:], a.shape[-2:])
        ]
        a = np.pad(a, pad)
    if dtype == "bfloat16":
        a = bf16_bits(a.astype(np.float32, copy=False))
    else:
        a = a.astype(DTYPES[check_dtype(dtype)], copy=False)
    if layout == "tile":
        a = np.concatenate([tilize(s) for s in a.reshape(-1, *shape[-2:])])
    return np.ascontiguousarray(a).view(np.uint8).reshape(-1), shape


def from_bytes(raw, shape, dtype, layout):
    """The values of a buffer's bytes (to_bytes' inverse); bfloat16 as float32."""
    raw = np.ascontiguousarray(raw, np.uint8)
    if dtype == "bfloat16":
        a = bf16_values(raw.view(np.uint16))
    else:
        a = raw.view(DTYPES[check_dtype(dtype)])
    if layout == "tile":
        rows, cols = shape[-2:]
        a = a.reshape(-1, rows * cols)
        a = np.stack([untilize(s, rows, cols) for s in a])
    return a.reshape(shape)
