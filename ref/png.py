"""Writing PNG files and mapping scalar images to colour, with numpy and zlib only."""

import struct
import zlib
from pathlib import Path

import numpy as np

# Anchor colours of a perceptual colour map (dark to bright), interpolated linearly.
VIRIDIS = np.array(
    [
        [68, 1, 84],
        [72, 36, 117],
        [65, 68, 135],
        [53, 95, 141],
        [42, 120, 142],
        [33, 145, 140],
        [34, 168, 132],
        [68, 191, 112],
        [122, 209, 81],
        [189, 223, 38],
        [253, 231, 37],
    ],
    dtype=np.float64,
)


def chunk(tag, data):
    """One PNG chunk: length, tag, data, CRC."""
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def write_png(path, image):
    """Writes a uint8 array (H, W) as grey or (H, W, 3) as RGB."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    color_type = 0 if image.ndim == 2 else 2
    rows = image.reshape(height, -1)
    raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    data += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    Path(path).write_bytes(data)


def read_png(path):
    """Reads a PNG written by `write_png` back into a uint8 array."""
    data = Path(path).read_bytes()
    pos, idat, shape = 8, b"", None
    while pos < len(data):
        (size,) = struct.unpack(">I", data[pos : pos + 4])
        tag, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + size]
        if tag == b"IHDR":
            width, height, _, color_type = struct.unpack(">IIBB", body[:10])
            shape = (height, width) if color_type == 0 else (height, width, 3)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + size
    rows = np.frombuffer(zlib.decompress(idat), dtype=np.uint8)
    return rows.reshape(shape[0], -1)[:, 1:].reshape(shape)


def colormap(values, low=None, high=None):
    """Maps a float image to RGB uint8 with the VIRIDIS anchors between low and high."""
    values = np.asarray(values, dtype=np.float64)
    low = float(values.min()) if low is None else low
    high = float(values.max()) if high is None else high
    x = np.clip((values - low) / max(high - low, 1e-30), 0.0, 1.0)
    pos = x * (len(VIRIDIS) - 1)
    i = np.minimum(pos.astype(np.int64), len(VIRIDIS) - 2)
    frac = (pos - i)[..., None]
    rgb = VIRIDIS[i] * (1 - frac) + VIRIDIS[i + 1] * frac
    return np.rint(rgb).astype(np.uint8)


def to_srgb8(linear):
    """Float RGB in [0, 1] (clamped) to display bytes with a 1/2.2 gamma."""
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0) ** (1 / 2.2)
    return np.rint(x * 255).astype(np.uint8)


def downscale(image, factor):
    """Box-filter downscale of a (H, W, 3) uint8 or float image by an integer factor."""
    h, w = image.shape[:2]
    shape = (h // factor, factor, w // factor, factor) + image.shape[2:]
    out = image.reshape(shape).astype(np.float64).mean(axis=(1, 3))
    return np.rint(out).astype(np.uint8) if image.dtype == np.uint8 else out
