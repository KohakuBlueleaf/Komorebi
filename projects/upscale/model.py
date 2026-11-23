"""QuickSRNet-Medium 2x on the host: the float pass, the emulation of the card's
formats, bicubic and PSNR. An image is float32 (rows, columns, channels); a layer's
weights are the checkpoint's (out, in, 3, 3)."""

import numpy as np

LAYERS = ["cnn.0", "cnn.2", "cnn.4", "cnn.6", "cnn.8", "cnn.10", "conv_last"]
TILE = 32


def load(path):
    """A converted checkpoint's weights (.npz of plain arrays): [(w, b)] by layer."""
    with np.load(path) as z:
        return [(z[f"{n}.weight"], z[f"{n}.bias"]) for n in LAYERS]


def taps(w):
    """(out, in, 3, 3) to nine (in, out) matrices, tap t = 3 dy + dx."""
    return [w[:, :, dy, dx].T for dy in range(3) for dx in range(3)]


def conv(x, w, b):
    """A 3 x 3 convolution with zero padding of one pixel, float32 sums."""
    rows, cols, _ = x.shape
    pad = np.zeros((rows + 2, cols + 2, x.shape[2]), np.float32)
    pad[1:-1, 1:-1] = x
    y = np.zeros((rows, cols, w.shape[0]), np.float32)
    for t, m in enumerate(taps(w)):
        dy, dx = divmod(t, 3)
        y += pad[dy : dy + rows, dx : dx + cols] @ m.astype(np.float32)
    return y + b.astype(np.float32)


def shuffle(y):
    """Pixel shuffle by 2 (torch's order): channel 4 c + 2 i + j at a low-resolution
    pixel (h, w) is colour c of the output pixel (2 h + i, 2 w + j)."""
    rows, cols, _ = y.shape
    y = y.reshape(rows, cols, 3, 2, 2)  # h, w, c, i, j
    return y.transpose(0, 3, 1, 4, 2).reshape(2 * rows, 2 * cols, 3)


def forward(x, weights):
    """The network in float32: six convolutions with Hardtanh(0, 1), the last one, the
    clip to [0, 1], the pixel shuffle."""
    for w, b in weights[:-1]:
        x = np.clip(conv(x, w, b), 0.0, 1.0)
    w, b = weights[-1]
    return shuffle(np.clip(conv(x, w, b), 0.0, 1.0))


# ---- the card's formats ------------------------------------------------------------


def round_bits(a, bits):
    """Round to `bits` significant bits, half to even (bfloat16: 8)."""
    m, e = np.frexp(np.asarray(a, np.float64))
    return np.ldexp(np.rint(np.ldexp(m, bits)), e - bits).astype(np.float32)


def bf16(a):
    return round_bits(a, 8)


def bfp8_b(a):
    """bfp8_b as the host encodes it: each run of 16 consecutive values of a tile row
    shares the largest exponent; magnitudes rounded half to even to 7 bits."""
    g = np.asarray(a, np.float64).reshape(-1, 16)
    top = np.abs(g).max(1, keepdims=True)
    _, e = np.frexp(np.where(top > 0, top, 1.0))
    unit = np.ldexp(1.0, e - 1 - 6)
    q = np.minimum(np.rint(np.abs(g) / unit), 127)
    return (
        np.where(top > 0, np.sign(g) * q * unit, 0.0)
        .astype(np.float32)
        .reshape(np.shape(a))
    )


def weight_tiles(w, b):
    """A layer's ten weight tiles as the card stores them, each TRANSPOSED (out by in,
    the turned product's operand): nine taps, then the bias tile (column 0)."""
    tiles = np.zeros((10, TILE, TILE), np.float32)
    for t, m in enumerate(taps(w)):
        tiles[t, : m.shape[1], : m.shape[0]] = m.T
    tiles[9, : len(b), 0] = b
    return tiles


def emulate(x, weights, keep=None):
    """The card's arithmetic: weights bfp8_b, activations bfloat16, exact products,
    float32 sums, the bias a tenth product, Hardtanh, packed to bfloat16 (the last layer
    too). `keep`, a list, gets each layer's output."""
    x = bf16(x)
    for w, b in weights:
        q = bfp8_b(weight_tiles(w, b))
        cin, cout = w.shape[1], w.shape[0]
        mats = [q[t, :cout, :cin].T for t in range(9)]
        rows, cols, _ = x.shape
        pad = np.zeros((rows + 2, cols + 2, cin), np.float32)
        pad[1:-1, 1:-1] = x
        y = np.zeros((rows, cols, cout), np.float32)
        for t, m in enumerate(mats):
            dy, dx = divmod(t, 3)
            y += pad[dy : dy + rows, dx : dx + cols] @ m
        y += q[9, :cout, 0]
        x = bf16(np.clip(y, 0.0, 1.0))
        if keep is not None:
            keep.append(x)
    return shuffle(x)


# ---- comparisons --------------------------------------------------------------------


def psnr(a, b):
    """PSNR in dB of two images with values in [0, 1]."""
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")


def box_down(image):
    """The 2 x 2 box average of an image (the network's low-resolution input)."""
    rows, cols, ch = image.shape
    return image.reshape(rows // 2, 2, cols // 2, 2, ch).mean((1, 3)).astype(np.float32)


def _cubic_weights(size, a=-0.5):
    """Keys' cubic for an upscale by 2 with half-pixel centres, edges clamped: for each
    output index, four source indices and their weights."""
    out = np.arange(2 * size)
    src = (out + 0.5) / 2 - 0.5
    base = np.floor(src).astype(np.int64) - 1
    idx = base[:, None] + np.arange(4)[None, :]
    d = np.abs(src[:, None] - idx)
    wt = np.where(
        d <= 1,
        (a + 2) * d**3 - (a + 3) * d**2 + 1,
        np.where(d < 2, a * d**3 - 5 * a * d**2 + 8 * a * d - 4 * a, 0.0),
    )
    return np.clip(idx, 0, size - 1), wt / wt.sum(1, keepdims=True)


def bicubic2(image):
    """Bicubic upscaling by 2 (Keys, a = -0.5), clipped to [0, 1]."""
    rows, cols, _ = image.shape
    ri, rw = _cubic_weights(rows)
    ci, cw = _cubic_weights(cols)
    tmp = (image[ri] * rw[:, :, None, None]).sum(1)
    out = (tmp[:, ci] * cw[None, :, :, None]).sum(2)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def to8(image):
    """[0, 1] to 8-bit display values."""
    return np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
