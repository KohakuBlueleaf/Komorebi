"""What test_class_movers.py and other tests share: K, N, made, nocs, tensors."""

from chain_fixture import Pages, quiet

from komorebi import formats, layout
from komorebi.device.tensor import Tensor

K, N = 8, 4


def tensors():
    def made(shape, address, dtype="bfloat16"):
        pages = shape[0] * shape[1] // 1024
        page = formats.tile_bytes(dtype)
        return Tensor(Pages(page, pages, address), shape, dtype, "tile")

    return {
        "x": made((128, K * 32), 0x100000),
        "w": made((K * 32, 1024), 0x1000000),
        "y": made((128, 512), 0x4000000, "float32"),
    }


def made(cls):
    return cls(l1=1 << 20, log=quiet, rect=(0, 0, 2, 4), **tensors())


def nocs():
    """A 2 x 4 grid's NoC words (made up: x + 1, y + 2), for the arguments' layout."""
    order = layout.core_order((2, 4))
    words = [(x + 1) | (y + 2) << 16 for x, y in order]
    return layout.nocs((2, 4), words, words)
