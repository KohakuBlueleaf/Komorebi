"""What test_broadcast_product.py and other tests share: K, N, project, tensors."""

from chain_fixture import Pages

import komorebi as km
from komorebi import formats
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
        "wt": made((1024, K * 32), 0x2000000),
        "y": made((128, 512), 0x4000000, "float32"),
    }


@km.tile_function
def project(x, w):
    return x @ w
