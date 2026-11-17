"""What test_class_constants.py and other tests share: Pages, grid, project."""

import komorebi as km
from komorebi.device.tensor import Tensor


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def grid(rows, columns):
    """A bfloat16 tensor of tiles, rows x columns of them."""
    return Tensor(Pages(2048, rows * columns), (rows * 32, columns * 32), "bfloat16",
                  "tile")  # fmt: skip


@km.tile_function
def project(x, w):
    return x @ w
