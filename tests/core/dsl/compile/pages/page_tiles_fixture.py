"""A tile tensor in DRAM pages of k tiles, as a class's compile needs it (no card): the
test_page_tiles and test_class_shared_pair examples."""

from komorebi.device.tensor import Tensor

TILE = {"bfloat16": 2048, "bfp8_b": 1088, "bfp4_b": 576}


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(rows, columns, dtype="bfp8_b", k=1):
    """A tile tensor of rows x columns tiles in pages of k tiles."""
    made_ = Tensor(
        Pages(k * TILE[dtype], rows * columns // k),
        (rows * 32, columns * 32),
        dtype,
        "tile",
    )
    made_.page_tiles = k
    return made_
