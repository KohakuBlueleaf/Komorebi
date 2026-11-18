"""What test_class_handler_constants.py and other tests share: Gate, Pages, TENSORS,
grid, quiet, twice."""

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


def quiet(line):
    pass


TENSORS = {"a": grid(1, 8), "out": grid(1, 8)}


@km.tile_function
def twice(a):
    return a + a


class Gate(km.Program):
    @km.handler
    def gate(
        self, a: km.Tensor, out: km.Tensor, tiles: int, size: km.constant[int] = 8
    ):
        results = km.Stream(out.pages(0, tiles), write=True, block=size)
        for x in km.Stream(a.pages(0, tiles), block=size):
            results.push(twice(x))

    @km.handler
    def other(self, a: km.Tensor):
        km.Stream(a.pages(0, 1)).drain()
