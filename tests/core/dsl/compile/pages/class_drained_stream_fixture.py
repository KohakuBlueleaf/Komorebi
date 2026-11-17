"""What test_class_drained_stream.py and other tests share: Ended, Pages, lists."""

import komorebi as km
from komorebi.device.tensor import Tensor


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def lists():
    """A uint32 tensor of 2048 B pages, 16 items of 128 B a page."""
    return Tensor(Pages(2048, 4), (4, 512), "uint32", "row")


class Ended(km.Program):
    out: km.L1[km.u32, 2, 32, 32] = km.own(push="pack")
    next_slots: km.L1[km.u32, 1, 32, 32] = km.own(push="mover_b")

    @km.handler
    def setup(self, lists: km.Tensor, count: int):
        out = km.Stream(self.out, mover="B")
        for tri in range(count):
            km.cxx("bin(tri, out);", takes=[out, tri], on="pack")
        out.end()
        for page in out:
            slot = self.next_slots.bump(page[1])
            lists.items(size=128).put(page, at=slot)
