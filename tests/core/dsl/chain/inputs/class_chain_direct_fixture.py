"""What test_class_chain_direct.py and other tests share: DirectSum, halves."""

from chain_fixture import Steps

import komorebi as km


def halves(self, a: km.Tensor, out: km.Tensor):
    """Each core's 2 tiles of out, each written by .write (a root sends it as written:
    a write stream's blocks go to DRAM)."""
    first = self.rank * 2
    left = km.tile.zeros(km.fp32)
    for tile in km.Stream(a.pages(first, 1)):
        left += tile
    out.pages(first, 1).write(left)
    right = km.tile.zeros(km.fp32)
    for tile in km.Stream(a.pages(first + 1, 1)):
        right += tile
    out.pages(first + 1, 1).write(right)


class DirectSum(Steps):
    """Steps whose sum may take s direct: a K share of it a core."""

    sum = km.handler(direct=("s",))(Steps.sum.function)
    halves = km.handler(halves)
