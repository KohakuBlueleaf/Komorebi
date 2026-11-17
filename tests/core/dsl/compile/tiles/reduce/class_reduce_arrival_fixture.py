"""What test_class_reduce_arrival.py and other tests share: Large, Small, tiles."""

from chain_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor


def tiles(rows, columns, address):
    """A bfloat16 tile tensor of rows x columns tiles."""
    shape = (rows * 32, columns * 32)
    return Tensor(Pages(2048, rows * columns, address), shape, "bfloat16", "tile")


class Small(km.Program):
    """Sums of 8 tiles up a tree of 2 (one round), then of 15 (four rounds), then one
    tile up a tree of 7 (three rounds): the first reduce the smallest."""

    sums: km.L1[km.fp32, 1, 8, 32, 32]

    @km.tile_function
    def project(x, w):
        return x @ w

    @km.handler
    def two(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        k0 = self.rank % 2 * 2
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + 2], block=2),
            km.Stream(w.tiles[k0 : k0 + 2, 0:8], block=16),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        if self.rank == self.leader:
            y.tiles[0:1, 0:8].write(whole)

    @km.handler
    def fifteen(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        k0 = self.rank % 15 * 2
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + 2], block=2),
            km.Stream(w.tiles[k0 : k0 + 2, 0:8], block=16),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=15)
        if self.rank == self.leader:
            y.tiles[0:1, 0:8].write(whole)

    @km.handler
    def seven(self, s: km.Tensor, t: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            total += tile
        whole = km.reduce(total, form="tree", group=7)
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)


class Large(Small):
    """The same reduces, the tree of 15 first."""

    @km.handler
    def first(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        k0 = self.rank % 15 * 2
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + 2], block=2),
            km.Stream(w.tiles[k0 : k0 + 2, 0:8], block=16),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=15)
        if self.rank == self.leader:
            y.tiles[0:1, 0:8].write(whole)
