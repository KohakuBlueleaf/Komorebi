"""The handlers' reduces share one arrival buffer (kArrival0): it is sized for the
largest of them, a slot of the most tiles a reduce sends and as many slots as the most
rounds, whichever handler comes first (no card)."""

from chain_fixture import Pages, quiet, tensor

import komorebi as km
from komorebi.tensor import Tensor


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


def arrival(cls):
    program = cls(
        l1=1 << 20,
        log=quiet,
        x=tiles(1, 30, 0x10000),
        w=tiles(30, 8, 0x20000),
        y=tiles(1, 8, 0x90000),
        s=tensor(120, 0xA0000),
        t=tensor(1, 0x120000),
    )
    header = program.built.folder / f"{cls.__name__.lower()}.h"
    line = next(
        line for line in header.read_text().splitlines() if "Arrival kArrival0" in line
    )
    return tuple(int(word) for word in line.split("{")[1].split("}")[0].split(","))


def test_a_shared_arrival_is_sized_for_the_largest_reduce():
    index, pages, per = arrival(Small)
    assert per == 8  # a slot of the most tiles a reduce sends (the sums' 8, not 1)
    assert pages == 4 * 8  # the most rounds: the tree of 15's four


def test_whichever_handler_comes_first():
    assert arrival(Large)[1:] == arrival(Small)[1:] == (32, 8)
