"""The handlers' reduces share one arrival buffer (kArrival0): it is sized for the
largest of them, a slot of the most tiles a reduce sends and as many slots as the most
rounds, whichever handler comes first (no card)."""

from types import SimpleNamespace

from chain_fixture import Pages, quiet, tensor

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile import tile_reduce


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


def test_a_tree_over_the_commands_group_has_a_slot_each_island_part_more():
    # over islands: the root part's rounds and a slot each other part (a line of 3
    # islands of 47, 34, 35 ranks: 6 + 2 = 8, past the flat tree's 7: given up)
    def slots(islands, size=None):
        made = SimpleNamespace(compiled=SimpleNamespace(islands=islands))
        return tile_reduce.tree_slots(made, size)

    assert slots((1, 0)) == tile_reduce.TREE_ROUNDS == 7  # one island: as before
    assert slots((3, 47)) == 8 and slots((2, 118)) == 8
    assert slots((2, 40)) == 7  # 6 + 1: the flat tree's 7 still the most
    assert slots((1, 0), size=15) == 4  # group=S on one island: its own rounds


def test_a_tree_in_groups_of_s_over_islands_has_a_slot_each_island_it_can_touch_more():
    """group=S on islands: the largest part's rounds and a slot each further island
    the group can touch (km::reduce_slots), as over the command's group; a group of 7
    split 5 | 2 at an island's edge needs 3 + 1, past one island's 3."""

    def slots(islands, size):
        made = SimpleNamespace(compiled=SimpleNamespace(islands=islands))
        return tile_reduce.tree_slots(made, size)

    assert slots((1, 0), 7) == 3
    assert slots((3, 47), 7) == 3 + 2 and slots((2, 59), 7) == 3 + 1
    assert slots((4, 29), 64) == 5 + 3  # larger than an island: parts of 29 at most
    assert slots((4, 29), 2) == 1 + 1  # two ranks touch two islands at most
