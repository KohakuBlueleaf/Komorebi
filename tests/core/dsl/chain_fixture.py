"""A class whose handlers make a chain (example 29's steps, and handlers a chain
refuses), its made program, tensors and a run on the fake backend: what the tests of
a chain stated with a class share (no card)."""

import numpy as np
from fake_port import FakeBackend, FakePort

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.session.run import Run

ADDRESS = 0x10_0000
WORDS = 1024
SCALE = """
#include "api/compute/eltwise_unary/binop_with_scalar.h"
copy_init(tile_buffer);
copy_tile(tile_buffer, tile_index, 0);
binop_with_scalar_tile_init();
mul_unary_tile(0, factor);
"""


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages, address):
        self.page_size, self.num_pages, self.address = page_size, pages, address

    def accessor_args(self):
        return [2, self.page_size]


def tensor(rows, address):
    """A float32 tensor of `rows` pages of one tile's words."""
    return Tensor(Pages(4096, rows, address), (rows, WORDS), "float32", "row")


class Block:
    def __init__(self, data):
        self.address, self.data, self.freed = ADDRESS, bytes(data), False

    def free(self):
        self.freed = True


class BlockBackend(FakeBackend):
    """A fake backend that keeps a chain's records."""

    def __init__(self, port):
        super().__init__(port)
        self.blocks = []

    def block(self, data, place=None):
        self.blocks.append(Block(data if place is None else place(ADDRESS)))
        return self.blocks[-1]

    def fence(self):
        pass


def quiet(line):
    pass


@km.tile_function
def plus(a, b):
    return a + b


class Handlers(km.Program):
    """Example 29's steps: SCALE, ADD (a core's width tiles from rank x width), SUM
    (a share of the tiles up a tree to the root); and handlers a chain refuses."""

    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, factor: int, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            scaled = km.cxx(SCALE, takes=[tile, factor], gives=km.Tile)
            results.push(scaled)

    @km.handler(direct=("b",))
    def add(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for x, y in zip(
            km.Stream(a.pages(first, width)), km.Stream(b.pages(first, width))
        ):
            results.push(plus(x, y))

    @km.handler
    def sum(self, s: km.Tensor, t: km.Tensor, tiles: int):
        part = self.partition(s.pages(0, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s[part]):
            total += tile
        whole = km.reduce(total, form="tree")
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)

    @km.handler
    def add_next(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for x, y in zip(
            km.Stream(a.pages(first, width)), km.Stream(b.pages(first + width, width))
        ):
            results.push(plus(x, y))

    @km.handler
    def early(self, a: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width), prefetch=True):
            results.push(tile)

    @km.handler
    def late(self, a: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width), prefetch=False):
            results.push(tile)

    @km.handler
    def looped(self, a: km.Tensor, n: int):
        for i in range(n):
            km.Stream(a.pages(i * self.rank, 1)).drain()

    @km.handler
    def scattered(self, a: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first + 1, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)

    @km.handler
    def owner_sum(self, s: km.Tensor, t: km.Tensor, tiles: int):
        part = self.partition(s.pages(0, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s[part]):
            total += tile
        whole = km.reduce(total, form="owner")
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)

    @km.handler
    def bodied(self, a: km.Tensor, n: int):
        km.cxx("km::Tensor<0> unused = a; (void)unused;", takes=[a], on="mover_a")


class Steps(Handlers):
    """The handlers with the chain's walk: example 29's steps as a km.chain method."""

    @km.chain
    def steps(self, x, y, z, s, t, factor: int, width: int, tiles: int):
        self.scale(x, y, factor, width)
        self.scale(y, z, THREE, width)
        self.add(z, km.past(y), s, width)
        self.sum(s, t, tiles)

    @km.chain
    def twice(self, x, y, factor: int, width: int):
        for _ in range(COUNT):
            self.scale(x, y, factor, width)


THREE = int(np.float32(3).view(np.uint32))
COUNT = 2
CORES, WIDTH = 8, 2
TILES = CORES * WIDTH


def tensors():
    """x, y, z, s (TILES pages each) and t (one page), at addresses apart."""
    rows = {"x": TILES, "y": TILES, "z": TILES, "s": TILES, "t": 1}
    return {
        name: tensor(count, 0x10000 * (index + 1))
        for index, (name, count) in enumerate(rows.items())
    }


def made(cls, given):
    """`cls` made with an example of each tensor parameter."""
    x = given["x"]
    return cls(
        l1=1 << 20, log=quiet, a=x, b=x, out=given["y"], s=given["s"], t=given["t"]
    )


def run_of(program):
    return Run(BlockBackend(FakePort()), (12, 10), program=program)


def bits(factor):
    return int(np.float32(factor).view(np.uint32))


def host_list(given, factor=5):
    """Example 29's chain as the host's list of calls."""
    x, y, z, s, t = (given[name] for name in "xyzst")
    return [
        Steps.scale(x, y, bits(factor), WIDTH),
        Steps.scale(y, z, THREE, WIDTH),
        Steps.add(z, km.past(y), s, WIDTH),
        Steps.sum(s, t, TILES),
    ]
