"""A class's build key holds the field sizes solved from the L1 left, not that L1: two
makes whose L1 solves the same sizes share a build folder (a mock session's and the
card's), and a solved size that differs gives another (no card)."""

import komorebi as km
from komorebi.device.tensor import Tensor


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 2048, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor():
    return Tensor(Pages(2048), (2048, 32), "bfloat16", "tile")


class Solved(km.Program):
    TILES = km.symbol()
    count: km.L1[km.fp32, TILES, 32, 32]

    @km.handler
    def nothing(self, source: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


class Fixed(km.Program):
    count: km.L1[km.fp32, 4, 32, 32]

    @km.handler
    def nothing(self, source: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


def made(cls, l1):
    return cls(log=lambda line: None, l1=l1, source=tensor())


def test_two_rooms_that_solve_the_same_sizes_share_a_build():
    one, two = made(Solved, 1_000_000), made(Solved, 1_000_000 + 2_000)
    assert "  plan: TILES = 240" in one.built.report
    assert "  plan: TILES = 240" in two.built.report
    assert one.built.folder == two.built.folder
    assert made(Fixed, 1_434_752).built.folder == made(Fixed, 1_345_584).built.folder


def test_a_solved_size_that_differs_gives_another_build():
    one, other = made(Solved, 1_000_000), made(Solved, 1_100_000)
    assert "  plan: TILES = 264" in other.built.report
    assert one.built.folder != other.built.folder
