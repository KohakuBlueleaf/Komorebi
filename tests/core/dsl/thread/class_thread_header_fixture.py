"""What test_class_thread_header.py and other tests share: COMPILER, Counted, HERE,
Pages, Plain, Span, Sums, TREE, grid, quiet, twice."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi import _env
from komorebi.device.tensor import Tensor

HERE = Path(__file__).resolve().parent / "class_thread"
TREE = _env.find_tree()[0]


COMPILER = pytest.mark.skipif(
    TREE is None or _env.x280_compiler(TREE) is None, reason="no X280 compiler"
)


def quiet(line):
    pass


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages, address):
        self.page_size, self.num_pages, self.address = page_size, pages, address
        self.aligned_page_size = page_size

    def accessor_args(self):
        return [2, self.page_size]


def grid(columns, address=0x1000):
    """A bfloat16 tensor of one row of `columns` tiles."""
    return Tensor(Pages(2048, columns, address), (32, columns * 32), "bfloat16", "tile")


@km.tile_function
def twice(a):
    return a + a


class Plain(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, tiles: int):
        results = km.Stream(out.pages(0, tiles), write=True)
        for x in km.Stream(a.pages(0, tiles)):
            results.push(twice(x))


class Span(km.Entry):
    first: int
    count: int


class Sums(km.Program):
    @km.handler
    def total(self, counts: km.Region, at: int):
        for span in counts.pages(at, 1).entries(Span, size=2, fixed=1):
            items = counts.items(span.first, span.count, size=64)
            for page in km.Stream(items, write=True):
                km.cxx("fill(page);", takes=[page], on="pack")


class Counted(Sums):
    SIZE = km.constant(8192)
    counts = km.region(SIZE)
    thread = km.x280_thread(str(HERE / "counted.c"), cluster=2)
