"""A program written as a class that the compiler does not take is refused when it is
made, with a sentence that names the handler and the line (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.dsl.compiler import Refused
from komorebi.tensor import Tensor


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype):
    return Tensor(
        Pages(2048 if dtype == "bfloat16" else 4096), (2048, 32), dtype, "tile"
    )


class Sliced(km.Program):
    @km.handler
    def copy(self, source: km.Tensor):
        km.Stream(source[1:4]).drain()


class Unled(km.Program):
    @km.handler
    def merge(self, source: km.Tensor, output: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(source.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        output.pages(0, 1).write(whole)


class Unsolved(km.Program):
    TILES = km.symbol()
    count: km.L1[km.fp32, TILES, 32, 32]

    @km.handler
    def nothing(self, source: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


def made(cls, **tensors):
    return cls(log=lambda line: None, **tensors)


def line_of(text):
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


def test_a_refusal_names_the_handler_and_the_line():
    line = line_of("km.Stream(source[1:4]).drain()")
    with pytest.raises(Refused, match=rf"^Sliced.copy, line {line}, `km.Stream\("):
        made(Sliced, source=tensor("bfloat16"))
    line = line_of("output.pages(0, 1).write(whole)")
    with pytest.raises(Refused, match=rf"^Unled.merge, line {line}, `output.pages"):
        made(Unled, source=tensor("float32"), output=tensor("float32"))


def test_a_symbol_left_to_the_build_needs_the_l1():
    line = line_of("count: km.L1[km.fp32, TILES, 32, 32]")
    with pytest.raises(Refused, match=rf"^Unsolved.count, line {line}, .*l1=session"):
        made(Unsolved, source=tensor("bfloat16"))


@km.tile_function
def product(a, b):
    return a @ b


class Product(km.Program):
    @km.handler
    def multiply(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 1), write=True)
        for x, y in zip(km.Stream(a.pages(0, 1)), km.Stream(b.pages(0, 1))):
            results.push(product(x, y))


def test_a_matrix_product_in_a_tile_function_is_refused_by_its_line():
    line = line_of("results.push(product(x, y))")
    product_line = line_of("return a @ b")
    with pytest.raises(
        Refused, match=rf"^Product.multiply, line {line}, .*line {product_line}, has a"
    ):
        made(Product, a=tensor("float32"), b=tensor("float32"), out=tensor("float32"))
