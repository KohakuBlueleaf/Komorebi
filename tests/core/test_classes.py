"""Programs written as classes, off the card: the words a call gives, the files and
buffers a build makes, a field's symbol solved against the L1, and refusals that name
the line."""

from types import SimpleNamespace

import pytest

import komorebi as km
from komorebi.dsl.compiler import Refused
from komorebi.dsl.program import lines_of
from komorebi.tensor import Tensor

ROWS = 32 * 64


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, address):
        self.page_size, self.num_pages, self.address = page_size, 64, address

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype, address=0x1000):
    page = 2048 if dtype == "bfloat16" else 4096
    return Tensor(Pages(page, address), (ROWS, 32), dtype, "tile")


class Sum(km.Program):
    # TILES: a symbol the build solves, not a name (F821 reads it as one)
    count: km.L1[km.fp32, ("TILES", 32, 32)]  # noqa: F821

    @km.handler
    def add(self, source: km.Tensor, tiles: int, output: km.Tensor, page: int):
        part = self.partition(source.pages(0, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(source[part], block=4):
            total += tile
        output.pages(page + self.rank, 1).write(total)


class Sliced(km.Program):
    @km.handler
    def copy(self, source: km.Tensor):
        for _ in km.Stream(source[1:4]):
            pass


class Unled(km.Program):
    @km.handler
    def merge(self, source: km.Tensor, output: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(source.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        output.pages(0, 1).write(whole)


def test_a_build_and_its_words():
    source, output = tensor("bfloat16"), tensor("float32", 0x9000)
    program = Sum(source=source, output=output, l1=1 << 20, log=lambda line: None)
    files = program.built.files
    assert int(Sum.add) == 1 and program.handler("add") == 1
    assert Sum.add.args(source, 64, output, 3) == [0x1000, 64, 0x9000, 3]
    assert "km::Stream read0(" in files["movement.cpp"]
    assert "add_binary_tile(0, 1, 0);" in files["compute.cpp"]
    # the read pair: four blocks of four tiles a half; the field takes what is left
    assert [(b.index, b.pages) for b in program.buffers[:2]] == [(0, 16), (1, 16)]
    taken = sum(b.page_bytes * b.pages for b in program.buffers[:4])
    field = program.buffers[-1]
    assert field.index == 8 and field.page_bytes == 4096
    assert field.pages == ((1 << 20) - taken) // 4096
    assert "plan: TILES = " in "\n".join(program.built.report)


def test_a_symbol_needs_the_l1():
    with pytest.raises(Refused, match="l1=session.l1"):
        Sum(source=tensor("bfloat16"), output=tensor("float32"), log=lambda _: None)


def test_refusals_name_the_line():
    with pytest.raises(Refused, match=r"Sliced.copy, line \d+, `for _ in .*a slice"):
        Sliced(source=tensor("bfloat16"), log=lambda line: None)
    with pytest.raises(
        Refused, match=r"Unled.merge, line \d+, `output.*only the leader"
    ):
        Unled(source=tensor("float32"), output=tensor("float32"), log=lambda _: None)


def test_a_timeout_names_the_line():
    program = Sum(
        source=tensor("bfloat16"),
        output=tensor("float32"),
        l1=1 << 20,
        log=lambda line: None,
    )
    error = SimpleNamespace(cores=[{"core": 5, "compute": {"where": "cinput"}}])
    (named,) = lines_of(program, error)
    assert named.startswith("core 5 compute: add, line ")
    assert "for tile in km.Stream(source[part], block=4):" in named
