"""A merge of classes from two files: each handler's inline body names its own file
in its #line, and a field's refusal quotes its own class's file (no card)."""

from pathlib import Path

import pytest
from merged_file_fixture import Elsewhere

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused

HERE = str(Path(__file__).resolve())
THERE = str(Path(Elsewhere.stepped.function.__code__.co_filename).resolve())
STEP = """
here(words);
"""


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self):
        self.page_size, self.num_pages, self.address = 4096, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


class Local(km.Program):
    @km.handler
    def local(self, words: km.Tensor):
        km.cxx(STEP, takes=[words], on="mover_b")


class Solved(km.Program):
    A_TILES = km.symbol()
    a_part: km.L1[km.fp32, A_TILES, 32, 32]

    @km.handler
    def local(self, words: km.Tensor):
        km.cxx(STEP, takes=[words], on="mover_b")


def quiet(line):
    pass


def made(cls):
    words = Tensor(Pages(), (64, 1024), "uint32", "tile")
    return cls(l1=1 << 20, log=quiet, words=words)


def line_of(path, text):
    lines = Path(path).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


def test_each_handlers_body_names_its_own_file():
    merged = km.merge(made(Local), made(Elsewhere), log=quiet, l1=1 << 20)
    assert merged.compiled.source == HERE  # the build's folder: the first's


def test_a_fields_refusal_quotes_its_own_classs_file():
    with pytest.raises(Refused) as error:
        km.merge(made(Solved), made(Elsewhere), log=quiet, l1=1 << 20)
    line = line_of(THERE, "b_part: km.L1")
    assert f"b_part, line {line}, `b_part: km.L1[km.fp32, B_TILES, 32, 32]`" in str(
        error.value
    )
