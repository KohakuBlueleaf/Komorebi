"""An inline body's text from a string constant of the class's module, or a sum of
such constants and strings, the same in several handlers; any other text is refused
by its line (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.dsl.compiler import Refused, compile_class
from komorebi.dsl.nodes import prune, render
from komorebi.tensor import Tensor

HEADER = """
#include "shared.h"
"""
STEP = """
step(words);
"""
NOT_TEXT = 3


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self):
        self.page_size, self.num_pages, self.address = 4096, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


class Shared(km.Program):
    @km.handler
    def first(self, words: km.Tensor):
        km.cxx(HEADER + STEP, takes=[words], on="mover_b")

    @km.handler
    def second(self, words: km.Tensor):
        km.cxx(STEP + "again(words);", takes=[words], on="mover_b")


class Numbered(km.Program):
    @km.handler
    def first(self, words: km.Tensor):
        km.cxx(NOT_TEXT, takes=[words], on="mover_b")


def line_of(text):
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


def test_a_body_of_module_constants_is_their_text_at_the_calls_line():
    words = Tensor(Pages(), (64, 1024), "uint32", "tile")
    compiled = compile_class(Shared, {}, {"words": words})
    first = "\n".join(render(prune(compiled.movement["first"]), 0))
    second = "\n".join(render(prune(compiled.movement["second"]), 0))
    assert "step(words);" in first and "step(words);" in second
    assert "again(words);" in second
    line = line_of("km.cxx(HEADER + STEP")
    assert f'#line {line} "' in first
    assert any('"shared.h"' in entry for entry in compiled.body_includes["movement"])


def test_a_body_of_another_value_is_refused_by_its_line():
    words = Tensor(Pages(), (64, 1024), "uint32", "tile")
    line = line_of("km.cxx(NOT_TEXT")
    with pytest.raises(
        Refused, match=rf"^Numbered.first, line {line}, .*a sum of them"
    ):
        compile_class(Numbered, {}, {"words": words})
