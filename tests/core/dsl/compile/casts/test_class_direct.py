"""direct() from a class: `went = self.direct(rank, message)` in a dynamic handler is
the runtime's direct call as a word; the call anywhere else is refused (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.dsl.compile.records import Refused


class Token(km.Message):  # the module's message: the loops below take it
    hops: int


class Unnamed(km.Program):
    @km.handler
    def ring(self):
        for message in self.arrive():
            self.direct(self.rank, message)


class InTest(km.Program):
    @km.handler
    def ring(self):
        for message in self.arrive():
            if self.direct(self.rank, message) == 0:
                self.forward(self.rank, message)


class Static(km.Program):
    @km.handler
    def copy(self, rank: int):
        went = self.direct(rank, rank)
        assert went


def line_of(text):
    lines = Path(__file__).read_text().splitlines()
    return next(number for number, line in enumerate(lines, 1) if text in line)


def made(cls):
    return cls(log=lambda line: None)


def test_direct_whose_value_is_dropped_or_tested_inline_is_refused():
    line = line_of("            self.direct(self.rank, message)")
    with pytest.raises(Refused, match=rf"^Unnamed.ring, line {line}, .*says whether"):
        made(Unnamed)
    line = line_of("if self.direct(self.rank, message) == 0:")
    with pytest.raises(Refused, match=rf"^InTest.ring, line {line}, .*says whether"):
        made(InTest)


def test_direct_outside_a_dynamic_handler_is_refused():
    line = line_of("went = self.direct(rank, rank)")
    with pytest.raises(Refused, match=rf"^Static.copy, line {line}"):
        made(Static)
