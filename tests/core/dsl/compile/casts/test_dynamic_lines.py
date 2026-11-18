"""A body's word in a dynamic handler's lines (an if, a message's word, a send's
rank) and a message's word taken by a body; a plain handler's lines still refuse a
body's word (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.dsl.compile.records import Refused


class Ray(km.Message):  # the module's message: the loops below take it
    origin: int
    depth: int


class Shadowed(km.Program):
    @km.handler
    def trace(self, origin: int):
        for message in self.arrive():
            km.cxx("", takes=[message.origin])
            self.consume()


class Plain(km.Program):
    @km.handler
    def count(self, n: int):
        left = km.cxx("left = 1;", gives=int)
        if left > 0:
            km.cxx("")


def line_of(text):
    lines = Path(__file__).read_text().splitlines()
    return next(number for number, line in enumerate(lines, 1) if text in line)


def made(cls):
    return cls(log=lambda line: None)


def test_a_message_word_taken_under_a_handlers_name_is_refused():
    line = line_of('km.cxx("", takes=[message.origin])')
    with pytest.raises(Refused, match=rf"^Shadowed.trace, line {line}, .*origin is a"):
        made(Shadowed)


def test_a_plain_handlers_lines_refuse_a_bodys_word():
    line = line_of("if left > 0:")
    with pytest.raises(Refused, match=rf"^Plain.count, line {line}, .*left is a word"):
        made(Plain)
