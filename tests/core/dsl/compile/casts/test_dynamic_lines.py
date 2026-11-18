"""A body's word in a dynamic handler's lines (an if, a message's word, a send's
rank) and a message's word taken by a body; a plain handler's lines still refuse a
body's word (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render

WALK = "hit = origin & 3;"


class Ray(km.Message):
    origin: int
    depth: int


class Walk(km.Program):
    @km.handler
    def trace(self, count: int):
        for message in self.arrive():
            hit = km.cxx(WALK, takes=[message.origin], gives=int)
            left = km.cxx("left = km_messages.quota();", gives=int)
            if message.depth == 0 or left == 0:
                self.consume()
            else:
                child = Ray(origin=message.origin + hit, depth=message.depth - 1)
                self.send((message.origin + hit) % self.size, child)
                self.consume()


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


def test_a_bodys_words_are_read_by_a_dynamic_handlers_lines():
    compiled = compile_class(Walk, {}, {})
    compute = "\n".join(render(prune(compiled.compute["trace"]), 0))
    assert "const uint32_t origin = message[kRayOrigin];" in compute
    assert "uint32_t hit = 0;" in compute and "uint32_t left = 0;" in compute
    assert "if ((message[kRayDepth] == 0 || left == 0)) {" in compute
    assert "child[kRayOrigin] = (message[kRayOrigin] + hit);" in compute
    sent = "km_messages.send_or_give_up(((message[kRayOrigin] + hit) % km_size)"
    assert sent in compute
    # the words are made before the lines that read them
    assert compute.index("uint32_t left = 0;") < compute.index("|| left == 0")


def test_a_message_word_taken_under_a_handlers_name_is_refused():
    line = line_of('km.cxx("", takes=[message.origin])')
    with pytest.raises(Refused, match=rf"^Shadowed.trace, line {line}, .*origin is a"):
        made(Shadowed)


def test_a_plain_handlers_lines_refuse_a_bodys_word():
    line = line_of("if left > 0:")
    with pytest.raises(Refused, match=rf"^Plain.count, line {line}, .*left is a word"):
        made(Plain)
