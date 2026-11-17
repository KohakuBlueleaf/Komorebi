"""direct() from a class: `went = self.direct(rank, message)` in a dynamic handler is
the runtime's direct call as a word; the call anywhere else is refused (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.nodes import prune, render


class Token(km.Message):
    hops: int


class Ring(km.Program):
    @km.handler
    def ring(self, hops: int):
        if self.seeding():
            seed = Token(hops=hops)
            self.send(self.rank, seed)
        for message in self.arrive():
            if message.hops == 0:
                self.consume()
            else:
                message.hops -= 1
                went = self.direct((self.rank + 1) % self.size, message)
                if went == 0:
                    self.forward((self.rank + 1) % self.size, message)


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


def test_a_named_direct_is_the_runtimes_call_as_a_word():
    compiled = compile_class(Ring, {}, {})
    compute = "\n".join(render(prune(compiled.compute["ring"]), 0))
    assert "const uint32_t went = (km_messages.direct(" in compute
    assert "? 1u : 0u);" in compute
    assert compute.index("km_messages.direct(") < compute.index("km_messages.forward(")


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
