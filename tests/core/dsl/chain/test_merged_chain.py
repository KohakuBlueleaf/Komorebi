"""A chain over a merged class: `chain=` at a merged class's make and the walk from a
km.chain method, km.merge of programs made with the walk, each handler's chain waits
read from its own class's file, and a chain's calls named through the merged class's
parents resolved to the merged program (no card)."""

import re

import pytest
from class_first_use_fixture import FakeRun
from class_runtime_form_fixture import Copy, quiet, tiles

import komorebi as km
from komorebi.chain.calls import ChainRefused
from komorebi.session.run import Run


@km.tile_function
def tripled(x):
    return x + x + x


class Scale(km.Program):
    @km.handler
    def scale(self, source: km.Tensor, target: km.Tensor, count: int):
        out = km.Stream(target.pages(self.rank * count, count), write=True)
        for tile in km.Stream(source.pages(self.rank * count, count)):
            out.push(tripled(tile))


class Both(Copy, Scale):
    """Copy (another file) and Scale merged, made with chain=True."""


class Walked(Copy, Scale):
    @km.chain
    def twice(self, a, b, c, count: int):
        self.copy(a, b, count)
        self.scale(b, c, count)


def made(cls, **given):
    return cls(source=tiles(64), target=tiles(64), l1=1 << 20, log=quiet, **given)


class ChainRun(FakeRun):
    _class_calls = Run._class_calls
    _walk = Run._walk
    field_handler = Run.field_handler

    def _group(self, cores):
        return tuple(cores)


def waits(program):
    """The chain's input waits of the movement kernel, by handler case."""
    text = (program.built.folder / "movement.cpp").read_text()
    return sorted(re.findall(r"km_chain_read\(movement, [^)]*\)", text))


def test_chain_at_a_merged_class_s_make_and_from_a_chain_method():
    plain = made(Both)
    assert plain.chain_handler == 0 and "km_chain" not in plain.handlers
    walked = made(Both, chain=True)
    assert walked.chain_handler == 3
    assert walked.handlers == {"copy": 1, "scale": 2, "km_chain": 3}
    assert made(Walked).chain_handler == 3  # its km.chain method gives the walk
    assert walked.built.folder != plain.built.folder  # the walk is in the key


def test_each_handler_s_waits_are_read_from_its_own_file():
    both = made(Both, chain=True)
    sources = both.compiled.handler_sources
    assert sources["copy"].endswith("class_runtime_form_fixture.py")
    assert sources["scale"] == __file__
    merged = waits(both)
    alone = waits(made(Copy, chain=True)) + waits(made(Scale, chain=True))
    assert merged and merged == sorted(alone)


def test_a_merge_of_programs_made_with_the_walk_has_it():
    merged = km.merge(made(Copy, chain=True), made(Scale), l1=1 << 20, log=quiet)
    assert merged.chain_handler == 3
    assert km.merge(made(Copy), made(Scale), l1=1 << 20, log=quiet).chain_handler == 0


def test_a_chain_names_handlers_through_the_merged_class_s_parents():
    a, b, c = tiles(64), tiles(64), tiles(64)
    for program in (made(Walked), made(Both, chain=True)):
        run = ChainRun([program])
        calls = run._class_calls(
            [Copy.copy(a, b, 8), Scale.scale(b, c, 8)], (0, 8), True
        )
        assert [run._handler(call.handler) for call in calls] == [(1, 0), (2, 0)]
        assert run._walk(calls) == 3
    program = made(Walked)
    run = ChainRun([program])
    calls = run._class_calls(program.twice(a, b, c, 8), (0, 8), True)
    assert [call.handler.method.name for call in calls] == ["copy", "scale"]


def test_a_merged_class_without_the_walk_is_refused_in_a_chain():
    run = ChainRun([made(Both)])
    with pytest.raises(ChainRefused, match="made without the chain's walk"):
        run._class_calls([Copy.copy(tiles(64), tiles(64), 8)], (0, 8), True)


class Widened(km.Program):
    WIDTH = km.constant(4)
    SPAN = 2 * WIDTH  # derived: 8 pages a core

    @km.handler
    def widen(self, source: km.Tensor, target: km.Tensor):
        out = km.Stream(target.pages(self.rank * self.SPAN, self.SPAN), write=True)
        for tile in km.Stream(source.pages(self.rank * self.SPAN, self.SPAN)):
            out.push(tripled(tile))


class WidenedBoth(Copy, Widened):
    """Copy and Widened merged: a call's ranges read Widened's constants."""


def test_a_call_s_ranges_read_its_own_class_s_constants_in_a_merged_program():
    a, b, c = tiles(64), tiles(64), tiles(64)
    alone = made(Widened, chain=True)
    expected = ChainRun([alone])._class_calls([Widened.widen(b, c)], (0, 8), True)
    merges = (
        made(WidenedBoth, chain=True, WIDTH=4),
        km.merge(made(Copy, chain=True), made(Widened, WIDTH=4), l1=1 << 20,
                 log=quiet),  # fmt: skip
    )
    for program in merges:
        assert getattr(program, "constant_values", None) is None
        run = ChainRun([program])
        calls = run._class_calls([Copy.copy(a, b, 8), Widened.widen(b, c)], (0, 8),
                                 True)  # fmt: skip
        assert calls[1].output == expected[0].output
        assert calls[1].width == expected[0].width == 8
