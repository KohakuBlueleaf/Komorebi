"""A region of an X280 thread as a handler's parameter: its two command words, its
pages and items as a tensor's, its name given at the submit; and the refusals (no
card)."""

import pytest

import komorebi as km
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.front import Region, region_words, word_of
from komorebi.dsl.front.nodes import prune, render
from komorebi.session.run import Run


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


def texts(cls, handler, **tensors):
    """(compiled, movement C++) of one handler of `cls`."""
    compiled = compile_class(cls, {}, tensors)
    return compiled, "\n".join(render(prune(compiled.movement[handler]), 0))


def test_a_region_is_two_words_and_a_tensor_of_its_pages():
    compiled, movement = texts(Sums, "total")
    # its bank address the parameter's word, its bank a word after the others
    assert compiled.words["total"] == [
        ("counts", "region"),
        ("at", "int"),
        ("counts", "bank"),
    ]
    # its pages are the thread's, 4096 B, in one bank (km_region.hpp)
    assert compiled.regions == ["counts"]


def test_the_host_gives_a_region_by_its_words():
    assert region_words("total", "counts", Region(6, 0x4000)) == (0x4000, 6)
    assert region_words("total", "counts", (7, 0x8000)) == (0x8000, 7)
    assert word_of("total", "counts", "region", (7, 0x8000)) == 0x8000
    assert word_of("total", "counts", "bank", (7, 0x8000)) == 7
    with pytest.raises(km.ProgramError, match="counts is a region"):
        word_of("total", "counts", "region", 5)


class Thread:
    """A run's thread as `Run._regions_named` reads it: its regions by name."""

    class Regions:
        def __init__(self, places):
            self.keys, self.places = {n: i for i, n in enumerate(places)}, places

        def address(self, name):
            return self.places[name]

    def __init__(self, **places):
        self.regions = self.Regions(places)


def test_a_submit_names_the_region_of_the_thread_that_has_it():
    compile_class(Sums, {}, {})
    run = Run.__new__(Run)
    run.threads = {1: Thread(sums=(6, 0x4000)), 2: Thread(items=(7, 0x8000))}
    assert run._regions_named(Sums.total, ("sums", 3)) == [(6, 0x4000), 3]
    with pytest.raises(ValueError, match="no thread of this run"):
        run._regions_named(Sums.total, ("lists", 3))
    run.threads[3] = Thread(sums=(7, 0x1000))
    with pytest.raises(ValueError, match="more than one thread"):
        run._regions_named(Sums.total, ("sums", 3))


class Placement:
    parts = [(0, 4), (4, 4)]


class Island:
    """An island tensor, as a program made with them holds one."""

    address, dtype, page_bytes, placement = 0x1000, "uint32", 4096, Placement()


class Other(km.Program):
    @km.handler
    def total(self, counts: km.Region, t: km.Tensor):
        km.Stream(counts.pages(0, 1)).drain()


def test_another_islands_core_reading_a_region_is_refused_with_the_island():
    with pytest.raises(Refused, match="region of island 1's DRAM"):
        compile_class(Other, {}, {"t": Island(), "counts": Region(6, 0, island=1)})
