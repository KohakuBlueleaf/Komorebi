"""A class's X280 thread and its generated header: a function per built handler (a
constant set's, a merged program's), the load's words, the regions; the thread
declared in the class or added after the make, its C built at the make; each
refusal (no card)."""

from pathlib import Path

import pytest
from class_thread_header_fixture import (
    COMPILER,
    HERE,
    Counted,
    Plain,
    Sums,
    grid,
    quiet,
)

import komorebi as km
from komorebi.errors import ProgramError

pytestmark = pytest.mark.tree  # needs a tt-metal tree: the tree-tests lane


class Drain(km.Program):
    @km.handler
    def drain(self, a: km.Tensor):
        km.Stream(a.pages(0, 1)).drain()

    @km.handler
    def count(self, a: km.Tensor):
        if self.rank == self.leader:
            total = a.pages(0, 1).read()
            return total


class Missing(Sums):
    counts = km.region(4096)
    thread = km.x280_thread(str(HERE / "missing.c"))


class Unnamed(Sums):
    others = km.region(4096)
    thread = km.x280_thread(str(HERE / "counted.c"))


def header(program, own=False):
    """The thread header of the build folder, or of the program's thread."""
    name = f"{type(program).__name__.lower()}_thread.h"
    if own:
        return (
            Path(program.threads[min(program.threads)].includes[0]) / name
        ).read_text()
    return (program.built.folder / name).read_text()


@COMPILER
def test_a_declared_thread_is_made_and_built_at_the_make():
    program = Counted(l1=1 << 20, log=quiet, SIZE=12288, thread_words=[7, 8])
    thread = program.thread
    assert thread.cluster == 2 and thread.regions == {"counts": 12288}
    assert thread.words == [7, 8]  # no tensor: no load's word


@COMPILER
def test_a_thread_added_after_the_make_gets_the_same_header():
    program = Sums(l1=1 << 20, log=quiet)
    program.thread(cluster=1, sources=[HERE / "added.c"], regions={"counts": 4096},
                   words=[3])  # fmt: skip
    assert program.threads[1].words == [3]


class MergedCounted(Counted):
    thread = km.x280_thread(str(HERE / "merged.c"), cluster=2)


class Countedrain(MergedCounted, Drain):
    pass


@COMPILER
def test_a_merge_makes_the_thread_one_class_declares():
    program = Countedrain(l1=1 << 20, log=quiet, a=grid(8), SIZE=12288,
                          thread_words=[7])  # fmt: skip
    thread = program.thread
    assert thread.cluster == 2 and thread.regions == {"counts": 12288}
    assert thread.words[0] == 7 and len(thread.words) == 2  # its word, then a's


def test_a_merge_of_two_threads_is_refused():
    class Drained(Drain):
        thread = km.x280_thread(str(HERE / "merged.c"))

    class Two(MergedCounted, Drained):
        pass

    with pytest.raises(ProgramError, match=r"MergedCounted and Drained each declare"):
        Two(l1=1 << 20, log=quiet, a=grid(8))
    with pytest.raises(ProgramError, match=r"none of its classes declares one"):
        type("Both", (Drain, Plain), {})(
            l1=1 << 20, log=quiet, a=grid(8), out=grid(8, 0x9000), thread_words=[1]
        )


@COMPILER
def test_the_c_naming_what_the_class_lacks_is_refused_at_the_make():
    with pytest.raises(ProgramError) as refused:
        Missing(l1=1 << 20, log=quiet)
    said = str(refused.value)
    assert said.startswith("Missing, line ")
    assert "missing.c:" in said and ": error:" in said
    assert "missing_nothing: the program has no handler nothing (['total'])" in said
    assert "MISSING_REGION_ITEMS: the program has no region items (['counts'])" in said


def test_refusals():
    with pytest.raises(ProgramError, match="has no region of that name"):
        Unnamed(l1=1 << 20, log=quiet)
    with pytest.raises(ProgramError, match="declares none"):
        Sums(l1=1 << 20, log=quiet, thread_words=[1])
    with pytest.raises(ProgramError, match="declares a region of the class's thread"):

        class Alone(Sums):
            counts = km.region(4096)

    with pytest.raises(ProgramError, match="an int or of the class's constants"):
        km.region(4.5)
    with pytest.raises(ProgramError, match="regions is not one of them"):
        km.x280_thread("t.c", regions={})
