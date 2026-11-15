"""A class's X280 thread and its generated header: a function per built handler (a
constant set's, a merged program's), the load's words, the regions; the thread
declared in the class or added after the make, its C built at the make; each
refusal (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi import _env
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor

HERE = Path(__file__).resolve().parent / "class_thread"
TREE = _env.find_tree()[0]
COMPILER = pytest.mark.skipif(
    TREE is None or _env.x280_compiler(TREE) is None, reason="no X280 compiler"
)


def quiet(line):
    pass


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages, address):
        self.page_size, self.num_pages, self.address = page_size, pages, address
        self.aligned_page_size = page_size

    def accessor_args(self):
        return [2, self.page_size]


def grid(columns, address=0x1000):
    """A bfloat16 tensor of one row of `columns` tiles."""
    return Tensor(Pages(2048, columns, address), (32, columns * 32), "bfloat16", "tile")


@km.tile_function
def twice(a):
    return a + a


class Scale(km.Program):
    @km.handler
    def scale(
        self, a: km.Tensor, out: km.Tensor, tiles: int, size: km.constant[int] = 8
    ):
        results = km.Stream(out.pages(0, tiles), write=True, block=size)
        for x in km.Stream(a.pages(0, tiles), block=size):
            results.push(twice(x))


class Plain(km.Program):
    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, tiles: int):
        results = km.Stream(out.pages(0, tiles), write=True)
        for x in km.Stream(a.pages(0, tiles)):
            results.push(twice(x))


class Drain(km.Program):
    @km.handler
    def drain(self, a: km.Tensor):
        km.Stream(a.pages(0, 1)).drain()

    @km.handler
    def count(self, a: km.Tensor):
        if self.rank == self.leader:
            total = a.pages(0, 1).read()
            return total


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


class Counted(Sums):
    SIZE = km.constant(8192)
    counts = km.region(SIZE)
    thread = km.x280_thread(str(HERE / "counted.c"), cluster=2)


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


def test_a_constant_set_s_handlers_each_fill_their_command():
    a, out = grid(8, 0x1000), grid(8, 0x9000)
    program = Scale(l1=1 << 20, log=quiet, a=a, out=out,
                    handlers={"scale": [{"size": 4}, {"size": 8}]})  # fmt: skip
    text = header(program)
    assert "#define SCALE_HANDLER_SCALE_SIZE4 1u" in text  # today's places stay
    assert "#define SCALE_SCALE_SIZE4_TILES 2u" in text
    assert "static inline void scale_scale_size4(" in text
    assert "static inline void scale_scale_size8(" in text
    assert "#define SCALE_WORD_A (SCALE_WORDS_OWN + 0u)" in text
    assert "#define SCALE_WORD_OUT (SCALE_WORDS_OWN + 1u)" in text
    assert "    km_command->a0 = scale_word(SCALE_WORD_A);" in text
    assert "    km_command->a2 = tiles;" in text
    assert "static inline kmx_dram scale_a_dram(uint32_t first) {" in text
    assert "_REGION_" not in text  # no thread, no region parameter


def test_a_merged_program_s_header_has_the_merged_numbers():
    class Both(Drain, Plain):
        pass

    text = header(Both(l1=1 << 20, log=quiet, a=grid(8), out=grid(8, 0x9000)))
    assert "#define BOTH_HANDLER_DRAIN 1u" in text
    assert "#define BOTH_HANDLER_SCALE 3u" in text
    assert "static inline void plain_scale(" in text  # each class's own name
    assert "static inline void drain_count(" in text
    assert "kmx_command(km_command, BOTH_PROGRAM, BOTH_HANDLER_SCALE," in text
    assert "    km_command->hdr |= (km_flags | KM_FLAG_RESULT) << 8;" in text  # count


@COMPILER
def test_a_declared_thread_is_made_and_built_at_the_make():
    program = Counted(l1=1 << 20, log=quiet, SIZE=12288, thread_words=[7, 8])
    thread = program.thread
    assert thread.cluster == 2 and thread.regions == {"counts": 12288}
    assert thread.words == [7, 8]  # no tensor: no load's word
    text = header(program, own=True)
    assert "#define COUNTED_SIZE 12288u" in text
    assert "#define COUNTED_REGION_COUNTS 0u" in text
    assert "#define COUNTED_REGION_COUNTS_BYTES 12288u" in text
    assert "#define COUNTED_WORDS_OWN" not in text
    assert "    km_command->a0 = counted_region_address(COUNTED_REGION_COUNTS);" in text
    assert "    km_command->a2 = counted_bank();" in text
    assert "return cluster == 1u ? 6u : 7u;" in text
    assert "_REGION_" not in header(program)  # the build folder's: no thread's regions


@COMPILER
def test_a_thread_added_after_the_make_gets_the_same_header():
    program = Sums(l1=1 << 20, log=quiet)
    program.thread(cluster=1, sources=[HERE / "added.c"], regions={"counts": 4096},
                   words=[3])  # fmt: skip
    text = header(program, own=True)
    assert "    km_command->a0 = sums_region_address(SUMS_REGION_COUNTS);" in text
    assert program.threads[1].words == [3]


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
