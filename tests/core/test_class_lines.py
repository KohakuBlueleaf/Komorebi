"""The class lines of a page's words as a table (entries), the class's defines, bodies
at a kernel's start and end, compute's start-up, and a write stream a body fills: what
each lowers to and each refusal (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.dsl.compiler import Refused, compile_class
from komorebi.dsl.nodes import prune, render
from komorebi.dsl.program import compute_source, movement_source
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype):
    return Tensor(
        Pages(2048 if dtype == "bfloat16" else 4096), (2048, 32), dtype, "tile"
    )


def words():
    """A tensor of word pages, 1,024 words a page."""
    return Tensor(Pages(4096), (8, 1024), "uint32", "row")


def line_of(text):
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


def texts(cls, handler, **tensors):
    """(movement, compute) C++ of one handler of `cls`."""
    compiled = compile_class(cls, {}, tensors)
    return (
        "\n".join(render(prune(compiled.movement[handler]), 0)),
        "\n".join(render(prune(compiled.compute[handler]), 0)),
    )


class Job(km.Entry):
    first: int
    count: int
    target: int


class Walked(km.Program):
    @km.handler
    def walk(self, tables: km.Tensor, source: km.Tensor, target: km.Tensor):
        for job in tables.pages(self.rank, 1).entries(Job, first=8, size=4, count=0):
            out = km.Stream(target.pages(job.target, job.count), write=True)
            for tile in km.Stream(source.pages(job.first, job.count)):
                out.push(tile)


def test_entries_are_read_on_every_cpu_and_walked_alike():
    movement, compute = texts(
        Walked, "walk", tables=words(), source=tensor("bfloat16"),
        target=tensor("bfloat16"),
    )  # fmt: skip
    # mover A reads the page and hands it to compute; mover B reads its own
    assert "movement.reserve(kEntries0, 1)" in movement
    assert "cb_push_back(kEntries0, 1);" in movement
    assert "get_write_ptr(kEntries0B)" in movement
    # the count (word 0), refused past the page's 254 entries of 4 words from word 8
    assert "km_entries0[0] <= 254 ? km_entries0[0] : 0" in movement
    loop = "for (uint32_t km_job_at = 0; km_job_at < km_entries0_count; km_job_at++)"
    assert loop in movement
    assert "job = km_entries0 + 8 + 4 * km_job_at;" in movement
    assert "km_job_target = job[kJobTarget];" in movement
    assert "km_job_target, km_job_count, kWrite0Block" in movement
    # compute takes the page as pushed, walks it, frees it after the loop
    assert "compute.wait_front(kEntries0, 1)" in compute
    assert "km::l1(get_tile_address(kEntries0, 0));" in compute
    assert "km_ok && km_job_at < km_entries0_count" in compute
    assert compute.rstrip().endswith("cb_pop_front(kEntries0, 1);")


class Spanned(km.Program):
    @km.handler
    def walk(self, directory: km.Tensor, source: km.Tensor):
        for span in directory.pages(self.rank, 1).entries(Job, size=3, fixed=1):
            km.Stream(source.pages(span.first, span.count)).drain()


def test_one_entry_known_at_build_is_a_block_of_no_loop():
    movement, compute = texts(
        Spanned, "walk", directory=words(), source=tensor("bfloat16")
    )
    assert "km_span_at" not in movement + compute
    assert "span = km_entries0 + 0;" in movement
    assert "km_entries0_count" not in movement


class Named(km.Program):
    @km.handler
    def walk(self, tables: km.Tensor, source: km.Tensor, first: int, count: int):
        jobs = tables.pages(self.rank, 1).entries(Job, first=8, size=4, count=0)
        end = first + count + (count == 0) * jobs.count
        for job in jobs[first:end]:
            km.cxx("total += job_count;", takes=[job], on="pack")


def test_a_named_table_is_sliced_as_python_slices_and_bodies_take_its_words():
    movement, compute = texts(Named, "walk", tables=words(), source=tensor("bfloat16"))
    assert "(first < km_entries0_count ? first : km_entries0_count)" in compute
    assert "(end < km_entries0_count ? end : km_entries0_count)" in compute
    assert "const uint32_t job_count = km_job_count;" in compute
    assert "end = ((first + count) + ((count == 0) * km_entries0_count));" in compute
    assert compute.rstrip().endswith("cb_pop_front(kEntries0, 1);")


class Unclassed(km.Program):
    @km.handler
    def walk(self, tables: km.Tensor):
        for _ in tables.pages(self.rank, 1).entries(first=8, size=4, count=0):
            pass


class Both(km.Program):
    @km.handler
    def walk(self, tables: km.Tensor):
        for _ in tables.pages(0, 1).entries(Job, size=4, count=0, fixed=1):
            pass


class Inside(km.Program):
    @km.handler
    def walk(self, tables: km.Tensor):
        for _ in tables.pages(0, 1).entries(Job, first=8, size=4, count=9):
            pass


class Unknown(km.Program):
    @km.handler
    def walk(self, tables: km.Tensor, source: km.Tensor):
        for job in tables.pages(0, 1).entries(Job, size=4, fixed=1):
            km.Stream(source.pages(job.start, 1)).drain()


class Nested(km.Program):
    @km.handler
    def walk(self, tables: km.Tensor, rounds: int):
        for _ in range(rounds):
            jobs = tables.pages(0, 1).entries(Job, size=4, fixed=1)
            km.cxx("use(jobs_count);", takes=[jobs])


class Stepped(km.Program):
    @km.handler
    def walk(self, tables: km.Tensor):
        jobs = tables.pages(0, 1).entries(Job, first=8, size=4, count=0)
        for _ in jobs[0:4:2]:
            pass


class Small(km.Program):
    @km.handler
    def walk(self, tables: km.Tensor):
        for _ in tables.pages(0, 1).entries(Job, size=2, fixed=1):
            pass


@pytest.mark.parametrize(
    "cls, text, why",
    [
        (Unclassed, "entries(first=8, size=4, count=0)", "entries name their class"),
        (Both, "count=0, fixed=1", "either the word of their count"),
        (Inside, "count=9", "the count's word 9 is not a word of the page before"),
        (Unknown, "job.start", "job, a Job, has no word start"),
        (Nested, "jobs = tables.pages(0, 1)", "a named table is read at the handler"),
        (Stepped, "jobs[0:4:2]", "a slice of its entries, with no step"),
        (Small, "entries(Job, size=2, fixed=1)", "Job names 3 words, and an entry"),
    ],
)
def test_entries_refusals_name_the_line(cls, text, why):
    line = line_of(text)
    with pytest.raises(Refused, match=rf"^{cls.__name__}.walk, line {line}, .*{why}"):
        compile_class(cls, {}, {"tables": words(), "source": tensor("bfloat16")})


def test_an_entry_class_is_ints():
    with pytest.raises(ProgramError, match="an entry's words are ints"):

        class Bad(km.Entry):
            name: str


class Defined(km.Program, defines={"SCALE": 3, "MODE": "fast"}):
    @km.handler
    def copy(self, source: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


def test_defines_of_the_class_and_of_the_make():
    made = Defined(source=tensor("bfloat16"), l1=1 << 20, log=lambda line: None)
    assert made.defines == {"SCALE": "3", "MODE": "fast"}
    made = Defined(
        source=tensor("bfloat16"), l1=1 << 20, log=lambda line: None,
        defines={"SCALE": 4, "CAUSAL": 1},
    )  # fmt: skip
    assert made.defines == {"SCALE": "4", "MODE": "fast", "CAUSAL": "1"}
    with pytest.raises(ProgramError, match="not one of the runtime's"):
        Defined(
            source=tensor("bfloat16"), l1=1 << 20, log=lambda line: None,
            defines={"KM_PAGE_LINKS": 1},
        )  # fmt: skip
    with pytest.raises(ProgramError, match="an int or a string"):
        Defined(
            source=tensor("bfloat16"), l1=1 << 20, log=lambda line: None,
            defines={"X": 1.5},
        )  # fmt: skip


class Started(km.Program, startup=km.Startup("ones", "ones", "ones", order="reverse")):
    ones: km.L1[km.bf16, 1, 32, 32] = km.own(push=None)

    @km.handler
    def copy(self, source: km.Tensor):
        km.cxx("build_ones(ones);", takes=[self.ones], at="start")
        km.cxx("restore();", at="end")
        km.cxx("noc_async_write_barrier();", on="mover_b", at="end")
        km.Stream(source.pages(0, 1)).drain()


def test_bodies_at_the_kernels_start_and_end_and_the_startup():
    compiled = compile_class(Started, {}, {"source": tensor("bfloat16")})
    compute = compute_source(compiled)
    startup = "compute_kernel_hw_startup<SrcOrder::Reverse>(kFieldOnes, kFieldOnes"
    assert startup in compute
    assert compute.index(startup) < compute.index("build_ones(ones);")
    assert compute.index("build_ones(ones);") < compute.index("compute.run(")
    assert compute.index("restore();") > compute.rindex("});")
    assert "const uint32_t ones = kFieldOnes;" in compute
    movement = movement_source(compiled)
    end = movement[movement.rindex("});") :]
    assert "#if !defined(COMPILE_FOR_BRISC)" in end
    assert "noc_async_write_barrier();" in end
    assert "restore();" not in movement


class StartTakes(km.Program):
    @km.handler
    def copy(self, source: km.Tensor, scale: int):
        km.cxx("use(scale);", takes=[scale], at="start")


class StartAt(km.Program):
    @km.handler
    def copy(self, source: km.Tensor):
        km.cxx("use();", at="middle")


def test_start_and_end_refusals():
    line = line_of('km.cxx("use(scale);", takes=[scale], at="start")')
    with pytest.raises(Refused, match=rf"line {line}, .*own fields alone"):
        compile_class(StartTakes, {}, {"source": tensor("bfloat16")})
    line = line_of('km.cxx("use();", at="middle")')
    with pytest.raises(Refused, match=rf"line {line}, .*the kernel's start or its"):
        compile_class(StartAt, {}, {"source": tensor("bfloat16")})
    with pytest.raises(ProgramError, match="a source order is of two inputs"):
        km.Startup("a", "out", order="reverse")


class Fed(km.Program):
    @km.handler
    def rows(self, source: km.Tensor, target: km.Tensor, rows: int):
        out = km.Stream(target.pages(0, rows), write=True)
        for row in range(rows):
            km.cxx("pack_row(out_buffer, out, row);", takes=[out, row])


class FedMoving(km.Program):
    @km.handler
    def rows(self, source: km.Tensor, target: km.Tensor, rows: int):
        out = km.Stream(target.pages(0, rows), write=True)
        for row in range(rows):
            km.Stream(source.pages(row, 1)).drain()
            km.cxx("pack_row(out_buffer, out, row);", takes=[out, row])


def test_a_write_stream_a_loops_bodies_fill():
    movement, compute = texts(
        Fed, "rows", source=tensor("bfloat16"), target=tensor("bfloat16")
    )
    assert "uint32_t km_fed_out = 0;" in compute
    assert "km::block_half(kWrite0, kWrite0Pattern, km_fed_out);" in compute
    assert "const bool km_room_out = compute.reserve_back(km_out_out" in compute
    assert "const uint32_t out_buffer = km_out_out;" in compute
    assert "km_fed_out++;" in compute
    # the movers move the whole stream once, after the loop
    assert "for (uint32_t row" not in movement
    assert "movement.move(km_write0)" in movement
    with pytest.raises(Refused, match="feed it in a loop of compute's lines alone"):
        compile_class(
            FedMoving, {}, {"source": tensor("bfloat16"), "target": tensor("bfloat16")}
        )
