"""The class lines of a page's words as a table (entries), the class's defines, bodies
at a kernel's start and end, compute's start-up, and a write stream a body fills: what
each lowers to and each refusal (no card)."""

from pathlib import Path

import pytest
from class_lines_fixture import Pages, tensor

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.errors import ProgramError


def words():
    """A tensor of word pages, 1,024 words a page."""
    return Tensor(Pages(4096), (8, 1024), "uint32", "row")


def line_of(text):
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


class Job(km.Entry):
    first: int
    count: int
    target: int


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


class FedMoving(km.Program):
    @km.handler
    def rows(self, source: km.Tensor, target: km.Tensor, rows: int):
        out = km.Stream(target.pages(0, rows), write=True)
        for row in range(rows):
            km.Stream(source.pages(row, 1)).drain()
            km.cxx("pack_row(out_buffer, out, row);", takes=[out, row])


def test_a_write_stream_a_loops_bodies_fill():
    with pytest.raises(Refused, match="feed it in a loop of compute's lines alone"):
        compile_class(
            FedMoving, {}, {"source": tensor("bfloat16"), "target": tensor("bfloat16")}
        )


def grid(dtype, rows, columns):
    """A tensor of tiles, rows x columns of them."""
    page = 2048 if dtype == "bfloat16" else 4096
    pages = Pages(page)
    pages.num_pages = rows * columns
    return Tensor(pages, (rows * 32, columns * 32), dtype, "tile")


class Gathered(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor, keys: int):
        down = km.Broadcast(self, along="column")
        for chunk in range(2):
            k0 = chunk * 3
            if down.sender():
                kv = down.cast(
                    k.tiles[k0 : min(k0 + 3, keys), 0:2].T,
                    v.tiles[k0 : min(k0 + 3, keys), 0:4],
                )
            else:
                kv = down.arrive()
            km.cxx("use(kv_buffer, kv);", takes=[kv])


def test_a_cast_of_a_transposed_rectangle_and_a_second_range_into_one_block():
    compiled = compile_class(Gathered, {}, {"k": grid("bfloat16", 8, 2),
                                            "v": grid("bfloat16", 8, 4)})  # fmt: skip
    slots = {b.name: b for b in compiled.buffers if b.role == "broadcast"}
    assert slots["kBroadcast0"].block == 3 * 2 + 3 * 4  # K's most, then V's


class Unbound(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor, keys: int):
        down = km.Broadcast(self, along="column")
        if down.sender():  # noqa: SIM108
            kv = down.cast(k.tiles[0:keys, 0:2].T)  # unbound
        else:
            kv = down.arrive()
        km.cxx("use(kv);", takes=[kv])


class Unstated(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor, rows: int, stride: int):
        down = km.Broadcast(self, along="column")
        if down.sender():
            kv = down.cast(k.tiles[0 : rows * stride : stride, 0:2])  # unstated
        else:
            kv = down.arrive()
        km.cxx("use(kv);", takes=[kv])


class Mixed(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor):
        down = km.Broadcast(self, along="column")
        if down.sender():
            kv = down.cast(k.tiles[0:1, 0:2], v.tiles[0:1, 0:2])  # mixed
        else:
            kv = down.arrive()
        km.cxx("use(kv);", takes=[kv])


class Twice(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor):
        down = km.Broadcast(self, along="column")
        if down.sender():  # noqa: SIM108
            kv = down.cast(k.tiles[0:1, 0:2])
        else:
            kv = down.arrive()
        km.cxx("use(kv);", takes=[kv])
        km.cxx("again(kv);", takes=[kv])  # twice


class Packed(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor):
        down = km.Broadcast(self, along="column")
        if down.sender():  # noqa: SIM108
            kv = down.cast(k.tiles[0:1, 0:2])
        else:
            kv = down.arrive()
        km.cxx("use(kv);", takes=[kv], on="pack")  # packed


class ColumnStep(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor):
        km.Stream(k.tiles[0:2, 0:2:2]).drain()  # column step


@pytest.mark.parametrize(
    "cls, text, why",
    [
        (Unbound, "# unbound", "the build knows no most for a side here"),
        (Unstated, "# unstated", "the build knows no most for a side here"),
        (Mixed, "# mixed", "a cast's block is of one format and page size"),
        (Twice, "# twice", "this block is taken above"),
        (Packed, "# packed", "a block of compute's three CPUs"),
        (ColumnStep, "# column step", "a step on the rows alone"),
    ],
)
def test_range_and_block_refusals_name_the_line(cls, text, why):
    line = line_of(text)
    tensors = {"k": grid("bfloat16", 8, 2), "v": grid("float32", 8, 2)}
    with pytest.raises(Refused, match=rf"^{cls.__name__}.copy, line {line}, .*{why}"):
        compile_class(cls, {}, tensors)


class Banded(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor, jobs: int, unread: int):
        down = km.Broadcast(self, along="column", length=2, mover="A", unread=unread)
        for job in range(jobs):
            for chunk in range(job + 2):
                if down.sender():
                    kv = down.cast(k.tiles[chunk : chunk + 1, 0:2].read())
                else:
                    kv = down.arrive()
                km.cxx("use(kv);", takes=[kv])


def test_a_broadcast_on_a_chosen_mover_in_bands_counted_over_its_loops_or_unread():
    compiled = compile_class(Banded, {}, {"k": grid("bfloat16", 8, 2),
                                          "v": grid("bfloat16", 8, 2)})  # fmt: skip
    assert compiled.broadcasts[0][3:] == ("A", 2, 0)  # mover A, bands of 2, its first


class Unplaced(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor):
        down = km.Broadcast(self, along="column", mover="C")  # unplaced
        km.cxx("use();", takes=[down])


class Unlengthed(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor, n: int):
        down = km.Broadcast(self, along="column", length=n)  # unlengthed
        km.cxx("use();", takes=[down])


class TwoTops(km.Program):
    @km.handler
    def copy(self, k: km.Tensor, v: km.Tensor):
        down = km.Broadcast(self, along="column")
        for i in range(1, 3):
            if down.sender():  # noqa: SIM108
                kv = down.cast(k.tiles[i : i + 1, 0:2])
            else:
                kv = down.arrive()
            km.cxx("use(kv);", takes=[kv])
        for i in range(2):
            if down.sender():  # noqa: SIM108 (two tops)
                kv = down.cast(k.tiles[i : i + 1, 0:2])
            else:
                kv = down.arrive()
            km.cxx("use(kv);", takes=[kv])


@pytest.mark.parametrize(
    "cls, text, why",
    [
        (Unplaced, "# unplaced", 'a broadcast\'s mover is "A"'),
        (Unlengthed, "# unlengthed", "a broadcast's length is cores a band"),
        (TwoTops, "(two tops)", "down casts in two loops"),
    ],
)
def test_broadcast_place_and_count_refusals(cls, text, why):
    line = line_of(text)
    tensors = {"k": grid("bfloat16", 8, 2), "v": grid("bfloat16", 8, 2)}
    with pytest.raises(Refused, match=rf"^{cls.__name__}.copy, line {line}, .*{why}"):
        compile_class(cls, {}, tensors)
