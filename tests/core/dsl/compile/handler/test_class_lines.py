"""The class lines of a page's words as a table (entries), the class's defines, bodies
at a kernel's start and end, compute's start-up, and a write stream a body fills: what
each lowers to and each refusal (no card)."""

from pathlib import Path

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render
from komorebi.dsl.program import compute_source, movement_source
from komorebi.errors import ProgramError


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


def grid(dtype, rows, columns):
    """A tensor of tiles, rows x columns of them."""
    page = 2048 if dtype == "bfloat16" else 4096
    pages = Pages(page)
    pages.num_pages = rows * columns
    return Tensor(pages, (rows * 32, columns * 32), dtype, "tile")


class Strided(km.Program):
    @km.handler
    def rows(self, q: km.Tensor, out: km.Tensor, rows: int, stride: int):
        r0 = self.rank
        copied = km.Stream(
            out.tiles[r0 : min(r0 + 4 * stride, r0 + rows * stride) : stride, 0:2],
            write=True,
            block=8,
        )
        for tile in km.Stream(
            q.tiles[r0 : min(r0 + 4 * stride, r0 + rows * stride) : stride, 0:2],
            block=8,
        ):
            copied.push(tile)


def test_a_rectangles_rows_with_a_step_are_a_walk_of_that_pitch():
    movement, compute = texts(Strided, "rows", q=grid("bfloat16", 16, 2),
                              out=grid("bfloat16", 16, 2))  # fmt: skip
    rows = "(((((r0 + (4 * stride)) < (r0 + (rows * stride))"
    assert "// the rectangle's rows, 2 tiles each, ((stride) * 2) apart" in movement
    assert ") / 4, 4 * ((stride) * 2), 4, ((stride) * 2), 2, 1, 2, 8, " in movement
    assert rows in movement and rows in compute  # the rows: (r1 - r0 + s - 1) / s
    assert "copy_tile(km_half0, km_tile0, 0);" in compute


class Rows(km.Program):
    @km.handler
    def copy(self, q: km.Tensor, k: km.Tensor):
        down = km.Broadcast(self, along="column")
        if down.sender():  # noqa: SIM108 (the compiler's form)
            got = down.cast(q.tiles[1:7:2, 0:2].read())
        else:
            got = down.arrive()
        for _tile in got:
            pass


def test_known_rows_with_a_step_and_their_slot():
    movement, _ = texts(Rows, "copy", q=grid("bfloat16", 16, 2),
                        k=grid("bfloat16", 16, 2))  # fmt: skip
    assert "km_row < 3; km_row++" in movement  # rows 1, 3 and 5
    assert "((1) * 2 + (0)) + km_row * 4, 2, km_address" in movement


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
    movement = "\n".join(render(prune(compiled.movement["copy"]), 0))
    compute = "\n".join(render(prune(compiled.compute["copy"]), 0))
    slots = {b.name: b for b in compiled.buffers if b.role == "broadcast"}
    assert slots["kBroadcast0"].block == 3 * 2 + 3 * 4  # K's most, then V's
    # K's columns in turn, a page each: slot h x 3 + key holds page (k0 + key) x 2 + h
    assert "km_column < 2; km_column++" in movement
    assert "((k0) * 2 + (0)) + km_row * 2 + km_column, 1," in movement
    assert "+ (km_column * 3 + km_row) * 2048))" in movement
    # V's rows after K's most: slot page 6, byte 12,288
    assert "() && km::read_pages(movement, command, V(v), ((k0) * 4 + (0))," in movement
    assert " + 12288)) || !down.cast(chunk)" in movement
    # mover A, which runs no broadcast here, skips a core the rectangle leaves out
    assert "#else  // mover A\nif (movement.arg(kArgIncluded) == 0) {" in movement
    # compute takes the block whole, then frees it
    assert "km::broadcast_receive(compute, kBroadcast0, kBroadcast0Pages);" in compute
    assert "const uint32_t kv_buffer = kBroadcast0;" in compute
    assert "const uint32_t kv = 0;" in compute
    assert compute.index("use(kv_buffer, kv);") < compute.index(
        "km::broadcast_done(kBroadcast0, kBroadcast0Pages);"
    )


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
    movement = "\n".join(render(prune(compiled.movement["copy"]), 0))
    compute = "\n".join(render(prune(compiled.compute["copy"]), 0))
    assert "uint32_t km_blocks_down = 0;\n#else  // mover B\n" in movement
    # the block's number counts every loop's passes, on mover A and compute
    assert "down.reserve(km_blocks_down)" in movement
    assert "!down.arrive(km_blocks_down)" in movement
    assert (
        movement.count("km_blocks_down++;") == compute.count("km_blocks_down++;") == 1
    )
    assert "kBroadcast0Window, km_blocks_down);" in compute
    # unread: each core's slot pushed as it is, the end padded to the window
    assert "if (unread) {" in movement
    assert "movement.reserve(kBroadcast0, kBroadcast0Pages)" in movement
    assert "} else if (down.sender()) {" in movement
    assert "% kBroadcast0Window * kBroadcast0Pages;" in movement
    assert "} else if (!down.finish()) {" in movement


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
