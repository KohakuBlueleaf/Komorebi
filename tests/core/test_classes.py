"""A program written as a class that the compiler does not take is refused when it is
made, with a sentence that names the handler and the line (no card)."""

import inspect
from pathlib import Path

import pytest

import komorebi as km
from komorebi import formats
from komorebi.dsl.compiler import Refused, compile_class
from komorebi.dsl.nodes import prune, render
from komorebi.dsl.program import compute_options
from komorebi.errors import ProgramError
from komorebi.session import Session
from komorebi.tensor import Tensor


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype):
    return Tensor(Pages(formats.tile_bytes(dtype)), (2048, 32), dtype, "tile")


class Sliced(km.Program):
    @km.handler
    def copy(self, source: km.Tensor):
        km.Stream(source[1:4]).drain()


class Unled(km.Program):
    @km.handler
    def merge(self, source: km.Tensor, output: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(source.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        output.pages(0, 1).write(whole)


class Unsolved(km.Program):
    TILES = km.symbol()
    count: km.L1[km.fp32, TILES, 32, 32]

    @km.handler
    def nothing(self, source: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


def made(cls, **tensors):
    return cls(log=lambda line: None, **tensors)


def line_of(text):
    """The number of the line of this file that holds `text`."""
    lines = Path(__file__).read_text().split("\n")
    return next(n for n, line in enumerate(lines, start=1) if text in line)


def test_a_refusal_names_the_handler_and_the_line():
    line = line_of("km.Stream(source[1:4]).drain()")
    with pytest.raises(Refused, match=rf"^Sliced.copy, line {line}, `km.Stream\("):
        made(Sliced, source=tensor("bfloat16"))
    line = line_of("output.pages(0, 1).write(whole)")
    with pytest.raises(Refused, match=rf"^Unled.merge, line {line}, `output.pages"):
        made(Unled, source=tensor("float32"), output=tensor("float32"))


def test_a_symbol_left_to_the_build_needs_the_l1():
    line = line_of("count: km.L1[km.fp32, TILES, 32, 32]")
    with pytest.raises(Refused, match=rf"^Unsolved.count, line {line}, .*l1=session"):
        made(Unsolved, source=tensor("bfloat16"))


@km.tile_function
def product(a, b):
    return a @ b


class Product(km.Program):
    @km.handler
    def multiply(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 1), write=True)
        for x, y in zip(km.Stream(a.pages(0, 1)), km.Stream(b.pages(0, 1))):
            results.push(product(x, y))


def test_a_product_in_a_tile_function_is_unpacked_and_issued_by_the_body():
    tensors = {"a": tensor("bfloat16"), "b": tensor("bfloat16")}
    compiled = compile_class(Product, {}, {**tensors, "out": tensor("float32")})
    compute = "\n".join(render(prune(compiled.compute["multiply"]), 0))
    assert "mm_no_mop_init_short(kRead0.mover_a, kRead1.mover_a, false, 1, 1, 1);" in (
        compute
    )
    assert "UNPACK((llk_unpack_AB_matmul(km_half0, km_half1, km_tile0 + " in compute
    assert "copy_tile(" not in compute  # no copy into a product's Dst tile
    assert compute_options(compiled)["math_fidelity"] == "HiFi4"
    assert compute_options(compiled)["fp32_dest_acc_en"]
    # the A family's operands: the same product, on a 16-bit Dst
    tensors = {"a": tensor("float16"), "b": tensor("float16")}
    compiled = compile_class(Product, {}, {**tensors, "out": tensor("float32")})
    compute = "\n".join(render(prune(compiled.compute["multiply"]), 0))
    assert "mm_no_mop_init_short(kRead0.mover_a, kRead1.mover_a, false, 1, 1, 1);" in (
        compute
    )
    assert not compute_options(compiled)["fp32_dest_acc_en"]
    # one operand of each family is refused at the line, naming both
    line = line_of("results.push(product(x, y))")
    with pytest.raises(
        Refused, match=rf"^Product.multiply, line {line}, .*given float16 with bfloat16"
    ):
        made(Product, a=tensor("float16"), b=tensor("bfloat16"), out=tensor("float32"))


@km.tile_function
def scaled(a, b):
    return a * b


@km.tile_function
def project(x, w):
    return x @ w


class OtherFunction(km.Program):
    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 4)))
        acc = km.tile.zeros(km.fp32, 2)
        for block in km.Stream(x.pages(0, 2), block=2):
            acc += scaled(block, self.weight)


class Unfilled(km.Program):
    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for block in km.Stream(x.pages(0, 2), block=2):
            acc += project(block, self.weight)  # unfilled


class Narrow(km.Program):
    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 4)))
        acc = km.tile.zeros(km.fp32)
        for block in km.Stream(x.pages(0, 2), block=2):
            acc += project(block, self.weight)  # narrow


class Mixed(km.Program):
    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 4)))
        acc = km.tile.zeros(km.fp32, 2)
        for block in km.Stream(x.pages(0, 2), block=2):
            acc += block
            acc += project(block, self.weight)  # mixed


PRODUCTS = [
    (OtherFunction, "acc += scaled(block, self.weight)", "is not `return a @ b`"),
    (Unfilled, "# unfilled", "is not filled above"),
    (Narrow, "# narrow", "is K rows of 2 tiles and acc holds 1"),
    (Mixed, "# mixed", "this loop has other tile lines"),
]


@pytest.mark.parametrize(("cls", "text", "message"), PRODUCTS)
def test_a_product_the_compiler_does_not_take_is_refused_by_its_line(
    cls, text, message
):
    line = line_of(text)
    match = rf"^{cls.__name__}.apply, line {line}, .*{message}"
    with pytest.raises(Refused, match=match):
        made(cls, x=tensor("bfloat16"), w=tensor("bfloat16"))


def test_a_product_of_two_families_is_refused_by_its_line():
    line = line_of("# narrow")
    with pytest.raises(Refused, match=rf"^Narrow.apply, line {line}, .*one exponent"):
        made(Narrow, x=tensor("float16"), w=tensor("bfloat16"))


def grid(dtype, rows, columns):
    """A tensor of rows x columns tiles."""
    return Tensor(
        Pages(2048 if dtype == "bfloat16" else 4096),
        (rows * 32, columns * 32),
        dtype,
        "tile",
    )


@km.tile_function
def gate(a, b):
    return a * b


@km.tile_function(fidelity="HiFi2")
def coarse(x, w):
    return x @ w


class Stepped(km.Program):
    @km.handler
    def copy(self, x: km.Tensor):
        km.Stream(x.tiles[0:4, 0:4:2]).drain()  # stepped


class Ragged(km.Program):
    @km.handler
    def copy(self, x: km.Tensor, n: int):
        km.Stream(x.tiles[0:2, 0:n]).drain()  # ragged


class Uneven(km.Program):
    @km.handler
    def copy(self, x: km.Tensor):
        km.Stream(x.tiles[0:4, 0:3], block=4).drain()  # uneven


class Owned(km.Program):
    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        self.weight.fill(km.Stream(w.tiles[0:2, 0:2]))
        acc = km.tile.zeros(km.fp32, 2)
        for block in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += project(block, self.weight)
        whole = km.reduce(acc, form="owner")  # owned
        if self.rank == self.leader:
            y.tiles[0:1, 0:2].write(whole)


class Tall(km.Program):
    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        self.weight.fill(km.Stream(w.tiles[0:2, 0:2]))
        acc = km.tile.zeros(km.fp32, 2)
        for block in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += project(block, self.weight)
        y.tiles[0:2, 0:1].write(acc)  # tall


class Vector(km.Program):
    @km.tile_function(fidelity="HiFi2")
    def gated(a, b):
        return a * b

    @km.handler
    def apply(self, a: km.Tensor, b: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 2), write=True)
        for x, y in zip(km.Stream(a.pages(0, 2)), km.Stream(b.pages(0, 2))):
            results.push(self.gated(x, y))  # vector


class Fidelities(km.Program):
    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        self.weight.fill(km.Stream(w.tiles[0:2, 0:2]))
        acc = km.tile.zeros(km.fp32, 2)
        for block in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += project(block, self.weight)
        for block in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += coarse(block, self.weight)  # fidelities
        y.tiles[0:1, 0:2].write(acc)


LINES = [
    (Stepped, "copy", "# stepped", "two slices, rows then columns"),
    (Ragged, "copy", "# ragged", "this rectangle's columns are a word"),
    (Uneven, "copy", "# uneven", "whole rows of it"),
    (Owned, "apply", "# owned", "by owner holds a slot of them"),
    (Tall, "apply", "# tall", "a page write of a rectangle is one row"),
    (Vector, "apply", "# vector", "fidelity= is a product's"),
    (Fidelities, "apply", "# fidelities", "the fidelity is the compute kernel's"),
]


@pytest.mark.parametrize(("cls", "handler", "text", "message"), LINES)
def test_a_line_of_rectangles_reduces_or_fidelity_is_refused_by_its_line(
    cls, handler, text, message
):
    line = line_of(text)
    match = rf"^{cls.__name__}.{handler}, line {line}, .*{message}"
    tensors = {
        "x": grid("bfloat16", 4, 4),
        "w": grid("bfloat16", 4, 4),
        "y": grid("bfloat16", 4, 4),
        "a": tensor("bfloat16"),
        "b": tensor("bfloat16"),
        "out": tensor("bfloat16"),
    }
    names = inspect.signature(getattr(cls, handler).function).parameters
    with pytest.raises(Refused, match=match):
        made(cls, **{name: tensors[name] for name in names if name in tensors})


class Kept(km.Program):
    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def load(self, w: km.Tensor):
        self.weight.fill(km.Stream(w.tiles[0:2, 0:2]))

    @km.handler
    def apply(self, x: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for block in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += project(block, self.weight)
        for block in km.Stream(x.tiles[1:2, 0:2], block=2):
            acc += project(block, self.weight)
        y.tiles[0:1, 0:2].write(acc)


def test_a_field_filled_by_one_handler_is_read_by_another():
    compiled = compile_class(
        Kept,
        {},
        {name: grid("bfloat16", 4, 4) for name in ("w", "x", "y")},
    )
    assert compiled.field_users["weight"] == (["load"], ["apply"])
    compute = "\n".join(render(prune(compiled.compute["apply"]), 0))
    assert compute.count("compute.wait_front(kFieldWeight") == 1
    assert "cb_pop_front(kFieldWeight" not in compute
    load = "\n".join(render(prune(compiled.compute["load"]), 0))
    assert "cb_pop_front(kFieldWeight" in load


def test_a_fidelity_is_tt_metals_word():
    with pytest.raises(km.ProgramError, match="fidelity='HiFi5' is not one of"):
        km.tile_function(fidelity="HiFi5")(gate.function)


class Note(km.Message):
    hops: int


class TwoRuns(km.Program):
    @km.handler
    def first(self, notes: int):
        if self.seeding():
            self.send(self.rank, Note(hops=notes))
        for message in self.arrive():
            self.forward(self.rank, message, to=TwoRuns.second)

    @km.handler
    def second(self, hops: int):
        for message in self.arrive():
            self.forward(hops, message)


class ReadWords(km.Program):
    counts: km.L1[km.u32, 4]

    @km.handler
    def first(self):
        for message in self.arrive():
            seen = self.counts[0] + 1
            self.forward(seen, message)


class ToStatic(km.Program):
    @km.handler
    def first(self):
        for message in self.arrive():
            self.forward(self.rank, message, to=ToStatic.copy)

    @km.handler
    def copy(self, source: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()


def test_one_dynamic_handler_of_a_class_has_lines_besides_its_loop():
    line = line_of("def second(self, hops: int):")
    with pytest.raises(Refused, match=rf"^TwoRuns.second, line {line}: first has"):
        made(TwoRuns)


def test_a_word_field_is_read_in_a_line_that_sets_one():
    line = line_of("seen = self.counts[0] + 1")
    with pytest.raises(Refused, match=rf"^ReadWords.first, line {line}, .*words the"):
        made(ReadWords)


def test_to_names_a_dynamic_handler():
    line = line_of("self.forward(self.rank, message, to=ToStatic.copy)")
    with pytest.raises(Refused, match=rf"^ToStatic.first, line {line}, .*to= names"):
        made(ToStatic, source=tensor("bfloat16"))


class Placed(km.Program):
    places: km.L1[km.u32, 4]

    @km.handler
    def place(self, source: km.Tensor):
        self.places[0] = self.core
        self.places[1] += self.rank
        km.Stream(source.pages(0, 1)).drain()


def test_a_word_field_is_written_by_the_unpack_cpu():
    compiled = compile_class(Placed, {}, {"source": tensor("bfloat16")})
    text = render(prune(compiled.compute["place"]), 0)
    assert "const uint32_t km_self_core = compute.me();" in text
    assert "UNPACK((km_field_places[0] = km_self_core));" in text
    assert "UNPACK((km_field_places[1] += km_rank));" in text
    assert compiled.word_fields == {"places": 4} and "places" not in compiled.fields


WALK = {"enable_trisc2_rvv": True}


class Bodies(km.Program, compute=WALK):
    table: km.L1[km.u32, 2, 32, 32]

    @km.handler
    def load(self, image: km.Tensor):
        self.table.fill(km.Stream(image.pages(0, 2)))

    @km.handler
    def walk(self, places: km.Tensor, items: km.Tensor, out: km.Tensor, mode: int):
        first, count = places.pages(self.rank, 1).read()[0:2]
        made = km.Stream(out.items(first, count, size=1024), write=True, pattern="B")
        base = km.cxx("base = table;", takes=[self.table], gives=int, on="pack")
        for item in km.Stream(items.items(first, count, size=512), pattern="A"):
            km.cxx("use(item, made, base, mode);", takes=[item, made, base, mode],
                   on="pack", include=["walk.hpp"])  # fmt: skip
        km.cxx("note(places);", takes=[places], on="mover_b")


def test_inline_bodies_reach_items_words_cpus_and_kept_fields():
    names = ("places", "items", "out")
    tensors = {"image": tensor("uint32"), **{name: tensor("float32") for name in names}}
    compiled = compile_class(Bodies, {}, tensors)
    movement = "\n".join(render(prune(compiled.movement["walk"]), 0))
    compute = "\n".join(render(prune(compiled.compute["walk"]), 0))
    assert "km::items(Items(items), 512, 8)" in movement
    assert "km::items(Out(out), 1024, 4)" in movement
    assert "cb_push_back(kWords, 1);" in movement  # mover A hands the words on
    assert "const Places places(km_word_places);" in movement
    assert "first = compute.handed(ckernel::ThreadId::UnpackThreadId);" in compute
    assert "cb_pop_front(kFieldTable" not in compute  # kept: load's next fill frees it
    assert "compute.hand(ckernel::ThreadId::PackThreadId, get_tile_l1_byte_address(" \
        "km_half0, km_tile0));" in compute  # fmt: skip
    assert '#if defined(TRISC_PACK)\n#include "walk.hpp"\n#endif' in (
        compiled.body_includes["compute"]
    )
    assert compiled.compute_options == WALK
    assert [b.page_bytes for b in compiled.buffers if b.role == "read"] == [512]
    assert len(compiled.handovers) == 4


class Budget:
    """A session's L1 as its load counts it (Session.l1_rows), without a card."""

    channel, largest_page, transfer_chunk = object(), 2048, None
    l1_rows = Session.l1_rows

    def __init__(self, free):
        self.free = free

    def l1(self):
        return 0, self.free

    def l1_held(self):
        return []


def test_a_class_built_for_a_session_leaves_the_loads_rows():
    alone = made(Unsolved, l1=1_000_000, source=tensor("bfloat16"))
    lines = []
    loaded = Unsolved(
        log=lines.append, session=Budget(1_000_000), source=tensor("bfloat16")
    )
    assert "  plan: TILES = 240" in alone.built.report
    assert "  plan: TILES = 213" in loaded.built.report  # 27 tiles the runtime's
    table = lines[-1]
    for row in ("control block and overlay area", "host transfer buffer"):
        assert row in table
    assert "slots: service (reader), 1 x 9,216 B (at most)" in table
    with pytest.raises(ProgramError, match="its buffers take 999424 B, 891632 B"):
        Unsolved(
            log=lambda line: None,
            session=Budget(1_000_000),
            plan={"TILES": 240},
            source=tensor("bfloat16"),
        )


class Owned(km.Program):
    state: km.L1[km.fp32, 2, 32, 32] = km.own(push=None)
    fifo: km.L1[km.bf16, 3, 32, 32] = km.own()

    @km.handler
    def step(self, source: km.Tensor):
        for tile in km.Stream(source.pages(0, 8)):
            km.cxx("pack_tile<true>(0, state, 1);", takes=[tile, self.state])
            km.cxx("cb_push_back(fifo, 1);", takes=[self.fifo], on="pack")
            km.cxx("cb_pop_front(fifo, 1);", takes=[self.fifo], on="unpack")


class OwnedFilled(km.Program):
    state: km.L1[km.fp32, 2, 32, 32] = km.own(push=None)
    fifo: km.L1[km.bf16, 3, 32, 32] = km.own()

    @km.handler
    def load(self, source: km.Tensor):
        self.fifo.fill(km.Stream(source.pages(0, 3)))


class OwnedTwice(km.Program):
    state: km.L1[km.fp32, 2, 32, 32] = km.own(push=None)
    fifo: km.L1[km.bf16, 3, 32, 32] = km.own()

    @km.handler
    def step(self, source: km.Tensor):
        for tile in km.Stream(source.pages(0, 8)):
            km.cxx("cb_push_back(fifo, 1);", takes=[tile, self.fifo], on="pack")

    @km.handler
    def give(self, source: km.Tensor):
        km.Stream(source.pages(0, 1)).drain()
        km.cxx("cb_push_back(fifo, 1);", takes=[self.fifo], on="mover_a")


class OwnedPushed(km.Program):
    state: km.L1[km.fp32, 2, 32, 32] = km.own(push=None)
    fifo: km.L1[km.bf16, 3, 32, 32] = km.own()

    @km.handler
    def give(self, source: km.Tensor):
        km.cxx("cb_push_back(state, 1);", takes=[self.state], on="pack")


def test_an_own_field_is_a_buffer_of_bodies_with_one_producer():
    compiled = compile_class(Owned, {}, {"source": tensor("bfloat16")})
    owned = {b.name: b for b in compiled.buffers if b.role == "field"}
    assert owned["kFieldState"].pages == (2,) and owned["kFieldFifo"].pages == (3,)
    assert owned["kFieldFifo"].what.endswith(
        "own: used by bodies, not filled by a line"
    )
    unpacked = compute_options(compiled)["unpack_to_dest_mode"]  # float32 in DST
    assert unpacked[owned["kFieldState"].index_a] == "UnpackToDestFp32"
    compute = "\n".join(render(prune(compiled.compute["step"]), 0))
    assert "const uint32_t state = kFieldState;" in compute
    assert "kFieldFifo)" not in compute  # no wait, no free: the bodies' own
    for cls, match in (
        (OwnedFilled, "is an own field: only bodies use it, no line fills it"),
        (OwnedTwice, "is pushed on pack too .step, line .*: a buffer has one"),
        (OwnedPushed, r"is km.own\(push=None\): written in place, never pushed"),
    ):
        with pytest.raises(Refused, match=match):
            compile_class(cls, {}, {"source": tensor("bfloat16")})


class Assembled(km.Program):
    words: km.L1[km.u32, 1, 32, 32] = km.own(push="mover_a")

    @km.handler
    def page(self, source: km.Tensor, out: km.Tensor):
        for tile in km.Stream(source.pages(0, 8), pattern="A"):
            km.cxx("look(tile);", takes=[tile], on="pack")
        km.cxx("hand(words, tile_tiles, tile_cycles);", takes=[tile, self.words],
               on="mover_a")  # fmt: skip
        report = km.Stream(out.pages(self.rank, 1), write=True, pattern="B")
        km.cxx("fill(report);", takes=[report], on="pack")
        km.cxx("add(report, words);", takes=[report, self.words], on="mover_b")


class AssembledOnA(km.Program):
    @km.handler
    def page(self, out: km.Tensor):
        report = km.Stream(out.pages(self.rank, 1), write=True, pattern="B")
        km.cxx("fill(report);", takes=[report], on="pack")
        km.cxx("add(report);", takes=[report], on="mover_a")


class AssembledFirst(km.Program):
    @km.handler
    def page(self, out: km.Tensor):
        report = km.Stream(out.pages(self.rank, 1), write=True, pattern="B")
        km.cxx("add(report);", takes=[report], on="mover_b")


class AssembledByA(km.Program):
    @km.handler
    def page(self, out: km.Tensor):
        report = km.Stream(out.pages(self.rank, 1), write=True, pattern="A")
        km.cxx("fill(report);", takes=[report], on="pack")
        km.cxx("add(report);", takes=[report], on="mover_b")


def test_mover_b_adds_words_to_a_page_before_the_movers_write_it():
    tensors = {"source": tensor("bfloat16"), "out": tensor("uint32")}
    compiled = compile_class(Assembled, {}, tensors)
    movement = "\n".join(render(prune(compiled.movement["page"]), 0))
    assert (
        "km::Moved km_moved0 = {0, 0};\nif (!movement.move(km_moved0, km_read0))"
        in movement
    )
    assert "const uint32_t tile_cycles = km_moved0.cycles;" in movement  # mover A's
    waited = movement.index("if (!movement.wait_front(km_out_report, kWrite0Block))")
    body = movement.rindex("add(report, words);")  # the body, not its comment
    assert waited < body < movement.index("if (!movement.move(km_write0))")
    assert "const uint32_t report = get_read_ptr(km_out_report);" in movement
    assert "const uint32_t words = kFieldWords;" in movement  # A to B by an own field
    for cls, match in (
        (AssembledOnA, "report is a write stream: its pages are pack's to fill"),
        (AssembledFirst, "report's page is mover B's after pack's body fills it"),
        (AssembledByA, "report's block is not mover B's to write"),
    ):
        with pytest.raises(Refused, match=match):
            compile_class(cls, {}, {"out": tensor("uint32")})


class Patterned(km.Program):
    @km.handler
    def copy(self, source: km.Tensor):
        km.Stream(source.pages(0, 8), pattern="A").drain()
        km.Stream(source.pages(8, 8)).drain()


def test_a_stated_pattern_is_used_and_an_unstated_one_is_the_default():
    compiled = compile_class(Patterned, {}, {"source": tensor("bfloat16")})
    assert [b.patterns for b in compiled.buffers] == [
        {"kRead0Pattern": "A"},
        {"kRead1Pattern": "AAAB"},
    ]


class Compared(km.Program):
    @km.handler
    def copy(self, source: km.Tensor, n: int):
        km.Stream(source.pages(0, n + (n < 4) * 2)).drain()


def test_a_comparison_inside_arithmetic_keeps_its_parentheses():
    compiled = compile_class(Compared, {}, {"source": tensor("bfloat16")})
    movement = "\n".join(render(prune(compiled.movement["copy"]), 0))
    assert "Source(source), 0, (n + ((n < 4) * 2))," in movement


class Unpacked(km.Program):
    @km.handler
    def copy(self, items: km.Tensor):
        km.Stream(items.items(0, 4, size=1000)).drain()


class Untaken(km.Program):
    table: km.L1[km.u32, 2, 32, 32]

    @km.handler
    def use(self, items: km.Tensor):
        km.cxx("use(table);", takes=[self.table], on="pack")


class Outside(km.Program):
    @km.handler
    def copy(self, items: km.Tensor):
        first, count, more = items.pages(0, 1).read()[0:2]


class Unknown(km.Program, compute={"dst_full_sync": True}):
    @km.handler
    def copy(self, items: km.Tensor):
        km.Stream(items.pages(0, 1)).drain()


@pytest.mark.parametrize(
    "cls, text, match",
    [
        (Unpacked, "size=1000)).drain()", "items of 1000 B in pages of 4096 B"),
        (Untaken, 'takes=[self.table], on="pack")', "no other handler keeps it"),
        (Outside, "first, count, more =", "words 0 to 1 of a page of 1024 words"),
    ],
)
def test_a_body_line_the_compiler_does_not_take_is_refused_by_its_line(
    cls, text, match
):
    line = line_of(text)
    with pytest.raises(Refused, match=rf"line {line}, .*{match}"):
        compile_class(cls, {}, {name: tensor("float32") for name in ("items", "out")})


def test_an_unknown_compute_option_is_refused_by_name():
    line = line_of('class Unknown(km.Program, compute={"dst_full_sync": True}):')
    with pytest.raises(Refused, match=rf"^Unknown, line {line}: .*'dst_full_sync'"):
        compile_class(Unknown, {}, {"items": tensor("float32")})


class Streamed(km.Program):
    sums: km.L1[km.fp32, 1, 4, 32, 32]
    part: km.L1[km.fp32, 2, 3, 32, 32]

    @km.handler
    def dst(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        k0 = self.rank * 2
        acc = km.tile.zeros(km.fp32, 2)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + 2], block=1),
            km.Stream(w.tiles[k0 : k0 + 2, 0:2], block=2),
        ):
            acc += coarse(xb, wb)
        y.tiles[1:2, 0:2].write(acc)

    @km.handler
    def split(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        k0 = self.rank % 2 * 2
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + 2], block=1),
            km.Stream(w.tiles[k0 : k0 + 2, 0:4], block=4),
        ):
            self.sums += coarse(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=2)
        if self.rank == self.leader:
            y.tiles[0:1, 0:4].write(whole)

    @km.handler
    def ragged(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        along = km.Broadcast(self, along="row")
        down = km.Broadcast(self, along="column")
        left = self.x * 3
        for chunk in range(2):
            k0 = chunk * 2
            if along.sender():
                xs = along.cast(x.tiles[0:2, k0 : k0 + 2].read())
            else:
                xs = along.arrive()
            if down.sender():
                ws = down.cast(w.tiles[k0 : k0 + 2, left : min(left + 3, 4)].read())
            else:
                ws = down.arrive()
            self.part += coarse(xs, ws)
        y.tiles[0:2, left : min(left + 3, 4)].write(self.part)


def test_products_over_two_streams_reduces_of_fields_and_word_bounds():
    tensors = {name: grid("bfloat16", 4, 4) for name in ("x", "w", "y")}
    compiled = compile_class(Streamed, {}, tensors)
    dst = "\n".join(render(prune(compiled.compute["dst"]), 0))
    assert (
        "matmul_block(km_half0, km_half1, km_tile0, km_tile0 * 2, 0, 0, 2, 1, "
        "kRead0Block);" in dst
    )
    split = "\n".join(render(prune(compiled.compute["split"]), 0))
    assert "for (uint32_t km_k" in split and "< km_valid0;" in split  # the K tiles
    assert "pack_tile<true>(3, kFieldSums, km_sub_column" in split
    assert "km::reduce(compute, kArrival0" in split
    moved = "\n".join(render(prune(compiled.movement["split"]), 0))
    assert "kReduceTree, kFieldSums, kFieldSumsPages)" in moved  # a sender's field
    ragged = "\n".join(render(prune(compiled.movement["ragged"]), 0))
    assert "((left + 3) < 4 ? (left + 3) : 4)) - (left)), km_address" in ragged
    assert "+ km_row * 6144)" in ragged  # the slot's rows at 3 columns' pitch
    slots = {b.name: b.pages[0] for b in compiled.buffers if b.role == "broadcast"}
    assert slots["kBroadcast1"] == 2 * 2 * 3  # window 2, the most: 2 x 3 tiles
    assert {"kFieldSums", "kFieldPart"} <= compiled.summed_fields


class Unproduct(km.Program):
    @km.handler
    def apply(self, a: km.Tensor, w: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 2), write=True)
        for h, g in zip(
            km.Stream(a.tiles[0:1, 0:2], block=1), km.Stream(w.tiles[0:2, 0:2], block=2)
        ):
            results.push(gate(h, g))


class Unblocked(km.Program):
    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=1), km.Stream(w.tiles[0:2, 0:3], block=3)
        ):
            acc += coarse(xb, wb)  # unblocked
        y.tiles[0:1, 0:2].write(acc)


class Swapped(km.Program):
    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 2)
        for wb, xb in zip(
            km.Stream(w.tiles[0:2, 0:2], block=2), km.Stream(x.tiles[0:1, 0:2], block=1)
        ):
            acc += coarse(xb, wb)  # swapped
        y.tiles[0:1, 0:2].write(acc)


class Unsummed(km.Program):
    sums: km.L1[km.fp32, 1, 2, 32, 32]

    @km.handler
    def apply(self, y: km.Tensor):
        whole = km.reduce(self.sums, form="tree", group=2)  # unsummed
        if self.rank == self.leader:
            y.tiles[0:1, 0:2].write(whole)


class Owners(km.Program):
    sums: km.L1[km.fp32, 1, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=1), km.Stream(w.tiles[0:2, 0:2], block=2)
        ):
            self.sums += coarse(xb, wb)
        whole = km.reduce(self.sums, form="owner")  # owners
        if self.rank == self.leader:
            y.tiles[0:1, 0:2].write(whole)


class Unbounded(km.Program):
    @km.handler
    def apply(self, x: km.Tensor, n: int):
        down = km.Broadcast(self, along="column")
        if down.sender():
            arrived_blocks = down.cast(x.tiles[0:2, 0:n].read())  # unbounded
        else:
            arrived_blocks = down.arrive()
        for _tile in arrived_blocks:
            pass


class Overwide(km.Program):
    part: km.L1[km.fp32, 1, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=1), km.Stream(w.tiles[0:2, 0:2], block=2)
        ):
            self.part += coarse(xb, wb)
        y.tiles[0:1, 0 : min(3, self.rank)].write(self.part)  # overwide


class Unfillable(km.Program):
    weight: km.L1[km.bf16, 2, 2, 32, 32]

    @km.handler
    def load(self, w: km.Tensor, n: int):
        self.weight.fill(km.Stream(w.tiles[0:2, 0:n]))  # unfillable


class Lone(km.Program):
    @km.handler
    def copy(self, x: km.Tensor):
        km.Stream(x.pages(0, min(2))).drain()  # lone


@pytest.mark.parametrize(
    "cls, handler, text, match",
    [
        (
            Unproduct,
            "apply",
            "for h, g in zip(",
            "blocks of different sizes are a product's",
        ),
        (
            Unblocked,
            "apply",
            "# unblocked",
            "each block of the second stream is the first's",
        ),
        (Swapped, "apply", "# swapped", "the loop counts the first stream's tiles"),
        (
            Unsummed,
            "apply",
            "# unsummed",
            "self.sums holds no sums above in this handler",
        ),
        (Owners, "apply", "# owners", "by owner holds a slot of its 2 tiles"),
        (Unbounded, "apply", "# unbounded", "the build knows no most for a side here"),
        (
            Overwide,
            "apply",
            "# overwide",
            "a side a word: the build must know it is at most",
        ),
        (Unfillable, "load", "# unfillable", "the build must know they are at most 2"),
        (Lone, "copy", "# lone", r"min\(...\) of level 1 takes two ints or more"),
    ],
)
def test_a_line_of_products_over_streams_or_word_bounds_is_refused_by_its_line(
    cls, handler, text, match
):
    line = line_of(text)
    tensors = {name: grid("bfloat16", 4, 4) for name in ("x", "w", "y", "a", "out")}
    names = inspect.signature(getattr(cls, handler).function).parameters
    with pytest.raises(Refused, match=rf"line {line}, .*{match}"):
        compile_class(cls, {}, {n: tensors[n] for n in names if n in tensors})


class KeptOwn(km.Program):
    tree: km.L1[km.bf16, 1, 32, 32]

    @km.handler
    def load(self, source: km.Tensor):
        self.tree.fill(km.Stream(source.pages(0, 1)))
        km.cxx("(void)tree;", takes=[self.tree], on="mover_b")

    @km.handler
    def use(self, out: km.Tensor):
        km.cxx("(void)tree;", takes=[self.tree], on="mover_b")


class FedOther(km.Program):
    @km.handler
    def copy(self, source: km.Tensor, out: km.Tensor, tiles: int, others: int):
        results = km.Stream(out.pages(0, others), write=True, block=4)
        for tile in km.Stream(source.pages(0, tiles), block=8):
            results.push(tile)  # fed other


class FilledAdded(km.Program):
    sums: km.L1[km.fp32, 1, 2, 32, 32]

    @km.handler
    def go(self, x: km.Tensor, w: km.Tensor, a: km.Tensor, k0: int, k1: int):
        self.sums.fill(km.Stream(a.pages(0, 2)))  # filled added
        km.cxx("(void)sums;", takes=[self.sums], on="mover_b")
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=2),
            km.Stream(w.tiles[k0:k1, 0:2], block=4),
        ):
            self.sums += coarse(xb, wb)  # added filled


class Rows(km.Program):
    @km.handler
    def part(self, x: km.Tensor, y: km.Tensor):
        mine = self.partition(x.tiles[0:4, 0:2])  # rows partition
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x[mine], block=2):
            total += tile
        y.pages(self.rank, 1).write(total)


class Prefixed(km.Program):
    @km.handler
    def copy(self, x: km.Tensor, tiles: int):
        km_row = tiles + 1  # prefixed
        km.Stream(x.pages(0, km_row)).drain()


class Truthy(km.Program):
    @km.handler
    def copy(self, x: km.Tensor, tiles: int):
        n = tiles or 1  # truthy
        km.Stream(x.pages(0, n)).drain()


class Below(km.Program):
    @km.handler
    def copy(self, x: km.Tensor, tiles: int, done: int):
        n = 1
        if tiles - done > 0:  # below
            n = 2
        km.Stream(x.pages(0, n)).drain()


class Reset(km.Program):
    @km.handler
    def copy(self, x: km.Tensor, tiles: int):
        tiles = tiles + 1  # reset
        km.Stream(x.pages(0, tiles)).drain()


class Ended(km.Program):
    @km.handler
    def copy(self, x: km.Tensor, tiles: int):
        if tiles > 4:
            bonus = 7
        km.Stream(x.pages(0, bonus)).drain()  # ended


class Deep(km.Program):
    weight: km.L1[km.bf16, 4, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, tiles: int):
        self.weight.fill(km.Stream(w.pages(0, 8)))
        acc = km.tile.zeros(km.fp32, 2)
        for xt in km.Stream(x.pages(0, tiles), block=2):
            acc += project(xt, self.weight)
        y.pages(0, 2).write(acc)


class TooDeep(km.Program):
    weight: km.L1[km.bf16, 4, 2, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 8)))
        acc = km.tile.zeros(km.fp32, 2)
        for xt in km.Stream(x.pages(0, 8), block=2):
            acc += project(xt, self.weight)  # too deep
        y.pages(0, 2).write(acc)


class TreeOfOne(km.Program):
    @km.handler
    def merge(self, x: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.pages(self.rank, 1)):
            total += tile
        whole = km.reduce(total, form="tree")
        if self.rank == self.leader:
            y.pages(0, 1).write(whole)


class HalfSync(km.Program, compute={"dst_full_sync_en": False}):
    @km.handler
    def apply(self, x: km.Tensor):
        acc = km.tile.zeros(km.fp32, 6)
        for tile in km.Stream(x.pages(0, 4), block=4):
            acc += tile


class Nested(km.Program):
    @km.handler
    def page(self, x: km.Tensor, n: int):
        if n > 0:
            for tile in km.Stream(x.pages(0, 8), pattern="A"):
                km.cxx("look(tile);", takes=[tile], on="pack")
        km.cxx("hand(tile_tiles);", takes=[tile], on="mover_a")  # nested


@pytest.mark.parametrize(
    "cls, handler, text, match",
    [
        (FedOther, "copy", "# fed other", "results is others pages in blocks of 4"),
        (FilledAdded, "go", "# added filled", "self.sums is pushed by mover A too"),
        (Rows, "part", "# rows partition", "a rectangle of rows is not one"),
        (Prefixed, "copy", "# prefixed", "its own names start km_"),
        (Truthy, "copy", "# truthy", r"write max\(a, 1\), or an if"),
        (Below, "copy", "# below", "compare the two: write tiles > done"),
        (Reset, "copy", "# reset", "tiles is a parameter, a command word"),
        (Ended, "copy", "# ended", "bonus is first assigned inside the if"),
        (TooDeep, "apply", "# too deep", "self.weight is 4 K rows, and the loop's"),
        (Nested, "page", "# nested", "the loop is inside the if at line"),
    ],
)
def test_a_line_the_hunt_found_unsound_is_refused_by_its_line(
    cls, handler, text, match
):
    line = line_of(text)
    tensors = {"source": tensor("bfloat16"), "out": tensor("bfloat16")}
    tensors |= {name: grid("bfloat16", 4, 4) for name in ("x", "w", "y")}
    tensors["a"] = tensor("float32")
    names = inspect.signature(getattr(cls, handler).function).parameters
    with pytest.raises(Refused, match=rf"line {line}, .*{match}"):
        compile_class(cls, {}, {n: tensors[n] for n in names if n in tensors})


def test_the_hunts_generated_lines_hold_what_the_build_says():
    compiled = compile_class(KeptOwn, {}, {"source": tensor("bfloat16")})
    compute = "\n".join(render(prune(compiled.compute["load"]), 0))
    assert compute.count("cb_pop_front(kFieldTree") == 1  # the last fill's, held
    tensors = {name: grid("bfloat16", 4, 4) for name in ("x", "w", "y")}
    compiled = compile_class(Deep, {}, tensors)
    compute = "\n".join(render(prune(compiled.compute["apply"]), 0))
    assert compute.count("if (tiles > 4) {") == 1  # one compare a command
    assert "compute.gave_up(km::kGaveBadItem, km::kWhereCInput);" in compute
    tensors = {"x": tensor("bfloat16"), "y": tensor("float32")}
    compiled = compile_class(TreeOfOne, {}, tensors)
    (arrival,) = [b for b in compiled.buffers if b.role == "arrival"]
    assert arrival.pages[0] == 7  # a tree's rounds, not a slot a sender
    with pytest.raises(Refused, match="'dst_full_sync_en' = False: the class's"):
        compile_class(HalfSync, {}, {"x": tensor("bfloat16")})


def test_two_builds_of_a_class_have_a_folder_each():
    from komorebi.dsl.program import sources_key  # noqa: PLC0415

    small, wide = {"x": grid("bfloat16", 4, 4)}, {"x": grid("bfloat16", 4, 8)}
    keys = {
        sources_key({"BLOCK": 4}, small, None, 4, None, None),
        sources_key({"BLOCK": 8}, small, None, 4, None, None),
        sources_key({"BLOCK": 4}, wide, None, 4, None, None),
    }
    assert len(keys) == 3


class Walked(km.Program):
    @km.handler
    def copy(self, x: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(x.tiles[0:2, 0:2], block=2):
            total += tile
        y.pages(0, 1).write(total)


def test_a_tensor_of_another_tile_width_is_refused_at_submit():
    program = made(Walked, x=grid("bfloat16", 4, 4), y=tensor("float32"))
    assert len(program.copy.args(grid("bfloat16", 2, 4), tensor("float32"))) == 2
    with pytest.raises(ProgramError, match="x is 8 tiles wide, .* built for 4"):
        program.copy.args(grid("bfloat16", 4, 8), tensor("float32"))
