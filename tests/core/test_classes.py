"""A program written as a class that the compiler does not take is refused when it is
made, with a sentence that names the handler and the line (no card)."""

import inspect
from pathlib import Path

import pytest

import komorebi as km
from komorebi.dsl.compiler import Refused, compile_class
from komorebi.dsl.nodes import prune, render
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


def test_a_matrix_product_in_a_tile_function_is_refused_by_its_line():
    line = line_of("results.push(product(x, y))")
    product_line = line_of("return a @ b")
    with pytest.raises(
        Refused, match=rf"^Product.multiply, line {line}, .*line {product_line}, has a"
    ):
        made(Product, a=tensor("float32"), b=tensor("float32"), out=tensor("float32"))


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


def test_a_product_of_float32_tiles_is_refused_by_its_line():
    line = line_of("# narrow")
    with pytest.raises(Refused, match=rf"^Narrow.apply, line {line}, .*bfloat16 tiles"):
        made(Narrow, x=tensor("float32"), w=tensor("bfloat16"))


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
        km.Stream(x.tiles[0:4:2, 0:4]).drain()  # stepped


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
    (Ragged, "copy", "# ragged", "a constant number of columns"),
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
    assert "const uint32_t self_core = compute.me();" in text
    assert "UNPACK((field_places[0] = self_core));" in text
    assert "UNPACK((field_places[1] += rank));" in text
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
        "half0, tile0));" in compute  # fmt: skip
    assert '#if defined(TRISC_PACK)\n#include "walk.hpp"\n#endif' in (
        compiled.body_includes["compute"]
    )
    assert compiled.compute_options == WALK
    assert [b.page_bytes for b in compiled.buffers if b.role == "read"] == [512]
    assert len(compiled.handovers) == 4


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


class Pushed(km.Program):
    @km.handler
    def copy(self, items: km.Tensor, out: km.Tensor):
        made = km.Stream(out.pages(0, 1), write=True)
        km.cxx("fill(made);", takes=[made])


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
        (Pushed, 'km.cxx("fill(made);"', "its pages are pack's to fill"),
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
