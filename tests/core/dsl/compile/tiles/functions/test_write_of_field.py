"""`pages.write(f(acc, self.b))` (no card): a field in a product's write, copied into
the DST tiles after acc's (or its row or column broadcast, `.row()` / `.col()`), f run
on both, acc written; the field waited for once, freed at the handler's end. Other
widths, DST too small, a product in f, broadcast sums or a pair are refused by name."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensors(bias="float32"):
    """float32 tensors of tiles w, x and y, and the bias's b of `bias` tiles."""
    page = formats.tile_bytes("float32")
    given = {n: Tensor(Pages(page), (2048, 256), "float32", "tile") for n in "wxy"}
    given["b"] = Tensor(Pages(formats.tile_bytes(bias)), (2048, 256), bias, "tile")
    return given


@km.tile_function
def biased(a, b):
    return km.silu(a + b)


@km.tile_function
def multiplied(a, b):
    return a @ b


@km.tile_function
def by_column(a, b):
    return km.silu(a + b.col())


@km.tile_function
def by_row(a, b):
    return a * b.row()


@km.tile_function
def sums_column(a, b):
    return a.col() + b


@km.tile_function
def two_ways(a, b):
    return a + b + b.col()


class Columns(km.Program):
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        self.bias.fill(km.Stream(b.pages(0, 1)))
        acc = km.tile.zeros(km.fp32, 2)
        for xb in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += xb
        y.tiles[0:1, 0:2].write(by_column(acc, self.bias))


class Rows(km.Program):
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        self.bias.fill(km.Stream(b.pages(0, 1)))
        acc = km.tile.zeros(km.fp32, 2)
        for xb in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += xb
        y.tiles[0:1, 0:2].write(by_row(acc, self.bias))


class SumsColumn(km.Program):
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        self.bias.fill(km.Stream(b.pages(0, 1)))
        acc = km.tile.zeros(km.fp32, 2)
        for xb in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += xb
        y.tiles[0:1, 0:2].write(sums_column(acc, self.bias))


class TwoWays(km.Program):
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        self.bias.fill(km.Stream(b.pages(0, 1)))
        acc = km.tile.zeros(km.fp32, 2)
        for xb in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += xb
        y.tiles[0:1, 0:2].write(two_ways(acc, self.bias))


class Pair(km.Program):
    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 1)
        other = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(x.tiles[0:1, 0:1]):
            acc += xb
            other += xb
        y.tiles[0:1, 0:1].write(by_column(acc, other))


class Wide(km.Program):
    """Two tiles of sums, a bias of two."""

    bias: km.L1[km.fp32, 2, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def product(a, b):
        return a @ b

    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        self.bias.fill(km.Stream(b.pages(0, 2)))
        for j in range(2):
            acc = km.tile.zeros(km.fp32, 2)
            for wb, xb in zip(
                km.Stream(w.tiles[0:1, 0:2], block=2),
                km.Stream(x.tiles[0:2, j : j + 2], block=4),
            ):
                acc += self.product(wb, xb)
            y.tiles[0:1, j : j + 2].write(biased(acc, self.bias))


class Shared(km.Program):
    """Two tiles of sums, a bf16 bias of one for each."""

    bias: km.L1[km.bf16, 1, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def product(a, b):
        return a @ b

    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        self.bias.fill(km.Stream(b.pages(0, 1)))
        acc = km.tile.zeros(km.fp32, 2)
        for wb, xb in zip(
            km.Stream(w.tiles[0:1, 0:2], block=2),
            km.Stream(x.tiles[0:2, 0:2], block=4),
        ):
            acc += self.product(wb, xb)
        y.tiles[0:1, 0:2].write(biased(acc, self.bias))


class Odd(km.Program):
    bias: km.L1[km.fp32, 3, 32, 32]

    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        self.bias.fill(km.Stream(b.pages(0, 3)))
        acc = km.tile.zeros(km.fp32, 2)
        for xb in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += xb
        y.tiles[0:1, 0:2].write(biased(acc, self.bias))


class Full(km.Program):
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        self.bias.fill(km.Stream(b.pages(0, 1)))
        acc = km.tile.zeros(km.fp32, 5)
        for xb in km.Stream(x.tiles[0:1, 0:5], block=5):
            acc += xb
        y.tiles[0:1, 0:5].write(biased(acc, self.bias))


class Product(km.Program):
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        self.bias.fill(km.Stream(b.pages(0, 1)))
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(x.tiles[0:1, 0:1]):
            acc += xb
        y.tiles[0:1, 0:1].write(multiplied(acc, self.bias))


class NoForm(km.Program):
    bias: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def go(self, w: km.Tensor, x: km.Tensor, b: km.Tensor, y: km.Tensor):
        self.bias.fill(km.Stream(b.pages(0, 1)))
        acc = km.tile.zeros(km.fp32, 1)
        for xb in km.Stream(x.tiles[0:1, 0:1]):
            acc += xb
        y.tiles[0:1, 0:1].write(biased(self.bias, acc))


def compute(cls, bias="float32"):
    compiled = compile_class(cls, {}, tensors(bias))
    return compiled, "\n".join(render(prune(compiled.compute["go"]), 0))


def refused(cls):
    with pytest.raises(Refused) as refusal:
        compute(cls)
    return str(refusal.value)


def test_the_fields_tiles_go_into_dst_after_the_sums_and_f_runs_on_both():
    compiled, text = compute(Wide)
    copied = text.index("// self.bias's tiles into DST 2 .. 3")
    wait = text.index("compute.wait_front(kFieldBias, kFieldBiasPages);", copied)
    first = text.index("copy_tile(kFieldBias, 0, 2);", wait)
    second = text.index("copy_tile(kFieldBias, 1, 3);", first)
    calls = text.index("MATH((km_tile::biased(0)));", second)
    calls = text.index("MATH((km_tile::biased(1)));", calls)
    packed = text.index("pack_tile(1, kOut0);", calls)
    freed = text.rindex("cb_pop_front(kFieldBias, kFieldBiasPages);")
    assert copied < wait < first < second < calls < packed < freed
    assert text.count("compute.wait_front(kFieldBias") == 1  # once a handler
    assert "kFieldBias" in compiled.unpack_fp32


def test_a_field_of_one_tile_serves_each_of_the_sums_tiles():
    compiled, text = compute(Shared, "bfloat16")
    assert "copy_tile(kFieldBias, 0, 2);\n    copy_tile(kFieldBias, 0, 3);" in text
    assert "kFieldBias" not in compiled.unpack_fp32  # bf16: unpacked as it is


@pytest.mark.parametrize("cls, along", [(Columns, "COL"), (Rows, "ROW")])
def test_a_fields_column_or_row_is_broadcast_into_dst(cls, along):
    compiled, text = compute(cls)
    kind = f"BroadcastType::{along}"
    init = text.index(f"unary_bcast_init<{kind}>(kFieldBias);")
    first = text.index(f"unary_bcast<{kind}>(kFieldBias, 0, 2);", init)
    second = text.index(f"unary_bcast<{kind}>(kFieldBias, 0, 3);", first)
    assert second < text.index(f"unary_bcast_uninit<{kind}>(kFieldBias);")
    assert "copy_tile(kFieldBias" not in text
    assert "bcast" in compiled.includes


def test_sums_read_by_a_broadcast_are_refused():
    message = refused(SumsColumn)
    assert "sums_column reads a, DST sums, by `.row()` or `.col()`: a row or " in (
        message
    )


def test_a_field_read_two_ways_is_refused():
    message = refused(TwoWays)
    assert "two_ways reads b in two ways (whole, `.row()`, `.col()`): its DST " in (
        message
    )


def test_a_pairs_broadcast_is_refused():
    message = refused(Pair)
    assert "by_column reads an operand by `.row()` or `.col()`, and its operands " in (
        message
    )
    assert "write by_column(acc, self.b) of a field" in message


def test_a_field_of_another_width_is_refused():
    message = refused(Odd)
    assert "biased takes acc (2 tiles) and self.bias (3 tiles): tile t of each, " in (
        message
    )
    assert "declare self.bias of 2 tiles, or of 1" in message


def test_sums_whose_copies_overflow_dst_are_refused():
    message = refused(Full)
    assert "acc is 5 tiles of DST from tile 0, and self.bias's copies take 5 " in (
        message
    )
    assert "sum at most 4 tiles in acc" in message


def test_a_product_in_the_function_is_refused():
    assert "multiplied has a matrix product (`@`), and its operand here is acc" in (
        refused(Product)
    )


def test_a_write_of_a_tile_function_of_no_form_names_the_forms():
    message = refused(NoForm)
    assert "a write of a tile function takes DST sums: f(acc); f(a, b) of a pair " in (
        message
    )
    assert "f(acc, self.s[r]), a step; f(acc, self.b), self.b a field" in message
