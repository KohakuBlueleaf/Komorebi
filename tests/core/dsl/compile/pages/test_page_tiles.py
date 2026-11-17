"""A read stream over a tensor in DRAM pages of k tiles (page_tiles): the walk in
tiles as at k = 1, the tensor read through km::pages_of<k>; and each refusal: a
block, a first column or a width not whole pages, a stream of its pages, any other
line over it, a direct input, a tensor of another k at a submit (no card)."""

from types import SimpleNamespace

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl import program as made
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render
from komorebi.errors import ProgramError

TILE = {"bfloat16": 2048, "bfp8_b": 1088, "bfp4_b": 576}


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(rows, columns, dtype="bfp8_b", k=1):
    """A tile tensor of rows x columns tiles in pages of k tiles."""
    made_ = Tensor(
        Pages(k * TILE[dtype], rows * columns // k),
        (rows * 32, columns * 32),
        dtype,
        "tile",
    )
    made_.page_tiles = k
    return made_


class Product(km.Program):
    KS = km.constant(42)
    W = km.constant(8)
    CHUNK = km.constant(2)
    BW = CHUNK * W

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def product(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, col0: int):
        k0 = self.rank % 4 * self.KS
        first = col0 + self.rank // 4 * self.W
        acc = km.tile.zeros(km.fp32, self.W)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + self.KS], block=self.CHUNK),
            km.Stream(
                w.tiles[k0 : k0 + self.KS, first : first + self.W], block=self.BW
            ),
        ):
            acc += self.project(xb, wb)
        y.tiles[0:1, first : first + self.W].write(acc)


CONSTANTS = {"KS": 42, "W": 8, "CHUNK": 2, "BW": 16}


def compiled(cls, k, constants=CONSTANTS, w=None):
    tensors = {
        "x": tensor(1, 168, "bfloat16"),
        "w": w or tensor(168, 240, "bfp8_b", k),
        "y": tensor(1, 240, "bfloat16"),
    }
    return compile_class(cls, {}, tensors, constants=constants)


def movement(result, handler="product"):
    return "\n".join(render(prune(result.movement[handler]), 0))


def test_k_wraps_the_tensor_and_keeps_the_walk_and_the_rest_as_at_one_tile():
    one, four = compiled(Product, 1), compiled(Product, 4)
    text1, text4 = movement(one), movement(four)
    assert "km::pages_of<" not in text1
    assert "km::kMoveRead, km::pages_of<4>(W(w)), km::Walk{" in text4
    assert text4.replace("km::pages_of<4>(W(w))", "W(w)") == text1
    assert [b.page_bytes for b in four.buffers if b.name == "kRead1"] == [1088]
    assert one.page_tiles[("product", "w")] == 1
    assert four.page_tiles[("product", "w")] == 4
    compute = ["\n".join(render(prune(c.compute["product"]), 0)) for c in (one, four)]
    assert compute[0] == compute[1]


def test_a_block_not_whole_pages_is_refused():
    with pytest.raises(Refused, match="block=16 over w in pages of 3 tiles: a block"):
        compiled(Product, 3, w=tensor(168, 240, "bfp8_b", 3))


class Narrow(Product):
    @km.handler
    def product(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, col0: int):
        acc = km.tile.zeros(km.fp32, 6)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:42], block=2),
            km.Stream(w.tiles[0:42, 2:8], block=12),
        ):
            acc += self.project(xb, wb)
        y.tiles[0:1, 0:6].write(acc)


class Ragged(Narrow):
    @km.handler
    def product(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, col0: int):
        acc = km.tile.zeros(km.fp32, 6)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:42], block=2),
            km.Stream(w.tiles[0:42, 4:10], block=12),
        ):
            acc += self.project(xb, wb)
        y.tiles[0:1, 0:6].write(acc)


def test_a_first_column_or_a_width_not_whole_pages_is_refused():
    with pytest.raises(Refused, match="first column 2 over pages of 4 tiles"):
        compiled(Narrow, 4, w=tensor(168, 240, "bfp8_b", 4))
    with pytest.raises(Refused, match="width 6 over pages of 4 tiles"):
        compiled(Ragged, 4, w=tensor(168, 240, "bfp8_b", 4))
    compiled(Narrow, 2, w=tensor(168, 240, "bfp8_b", 2))


class OfPages(km.Program):
    @km.handler
    def drain(self, w: km.Tensor, first: int):
        km.Stream(w.pages(first, 16), block=16).drain()


def test_a_stream_of_its_pages_is_refused():
    with pytest.raises(Refused, match="a stream of it reads its tiles"):
        compile_class(OfPages, {}, {"w": tensor(168, 240, "bfp8_b", 4)})
    compile_class(OfPages, {}, {"w": tensor(168, 240, "bfp8_b", 1)})


class Filled(km.Program):
    wf: km.L1[km.bfp8_b, 2, 8, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def load(self, w: km.Tensor):
        self.wf.fill(km.Stream(w.tiles[0:2, 0:8]))

    @km.handler
    def held(self, x: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32, 8)
        for xb in km.Stream(x.tiles[0:1, 0:2], block=2):
            acc += self.project(xb, self.wf)
        y.tiles[0:1, 0:8].write(acc)


def test_any_other_line_over_it_is_refused_by_name():
    given = {"x": tensor(1, 168, "bfloat16"), "y": tensor(1, 240, "bfloat16")}
    with pytest.raises(Refused, match=r"w is in pages of 4 tiles \(page_tiles=4\)"):
        compile_class(Filled, {}, given | {"w": tensor(168, 240, "bfp8_b", 4)})
    compile_class(Filled, {}, given | {"w": tensor(168, 240, "bfp8_b", 1)})


def test_a_direct_input_in_pages_of_k_tiles_is_refused():
    compiled_ = SimpleNamespace(
        name="Layer", direct_area=(16, 1088, [("product", "w")])
    )
    lines = ["km::kMoveRead, km::pages_of<4>(W(w)), km::Walk{0, 1, 1, 1, 1, 1},"]
    with pytest.raises(ProgramError, match="w is in pages of several tiles"):
        made.direct_reads(compiled_, "product", lines)
    plain = ["km::kMoveRead, W(w), km::Walk{0, 1, 1, 1, 1, 1},"]
    assert "km::direct_or(W(w)" in made.direct_reads(compiled_, "product", plain)[0]


def test_a_submit_of_a_tensor_of_another_k_is_refused():
    program = Product(
        x=tensor(1, 168, "bfloat16"),
        w=tensor(168, 240, "bfp8_b", 4),
        y=tensor(1, 240, "bfloat16"),
        KS=42,
        W=8,
        CHUNK=2,
    )
    method = type(program).product
    assert method.paged == {"x": 1, "w": 4, "y": 1}
    x, y = tensor(1, 168, "bfloat16"), tensor(1, 240, "bfloat16")
    method.args(x, tensor(168, 240, "bfp8_b", 4), y, 0)
    with pytest.raises(ProgramError, match="w is in pages of 2 tile"):
        method.args(x, tensor(168, 240, "bfp8_b", 2), y, 0)
