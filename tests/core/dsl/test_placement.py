"""A product's placement (no card): km.srcb / km.srca mark an operand, a marker the
other way turns the product (W's tiles stored transposed, X's read transposed, DST
transposed back), and the build prints both placements' bits."""

import numpy as np
import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class, product_placement
from komorebi.dsl.program import compute_source, placements_of
from komorebi.errors import ProgramError


class Pages:
    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def tensor(dtype, turned=False):
    made = Tensor(Pages(formats.tile_bytes(dtype)), (64, 128), dtype, "tile")
    made.transposed_tiles = turned
    return made


@km.tile_function(fidelity="HiFi2")
def unmarked(x, w):
    return x @ w


@km.tile_function(fidelity="HiFi2")
def weight_in_b(x, w):
    return x @ km.srcb(w)


def x_in_a(x, w):
    return km.srca(x) @ w


def x_in_b(x, w):
    return km.srcb(x) @ w


def both_in_b(x, w):
    return km.srcb(x) @ km.srcb(w)


class Sums(km.Program):
    sums: km.L1[km.fp32, 1, 4, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(w.tiles[0:2, 0:4], block=8),
        ):
            self.sums += weight_in_b(xb, wb)
        y.tiles[0:1, 0:4].write(self.sums)


class Plain(km.Program):
    sums: km.L1[km.fp32, 1, 4, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:2], block=2),
            km.Stream(w.tiles[0:2, 0:4], block=8),
        ):
            self.sums += unmarked(xb, wb)
        y.tiles[0:1, 0:4].write(self.sums)


def test_a_marker_places_one_operand_and_the_other_follows():
    assert product_placement(unmarked.function) == "none"
    assert product_placement(x_in_b) == "plain"
    assert product_placement(weight_in_b.function) == "turned"
    assert product_placement(x_in_a) == "turned"
    assert "both operands are marked km.srcb" in product_placement(both_in_b)


def test_the_turned_product_reads_x_transposed_and_turns_dst_back():
    tensors = {"x": tensor("bfloat16"), "w": tensor("bfp8_b", True)}
    tensors["y"] = tensor("float32")
    compiled = compile_class(Sums, {}, tensors)
    source = compute_source(compiled)
    assert "matmul_block_init(kRead1.mover_a, kRead0.mover_a, 1, 1, 4, 1);" in source
    assert "KM_UNPACK_FORMATS(kRead0.mover_a, kRead1.mover_a);" in source
    assert source.count("transpose_dest<DST_ACCUM_MODE, true>(3);") == 1
    (line,) = placements_of(compiled, tensors)[1:]
    assert "left bfloat16, right bfp8_b, HiFi2" in line
    assert "plain (left in SrcB): left keeps 7, right 7" in line
    assert "turned (right in SrcB): left 8, right 7; built turned" in line


def test_the_weights_tiles_are_checked_against_the_placement():
    plain = {"x": tensor("bfloat16"), "w": tensor("bfp8_b"), "y": tensor("float32")}
    with pytest.raises(ProgramError, match="is turned and tensor w is not stored"):
        placements_of(compile_class(Sums, {}, plain), plain)
    turned = {**plain, "w": tensor("bfp8_b", True)}
    with pytest.raises(ProgramError, match="is plain and tensor w is stored"):
        placements_of(compile_class(Plain, {}, turned), turned)
    (line,) = placements_of(compile_class(Plain, {}, plain), plain)[1:]
    assert line.endswith("built plain, no marker")


def test_the_host_stores_each_tile_transposed():
    w = np.random.default_rng(2).standard_normal((64, 96)).astype(np.float32)
    raw, shape = formats.to_bytes(w, "bfloat16", "tile", turned=True)
    tiles = formats.bf16_values(raw.view(np.uint16)).reshape(-1, 1024)
    want = formats.tilize(formats.bf16_values(formats.bf16_bits(w[32:64, 0:32].T)))
    assert np.array_equal(tiles[3], want[0])
    back = formats.from_bytes(raw, shape, "bfloat16", "tile", turned=True)
    assert np.array_equal(back, formats.bf16_values(formats.bf16_bits(w)))
