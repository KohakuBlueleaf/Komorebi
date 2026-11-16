"""A row longer than its stream's block, and a per-row operand in the stream form (no
card): the range read once a pass, each sum pass's values and sums through a page
pair of the row's, the writes in the last pass alone; a field read by `.col()` unpacked
with column 0 broadcast, a stream tile's `.row()` from its block's half; what the
passes do not take refused by name."""

import pytest
from test_class_lines import Pages

import komorebi as km
from komorebi.dsl.compiler import Refused, compile_class, known_count
from komorebi.dsl.program import compute_source, movement_source
from komorebi.tensor import Tensor

W = 32  # tiles a row


@km.tile_function
def layer(row, weight, bias):
    mean = km.reduce_sum(row, exact=True) * (1.0 / (32 * W))
    d = row - mean.col()
    var = km.reduce_sum(d * d, exact=True) * (1.0 / (32 * W))
    return d * km.rsqrt(var + 1e-5).col() * weight.row() + bias.row()


@km.tile_function
def apply(x, w, s):
    return x * km.rsqrt(s.col() * (1.0 / 2048) + 1e-6) * w.row()


class Long(km.Program):
    weight: km.L1[km.fp32, 16, 32, 32]
    bias: km.L1[km.fp32, 16, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 16)))
        self.bias.fill(km.Stream(b.pages(0, 16)))
        results = km.Stream(out.pages(0, 2 * W), write=True, block=8)
        for row in km.Stream(x.pages(0, 2 * W), block=8, row=W):
            results.push(layer(row, self.weight, self.bias))


class Plain(km.Program):
    @km.handler
    def rows(self, x: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 2 * W), write=True, block=8)
        for t in km.Stream(x.pages(0, 2 * W), block=8, row=W):
            results.push(t)


class Counted(km.Program):
    weight: km.L1[km.fp32, 16, 32, 32]
    bias: km.L1[km.fp32, 16, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor, rows: int):
        self.weight.fill(km.Stream(w.pages(0, 16)))
        self.bias.fill(km.Stream(b.pages(0, 16)))
        results = km.Stream(out.pages(0, rows * W), write=True, block=8)
        for row in km.Stream(x.pages(0, rows * W), block=8, row=W):
            results.push(layer(row, self.weight, self.bias))


class Halves(km.Program):
    weight: km.L1[km.bf16, 16, 32, 32]
    bias: km.L1[km.fp32, 16, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, b: km.Tensor, out: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 16)))
        self.bias.fill(km.Stream(b.pages(0, 16)))
        results = km.Stream(out.pages(0, 2 * W), write=True, block=8)
        for row in km.Stream(x.pages(0, 2 * W), block=8, row=W):
            results.push(layer(row, self.weight, self.bias))


class Split(km.Program):
    s: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, s: km.Tensor, out: km.Tensor):
        self.s.fill(km.Stream(s.pages(0, 1)))
        results = km.Stream(out.pages(0, 4), write=True, block=2)
        for xt, wt in zip(
            km.Stream(x.pages(0, 4), block=2), km.Stream(w.pages(0, 4), block=2)
        ):
            results.push(apply(xt, wt, self.s))


def built(cls):
    tensors = {n: Tensor(Pages(4096), (64, 32 * W), "float32", "tile") for n in "xo"}
    tensors["out"] = tensors.pop("o")
    for n in "wb":
        tensors[n] = Tensor(Pages(4096), (16 * 32, 32), "float32", "tile")
    if cls is Halves:
        tensors["w"] = Tensor(Pages(2048), (16 * 32, 32), "bfloat16", "tile")
    if cls is Split:
        tensors = {n: Tensor(Pages(4096), (32, 128), "float32", "tile") for n in "xw"}
        tensors["s"] = Tensor(Pages(4096), (32, 32), "float32", "tile")
        tensors["out"] = Tensor(Pages(4096), (32, 128), "float32", "tile")
    if cls is Plain:
        tensors = {n: tensors[n] for n in ("x", "out")}
    return compile_class(cls, {}, tensors)


def test_the_row_is_read_once_a_pass_and_written_in_the_last():
    compiled = built(Long)
    case = compute_source(compiled)
    case = case[case.index("case kHandlerRows") :]
    assert "km_block0 < km_blocks0 * 3;" in case  # two sum levels, then the writes
    assert "const uint32_t km_part0 = km_block0 % km_blocks0 % 4;" in case
    assert "km_block0 / km_blocks0 == 2 && compute.reserve_back(" in case
    assert "km::block_half(kWrite0, kWrite0Pattern, km_block0 % km_blocks0)" in case
    assert "MATH((km_tile::layer_sum1(0)));" in case
    assert "copy_tile(kChunkRowsLayer, 1, 3);" in case  # the sums, back for pass 2
    assert "copy_tile(kFieldBias, km_part0 * 4 + km_chunk0, 5);" in case
    pages = {b.name: b.pages for b in compiled.buffers}
    assert pages["kChunkRowsLayer"] == (4,)  # a pair a row, 2 rows
    assert ", 3, kRead0, kRead0Pattern" in movement_source(compiled)


def test_what_the_passes_do_not_take_is_refused():
    with pytest.raises(Refused, match="a row read in parts is a row function's"):
        built(Plain)
    with pytest.raises(Refused, match="its rows a constant count"):
        built(Counted)
    with pytest.raises(Refused, match=r"self.weight \(bfloat16\) is a wide row's"):
        built(Halves)
    assert known_count("(2 * 32)") == 64 and known_count("rows * 32") is None


def test_a_field_by_col_is_broadcast_and_a_stream_row_reads_its_half():
    case = compute_source(built(Split))
    case = case[case.index("case kHandlerApply") :]
    assert "unary_bcast<BroadcastType::COL>(kFieldS, 0, " in case
    assert "unary_bcast<BroadcastType::ROW>(km_half1, " in case
