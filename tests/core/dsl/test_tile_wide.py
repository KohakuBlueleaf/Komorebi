"""A row of more than 2 tiles in chunks of 2 (no card): the parts of the sums'
passes, the per-row values through a page of their own, each chunk's call with the
fields' chunk, an odd row's last tile a chunk of one; too many fields and a call not
pushed refused by name."""

import pytest
from test_class_lines import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import Refused, compile_class
from komorebi.dsl.program import compute_source
from komorebi.dsl.tile import trace
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam
from komorebi.dsl.tile.rows import chunk_program

W = 8  # tiles a row


def rows_ir(fn, residents):
    x = TensorParam("x", False, F32)
    held = [TensorParam(f"g{k}", False, F32) for k in range(residents)]
    ir = KernelIR("f", [x, *held, TensorParam("y", True, F32)], [], {})
    with trace.Tracer(ir):
        xs, ys = trace.TensorProxy(x), trace.TensorProxy(ir.tensors[-1])
        rows = [trace.resident(trace.TensorProxy(g).tiles[0, :]) for g in held]
        for t in trace.grid(trace.num_tile_rows(xs)):
            trace.store(ys.tiles[t, :], fn(trace.load(xs.tiles[t, :]), *rows))
    return ir


def layer(row, w, b):
    mean = trace.reduce_sum(row, exact=True) * (1.0 / (32 * W))
    d = row - mean.col()
    var = trace.reduce_sum(d * d, exact=True) * (1.0 / (32 * W))
    return d * trace.rsqrt(var + 1e-5).col() * w.row() + b.row()


def test_layer_norm_takes_a_part_a_pass():
    unit = chunk_program(rows_ir(layer, 2), W)
    assert list(unit.parts) == ["start", "sum0", "finish0", "sum1", "finish1"]
    (done,) = [p.done for p in unit.passes if p.name == "chunks"]
    assert done[0].startswith("a row of 8 tiles in 4 chunks of 2; 2 sum(s) in 2")


def test_an_odd_row_ends_in_a_chunk_of_one():
    unit = chunk_program(rows_ir(layer, 2), 7)
    assert list(unit.parts) == [
        *("start", "sum0", "sum0_one", "finish0", "sum1", "sum1_one", "finish1"),
        "apply_one",
    ]


def test_five_fields_are_refused():
    def five(row, *g):
        r = trace.rsqrt(trace.reduce_sum(row * row, exact=True)).col()
        return row * r * g[0].row() + g[1].row() + g[2].row() + g[3].row() + g[4].row()

    with pytest.raises(PlanError, match="5 resident rows; a wide row's call holds 4"):
        chunk_program(rows_ir(five, 5), W)


@km.tile_function
def rms(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / (32 * W))
    return row * km.rsqrt(ms + 1e-6).col() * weight.row()


class Wide(km.Program):
    weight: km.L1[km.fp32, 4, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 4)))
        results = km.Stream(out.pages(0, 2 * W), write=True, block=W)
        for row in km.Stream(x.pages(0, 2 * W), block=W, row=W):
            results.push(rms(row, self.weight))


class Named(Wide):
    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 4)))
        results = km.Stream(out.pages(0, 2 * W), write=True, block=W)
        for row in km.Stream(x.pages(0, 2 * W), block=W, row=W):
            y = rms(row, self.weight)
            results.push(y)


def built(cls):
    tensors = {
        n: Tensor(Pages(4096), (2 * 32, 32 * W), "float32", "tile")
        for n in ("x", "out")
    }
    tensors["w"] = Tensor(Pages(4096), (4 * 32, 32), "float32", "tile")
    return compile_class(cls, {}, tensors)


def test_the_values_go_through_their_page_and_each_chunk_is_a_call():
    case = compute_source(built(Wide))
    case = case[case.index("case kHandlerRows") :]
    assert case.count("MATH((km_tile::rms_sum0(0)));") == 1
    assert "pack_tile(2, kChunkRowsRms);" in case
    assert "copy_tile(kChunkRowsRms, 0, 2);" in case
    assert "copy_tile(kFieldWeight, km_chunk0, 4);" in case
    assert "pack_tile<true>(1, km_out0, km_tile0 + 2 * km_chunk0 + 1);" in case


def test_a_wide_row_call_not_pushed_is_refused():
    with pytest.raises(Refused, match="writes the row in chunks, pushed as it goes"):
        built(Named)


@km.tile_function
def rms7(row, weight):
    ms = km.reduce_sum(row * row, exact=True) * (1.0 / (32 * 7))
    return row * km.rsqrt(ms + 1e-6).col() * weight.row()


class Odd(km.Program):
    weight: km.L1[km.fp32, 4, 32, 32]

    @km.handler
    def rows(self, x: km.Tensor, w: km.Tensor, out: km.Tensor):
        self.weight.fill(km.Stream(w.pages(0, 4)))
        results = km.Stream(out.pages(0, 14), write=True, block=7)
        for row in km.Stream(x.pages(0, 14), block=7, row=7):
            results.push(rms7(row, self.weight))


def test_an_odd_row_s_last_tile_is_a_call_of_its_own():
    tensors = {
        n: Tensor(Pages(4096), (64, 224), "float32", "tile") for n in ("x", "out")
    }
    tensors["w"] = Tensor(Pages(4096), (128, 32), "float32", "tile")
    case = compute_source(compile_class(Odd, {}, tensors))
    case = case[case.index("case kHandlerRows") :]
    assert "copy_tile(km_half0, km_tile0 + 6, 0);\nMATH" in case.replace(" " * 24, "")
    assert "copy_tile(kFieldWeight, 3, 4);" in case
    assert "MATH((km_tile::rms7_apply_one(0)));" in case
    assert "pack_tile<true>(0, km_out0, km_tile0 + 6);" in case
