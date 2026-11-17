"""A per-column operand in a call on a stream's tiles (no card): a field of one plain
row read by `.row()`, unpacked with row 0 broadcast to every row into its DST tile;
a field read whole in that form refused by name."""

import pytest
from test_class_lines import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source
from komorebi.dsl.tile import trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam


def test_a_row_operand_is_read_as_a_whole_tile():
    x, b, y = (TensorParam(n, n == "y", F32) for n in "xby")
    ir = KernelIR("f", [x, b, y], [], {})
    with trace.Tracer(ir):
        xs, bs, ys = (trace.TensorProxy(t) for t in (x, b, y))
        for t in trace.grid(trace.num_tiles(xs)):
            value = trace.load(xs.tiles[t]) + trace.load(bs.tiles[t]).row()
            trace.store(ys.tiles[t], value)
    unit = unit_program(ir)
    assert unit.operands == ["x", "b"] and unit.row_operands == [1]


@km.tile_function
def affine(x, scale, shift):
    return x * scale.row() + shift.row()


@km.tile_function
def whole(x, scale):
    return x * scale


class Affine(km.Program):
    scale: km.L1[km.fp32, 1, 32, 32]
    shift: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def line(self, a: km.Tensor, s: km.Tensor, b: km.Tensor, out: km.Tensor):
        self.scale.fill(km.Stream(s.pages(0, 1)))
        self.shift.fill(km.Stream(b.pages(0, 1)))
        results = km.Stream(out.pages(0, 16), write=True, block=8)
        for x in km.Stream(a.pages(0, 16), block=8):
            results.push(affine(x, self.scale, self.shift))


class Whole(Affine):
    @km.handler
    def line(self, a: km.Tensor, s: km.Tensor, b: km.Tensor, out: km.Tensor):
        self.scale.fill(km.Stream(s.pages(0, 1)))
        results = km.Stream(out.pages(0, 16), write=True, block=8)
        for x in km.Stream(a.pages(0, 16), block=8):
            results.push(whole(x, self.scale))


def built(cls):
    tensors = {
        n: Tensor(Pages(4096), (16 * 32, 32), "float32", "tile")
        for n in ("a", "s", "b", "out")
    }
    return compile_class(cls, {}, tensors)


def test_each_field_is_unpacked_with_row_0_broadcast():
    source = compute_source(built(Affine))
    assert '#include "api/compute/bcast.h"' in source
    case = source[source.index("case kHandlerLine") :]
    assert "unary_bcast<BroadcastType::ROW>(kFieldScale, 0, 1);" in case
    assert "unary_bcast<BroadcastType::ROW>(kFieldShift, 0, 2);" in case
    assert case.count("copy_init(kRead0.mover_a);") == 3  # the setup's, then each's


def test_a_field_read_whole_is_refused():
    with pytest.raises(Refused, match=r"self.scale .* read by `scale.row\(\)` alone"):
        built(Whole)
