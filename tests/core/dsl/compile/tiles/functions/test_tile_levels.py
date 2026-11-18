"""Several sums and several results in a tile function (no card): the layer norm's two
sums in turn over the row, a weight and a bias row; a state and an output stored into
Dst tiles 0 and 1 and pushed into two write streams under one commit; past what Dst
and the rows tile hold, refused naming both counts."""

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.program import compute_source
from komorebi.dsl.tile import trace, unit_program
from komorebi.dsl.tile.forms.rows import row_program
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam

W = 2  # tiles a row


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


def results_ir(fn, results):
    x, h = TensorParam("x", False, F32), TensorParam("h", False, F32)
    outs = [TensorParam(f"y{k}", True, F32) for k in range(results)]
    ir = KernelIR("f", [x, h, *outs], [], {})
    with trace.Tracer(ir):
        xs, hs = trace.TensorProxy(x), trace.TensorProxy(h)
        for t in trace.grid(trace.num_tiles(xs)):
            values = fn(trace.load(xs.tiles[t]), trace.load(hs.tiles[t]))
            for out, value in zip(outs, values):
                trace.store(trace.TensorProxy(out).tiles[t], value)
    return ir


def layer(row, w, b):
    mean = trace.reduce_sum(row, exact=True) * (1.0 / (32 * W))
    d = row - mean.col()
    var = trace.reduce_sum(d * d, exact=True) * (1.0 / (32 * W))
    return d * trace.rsqrt(var + 1e-5).col() * w.row() + b.row()


def test_layer_norm_takes_two_passes_and_two_rows():
    unit = row_program(rows_ir(layer, 2), W, W, 2 * W)
    (lanes,) = [p.done for p in unit.passes if p.name == "lanes"]
    assert lanes == [
        "2 sum(s) in 2 pass(es) over the row; 2 per-row value(s) in slots of tile 2; "
        "2 per-column value(s) in tile(s) from 4"
    ]


def test_five_per_row_values_are_refused():
    def five(row):
        m = trace.reduce_sum(row, exact=True)
        for _ in range(4):
            m = trace.reduce_sum(row * m.col(), exact=True)
        return row * m.col()

    with pytest.raises(PlanError, match="5 per-row values .* the rows tile holds 4"):
        row_program(rows_ir(five, 0), W, W, 2 * W)


def test_five_resident_rows_are_refused():
    def five(row, *g):
        r = trace.rsqrt(trace.reduce_sum(row * row, exact=True)).col()
        return row * r * g[0].row() + g[1].row() + g[2].row() + g[3].row() + g[4].row()

    with pytest.raises(PlanError, match="5 resident rows; Dst's 8 fp32 tiles hold 4"):
        row_program(rows_ir(five, 5), W, W, 2 * W)


def test_two_results_are_stored_into_dst_0_and_1():
    unit = unit_program(results_ir(lambda x, h: (0.9 * h + x, x * h), 2))
    assert unit.results == 2
    packs = [c.text for c in unit.tile["pack"] if "pack_tile" in c.text]
    assert packs == ["pack_tile(dst + 0, out0)", "pack_tile(dst + 1, out1)"]


def test_five_results_are_refused():
    with pytest.raises(PlanError, match="5 results; a call holds 4 Dst tiles"):
        unit_program(results_ir(lambda x, h: (x, h, x + h, x * h, x - h), 5))


@km.tile_function
def decay(x, h):
    h = 0.9 * h + x
    return h, km.silu(h)


class Decay(km.Program):
    @km.handler
    def line(self, a: km.Tensor, h: km.Tensor, out: km.Tensor, state: km.Tensor):
        results = km.Stream(out.pages(0, 16), write=True, block=8)
        states = km.Stream(state.pages(0, 16), write=True, block=8)
        for x, s in zip(
            km.Stream(a.pages(0, 16), block=8), km.Stream(h.pages(0, 16), block=8)
        ):
            s2, y = decay(x, s)
            states.push(s2)
            results.push(y)


class Unpushed(Decay):
    @km.handler
    def line(self, a: km.Tensor, h: km.Tensor, out: km.Tensor, state: km.Tensor):
        results = km.Stream(out.pages(0, 16), write=True, block=8)
        for x, s in zip(
            km.Stream(a.pages(0, 16), block=8), km.Stream(h.pages(0, 16), block=8)
        ):
            s2, y = decay(x, s)
            results.push(y)


class Pushed(Decay):
    @km.handler
    def line(self, a: km.Tensor, h: km.Tensor, out: km.Tensor, state: km.Tensor):
        results = km.Stream(out.pages(0, 16), write=True, block=8)
        for x, s in zip(
            km.Stream(a.pages(0, 16), block=8), km.Stream(h.pages(0, 16), block=8)
        ):
            results.push(decay(x, s))


def built(cls):
    tensors = {
        n: Tensor(Pages(4096), (16 * 32, 32), "float32", "tile")
        for n in ("a", "h", "out", "state")
    }
    return compile_class(cls, {}, tensors)


def test_two_results_are_pushed_under_one_commit():
    case = compute_source(built(Decay))
    case = case[case.index("case kHandlerLine") :]
    # one commit for two results: a commit each gives the same values
    assert case.count("tile_regs_commit();") == 1


@pytest.mark.parametrize(
    ("cls", "message"),
    [
        (Unpushed, r"the results s2, y of the call at line \d+ are pushed in its loop"),
        (Pushed, "decay returns 2 tile\\(s\\), and the line takes 1"),
    ],
)
def test_results_not_each_named_and_pushed_are_refused(cls, message):
    with pytest.raises(Refused, match=message):
        built(cls)
