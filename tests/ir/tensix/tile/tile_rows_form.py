"""Today's trace of tile functions called on a row of n tiles (the row form), the
port's reference (in the IR's text in tile_rows_form.kom). Writes each KernelIR
(`<case>.ir`), the cases with their widths and fields (`cases.txt`), and the refusal
of a reduction on one tile (`one_tile.refused`). Usage: <this> <folder>."""

import inspect
import sys
from pathlib import Path

import komorebi as km
from komorebi.dsl.tile import trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam

BF, F = "bfloat16", "float32"


def rms_norm(x, w):
    """A row's root mean square at 2 tiles, times a resident weight row."""
    ms = km.reduce_sum(x * x, exact=True) * 0.015625
    return x * km.rsqrt(ms + 1e-6).col() * w.row()


def rms_norm_wide(x, w):
    """The same at 4 tiles."""
    ms = km.reduce_sum(x * x, exact=True) * 0.0078125
    return x * km.rsqrt(ms + 1e-6).col() * w.row()


def rms_bf16(x, w):
    """At 3 tiles, the weight row in bf16."""
    ms = km.reduce_sum(x * x, exact=True) * 0.010416666666666666
    return x * km.rsqrt(ms + 1e-6).col() * w.row()


def softmax_rows(x):
    """A row's max, then the sum of the exponentials: two levels."""
    top = km.reduce_max(x, exact=True)
    e = km.exp(x - top.col())
    return e * km.reciprocal(km.reduce_sum(e, exact=True)).col()


def centred(x):
    """A row less its mean."""
    return x - (km.reduce_sum(x, exact=True) * 0.015625).col()


def row_max(x):
    """A row less its largest value."""
    return x - km.reduce_max(x, exact=True).col()


def row_min(x):
    """A row less its smallest value."""
    return x - km.reduce_min(x, exact=True).col()


def layer_norm(x, w, b):
    """A row's mean and variance, two resident rows."""
    d = x - (km.reduce_sum(x, exact=True) * 0.0078125).col()
    variance = km.reduce_sum(d * d, exact=True) * 0.0078125
    return d * km.rsqrt(variance + 1e-5).col() * w.row() + b.row()


def gated(x, g, w):
    """Two streamed rows and a resident one."""
    y = x * g
    return y * km.rsqrt(km.reduce_sum(y * y, exact=True) + 1e-6).col() * w.row()


# case, function, its tensors' formats, the row's width, its fields (the last ones)
CASES = [
    ("rms_norm", rms_norm, [F, F], 2, 1),
    ("rms_norm_wide", rms_norm_wide, [F, F], 4, 1),
    ("rms_bf16", rms_bf16, [F, BF], 3, 1),
    ("softmax_rows", softmax_rows, [F], 3, 0),
    ("centred", centred, [F], 2, 0),
    ("row_max", row_max, [F], 4, 0),
    ("row_min", row_min, [F], 2, 0),
    ("layer_norm", layer_norm, [F, F, F], 4, 2),
    ("gated", gated, [F, F, F], 2, 1),
]


def traced(name, function, formats, fields):
    """The KernelIR of `function` as today's call site traces it on a row of tiles
    (compile/tiles/functions.py, tile_unit, at a width above 1)."""
    names = list(inspect.signature(function).parameters)
    tensors = [TensorParam(p, False, f) for p, f in zip(names, formats, strict=True)]
    result = TensorParam("result", True, F32)
    ir = KernelIR(name, [*tensors, result], [], {})
    held = len(names) - fields  # the fields are the last parameters
    with trace.Tracer(ir):
        proxies = [trace.TensorProxy(t) for t in tensors]
        out = trace.TensorProxy(result)
        rows = [trace.resident(p.tiles[0, :]) for p in proxies[held:]]
        for t in trace.grid(trace.num_tile_rows(proxies[0])):
            loads = [trace.load(p.tiles[t, :]) for p in proxies[:held]]
            trace.store(out.tiles[t, :], function(*loads, *rows))
    return ir


def one_tile():
    """Today's refusal of a reduction called on one tile (tile_unit at width 1)."""
    x, result = TensorParam("x", False, F), TensorParam("result", True, F32)
    ir = KernelIR("one", [x, result], [], {})
    with trace.Tracer(ir):
        xs, out = trace.TensorProxy(x), trace.TensorProxy(result)
        for t in trace.grid(trace.num_tiles(xs)):
            a = trace.load(xs.tiles[t])
            trace.store(out.tiles[t], a - km.reduce_sum(a, exact=True).col())
    try:
        unit_program(ir, "fp32", load_macro=True)
    except PlanError as error:
        return str(error)
    raise AssertionError("a reduction on one tile was taken")


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cases = []
    for case, function, formats, width, fields in CASES:
        cases.append(f"{case}\t{width}\t{fields}")
        ir = traced(case, function, formats, fields)
        (folder / f"{case}.ir").write_text(ir.dump() + "\n")
    (folder / "cases.txt").write_text("".join(f"{c}\n" for c in cases))
    (folder / "one_tile.refused").write_text(one_tile() + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
