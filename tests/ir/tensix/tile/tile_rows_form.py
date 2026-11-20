"""Today's trace and row form of tile functions on a row of n tiles, the port's
reference (in the IR's text in tile_rows_form.kom). Writes each KernelIR, the cases,
each 2-tile unit's listing, header and emitted listing (`<case>.rows.*`) or refusal,
and the refusal of a reduction on one tile. Usage: <this> <folder>."""

import inspect
import sys
from pathlib import Path

from tile_rows import ENCC, add, gt, line, load, mov, mul, store, w

import komorebi as km
from komorebi.dsl.tile import emit, trace, unit_program
from komorebi.dsl.tile.forms.rows import (
    Layout,
    _check_broadcast,
    _check_lanes,
    row_program,
)
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam
from komorebi.dsl.tile.unit import UnitError, UnitProgram

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


def rms_plain(x):
    """A row's root mean square alone, at 2 tiles."""
    ms = km.reduce_sum(x * x, exact=True) * 0.015625
    return x * km.rsqrt(ms + 1e-6).col()


def peak(x):
    """A row less its largest value, at 2 tiles."""
    return x - km.reduce_max(x, exact=True).col()


def weighted_bf16(x, w):
    """rms_norm at 2 tiles, the weight row in bf16."""
    ms = km.reduce_sum(x * x, exact=True) * 0.015625
    return x * km.rsqrt(ms + 1e-6).col() * w.row()


def softmax_pair(x):
    """softmax_rows at 2 tiles: two reductions, the levels."""
    top = km.reduce_max(x, exact=True)
    e = km.exp(x - top.col())
    return e * km.reciprocal(km.reduce_sum(e, exact=True)).col()


def weighted_sum(x, w):
    """A sum that reads the resident row by .row(): the levels."""
    return x * km.reduce_sum(x * w.row(), exact=True).col()


def peak_square(x):
    """A row's largest square: a product summed by max, not by a multiply-add."""
    return x - km.reduce_max(x * x, exact=True).col()


def exp_sum(x):
    """A sum of exponentials over the row: the mad-reader check repairs the code."""
    return km.reduce_sum(km.exp(x), exact=True).col() * x


def exp_scaled(x):
    """More live values than the work registers: refused by the layout."""
    return km.exp(x) * km.reduce_sum(x, exact=True).col()


def recip_centred(x):
    """A row less the reciprocal of its sum of squares."""
    return x - km.reciprocal(km.reduce_sum(x * x, exact=True)).col()


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
    ("rms_plain", rms_plain, [F], 2, 0),
    ("peak", peak, [F], 2, 0),
    ("weighted_bf16", weighted_bf16, [F, BF], 2, 1),
    ("softmax_pair", softmax_pair, [F], 2, 0),
    ("weighted_sum", weighted_sum, [F, F], 2, 1),
    ("peak_square", peak_square, [F], 2, 0),
    ("exp_sum", exp_sum, [F], 2, 0),
    ("exp_scaled", exp_scaled, [F], 2, 0),
    ("recip_centred", recip_centred, [F], 2, 0),
]


# the row form's branches the corpus does not reach, on functions the IR does not
# write or calls it does not make: case, function, formats, fields, width, Dst
BRANCHES = [
    ("plus_field", lambda x, w: x * km.reduce_sum(x, exact=True).col() + w, [F, F], 1),
    ("row_of_load", lambda x: x * km.reduce_sum(x, exact=True).col() * x.row(), [F], 0),
    ("no_col", lambda x: x + km.reduce_sum(x, exact=True), [F], 0),
    ("row_col", lambda x: x * x.col(), [F], 0),
    (
        "field_col",
        lambda x, w: x * km.reduce_sum(x, exact=True).col() * w.col(),
        [F, F],
        1,
    ),
    ("no_reduce", lambda x: x * 2.0, [F], 0),
    ("not_exact", lambda x: x * km.reduce_sum(x).col(), [F], 0),
    ("fp16_row", lambda x: x * km.reduce_sum(x, exact=True).col(), ["float16"], 0),
    (
        "two_fields",
        lambda x, w, b: x * km.reduce_sum(x, exact=True).col() * w.row() + b.row(),
        [F, F, F],
        2,
    ),
    (
        "two_cols",
        lambda x: x * km.reduce_sum(x, exact=True).col()
        + km.reduce_sum(x, exact=True).col(),
        [F],
        0,
    ),
]
# functions of a row and a tile of it (a tile load the row form's trace does not make)
TILED = [
    ("tile_with_row", lambda row, tile: row * tile),
    ("tile_with_col", lambda row, tile: tile + km.reduce_sum(row, exact=True)),
]
# calls of row_program the corpus does not make: case, the function's case, width, Dst
CALLS = [("width_3", "centred", 3, "fp32"), ("fp16a_dest", "centred", 2, "fp16a")]


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


def tiled(name, function):
    """The KernelIR of `function` of a row of x and its first tile."""
    x, result = TensorParam("x", False, F), TensorParam("result", True, F32)
    ir = KernelIR(name, [x, result], [], {})
    with trace.Tracer(ir):
        xs, out = trace.TensorProxy(x), trace.TensorProxy(result)
        for t in trace.grid(trace.num_tile_rows(xs)):
            row, tile = trace.load(xs.tiles[t, :]), trace.load(xs.tiles[t])
            trace.store(out.tiles[t, :], function(row, tile))
    return ir


def shft(a, out):
    return w("SFPSHFT2", "turned", lreg_src_c=a, lreg_dest=out, instr_mod1=3)


def swap(a, b):
    return w("SFPSWAP", "max, min", lreg_src_c=a, lreg_dest=b, instr_mod1=1)


def summed():
    """A lane row's sum by rotations of 1, 2 and 4 lanes (v1 loaded, v11 the sum)."""
    words = [load("v1", 0), shft("v1", "v2"), add("v1", "v2", "v3")]
    words += [shft("v3", "v4"), shft("v4", "v5"), add("v3", "v5", "v6")]
    turned = ["v6", "v7", "v8", "v9", "v10"]
    words += [shft(a, b) for a, b in zip(turned, turned[1:], strict=False)]
    return [*words, add("v6", "v10", "v11"), store("v11", 128)]


# the row form's checks on code of a few words: case, check, words (a row of 2 tiles,
# per-row slots in tile 2, per-column vectors in tile 4)
CHECKS = [
    ("lanes_partial", "lanes", [load("v1", 0), store("v1", 128)]),
    (
        "lanes_two",
        "lanes",
        [load("v1", 0), shft("v1", "v2"), add("v1", "v2", "v3"), store("v3", 128)],
    ),
    ("lanes_full", "lanes", summed()),
    (
        "lanes_copy",
        "lanes",
        [load("v1", 0), mov("v1", "v2"), shft("v2", "v3"), add("v1", "v3", "v4")]
        + [store("v4", 128)],
    ),
    ("lanes_slot", "lanes", [load("v1", 128), store("v1", 136)]),
    (
        "lanes_swap",
        "lanes",
        [load("v1", 0), load("v2", 2), swap("v1", "v2"), store("v2", 128)],
    ),
    (
        "broadcast_ok",
        "broadcast",
        [load("v1", 0), load("v2", 128), load("v3", 256), mul("v1", "v2", "v4")]
        + [mul("v4", "v3", "v5"), store("v5", 0)],
    ),
    ("broadcast_other", "broadcast", [load("v1", 64), store("v1", 0)]),
    ("broadcast_alone", "broadcast", [load("v1", 128), store("v1", 0)]),
    (
        "broadcast_missing_own",
        "broadcast",
        [load("v1", 128), load("v2", 256), mul("v1", "v2", "v3"), store("v3", 0)],
    ),
    (
        "broadcast_flags",
        "broadcast",
        [load("v1", 0), load("v2", 64), gt("v1", "v2"), mov("v2", "v1"), ENCC]
        + [store("v1", 0)],
    ),
    ("broadcast_outside", "broadcast", [load("v1", 0), store("v1", 128)]),
]


def checks(folder):
    """Today's lanes and broadcast checks on CHECKS: each case's words in
    `check_<case>.in`, its verdict (or refusal) a line of the result."""
    layouts = {
        "copy": Layout("tile", 0),
        "bcast_col": Layout("rows", 2),
        "bcast_row": Layout("cols", 4),
    }
    lines = []
    for case, check, words in CHECKS:
        (folder / f"check_{case}.in").write_text("".join(f"{line(x)}\n" for x in words))
        unit = UnitProgram("c", "fp32")
        try:
            if check == "lanes":
                _check_lanes(unit, words, layouts["bcast_col"])
            else:
                _check_broadcast(unit, words, layouts, 2)
            verdict = unit.checks[-1].verdict
        except UnitError as error:
            verdict = str(error)
        lines.append(f"{case}\t{check}\t{verdict}")
    return lines


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
    cases, refused = [], []
    for case, function, formats, width, fields in CASES:
        cases.append(f"{case}\t{width}\t{fields}")
        ir = traced(case, function, formats, fields)
        (folder / f"{case}.ir").write_text(ir.dump() + "\n")
        if width != 2:
            continue
        try:  # as tile_unit calls it on a row of 2 tiles
            unit = row_program(ir, width, width, 2 * width)
            text = unit.dump()
            words = emit(unit)
        except (PlanError, UnitError) as error:
            refused.append(f"{case}\t{error}")
            continue
        (folder / f"{case}.rows.txt").write_text(text + "\n")
        (folder / f"{case}.rows.hpp").write_text(words)
        (folder / f"{case}.rows.emitted.txt").write_text(unit.dump() + "\n")
    (folder / "cases.txt").write_text("".join(f"{c}\n" for c in cases))
    (folder / "one_tile.refused").write_text(one_tile() + "\n")
    (folder / "rows_refused.txt").write_text("".join(f"{r}\n" for r in refused))
    (folder / "branches.txt").write_text("".join(f"{b}\n" for b in branches(folder)))
    (folder / "checks.txt").write_text("".join(f"{c}\n" for c in checks(folder)))


def branches(folder):
    """Today's row_program on BRANCHES and CALLS: each case's width, Dst and verdict
    (`made`, or its refusal's text), its KernelIR in `branch_<case>.ir`."""
    made = [(c, traced(c, fn, fm, fd), 2, "fp32") for c, fn, fm, fd in BRANCHES]
    made += [(c, tiled(c, fn), 2, "fp32") for c, fn in TILED]
    corpus = {c: (fn, fm, fd) for c, fn, fm, _, fd in CASES}
    made += [(c, traced(of, *corpus[of]), w, dest) for c, of, w, dest in CALLS]
    lines = []
    for case, ir, width, dest in made:
        (folder / f"branch_{case}.ir").write_text(ir.dump() + "\n")
        try:
            row_program(ir, width, width, 2 * width, dest=dest)
            verdict = "made"
        except (PlanError, UnitError) as error:
            verdict = str(error)
        lines.append(f"{case}\t{width}\t{dest}\t{verdict}")
    return lines


if __name__ == "__main__":
    main(sys.argv[1])
