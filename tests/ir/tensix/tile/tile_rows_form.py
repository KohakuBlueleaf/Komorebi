"""Today's trace and row form of tile functions on a row of n tiles, the port's
reference (in tile_rows_form.kom). Writes each KernelIR, the cases, each unit's
listing, header and emitted listing (`<case>.rows.*` at 2 tiles, else `.chunk.*` and
its cost), its band at 2 tiles (`.band.*`) or the refusals, and a few words' runs."""

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
    band_groups,
    chunk_cost,
    chunk_program,
    row_program,
)
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam
from komorebi.dsl.tile.unit import Insn, UnitError, UnitProgram

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


def layer_pair(x, w, b):
    """layer_norm at 2 tiles: two levels, two resident rows."""
    d = x - (km.reduce_sum(x, exact=True) * 0.015625).col()
    variance = km.reduce_sum(d * d, exact=True) * 0.015625
    return d * km.rsqrt(variance + 1e-5).col() * w.row() + b.row()


def kept_sum(x):
    """A sum a later level's per-row value reads: kept in a slot of its own."""
    a = km.reduce_sum(x, exact=True)
    m = km.reduce_max(x - (a * 0.015625).col(), exact=True)
    return x * (m + a).col()


def max_weighted(x, w):
    """A largest value that reads the resident row by .row(): the levels."""
    return x - km.reduce_max(x * w.row(), exact=True).col()


def three_levels(x):
    """softmax_pair, then its root mean square: three passes over the row."""
    top = km.reduce_max(x, exact=True)
    e = km.exp(x - top.col())
    y = e * km.reciprocal(km.reduce_sum(e, exact=True)).col()
    return y * km.rsqrt(km.reduce_sum(y * y, exact=True) + 1e-6).col()


def three_plain(x):
    """Three passes over the row, each reading the value before it, no constant."""
    a = km.reduce_max(x, exact=True)
    b = km.reduce_sum(x - a.col(), exact=True)
    return x - km.reduce_min(x * b.col(), exact=True).col()


def kept_through(x):
    """A sum a later reduction reads through a value made of it: kept in a slot."""
    a = km.reduce_max(x, exact=True)
    p = km.reduce_sum(x - (a * 2.0).col(), exact=True)
    return x * p.col()


def two_weights(x, w, v):
    """A sum that reads two resident rows by .row(): a column of each."""
    return x - km.reduce_sum(x * w.row() * v.row(), exact=True).col()


def rms_eight(x, w):
    """rms_norm over 8 tiles: four chunks."""
    ms = km.reduce_sum(x * x, exact=True) * 0.00390625
    return x * km.rsqrt(ms + 1e-6).col() * w.row()


def softmax_five(x):
    """softmax_rows over 5 tiles: an odd row's last chunk of one tile."""
    top = km.reduce_max(x, exact=True)
    e = km.exp(x - top.col())
    return e * km.reciprocal(km.reduce_sum(e, exact=True)).col()


def extremes(x):
    """A largest and a smallest value over 4 tiles: two starts, in a set's order."""
    return x - km.reduce_max(x, exact=True).col() + km.reduce_min(x, exact=True).col()


def wide_weighted(x, w):
    """A sum that reads the resident row by .row() over 4 tiles."""
    return x * km.reduce_sum(x * w.row(), exact=True).col()


def wide_mixed(x, w):
    """Two sums over 4 tiles, one reading the resident row by .row(), one not."""
    a = km.reduce_sum(x * w.row(), exact=True)
    return x * a.col() - km.reduce_max(x, exact=True).col()


def five_values(x):
    """Five per-row values: more than the rows tile's slots."""
    a = x * km.reduce_sum(x, exact=True).col() + km.reduce_max(x, exact=True).col()
    b = a - km.reduce_min(x, exact=True).col()
    return (
        b * km.reduce_sum(x * x, exact=True).col()
        + km.reduce_max(x * x, exact=True).col()
    )


def five_fields(x, a, b, c, d, e):
    """Five resident rows: more per-column tiles than Dst holds after the row's."""
    y = x * km.reduce_sum(x, exact=True).col() * a.row() * b.row()
    return y * c.row() * d.row() * e.row()


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
    ("layer_pair", layer_pair, [F, F, F], 2, 2),
    ("kept_sum", kept_sum, [F], 2, 0),
    ("max_weighted", max_weighted, [F, F], 2, 1),
    ("three_levels", three_levels, [F], 2, 0),
    ("three_plain", three_plain, [F], 2, 0),
    ("kept_through", kept_through, [F], 2, 0),
    ("two_weights", two_weights, [F, F, F], 2, 2),
    ("rms_eight", rms_eight, [F, F], 8, 1),
    ("softmax_five", softmax_five, [F], 5, 0),
    ("extremes", extremes, [F], 4, 0),
    ("wide_weighted", wide_weighted, [F, F], 4, 1),
    ("wide_mixed", wide_mixed, [F, F], 4, 1),
    ("five_values", five_values, [F], 2, 0),
    ("five_fields", five_fields, [F] * 6, 2, 5),
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
        "field_levels",
        lambda x, w: x
        * km.reduce_sum(x, exact=True).col()
        * km.reduce_max(x, exact=True).col()
        + w,
        [F, F],
        1,
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
    cases, refused, costs, band_refused = [], [], [], []
    for case, function, formats, width, fields in CASES:
        cases.append(f"{case}\t{width}\t{fields}")
        ir = traced(case, function, formats, fields)
        (folder / f"{case}.ir").write_text(ir.dump() + "\n")
        if width == 2:
            band_refused += banded(folder, case, ir)
        form = "rows" if width == 2 else "chunk"
        try:  # as tile_unit calls it on a row of 2 tiles, or of more
            if width == 2:
                unit = row_program(ir, width, width, 2 * width)
            else:
                unit = chunk_program(ir, width)
                costs.append(
                    f"{case}\t" + "\t".join(map(str, chunk_cost(unit, width, fields)))
                )
            text = unit.dump()
            words = emit(unit)
        except (PlanError, UnitError) as error:
            refused.append(f"{case}\t{error}")
            continue
        (folder / f"{case}.{form}.txt").write_text(text + "\n")
        lines = dst_lines(width, formats, fields)
        (folder / f"{case}.dst").write_text("\n".join(lines) + "\n")
        (folder / f"{case}.{form}.hpp").write_text(words)
        (folder / f"{case}.{form}.emitted.txt").write_text(unit.dump() + "\n")
        if form == "chunk":
            (folder / f"{case}.chunk.fields").write_text(chunk_fields(unit))
    (folder / "cases.txt").write_text("".join(f"{c}\n" for c in cases))
    (folder / "one_tile.refused").write_text(one_tile() + "\n")
    (folder / "rows_refused.txt").write_text("".join(f"{r}\n" for r in refused))
    (folder / "branches.txt").write_text("".join(f"{b}\n" for b in branches(folder)))
    (folder / "checks.txt").write_text("".join(f"{c}\n" for c in checks(folder)))
    (folder / "band.txt").write_text(band() + "\n")
    (folder / "band_refused.txt").write_text("".join(f"{r}\n" for r in band_refused))
    (folder / "band_words.txt").write_text(
        "".join(f"{b}\n" for b in band_words(folder))
    )
    (folder / "chunk_costs.txt").write_text("".join(f"{c}\n" for c in costs))
    (folder / "chunk_branches.txt").write_text(
        "".join(f"{b}\n" for b in chunks(folder))
    )


def results(name, functions, formats, fields):
    """The KernelIR of `functions` of one row, each stored to a result of its own (the
    chunk form's several results, which the IR's trace does not make)."""
    names = list(inspect.signature(functions[0]).parameters)
    tensors = [TensorParam(p, False, f) for p, f in zip(names, formats, strict=True)]
    outs = [TensorParam(f"result{k}", True, F32) for k in range(len(functions))]
    ir = KernelIR(name, [*tensors, *outs], [], {})
    held = len(names) - fields
    with trace.Tracer(ir):
        proxies = [trace.TensorProxy(t) for t in tensors]
        stores = [trace.TensorProxy(o) for o in outs]
        rows = [trace.resident(p.tiles[0, :]) for p in proxies[held:]]
        for t in trace.grid(trace.num_tile_rows(proxies[0])):
            loads = [trace.load(p.tiles[t, :]) for p in proxies[:held]]
            for out, function in zip(stores, functions, strict=True):
                trace.store(out.tiles[t, :], function(*loads, *rows))
    return ir


def _sum(x):
    return km.reduce_sum(x, exact=True)


# chunk_program's branches the corpus does not reach: case, KernelIR, width
CHUNKS = [
    ("narrow", lambda: traced("narrow", centred, [F], 0), 2),
    (
        "two_results",
        lambda: results(
            "two_results",
            [lambda x, w: x * _sum(x).col() * w.row(), lambda x, w: x - _sum(x).col()],
            [F, F],
            1,
        ),
        4,
    ),
    (
        "three_results",
        lambda: results(
            "three_results",
            [lambda x, w, v: x * w.row() * _sum(x).col()] * 3,
            [F, F, F],
            2,
        ),
        4,
    ),
    (
        "five_sums",
        lambda: traced(
            "five_sums",
            lambda x: x
            * (
                _sum(x) + _sum(x * x) + _sum(x * 2.0) + _sum(x * 3.0) + _sum(x * 4.0)
            ).col(),
            [F],
            0,
        ),
        4,
    ),
    (
        "five_wide_fields",
        lambda: traced("five_wide_fields", five_fields, [F] * 6, 5),
        4,
    ),
    (
        "value_alone",
        lambda: traced("value_alone", lambda x: (_sum(x) * 0.25).col(), [F], 0),
        3,
    ),
]


def dst_lines(width, formats, fields):
    """Today's handler's Dst layout of a row form's call (compile/tiles/functions.py):
    the row in Dst 0 .. width - 1 and field k in 2 width + k, packed from the row's
    tiles; past 2 tiles, row_chunks: a chunk in Dst 0 and 1, field k in 4 + k."""
    tiles, first = (width, 2 * width) if width <= 2 else (2, 4)
    held = len(formats) - fields
    lines = [
        "param\t" + ",".join(map(str, range(tiles))) if k == 0 else "param\t"
        for k in range(held)
    ]
    lines += [f"param\t{first + k}" for k in range(fields)]
    return [*lines, "result\t" + ",".join(map(str, range(tiles)))]


def chunk_fields(unit):
    """What the handler reads of a chunk unit that its listing does not print."""
    return (
        f"results\t{unit.results}\n"
        f"chunk_results\t{','.join(map(str, unit.chunk_results))}\n"
        f"fields_in_sums\t{int(unit.chunk_fields_in_sums)}\n"
    )


def chunks(folder):
    """Today's chunk_program on CHUNKS: each case's width and verdict (`made`, or its
    refusal's text), its KernelIR in `chunk_<case>.ir` and its unit in
    `chunk_<case>.chunk.txt`."""
    lines = []
    for case, make, width in CHUNKS:
        ir = make()
        (folder / f"chunk_{case}.ir").write_text(ir.dump() + "\n")
        try:
            unit = chunk_program(ir, width)
            (folder / f"chunk_{case}.chunk.txt").write_text(unit.dump() + "\n")
            (folder / f"chunk_{case}.chunk.fields").write_text(chunk_fields(unit))
            verdict = "made"
        except (PlanError, UnitError) as error:
            verdict = str(error)
        lines.append(f"{case}\t{width}\t{verdict}")
    return lines


def band():
    """Today's refusal of the band (`rows`) in the levels, on softmax_pair."""
    ir = traced("softmax_pair", softmax_pair, [F], 0)
    try:
        row_program(ir, 2, 2, 4, rows=True)
    except PlanError as error:
        return str(error)
    raise AssertionError("the levels took the band")


def banded(folder, case, ir):
    """Today's band of the row form at 2 tiles (`row_program`'s `rows`): its listing,
    header, emitted listing and math cost at 0 .. 33 live rows; or its refusal (a list
    of one line)."""
    try:
        unit = row_program(ir, 2, 2, 4, rows=True)
        text = unit.dump()
        costs = [(r, *unit.cost("math", r)[:2]) for r in range(34)]
        words = emit(unit)
    except (PlanError, UnitError) as error:
        return [f"{case}\t{error}"]
    (folder / f"{case}.band.txt").write_text(text + "\n")
    (folder / f"{case}.band").write_text(
        "".join(f"{r}\t{n}\t{c!r}\n" for r, n, c in costs)
    )
    (folder / f"{case}.band.hpp").write_text(words)
    (folder / f"{case}.band.emitted.txt").write_text(unit.dump() + "\n")
    return []


# the band's runs on words the row form does not make (slots in tile 2, at an offset if
# given): Dst addresses 0, 8, 32, 40 are rows 0, 8, 16, 24; the slots' 136 row 8
BAND_WORDS = [
    # a run ending in a MAD before another run: an SFPNOP after it
    (
        "nop",
        [load("v1", 0), load("v3", 8), mul("v1", "v1", "v2"), mul("v3", "v3", "v4")]
        + [store("v4", 8), store("v2", 0)],
    ),
    # words before any store's band run with band 0; a word no store needs, in the
    # run before it; the last run ending in a MAD: no SFPNOP
    (
        "lead",
        [load("v5", 0), load("v1", 32), store("v1", 32), load("v2", 40)]
        + [store("v2", 40), mul("v2", "v2", "v3")],
    ),
    # a value two stores need: the first band of the two
    (
        "shared",
        [load("v1", 40), add("v1", "v1", "v2"), store("v2", 40), store("v2", 8)],
    ),
    # the per-row slots; slots 2 rows into their tile (136: row 0); a store past the
    # slots' tile (200, Dst tile 3: row 8)
    ("slots", [load("v1", 0), store("v1", 136), load("v2", 8), store("v2", 162)]),
    ("offset", [load("v1", 0), store("v1", 136)], 2),
    ("far", [load("v1", 0), store("v1", 200)]),
    # a word that writes two values two bands need: the first band of the two
    (
        "swapped",
        [load("v1", 0), load("v2", 8), swap("v1", "v2"), store("v1", 8)]
        + [store("v2", 0)],
    ),
    # lane flags, set or pushed: refused
    ("flags", [load("v1", 0), load("v2", 8), gt("v1", "v2"), ENCC, store("v1", 0)]),
    ("pushed", [load("v1", 0), w("SFPPUSHC", "push"), store("v1", 0)]),
]


def band_words(folder):
    """Today's `band_groups` on BAND_WORDS: each case's words in `band_<case>.in`, its
    items in `band_<case>.out` (a guarded run: its test and its passes at 0 .. 33 live
    rows, then its words indented), its slots' offset and verdict (what it did, or its
    refusal) a line of the result."""
    lines = []
    for case, words, *offset in BAND_WORDS:
        offset = offset[0] if offset else 0
        layouts = {"bcast_col": Layout("rows", 2, offset)}
        (folder / f"band_{case}.in").write_text("".join(f"{line(x)}\n" for x in words))
        try:
            items, done = band_groups(words, layouts)
        except UnitError as error:
            lines.append(f"{case}\t{offset}\t{error}")
            continue
        out = []
        for item in items:
            if isinstance(item, Insn):
                out.append(line(item))
                continue
            passes = " ".join(str(item.count.of(r)) for r in range(34))
            out.append(f"{item.name} if {item.count.text}: {passes}")
            out += [f"  {line(x)}" for x in item.body]
        (folder / f"band_{case}.out").write_text("".join(f"{x}\n" for x in out))
        lines.append(f"{case}\t{offset}\t{'; '.join(done)}")
    return lines


def branches(folder):
    """Today's row_program on BRANCHES and CALLS: each case's width, Dst and verdict
    (`made`, or its refusal's text), its KernelIR in `branch_<case>.ir` and the unit
    it makes in `branch_<case>.rows.txt`."""
    made = [(c, traced(c, fn, fm, fd), 2, "fp32") for c, fn, fm, fd in BRANCHES]
    made += [(c, tiled(c, fn), 2, "fp32") for c, fn in TILED]
    corpus = {c: (fn, fm, fd) for c, fn, fm, _, fd in CASES}
    made += [(c, traced(of, *corpus[of]), w, dest) for c, of, w, dest in CALLS]
    lines = []
    for case, ir, width, dest in made:
        (folder / f"branch_{case}.ir").write_text(ir.dump() + "\n")
        try:
            unit = row_program(ir, width, width, 2 * width, dest=dest)
            (folder / f"branch_{case}.rows.txt").write_text(unit.dump() + "\n")
            verdict = "made"
        except (PlanError, UnitError) as error:
            verdict = str(error)
        lines.append(f"{case}\t{width}\t{dest}\t{verdict}")
    return lines


if __name__ == "__main__":
    main(sys.argv[1])
