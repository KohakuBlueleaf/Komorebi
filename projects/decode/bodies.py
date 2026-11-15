"""The operations' vector bodies as tile functions (komorebi.dsl.tile): the residual
add, the squares, the norm's 1 / rms, the scale, the gate, the heads' norm and rotation,
emitted into kernels/generated/. Run it before a card job (`python
projects/decode/bodies.py`); the card refuses headers built from other sources."""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi.dsl.tile import emit, trace, unit_program  # noqa: E402
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam  # noqa: E402
from komorebi.dsl.tile.rows import row_program  # noqa: E402

GENERATED = HERE / "kernels" / "generated"
EPS = 1e-6  # Qwen3's norms (model.py)
ROW, HEAD = 2048, 128  # the columns a norm's mean is over: the row, a head
# The row form's Dst tiles: the row in 0, its per-row value's lanes in 1; no .row()
# operand (the per-column tile is not read).
ROWS_TILE, COLS_TILE = 1, 2

# Stream bodies, {name: (operand tiles, the expression)}: operand k is Dst tile dst + k,
# the result is stored into tile dst.
STREAMS = {
    "add": (2, lambda a, b: a + b),
    "square": (1, lambda a: a * a),
    "square_add": (2, lambda a, b: a + b * b),
    "squares3": (3, lambda a, b, c: a * a + b * b + c * c),
    "sum3": (3, lambda a, b, c: a + b + c),
    "mul": (2, lambda a, b: a * b),
    "mul3": (3, lambda a, b, c: a * b * c),
    "mad": (3, lambda a, b, c: a + b * c),
    "gate": (3, lambda a, b, c: trace.silu(a * b) * c),
}


def stream(name, inputs, fn):
    """The IR of y = fn(a, ..) over a stream of fp32 tiles."""
    params = [TensorParam(n, False, F32) for n in "abc"[:inputs]]
    y = TensorParam("y", True, F32)
    ir = KernelIR(name, [*params, y], [], {})
    with trace.Tracer(ir):
        proxies = [trace.TensorProxy(p) for p in params]
        ys = trace.TensorProxy(y)
        for t in trace.grid(trace.num_tiles(proxies[0])):
            trace.store(ys.tiles[t], fn(*[trace.load(p.tiles[t]) for p in proxies]))
    return ir


def rms(name, columns):
    """Each row's 1 / sqrt(mean square + EPS) in every column of one tile, from a tile
    whose row sums are the sums of squares over `columns` (the row form, a row of one
    tile; `row * 0` makes the stored value a row of the form's own)."""
    x, y = TensorParam("x", False, F32), TensorParam("y", True, F32)
    ir = KernelIR(name, [x, y], [], {})
    with trace.Tracer(ir):
        xs, ys = trace.TensorProxy(x), trace.TensorProxy(y)
        for t in trace.grid(trace.num_tile_rows(xs)):
            row = trace.load(xs.tiles[t, :])
            mean = trace.reduce_sum(row, exact=True) * (1.0 / columns)
            r = trace.rsqrt(mean + EPS)
            trace.store(ys.tiles[t, :], row * 0.0 + r.col())
    return ir


def units():
    """{header name: unit program}: the streams (each with the load-macro rule, which
    refuses what does not fit; `emit` refuses a template its setup does not store),
    the row's and the head's 1 / rms; each on the command's live rows (the band)."""
    out = {}
    for name, (inputs, fn) in STREAMS.items():
        ir = stream(name, inputs, fn)
        out[name] = unit_program(ir, load_macro=True, rows=True)
    for name, columns in (("rms_row", ROW), ("rms_head", HEAD)):
        ir = rms(name, columns)
        out[name] = row_program(ir, 1, ROWS_TILE, COLS_TILE, rows=True)
    return out


def sources_hash():
    """The hash of what the headers come from: the tile package and this file."""
    tile = Path(km.__file__).resolve().parent / "dsl" / "tile"
    digest = hashlib.sha256()
    for path in sorted(tile.glob("*.py")) + [Path(__file__).resolve()]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generate():
    """Writes every header into GENERATED and their costs beside them; returns the
    costs (math CPU cycles a unit at 1 and 32 live rows)."""
    GENERATED.mkdir(exist_ok=True)
    costs = {}
    for name, unit in units().items():
        (GENERATED / f"{name}.hpp").write_text(emit(unit))
        macro = next((p.done for p in unit.passes if p.name == "load_macro"), None)
        costs[name] = {
            "cycles": unit.cost("math")[1],
            "cycles_1_row": unit.cost("math", 1)[1],
            "load_macro": macro,
        }
    saved = {"sources": sources_hash(), "costs": costs}
    (GENERATED / "bodies.json").write_text(json.dumps(saved, indent=1))
    return costs


def require():
    """Refuses missing headers or headers built from other sources (a card job emits
    nothing: run this file first)."""
    try:
        saved = json.loads((GENERATED / "bodies.json").read_text())
    except (OSError, ValueError):
        raise km.KomorebiError(
            "no emitted bodies: run projects/decode/bodies.py"
        ) from None
    if saved["sources"] != sources_hash():
        raise km.KomorebiError(
            "emitted bodies are stale: run projects/decode/bodies.py"
        )


if __name__ == "__main__":
    print(json.dumps(generate(), indent=1))
