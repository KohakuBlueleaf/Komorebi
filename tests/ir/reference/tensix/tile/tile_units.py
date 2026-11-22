"""Today's tile compiler on a corpus of tile functions, the port's reference (written in
the IR's text in tile_units.kom). Writes each KernelIR, chain, unit as data, listing
(`.txt`), header and listing after the emit (`.hpp`, `.emitted.txt`), each word's
encoding, refusals, Dst, band, a band's cost by live rows. Usage: <this> <folder>."""

import inspect
import operator
import random
import sys
from pathlib import Path

import komorebi as km
from komorebi.dsl.compile.tiles.functions import SCALAR, TILE_FORMATS, tile_dest
from komorebi.dsl.tile import emit, trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam
from komorebi.dsl.tile.lower import stream_chain
from komorebi.dsl.tile.unit import (
    CPUS,
    OPS,
    Call,
    Insn,
    Loop,
    Product,
    Runtime,
    Sync,
    UnitError,
    decode,
    encode,
)

BF, F = "bfloat16", "float32"
SEP = "\x1f"  # between the items of a list in one field


def half(a):
    """A function the corpus calls from another; `called` also compares a number on
    the left (`0.0 < b`, which Python reflects)."""
    return a * 0.5


# (case, function, formats, unit_program's options): the lessons' and the examples'
# functions, and one of each vector function at the 1-tile width.
CASES = [
    ("times", lambda a, k: a * k, [BF, SCALAR], {}),
    ("plus", lambda a, b: a + b, [BF, BF], {}),
    ("plus_band", lambda a, b: a + b, [F, F], {"rows": True}),
    # the band beside a scalar, in a bf16 Dst, without a load macro, on the per-row
    # path (with lane flags), after products, over a vector function; a product alone
    ("times_band", lambda a, k: a * k, [BF, SCALAR], {"rows": True}),
    ("plus_bf16_band", lambda a, b: a + b, [BF, BF], {"rows": True}),
    ("smooth_band", lambda a: a * 0.5 + abs(a) * 0.5, [F], {"rows": True}),
    ("by_col_band", lambda a, b: a * b.col(), [F, F], {"rows": True}),
    (
        "col_select_band",
        lambda a, b: km.where(b.col() > 0.0, a, -a),
        [F, F],
        {"rows": True},
    ),
    ("project_silu_band", lambda x, w: km.silu(x @ w), [BF, BF], {"rows": True}),
    ("exp_band", lambda a: km.exp(a), [F], {"rows": True}),
    ("project_band", lambda x, w: x @ w, [BF, BF], {"rows": True}),
    ("plus_bfp8", lambda a, b: a + b, ["bfp8_b", "bfp8_b"], {}),
    ("gate", lambda a, b: km.silu(a) * b, [F, F], {}),
    ("silu_gate_bf16", lambda a, b: km.silu(a) * b, [BF, BF], {}),
    ("clamp01", lambda a: km.minimum(km.maximum(a, 0.0), 1.0), [F], {}),
    ("leaky", lambda a: km.where(a > 0.0, a, a * 0.01), [F], {}),
    ("smooth", lambda a: a * 0.5 + abs(a) * 0.5, [F], {}),
    ("project", lambda x, w: x @ w, [BF, BF], {}),
    ("project_hifi2", lambda x, w: x @ w, [BF, BF], {"fidelity": "HiFi2"}),
    # a tile's row 0 and column 0 spread (bcast_row, bcast_col: the per-row path, its
    # pre-pass taken and kept in the rows)
    ("by_row", lambda a, b: a * b.row(), [F, F], {}),
    ("by_col", lambda a, b: a * b.col(), [F, F], {}),
    ("col_prepass", lambda a, b: a * km.rsqrt(b.col() * 0.5 + 1.0), [F, F], {}),
    ("col_cheap", lambda a, b: a * (b.col() + 1.0), [F, F], {}),
    # the per-row passes where they act: copies dropped, a per-row value read beside
    # the operand, a select's run under lane flags moved whole, nothing moved
    ("col_relu", lambda a, b: a * km.relu(b.col()), [F, F], {}),
    ("col_min", lambda a, b: km.minimum(a, b.col()), [F, F], {}),
    ("col_select", lambda a, b: km.where(b.col() > 0.0, a, -a), [F, F], {}),
    ("col_silu", lambda a, b: km.silu(a * b.col()), [F, F], {}),
    # a pre-pass estimated with its constants' loads (kept in the rows)
    ("col_poly", lambda a, b: a * ((b.col() * 0.5 + 1.0) * b.col() + 2.0), [F, F], {}),
    # products before the vector code: one, one beside a loaded tile, two
    ("project_silu", lambda x, w: km.silu(x @ w), [BF, BF], {}),
    ("project_bias", lambda x, w, b: x @ w + b, [BF, BF, F], {}),
    ("two_products", lambda x, w, v: (x @ w) * (x @ v), [BF, BF, BF], {}),
    ("scaled_sum", lambda a, b, k: (a + b) * k, [F, F, SCALAR], {}),
    ("exp", lambda a: km.exp(a), [F], {}),
    ("sigmoid", lambda a: km.sigmoid(a), [F], {}),
    ("silu", lambda a: km.silu(a), [F], {}),
    ("tanh", lambda a: km.tanh(a), [F], {}),
    ("erf", lambda a: km.erf(a), [F], {}),
    ("log", lambda a: km.log(a), [F], {}),
    ("sqrt", lambda a: km.sqrt(a), [F], {}),
    ("rsqrt", lambda a: km.rsqrt(a), [F], {}),
    ("recip", lambda a: km.reciprocal(a), [F], {}),
    ("relu", lambda a: km.relu(a), [F], {}),
    ("square", lambda a: km.square(a), [F], {}),
    ("neg", lambda a: -a, [F], {}),
    ("divide", lambda a, b: a / b, [F, F], {}),
    ("floor", lambda a: km.floor(a), [F], {}),
    ("twice_k", lambda a, k: a * k + k * a, [F, SCALAR], {}),
    # a value takes the register of an operand dying there, not the lowest free one
    ("chained", lambda a, b, c: a * b * c - a, [F, F, F], {}),
    # two reciprocals share the register of their 2.0
    ("two_recips", lambda a, b: km.reciprocal(a) * km.reciprocal(b), [F, F], {}),
    # rsqrt's three constant registers and recip's 2.0: one more than there are
    ("four_registers", lambda a: km.rsqrt(a) * km.reciprocal(a), [F], {}),
    # the selects of SFPGT's inverse tests, and the tests against +0.0 (not -0.0)
    ("pick_ge", lambda a, b: km.where(a >= b, a, b), [F, F], {}),
    (
        "pick_le_ne",
        lambda a, b: km.where(a <= b, a, b) + km.where(a != b, a, b),
        [F, F],
        {},
    ),
    ("pick_eq", lambda a, b: km.where(a == b, a, b), [F, F], {}),
    (
        "zero_tests",
        lambda a, b: km.where(a == 0.0, b, a)
        + km.where(a != 0.0, a, b)
        + km.where(a == -0.0, b, a),
        [F, F],
        {},
    ),
    ("five", lambda a, b, c, d, e: a * b + (c - d) * e, [F] * 5, {}),
    ("by5", lambda a, j, k, m, n, p: a * j * k * m * n * p, [F, *[SCALAR] * 5], {}),
    (
        "called",
        lambda a, b: half(a) + half(km.where(operator.lt(0.0, b), b, a)),
        [F, F],
        {},
    ),
]


def traced(name, function, formats):
    """The KernelIR of `function` on one tile of each format, as today's call site
    traces it (compile/tiles/functions.py, tile_unit, at the 1-tile width)."""
    names = list(inspect.signature(function).parameters)
    tensors = [
        TensorParam(p, False, TILE_FORMATS[f])
        for p, f in zip(names, formats, strict=True)
        if f != SCALAR
    ]
    values = [
        trace.ValueParam(p, "float")
        for p, f in zip(names, formats, strict=True)
        if f == SCALAR
    ]
    result = TensorParam("result", True, F32)
    ir = KernelIR(name, [*tensors, result], values, {})
    with trace.Tracer(ir):
        proxies = [trace.TensorProxy(t) for t in tensors]
        out = trace.TensorProxy(result)
        for t in trace.grid(trace.num_tiles(proxies[0])):
            loads = iter([trace.load(p.tiles[t]) for p in proxies])
            given = iter(values)
            args = [next(given if f == SCALAR else loads) for f in formats]
            trace.store(out.tiles[t], function(*args))
    return ir


def value(v):
    """An argument's text: `i<int>` or `s<name>`."""
    return f"i{v}" if isinstance(v, int) else f"s{v}"


def items(lines, things, depth):
    """Each item of a loop nest as a line, its depth first."""
    for item in things:
        if isinstance(item, Loop):
            count = item.count
            if isinstance(count, Runtime):
                count = f"r{int(count.guard)}{SEP}{count.text}{SEP}{count.of(32)}"
            else:
                count = f"n{count}"
            lines.append(f"LOOP\t{depth}\t{item.name}\t{count}\t{int(item.unroll)}")
            items(lines, item.body, depth + 1)
        elif isinstance(item, Insn):
            args = SEP.join(f"{n}={value(item.args[n])}" for n, _ in OPS[item.op].args)
            uses = SEP.join(value(u) for u in item.uses)
            lines.append(
                f"INSN\t{depth}\t{item.op}\t{args}\t{item.note}\t{uses}\t{item.runtime}"
            )
        elif isinstance(item, Sync):
            lines.append(
                f"SYNC\t{depth}\t{item.cpu}\t{item.other}\t{item.text}\t{item.note}"
            )
        elif isinstance(item, Product):
            words = SEP.join(f"{op}:{block}" for op, block in item.words)
            lines.append(
                f"PRODUCT\t{depth}\t{item.text}\t{item.note}\t{int(item.synced)}\t{words}"
            )
            for kind, insns in (("PINSN", item.insns), ("PWORMHOLE", item.wormhole)):
                at = len(lines)
                items(lines, insns, depth + 1)
                lines[at:] = [line.replace("INSN", kind, 1) for line in lines[at:]]
        elif isinstance(item, Call):
            lines.append(f"CALL\t{depth}\t{item.text}\t{item.note}")
        else:
            raise TypeError(item)


def constant(v):
    """A constant register's value: `f<repr>` of a float, `i` an int, `t` a text."""
    if isinstance(v, float):
        return f"f{v!r}"
    if isinstance(v, int):
        return f"i{v}"
    return f"t{v}"


def unit_lines(unit):
    scalars = SEP.join(unit.scalars)
    lines = [f"UNIT\t{unit.name}\t{unit.dest}\t{int(unit.band)}\t{scalars}"]
    lines += [f"CONST\t{r}\t{constant(v)}" for r, v in unit.constants.items()]
    lines += [f"CLOBBER\t{c}" for c in unit.clobbers]
    for cpu in CPUS:
        lines.append(f"INIT\t{cpu}")
        items(lines, unit.init[cpu], 0)
        lines.append(f"TILE\t{cpu}")
        items(lines, unit.tile[cpu], 0)
    for part, things in unit.parts.items():
        lines.append(f"PART\t{part}")
        items(lines, things, 0)
    lines += [f"PASS\t{p.kind}\t{p.name}\t{SEP.join(p.done)}" for p in unit.passes]
    lines += [f"CHECK\t{c.rule}\t{c.text}\t{c.verdict}" for c in unit.checks]
    lines += [f"NOTCHECKED\t{rule}" for rule in unit.not_checked]
    return lines


def chain_lines(chain, lines):
    """A chain as text, its earlier stages first: operands, scalars, the operations
    counted, the function's facts and its SFPI code; then the templates chosen."""
    if chain.prev is not None:
        chain_lines(chain.prev, lines)
    fn = chain.fn
    lines.append(f"CHAIN {fn.name} out={chain.out} pairs={chain.pairs}")
    for k, leaf in enumerate(chain.operands):
        source = leaf.source
        if isinstance(source, str):
            what = source
        elif source.kind == "load":
            what = f"load {source.attrs['view'].text()}"
        else:
            what = source.kind
        offset = None if leaf.offset is None else leaf.offset.text()
        lines.append(f"operand {k} {what} {leaf.kind} {offset}")
    lines += [f"value {k} {s.key()}" for k, s in enumerate(chain.values)]
    lines.append("ops " + " ".join(f"{k}={n}" for k, n in chain.ops.items()))
    lines.append(
        f"fn operands={fn.operands} scalars={fn.scalars} init={fn.init} fp32={fn.fp32}"
    )
    lines += [f"header {h}" for h in fn.headers]
    lines += [f"along {k} {v}" for k, v in fn.along.items()]
    lines += fn.code.split("\n")
    return lines


def encodings():
    """For each operation, words of chosen and random fields, each `name fields word
    decoded`; then a field one bit too wide, with today's refusal."""
    lines = []
    for name, op in OPS.items():
        rng = random.Random(name)
        chosen = [
            {arg: 0 for arg, _ in op.args},
            {arg: (1 << width) - 1 for arg, width in op.args},
            *[
                {arg: rng.randrange(1 << width) for arg, width in op.args}
                for _ in range(8)
            ],
        ]
        for args in chosen:
            word = encode(name, args)
            fields = ",".join(f"{a}={v}" for a, v in args.items())
            again = decode(word)
            back = ",".join(f"{a}={v}" for a, v in again[1].items())
            lines.append(f"WORD\t{name}\t{fields}\t{word}\t{again[0]}\t{back}")
        for arg, width in op.args:
            args = {a: 0 for a, _ in op.args}
            args[arg] = 1 << width
            try:
                encode(name, args)
            except UnitError as error:
                fields = ",".join(f"{a}={v}" for a, v in args.items())
                lines.append(f"WIDE\t{name}\t{fields}\t{error}")
    lines.extend(
        f"UNKNOWN\t{word}\t{decode(word) is None}"
        for word in (0x00000000, 0xFF000000, 0x12345678)
    )
    return lines


def dst_lines(function, unit):
    """Today's handler's Dst layout of a call (compile/tiles/functions.py, tile_unit):
    Dst tile k holds the parameter unit.operands[k] names; result k is in Dst tile k."""
    names = list(inspect.signature(function).parameters)
    taken = [[k for k, o in enumerate(unit.operands) if o == n] for n in names]
    lines = ["param\t" + ",".join(map(str, tiles)) for tiles in taken]
    return [*lines, "result\t" + ",".join(map(str, range(unit.results)))]


def fields(unit):
    """What a caller reads of a unit that its listing does not print."""
    products = [f"{p.dst}:{p.a}:{p.w}" for p in unit.products]
    return [
        "operands\t" + ",".join(o or "-" for o in unit.operands),
        "row\t" + ",".join(map(str, unit.row_operands)),
        "col\t" + ",".join(map(str, unit.col_operands)),
        f"results\t{unit.results}",
        "products\t" + ",".join(products),
        f"fidelity\t{unit.fidelity or '-'}",
    ]


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    refused, not_emitted, cases = [], [], []
    for case, function, formats, options in CASES:
        cases.append(f"{case}\t{tile_dest(formats)}\t{int(bool(options.get('rows')))}")
        ir = traced(case, function, formats)
        (folder / f"{case}.ir").write_text(ir.dump() + "\n")
        try:
            chain = stream_chain(ir)
            made = chain_lines(chain, [])
            made += [f"template {k}: {v}" for k, v in chain.templates.items()]
            (folder / f"{case}.chain").write_text("\n".join(made) + "\n")
            unit = unit_program(ir, tile_dest(formats), load_macro=True, **options)
            text = unit.dump()
        except (PlanError, UnitError) as error:
            refused.append(f"{case}\t{error}")
            continue
        (folder / f"{case}.unit").write_text("\n".join(unit_lines(unit)) + "\n")
        (folder / f"{case}.txt").write_text(text + "\n")
        (folder / f"{case}.fields").write_text("\n".join(fields(unit)) + "\n")
        (folder / f"{case}.dst").write_text("\n".join(dst_lines(function, unit)) + "\n")
        if unit.band:  # the run-time counts: the cost at 0 .. 33 live rows
            costs = [(r, *unit.cost("math", r)[:2]) for r in range(34)]
            (folder / f"{case}.band").write_text(
                "".join(f"{r}\t{n}\t{c!r}\n" for r, n, c in costs)
            )
        try:
            words = emit(unit)  # as the class's files are written (program.write)
        except UnitError as error:
            not_emitted.append(f"{case}\t{error}")
            continue
        (folder / f"{case}.hpp").write_text(words)
        (folder / f"{case}.emitted.txt").write_text(unit.dump() + "\n")
    (folder / "encode.txt").write_text("\n".join(encodings()) + "\n")
    (folder / "refused.txt").write_text("".join(f"{r}\n" for r in refused))
    (folder / "not_emitted.txt").write_text("".join(f"{r}\n" for r in not_emitted))
    (folder / "cases.txt").write_text("".join(f"{c}\n" for c in cases))


if __name__ == "__main__":
    main(sys.argv[1])
