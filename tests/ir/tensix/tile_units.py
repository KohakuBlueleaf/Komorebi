"""Today's tile compiler on a corpus of tile functions, as the reference for the port's
unit program (backend/tensix/tile/unit.rs): for each function its unit program as
data (`<case>.unit`, tab-separated lines the Rust test reads) and as today's listing
(`<case>.txt`); the encoding of words over the whole operation table (`encode.txt`);
the functions today refuses (`refused.txt`). No card, no compiler.
Usage: python tile_units.py <folder>."""

import random
import sys
from pathlib import Path

import komorebi as km
from komorebi.dsl.compile.tiles.functions import SCALAR, TILE_FORMATS, tile_dest
from komorebi.dsl.tile import trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam
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

# (case, function, formats, unit_program's options): the lessons' and the examples'
# functions, and one of each vector function at the 1-tile width.
CASES = [
    ("times", lambda a, k: a * k, [BF, SCALAR], {}),
    ("plus", lambda a, b: a + b, [BF, BF], {}),
    ("plus_band", lambda a, b: a + b, [F, F], {"rows": True}),
    ("gate", lambda a, b: km.silu(a) * b, [F, F], {}),
    ("silu_gate_bf16", lambda a, b: km.silu(a) * b, [BF, BF], {}),
    ("clamp01", lambda a: km.minimum(km.maximum(a, 0.0), 1.0), [F], {}),
    ("leaky", lambda a: km.where(a > 0.0, a, a * 0.01), [F], {}),
    ("smooth", lambda a: a * 0.5 + abs(a) * 0.5, [F], {}),
    ("project", lambda x, w: x @ w, [BF, BF], {}),
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
]


def traced(name, function, formats):
    """The KernelIR of `function` on one tile of each format, as today's call site
    traces it (compile/tiles/functions.py, tile_unit, at the 1-tile width)."""
    names = "abcdefgh"[: len(formats)]
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
            lines.append(f"INSN\t{depth}\t{item.op}\t{args}\t{item.note}\t{uses}")
        elif isinstance(item, Sync):
            lines.append(
                f"SYNC\t{depth}\t{item.cpu}\t{item.other}\t{item.text}\t{item.note}"
            )
        elif isinstance(item, Product):
            words = SEP.join(f"{op}:{block}" for op, block in item.words)
            lines.append(
                f"PRODUCT\t{depth}\t{item.text}\t{item.note}\t{int(item.synced)}\t{words}"
            )
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
    lines = [f"UNIT\t{unit.name}\t{unit.dest}"]
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


def main(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    refused = []
    for case, function, formats, options in CASES:
        try:
            ir = traced(case, function, formats)
            unit = unit_program(ir, tile_dest(formats), load_macro=True, **options)
            text = unit.dump()
        except (PlanError, UnitError) as error:
            refused.append(f"{case}\t{error}")
            continue
        (folder / f"{case}.unit").write_text("\n".join(unit_lines(unit)) + "\n")
        (folder / f"{case}.txt").write_text(text + "\n")
    (folder / "encode.txt").write_text("\n".join(encodings()) + "\n")
    (folder / "refused.txt").write_text("".join(f"{r}\n" for r in refused))


if __name__ == "__main__":
    main(sys.argv[1])
