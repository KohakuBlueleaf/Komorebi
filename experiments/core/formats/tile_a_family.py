"""A tile function of the A family (README.md): the gate silu(a @ w1) * (a @ w3) and
x * y on fp16 and bfp8 tiles (a 16-bit Dst) beside bf16 and bfp8_b (an fp32 Dst),
against float64; `--set fix`: x * y of two formats; `--set edges`: fp16's range edges.
`build` on a mock device, `card`."""

import argparse
import json
import os
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.device import native
from komorebi.session import process
from komorebi.session import session as km_session

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from placement import DESCRIPTOR, JIT_CACHE, RESULTS, build_only, errors, log

CORES, TILES, REPEAT = 12, 64, 3  # cores, tiles a core, commands a handler
FORMS = {  # (X's format, W's format)
    "float16": ("float16", "float16"),
    "float16 x bfp8": ("float16", "bfp8"),
    "bfloat16": ("bfloat16", "bfloat16"),
    "bfloat16 x bfp8_b": ("bfloat16", "bfp8_b"),
}
STARTED = time.monotonic()


class Gate(km.Program):
    """gate: each core's TILES tiles of a, w1 and w3 from rank x TILES, the gate of
    each tile's two products into the same tile of out; mul: a * w1, tile by tile."""

    @km.tile_function
    def gate(a, w1, w3):
        return km.silu(a @ w1) * (a @ w3)

    @km.tile_function
    def times(x, y):
        return x * y

    @km.tile_function
    def product(x, y):
        return x @ y

    @km.handler
    def run(self, a: km.Tensor, w1: km.Tensor, w3: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x, p, q in zip(
            km.Stream(a.pages(first, TILES), block=8),
            km.Stream(w1.pages(first, TILES), block=8),
            km.Stream(w3.pages(first, TILES), block=8),
        ):
            results.push(self.gate(x, p, q))

    @km.handler
    def mul(self, a: km.Tensor, w1: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x, p in zip(
            km.Stream(a.pages(first, TILES), block=8),
            km.Stream(w1.pages(first, TILES), block=8),
        ):
            results.push(self.times(x, p))

    @km.handler
    def prod(self, a: km.Tensor, w1: km.Tensor, out: km.Tensor):
        first = self.rank * TILES
        results = km.Stream(out.pages(first, TILES), write=True, block=8)
        for x, p in zip(
            km.Stream(a.pages(first, TILES), block=8),
            km.Stream(w1.pages(first, TILES), block=8),
        ):
            results.push(self.product(x, p))


def range_inputs(seed=4):
    """float16's edges by tile (t % 3): 0, a normal x 8,000 and w1 normal (sums of
    std 45,000: some above 65,504); 1, a and w1 of magnitude 10^U(-6, -1) (values and
    products under 2^-14); 2, a normal and w1 normal / sqrt(32)."""
    rng = np.random.default_rng(seed)
    shape = (CORES * TILES, 32, 32)
    group = np.arange(CORES * TILES) % 3
    a = rng.standard_normal(shape)
    w1 = rng.standard_normal(shape) / np.sqrt(32)
    a[group == 0] *= 8000
    w1[group == 0] *= np.sqrt(32)
    for m in (a, w1):
        small = rng.uniform(-6, -1, m[group == 1].shape)
        m[group == 1] = np.sign(m[group == 1]) * 10.0**small
    w3 = np.zeros(shape)
    return [m.reshape(-1, 32).astype(np.float32) for m in (a, w1, w3)], group


def inputs(seed=3):
    """a normal, w1 and w3 normal / sqrt(32): CORES * TILES tiles each (float32)."""
    rng = np.random.default_rng(seed)
    shape = (CORES * TILES * 32, 32)
    a = rng.standard_normal(shape).astype(np.float32)
    w1 = (rng.standard_normal(shape) / np.sqrt(32)).astype(np.float32)
    w3 = (rng.standard_normal(shape) / np.sqrt(32)).astype(np.float32)
    return a, w1, w3


def stored(values, dtype):
    """The values a tensor of `dtype` holds (the host's encode and decode)."""
    return formats.from_bytes(*formats.to_bytes(values, dtype, "tile"), dtype, "tile")


def tiles_of(m):
    return m.astype(np.float64).reshape(-1, 32, 32)


def want_gate(a, w1, w3):
    """silu(a @ w1) * (a @ w3) of each tile, float64."""
    p, q = tiles_of(a) @ tiles_of(w1), tiles_of(a) @ tiles_of(w3)
    return (p / (1 + np.exp(-p)) * q).reshape(-1, 32)


def made(session, data, form, empty):
    """The tensors of a form: a, w1, w3 in its formats, out float32."""
    x_format, w_format = FORMS[form]
    out = {"out": session.empty(data[0].shape, "float32", "tile")}
    for name, values, dtype in zip(
        ("a", "w1", "w3"), data, (x_format, *[w_format] * 2)
    ):
        if empty:
            out[name] = session.empty(values.shape, dtype, "tile")
        else:
            out[name] = session.tensor(values, dtype, "tile")
    return out


FIX = ("float16 x bfp8", "bfloat16 x bfp8_b", "float16")  # the fix set's classes


def build(args):
    """Mock device: the set's classes built."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    process.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    with km.Session(log=log) as session:
        for form in FORMS if args.set == "base" else FIX:
            tensors = made(session, inputs(), form, empty=True)
            session.load(Gate(log=log, session=session, **tensors))
    return 0


def submitted(session, run, handler, tensors, shape):
    """REPEAT commands of `handler` on (tensors..., y); y and the microseconds."""
    y = session.empty(shape, "float32", "tile")
    times = []
    for _ in range(REPEAT):
        sent = run.submit(handler, (0, CORES), args=(*tensors, y))
        event = run.wait(sent, timeout=30.0)
        if event.kind != "done":
            raise RuntimeError(f"{handler.name}: {event}")
        times.append(event.microseconds)
    return y, times


def edges(got, want, group):
    """By range group: the error where both are finite and nonzero, the datums that
    read inf (where want is above 65,504 or not), and those that read 0 (where want
    is under 2^-14 or not)."""
    out = {}
    for g, name in enumerate(("large", "small", "ordinary")):
        rows = np.repeat(group == g, 32)
        y, w = got[rows], want[rows]
        big, tiny = np.abs(w) > 65504, np.abs(w) < 2.0**-14
        fine = np.isfinite(y) & (y != 0) & ~big & ~tiny
        out[name] = {
            **errors(y[fine], w[fine]),
            "want_above_65504": int(big.sum()),
            "inf_where_above": int(np.isinf(y[big]).sum()),
            "inf_elsewhere": int(np.isinf(y[~big]).sum()),
            "want_under_2^-14": int(tiny.sum()),
            "zero_where_under": int((y[tiny] == 0).sum()),
            "zero_elsewhere": int((y[~tiny] == 0).sum()),
        }
        if big.any():  # what a value above float16's range reads
            ratio = y[big] / w[big]
            out[name]["above"] = {
                "zero": int((y[big] == 0).sum()),
                "nan": int(np.isnan(y[big]).sum()),
                "largest_read": float(np.nanmax(np.abs(y[big]))),
                "read_over_want_min": float(np.nanmin(ratio)),
                "read_over_want_max": float(np.nanmax(ratio)),
                "first": [[float(u), float(v)] for u, v in zip(w[big][:8], y[big][:8])],
            }
    return out


def card_fix(edges_only=False):
    """One card: x * y of two formats in each family (not with `edges_only`), then
    float16's range edges."""
    data = inputs()
    rows = []
    with km.Session(log=log) as session:
        for form in () if edges_only else FIX[:2]:
            x_format, w_format = FORMS[form]
            tensors = made(session, data, form, empty=False)
            run = session.load(Gate(log=log, session=session, **tensors))
            try:
                words = (tensors["a"], tensors["w1"])
                y, times = submitted(session, run, Gate.mul, words, data[0].shape)
            finally:
                run.stop()
            a, w1 = stored(data[0], x_format), stored(data[1], w_format)
            got = y.read().astype(np.float64)
            row = {"form": form, "function": "mul"}
            row |= errors(got, a.astype(np.float64) * w1) | {"microseconds": times}
            rows.append(row)
            log(json.dumps(row))
        values, group = range_inputs()
        tensors = made(session, values, "float16", empty=False)
        run = session.load(Gate(log=log, session=session, **tensors))
        outs = {}
        try:
            for name, handler in (("prod", Gate.prod), ("mul", Gate.mul)):
                words = (tensors["a"], tensors["w1"])
                outs[name] = submitted(session, run, handler, words, values[0].shape)
        finally:
            run.stop()
        a, w1 = (stored(v, "float16").astype(np.float64) for v in values[:2])
        wants = {"prod": (tiles_of(a) @ tiles_of(w1)).reshape(-1, 32), "mul": a * w1}
        for name, (y, times) in outs.items():
            got = y.read().astype(np.float64)
            row = {"form": "float16 range", "function": name, "microseconds": times}
            row |= edges(got, wants[name], group)
            rows.append(row)
            log(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    name = "tile_a_family_edges.json" if edges_only else "tile_a_family_fix.json"
    with open(RESULTS / name, "w") as f:
        json.dump({"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def card(args):
    """One card: each form's class loaded in turn, gate then mul, REPEAT commands."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    if args.set != "base":
        return card_fix(edges_only=args.set == "edges")
    data = inputs()
    rows = []
    with km.Session(log=log) as session:
        for form, (x_format, w_format) in FORMS.items():
            tensors = made(session, data, form, empty=False)
            program = Gate(log=log, session=session, **tensors)
            run = session.load(program)
            outs = {}
            try:
                for name, handler, args_ in (
                    ("gate", Gate.run, ("a", "w1", "w3")),
                    ("mul", Gate.mul, ("a", "w1")),
                ):
                    y = session.empty(data[0].shape, "float32", "tile")
                    times = []
                    for _ in range(REPEAT):
                        words = (*[tensors[n] for n in args_], y)
                        sent = run.submit(handler, (0, CORES), args=words)
                        event = run.wait(sent, timeout=30.0)
                        if event.kind != "done":
                            raise RuntimeError(f"{form} {name}: {event}")
                        times.append(event.microseconds)
                    outs[name] = (y, times)
            finally:
                run.stop()
            a, w1, w3 = (
                stored(v, f) for v, f in zip(data, (x_format, w_format, w_format))
            )
            wants = {"gate": want_gate(a, w1, w3), "mul": a.astype(np.float64) * w1}
            for name, (y, times) in outs.items():
                got = y.read().astype(np.float64)
                row = {"form": form, "function": name, **errors(got, wants[name])}
                row["microseconds"] = times
                if name == "mul":  # the store's rounding: nearest-even or toward zero
                    want = wants[name]
                    dtype = np.float16 if x_format == "float16" else np.float32
                    nearest = want.astype(dtype).astype(np.float64)
                    toward = np.where(
                        np.abs(nearest) > np.abs(want),
                        np.nextafter(nearest.astype(dtype), dtype(0)),
                        nearest.astype(dtype),
                    ).astype(np.float64)
                    finite = np.abs(want) >= 2.0**-14
                    row["equal_nearest_even"] = float((got == nearest)[finite].mean())
                    row["equal_toward_zero"] = float((got == toward)[finite].mean())
                rows.append(row)
                log(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "tile_a_family.json", "w") as f:
        json.dump({"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    p.add_argument("--set", choices=["base", "fix", "edges"], default="base")
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
