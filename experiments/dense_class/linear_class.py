"""Y = X W as a class (the page's Linear, W's columns resident in each core's L1, X
streamed past them) and a SwiGLU gate after it, against the host's product (build/
linear/ beside this file). See README.md.
Example: python linear_class.py --rows 1 --check exact random swiglu
"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km

STARTED = time.monotonic()
K_TILES = 128  # K = 4096
COLUMNS = 2  # W's column tiles a core holds: N = cores x COLUMNS tiles
WEIGHT = K_TILES * COLUMNS  # the field's tiles
BLOCK = 8  # X's tiles a block
GATE_BLOCK = 2  # the gate's tiles a block


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function(tolerance=1e-6)
def silu_gate(a, b):
    """The SwiGLU gate of two tiles."""
    return km.silu(a) * b


class Linear(km.Program):
    """apply: core r holds W's columns r x COLUMNS .. + COLUMNS - 1 (K rows of them)
    and makes those columns of Y for each of X's `rows` row tiles; gate: out = silu(a)
    x b over each core's `tiles` pages."""

    weight: km.L1[km.bf16, K_TILES, COLUMNS, 32, 32]

    @km.tile_function
    def project(x, w):
        return x @ w

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, rows: int):
        self.weight.fill(km.Stream(w.pages(self.rank * WEIGHT, WEIGHT)))
        for row in range(rows):
            acc = km.tile.zeros(km.fp32, COLUMNS)
            for block in km.Stream(x.pages(row * K_TILES, K_TILES), block=BLOCK):
                acc += self.project(block, self.weight)
            y.pages((self.rank * rows + row) * COLUMNS, COLUMNS).write(acc)

    @km.handler
    def gate(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=GATE_BLOCK)
        for h, g in zip(
            km.Stream(a.pages(first, tiles), block=GATE_BLOCK),
            km.Stream(b.pages(first, tiles), block=GATE_BLOCK),
        ):
            results.push(silu_gate(h, g))


def by_core(w, cores):
    """W [K, N] as each core's columns, [cores, K, COLUMNS x 32]: a core's K rows of
    tiles one after another (the field's order)."""
    return w.reshape(w.shape[0], cores, COLUMNS * 32).transpose(1, 0, 2)


def from_cores(blocks, rows):
    """Y [M, N] from the cores' blocks [cores, M, COLUMNS x 32]."""
    cores = blocks.shape[0]
    return blocks.transpose(1, 0, 2).reshape(rows * 32, cores * COLUMNS * 32)


def bf16(values):
    """float32 values rounded to bfloat16 (to nearest even), as float64."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32).astype(np.float64)


def timed_apply(run, args, x, w, y, rows, cores, repeat):
    """`repeat` apply commands; their events' microseconds."""
    times = []
    for _ in range(repeat):
        submitted = run.submit(Linear.apply, (0, cores), args=(x, w, y, rows))
        event = run.wait(submitted, timeout=30.0)
        if event.kind != "done":
            raise RuntimeError(f"apply: {event}")
        times.append(event.microseconds)
    return times


def made(session, rows, cores):
    """The class made with tensors of the shape: X [M, K], W by core, Y by core."""
    m, k, n = rows * 32, K_TILES * 32, cores * COLUMNS * 32
    tensors = {
        "x": session.empty((m, k), "bfloat16", "tile"),
        "w": session.empty((cores, k, COLUMNS * 32), "bfloat16", "tile"),
        "y": session.empty((cores, m, COLUMNS * 32), "bfloat16", "tile"),
        "a": session.empty((cores, m, COLUMNS * 32), "bfloat16", "tile"),
        "b": session.empty((cores, m, COLUMNS * 32), "bfloat16", "tile"),
        "out": session.empty((cores, m, COLUMNS * 32), "bfloat16", "tile"),
    }
    program = Linear(log=log, **tensors)
    return program, (m, k, n)


def check_exact(session, args, cores):
    """Small integers (x in -1, 0, 1; w in -1, 0, 0, 0, 1): every sum exact in fp32 and
    under 256, so exact in bfloat16 too; Y equal to the host's product."""
    rng = np.random.default_rng(args.seed)
    program, (m, k, n) = made(session, args.rows, cores)
    x = rng.integers(-1, 2, (m, k)).astype(np.float32)
    w = rng.choice(np.array([-1, 0, 0, 0, 1], np.float32), (k, n))
    tx = session.tensor(x, "bfloat16", "tile")
    tw = session.tensor(by_core(w, cores), "bfloat16", "tile")
    shape = (cores, m, COLUMNS * 32)
    ty = session.tensor(np.zeros(shape, np.float32), "bfloat16", "tile")
    run = session.load(program)
    try:
        times = timed_apply(run, args, tx, tw, ty, args.rows, cores, 1)
    finally:
        run.stop()
    got = from_cores(ty.read(), args.rows)
    want = x.astype(np.float64) @ w.astype(np.float64)
    return {
        "kernel_text": program.sizes.text,
        "microseconds": times,
        "exact": bool((got == want).all()),
        "wrong": int((got != want).sum()),
        "largest_sum": float(np.abs(want).max()),
        "sums_exact_in_bf16": bool(np.abs(want).max() <= 256),
    }


def check_random(session, args, cores, swiglu):
    """Normal values in bfloat16: Y (bfloat16) against float64 of the same inputs, as
    the dense project judges (relative RMS); `swiglu`: two products and the gate."""
    rng = np.random.default_rng(args.seed)
    program, (m, k, n) = made(session, args.rows, cores)
    x = bf16(rng.standard_normal((m, k)))
    weights = [
        bf16(rng.standard_normal((k, n)) / 64) for _ in range(2 if swiglu else 1)
    ]
    tx = session.tensor(x, "bfloat16", "tile")
    tws = [session.tensor(by_core(w, cores), "bfloat16", "tile") for w in weights]
    shape = (cores, m, COLUMNS * 32)
    tys = [session.tensor(np.zeros(shape, np.float32), "bfloat16", "tile") for _ in tws]
    out = session.tensor(np.zeros(shape, np.float32), "bfloat16", "tile")
    run = session.load(program)
    record = {"kernel_text": program.sizes.text}
    try:
        times = timed_apply(
            run, args, tx, tws[0], tys[0], args.rows, cores, args.repeat
        )
        record["microseconds"] = times
        record["median_microseconds"] = float(np.median(times))
        if swiglu:
            times += timed_apply(run, args, tx, tws[1], tys[1], args.rows, cores, 1)
            pages = args.rows * COLUMNS
            submitted = run.submit(
                Linear.gate, (0, cores), args=(tys[0], tys[1], out, pages)
            )
            event = run.wait(submitted, timeout=30.0)
            record["gate_microseconds"] = event.microseconds
    finally:
        run.stop()
    want = x @ weights[0]
    got = from_cores(tys[0].read(), args.rows)
    record["relative_rms"] = float(
        np.sqrt(((got - want) ** 2).mean()) / np.sqrt((want**2).mean())
    )
    if swiglu:
        h1, h3 = (from_cores(t.read(), args.rows) for t in tys)
        gate = h1 / (1 + np.exp(-h1)) * h3  # of the card's bfloat16 products
        whole = (x @ weights[0]) / (1 + np.exp(-(x @ weights[0]))) * (x @ weights[1])
        got = from_cores(out.read(), args.rows)
        record["gate_relative_rms"] = float(
            np.sqrt(((got - gate) ** 2).mean()) / np.sqrt((gate**2).mean())
        )
        record["swiglu_relative_rms"] = float(
            np.sqrt(((got - whole) ** 2).mean()) / np.sqrt((whole**2).mean())
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, default=1, help="X's row tiles (M / 32)")
    parser.add_argument("--cores", type=int, default=64, help="N = cores x 64")
    parser.add_argument("--check", nargs="+", default=["exact", "random", "swiglu"])
    parser.add_argument("--repeat", type=int, default=3, help="timed apply commands")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    report = {"m_k_n": [args.rows * 32, K_TILES * 32, args.cores * COLUMNS * 32]}
    ok = True
    with km.Session(log=log) as session:
        log(f"device open; {args.cores} cores, M = {args.rows * 32}")
        if "exact" in args.check:
            report["exact"] = check_exact(session, args, args.cores)
            ok &= report["exact"]["exact"]
            log(f"exact: {report['exact']}")
        for name in ("random", "swiglu"):
            if name in args.check:
                report[name] = check_random(session, args, args.cores, name == "swiglu")
                ok &= report[name]["relative_rms"] < 1e-2
                log(f"{name}: {report[name]}")
    report["ok"] = bool(ok)
    print(json.dumps(report, default=float), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
