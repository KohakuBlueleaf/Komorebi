"""Y = X W as a class in the dense project's two plans, picked by the shape as its plan
picker does (the K split for one row tile of X, the grid plan for more), and a SwiGLU
gate after it; against the host's product (build/linear/ beside this file). README.md.
Example: python linear_class.py --rows 1 32 --check exact swiglu"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km

STARTED = time.monotonic()
K_TILES = 128  # K = 4096
N_TILES = 128  # N = 4096
# The K split (the dense project's 14,1,16): groups of SPLIT cores, a group WIDTH
# columns of Y, a core 1/SPLIT of K; X and W stream in step, CHUNK K tiles a block.
SPLIT = 14
WIDTH = 16
SPLIT_CORES = N_TILES // WIDTH * SPLIT  # 112
CHUNK = 2
W_BLOCK = CHUNK * WIDTH  # W's tiles a block: CHUNK rows of the group's columns
# The grid on COLUMNS x ROWS cores: core (x, y) Y's block (y, x) of OH x OW tiles, the
# last column 7 wide; K chunks of KC tiles cast, X's along a row, W's down a column.
COLUMNS = 12
ROWS = 8
OH = 4  # the block's rows at M = 1024; the class's symbol, the plan gives the shape's
KC = 2  # K tiles a chunk: at 4 the casts' slots and the K split's are over a core's L1
OW = -(-N_TILES // COLUMNS)  # 11
# The kernel buffer: the five kernels' text is over the default 70,656 B (the dense
# project's linear kernels take 88 KiB; this class's 90,192 B, 89 KiB).
KERNEL_BYTES = 89 * 1024
GATE_BLOCK = 2  # the gate's tiles a block


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function(tolerance=1e-6)
def silu_gate(a, b):
    """The SwiGLU gate of two tiles."""
    return km.silu(a) * b


class Linear(km.Program):
    """split: Y = X W for one row tile of X, in groups of SPLIT cores (the K split);
    grid: Y = X W on a COLUMNS x ROWS rectangle (the grid plan), OH row tiles a core;
    gate: out = silu(a) x b over each core's `tiles` pages."""

    OH = km.symbol()  # the plan's
    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]
    part: km.L1[km.fp32, OH, OW, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def split(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        share = self.rank % SPLIT  # this core's part of K
        first = self.rank // SPLIT * WIDTH  # the group's columns of Y
        k0 = share * K_TILES // SPLIT
        k1 = (share + 1) * K_TILES // SPLIT
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=CHUNK),
            km.Stream(w.tiles[k0:k1, first : first + WIDTH], block=W_BLOCK),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=SPLIT)
        if self.rank == self.leader:
            y.tiles[0:1, first : first + WIDTH].write(whole)

    @km.handler
    def grid(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        along = km.Broadcast(self, along="row", window=2)
        down = km.Broadcast(self, along="column", window=2)
        top = self.y * OH
        left = self.x * OW
        for chunk in range(K_TILES // KC):
            k0 = chunk * KC
            if along.sender():
                xs = along.cast(x.tiles[top : top + OH, k0 : k0 + KC].read())
            else:
                xs = along.arrive()
            if down.sender():
                ws = down.cast(
                    w.tiles[k0 : k0 + KC, left : min(left + OW, N_TILES)].read()
                )
            else:
                ws = down.arrive()
            self.part += self.project(xs, ws)
        y.tiles[top : top + OH, left : min(left + OW, N_TILES)].write(self.part)

    @km.handler
    def gate(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=GATE_BLOCK)
        for h, g in zip(
            km.Stream(a.pages(first, tiles), block=GATE_BLOCK),
            km.Stream(b.pages(first, tiles), block=GATE_BLOCK),
        ):
            results.push(silu_gate(h, g))


def bf16(values):
    """float32 values rounded to bfloat16 (to nearest even), as float64."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32).astype(np.float64)


def plan_of(rows):
    """The plan picker: the K split for one row tile, else the grid plan; (handler,
    its group or None for the rectangle's)."""
    if rows == 1:
        return Linear.split, (0, SPLIT_CORES)
    if rows % ROWS:
        raise SystemExit(f"the grid plan takes row tiles in multiples of {ROWS}")
    return Linear.grid, None


def made(session, rows):
    """The class made with tensors of the largest shape of `rows` (X [M, K], W [K, N],
    Y [M, N]); the grid's OH x OW block of 1024 rows when none is larger, so one row
    tile and 32 load the same program."""
    rows = max(*rows, ROWS * OH)
    m, k, n = rows * 32, K_TILES * 32, N_TILES * 32
    tensors = {
        name: session.empty(shape, "bfloat16", "tile")
        for name, shape in (
            ("x", (m, k)),
            ("w", (k, n)),
            ("y", (m, n)),
            ("a", (m, n)),
            ("b", (m, n)),
            ("out", (m, n)),
        )
    }
    oh = rows // ROWS
    program = Linear(
        log=log,
        plan={"OH": oh},
        rect=(0, 0, COLUMNS, ROWS),
        session=session,
        **tensors,
    )
    return program


def timed(run, handler, group, args, repeat):
    """`repeat` commands of the handler; their events' microseconds."""
    times = []
    for _ in range(repeat):
        submitted = run.submit(handler, group, args=args)
        event = run.wait(submitted, timeout=30.0)
        if event.kind != "done":
            raise RuntimeError(f"{handler.name}: {event}")
        times.append(event.microseconds)
    return times


def check_exact(session, run, plan, args):
    """Small integers (x in -1, 0, 1; w in -1, 0, 0, 0, 1): every sum exact in fp32 and
    under 256, so exact in bfloat16 too. Returns what reads Y after the run."""
    rng = np.random.default_rng(args.seed)
    m, k, n = args.rows * 32, K_TILES * 32, N_TILES * 32
    x = rng.integers(-1, 2, (m, k)).astype(np.float32)
    w = rng.choice(np.array([-1, 0, 0, 0, 1], np.float32), (k, n))
    tx, tw = (session.tensor(v, "bfloat16", "tile") for v in (x, w))
    ty = session.tensor(np.zeros((m, n), np.float32), "bfloat16", "tile")
    handler, group = plan
    times = timed(run, handler, group, (tx, tw, ty), 1)

    def finish():
        got = ty.read().astype(np.float64)
        want = x.astype(np.float64) @ w.astype(np.float64)
        return {
            "microseconds": times,
            "exact": bool((got == want).all()),
            "wrong": int((got != want).sum()),
            "largest_sum": float(np.abs(want).max()),
            "sums_exact_in_bf16": bool(np.abs(want).max() <= 256),
        }

    return finish


def check_random(session, run, plan, args, swiglu):
    """Normal values in bfloat16: Y (bfloat16) against float64 of the same inputs, as
    the dense project judges (relative RMS); `swiglu`: two products and the gate."""
    rng = np.random.default_rng(args.seed)
    m, k, n = args.rows * 32, K_TILES * 32, N_TILES * 32
    x = bf16(rng.standard_normal((m, k)))
    weights = [
        bf16(rng.standard_normal((k, n)) / 64) for _ in range(2 if swiglu else 1)
    ]
    tx = session.tensor(x, "bfloat16", "tile")
    tws = [session.tensor(w, "bfloat16", "tile") for w in weights]
    tys = [
        session.tensor(np.zeros((m, n), np.float32), "bfloat16", "tile") for _ in tws
    ]
    out = session.tensor(np.zeros((m, n), np.float32), "bfloat16", "tile")
    handler, group = plan
    record = {}
    times = timed(run, handler, group, (tx, tws[0], tys[0]), args.repeat)
    record["microseconds"] = times
    record["median_microseconds"] = float(np.median(times))
    if swiglu:
        timed(run, handler, group, (tx, tws[1], tys[1]), 1)
        cores = args.gate_cores
        pages = m * n // 1024 // cores
        submitted = run.submit(
            Linear.gate, (0, cores), args=(tys[0], tys[1], out, pages)
        )
        event = run.wait(submitted, timeout=30.0)
        record["gate_microseconds"] = event.microseconds

    def finish():
        want = x @ weights[0]
        got = tys[0].read().astype(np.float64)
        record["relative_rms"] = rms(got, want)
        if swiglu:
            h1, h3 = (t.read().astype(np.float64) for t in tys)
            gate = h1 / (1 + np.exp(-h1)) * h3  # of the card's bfloat16 products
            whole = want / (1 + np.exp(-want)) * (x @ weights[1])
            got = out.read().astype(np.float64)
            record["gate_relative_rms"] = rms(got, gate)
            record["swiglu_relative_rms"] = rms(got, whole)
        return record

    return finish


def rms(got, want):
    """The relative RMS of `got` against `want`."""
    return float(np.sqrt(((got - want) ** 2).mean()) / np.sqrt((want**2).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1], help="X's row tiles (M / 32)"
    )
    parser.add_argument("--check", nargs="+", default=["exact", "random", "swiglu"])
    parser.add_argument("--repeat", type=int, default=3, help="timed commands")
    parser.add_argument("--gate-cores", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    reports, finishes, ok = [], [], True
    with km.Session(log=log, kernel_bytes=KERNEL_BYTES) as session:
        log(f"device open, grid {session.grid}; row tiles {args.rows}")
        program = made(session, args.rows)
        run = session.load(program)  # one load for every shape: the mock's build key
        try:
            for rows in args.rows:
                shape = argparse.Namespace(**{**vars(args), "rows": rows})
                handler, group = plan_of(rows)
                if handler is Linear.grid and rows != max(*args.rows, ROWS * OH):
                    raise SystemExit("the grid plan runs the largest shape of --rows")
                group = group or program.rectangle["group"]
                report = {"m_k_n": [rows * 32, K_TILES * 32, N_TILES * 32]}
                report |= {"plan": handler.name, "group": list(group)}
                report["kernel_text"] = program.sizes.text
                plan = (handler, group)
                if "exact" in args.check:
                    finishes.append(
                        (report, "exact", check_exact(session, run, plan, shape))
                    )
                for name in ("random", "swiglu"):
                    if name in args.check:
                        check = check_random(
                            session, run, plan, shape, name == "swiglu"
                        )
                        finishes.append((report, name, check))
                reports.append(report)
        finally:
            run.stop()
        for report, name, finish in finishes:
            report[name] = finish()
            log(f"{report['m_k_n'][0]} x {name}: {report[name]}")
            ok &= report[name]["exact"] if name == "exact" else True
            ok &= report[name].get("relative_rms", 0) < 1e-2
    for report in reports:
        print(json.dumps(report, default=float), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
