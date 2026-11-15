"""Y = X W as a class in the dense project's two plans, picked by the shape as its plan
picker does (the K split for one row tile of X, the grid plan for more), and a SwiGLU
gate after it; against the host's product (build/linear-<hash>/ here). README.md.
Example: python linear_class.py --rows 1 32 --check exact swiglu"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km

STARTED = time.monotonic()
# The grid plan's rows of cores (the rectangle's, the host's) and OH, the block's rows
# at M = 1024: the class's symbol, the plan gives the shape's.
ROWS = 8
OH = 4


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

    K_TILES = km.constant(128)  # K = 4096
    N_TILES = km.constant(128)  # N = 4096
    # The K split (the dense project's 14,1,16): groups of SPLIT cores, a group WIDTH
    # columns of Y, a core 1/SPLIT of K; X and W stream in step, CHUNK K tiles a block.
    SPLIT = km.constant(14)
    WIDTH = km.constant(16)
    SPLIT_CORES = N_TILES // WIDTH * SPLIT  # 112
    CHUNK = km.constant(2)
    W_BLOCK = CHUNK * WIDTH  # W's tiles a block: CHUNK rows of the group's columns
    # The grid on COLUMNS x ROWS cores: core (x, y) Y's block (y, x) of OH x OW tiles,
    # the last column 7 wide; K chunks of KC tiles cast, X's along a row, W's down one.
    COLUMNS = km.constant(12)
    KC = km.constant(2)  # at 4 the casts' slots and the K split's are over a core's L1
    OW = -(-N_TILES // COLUMNS)  # 11
    GATE_BLOCK = km.constant(2)  # the gate's tiles a block
    OH = km.symbol()  # the plan's
    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]
    part: km.L1[km.fp32, OH, OW, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def split(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        share = self.rank % self.SPLIT  # this core's part of K
        first = self.rank // self.SPLIT * self.WIDTH  # the group's columns of Y
        k0 = share * self.K_TILES // self.SPLIT
        k1 = (share + 1) * self.K_TILES // self.SPLIT
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=self.CHUNK),
            km.Stream(w.tiles[k0:k1, first : first + self.WIDTH], block=self.W_BLOCK),
        ):
            self.sums += self.project(xb, wb)
        whole = km.reduce(self.sums, form="tree", group=self.SPLIT)
        if self.rank == self.leader:
            y.tiles[0:1, first : first + self.WIDTH].write(whole)

    @km.handler
    def grid(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        along = km.Broadcast(self, along="row", window=2)
        down = km.Broadcast(self, along="column", window=2)
        top = self.y * self.OH
        left = self.x * self.OW
        for chunk in range(self.K_TILES // self.KC):
            k0 = chunk * self.KC
            if along.sender():
                xs = along.cast(x.tiles[top : top + self.OH, k0 : k0 + self.KC].read())
            else:
                xs = along.arrive()
            if down.sender():
                ws = down.cast(
                    w.tiles[
                        k0 : k0 + self.KC, left : min(left + self.OW, self.N_TILES)
                    ].read()
                )
            else:
                ws = down.arrive()
            self.part += self.project(xs, ws)
        y.tiles[top : top + self.OH, left : min(left + self.OW, self.N_TILES)].write(
            self.part
        )

    @km.handler
    def gate(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=self.GATE_BLOCK)
        for h, g in zip(
            km.Stream(a.pages(first, tiles), block=self.GATE_BLOCK),
            km.Stream(b.pages(first, tiles), block=self.GATE_BLOCK),
        ):
            results.push(silu_gate(h, g))


def bf16(values):
    """float32 values rounded to bfloat16 (to nearest even), as float64."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32).astype(np.float64)


def plan_of(program, rows):
    """The plan picker: the K split for one row tile, else the grid plan; (the
    program's handler, its group or None for the rectangle's)."""
    if rows == 1:
        return program.split, (0, program.SPLIT_CORES)
    if rows % ROWS:
        raise SystemExit(f"the grid plan takes row tiles in multiples of {ROWS}")
    return program.grid, None


def made(session, rows, k_tiles, hints=None, least=ROWS * OH):
    """The class made with K_TILES `k_tiles` and tensors of the largest shape of
    `rows` (X [M, K], W [K, N], Y [M, N]), each with its island hints of `hints`; the
    grid's block of `least` rows when none is larger (one load for both)."""
    rows = max(*rows, least)
    m, k, n = rows * 32, k_tiles * 32, Linear.N_TILES.default * 32
    hints = hints or {}
    tensors = {
        name: session.empty(shape, "bfloat16", "tile", **hints.get(name, {}))
        for name, shape in (("x", (m, k)), ("w", (k, n)), ("y", (m, n)))
    }
    for name in ("a", "b", "out"):  # Y's shape: two cards hold one, their table 8
        tensors[name] = (
            tensors["y"] if hints else session.empty((m, n), "bfloat16", "tile")
        )
    oh = rows // ROWS
    program = Linear(
        log=log,
        plan={"OH": oh},
        K_TILES=k_tiles,
        rect=(0, 0, Linear.COLUMNS.default, ROWS),
        session=session,
        **tensors,
    )
    return program


def timed(run, handler, group, args, repeat, parts=None):
    """`repeat` commands of the handler; their events' microseconds (two islands: the
    slower part's, each part's appended to `parts`)."""
    times = []
    for _ in range(repeat):
        submitted = run.submit(handler, group, args=args)
        event = run.wait(submitted, timeout=30.0)
        each = (event,) if hasattr(event, "kind") else event  # an island's part each
        if any(part.kind != "done" for part in each):
            raise RuntimeError(f"{handler.method.name}: {event}")
        times.append(max(part.microseconds for part in each))
        if parts is not None:
            parts.append([part.microseconds for part in each])
    return times


def check_exact(session, run, plan, args):
    """Small integers (x in -1, 0, 1; w in -1, 0, 0, 0, 1): every sum exact in fp32 and
    under 256, so exact in bfloat16 too. Returns what reads Y after the run."""
    rng = np.random.default_rng(args.seed)
    program = plan[0].program
    m, k, n = args.rows * 32, program.K_TILES * 32, program.N_TILES * 32
    x = rng.integers(-1, 2, (m, k)).astype(np.float32)
    w = rng.choice(np.array([-1, 0, 0, 0, 1], np.float32), (k, n))
    hints = args.hints
    tx = session.tensor(x, "bfloat16", "tile", **hints.get("x", {}))
    tw = session.tensor(w, "bfloat16", "tile", **hints.get("w", {}))
    ty = session.tensor(
        np.zeros((m, n), np.float32), "bfloat16", "tile", **hints.get("y", {})
    )
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

    return now_if_islands(finish, args, [tx, tw, ty])


def check_random(session, run, plan, args, swiglu):
    """Normal values in bfloat16: Y (bfloat16) against float64 of the same inputs, as
    the dense project judges (relative RMS); `swiglu`: two products and the gate."""
    rng = np.random.default_rng(args.seed)
    program = plan[0].program
    m, k, n = args.rows * 32, program.K_TILES * 32, program.N_TILES * 32
    x = bf16(rng.standard_normal((m, k)))
    weights = [
        bf16(rng.standard_normal((k, n)) / 64) for _ in range(2 if swiglu else 1)
    ]
    hints = args.hints
    tx = session.tensor(x, "bfloat16", "tile", **hints.get("x", {}))
    tws = [session.tensor(w, "bfloat16", "tile", **hints.get("w", {})) for w in weights]
    tys = [
        session.tensor(
            np.zeros((m, n), np.float32), "bfloat16", "tile", **hints.get("y", {})
        )
        for _ in tws
    ]
    out = session.tensor(np.zeros((m, n), np.float32), "bfloat16", "tile")
    handler, group = plan
    record, parts = {}, []
    times = timed(run, handler, group, (tx, tws[0], tys[0]), args.repeat, parts)
    record["microseconds"] = times
    if len(parts[0]) > 1:
        record["island_microseconds"] = parts
        record |= w_held(tws[0], program)
    record["median_microseconds"] = float(np.median(times))
    if swiglu:
        timed(run, handler, group, (tx, tws[1], tys[1]), 1)
        cores = args.gate_cores
        pages = m * n // 1024 // cores
        submitted = run.submit(
            program.gate, (0, cores), args=(tys[0], tys[1], out, pages)
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

    return now_if_islands(finish, args, [tx, *tws, *tys, out])


def w_held(w, program):
    """W's bytes each island holds (its part of the columns) and one core of the K
    split reads (its K rows of its group's WIDTH columns)."""
    page = w.aligned_page_bytes
    per_core = -(-program.K_TILES // program.SPLIT) * program.WIDTH * page
    return {
        "w_bytes_an_island": [
            w.placement.part_pages(island) * page
            for island in range(len(w.placement.parts))
        ],
        "w_bytes_a_core": per_core,
    }


def islands_args(session, args):
    """Two cards (the owner's form, the output dim sharded): the split plan's group of
    SPLIT_CORES ranks centred on island 1's first rank, so each island runs half the
    groups; X a copy on both, W and Y split by their tile columns over that group."""
    if args.rows != [1]:
        raise SystemExit("two cards: the split plan only (--rows 1)")
    edge = session.graph.first(1)
    cores = Linear.N_TILES.default // Linear.WIDTH.default * Linear.SPLIT.default
    group = (edge - cores // 2, cores)
    split = {"split": 1, "group": group}
    return {"x": {"copy": True}, "w": split, "y": split}, group


def now_if_islands(finish, args, tensors):
    """`finish`, or on two cards its result read now and the tensors freed: a run's
    tensor table holds 8 entries."""
    if not args.hints:
        return finish
    result = finish()
    for tensor in tensors:
        tensor.free()
    return lambda: result


def rms(got, want):
    """The relative RMS of `got` against `want`."""
    return float(np.sqrt(((got - want) ** 2).mean()) / np.sqrt((want**2).mean()))


def shapes_run(session, program, args):
    """Each shape of `args.rows` by the plan picker on one load of `program` (one
    load for every shape: the mock's build key), its checks read after the run;
    (the reports, whether all were right)."""
    reports, finishes, ok = [], [], True
    run = session.load(program)
    for spare in args.spares:  # two cards: the table sized at the load has room now
        spare.free()
    try:
        for rows in args.rows:
            shape = argparse.Namespace(**{**vars(args), "rows": rows})
            handler, group = plan_of(program, rows)
            if handler.method is Linear.grid and rows != max(*args.rows, args.block):
                raise SystemExit("the grid plan runs the largest shape of --rows")
            group = args.group or group or program.rectangle["group"]
            report = {"m_k_n": [rows * 32, program.K_TILES * 32, program.N_TILES * 32]}
            report |= {"plan": handler.method.name, "group": list(group)}
            report["kernel_text"] = program.sizes.text
            plan = (handler, group)
            if "exact" in args.check:
                finishes.append(
                    (report, "exact", check_exact(session, run, plan, shape))
                )
            for name in ("random", "swiglu"):
                if name in args.check:
                    check = check_random(session, run, plan, shape, name == "swiglu")
                    finishes.append((report, name, check))
            reports.append(report)
    finally:
        run.stop()
    for report, name, finish in finishes:
        report[name] = finish()
        log(f"{report['m_k_n'][0]} x {name}: {report[name]}")
        ok &= report[name]["exact"] if name == "exact" else True
        ok &= report[name].get("relative_rms", 0) < 1e-2
    return reports, ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1], help="X's row tiles (M / 32)"
    )
    parser.add_argument("--check", nargs="+", default=["exact", "random", "swiglu"])
    parser.add_argument("--repeat", type=int, default=3, help="timed commands")
    parser.add_argument("--gate-cores", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cards", default="", help="0,1: two cards (the split plan)")
    parser.add_argument(
        "--block", type=int, default=ROWS * OH, help="the grid plan's rows (its field)"
    )
    parser.add_argument(
        "--k-tiles",
        type=int,
        nargs="+",
        default=[Linear.K_TILES.default],
        help="K / 32: a program of the class each, loaded one after the other",
    )
    args = parser.parse_args()
    reports, ok = [], True
    cards = [int(card) for card in args.cards.split(",")] if args.cards else []
    opened = {"cards": cards} if len(cards) > 1 else {}
    with km.Session(log=log, **opened) as session:
        log(f"device open, grid {session.grid}; row tiles {args.rows}")
        args.hints, args.group = islands_args(session, args) if opened else ({}, None)
        # a run's tensor table is sized at its load: entries for the checks' tensors
        args.spares = [
            session.empty((32, 32), "bfloat16") for _ in range(6 * bool(opened))
        ]
        for k_tiles in args.k_tiles:
            program = made(session, args.rows, k_tiles, args.hints, args.block)
            found, ok_k = shapes_run(session, program, args)
            reports += found
            ok &= ok_k
    for report in reports:
        print(json.dumps(report, default=float), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
