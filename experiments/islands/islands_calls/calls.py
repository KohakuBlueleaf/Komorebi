"""One card's calls on islands, by value: DOT commands whose block every island's part
reads (example 01's kernels), each island's X280 threads by flat number, and a turned
product whose W every island stores with transposed tiles. One JSON line; exits 1
when a check fails. See README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[2] / "examples" / "01_mapcollect" / "kernels"
TILE = 32
K_TILES = 8  # X's columns and W's rows of tiles
N_TILES = 4  # W's and Y's columns of tiles: a core a column, a row an island
CLUSTERS = (1, 2)  # each island's threads
CODE = 7  # thread/island.c's result code
TURNED_RMS = 2e-2  # bound: 1.7e-3 measured on one card (formats README); wrong tiles ~1
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--jobs", type=int, default=32, help="DOT commands")
    parser.add_argument("--words", type=int, default=64, help="words a block")
    return parser.parse_args()


class Turned(km.Program):
    """Y = X W with W in SrcB (turned): core `rank` of the group gives Y's tile
    (rank // N_TILES, rank % N_TILES), so each island's cores a row of Y."""

    sums: km.L1[km.fp32, 1, 1, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def turned(x, w):
        return x @ km.srcb(w)

    @km.handler
    def apply(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        row = self.rank // N_TILES
        column = self.rank % N_TILES
        for xb, wb in zip(
            km.Stream(x.tiles[row : row + 1, 0:K_TILES], block=2),
            km.Stream(w.tiles[0:K_TILES, column : column + 1], block=2),
        ):
            self.sums += self.turned(xb, wb)
        y.tiles[row : row + 1, column : column + 1].write(self.sums)


def mapcollect(tiles):
    """Example 01's kernels; island.c on CLUSTERS of every island."""
    made = km.Program(
        "mapcollect",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={
            "reader": tiles.accessor_args(),
            "writer": tiles.accessor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=KERNELS / "mapcollect.h",
    )
    made.thread(
        clusters=CLUSTERS,
        sources=[HERE / "thread" / "island.c"],
        regions={"scratch": 64},
    )
    return made


def dot(weights, x0):
    """sum of w[i] * (x0 + i), mod 2^32, as the writer computes it."""
    x = (x0 + np.arange(len(weights), dtype=np.uint64)) & 0xFFFFFFFF
    return int((weights.astype(np.uint64) * x & 0xFFFFFFFF).sum() & 0xFFFFFFFF)


def blocks(run, group, args):
    """DOT commands on `group` (both islands), each with its own block: every part's
    reader checks the block's sum (it gives up otherwise), the writer gives the dot."""
    generator = np.random.default_rng(1)
    tables = generator.integers(0, 1 << 32, (args.jobs, args.words), dtype=np.uint64)
    tables = tables.astype(np.uint32)
    x0s = generator.integers(0, 1 << 32, args.jobs, dtype=np.uint64)
    sums = [int(table.astype(np.uint64).sum() & 0xFFFFFFFF) for table in tables]
    handles = [
        run.submit(
            "dot", group, [int(x0s[job]), sums[job]], result=True, block=tables[job]
        )
        for job in range(args.jobs)
    ]
    events = run.wait(handles, timeout=30.0)
    wants = [dot(tables[job], int(x0s[job])) for job in range(args.jobs)]
    return {
        "dot_parts": sorted({len(event.parts) for event in events}),
        "dot_not_done": sum(event.kind != "done" for event in events),
        "dot_wrong": sum(event.result != want for event, want in zip(events, wants)),
        "dot_parts_wrong": sum(
            part.result != want
            for event, want in zip(events, wants)
            for part in event.parts
        ),
    }


def threads(run):
    """run.thread_results() (every island's, by flat number), then
    run.thread_result() (the lowest number's): each thread's words name its island,
    cluster and number."""
    results = dict(run.thread_results(timeout=10.0))  # a mock run's: none
    first = run.thread_result(timeout=10.0)  # the same, already collected
    if not results:
        return {"thread_numbers": [], "threads_right": False}
    return {
        "thread_numbers": sorted(results),
        "thread_first": [first.code, *first.words],
        "thread_words": {
            str(number): [result.state, result.code, *result.words]
            for number, result in results.items()
        },
        "threads_right": all(
            result.state == "done"
            and result.code == CODE
            and result.words[2] == number
            and number == 3 * result.words[0] + result.words[1]
            for number, result in results.items()
        )
        and first is results[min(results)],
    }


def turned(run, x, w, y, group, w_held):
    """One APPLY on `group` (a row of Y an island), the run stopped; then Y against
    float64 of X and W as held (`w_held`), and W read back against it."""
    event = run.wait(run.submit(Turned.apply, group, args=(x, w, y)), timeout=30.0)
    stopped = run.stop()
    log(f"APPLY {event.kind}; stopped; reading X, W and Y")
    want = x.read().astype(np.float64) @ w_held.astype(np.float64)
    got = y.read().astype(np.float64)
    scale = np.sqrt((want**2).mean())
    return {
        "turned_kind": event.kind,
        "turned_parts": len(event.parts),
        "turned_kernel_errors": [part["error_name"] for part in stopped.islands],
        "turned_rms_rel": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "turned_max_over_rms": float(np.abs(got - want).max() / scale),
        "w_read_back_exact": bool((np.asarray(w.read()) == w_held).all()),
    }


def bf16(values):
    """`values` (float32) as bfloat16 holds them (nearest, ties to even), as float32."""
    bits = values.view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "jobs": args.jobs, "words": args.words}
    with km.Session(cards=cards, log=log) as session:
        graph = session.graph
        # the group: island 0's last N_TILES ranks and island 1's first N_TILES
        group = (graph.first(1) - N_TILES, 2 * N_TILES)
        report["group"] = list(group)
        tiles = session.tensor(np.zeros((TILE, TILE), np.float32), "bfloat16", "tile")
        run = session.load(mapcollect(tiles), deadline=2.0)
        log(f"loaded mapcollect with threads on clusters {CLUSTERS}; DOT on {group}")
        report |= blocks(run, group, args)
        log(f"DOT: {report['dot_not_done']} not done, {report['dot_wrong']} wrong")
        report |= threads(run)
        log(f"threads: {report['thread_numbers']} right {report['threads_right']}")
        stopped = run.stop()
        report["kernel_errors"] = [part["error_name"] for part in stopped.islands]
        log(f"stopped: {report['kernel_errors']}")

        generator = np.random.default_rng(3)
        x_values = generator.standard_normal((2 * TILE, K_TILES * TILE))
        w_values = generator.standard_normal((K_TILES * TILE, N_TILES * TILE)) / 32
        w_held = bf16(w_values.astype(np.float32))
        x = session.tensor(x_values.astype(np.float32), "bfloat16", "tile", split=0)
        w = session.tensor(w_held, "bfloat16", "tile", copy=True, transposed_tiles=True)
        y = session.empty((2 * TILE, N_TILES * TILE), "float32", "tile", split=0)
        try:
            w.transposed_tiles = False
            report["turned_after_data"] = "taken"
        except km.KomorebiError:
            report["turned_after_data"] = "refused"
        program = Turned(log=log, session=session, x=x, w=w, y=y)
        run = session.load(program, deadline=2.0)
        log(f"loaded Turned; APPLY on {group}")
        report |= turned(run, x, w, y, group, w_held)
        log(f"APPLY: rms {report['turned_rms_rel']:.3e}")
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        report["dot_parts"] == [2]
        and report["dot_not_done"] == 0
        and report["dot_wrong"] == 0
        and report["threads_right"]
        and report["thread_numbers"] == [1, 2, 4, 5]
        and report["kernel_errors"] == ["none", "none"]
        and report["turned_after_data"] == "refused"
        and report["turned_kind"] == "done"
        and report["turned_parts"] == 2
        and report["turned_rms_rel"] < TURNED_RMS
        and report["w_read_back_exact"]
        and report["turned_kernel_errors"] == ["none", "none"]
    )
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
