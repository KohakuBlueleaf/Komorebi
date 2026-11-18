"""One card's side of two_processes.py: loads example 01's program on the visible card,
prints READY, waits (bounded) for a line on stdin, then times rounds of NOP and MAP
commands to the whole grid, checks the last MAP round exactly and prints one JSON line.
"""

import argparse
import json
import select
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.layout import partition

KERNELS = Path(__file__).resolve().parents[3] / "examples" / "01_mapcollect" / "kernels"
TILE = 32
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nop-rounds", type=int, default=10)
    parser.add_argument("--nop-jobs", type=int, default=20000, help="per round")
    parser.add_argument("--map-rounds", type=int, default=10)
    parser.add_argument("--map-jobs", type=int, default=200, help="per round")
    parser.add_argument("--go-timeout", type=float, default=30.0, help="seconds")
    return parser.parse_args()


def timed(run, handler, jobs, cores, words):
    """Microseconds per command of one round: submit_many to the whole grid, drain."""
    start = time.perf_counter()
    run.submit_many(handler, np.zeros(jobs, np.int64), np.full(jobs, cores), words)
    waited = run.drain(timeout=20.0)
    if waited != jobs:
        raise RuntimeError(f"{waited} events of {jobs}")
    return (time.perf_counter() - start) * 1e6 / jobs


def main():
    args = parse()
    report = {}
    with km.Session(log=log) as session:
        grid_width, grid_height = session.grid
        cores = grid_width * grid_height
        input_tiles = 2 * cores
        generator = np.random.default_rng(1)
        exact = generator.integers(-2, 3, (input_tiles * TILE, TILE)).astype(np.float32)
        source = session.tensor(exact, "bfloat16", "tile")
        output = session.tensor(np.zeros((cores * TILE, TILE), np.float32),
                                "bfloat16", "tile")  # fmt: skip
        program = km.Program(
            "mapcollect",
            kernels=KERNELS,
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
                km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
            ],
            compile_args={
                "reader": source.accessor_args(),
                "writer": output.accessor_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            handlers=KERNELS / "mapcollect.h",
        )
        run = session.load(program, deadline=1.0)
        log(f"loaded: {cores} cores")
        print("READY", flush=True)
        ready, _, _ = select.select([sys.stdin], [], [], args.go_timeout)
        if not ready or not sys.stdin.readline():
            raise RuntimeError(f"no go line in {args.go_timeout} s")
        log("go")
        window = [time.time()]
        nop = [timed(run, 0, args.nop_jobs, cores, None)
               for _ in range(args.nop_rounds)]  # fmt: skip
        log(f"{args.nop_rounds} NOP rounds")
        jobs = np.arange(args.map_jobs)
        first_tiles = jobs % (input_tiles - cores + 1)
        words = np.stack(
            [
                np.full(args.map_jobs, source.address),
                np.full(args.map_jobs, cores),  # tiles: one a core
                np.full(args.map_jobs, output.address),
                np.zeros(args.map_jobs, np.int64),  # first_page
                np.zeros(args.map_jobs, np.int64),  # slow_cycles
                np.full(args.map_jobs, 0xFFFFFFFF),  # slow_rank: none
                first_tiles,
            ]
        ).T
        mapping = [timed(run, program.handler("map"), args.map_jobs, cores, words)
                   for _ in range(args.map_rounds)]  # fmt: skip
        window.append(time.time())
        log(f"{args.map_rounds} MAP rounds")
        run.stop()
        result = run.result()
        (got,) = session.read_bf16(output)
        got = got.reshape(cores, TILE, TILE)
        tiles = exact.reshape(-1, TILE, TILE).astype(np.float64)
        last = int(first_tiles[-1])
        want = np.stack(
            [
                tiles[last:][partition(range(cores), rank, cores)].sum(0)
                for rank in range(cores)
            ]
        )
        report = {
            "cores": cores,
            "kernel_error": result["error_name"],
            "commands": result["commands"],
            "nop_us": [round(value, 3) for value in nop],
            "map_us": [round(value, 3) for value in mapping],
            "map_wrong_tiles": int((got != want).any(axis=(1, 2)).sum()),
            "zeros": not got.any(),  # every word of the output read zero
            "window": window,  # host time (s) of the go and of the last round's end
        }
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    ok = report["kernel_error"] == "none" and report["map_wrong_tiles"] == 0
    if not ok and report.get("zeros"):
        raise SystemExit("child.py: no result words (reads were zeros)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
