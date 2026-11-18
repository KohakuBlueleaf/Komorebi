"""Both cards as islands of one 1 x 2 fabric mesh in one process: the X280 runtime over
the mesh, a dispatcher on each card's X280 cluster 0 and example 01's program on each
grid; times NOP and MAP rounds on each island alone and on both at once, checks the
last MAP round of each exactly and prints one JSON line. See README.md.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.device import mock, native
from komorebi.layout import partition

KERNELS = Path(__file__).resolve().parents[3] / "examples" / "01_mapcollect" / "kernels"
TILE = 32
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--islands", type=int, default=2)
    parser.add_argument("--nop-rounds", type=int, default=5)
    parser.add_argument("--nop-jobs", type=int, default=20000, help="per round")
    parser.add_argument("--map-rounds", type=int, default=3)
    parser.add_argument("--map-jobs", type=int, default=2000, help="per round")
    parser.add_argument("--no-load", action="store_true", help="open and close only")
    return parser.parse_args()


def timed(run, handler, jobs, cores, words):
    """Microseconds per command of one round: submit_many to the whole grid, drain."""
    start = time.perf_counter()
    run.submit_many(handler, np.zeros(jobs, np.int64), np.full(jobs, cores), words)
    waited = run.drain(timeout=20.0)
    if waited != jobs:
        raise RuntimeError(f"{waited} events of {jobs}")
    return (time.perf_counter() - start) * 1e6 / jobs


class Island:
    """Example 01's program loaded on one island through a Session given its device."""

    def __init__(self, index, chip, device):
        self.index = index
        self.session = km.Session(card=chip, device=device, log=self.log)
        grid_width, grid_height = self.session.grid
        self.cores = grid_width * grid_height
        self.input_tiles = 2 * self.cores
        generator = np.random.default_rng(1 + index)
        self.exact = generator.integers(-2, 3, (self.input_tiles * TILE, TILE)).astype(
            np.float32
        )
        self.source = self.session.tensor(self.exact, "bfloat16", "tile")
        self.output = self.session.tensor(
            np.zeros((self.cores * TILE, TILE), np.float32), "bfloat16", "tile"
        )
        self.program = km.Program(
            "mapcollect",
            kernels=KERNELS,
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
                km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
            ],
            compile_args={
                "reader": self.source.accessor_args(),
                "writer": self.output.accessor_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            handlers=KERNELS / "mapcollect.h",
        )
        self.run = None
        self.first_tiles = None

    def log(self, message):
        log(f"island {self.index}: {message}")

    def load(self):
        self.run = self.session.load(self.program, deadline=1.0)
        self.log(f"loaded: {self.cores} cores")

    def nop(self, rounds, jobs):
        return [timed(self.run, 0, jobs, self.cores, None) for _ in range(rounds)]

    def map(self, rounds, jobs):
        index = np.arange(jobs)
        self.first_tiles = (index + 7 * self.index) % (
            self.input_tiles - self.cores + 1
        )
        words = np.stack(
            [
                np.full(jobs, self.source.address),
                np.full(jobs, self.cores),  # tiles: one a core
                np.full(jobs, self.output.address),
                np.zeros(jobs, np.int64),  # first_page
                np.zeros(jobs, np.int64),  # slow_cycles
                np.full(jobs, 0xFFFFFFFF),  # slow_rank: none
                self.first_tiles,
            ]
        ).T
        handler = self.program.handler("map")
        return [
            timed(self.run, handler, jobs, self.cores, words) for _ in range(rounds)
        ]

    def stop_and_check(self):
        """Stops the run, then the last MAP round's tiles against the host's sums."""
        self.run.stop()
        result = self.run.result()
        (got,) = self.session.read_bf16(self.output)
        got = got.reshape(self.cores, TILE, TILE)
        tiles = self.exact.reshape(-1, TILE, TILE).astype(np.float64)
        last = int(self.first_tiles[-1])
        want = np.stack(
            [
                tiles[last:][partition(range(self.cores), rank, self.cores)].sum(0)
                for rank in range(self.cores)
            ]
        )
        return {
            "kernel_error": result["error_name"],
            "commands": result["commands"],
            "map_wrong_tiles": int((got != want).any(axis=(1, 2)).sum()),
        }


def both(islands, work):
    """Runs work(island) on every island at once, one thread each; their results."""
    out, errors = [None] * len(islands), []

    def one(island):
        try:
            out[island.index] = work(island)
        except BaseException as error:  # noqa: BLE001 (raised again below)
            errors.append(error)

    threads = [threading.Thread(target=one, args=(island,)) for island in islands]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60.0)
    if errors or any(thread.is_alive() for thread in threads):
        raise RuntimeError(f"a thread failed or did not end: {errors}")
    return out


def median(values):
    return round(float(np.median(values)), 3)


def main():
    args = parse()
    if mock.active():  # it opens the mesh and the X280 runtime itself, not a session
        log("card only: it drives the native islands, which a prebuild has not")
        sys.exit(1)
    report = {"islands": args.islands}
    module = native.load_module()
    log(f"opening {args.islands} islands as one 1 x {args.islands} fabric mesh")
    mesh = module.Islands(args.islands)
    report["chips"] = mesh.chips()
    log(f"mesh open: chips {report['chips']}")
    islands = []
    try:
        mesh.open_x280()
        log("X280 runtime open over the mesh")
        if not args.no_load:
            islands = [
                Island(index, chip, mesh.device(index))
                for index, chip in enumerate(report["chips"])
            ]
            for island in islands:
                island.load()
            nops = args.nop_rounds, args.nop_jobs
            report["nop_alone_us"] = [median(island.nop(*nops)) for island in islands]
            log(f"NOP alone: {report['nop_alone_us']}")
            report["nop_both_us"] = [
                median(value) for value in both(islands, lambda one: one.nop(*nops))
            ]
            log(f"NOP both: {report['nop_both_us']}")
            maps = args.map_rounds, args.map_jobs
            report["map_alone_us"] = [median(island.map(*maps)) for island in islands]
            log(f"MAP alone: {report['map_alone_us']}")
            report["map_both_us"] = [
                median(value) for value in both(islands, lambda one: one.map(*maps))
            ]
            log(f"MAP both: {report['map_both_us']}")
            report["checks"] = [island.stop_and_check() for island in islands]
            log(f"checks: {report['checks']}")
    finally:
        for island in islands:
            if island.run is not None and not island.run.stopped:
                island.run.abort()
            island.session.close()
        log("closing the mesh")
        mesh.close()
        log("mesh closed")
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    checks = report.get("checks", [])
    ok = all(
        check["kernel_error"] == "none" and check["map_wrong_tiles"] == 0
        for check in checks
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
