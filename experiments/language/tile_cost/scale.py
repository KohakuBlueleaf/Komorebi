"""What a command of one tile function costs against its tiles a core: a scale, exp,
SiLU and an add over DRAM streams in blocks of 4 (README.md says how it is read).
    python scale.py --cores 1 8 36 --tiles 4 8 16 32 64 --repeats 5
"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km

STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function(tolerance=1e-2)
def scaled(a):
    return a * 2.0


@km.tile_function(tolerance=1e-2)
def exponential(a):
    return km.exp(a)


@km.tile_function(tolerance=1e-2)
def silu(a):
    return km.silu(a)


@km.tile_function(tolerance=1e-2)
def added(a, b):
    return a + b


class Stream(km.Program):
    """Each core takes `tiles` pages from rank x tiles."""

    @km.handler
    def scale_layer(self, a: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x in km.Stream(a.pages(first, tiles), block=4):
            results.push(scaled(x))

    @km.handler
    def exp_layer(self, a: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x in km.Stream(a.pages(first, tiles), block=4):
            results.push(exponential(x))

    @km.handler
    def silu_layer(self, a: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x in km.Stream(a.pages(first, tiles), block=4):
            results.push(silu(x))

    @km.handler
    def add_layer(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, tiles: int):
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=4)
        for x, y in zip(
            km.Stream(a.pages(first, tiles), block=4),
            km.Stream(b.pages(first, tiles), block=4),
        ):
            results.push(added(x, y))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, nargs="+", default=[1, 8, 36])
    parser.add_argument("--tiles", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="the tensors'",
    )
    parser.add_argument("--save", help="the runs' microseconds as JSON")
    args = parser.parse_args()
    most = max(c * t for c in args.cores for t in args.tiles)
    rng = np.random.default_rng(1)
    values = rng.uniform(-1.0, 1.0, (most * 32, 32)).astype(np.float32)
    runs = []
    with km.Session(log=log) as session:
        ta = session.tensor(values, args.dtype, "tile")
        tb = session.tensor(values[::-1].copy(), args.dtype, "tile")
        out = session.empty((most * 32, 32), args.dtype, "tile")
        run = session.load(Stream(a=ta, b=tb, out=out, session=session, log=log))
        try:
            for name, handler, extra in (
                ("scale", Stream.scale_layer, ()),
                ("exp", Stream.exp_layer, ()),
                ("silu", Stream.silu_layer, ()),
                ("add", Stream.add_layer, (tb,)),
            ):
                for cores in args.cores:
                    for tiles in args.tiles:
                        micro = []
                        for _ in range(args.repeats + 1):
                            event = run.wait(
                                run.submit(
                                    handler,
                                    (0, cores),
                                    args=(ta, *extra, out, tiles),
                                ),
                                timeout=30.0,
                            )
                            if event.kind != "done":
                                raise RuntimeError(f"{name}: {event}")
                            micro.append(round(event.microseconds, 2))
                        row = {
                            "handler": name,
                            "dtype": args.dtype,
                            "cores": cores,
                            "tiles": tiles,
                            "us": micro[1:],
                            "first_us": micro[0],
                        }
                        runs.append(row)
                        log(json.dumps(row))
        finally:
            run.stop()
    if args.save:
        with open(args.save, "w") as out_file:
            json.dump(runs, out_file)
    print(json.dumps({"runs": len(runs)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
