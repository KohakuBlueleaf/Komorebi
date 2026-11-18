"""What a loop of products that waits for its own reads costs.

The shape of a 3x3 convolution's shifts and taps: k loops in a row, each a stream of 2
tiles of x by a stream of 2 tiles of a 0/1 matrix, summed in Dst (2 tile products) and
written as one tile. One class at HiFi4 (the shifts' fidelity) and one at HiFi2 (the
convolution's); k = 1, 2, 4, 8 loops, on one core and on 36 (the shift's group), the
event's microseconds a command.

    python taps.py --repeats 5
"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km

STARTED = time.monotonic()
COUNTS = (1, 2, 4, 8)


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def make(fidelity):
    class Taps(km.Program):
        @km.tile_function(fidelity=fidelity)
        def product(a, b):
            return a @ b

        @km.handler
        def taps1(self, x: km.Tensor, s: km.Tensor, z: km.Tensor):
            for j in range(1):
                acc = km.tile.zeros(km.fp32, 1)
                for xb, sb in zip(
                    km.Stream(x.tiles[self.rank : self.rank + 1, 0:2], block=2),
                    km.Stream(s.tiles[0:2, 0:1], block=2),
                ):
                    acc += self.product(xb, sb)
                z.tiles[self.rank : self.rank + 1, j : j + 1].write(acc)

        @km.handler
        def taps2(self, x: km.Tensor, s: km.Tensor, z: km.Tensor):
            for j in range(2):
                acc = km.tile.zeros(km.fp32, 1)
                for xb, sb in zip(
                    km.Stream(x.tiles[self.rank : self.rank + 1, 0:2], block=2),
                    km.Stream(s.tiles[0:2, 0:1], block=2),
                ):
                    acc += self.product(xb, sb)
                z.tiles[self.rank : self.rank + 1, j : j + 1].write(acc)

        @km.handler
        def taps4(self, x: km.Tensor, s: km.Tensor, z: km.Tensor):
            for j in range(4):
                acc = km.tile.zeros(km.fp32, 1)
                for xb, sb in zip(
                    km.Stream(x.tiles[self.rank : self.rank + 1, 0:2], block=2),
                    km.Stream(s.tiles[0:2, 0:1], block=2),
                ):
                    acc += self.product(xb, sb)
                z.tiles[self.rank : self.rank + 1, j : j + 1].write(acc)

        @km.handler
        def taps8(self, x: km.Tensor, s: km.Tensor, z: km.Tensor):
            for j in range(8):
                acc = km.tile.zeros(km.fp32, 1)
                for xb, sb in zip(
                    km.Stream(x.tiles[self.rank : self.rank + 1, 0:2], block=2),
                    km.Stream(s.tiles[0:2, 0:1], block=2),
                ):
                    acc += self.product(xb, sb)
                z.tiles[self.rank : self.rank + 1, j : j + 1].write(acc)

    return Taps


CLASSES = {"HiFi4": make("HiFi4"), "HiFi2": make("HiFi2")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cores", type=int, nargs="+", default=[1, 36])
    parser.add_argument("--save", help="the runs' microseconds as JSON")
    args = parser.parse_args()
    rng = np.random.default_rng(1)
    ranks = max(args.cores)
    x = rng.uniform(-1.0, 1.0, (ranks * 32, 64)).astype(np.float32)
    s = rng.uniform(-1.0, 1.0, (64, 32)).astype(np.float32)
    runs = []
    with km.Session(log=log) as session:
        tx = session.tensor(x, "bfloat16", "tile")
        ts = session.tensor(s, "bfloat16", "tile")
        tz = session.empty((ranks * 32, 32 * max(COUNTS)), "bfloat16", "tile")
        for fidelity, cls in CLASSES.items():
            run = session.load(cls(x=tx, s=ts, z=tz, session=session, log=log))
            try:
                for count in COUNTS:
                    handler = getattr(cls, f"taps{count}")
                    for cores in args.cores:
                        micro = []
                        for _ in range(args.repeats + 1):
                            event = run.wait(
                                run.submit(handler, (0, cores), args=(tx, ts, tz)),
                                timeout=30.0,
                            )
                            if event.kind != "done":
                                raise RuntimeError(f"{handler.name}: {event}")
                            micro.append(round(event.microseconds, 2))
                        runs.append(
                            {
                                "fidelity": fidelity,
                                "loops": count,
                                "cores": cores,
                                "us": micro,
                            }
                        )
                        log(json.dumps(runs[-1]))
            finally:
                run.stop()
    if args.save:
        with open(args.save, "w") as out_file:
            json.dump(runs, out_file)
    print(json.dumps({"runs": len(runs)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
