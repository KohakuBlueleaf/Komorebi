"""What a command waits for that no tile function computes.

Four small resident classes' commands, timed by their events on one core:

- `loops`: k loops in a row, each a stream of 2 bfloat16 tiles read, scaled and written
  (the shape of a conv's taps: every loop waits for its own read);
- `strided`, `half`: 16 and 8 blocks of 2 tiles, one block of every other tile row of a
  tensor 4 tiles wide (a group norm's apply without its fields);
- `apply0`, `apply1`, `apply2`: the same 16 blocks on 2 cores with no field, one field
  and two fields (fp32 tiles read by `.col()`, each filled from one DRAM page).

    python waits.py --repeats 5
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


class Loops(km.Program):
    @km.handler
    def loops(self, a: km.Tensor, out: km.Tensor, count: int):
        for i in range(count):
            first = i * 2
            results = km.Stream(out.pages(first, 2), write=True, block=2)
            for x in km.Stream(a.pages(first, 2), block=2):
                results.push(scaled(x))


class Strided(km.Program):
    """A conv's or a group norm's reads: `count` blocks of 2 tiles, one block of each
    other tile row (a tensor 4 tiles wide), scaled and written the same way."""

    @km.handler
    def strided(self, a: km.Tensor, out: km.Tensor):
        results = km.Stream(out.tiles[0:32:2, 1:3], write=True, block=2)
        for x in km.Stream(a.tiles[0:32:2, 1:3], block=2):
            results.push(scaled(x))

    @km.handler
    def half(self, a: km.Tensor, out: km.Tensor):
        results = km.Stream(out.tiles[0:16:2, 1:3], write=True, block=2)
        for x in km.Stream(a.tiles[0:16:2, 1:3], block=2):
            results.push(scaled(x))


@km.tile_function(tolerance=1e-2)
def affine0(x):
    return x * 2.0 + 1.0


@km.tile_function(tolerance=1e-2)
def affine1(x, scale):
    return x * scale.col() + 1.0


@km.tile_function(tolerance=1e-2)
def affine2(x, scale, shift):
    return x * scale.col() + shift.col()


class Apply(km.Program):
    """A group norm's apply with no field, one and two (fields of one fp32 tile)."""

    scale: km.L1[km.fp32, 1, 32, 32]
    shift: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def apply0(self, x: km.Tensor, y: km.Tensor):
        results = km.Stream(y.tiles[0:32:2, 1:3], write=True, block=2)
        for xb in km.Stream(x.tiles[0:32:2, 1:3], block=2):
            results.push(affine0(xb))

    @km.handler
    def apply1(self, x: km.Tensor, sc: km.Tensor, y: km.Tensor):
        self.scale.fill(km.Stream(sc.pages(0, 1)))
        results = km.Stream(y.tiles[0:32:2, 1:3], write=True, block=2)
        for xb in km.Stream(x.tiles[0:32:2, 1:3], block=2):
            results.push(affine1(xb, self.scale))

    @km.handler
    def apply2(self, x: km.Tensor, sc: km.Tensor, sh: km.Tensor, y: km.Tensor):
        self.scale.fill(km.Stream(sc.pages(0, 1)))
        self.shift.fill(km.Stream(sh.pages(0, 1)))
        results = km.Stream(y.tiles[0:32:2, 1:3], write=True, block=2)
        for xb in km.Stream(x.tiles[0:32:2, 1:3], block=2):
            results.push(affine2(xb, self.scale, self.shift))


def timed(run, handler, args, repeats, cores=1):
    micro = []
    for _ in range(repeats + 1):
        event = run.wait(run.submit(handler, (0, cores), args=args), timeout=30.0)
        if event.kind != "done":
            raise RuntimeError(f"{handler.name}: {event}")
        micro.append(round(event.microseconds, 2))
    return micro


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--only", nargs="*", default=["loops", "strided", "apply"], help="which parts"
    )
    parser.add_argument("--save", help="the runs' microseconds as JSON")
    args = parser.parse_args()
    rng = np.random.default_rng(1)
    most = 2 * max(args.counts)
    values = rng.uniform(-1.0, 1.0, (most * 32, 32)).astype(np.float32)
    runs = []
    with km.Session(log=log) as session:
        ta = session.tensor(values, "bfloat16", "tile")
        out = session.empty((most * 32, 32), "bfloat16", "tile")
        run = session.load(Loops(a=ta, out=out, session=session, log=log))
        try:
            for count in args.counts if "loops" in args.only else ():
                micro = timed(run, Loops.loops, (ta, out, count), args.repeats)
                runs.append({"handler": "loops", "count": count, "us": micro})
                log(json.dumps(runs[-1]))
        finally:
            run.stop()
        wide_values = rng.uniform(-1.0, 1.0, (32 * 32, 128)).astype(np.float32)
        tw = session.tensor(wide_values, "bfloat16", "tile")
        wide_out = session.empty((32 * 32, 128), "bfloat16", "tile")
        run = session.load(Strided(a=tw, out=wide_out, session=session, log=log))
        try:
            for name, handler in (
                ("strided", Strided.strided),
                ("half", Strided.half),
            )[: 2 if "strided" in args.only else 0]:
                micro = timed(run, handler, (tw, wide_out), args.repeats)
                runs.append({"handler": name, "count": 1, "us": micro})
                log(json.dumps(runs[-1]))
        finally:
            run.stop()
        field = rng.uniform(-1.0, 1.0, (64, 32)).astype(np.float32)
        tsc = session.tensor(field, "float32", "tile")
        tsh = session.tensor(field[::-1].copy(), "float32", "tile")
        run = session.load(
            Apply(x=tw, y=wide_out, sc=tsc, sh=tsh, session=session, log=log)
        )
        try:
            for name, handler, call in (
                ("apply0", Apply.apply0, (tw, wide_out)),
                ("apply1", Apply.apply1, (tw, tsc, wide_out)),
                ("apply2", Apply.apply2, (tw, tsc, tsh, wide_out)),
            )[: 3 if "apply" in args.only else 0]:
                micro = timed(run, handler, call, args.repeats, cores=2)
                runs.append({"handler": name, "count": 1, "cores": 2, "us": micro})
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
