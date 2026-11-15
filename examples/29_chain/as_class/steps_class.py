"""Example 29's steps written as a class: SCALE x into y, SCALE y by 3 into z, ADD y to
z into s (y written further back: km.past), SUM s's tiles up a tree into t. The chain
stated both ways (the host's list, the class's km.chain method) and as a command a
step, every word against numpy, the times beside ../chain_steps.py's. See ../README.md.
"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km

WORDS = 1024  # float32 words a page: one tile's
STARTED = time.monotonic()

SCALE = """
#include "api/compute/eltwise_unary/binop_with_scalar.h"
copy_init(tile_buffer);
copy_tile(tile_buffer, tile_index, 0);
binop_with_scalar_tile_init();
mul_unary_tile(0, __builtin_bit_cast(uint32_t, factor));
"""


@km.tile_function
def plus(a, b):
    return a + b


class StepHandlers(km.Program):
    """Each core's `width` tiles from rank x width (SCALE, ADD), or its share of
    `tiles` tiles summed up a tree to the group's root (SUM); float32 tiles, the
    factor a float word. Made alone: no chain's walk in its kernels."""

    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, factor: float, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            scaled = km.cxx(SCALE, takes=[tile, factor], gives=km.Tile)
            results.push(scaled)

    @km.handler
    def add(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for x, y in zip(
            km.Stream(a.pages(first, width)), km.Stream(b.pages(first, width))
        ):
            results.push(plus(x, y))

    @km.handler
    def sum(self, s: km.Tensor, t: km.Tensor, tiles: int):
        part = self.partition(s.pages(0, tiles))
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s[part]):
            total += tile
        whole = km.reduce(total, form="tree")
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)


class Steps(StepHandlers):
    """The handlers with the chain's walk: the four steps as a km.chain method."""

    @km.chain
    def steps(self, x, y, z, s, t, factor: float, width: int, tiles: int):
        self.scale(x, y, factor, width)
        self.scale(y, z, 3.0, width)
        self.add(z, km.past(y), s, width)
        self.sum(s, t, tiles)


def step_args(tensors, factor, width, tiles):
    """Each step's handler and arguments, as a command a step submits them."""
    x, y, z, s, t = (tensors[name] for name in "xyzst")
    return [
        (Steps.scale, (x, y, float(factor), width)),
        (Steps.scale, (y, z, 3.0, width)),
        (Steps.add, (z, y, s, width)),
        (Steps.sum, (s, t, tiles)),
    ]


def host_list(tensors, factor, width, tiles):
    """The chain as the host's list of calls, the same as Steps.steps: y, which ADD
    reads, written two calls back (km.past)."""
    x, y, z, s, t = (tensors[name] for name in "xyzst")
    return [
        Steps.scale(x, y, float(factor), width),
        Steps.scale(y, z, 3.0, width),
        Steps.add(z, km.past(y), s, width),
        Steps.sum(s, t, tiles),
    ]


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8, help="cores 0 .. N - 1")
    parser.add_argument("--width", type=int, default=2, help="tiles a core")
    parser.add_argument("--runs", type=int, default=200, help="timed runs each way")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--kernel-bytes",
        type=int,
        default=None,
        help="the session's kernel buffer (default: the kernels')",
    )
    parser.add_argument(
        "--skip-plain",
        action="store_true",
        help="no command a step of the handlers built without the walk",
    )
    return parser.parse_args()


def expected(x, factor):
    y = x * factor
    z = y * 3
    s = z + y
    return {"y": y, "z": z, "s": s, "t": s.sum(axis=0, dtype=np.float32)[None]}


def outputs(session, tensors):
    got = session.read(*(tensors[name] for name in "yzst"))
    return {
        name: raw.view(np.float32).reshape(-1, WORDS)
        for name, raw in zip("yzst", got, strict=True)
    }


def bad_words(got, want):
    return {name: int(np.sum(got[name] != want[name])) for name in want}


def by_steps(run, steps, cores):
    """A command a step, each waited for: the dispatcher's microseconds, summed."""
    total = 0.0
    for handler, arguments in steps:
        handle = run.submit(handler, cores, args=arguments, exclusive=True)
        total += run.wait(handle).microseconds
    return total


def by_chain(run, chain):
    """The chain as one command, waited for: its microseconds."""
    return run.wait(run.submit(chain)).microseconds


def made(cls, tensors, **given):
    """`cls` made with an example of each tensor parameter."""
    x = tensors["x"]
    return cls(a=x, b=x, out=tensors["y"], s=tensors["s"], t=tensors["t"], **given)


def plain_steps(session, tensors, args, steps):
    """A command a step of the handlers built without the walk: the plain command's
    time the walk is measured against."""
    program = made(StepHandlers, tensors, session=session, log=log)
    run = session.load(program, wait=0.5, deadline=0.5)
    cores = (0, args.cores)
    for _ in range(3):
        by_steps(run, steps, cores)
    start = time.perf_counter()
    times = [by_steps(run, steps, cores) for _ in range(args.runs)]
    host = round((time.perf_counter() - start) * 1e6 / args.runs, 2)
    run.stop()
    return host, round(float(np.median(times)), 2), run.result()["error_name"]


def main():
    args = parse()
    generator = np.random.default_rng(args.seed)
    tiles = args.cores * args.width
    x = generator.integers(-50, 51, (tiles, WORDS)).astype(np.float32)
    result = {"cores": args.cores, "width": args.width, "tiles": tiles}
    cores = (0, args.cores)
    with km.Session(log=log, kernel_bytes=args.kernel_bytes) as session:
        tensors = {"x": session.tensor(x, "float32")}
        for name, rows in (("y", tiles), ("z", tiles), ("s", tiles), ("t", 1)):
            zeros = np.zeros((rows, WORDS), np.float32)
            tensors[name] = session.tensor(zeros, "float32")
        plain = {"checks": {}, "host": {}, "card": {}, "kernel": "none"}
        if not args.skip_plain:  # first: the handlers built without the walk
            steps = step_args(tensors, 1, args.width, tiles)
            host, median, plain["kernel"] = plain_steps(session, tensors, args, steps)
            plain["checks"]["without the walk"] = bad_words(
                outputs(session, tensors), expected(x, 1)
            )
            plain["host"]["steps without the walk"] = host
            plain["card"]["steps without the walk"] = median
        result["kernel_without_walk"] = plain["kernel"]
        program = made(Steps, tensors, session=session, log=log)
        run = session.load(program, wait=0.5, deadline=0.5)
        x_, y, z, s, t = (tensors[name] for name in "xyzst")
        chains = {}
        for factor in (2, 5, 7, 3, 1):
            listed = run.chain(
                host_list(tensors, factor, args.width, tiles), cores=cores
            )
            method = run.chain(
                program.steps(x_, y, z, s, t, float(factor), args.width, tiles),
                cores=cores,
            )
            chains[factor] = (listed, method)
        same = all(
            [c.args for c in listed.calls] == [c.args for c in method.calls]
            for listed, method in chains.values()
        )
        result["same_records"] = same
        log(f"loaded; the chains' records written, both forms alike: {same}")
        checks = {}
        for way, factor in (("steps", 2), ("list", 5), ("method", 7), ("list", 3)):
            listed, method = chains[factor]
            if way == "steps":
                by_steps(run, step_args(tensors, factor, args.width, tiles), cores)
            else:
                by_chain(run, listed if way == "list" else method)
            checks[f"{way} x{factor}"] = bad_words(
                outputs(session, tensors), expected(x, factor)
            )
        timed, card = {}, {}
        listed, method = chains[1]
        for way in ("steps", "list", "method"):
            start = time.perf_counter()
            steps = step_args(tensors, 1, args.width, tiles)
            times = [
                (
                    by_steps(run, steps, cores)
                    if way == "steps"
                    else by_chain(run, listed if way == "list" else method)
                )
                for _ in range(args.runs)
            ]
            timed[way] = round((time.perf_counter() - start) * 1e6 / args.runs, 2)
            card[way] = round(float(np.median(times)), 2)
        checks["after timing"] = bad_words(outputs(session, tensors), expected(x, 1))
        run.stop()
        for pair in chains.values():
            for chain in pair:
                chain.free()
        result["kernel"] = run.result()["error_name"]
    result["checks"] = {**plain["checks"], **checks}
    timed.update(plain["host"])
    card.update(plain["card"])
    result["microseconds_a_run"] = timed  # the host's, submit to the last wait
    result["card_microseconds_a_run"] = card  # the dispatcher's, median
    checks = result["checks"]
    result["ok"] = bool(
        result["kernel"] == "none"
        and result["kernel_without_walk"] == "none"
        and same
        and all(not any(bad.values()) for bad in checks.values())
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
