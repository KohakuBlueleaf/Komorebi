"""Chained casts: a class whose handlers cast along a row (`if b.sender(): v =
b.cast(...)`, `else: v = b.arrive()`), 8 calls in one chain command (call c casting
a's tiles 2c and 2c + 1), then two handlers in turn (spread, 1 block; triple, 3 blocks
summed); every core's block of every call exact. See ../README.md."""

import json
import sys

import numpy as np

import komorebi as km

CORES, WIDTH, CALLS = 8, 2, 8  # a row of 8 cores, 2 tiles a block, 8 calls a chain


class Casts(km.Program):
    """Each core multiplies the identity by the row's cast block: its block is the
    block cast, exact (a's values are bf16 multiples of 1/64)."""

    @km.tile_function(fidelity="HiFi4")
    def project(x, w):
        return x @ w

    @km.handler
    def spread(self, x: km.Tensor, a: km.Tensor, out: km.Tensor, call: int):
        along = km.Broadcast(self, along="row", window=1)
        if along.sender():
            block = along.cast(a.pages(call * 2, 2).read())
        else:
            block = along.arrive()
        acc = km.tile.zeros(km.fp32, 2)
        for xb in km.Stream(x.tiles[0:1, 0:1], block=1):
            acc += self.project(xb, block)
        out.pages((call * 8 + self.rank) * 2, 2).write(acc)

    @km.handler
    def triple(self, x: km.Tensor, a: km.Tensor, out: km.Tensor, call: int):
        along = km.Broadcast(self, along="row", window=1)
        acc = km.tile.zeros(km.fp32, 2)
        for i in range(3):
            if along.sender():
                block = along.cast(
                    a.tiles[0:1, call * 6 + i * 2 : call * 6 + i * 2 + 2].read()
                )
            else:
                block = along.arrive()
            for xb in km.Stream(x.tiles[0:1, 0:1], block=1):
                acc += self.project(xb, block)
        out.pages((call * 8 + self.rank) * 2, 2).write(acc)


def main():
    rng = np.random.default_rng(1)
    a = rng.integers(-64, 64, (32, 3 * CALLS * WIDTH * 32)).astype(np.float32) / 64
    eye = np.eye(32, dtype=np.float32)
    with km.Session() as session:
        tx = session.tensor(eye, "bfloat16", "tile")
        ta = session.tensor(a, "bfloat16", "tile")
        one = session.empty((CALLS * CORES * 32, WIDTH * 32), "float32", "tile")
        two = session.empty((CALLS * CORES * 32, WIDTH * 32), "float32", "tile")
        program = Casts(
            session=session, chain=True, rect=(0, 0, CORES, 1), x=tx, a=ta, out=one
        )
        run = session.load(program)
        group = program.rectangle["group"]
        spread = run.chain(
            [Casts.spread(tx, ta, one, c, cores=group) for c in range(CALLS)]
        )
        first = run.wait(run.submit(spread), timeout=30.0)
        handlers = [Casts.spread, Casts.triple]
        mixed = run.chain(
            [handlers[c % 2](tx, ta, two, c, cores=group) for c in range(CALLS)]
        )
        second = run.wait(run.submit(mixed), timeout=30.0)
        run.stop()
        kernel = run.result()["error_name"]
        got_one = one.read().reshape(CALLS * CORES, 32, WIDTH * 32)
        got_two = two.read().reshape(CALLS * CORES, 32, WIDTH * 32)

    def tiles(first):  # a's tiles first and first + 1
        return a[:, first * 32 : (first + WIDTH) * 32]

    want_one = [tiles(2 * c) for c in range(CALLS)]
    want_two = [
        tiles(2 * c) if c % 2 == 0 else sum(tiles(6 * c + 2 * i) for i in range(3))
        for c in range(CALLS)
    ]
    report = {
        "events": [first.kind, second.kind],
        "card_us": [first.microseconds, second.microseconds],
        "kernel": kernel,
        "spread_exact": all(
            np.array_equal(got_one[i], want_one[i // CORES])
            for i in range(len(got_one))
        ),
        "two_handlers_exact": all(
            np.array_equal(got_two[i], want_two[i // CORES])
            for i in range(len(got_two))
        ),
    }
    report["ok"] = (
        report["spread_exact"]
        and report["two_handlers_exact"]
        and kernel == "none"
        and first.kind == second.kind == "done"
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
