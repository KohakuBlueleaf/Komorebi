"""The account of linear_class.py's two plans: their parts as commands of their own
(the K split's W fill, its X stream and products, the grid's casts with the blocks
dropped), at the same shapes (build/parts/ beside this file).
Example: python linear_parts.py --repeat 3"""

import argparse
import json
import sys

import numpy as np
from linear_class import K_TILES, KERNEL_BYTES, N_TILES, OH, log

import komorebi as km

# The plans as the class's first version ran them (README.md's rows): the K split in
# groups of 3 cores of 4 columns, W filled whole; the grid on 8 x 8 of 16 columns.
SPLIT = 3
WIDTH = 4
PART = -(-K_TILES // SPLIT)
SPLIT_CORES = N_TILES // WIDTH * SPLIT  # 96
STREAM_BLOCK = 8
GRID = 8
KC = 4
OW = N_TILES // GRID  # 16


class Parts(km.Program):
    """fill: the K split's W read alone (the field stays); products: its X stream and
    products on that field, each core's sums to its own pages of `sums`; casts: the
    grid plan's two casts, the blocks dropped."""

    weight: km.L1[km.bf16, PART, WIDTH, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def fill(self, w: km.Tensor):
        share = self.rank % SPLIT
        first = self.rank // SPLIT * WIDTH
        k0 = share * K_TILES // SPLIT
        k1 = (share + 1) * K_TILES // SPLIT
        self.weight.fill(km.Stream(w.tiles[k0:k1, first : first + WIDTH]))

    @km.handler
    def products(self, x: km.Tensor, sums: km.Tensor):
        share = self.rank % SPLIT
        k0 = share * K_TILES // SPLIT
        k1 = (share + 1) * K_TILES // SPLIT
        acc = km.tile.zeros(km.fp32, WIDTH)
        for block in km.Stream(x.tiles[0:1, k0:k1], block=STREAM_BLOCK):
            acc += self.project(block, self.weight)
        sums.tiles[0:1, self.rank * WIDTH : self.rank * WIDTH + WIDTH].write(acc)

    @km.handler
    def casts(self, x: km.Tensor, w: km.Tensor):
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
                ws = down.cast(w.tiles[k0 : k0 + KC, left : left + OW].read())
            else:
                ws = down.arrive()
            for _tile in xs:
                pass
            for _tile in ws:
                pass


def made(session):
    """Parts with tensors of both shapes: decode (M 32) and prefill (M 1024)."""
    k, n = K_TILES * 32, N_TILES * 32
    tensors = {
        "x": session.empty((1024, k), "bfloat16", "tile"),
        "w": session.empty((k, n), "bfloat16", "tile"),
        "sums": session.empty((32, SPLIT_CORES * WIDTH * 32), "float32", "tile"),
    }
    return Parts(
        log=log,
        rect=(0, 0, GRID, GRID),
        l1=session.l1()[1],
        **tensors,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    k, n = K_TILES * 32, N_TILES * 32
    report = {}
    with km.Session(log=log, kernel_bytes=KERNEL_BYTES) as session:
        program = made(session)
        x32 = session.tensor(rng.standard_normal((32, k)), "bfloat16", "tile")
        x1024 = session.tensor(rng.standard_normal((1024, k)), "bfloat16", "tile")
        w = session.tensor(rng.standard_normal((k, n)) / 64, "bfloat16", "tile")
        sums = session.empty((32, SPLIT_CORES * WIDTH * 32), "float32", "tile")
        run = session.load(program)
        try:
            rect = program.rectangle["group"]
            split = (0, SPLIT_CORES)
            commands = [
                ("fill", Parts.fill, split, (w,)),
                ("products", Parts.products, split, (x32, sums)),
                ("casts", Parts.casts, rect, (x1024, w)),
            ]
            for name, handler, group, words in commands:
                times = []
                for _ in range(args.repeat):
                    event = run.wait(run.submit(handler, group, args=words), 30.0)
                    if event.kind != "done":
                        raise RuntimeError(f"{name}: {event}")
                    times.append(event.microseconds)
                report[name] = {
                    "microseconds": times,
                    "median": float(np.median(times)),
                }
                log(f"{name}: {report[name]}")
        finally:
            run.stop()
    report["kernel_text"] = program.sizes.text
    print(json.dumps(report), flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
