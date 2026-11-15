"""The account of linear_class.py's two plans: their parts as commands of their own
(the K split's W fill, its X stream and products, the grid's casts with the blocks
dropped), at the same shapes (build/parts/ beside this file).
Example: python linear_parts.py --repeat 3"""

import argparse
import json
import sys

import numpy as np
from linear_class import KERNEL_BYTES, OH, Linear, log

import komorebi as km


class Parts(km.Program):
    """fill: the K split's W read alone (the field stays); products: its X stream and
    products on that field, each core's sums to its own pages of `sums`; casts: the
    grid plan's two casts, the blocks dropped."""

    K_TILES = km.constant(Linear.K_TILES.default)
    N_TILES = km.constant(Linear.N_TILES.default)
    # The plans as the class's first version ran them (README.md's rows): the K split
    # in groups of 3 cores of 4 columns, W filled whole; the grid on 8 x 8 of 16.
    SPLIT = km.constant(3)
    WIDTH = km.constant(4)
    PART = -(-K_TILES // SPLIT)
    SPLIT_CORES = N_TILES // WIDTH * SPLIT  # 96
    STREAM_BLOCK = km.constant(8)
    GRID = km.constant(8)
    KC = km.constant(4)
    OW = N_TILES // GRID  # 16
    OH = km.constant(OH)

    weight: km.L1[km.bf16, PART, WIDTH, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def fill(self, w: km.Tensor):
        share = self.rank % self.SPLIT
        first = self.rank // self.SPLIT * self.WIDTH
        k0 = share * self.K_TILES // self.SPLIT
        k1 = (share + 1) * self.K_TILES // self.SPLIT
        self.weight.fill(km.Stream(w.tiles[k0:k1, first : first + self.WIDTH]))

    @km.handler
    def products(self, x: km.Tensor, sums: km.Tensor):
        share = self.rank % self.SPLIT
        k0 = share * self.K_TILES // self.SPLIT
        k1 = (share + 1) * self.K_TILES // self.SPLIT
        acc = km.tile.zeros(km.fp32, self.WIDTH)
        for block in km.Stream(x.tiles[0:1, k0:k1], block=self.STREAM_BLOCK):
            acc += self.project(block, self.weight)
        sums.tiles[
            0:1, self.rank * self.WIDTH : self.rank * self.WIDTH + self.WIDTH
        ].write(acc)

    @km.handler
    def casts(self, x: km.Tensor, w: km.Tensor):
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
                ws = down.cast(w.tiles[k0 : k0 + self.KC, left : left + self.OW].read())
            else:
                ws = down.arrive()
            for _tile in xs:
                pass
            for _tile in ws:
                pass


def made(session):
    """Parts with tensors of both shapes: decode (M 32) and prefill (M 1024)."""
    k, n = Parts.K_TILES.default * 32, Parts.N_TILES.default * 32
    split, width = Parts.SPLIT.default, Parts.WIDTH.default
    columns = Parts.N_TILES.default // width * split * width * 32  # SPLIT_CORES's
    tensors = {
        "x": session.empty((1024, k), "bfloat16", "tile"),
        "w": session.empty((k, n), "bfloat16", "tile"),
        "sums": session.empty((32, columns), "float32", "tile"),
    }
    grid = Parts.GRID.default
    return Parts(
        log=log,
        rect=(0, 0, grid, grid),
        session=session,
        **tensors,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    report = {}
    with km.Session(log=log, kernel_bytes=KERNEL_BYTES) as session:
        program = made(session)
        k, n = program.K_TILES * 32, program.N_TILES * 32
        x32 = session.tensor(rng.standard_normal((32, k)), "bfloat16", "tile")
        x1024 = session.tensor(rng.standard_normal((1024, k)), "bfloat16", "tile")
        w = session.tensor(rng.standard_normal((k, n)) / 64, "bfloat16", "tile")
        columns = program.SPLIT_CORES * program.WIDTH * 32
        sums = session.empty((32, columns), "float32", "tile")
        run = session.load(program)
        try:
            rect = program.rectangle["group"]
            split = (0, program.SPLIT_CORES)
            commands = [
                ("fill", program.fill, split, (w,)),
                ("products", program.products, split, (x32, sums)),
                ("casts", program.casts, rect, (x1024, w)),
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
