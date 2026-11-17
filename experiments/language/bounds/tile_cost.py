"""Which CPU spends a decode product's time a W tile: on one card, each core's share of
gate's W (KS x W8 tiles) read and dropped (the movers alone), multiplied from L1 (the
compute alone) and streamed and multiplied (both), in bfp8_b and bfp4_b. README.md.
Example: python tile_cost.py --card 0 --format bfp8_b bfp4_b"""

import argparse
import json
import sys

import numpy as np

import komorebi as km
from experiments.language.bounds import bounds as bd
from komorebi.bench import power

CORES, SHARES = 120, 4  # every core; 4 K shares a column block
log = bd.log


class Probe(km.Program):
    """A core's share of W: rows k0 .. k0 + KS of a block of W8 columns (rank // 4),
    k0 by rank % 4; its row tile of Y written at row rank % 4."""

    KS = km.constant(42)
    W8 = km.constant(8)
    CHUNK = km.constant(2)
    BW = CHUNK * W8
    BX = km.constant(8)  # X's block against the field
    wf: km.L1[km.bfp8_b, KS, W8, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def streamed(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        k0 = self.rank % 4 * self.KS
        first = self.rank // 4 * self.W8
        acc = km.tile.zeros(km.fp32, self.W8)
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + self.KS], block=self.CHUNK),
            km.Stream(
                w.tiles[k0 : k0 + self.KS, first : first + self.W8], block=self.BW
            ),
        ):
            acc += self.project(xb, wb)
        y.tiles[self.rank % 4 : self.rank % 4 + 1, first : first + self.W8].write(acc)

    @km.handler
    def drained(self, w: km.Tensor):
        k0 = self.rank % 4 * self.KS
        first = self.rank // 4 * self.W8
        km.Stream(
            w.tiles[k0 : k0 + self.KS, first : first + self.W8], block=self.BW
        ).drain()

    @km.handler
    def load(self, w: km.Tensor):
        k0 = self.rank % 4 * self.KS
        first = self.rank // 4 * self.W8
        self.wf.fill(km.Stream(w.tiles[k0 : k0 + self.KS, first : first + self.W8]))

    @km.handler
    def held(self, x: km.Tensor, y: km.Tensor):
        k0 = self.rank % 4 * self.KS
        first = self.rank // 4 * self.W8
        acc = km.tile.zeros(km.fp32, self.W8)
        for xb in km.Stream(x.tiles[0:1, k0 : k0 + self.KS], block=self.BX):
            acc += self.project(xb, self.wf)
        y.tiles[self.rank % 4 : self.rank % 4 + 1, first : first + self.W8].write(acc)


class Probe4(Probe):
    """Probe with the kept field in bfp4_b."""

    wf: km.L1[km.bfp4_b, Probe.KS, Probe.W8, 32, 32]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--format", nargs="+", default=["bfp8_b", "bfp4_b"])
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    shape = bd.PRODUCTS["gate"]
    out = []
    with km.Session(card=args.card, log=log) as session:
        x = session.tensor(bd.activations(1, shape["k"], args.seed), "bfloat16", "tile")
        y = session.empty((SHARES * 32, shape["n"]), "bfloat16", "tile")
        for fmt in args.format:
            pages, _, _ = bd.weights("gate", fmt, args.seed, 256)
            w = session.empty((shape["k"], shape["n"]), fmt, "tile")
            session.write(w, pages)
            form = Probe4 if fmt == "bfp4_b" else Probe
            program = form(log=log, session=session, x=x, w=w, y=y)
            run = session.load(program)
            group = (0, CORES)
            row = {"format": fmt, "w_tiles_a_core": 42 * 8, "cores": CORES}
            try:
                for name, call in (
                    ("streamed", (program.streamed, (x, w, y))),
                    ("drained", (program.drained, (w,))),
                    ("load", (program.load, (w,))),
                    ("held", (program.held, (x, y))),
                ):
                    handler, words = call
                    times = []
                    for _ in range(args.repeat if name != "load" else 1):
                        event = run.wait(run.submit(handler, group, args=words), 10.0)
                        if event.kind != "done":
                            raise km.KomorebiError(f"{name}: {event}")
                        times.append(event.microseconds)
                    row[name + "_us"] = float(np.median(times))
            finally:
                run.stop()
            row["kernel"] = run.result()["error_name"]
            w.free()
            log(json.dumps(row))
            out.append(row)
    for row in out:
        print(json.dumps(row), flush=True)
    print(json.dumps(power.clocked({"cards": [args.card]}, card=args.card)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
