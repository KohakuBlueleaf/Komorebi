"""A scene in DRAM as two treelets, walked by dynamic runs that hold one treelet in L1,
against a Python walk of the same rules: run 1 walks treelet 0 and bins each ray that
reaches depth 0; a plain handler reads treelet 1 from DRAM into L1; run 2 seeds the
binned rays and walks treelet 1. A core's totals go to its results words."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import finish, log, timed

import komorebi as km

NODES = 256  # words of a treelet: one page, held at `scene` on every core
BIN = 1024  # words of a core's bin: its count, then the binned origins
COUNTERS = "uint32_t ray_count = 0, ray_sum = 0, ray_starved = 0, ray_binned = 0;"
RESET = "ray_count = 0; ray_sum = 0; ray_starved = 0; ray_binned = 0;"
STEP = """
volatile tt_l1_ptr uint32_t* nodes = km::l1(scene);
const uint32_t hit = nodes[message[kRayOrigin] & 255];
message[kRayDepth] -= 1;
uint32_t child[kMessageWords] = {};
child[kRayOrigin] = message[kRayOrigin] + hit;
child[kRayDepth] = message[kRayDepth];
km_messages.send(child[kRayOrigin] % km_messages.count(), child);
km_messages.forward(message[kRayOrigin] % km_messages.count(), message);
"""
BINNED = """
if (message[kRayDepth] == 0) {
    UNPACK(({
        volatile tt_l1_ptr uint32_t* bin = km::l1(bins);
        if (ray_binned + 1 < kBin) { bin[1 + ray_binned] = message[kRayOrigin]; }
    }));
    ray_binned += 1;
    km_messages.consume();
} else if (km_messages.quota() == 0) {
    ray_starved += 1;
    km_messages.consume();
} else {
""" + STEP + "}\n"
COUNTED = """
if (message[kRayDepth] == 0 || km_messages.quota() == 0) {
    ray_starved += message[kRayDepth] != 0;
    ray_count += 1;
    ray_sum += message[kRayOrigin];
    km_messages.consume();
} else {
""" + STEP + "}\n"
SEED = """
volatile tt_l1_ptr uint32_t* bin = km::l1(bins);
const uint32_t binned = bin[0];
for (uint32_t i = 0; i < binned && i + 1 < kBin; ++i) {
    uint32_t seed[kMessageWords] = {};
    seed[kRayOrigin] = bin[1 + i];
    seed[kRayDepth] = depth;
    km_messages.send(km_messages.rank(), seed);
}
"""
REPORT = """
UNPACK(({
    volatile tt_l1_ptr uint32_t* words = km::l1(results);
    words[0] = ray_count;
    words[1] = ray_sum;
    words[2] = ray_starved;
    words[3] = ray_binned;
    volatile tt_l1_ptr uint32_t* bin = km::l1(bins);
    if (ray_binned > 0) { bin[0] = ray_binned; }
}));
"""
LOAD = """
km::read_pages(movement, command, tree, treelet, 1, scene);
noc_async_read_barrier();
"""


class Ray(km.Message):
    origin: int
    depth: int


class Bin(km.Program, defines={"kBin": BIN}):
    """Run 1: seeds in lines; a ray at depth 0 goes to its core's bin."""

    @km.handler
    def trace(self, count: int, depth: int, scene: int, bins: int, results: int):
        km.cxx(COUNTERS, at="start")
        km.cxx(RESET)
        if self.seeding():
            part = self.partition(count)
            for index in range(part.count):
                seed = Ray(origin=part.start + index, depth=depth)
                self.send(self.rank, seed)
        for message in self.arrive():  # noqa: B007 - BINNED's C++ names it
            km.cxx(BINNED, takes=[scene, bins])
        km.cxx(REPORT, takes=[results, bins])


class Load(km.Program):
    """Between the runs: treelet `treelet` of `tree` into every core's scene words."""

    @km.handler
    def load(self, tree: km.Tensor, treelet: int, scene: int):
        km.cxx(LOAD, takes=[tree, treelet, scene], on="mover_b")


class Walk(km.Program, defines={"kBin": BIN}):
    """Run 2: each core seeds its bin's rays; a ray at depth 0 is counted."""

    @km.handler
    def trace(self, depth: int, scene: int, bins: int, results: int):
        km.cxx(COUNTERS, at="start")
        km.cxx(RESET)
        if self.seeding():
            km.cxx(SEED, takes=[bins, depth])
        for message in self.arrive():  # noqa: B007 - COUNTED's C++ names it
            km.cxx(COUNTED, takes=[scene])
        km.cxx(REPORT, takes=[results, bins])


def leaves(origins, depth, scene):
    """The rules in Python: the origins of the rays that reach depth 0."""
    out, pending = [], [(origin, depth) for origin in origins]
    while pending:
        origin, left = pending.pop()
        if left == 0:
            out.append(origin)
            continue
        hit = int(scene[origin & 255])
        pending += [((origin + hit) % (1 << 32), left - 1), (origin, left - 1)]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    tree = rng.integers(0, 1024, (2, NODES)).astype(np.int32)
    report = {"cores": args.cores, "seeds": args.seeds, "depth": args.depth}
    group = (0, args.cores)
    options = {"max_live": 1 << 14, "queue_messages": 1 << 14}  # README: the quota
    rows = {}
    with km.Session(log=log) as session:
        grid = session.grid
        every = grid[0] * grid[1]
        trees = session.tensor(tree, "int32")
        shard = {"shard_grid": grid}
        start = np.tile(tree[0], (every, 1))
        scenes = session.tensor(start, "int32", **shard)
        bins = session.tensor(np.zeros((every, BIN), np.int32), "int32", **shard)
        stages = [
            ("bin", Bin, (args.seeds, args.depth, scenes.address, bins.address)),
            ("walk", Walk, (args.depth, scenes.address, bins.address)),
        ]
        for name, cls, words in stages:
            if name == "walk":  # treelet 1 into L1, by a plain handler
                run = session.load(Load(tree=trees, log=log))
                try:
                    call = (trees, 1, scenes.address)
                    report["load_us"] = timed(run, Load.load, (0, every), call)
                finally:
                    run.stop()
            results = session.tensor(np.zeros((every, 8), np.int32), "int32", **shard)
            run = session.load(cls(log=log).dynamic("trace", **options))
            try:
                call = (*words, results.address)
                handle = run.submit(cls.trace, group, args=call, exclusive=True)
                run.drain(timeout=20.0)
            finally:
                run.stop()
            event = run.event(handle)
            report[f"{name}_event"] = event.kind
            report[f"{name}_us"] = round(event.microseconds, 1)
            rows[name] = results.read().astype(np.int64) & 0xFFFFFFFF
        report["scene_after"] = int(
            np.count_nonzero(scenes.read()[: args.cores] != tree[1])
        )
    first = leaves(range(args.seeds), args.depth, tree[0])
    final = leaves(first, args.depth, tree[1])
    want = [len(final), sum(final) % (1 << 32)]
    report["binned"] = int(rows["bin"][:, 3].sum())
    report["starved"] = int(rows["bin"][:, 2].sum() + rows["walk"][:, 2].sum())
    got = [int(rows["walk"][:, 0].sum()), int(rows["walk"][:, 1].sum() % (1 << 32))]
    report["walk"], report["walk_python"] = got, want
    wrong = int(got != want) + int(report["binned"] != len(first))
    wrong += int(report["starved"] != 0) + int(report["scene_after"] != 0)
    report["treelet_wrong"] = {"count": wrong}
    log(f"binned {report['binned']} (Python {len(first)}); walk {got}, Python {want}")
    finish(report, [("treelet_wrong", "count", 0)])


if __name__ == "__main__":
    sys.exit(main())
