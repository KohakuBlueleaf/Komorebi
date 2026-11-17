"""Dynamic work whose amount the data decides, against a Python walk of the same rules:
(V1b) a ray walks a scene held in L1 and spawns a ray at each step, all in one body
(the walk, the quota test, send, forward, consume); (V2) the spawn count is a word of
the message, in lines. Each core's totals go to its results words."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import finish, log

import komorebi as km

NODES = 256  # scene words on every core
COUNTERS = "uint32_t ray_count = 0, ray_sum = 0, ray_starved = 0;"
RESET = "ray_count = 0; ray_sum = 0; ray_starved = 0;"
WALK = """
volatile tt_l1_ptr uint32_t* nodes = km::l1(scene);
const uint32_t hit = nodes[message[kRayOrigin] & 255];
if (message[kRayDepth] == 0 || km_messages.quota() == 0) {
    ray_starved += message[kRayDepth] != 0;
    ray_count += 1;
    ray_sum += message[kRayOrigin];
    km_messages.consume();
} else {
    message[kRayDepth] -= 1;
    uint32_t child[kMessageWords] = {};
    child[kRayOrigin] = message[kRayOrigin] + hit;
    child[kRayDepth] = message[kRayDepth];
    km_messages.send(child[kRayOrigin] % km_messages.count(), child);
    km_messages.forward(message[kRayOrigin] % km_messages.count(), message);
}
"""
REPORT = """
UNPACK(({
    volatile tt_l1_ptr uint32_t* words = km::l1(results);
    words[0] = ray_count;
    words[1] = ray_sum;
    words[2] = ray_starved;
}));
"""
REPORT_LINES = """
UNPACK(({
    volatile tt_l1_ptr uint32_t* words = km::l1(results);
    words[0] = consumed;
    words[1] = total;
    words[2] = 0;
}));
"""


class Ray(km.Message):
    origin: int
    depth: int
    fanout: int


class Walk(km.Program):
    """V1b: the seeds in lines, every message's step in one body."""

    @km.handler
    def trace(self, count: int, depth: int, scene: int, results: int):
        km.cxx(COUNTERS, at="start")
        km.cxx(RESET)
        if self.seeding():
            part = self.partition(count)
            for index in range(part.count):
                seed = Ray(origin=part.start + index, depth=depth)
                self.send(self.rank, seed)
        for message in self.arrive():  # noqa: B007 - WALK's C++ names it
            km.cxx(WALK, takes=[scene])
        km.cxx(REPORT, takes=[results])


class Fan(km.Program):
    """V2: a message spawns `fanout` children, depth - 1, in lines."""

    @km.handler
    def trace(self, count: int, depth: int, fanout: int, results: int):
        consumed = 0
        total = 0
        if self.seeding():
            part = self.partition(count)
            for index in range(part.count):
                seed = Ray(origin=part.start + index, depth=depth, fanout=fanout)
                self.send(self.rank, seed)
        for message in self.arrive():
            if message.depth == 0:
                consumed += 1
                total += message.origin
                self.consume()
            else:
                for i in range(message.fanout):
                    child = Ray(
                        origin=message.origin * 3 + i,
                        depth=message.depth - 1,
                        fanout=message.fanout,
                    )
                    self.send((message.origin + i) % self.size, child)
                self.consume()
        km.cxx(REPORT_LINES, takes=[results, consumed, total])


def walk(seeds, depth, scene):
    """V1b's rules in Python: (rays consumed, the sum of their origins mod 2^32)."""
    count = total = 0
    pending = [(origin, depth) for origin in range(seeds)]
    while pending:
        origin, left = pending.pop()
        if left == 0:
            count, total = count + 1, (total + origin) % (1 << 32)
            continue
        hit = int(scene[origin & 255])
        pending += [((origin + hit) % (1 << 32), left - 1), (origin, left - 1)]
    return count, total


def fan(seeds, depth, fanout):
    count = total = 0
    pending = [(origin, depth) for origin in range(seeds)]
    while pending:
        origin, left = pending.pop()
        if left == 0:
            count, total = count + 1, (total + origin) % (1 << 32)
            continue
        pending += [((origin * 3 + i) % (1 << 32), left - 1) for i in range(fanout)]
    return count, total


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--fanout", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    scene = rng.integers(0, 1024, NODES).astype(np.int32)
    report = {"cores": args.cores, "seeds": args.seeds, "depth": args.depth}
    group = (0, args.cores)
    options = {"max_live": 1 << 12, "queue_messages": 1 << 12}
    totals = {}
    with km.Session(log=log) as session:
        grid = session.grid
        every = grid[0] * grid[1]
        scenes = session.tensor(np.tile(scene, (every, 1)), "int32", shard_grid=grid)
        for name, cls, words in (
            ("walk", Walk, (args.seeds, args.depth, scenes.address)),
            ("fan", Fan, (args.seeds, args.depth, args.fanout)),
        ):
            zeros = np.zeros((every, 8), np.int32)
            results = session.tensor(zeros, "int32", shard_grid=grid)
            program = cls(log=log).dynamic("trace", **options)
            run = session.load(program)
            try:
                call = (*words, results.address)
                handle = run.submit(cls.trace, group, args=call, exclusive=True)
                run.drain(timeout=20.0)
            finally:
                run.stop()
            event = run.event(handle)
            report[f"{name}_event"] = event.kind
            report[f"{name}_us"] = round(event.microseconds, 1)
            rows = results.read().astype(np.int64) & 0xFFFFFFFF
            totals[name] = (
                int(rows[:, 0].sum()),
                int(rows[:, 1].sum() % (1 << 32)),
                int(rows[:, 2].sum()),
            )
    want = {
        "walk": walk(args.seeds, args.depth, scene),
        "fan": fan(args.seeds, args.depth, args.fanout),
    }
    checks = []
    for name, (count, total, starved) in totals.items():
        report[name] = {"count": count, "sum": total, "starved": starved}
        report[f"{name}_numpy"] = list(want[name])
        wrong = int((count, total) != want[name]) + int(starved != 0)
        report[f"{name}_wrong"] = {"count": wrong}
        checks.append((f"{name}_wrong", "count", 0))
        log(f"{name}: {report[name]}, Python {want[name]}")
    finish(report, checks)


if __name__ == "__main__":
    sys.exit(main())
