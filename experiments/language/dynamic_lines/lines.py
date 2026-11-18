"""A dynamic handler's step in lines (README.md): a body's words (the scene's hit, the
quota left) read by the handler's if, its message words and its send's rank, against
the same rules in one body and in Python. A JSON line a form: totals, microseconds
of 5 runs, kernel text bytes. argv: --cards N (one card; the first visible if none)."""

import argparse
import json
import statistics
import time

import numpy as np

import komorebi as km

STARTED = time.monotonic()
NODES = 256  # scene words on every core
RUNS = 5
SEEDS, DEPTH = 16, 6
OPTIONS = {"max_live": 1 << 12, "queue_messages": 1 << 12}

HIT = """
volatile tt_l1_ptr uint32_t* nodes = km::l1(scene);
hit = nodes[origin & 255];
"""
QUOTA = "left = km_messages.quota();"
REPORT = """
UNPACK(({
    volatile tt_l1_ptr uint32_t* words = km::l1(results);
    words[0] = consumed;
    words[1] = total;
    words[2] = starved;
}));
"""
COUNTERS = "uint32_t consumed = 0, total = 0, starved = 0;"
RESET = "consumed = 0; total = 0; starved = 0;"
WALK = """
volatile tt_l1_ptr uint32_t* nodes = km::l1(scene);
const uint32_t hit = nodes[message[kRayOrigin] & 255];
if (message[kRayDepth] == 0 || km_messages.quota() == 0) {
    starved += message[kRayDepth] != 0;
    consumed += 1;
    total += message[kRayOrigin];
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


def log(message):
    print(f"[{time.monotonic() - STARTED:6.2f}] {message}", flush=True)


class Ray(km.Message):
    origin: int
    depth: int


class Lines(km.Program):
    """The step in lines: the hit and the quota are bodies' words."""

    @km.handler
    def trace(self, count: int, depth: int, scene: int, results: int):
        consumed = 0
        total = 0
        starved = 0
        if self.seeding():
            part = self.partition(count)
            for index in range(part.count):
                seed = Ray(origin=part.start + index, depth=depth)
                self.send(self.rank, seed)
        for message in self.arrive():
            hit = km.cxx(HIT, takes=[scene, message.origin], gives=int)
            left = km.cxx(QUOTA, gives=int)
            if message.depth == 0 or left == 0:
                if message.depth != 0:
                    starved += 1
                consumed += 1
                total += message.origin
                self.consume()
            else:
                message.depth -= 1
                child = Ray(origin=message.origin + hit, depth=message.depth)
                self.send(child.origin % self.size, child)
                self.forward(message.origin % self.size, message)
        km.cxx(REPORT, takes=[results, consumed, total, starved])


class Body(km.Program):
    """The same step in one body, as before a body's word could be read in lines."""

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


def walk(seeds, depth, scene):
    """The rules in Python: (rays consumed, the sum of their origins mod 2^32)."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="")
    cards = [int(c) for c in parser.parse_args().cards.split(",") if c]
    scene = np.random.default_rng(1).integers(0, 1024, NODES).astype(np.int32)
    want = walk(SEEDS, DEPTH, scene)
    with km.Session(log=log, card=next(iter(cards), None)) as session:
        grid = session.grid
        every = grid[0] * grid[1]
        group = (0, min(16, every))
        scenes = session.tensor(np.tile(scene, (every, 1)), "int32", shard_grid=grid)
        for name, cls in (("lines", Lines), ("body", Body)):
            micros, totals, kinds, text = [], set(), set(), None
            for _ in range(RUNS):
                zeros = np.zeros((every, 8), np.int32)
                results = session.tensor(zeros, "int32", shard_grid=grid)
                program = cls(log=lambda line: None).dynamic("trace", **OPTIONS)
                run = session.load(program)
                try:
                    words = (SEEDS, DEPTH, scenes.address, results.address)
                    handle = run.submit(cls.trace, group, args=words, exclusive=True)
                    run.drain(timeout=20.0)
                finally:
                    run.stop()
                event = run.event(handle)
                kinds.add(event.kind)
                micros.append(event.microseconds)
                text = program.sizes.text if program.sizes else None
                rows = results.read().astype(np.int64) & 0xFFFFFFFF
                totals.add(
                    (
                        int(rows[:, 0].sum()),
                        int(rows[:, 1].sum() % (1 << 32)),
                        int(rows[:, 2].sum()),
                    )
                )
                results.free()
            row = {
                "form": name,
                "kinds": sorted(kinds),
                "totals": sorted(totals),
                "python": list(want),
                "equal": totals == {(*want, 0)},
                "us_median": round(statistics.median(micros), 1),
                "us": [round(m, 1) for m in micros],
                "text_bytes": text,
            }
            print(json.dumps(row), flush=True)
        scenes.free()
    log("done")


if __name__ == "__main__":
    main()
