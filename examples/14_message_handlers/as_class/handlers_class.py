"""../message_handlers.py as a class: three dynamic handlers, each a method looping
over the messages that name it, compiled (build/messagehandlers-<hash>/ here);
each core's counts a field of words the host reads after the run, checked against the
replay as the hand program's (beside it with --beside). See README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import message_handlers as hand  # noqa: E402  (the hand program, its options, check)
import replay  # noqa: E402

from komorebi.layout import counter_tiles  # noqa: E402

STARTED = time.monotonic()
# The words of a core's totals, as the hand program's results words.
LOOKUPS, VALUES, REPORTS, OWN, SUM_LOW, SUM_HIGH, PROBES, WRONG = range(8)


class Thread(km.Message):
    """A thread: its number, the rank that seeded it, the key it looks up, the value
    found, the lookups left, the sum of its values (mod 2^32), the cores asked."""

    thread: int
    origin: int
    key: int
    value: int
    steps: int
    sum: int
    probes: int


class MessageHandlers(km.Program):
    """lookup at a key's owner, value at the value's owner, report at the thread's
    origin; the table a 32-bit mixer (words wrap at 2^32), so the cores load nothing."""

    WORDS = km.constant(16)  # the words of a core's totals
    totals: km.L1[km.u32, WORDS]

    @km.handler
    def lookup(self, threads: int, steps: int):
        for word in range(self.WORDS):
            self.totals[word] = 0
        if self.seeding():  # the seeds have the RUN command's handler: lookup
            part = self.partition(threads)
            for index in range(part.count):
                key = part.start + index + 1  # first key: the mix of thread + 1
                key ^= key >> 16
                key *= 0x7FEB352D
                key ^= key >> 15
                key *= 0x846CA68B
                key ^= key >> 16
                seed = Thread(
                    thread=part.start + index, origin=self.rank, key=key, steps=steps
                )
                self.send(key % self.size, seed)
        for message in self.arrive():
            self.totals[LOOKUPS] += 1
            message.probes += 1
            held = message.key ^ self.rank * 0x9E3779B1  # whether this rank holds it
            held ^= held >> 16
            held *= 0x7FEB352D
            held ^= held >> 15
            held *= 0x846CA68B
            held ^= held >> 16
            last = (message.key % self.size + self.size - 1) % self.size
            if self.rank != last and held & 3 == 0:
                after = (self.rank + 1) % self.size
                self.totals[OWN] += after == self.rank
                self.forward(after, message)  # still lookup
            else:
                value = message.key + 0x632BE5AB  # the key's value
                value ^= value >> 16
                value *= 0x7FEB352D
                value ^= value >> 15
                value *= 0x846CA68B
                value ^= value >> 16
                message.value = value
                owner = value % self.size
                self.totals[OWN] += owner == self.rank
                self.forward(owner, message, to=MessageHandlers.value)

    @km.handler
    def value(self):
        for message in self.arrive():
            self.totals[VALUES] += 1
            message.sum += message.value
            message.steps -= 1
            if message.steps != 0:
                key = message.value ^ 0x85EBCA6B  # the next key
                key ^= key >> 16
                key *= 0x7FEB352D
                key ^= key >> 15
                key *= 0x846CA68B
                key ^= key >> 16
                message.key = key
                owner = key % self.size
                self.totals[OWN] += owner == self.rank
                self.forward(owner, message, to=MessageHandlers.lookup)
            else:
                self.totals[OWN] += message.origin == self.rank
                self.forward(message.origin, message, to=MessageHandlers.report)

    @km.handler
    def report(self):
        for message in self.arrive():
            self.totals[REPORTS] += 1
            self.totals[WRONG] += message.origin != self.rank
            self.totals[SUM_HIGH] += self.totals[SUM_LOW] + message.sum < message.sum
            self.totals[SUM_LOW] += message.sum
            self.totals[PROBES] += message.probes
            self.consume()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def checked(args, count, rows, event):
    """Each core's totals against the replay, as the hand program checks its
    results words (the class has no run word: the totals are zeroed by each run)."""
    workers = len(counter_tiles(0, count, args.counters).workers)
    expected = replay.replay(args.threads, args.steps, workers)
    got = {
        "lookups": rows[:workers, LOOKUPS],
        "values": rows[:workers, VALUES],
        "reports": rows[:workers, REPORTS],
        "own": rows[:workers, OWN],
        "probes": rows[:workers, PROBES],
        "sum": (rows[:workers, SUM_LOW] | rows[:workers, SUM_HIGH] << 32).astype(
            np.uint64
        ),
    }
    differ = [name for name in got if not (got[name] == expected[name]).all()]
    exact = not differ and not rows[:, WRONG].any()
    handled = int(sum(got[name].sum() for name in ("lookups", "values", "reports")))
    microseconds = (
        round(event.microseconds, 1) if event and event.kind == "done" else None
    )
    return {
        "event": event.kind if event else None,
        "run_microseconds": microseconds,
        "exact": bool(exact),
        "differ": differ,
        "messages": handled,
        "by_handler": [
            int(got[name].sum()) for name in ("lookups", "values", "reports")
        ],
        "own_core_forwards": int(got["own"].sum()),
        "messages_per_second": (
            round(handled / microseconds * 1e6) if microseconds else None
        ),
    }


def run_class(session, args):
    """One load of the class, its RUN command (lookup), the end and the check."""
    grid = session.grid
    cores = grid[0] * grid[1]
    totals = session.tensor(
        np.zeros((cores, MessageHandlers.WORDS.default), np.uint32),
        "uint32",
        shard_grid=grid,
    )
    program = MessageHandlers(totals=totals, log=log).dynamic(
        "lookup", "value", "report", **hand.options(args)
    )
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    count = args.count or cores
    handle = run.submit(
        MessageHandlers.lookup,
        (0, count),
        args=(args.threads, args.steps),
        exclusive=True,
    )
    timed_out = None
    try:
        run.drain(timeout=args.deadline + 5.0)
        run.stop()
    except km.CommandTimeout as error:
        timed_out = str(error)
        log(f"timeout: {error}")
    rows = (totals.read().astype(np.int64) & 0xFFFFFFFF)[:count]
    gave = {
        state["core"]: state["dynamic"]["gave_up"]
        for state in run.state()[:count]
        if state["dynamic"]["gave_up"]
    }
    record = checked(args, count, rows, run.event(handle))
    record |= {"kernel_text": program.sizes.text, "messages_gave": gave}
    record["timed_out"] = timed_out
    record["ok"] = record["exact"] and record["event"] == "done" and not gave
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--beside", action="store_true", help="the hand program first")
    parser.add_argument("--count", type=int, default=0, help="group: cores 0 to n - 1")
    parser.add_argument("--threads", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=10, help="lookups per thread")
    parser.add_argument("--repeat", type=int, default=1, help="runs, a load each")
    parser.add_argument("--max-live", type=int, default=1 << 14)
    parser.add_argument("--queue-messages", type=int, default=1 << 14)
    parser.add_argument("--counters", type=int, default=2)
    parser.add_argument("--deadline", type=float, default=5.0)
    parser.add_argument("--wait", type=float, default=1.0)
    args = parser.parse_args()
    report = {"hand": [], "class": []}
    with km.Session(log=log) as session:
        log("device open")
        for _ in range(args.repeat):
            if args.beside:
                one = hand.run_once(session, args)
                one["kernel_text"] = None
                report["hand"].append(one)
                log(f"hand: {one['line']}")
            report["class"].append(run_class(session, args))
            log(f"class: {json.dumps(report['class'][-1], default=str)}")
    ok = all(one["ok"] for runs in report.values() for one in runs)
    for name, runs in list(report.items()):
        timed = [one["run_microseconds"] for one in runs if one["run_microseconds"]]
        if timed:
            report[f"{name}_run_microseconds"] = [
                min(timed),
                float(np.median(timed)),
                max(timed),
            ]
    report["ok"] = bool(ok)
    print(json.dumps(power.clocked(report), default=str), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
