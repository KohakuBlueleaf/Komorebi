"""Handlers sharing buffers: Three (30 indices as one set) built at shared indices of
other pages, every word checked, the time of a change of pages; Orders (the same pages
in other orders) built by shape, every word checked. One JSON line.
Example: python shared_buffers.py --cores 12"""

import argparse
import json
import statistics
import sys
import time

import numpy as np

import komorebi as km
from komorebi.bench import power

ITEMS = 64  # items a core of each tensor, in each handler
WORDS = 256  # a row of each tensor: its page, 1,024 B
CHANGES = 200  # commands of each timed run
ROUNDS = 5  # timed runs of each kind, the median kept
# The pack CPU's body of each of Three's handlers: an item's words (its bytes / 4).
MIX_256 = "shared_k::mix(ia, ib, ic, total, mixed, 64);"
MIX_512 = "shared_k::mix(ia, ib, ic, total, mixed, 128);"
MIX_1024 = "shared_k::mix(ia, ib, ic, total, mixed, 256);"


def log(line):
    print(line, file=sys.stderr, flush=True)


class Three(km.Program):
    """Three handlers of one form, items of 256, 512 and 1,024 B: each reads items of
    a, b and c and writes their sum to s and their exclusive-or to x, in one body on
    pack. Each handler takes 10 indices: 3 read pairs, 2 write pairs."""

    @km.handler
    def first(
        self,
        a: km.Tensor,
        b: km.Tensor,
        c: km.Tensor,
        s: km.Tensor,
        x: km.Tensor,
        items: int,
    ):
        base = self.rank * items
        total = km.Stream(s.items(first=base, count=items, size=256), write=True)
        mixed = km.Stream(x.items(first=base, count=items, size=256), write=True)
        for ia, ib, ic in zip(
            km.Stream(a.items(first=base, count=items, size=256)),
            km.Stream(b.items(first=base, count=items, size=256)),
            km.Stream(c.items(first=base, count=items, size=256)),
        ):
            km.cxx(
                MIX_256,
                takes=[ia, ib, ic, total, mixed],
                on="pack",
                include=["../../kernels/shared_buffers.hpp"],
            )

    @km.handler
    def second(
        self,
        a: km.Tensor,
        b: km.Tensor,
        c: km.Tensor,
        s: km.Tensor,
        x: km.Tensor,
        items: int,
    ):
        base = self.rank * items
        total = km.Stream(s.items(first=base, count=items, size=512), write=True)
        mixed = km.Stream(x.items(first=base, count=items, size=512), write=True)
        for ia, ib, ic in zip(
            km.Stream(a.items(first=base, count=items, size=512)),
            km.Stream(b.items(first=base, count=items, size=512)),
            km.Stream(c.items(first=base, count=items, size=512)),
        ):
            km.cxx(
                MIX_512,
                takes=[ia, ib, ic, total, mixed],
                on="pack",
                include=["../../kernels/shared_buffers.hpp"],
            )

    @km.handler
    def third(
        self,
        a: km.Tensor,
        b: km.Tensor,
        c: km.Tensor,
        s: km.Tensor,
        x: km.Tensor,
        items: int,
    ):
        base = self.rank * items
        total = km.Stream(s.items(first=base, count=items, size=1024), write=True)
        mixed = km.Stream(x.items(first=base, count=items, size=1024), write=True)
        for ia, ib, ic in zip(
            km.Stream(a.items(first=base, count=items, size=1024)),
            km.Stream(b.items(first=base, count=items, size=1024)),
            km.Stream(c.items(first=base, count=items, size=1024)),
        ):
            km.cxx(
                MIX_1024,
                takes=[ia, ib, ic, total, mixed],
                on="pack",
                include=["../../kernels/shared_buffers.hpp"],
            )


# Orders' bodies: each read item times 3 plus the handler's key, into the write
# stream of its size (an item's words: 64, 128, 256).
TURN_0 = (
    "shared_k::scale(ia, oa, 64, 1); shared_k::scale(ib, ob, 128, 1); "
    "shared_k::scale(ic, oc, 256, 1);"
)
TURN_1 = (
    "shared_k::scale(ia, oa, 128, 2); shared_k::scale(ib, ob, 256, 2); "
    "shared_k::scale(ic, oc, 64, 2);"
)
TURN_2 = (
    "shared_k::scale(ia, oa, 256, 3); shared_k::scale(ib, ob, 64, 3); "
    "shared_k::scale(ic, oc, 128, 3);"
)


class Orders(km.Program):
    """Three handlers reading items of p (256 B), q (512 B) and r (1,024 B) in three
    orders, each item times 3 plus the handler's key into u, v and w: 36 indices by a
    stream's place in its handler, 12 by shape."""

    @km.handler
    def turn0(
        self,
        p: km.Tensor,
        q: km.Tensor,
        r: km.Tensor,
        u: km.Tensor,
        v: km.Tensor,
        w: km.Tensor,
        items: int,
    ):
        base = self.rank * items
        oa = km.Stream(u.items(first=base, count=items, size=256), write=True)
        ob = km.Stream(v.items(first=base, count=items, size=512), write=True)
        oc = km.Stream(w.items(first=base, count=items, size=1024), write=True)
        for ia, ib, ic in zip(
            km.Stream(p.items(first=base, count=items, size=256)),
            km.Stream(q.items(first=base, count=items, size=512)),
            km.Stream(r.items(first=base, count=items, size=1024)),
        ):
            km.cxx(
                TURN_0,
                takes=[ia, ib, ic, oa, ob, oc],
                on="pack",
                include=["../../kernels/shared_buffers.hpp"],
            )

    @km.handler
    def turn1(
        self,
        p: km.Tensor,
        q: km.Tensor,
        r: km.Tensor,
        u: km.Tensor,
        v: km.Tensor,
        w: km.Tensor,
        items: int,
    ):
        base = self.rank * items
        oa = km.Stream(v.items(first=base, count=items, size=512), write=True)
        ob = km.Stream(w.items(first=base, count=items, size=1024), write=True)
        oc = km.Stream(u.items(first=base, count=items, size=256), write=True)
        for ia, ib, ic in zip(
            km.Stream(q.items(first=base, count=items, size=512)),
            km.Stream(r.items(first=base, count=items, size=1024)),
            km.Stream(p.items(first=base, count=items, size=256)),
        ):
            km.cxx(
                TURN_1,
                takes=[ia, ib, ic, oa, ob, oc],
                on="pack",
                include=["../../kernels/shared_buffers.hpp"],
            )

    @km.handler
    def turn2(
        self,
        p: km.Tensor,
        q: km.Tensor,
        r: km.Tensor,
        u: km.Tensor,
        v: km.Tensor,
        w: km.Tensor,
        items: int,
    ):
        base = self.rank * items
        oa = km.Stream(w.items(first=base, count=items, size=1024), write=True)
        ob = km.Stream(u.items(first=base, count=items, size=256), write=True)
        oc = km.Stream(v.items(first=base, count=items, size=512), write=True)
        for ia, ib, ic in zip(
            km.Stream(r.items(first=base, count=items, size=1024)),
            km.Stream(p.items(first=base, count=items, size=256)),
            km.Stream(q.items(first=base, count=items, size=512)),
        ):
            km.cxx(
                TURN_2,
                takes=[ia, ib, ic, oa, ob, oc],
                on="pack",
                include=["../../kernels/shared_buffers.hpp"],
            )


SIZES = {"first": 256, "second": 512, "third": 1024}  # an item's bytes
KEYS = {"turn0": 1, "turn1": 2, "turn2": 3}  # Orders' handlers' keys
ORDER_SIZES = (256, 512, 1024)  # p, q, r (and u, v, w): an item's bytes


def wanted(inputs, size, cores):
    """Each handler's s and x (words) from the inputs, zero past what it writes."""
    a, b, c = (t.reshape(-1) for t in inputs)
    words = cores * ITEMS * size // 4
    s = np.zeros_like(a)
    x = np.zeros_like(a)
    s[:words] = a[:words] + b[:words] + c[:words]
    x[:words] = a[:words] ^ b[:words] ^ c[:words]
    return s, x


def checked(session, outputs, want):
    """Words of the outputs unlike numpy's."""
    wrong = 0
    for tensor, words in zip(outputs, want):
        got = session.read(tensor)[0].view(np.uint32).reshape(-1)
        wrong += int((got != words).sum())
    return wrong


def batch(run, program, handlers, cores, words):
    """`handlers` (one a command) queued at once (their words made before) and
    waited: microseconds a command, and whether every command was done."""
    numbers = np.array([int(getattr(program, name)) for name in handlers])
    rows = np.array([getattr(Three, name).args(*words[name]) for name in handlers])
    started = time.perf_counter()
    handles = list(run.submit_many(numbers, 0, cores, rows))
    events = run.wait(handles, timeout=60.0)
    seconds = time.perf_counter() - started
    done = all(event.kind == "done" for event in events)
    return seconds * 1e6 / len(handlers), done


def sharing(program):
    """The report's lines of how the handlers share buffers."""
    lines = program.built.report
    first = next(i for i, line in enumerate(lines) if line.startswith("  buffers: "))
    end = first + 1
    while end < len(lines) and lines[end].startswith("    "):
        end += 1
    return [line.strip() for line in lines[first:end]]


def orders(session, cores, rng):
    """Orders made (by shape), each handler run once, every word of u, v and w
    checked: {handler: words wrong}, its report's lines."""
    rows = cores * ITEMS * max(ORDER_SIZES) // (4 * WORDS)
    values = [
        rng.integers(0, 2**32, (rows, WORDS), dtype=np.uint32) for _ in ORDER_SIZES
    ]
    p, q, r = (session.empty((rows, WORDS), "uint32") for _ in ORDER_SIZES)
    for tensor, value in zip((p, q, r), values):
        session.write(tensor, value.view(np.uint8))
    u, v, w = (session.empty((rows, WORDS), "uint32") for _ in ORDER_SIZES)
    program = Orders(p=p, q=q, r=r, u=u, v=v, w=w, session=session, log=log)
    lines = sharing(program)
    for line in lines:
        print(f"Orders: {line}", flush=True)
    zero = np.zeros((rows, WORDS * 4), np.uint8)
    wrong = {}
    run = session.load(program)
    try:
        for name, key in KEYS.items():
            for tensor in (u, v, w):
                session.write(tensor, zero)
            event = run.wait(
                run.submit(getattr(Orders, name), (0, cores),
                           args=(p, q, r, u, v, w, ITEMS)),
                30.0,
            )  # fmt: skip
            want = []
            for value, size in zip(values, ORDER_SIZES):
                words = cores * ITEMS * size // 4
                out = np.zeros(rows * WORDS, np.uint32)
                flat = value.reshape(-1)
                out[:words] = flat[:words] * np.uint32(3) + np.uint32(key)
                want.append(out)
            wrong[name] = checked(session, (u, v, w), want)
            wrong[name] += 0 if event.kind == "done" else 1
            log(f"Orders.{name}: {event.kind}, {wrong[name]} words wrong")
    finally:
        run.stop()
    return wrong, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=12, help="0: the session's ranks")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    report = {}
    with km.Session(log=log) as session:
        cores = args.cores or session.ranks
        rng = np.random.default_rng(args.seed)
        rows = cores * ITEMS * max(SIZES.values()) // (4 * WORDS)
        values = [
            rng.integers(0, 2**32, (rows, WORDS), dtype=np.uint32) for _ in range(3)
        ]
        a, b, c = (session.empty((rows, WORDS), "uint32") for _ in range(3))
        for tensor, value in zip((a, b, c), values):
            session.write(tensor, value.view(np.uint8))
        outputs = {
            name: tuple(session.empty((rows, WORDS), "uint32") for _ in range(2))
            for name in SIZES
        }
        given = {"a": a, "b": b, "c": c, "s": outputs["first"][0]}
        program = Three(**given, x=outputs["first"][1], session=session, log=log)
        report["folder"] = program.built.folder.name
        report["cores"] = cores
        report["three"] = sharing(program)
        for line in report["three"]:
            print(f"Three: {line}", flush=True)
        words = {name: (a, b, c, *outputs[name], ITEMS) for name in SIZES}
        zero = np.zeros((rows, WORDS * 4), np.uint8)
        for pair in outputs.values():
            for tensor in pair:
                session.write(tensor, zero)
        want = {name: wanted(values, size, cores) for name, size in SIZES.items()}
        run = session.load(program)
        try:
            wrong = {}
            for name in SIZES:
                handler = getattr(Three, name)
                event = run.wait(
                    run.submit(handler, (0, cores), args=words[name]), 30.0
                )
                wrong[name] = checked(session, outputs[name], want[name])
                log(f"{name}: {event.kind}, {wrong[name]} words wrong")
                wrong[name] += 0 if event.kind == "done" else 1
            alone, seconds, both = [], [], []
            for _ in range(ROUNDS):
                alone.append(batch(run, program, ["first"] * CHANGES, cores, words))
                seconds.append(batch(run, program, ["second"] * CHANGES, cores, words))
                both.append(
                    batch(
                        run, program, ["first", "second"] * (CHANGES // 2), cores, words
                    )
                )
            for name in ("first", "second"):
                wrong[name] += checked(session, outputs[name], want[name])
        finally:
            run.stop()
        order_wrong, report["orders"] = orders(session, cores, rng)
    one, other, two = (
        statistics.median(time for time, _ in runs) for runs in (alone, seconds, both)
    )
    # second moves twice first's bytes: a change's cost is the alternating command
    # against the mean of the two handlers alone (and, as a bound, against first's)
    change = two - (one + other) / 2
    report |= {
        "wrong": wrong,
        "orders_wrong": order_wrong,
        "first_alone_us": round(one, 3),
        "second_alone_us": round(other, 3),
        "alternating_us": round(two, 3),
        "against_first_us": round(two - one, 3),
        "change_us": round(change, 3),
        "all_done": all(done for _, done in alone + seconds + both),
    }
    log(f"a command: {one:.3f} us first alone, {other:.3f} us second alone")
    log(f"{two:.3f} us alternating, {two - one:.3f} us more than first alone")
    log(f"a command of the other handler's pages: {change:.3f} us more")
    report["ok"] = (
        report["all_done"] and not any(wrong.values()) and not any(order_wrong.values())
    )
    print(json.dumps(power.clocked(report), default=float), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
