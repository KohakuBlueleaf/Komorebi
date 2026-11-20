"""The head of DeepSeek-V4.1-Flash and the greedy choice of the token: logits over the
129,280 ids dealt over islands (a quarter each), the largest logit's id, of equal ones
the first (README.md). `python head.py card --cards A,B` checks it against float64."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats

VOCAB, DIM = 129280, 5120
K_TILES = DIM // 32  # 160
CORES, WIDTH = 73, 14  # cores an island and tile slots a core
SLICE = VOCAB // 32 // 4  # 1,010: the tile columns of a slice head.s0 .. s3
SHORT = CORES * WIDTH - SLICE  # 12: the first ranks' starts 13 apart, the others' 14
PART = 2  # the row functions' blocks: a core's row read in parts of 2 tiles
NOWHERE = 1e9  # an id past the vocabulary: no candidate
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def row_max(row):
    return km.reduce_max(row, exact=True).col()


@km.tile_function
def negated_min(row):
    return km.reduce_max(row * -1.0, exact=True).col()


@km.tile_function
def candidate(t, g, ids):
    return km.where(t == g.col(), ids, NOWHERE)


@km.tile_function
def show(t, s):
    return t * 0.0 + s.col()


@km.tile_function
def show_negated(t, s):
    return t * 0.0 - s.col()


# one set of compute options for every class of a load
COMPUTE = {"math_fidelity": "HiFi2", "dst_full_sync_en": True, "fp32_dest_acc_en": True}


class HeadLogits(km.Program, compute=COMPUTE):
    """logits of one island (README.md): a core's columns of W (the pages from `at` of
    a one-column arena) in two passes, each pass's sums in DST, to its slots of y."""

    K_TILES = km.constant(K_TILES)
    WIDTH = km.constant(WIDTH)
    HALF = WIDTH // 2
    SLICE = km.constant(SLICE)
    SHORT = km.constant(SHORT)
    CHUNK = km.constant(2)  # K tiles a block

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def logits(self, x: km.Tensor, w: km.Tensor, y: km.Tensor, at: int):
        slot = self.rank * self.WIDTH
        start = at + slot - min(self.rank, self.SHORT)
        for half in range(2):
            r0 = start + half * self.HALF
            acc = km.tile.zeros(km.fp32, self.HALF)
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0 : self.K_TILES], block=self.CHUNK),
                km.Stream(
                    w.tiles[
                        r0 : r0 + self.K_TILES * self.SLICE : self.SLICE, 0 : self.HALF
                    ],
                    block=self.CHUNK * self.HALF,
                ),
            ):
                acc += self.project(xb, wb)
            c0 = slot + half * self.HALF
            y.tiles[0:1, c0 : c0 + self.HALF].write(acc)


class HeadChoice(km.Program, compute=COMPUTE):
    """choose and least of one island (module docstring); a class of its own beside
    HeadLogits, whose cores switch to it (one class's writer is over NCRISC's code
    region on islands)."""

    WIDTH = km.constant(WIDTH)
    PART = km.constant(PART)
    gx: km.L1[km.fp32, 1, 32, 32]
    least_part: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def choose(self, y: km.Tensor, ids: km.Tensor, c: km.Tensor, g: km.Tensor):
        first = self.rank * self.WIDTH
        for row in km.Stream(
            y.tiles[0:1, first : first + self.WIDTH], block=self.PART, row=self.WIDTH
        ):
            self.gx.write(row_max(row))
        km.all_reduce(self.gx, op="max", local=True)
        picked = km.Stream(
            c.tiles[0:1, first : first + self.WIDTH], write=True, block=self.PART
        )
        for t, i in zip(
            km.Stream(y.tiles[0:1, first : first + self.WIDTH], block=self.PART),
            km.Stream(ids.tiles[0:1, first : first + self.WIDTH], block=self.PART),
        ):
            picked.push(candidate(t, self.gx, i))
        largest = km.Stream(g.pages(self.rank, 1), write=True, block=1)
        for t in km.Stream(y.tiles[0:1, first : first + 1], block=1):
            largest.push(show(t, self.gx))

    @km.handler
    def least(self, c: km.Tensor, r: km.Tensor):
        first = self.rank * self.WIDTH
        for row in km.Stream(
            c.tiles[0:1, first : first + self.WIDTH], block=self.PART, row=self.WIDTH
        ):
            self.least_part.write(negated_min(row))
        km.all_reduce(self.least_part, op="max", local=True)
        if self.rank == self.leader:
            out = km.Stream(r.pages(0, 1), write=True, block=1)
            for t in km.Stream(c.pages(0, 1), block=1):
                out.push(show_negated(t, self.least_part))


def slot_columns():
    """Each slot's tile column of the slice: rank r's 14 from r x 14 - min(r, SHORT), so
    a short rank's last slot repeats the next rank's first column."""
    r = np.arange(CORES)
    first = r * WIDTH - np.minimum(r, SHORT)
    return (first[:, None] + np.arange(WIDTH)[None, :]).reshape(-1)


def slice_pages(rows):
    """Slice pages as the loader holds head.sN (bf16 [DIM, 32,320] in the tile order)
    from `rows`, the head's 32,320 rows of the slice (float32 of bf16 values)."""
    raw, _ = formats.to_bytes(np.asarray(rows, np.float32).T, "bfloat16", "tile")
    return np.asarray(raw)


def slot_pages(q):
    """Slice q's ids: a float32 tile row, every row each slot's global ids."""
    cols = np.repeat(slot_columns(), 32) * 32 + np.tile(np.arange(32), CORES * WIDTH)
    ids = (q * SLICE * 32 + cols).astype(np.float32)
    return np.asarray(formats.to_bytes(np.tile(ids, (32, 1)), "float32", "tile")[0])


class Tensors:
    """The head's tensors, a copy on every island (each island writes its own)."""

    def __init__(self, session, islands):
        copy = {"copy": True} if islands else {}
        t = lambda shape, dt: session.empty(shape, dt, "tile", **copy)  # noqa: E731
        n = CORES * WIDTH * 32
        self.x = t((32, DIM), "bfloat16")
        self.w = t((K_TILES * SLICE * 32, 32), "bfloat16")  # an arena's form
        self.y = t((32, n), "float32")
        self.ids = t((32, n), "float32")
        self.c = t((32, n), "float32")
        self.g = t((32, CORES * 32), "float32")
        self.r = t((32, 32), "float32")
        self.session, self.islands = session, islands

    def put(self, tensor, i, raw):
        pages = np.ascontiguousarray(raw).view(np.uint8).reshape(tensor.pages, -1)
        if self.islands:
            self.session.sessions[i].write(tensor.parts[i], pages)
        else:
            self.session.write(tensor, pages)

    def get(self, tensor, i, first=0, pages=1):
        s = self.session.sessions[i] if self.islands else self.session
        part = tensor.parts[i] if self.islands else tensor
        (raw,) = s.read(part, first=first, pages=pages)
        return np.asarray(raw).reshape(-1).view(np.uint8)

    def write_x(self, i, x):
        raw, _ = formats.to_bytes(x, "bfloat16", "tile")
        self.put(self.x, i, raw)

    def tile_values(self, tensor, i, page):
        """Page `page` of a float32 tile tensor as its 32 x 32 values."""
        bits = self.get(tensor, i, page, 1)
        return np.asarray(formats.from_bytes(bits, (32, 32), "float32", "tile"))


class HeadRun:
    """The head loaded on a session (an island an entry of `islands`, or its one card),
    slice quarters[i] on island i: `choose(x)` returns each row's id and largest logit
    and the commands' microseconds; `head_of(q)` gives slice q's rows of the head."""

    def __init__(self, session, head_of, islands=None, quarters=(0,)):
        self.session = session
        count = len(islands) if islands else 1
        self.ts = ts = Tensors(session, bool(islands))
        self.quarters = quarters
        for i, q in enumerate(quarters):
            ts.put(ts.w, i, slice_pages(head_of(q)))
            ts.put(ts.ids, i, slot_pages(q))
            log(f"island {i}: slice {q} written")
        self.firsts = [session.graph.first(i) for i in range(count)] if islands else [0]
        # one compute build for both (a list shares cores): the logits take the choice's
        # float32 unpacking (beside) and its full DST sync
        self.cp = HeadChoice(
            log=log,
            session=session,
            form="runtime",
            y=ts.y,
            ids=ts.ids,
            c=ts.c,
            g=ts.g,
            r=ts.r,
        )
        self.lp = HeadLogits(
            log=log,
            session=session,
            form="runtime",
            beside=self.cp,
            x=ts.x,
            w=ts.w,
            y=ts.y,
        )
        # every core holds both; a group switches program between the commands
        self.run = session.load([self.cp, self.lp])

    def _wait(self, ticket, name):
        event = self.run.wait(ticket, timeout=60.0)
        parts = getattr(event, "parts", (event,))
        if any(p.kind != "done" for p in parts):
            raise km.KomorebiError(f"{name}: {event}")
        return max(p.microseconds for p in parts)

    def commands(self):
        """Each command sent to every island and waited for before the next (a
        collective's ports stay held until it ends): their microseconds."""
        ts, run, lp, cp = self.ts, self.run, self.lp, self.cp
        steps = (
            ("logits", lp.logits, (ts.x, ts.w, ts.y, 0)),
            ("choose", cp.choose, (ts.y, ts.ids, ts.c, ts.g)),
            ("least", cp.least, (ts.c, ts.r)),
        )
        us = {}
        for name, handler, args in steps:
            tickets = [run.submit(handler, (f, CORES), args=args) for f in self.firsts]
            us[name] = [self._wait(t, name) for t in tickets]
        return us

    def choose(self, x):
        ts = self.ts
        t0 = time.monotonic()
        for i in range(len(self.firsts)):
            ts.write_x(i, x)
        us = self.commands()
        best_v = np.full(32, -np.inf)
        best_i = np.full(32, -1, np.int64)
        for i in range(len(self.firsts)):
            v = ts.tile_values(ts.g, i, 0)[:, 0].astype(np.float64)
            idx = ts.tile_values(ts.r, i, 0)[:, 0].astype(np.int64)
            take = (
                v > best_v
            )  # islands in id order: an equal later one keeps the earlier
            best_v = np.where(take, v, best_v)
            best_i = np.where(take, idx, best_i)
        us["wall_ms"] = (time.monotonic() - t0) * 1e3
        return best_i, best_v, us

    def logits(self, i):
        """Island i's logits (float32 [32, its slice's 32,320 ids])."""
        ts = self.ts
        bits = ts.get(ts.y, i, 0, ts.y.pages)
        return slot_logits(formats.from_bytes(bits, ts.y.shape, "float32", "tile"))

    def stop(self):
        self.run.stop()


def slot_logits(y):
    """The slice's logits in id order from y's slots (float32 [32, 32,320])."""
    out = np.empty((32, SLICE, 32), np.float32)
    out[:, slot_columns()] = np.asarray(y, np.float32).reshape(32, CORES * WIDTH, 32)
    return out.reshape(32, SLICE * 32)


def as_bf16(values):
    bits = (np.asarray(values, np.float32).view(np.uint32) + 0x8000) & 0xFFFF0000
    return bits.view(np.float32)


def made_inputs(seed, plant):
    """Random bf16 head rows (normal / sqrt(DIM)) and 32 random bf16 rows of x; with
    `plant`, rows 0..3 of x tie two ids exactly (equal head rows: the first must win).
    """
    rng = np.random.default_rng(seed)
    head = as_bf16(
        rng.standard_normal((VOCAB, DIM), dtype=np.float32) / np.float32(np.sqrt(DIM))
    )
    x = as_bf16(rng.standard_normal((32, DIM), dtype=np.float32))
    if plant:
        pairs = ((5, 70000), (3, 129279), (64000, 64001), (100, 32319))
        for r, (a, b) in enumerate(pairs):
            head[b] = head[a]
            x[r] = as_bf16(head[a] * 40.0)
    return head, x


def card(args):
    """The head's shares --quarter, --quarter + 1, ... of four (a card each) on --cards:
    each row's id and largest logit, the logits against float64 of the stored values,
    the commands' times; saved to --out (npz) for `combine`."""
    cards = [int(c) for c in args.cards.split(",")]
    log("making the head")
    head, x = made_inputs(args.seed, args.plant)
    opened = {"cards": cards, "links": 1} if len(cards) > 1 else {"card": cards[0]}
    islands = list(range(len(cards))) if len(cards) > 1 else None
    quarters = range(args.quarter, args.quarter + len(cards))
    span = SLICE * 32
    with km.Session(log=log, **opened) as session:
        hr = HeadRun(
            session, lambda q: head[q * span : (q + 1) * span], islands, quarters
        )
        try:
            runs = [hr.choose(x) for _ in range(args.repeat)]
            logits = [hr.logits(i) for i in range(len(hr.firsts))]
        finally:
            hr.stop()
    got = np.concatenate(logits, axis=1).astype(np.float64)
    lo, hi = quarters[0] * span, (quarters[-1] + 1) * span
    log("float64 reference")
    want = x.astype(np.float64) @ head[lo:hi].T.astype(np.float64)
    order = np.argsort(-want, axis=1, kind="stable")
    top1, top2 = order[:, 0], order[:, 1]
    rows = np.arange(32)
    err = np.abs(got - want)
    ids, vals, _ = runs[-1]
    report = {
        "cards": cards,
        "columns": [lo, hi],
        "rows_equal": int((ids == top1 + lo).sum()),
        "rows": 32,
        "ids": ids[:6].tolist(),
        "want": (top1[:6] + lo).tolist(),
        "runs_same": all(np.array_equal(r[0], ids) for r in runs),
        "smallest_gap": float((want[rows, top1] - want[rows, top2]).min()),
        "logit_rel_rms": float(np.sqrt((err**2).mean() / (want**2).mean())),
        "logit_max_err": float(err.max()),
        "logit_max_err_at": [int(v) for v in np.unravel_index(err.argmax(), err.shape)],
        "logit_max_err_rows": [float(e) for e in err.max(1)[:6]],
        "value_max_err": float(np.abs(vals - want[rows, top1]).max()),
        "us": [r[2] for r in runs],
    }
    print(json.dumps(report, default=float), flush=True)
    if args.out:
        np.savez(
            args.out,
            ids=ids,
            vals=vals,
            want_id=top1 + lo,
            want_val=want[rows, top1],
            second=want[rows, top2],
            lo=lo,
            hi=hi,
            us=json.dumps(report["us"]),
        )
    return 0


def combine(args):
    """The shares saved by `card --quarter` runs (--parts a.npz,b.npz,...), in id order:
    each row's id kept as the islands' host step keeps it, against float64's argmax over
    every share (of equal largest logits the first)."""
    parts = [np.load(p) for p in args.parts.split(",")]
    parts.sort(key=lambda d: int(d["lo"]))
    best_v, best_i = np.full(32, -np.inf), np.full(32, -1, np.int64)
    want_v, want_i = np.full(32, -np.inf), np.full(32, -1, np.int64)
    for d in parts:
        take = d["vals"] > best_v
        best_v, best_i = np.where(take, d["vals"], best_v), np.where(
            take, d["ids"], best_i
        )
        take = d["want_val"] > want_v
        want_v, want_i = np.where(take, d["want_val"], want_v), np.where(
            take, d["want_id"], want_i
        )
    us = [json.loads(str(d["us"])) for d in parts]
    report = {
        "parts": [[int(d["lo"]), int(d["hi"])] for d in parts],
        "rows_equal": int((best_i == want_i).sum()),
        "rows": 32,
        "ids": best_i[:6].tolist(),
        "want": want_i[:6].tolist(),
        "slowest_island_us": [  # a command's time counts from its sending: the last
            max(max(u[k]["least"]) for u in us) for k in range(min(map(len, us)))
        ],
    }
    print(json.dumps(report, default=float), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["card", "combine"])
    p.add_argument("--cards", default="0")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--plant", action="store_true")
    p.add_argument(
        "--quarter", type=int, default=0, help="the first card's share of four"
    )
    p.add_argument("--out", help="card: an npz of the ids, values and reference")
    p.add_argument("--parts", help="combine: the cards' npz files")
    args = p.parse_args()
    return {"card": card, "combine": combine}[args.what](args)


if __name__ == "__main__":
    sys.exit(main())
