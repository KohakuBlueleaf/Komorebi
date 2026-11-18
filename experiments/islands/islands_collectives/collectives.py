"""Collectives across islands: an all-gather (km::gather_pages, a neighbour island's
pages DRAM to DRAM by the link cores) and an all-reduce of two islands' partials
(gathered to their keepers, folded, gathered back), each word checked on the cores and,
up to 4 MiB, against numpy. Prints one JSON line. See README.md."""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import islands

KERNELS = Path(__file__).resolve().parent / "kernels"
PAGE = 4096
PAGE_PRIME = 0x9E3779B1  # collectives.h kPagePrime, kWordPrime
WORD_PRIME = 0x85EBCA77
MODE_RING = 1  # collectives.h kModeRing
REPORT_MAGIC, CHECK_MAGIC = 0x434F4C4C, 0x43484B21
NUMPY_PAGES = 1024  # rows up to 4 MiB are read back and compared on the host too
KERNEL_BYTES = 104 * 1024  # the reader's calls (mock build 97,344 B)
X280_HZ = 50e6  # a command's ticks (the X280's mtime)
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument(
        "--split", type=int, default=0, help="one card split into N islands in a line"
    )
    parser.add_argument(
        "--latency", action="store_true", help="one rank's small gathers, timed by step"
    )
    parser.add_argument("--repeat", type=int, default=10, help="commands a latency row")
    parser.add_argument(
        "--gather", type=int, nargs="*", default=[], metavar="KIB", help="sizes"
    )
    parser.add_argument(
        "--reduce", type=int, nargs="*", default=[], metavar="KIB", help="sizes"
    )
    parser.add_argument(
        "--ways", nargs="+", default=["0>1", "1>0", "both"], help="gather directions"
    )
    parser.add_argument(
        "--busy",
        type=int,
        nargs="+",
        default=[0],
        help="rounds of the compute's matrix loop beside each row (0: idle)",
    )
    parser.add_argument(
        "--issuers",
        type=int,
        nargs="+",
        default=[1],
        help="ranks posting a gather's shares (0: all)",
    )
    parser.add_argument("--json", help="also write the JSON line here")
    return parser.parse_args()


def pattern(first, pages, salt):
    """collectives.h's FILL pattern: word w of page p, p * PAGE_PRIME + w * WORD_PRIME +
    salt (uint32)."""
    p = np.arange(first, first + pages, dtype=np.uint64)[:, None]
    w = np.arange(PAGE // 4, dtype=np.uint64)[None, :]
    return ((p * PAGE_PRIME + w * WORD_PRIME + salt) % 2**32).astype(np.uint32)


def aiclk(cards):
    """Each card's AICLK now (MHz), from the driver's sysfs."""
    out = []
    for card in cards:
        path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
        out.append(int(path.read_text().split()[0]) if path.exists() else 0)
    return out


class Bench:
    """The session's islands, the program and its tensors: commands of each handler
    over an island's ranks, and what they report."""

    def __init__(self, session, pages, cards):
        self.session = session
        self.cards = cards
        graph = session.graph
        self.islands = len(graph.islands)
        self.own = [(graph.first(i), graph.count(i)) for i in range(self.islands)]
        shape = (pages, PAGE // 4)
        held = [self.own[i] for i in range(self.islands)]
        # a source and a target an island, the reduce's result (a copy), a report
        # page a rank and the stamps' page, each island's
        self.a = [session.empty(shape, "uint32", group=g) for g in held]
        self.b = [session.empty(shape, "uint32", group=g) for g in held]
        self.result = session.empty(shape, "uint32", copy=True)
        ranks = max(count for _, count in self.own)
        self.reports = [
            session.empty((ranks, PAGE // 4), "uint32", group=g) for g in held
        ]
        self.program = program(self.a[0])
        self.run = session.load(self.program, deadline=2.0)
        self.salt = 0x1000

    def handler(self, name):
        return self.program.handler(name)

    def together(self, submitted, timeout=30.0):
        """[(island, handle)] waited together: their events."""
        return self.session.wait_together(submitted, timeout)

    def submit(self, island, name, words, group=None):
        group = group or self.own[island]
        return island, self.run.island(island).submit(self.handler(name), group, words)

    def fill(self, island, tensor, pages):
        """FILL of pages 0 .. pages - 1 of `tensor` by the island's ranks: its salt."""
        self.salt += 0x10001
        words = [tensor.address, 0, pages, self.salt]
        self.together([self.submit(island, "fill", words)])
        return self.salt

    def check(self, island, tensor, pages, salt, other=0):
        """CHECK by the island's ranks: the bad words over the ranks, the first bad
        page (None: none)."""
        report = self.reports[island]
        words = [tensor.address, 0, pages, salt, other, report.address]
        self.together([self.submit(island, "check", words)])
        ranks = self.own[island][1]
        (got,) = self.session.read(report, first=0, pages=ranks)
        rows = got.view(np.uint32).reshape(ranks, -1)[:, :5]
        missing = int((rows[:, 0] != CHECK_MAGIC).sum())
        bad = int(rows[:, 2].sum()) + missing
        firsts = [int(r[3]) for r in rows if r[2]]
        return bad, (min(firsts) if firsts else None)

    def numpy_check(self, island, tensor, pages, want):
        """The pages read back from the island's part and compared with `want`."""
        (got,) = self.session.read(tensor, first=0, pages=pages)
        return int((got.view(np.uint32).reshape(pages, -1) != want).sum())

    def report_rows(self, island, ranks):
        (got,) = self.session.read(self.reports[island], first=0, pages=ranks)
        return got.view(np.uint32).reshape(-1, PAGE // 4)[:ranks, :8]

    def stamps(self, island, toward):
        """Link core 0's KM_PUT_STAMPS words (toward island `toward`) and the reading
        rank's wall clock."""
        words = [toward, self.reports[island].address]
        self.together([self.submit(island, "stamps", words, (self.own[island][0], 1))])
        (got,) = self.session.read(self.reports[island], first=0, pages=1)
        return [int(w) for w in got.view(np.uint32).reshape(-1)[:9]]

    def crossings(self):
        return self.run.crossings()


def program(tensor):
    """The experiment's program; every tensor has `tensor`'s accessor arguments."""
    return km.Program(
        "collectives",
        kernels=KERNELS,
        handlers=KERNELS / "collectives.h",
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=16 * PAGE, pages=1),
            km.Buffer(index=1, format="bfloat16", page_bytes=2048, pages=1),
            km.Buffer(index=2, format="bfloat16", page_bytes=2048, pages=1),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=1),
        ],
        compile_args={"reader": tensor.accessor_args()},
    )


def cycles_us(cycles, mhz):
    """A difference of two 32-bit wall clocks (cycles) in us at `mhz`."""
    cycles = (int(cycles) + 2**31) % 2**32 - 2**31
    return round(cycles / mhz, 2) if mhz else None


def latency(bench, args, out):
    """One rank of island 0 (alone in its command) gathers 1 and 64 pages into island
    1, by km::gather_pages and by the ring put (put_pages, put_piece); a command each,
    --repeat times; the steps of the last: the rank's clock, island 0's lead page CPU
    (taken, across, pushed) and island 1's (arrived, confirmed)."""
    rank = (bench.own[0][0], 1)
    salt = bench.fill(0, bench.a[0], 64)
    rows = []
    nops = []
    bench.together([bench.submit(0, "nop", [], rank)])  # the first: not timed
    for _ in range(args.repeat):
        (event,) = bench.together([bench.submit(0, "nop", [], rank)])
        nops.append(event.ticks / X280_HZ * 1e6)
    nop = float(np.median(nops))
    log(f"NOP one rank: median {nop:.2f} us of {args.repeat}")
    out["nop_us"] = round(nop, 2)
    for mode, name in ((0, "gather"), (MODE_RING, "ring")):
        for pages in (1, 64):
            times = []
            for _ in range(args.repeat):
                words = [
                    bench.a[0].address,
                    0,
                    pages,
                    1,
                    1,
                    bench.reports[0].address,
                    bench.b[1].address,
                    0,
                    mode,
                ]
                (event,) = bench.together([bench.submit(0, "gather", words, rank)])
                times.append(event.ticks / X280_HZ * 1e6)
            mhz = aiclk(bench.cards)
            report = bench.report_rows(0, 1)[0]
            sent = bench.stamps(0, 1)
            landed = bench.stamps(1, 0)
            clock = mhz[0]
            start, posted, done = (int(report[k]) for k in (2, 3, 4))
            steps = {
                "posted": cycles_us(posted - start, clock),
                "taken": cycles_us(sent[0] - posted, clock),
                "across": cycles_us(sent[1] - sent[0], clock),
                "pushed": cycles_us(sent[2] - sent[1], clock),
                "far_wait": cycles_us(landed[4] - landed[3], mhz[1]),
                "posted_to_done": cycles_us(done - posted, clock),
                "start_to_done": cycles_us(done - start, clock),
            }
            bad, first = bench.check(1, bench.b[1], pages, salt)
            row = {
                "call": name,
                "pages": pages,
                "kib": pages * PAGE // 1024,
                "cmd_us": round(float(np.median(times)), 2),
                "cmd_us_all": [round(t, 1) for t in times],
                "added_us": round(float(np.median(times)) - nop, 2),
                "steps_us": steps,
                "aiclk": mhz,
                "stamps_sent": sent,
                "stamps_far": landed,
                "bad_words": bad,
                "first_bad": first,
            }
            log(
                f"{name} {pages} pages: cmd {row['cmd_us']} us (added "
                f"{row['added_us']}), steps {steps}, bad {bad}"
            )
            rows.append(row)
    out["latency"] = rows


def gather_rows(bench, args, out):
    """GATHER of each size, each way and both at once, idle and busy: all ranks of
    the sending island post their shares; the time added to the command (against the
    same command with no pages), the rate, every word checked."""
    rows = []
    for kib, busy, issuers in itertools.product(args.gather, args.busy, args.issuers):
        pages = kib * 1024 // PAGE
        base = baseline(bench, busy)
        for way in args.ways:
            senders = (0, 1) if way == "both" else (int(way[0]),)
            salts = {s: bench.fill(s, bench.a[s], pages) for s in senders}
            before = bench.crossings()
            submitted = []
            for s in senders:
                to = 1 - s
                words = [
                    bench.a[s].address,
                    0,
                    pages,
                    to,
                    issuers or bench.own[s][1],
                    bench.reports[s].address,
                    bench.b[to].address,
                    busy,
                    0,
                ]
                submitted.append(bench.submit(s, "gather", words))
            events = bench.together(submitted)
            moved = bench.crossings() - before
            mhz = aiclk(bench.cards)
            row = {"call": "gather", "kib": kib, "way": way, "busy": busy}
            row["issuers"] = issuers or bench.own[0][1]
            row["cmd_us"] = [round(e.ticks / X280_HZ * 1e6, 1) for e in events]
            row["base_us"] = [round(base[s], 1) for s in senders]
            row["added_us"] = [
                round(c - b, 1) for c, b in zip(row["cmd_us"], row["base_us"])
            ]
            row["gb_s"] = [
                round(pages * PAGE / (c * 1e-6) / 1e9, 2) for c in row["cmd_us"]
            ]
            row["span_us"] = [span(bench, s, mhz) for s in senders]
            row["crossed"] = {
                f"{a}->{b}": moved.total(a, b)["bytes"] for a, b in ((0, 1), (1, 0))
            }
            row["bad_words"] = 0
            for s in senders:
                bad, first = bench.check(1 - s, bench.b[1 - s], pages, salts[s])
                row["bad_words"] += bad
                if pages <= NUMPY_PAGES:
                    want = pattern(0, pages, salts[s])
                    row["bad_words"] += bench.numpy_check(
                        1 - s, bench.b[1 - s], pages, want
                    )
            row["aiclk"] = mhz
            log(f"{row}")
            rows.append(row)
    out["gather"] = rows


def reduce_rows(bench, args, out):
    """REDUCE of each size (the tensor's bytes, each island keeps half), idle and
    busy, both islands at once; the result checked on both islands' cores and, up to
    4 MiB, by numpy."""
    rows = []
    for kib in args.reduce:
        kept = kib * 1024 // PAGE // 2  # pages each island keeps
        for busy in args.busy:
            base = baseline(bench, busy, "reduce")
            salts = [bench.fill(i, bench.a[i], 2 * kept) for i in range(2)]
            before = bench.crossings()
            submitted = []
            for i in range(2):
                other = 1 - i
                words = [
                    bench.a[i].address,
                    bench.b[other].address,
                    bench.b[i].address,
                    bench.result.address,
                    kept,
                    bench.own[i][1] | other << 16,
                    bench.reports[i].address,
                    busy,
                    i * kept,
                    other * kept,
                ]
                submitted.append(bench.submit(i, "reduce", words))
            events = bench.together(submitted)
            moved = bench.crossings() - before
            mhz = aiclk(bench.cards)
            row = {"call": "reduce", "kib": kib, "busy": busy}
            row["cmd_us"] = [round(e.ticks / X280_HZ * 1e6, 1) for e in events]
            row["base_us"] = [round(b, 1) for b in base]
            row["added_us"] = [
                round(c - b, 1) for c, b in zip(row["cmd_us"], row["base_us"])
            ]
            row["span_us"] = [span(bench, i, mhz) for i in range(2)]
            row["crossed"] = {
                f"{a}->{b}": moved.total(a, b)["bytes"] for a, b in ((0, 1), (1, 0))
            }
            row["bad_words"] = 0
            for i in range(2):
                bad, _ = bench.check(i, bench.result, 2 * kept, salts[0], salts[1])
                row["bad_words"] += bad
            if 2 * kept <= NUMPY_PAGES:
                want = pattern(0, 2 * kept, salts[0]) + pattern(0, 2 * kept, salts[1])
                row["bad_words"] += bench.numpy_check(0, bench.result, 2 * kept, want)
            row["aiclk"] = mhz
            log(f"{row}")
            rows.append(row)
    out["reduce"] = rows


def baseline(bench, busy, name="gather"):
    """The command with no pages (and the same busy loop) on each island at once."""
    submitted = []
    for i in range(2):
        if name == "gather":
            words = [bench.a[i].address, 0, 0, 1 - i, 0, 0, 0, busy, 0]
        else:
            words = [0, 0, 0, 0, 0, bench.own[i][1] | (1 - i) << 16, 0, busy, 0, 0]
        submitted.append(bench.submit(i, name, words))
    events = bench.together(submitted)
    return [e.ticks / X280_HZ * 1e6 for e in events]


def span(bench, island, mhz):
    """From the first rank's start to the last rank's end, by the ranks' reports
    (us at the card's AICLK)."""
    ranks = bench.own[island][1]
    rows = bench.report_rows(island, ranks)
    good = rows[rows[:, 0] == REPORT_MAGIC]
    if not len(good):
        return None
    start = good[:, 2].astype(np.int64)
    end = good[:, 5].astype(np.int64)
    ends = (end - start.min()) % 2**32
    return cycles_us(int(ends.max()), mhz[min(island, len(mhz) - 1)])


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    out = {"cards": cards, "links": args.links, "page": PAGE}
    sizes = [*args.gather, *args.reduce, 256]
    pages = max(sizes) * 1024 // PAGE
    options = {"cards": cards, "log": log, "kernel_bytes": KERNEL_BYTES}
    if args.split:
        options.update(islands=args.split, graph=islands.line_pairs(args.split))
    else:
        options["links"] = args.links
    with km.Session(**options) as session:
        bench = Bench(session, pages, cards)
        log(f"loaded, {bench.islands} islands, ranks {bench.own}")
        try:
            if args.latency:
                latency(bench, args, out)
            if args.gather:
                gather_rows(bench, args, out)
            if args.reduce:
                reduce_rows(bench, args, out)
        except km.KomorebiError as error:
            out["error"] = str(error)[:600]
            log(f"failed: {out['error']}")
        try:
            results = bench.run.stop().islands
            out["kernel_errors"] = [result["error_name"] for result in results]
        except km.KomorebiError as error:
            out["kernel_errors"] = str(error)[:300]
    rows = [*out.get("latency", []), *out.get("gather", []), *out.get("reduce", [])]
    out["ok"] = bool(
        "error" not in out
        and all(e == "none" for e in out["kernel_errors"])
        and all(row["bad_words"] == 0 for row in rows)
    )
    out["wall_s"] = round(time.monotonic() - STARTED, 1)
    line = json.dumps(out)
    print(line, flush=True)
    if args.json:
        Path(args.json).write_text(line + "\n")
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
