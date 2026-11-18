"""Five processors of one Tensix core (BRISC, NCRISC, three TRISCs): whether a TRISC
reaches the NoC, the same loop on each, NoC rates from one core, the wall clock against
the host's, round trips through L1 between processors. One JSON line per part.

Example: python five.py probe --probe register:0:0; python five.py loop ping clock noc
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.device import mock
from komorebi.tools.feeder import fed

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
HEADER = KERNELS / "five.h"
C = km.abi.constants(HEADER, known=km.abi.headers().constants)
NAMES = ["BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"]
PARTS = ["probe", "loop", "ping", "clock", "noc"]
MODES = {
    "register": C["kProbeRegister"],
    "write": C["kProbeWrite"],
    "read": C["kProbeRead"],
}
PAIRS = {
    "BRISC-NCRISC": C["kPairBriscNcrisc"],
    "BRISC-TRISC0": C["kPairBriscTrisc"],
    "TRISC0-TRISC2": C["kPairTriscTrisc"],
    "TRISC0-BRISC": C["kPairTriscBrisc"],
}
NS = (16, 64, 119)  # cores started or collected from


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("parts", nargs="+", choices=PARTS)
    p.add_argument(
        "--probe",
        nargs="+",
        default=["register:0:0"],
        help="mode:noc:trisc, mode register, write or read (X1), in order",
    )
    p.add_argument("--iterations", type=int, default=1_000_000, help="LOOP (X2)")
    p.add_argument("--rounds", type=int, default=10_000, help="PING round trips (X5)")
    p.add_argument("--reps", type=int, default=100, help="NOC repetitions a case (X3)")
    p.add_argument("--host-wait", type=float, default=2.0, help="CLOCK host seconds")
    p.add_argument(
        "--swap-nocs",
        action="store_true",
        help="the reader (BRISC) on NoC 1 and the writer (NCRISC) on NoC 0; NOC leaves "
        "TRISC 0 out (its table is filled on NoC 0 by BRISC)",
    )
    return p.parse_args()


class Bench:
    """The open session, the loaded program, its tensors and the cores' coordinates."""

    def __init__(self, session, swap_nocs=False):
        self.session = session
        self.grid = session.grid
        self.cores = self.grid[0] * self.grid[1]
        self.nocs = session.nocs()
        rows = np.zeros((self.cores, C["kRowWords"]), np.uint32)
        source = C["kReadSource"]
        rows[:, source : source + 16] = np.arange(self.cores)[
            :, None
        ] * 256 + np.arange(16)
        self.rows = session.tensor(rows, "uint32", shard_grid=self.grid)
        blocks = np.tile(
            np.arange(C["kBlockWords"], dtype=np.uint32), (self.cores, C["kProcessors"])
        )
        self.blocks = session.tensor(blocks, "uint32", shard_grid=self.grid)
        self.dram = session.tensor(np.zeros((64, 16), np.uint32), "uint32")
        words = [self.rows.address, self.blocks.address]
        program = km.Program(
            "five_processors",
            kernels=KERNELS,
            args=lambda core: (words, words, words),
            handlers=HEADER,
            opt={"reader": "O2", "writer": "O2", "compute": "O2"},
            noc={"reader": 1, "writer": 0} if swap_nocs else {},
        )
        self.swap_nocs = swap_nocs
        self.run = session.load(program, deadline=10.0, wait=10.0)
        self.mhz = fed.aiclk_mhz()
        fed.log(f"loaded: grid {self.grid}, {self.cores} cores, AICLK {self.mhz} MHz")

    def submit(self, handler, group, args, timeout=20.0):
        """One command, waited for; its host microseconds (submit to event)."""
        event = self.run.wait(self.run.submit(handler, group, args), timeout=timeout)
        if event.kind != "done":
            raise km.KomorebiError(f"{handler} {group} {args}: {event}")
        return event.microseconds

    def row(self, rank):
        """Core `rank`'s row, read while the run is loaded."""
        pages = self.session.read(self.rows, first=rank, pages=1)
        if isinstance(pages, (list, tuple)):
            pages = pages[0]
        return np.asarray(pages).view(np.uint32).reshape(-1)

    def every_row(self):
        """Every core's row, read while the run is loaded: [cores, kRowWords]."""
        pages = self.session.read(self.rows)
        if isinstance(pages, (list, tuple)):
            pages = pages[0]
        return np.asarray(pages).view(np.uint32).reshape(self.cores, -1)

    def dram_page(self, page):
        pages = self.session.read(self.dram, first=page, pages=1)
        if isinstance(pages, (list, tuple)):
            pages = pages[0]
        return np.asarray(pages).view(np.uint32).reshape(-1)

    def result(self, row, processor):
        base = C["kResult"] + processor * C["kResultWords"]
        return row[base : base + C["kResultWords"]].astype(np.int64)

    def xy(self, rank):
        """The translated NoC coordinates of core `rank` as x | y << 16."""
        return self.nocs.words[rank]


def probe(bench, specs):
    """X1: each mode:noc:trisc in order; the TRISC's words and the movers'."""
    out = []
    target = 1
    for spec in specs:
        mode, noc, trisc = spec.split(":")
        noc, trisc = int(noc), int(trisc)
        fed.log(f"probe {spec}: submitting")
        host_us = bench.submit(
            "probe", (0, 1), [MODES[mode], noc, bench.xy(target), trisc]
        )
        row = bench.row(0)
        tr = bench.result(row, C["kTrisc0"] + trisc)
        p = tr[C["kRProbe"] : C["kRProbe"] + 8]
        movers = [bench.result(row, m)[C["kRProbe"] : C["kRProbe"] + 6] for m in (0, 1)]
        case = {
            "probe": spec,
            "host_us": round(host_us, 1),
            "trisc_words": [hex(int(v)) for v in p],
            "brisc_ids": [hex(int(v)) for v in movers[0][:4]],
            "movers_saw_done": [int(m[4]) for m in movers],
        }
        if mode == "register":
            case["node_id"] = hex(int(p[0]))
            case["id_logical"] = hex(int(p[1]))
            case["same_as_brisc"] = bool(
                p[0] == movers[0][2 * noc] and p[1] == movers[0][2 * noc + 1]
            )
            case["cycles"] = int(p[2])
        else:
            case["outcome"] = {1: "answered", 2: "no answer", 3: "buffer busy"}.get(
                int(p[3]), int(p[3])
            )
            case["counter"] = [int(p[0]), int(p[1])]
            case["cycles"] = int(p[2])
            if mode == "write":
                landed = bench.row(target)[C["kTrisLanding"] : C["kTrisLanding"] + 4]
                case["landed"] = [hex(int(v)) for v in landed]
                case["landed_ok"] = bool(
                    all((int(v) & 0xFF) == noc << 4 | w for w, v in enumerate(landed))
                    and all(int(v) >> 24 == 0x7A for v in landed)
                )
            else:
                case["read"] = [int(v) for v in p[4:8]]
                case["read_ok"] = case["read"] == [target * 256 + w for w in range(4)]
        fed.log(f"probe {spec}: {case}")
        out.append(case)
    return {"part": "probe", "target_rank": target, "cases": out}


def loop(bench, iterations):
    """X2: the loop on each processor alone, then on all five; seven runs each."""
    masks = [1 << p for p in range(5)] + [31]
    runs = []
    first_sums = {}
    attempts = 7
    for attempt in range(attempts):
        for mask in masks:
            bench.submit("loop", (0, 1), [mask, iterations, 0])
            row = bench.row(0)
            ticks = {}
            for p in range(5):
                if mask >> p & 1:
                    r = bench.result(row, p)
                    start = r[C["kRStart"]] | r[C["kRStart"] + 1] << 32
                    end = r[C["kREnd"]] | r[C["kREnd"] + 1] << 32
                    ticks[NAMES[p]] = int(end - start)
                    if attempt == 0 and mask != 31:
                        first_sums[NAMES[p]] = hex(int(r[C["kRSum"]]))
            runs.append({"mask": mask, "ticks": ticks})

    def ticks_of(name, mask):
        return [r["ticks"][name] for r in runs if r["mask"] == mask]

    alone = {NAMES[p]: ticks_of(NAMES[p], 1 << p) for p in range(5)}
    together = {name: ticks_of(name, 31) for name in NAMES}
    out = {
        "part": "loop",
        "iterations": iterations,
        "attempts": attempts,
        "aiclk_mhz": bench.mhz,
        "alone_ticks_median": {n: int(np.median(t)) for n, t in alone.items()},
        "all_five_ticks_median": {n: int(np.median(t)) for n, t in together.items()},
        "slowdown_median": {
            n: round(float(np.median(together[n]) / np.median(alone[n])), 3)
            for n in NAMES
        },
        "slowdown_range": {
            n: [
                round(min(together[n]) / np.median(alone[n]), 3),
                round(max(together[n]) / np.median(alone[n]), 3),
            ]
            for n in NAMES
        },
        "cycles_per_iteration_alone": {
            n: round(float(np.median(t)) / iterations, 3) for n, t in alone.items()
        },
        "first_sums": first_sums,
        "sums_equal": len(set(first_sums.values())) == 1,
        "runs": runs,
    }
    return out


def ping(bench, rounds):
    """X5: round trips through one L1 word pair, each pair of processors."""
    out = {}
    for name, pair in PAIRS.items():
        bench.submit("ping", (0, 1), [pair, rounds])
        first = C["kTrisc0"] if name.startswith("TRISC0") else C["kBrisc"]
        r = bench.result(bench.row(0), first)
        ticks = int(r[C["kRPing"]] | r[C["kRPing"] + 1] << 32)
        done = int(r[C["kRPing"] + 2])
        out[name] = {
            "round_trips": done,
            "timeouts": int(r[C["kRPing"] + 3]),
            "ticks": ticks,
            "ticks_per_round_trip": round(ticks / max(done, 1), 1),
        }
        fed.log(f"ping {name}: {out[name]}")
    return {"part": "ping", "aiclk_mhz": bench.mhz, "pairs": out}


def clock(bench, host_wait):
    """X4: two stamps of core 0's wall clock `host_wait` host seconds apart while the
    other cores run the loop on all five processors."""
    busy_cycles = int((host_wait + 1.0) * bench.mhz * 1e6)
    busy = bench.run.submit("loop", (1, bench.cores - 1), [31, 0, busy_cycles >> 10])
    marks = []
    aiclk = []
    for stamp in range(2):
        if stamp:
            time.sleep(host_wait)
        before = time.perf_counter()
        bench.submit("stamp", (0, 1), [stamp])
        after = time.perf_counter()
        marks.append((before, after))
        aiclk.append(fed.aiclk_mhz())
    event = bench.run.wait(busy, timeout=host_wait + 20.0)
    r = bench.result(bench.row(0), C["kBrisc"])
    stamps = [
        int(r[C["kRStamp"] + 2 * s] | r[C["kRStamp"] + 2 * s + 1] << 32) for s in (0, 1)
    ]
    host = (marks[1][0] + marks[1][1]) / 2 - (marks[0][0] + marks[0][1]) / 2
    window = (marks[0][1] - marks[0][0]) / 2 + (marks[1][1] - marks[1][0]) / 2
    ticks = stamps[1] - stamps[0]
    return {
        "part": "clock",
        "host_seconds": round(host, 6),
        "host_uncertainty_seconds": round(window, 6),
        "ticks": ticks,
        "ticks_per_second_mhz": round(ticks / host / 1e6, 3) if host else None,
        "bound_mhz": [
            round(ticks / (host + window) / 1e6, 3),
            round(ticks / max(host - window, 1e-9) / 1e6, 3),
        ],
        "aiclk_sysfs_mhz": aiclk,
        "busy_event": event.kind,
        "busy_cores": bench.cores - 1,
    }


def noc_cases(bench):
    """X3's cases: (label, mover mask, kind, count, first target)."""
    far = bench.cores - 1
    cases = []
    processors = ((1, "BRISC"), (2, "NCRISC"), (4, "TRISC0"))
    for mask, mover in processors[:2] if bench.swap_nocs else processors:
        cases += [
            (f"{mover} write neighbour", mask, C["kNocWriteOne"], 1, 1),
            (f"{mover} write farthest", mask, C["kNocWriteOne"], 1, far),
            (f"{mover} write DRAM bank 0", mask, C["kNocWriteDram"], 1, 0),
            (f"{mover} write DRAM bank 7", mask, C["kNocWriteDram"], 1, 7),
            (f"{mover} read neighbour", mask, C["kNocReadOne"], 1, 1),
            (f"{mover} read farthest", mask, C["kNocReadOne"], 1, far),
            (f"{mover} read DRAM bank 0", mask, C["kNocReadDram"], 1, 0),
            (f"{mover} read DRAM bank 7", mask, C["kNocReadDram"], 1, 7),
        ]
        for n in NS:
            cases.append((f"{mover} write {n} cores", mask, C["kNocWriteMany"], n, 1))
        for n in NS:
            cases.append((f"{mover} read {n} cores", mask, C["kNocReadMany"], n, 1))
        cases.append((f"{mover} multicast", mask, C["kNocMulticast"], 0, 0))
    for n in NS:
        cases.append((f"both write {n} cores", 3, C["kNocWriteMany"], n, 1))
    return cases


def multicast_rects(bench):
    """The largest rectangles without core 0 inside one panel (consecutive NoC0
    columns): (receivers, NoC 0 start, end, NoC 1 start, end) in translated words."""
    gx, gy = bench.grid
    rects = []
    for x0, x1 in bench.nocs.panels:
        y0 = 1 if x0 == 0 else 0  # core 0 is (0, 0): leave its row out of its panel
        a = bench.nocs.noc_xy[(x0, y0)]
        b = bench.nocs.noc_xy[(x1, gy - 1)]
        start, end = a[0] | a[1] << 16, b[0] | b[1] << 16
        cores = (x1 - x0 + 1) * (gy - y0)
        rects.append((cores, start, end, end, start, (x0, y0, x1, gy - 1)))
    return sorted(rects, reverse=True)


def words16(row, at):
    return row[at : at + 16].astype(np.int64)


def noc_wrong(bench, mask, kind, count, first, rect):
    """The 64-byte lines a NOC case got wrong: each target's landing against processor
    m's source on core 0, or what core 0 read against the target's words."""
    rows = bench.every_row()
    movers = [m for m in (0, 1, 2) if mask >> m & 1]
    wrong = 0
    for index, m in enumerate(movers):
        source = words16(rows[0], C["kSource"] + 16 * m)
        targets = list(range(first, first + count))
        if len(movers) == 2:  # both: BRISC the first half, NCRISC the rest
            half = count // 2
            targets = targets[:half] if index == 0 else targets[half:]
        if kind == C["kNocMulticast"]:
            x0, y0, x1, y1 = rect[5]
            gx = bench.grid[0]
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    got = words16(rows[y * gx + x], C["kMulticastLanding"])
                    wrong += int((got != source).any())
        elif kind in (C["kNocWriteOne"], C["kNocWriteMany"]):
            for t in targets:
                wrong += int((words16(rows[t], C["kLanding"] + 16 * m) != source).any())
        elif kind == C["kNocWriteDram"]:
            got = bench.dram_page(first).astype(np.int64)
            wrong += int((got != source).any())
        elif kind in (C["kNocReadOne"], C["kNocReadMany"]):
            got = words16(rows[0], C["kReadInto"] + 16 * m)
            want = targets[-1] * 256 + np.arange(16)
            wrong += int((got != want).any())
        elif kind == C["kNocReadDram"]:
            got = words16(rows[0], C["kReadInto"] + 16 * m)
            want = bench.dram_page(first).astype(np.int64)
            wrong += int((got != want).any())
    return wrong


def noc(bench, reps):
    """X3: per case the ticks of each repetition (start to the last flush)."""
    out = []
    rects = multicast_rects(bench)
    for label, mask, kind, n, first in noc_cases(bench):
        targets = [rects[0]] if kind == C["kNocMulticast"] else [None]
        if kind == C["kNocMulticast"] and len(rects) > 1:
            targets.append(rects[1])
        for rect in targets:
            args = [mask, kind, n, reps, first, 0, 0, 0, 0, bench.dram.address]
            name = label
            if rect is not None:
                args[2] = rect[0]
                args[5:9] = rect[1:5]
                name = f"{label} to {rect[0]} cores {rect[5]}"
            bench.submit("noc", (0, 1), args)
            row = bench.row(0)
            ticks = []
            failures = 0
            for m in (0, 1, 2):
                if mask >> m & 1:
                    r = bench.result(row, m)
                    failures += int(r[C["kRNoc"] + 1])
                    base = C["kTimes"] + m * C["kTimesWords"]
                    ticks.append(row[base : base + reps].astype(np.int64))
            each = np.max(np.stack(ticks), axis=0)  # both: the later end
            several = (C["kNocWriteMany"], C["kNocReadMany"], C["kNocMulticast"])
            count = args[2] if kind in several else 1
            case = {
                "case": name,
                "count": count,
                "failures": failures,
                "lines_wrong": noc_wrong(bench, mask, kind, count, first, rect),
                "median_ticks": int(np.median(each)),
                "min_ticks": int(each.min()),
                "max_ticks": int(each.max()),
                "median_us": round(float(np.median(each)) / bench.mhz, 3),
                "median_us_a_core": round(
                    float(np.median(each)) / bench.mhz / count, 4
                ),
            }
            fed.log(f"noc {name}: {case}")
            out.append(case)
    return {
        "part": "noc",
        "reps": reps,
        "aiclk_mhz": bench.mhz,
        "nocs": "BRISC 1, NCRISC 0" if bench.swap_nocs else "BRISC 0, NCRISC 1",
        "cases": out,
    }


def main():
    args = parse()
    results = []
    with km.Session(log=fed.log) as session:
        bench = Bench(session, args.swap_nocs)
        try:
            if mock.active():  # a prebuild: the load built the kernels; reads are zeros
                print(json.dumps({"parts": args.parts, "prebuild": True}), flush=True)
                return
            for part in args.parts:
                if part == "probe":
                    results.append(probe(bench, args.probe))
                elif part == "loop":
                    results.append(loop(bench, args.iterations))
                elif part == "ping":
                    results.append(ping(bench, args.rounds))
                elif part == "clock":
                    results.append(clock(bench, args.host_wait))
                elif part == "noc":
                    results.append(noc(bench, args.reps))
                print(json.dumps(results[-1]), flush=True)
        finally:
            bench.run.stop()
            fed.log("run stopped")


if __name__ == "__main__":
    main()
