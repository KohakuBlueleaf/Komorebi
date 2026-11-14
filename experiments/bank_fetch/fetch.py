"""The copy's rate: work cluster 1's thread fetches a range of another DRAM bank into
its own region with kmx_fetch on 1, 4 and 12 Tensix cores, then stores part of it back
with kmx_store; every byte checked (a hash, the bytes themselves up to 1 MiB). One JSON
line. Example: python fetch.py --bytes 4096 --cores 1"""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[1] / "examples" / "08_parallel_thread" / "kernels"  # NOPs
ROW = 16384  # bytes of a source page: bank b holds rows b, b + 8, ... one after another
BANKS = 8
TICKS = 50  # kmx_time() ticks a microsecond
WHOLE = 1 << 20  # read the region back whole up to this many bytes
T0 = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - T0:7.2f}] {message}", flush=True)


class Clock(threading.Thread):
    """Samples the card's AICLK (MHz) until stopped."""

    def __init__(self, card):
        super().__init__(daemon=True)
        self.path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            text = self.path.read_text().strip() if self.path.exists() else ""
            if text.isdigit():
                self.samples.append(int(text))
            self.stop.wait(0.02)

    def stats(self):
        samples = self.samples or [0]
        return min(samples), round(sum(samples) / len(samples), 1)


def hashed(data):
    """copy.c's hash: the sum of 64-bit word j times 2 j + 1, mod 2**64."""
    words = data.view(np.uint64)
    return int(np.sum(words * (2 * np.arange(words.size, dtype=np.uint64) + 1)))


def source_rows(args, rng):
    """[8 (n + m), ROW / 4] uint32 rows whose bank `args.bank` holds the data (its first
    n rows), then m zero rows for the store; the other banks' rows zero."""
    n, m = -(-args.bytes // ROW), -(-args.store // ROW)
    rows = np.zeros((BANKS * (n + m), ROW // 4), np.uint32)
    mine = rng.integers(0, 1 << 32, (n, ROW // 4), dtype=np.uint64).astype(np.uint32)
    rows[args.bank :: BANKS][:n] = mine
    return rows, mine.reshape(-1).view(np.uint8)[: args.bytes]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bytes", type=int, default=4096)
    parser.add_argument("--cores", type=int, nargs="+", default=[1])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bank", type=int, default=0, help="the source's bank")
    parser.add_argument("--store", type=int, default=0, help="bytes stored back")
    parser.add_argument("--card", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows, want = source_rows(args, rng)
    log(f"source rows built: {rows.nbytes >> 20} MiB, {args.bytes} bytes in bank")
    clock = Clock(args.card)
    with km.Session(log=log) as session:
        source = session.tensor(rows, "uint32")
        log(f"source written at {source.address:#x}")
        program = km.Program("bank_fetch", kernels=KERNELS)
        store_at = source.address + -(-args.bytes // ROW) * ROW
        words = [args.bank, source.address, args.bytes, args.repeats]
        words += [len(args.cores), *args.cores, *[0] * (4 - len(args.cores))]
        words += [args.store, store_at]
        program.thread(
            cluster=1,
            sources=[HERE / "thread" / "copy.c"],
            regions={"dst": args.bytes},
            words=words,
            deadline=30.0,
        )
        run = session.load(program)
        log("loaded")
        clock.start()
        result = run.thread_result(timeout=40.0)
        clock.stop.set()
        log(f"thread: {result.state}, code {result.code}, error {result.error}")
        run.stop()
        log("stopped")
        region = run.thread.read("dst") if args.bytes <= WHOLE else None
        log("region read" if region is not None else "region not read (hash only)")
        stored = None
        if args.store:
            first = BANKS * -(-args.bytes // ROW)
            pages = session.read(
                source, first=first, pages=BANKS * -(-args.store // ROW)
            )
            stored = pages[0][args.bank :: BANKS].reshape(-1)[: args.store]
            log("stored range read")
    words = list(result.words) + [0] * 34
    want_hash = hashed(want)
    report = {"bytes": args.bytes, "bank": args.bank, "repeats": args.repeats}
    report["aiclk_min_mean"] = clock.stats()
    report["thread"] = [result.state, result.code, result.error]
    report["cores"] = {}
    for i, count in enumerate(args.cores):
        out = words[8 * i : 8 * i + 8]
        call_us, event_us = out[0] / TICKS, out[1] / TICKS
        report["cores"][count] = {
            "status": out[4],
            "call_us": round(call_us, 2),
            "event_us": round(event_us, 2),
            "worst_us": round(out[5] / TICKS, 2),
            "gb_s": round(args.bytes / call_us / 1e3, 3) if out[0] else None,
            "exact": (out[2] | out[3] << 32) == want_hash,
        }
    if region is not None:
        report["region_exact"] = bool((region == want).all())
    if stored is not None:
        report["store"] = {
            "status": words[32],
            "call_us": round(words[33] / TICKS, 2),
            "exact": bool((stored == want[: args.store]).all()),
        }
    report["ok"] = (
        result.ran
        and result.code == 0
        and all(c["exact"] and c["status"] == 0 for c in report["cores"].values())
        and report.get("region_exact", True)
        and report.get("store", {"exact": True})["exact"]
    )
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
