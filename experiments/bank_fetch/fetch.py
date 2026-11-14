"""The call's time: work cluster 1's thread fetches another bank's range into its region
on 1, 4 and 12 Tensix cores three ways, stores part back, times a sweep on four X280
cores and one; every byte checked (a hash per way, the bytes up to 1 MiB). One JSON
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
WAYS = ("call", "clean", "before")  # copy.c's ways: kmx_fetch, the clean call, a sweep
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
    words = list(result.words) + [0] * 40
    want_hash = hashed(want)
    report = {"bytes": args.bytes, "bank": args.bank, "repeats": args.repeats}
    report["aiclk_min_mean"] = clock.stats()
    report["thread"] = [result.state, result.code, result.error]
    report["cores"] = {}
    for i, count in enumerate(args.cores):
        out = words[12 * i : 12 * i + 12]
        row = {"status": out[1], "event_us": round(out[0] / TICKS, 2)}
        row["worst_us"] = round(out[2] / TICKS, 2)
        for way, name in enumerate(WAYS):
            ticks, low, high = out[3 + 3 * way : 6 + 3 * way]
            row[f"{name}_us"] = round(ticks / TICKS, 2)
            rate = args.bytes * TICKS / ticks / 1e3 if 0 < ticks < 0xFFFFFFFF else 0
            row[f"{name}_gb_s"] = round(rate, 3)
            row[f"{name}_exact"] = (low | high << 32) == want_hash
        row["exact"] = all(row[f"{name}_exact"] for name in WAYS)
        report["cores"][count] = row
    if region is not None:
        report["region_exact"] = bool((region == want).all())
    if stored is not None:
        report["store"] = {
            "status": words[36],
            "call_us": round(words[37] / TICKS, 2),
            "exact": bool((stored == want[: args.store]).all()),
        }
    report["sweep_us"] = {"four": words[38] / TICKS, "one": words[39] / TICKS}
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
