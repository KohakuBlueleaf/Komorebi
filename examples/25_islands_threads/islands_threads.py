"""Commands across islands from the card: a NOP for island 1's ranks given to island 0's
dispatcher alone; island 0's X280 thread sending MAP to every rank of both islands, a
FOLD to island 1's thread, and a HANG on island 1 that its deadline gives up. Example
01's kernels, unchanged. Prints one JSON line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
MAPCOLLECT = HERE.parent / "01_mapcollect"
sys.path.insert(0, str(MAPCOLLECT))
from mapcollect import rank_sums  # noqa: E402 (example 01's sums, beside it)

KERNELS = MAPCOLLECT / "kernels"
TILE = 32
TICKS_PER_US = 50  # the X280 timer, 50 MHz
THREAD_HANDLERS = {"maps": 1, "ask": 2, "fold": 3, "hang": 4}  # across.c
EVENT_TIMEOUT, EVENT_ISLAND = 1, 0x80  # KM_EVENT_TIMEOUT, KM_EVENT_ISLAND
BAD = 0xFFFFFFFF
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument(
        "--links", type=int, default=0, help="the session's page links (0: link 0)"
    )
    parser.add_argument(
        "--steps", default="nop,map,ask,hang", help="of nop, map, ask, hang"
    )
    parser.add_argument("--nop-jobs", type=int, default=2000, help="per round")
    parser.add_argument("--rounds", type=int, default=3, help="NOP rounds")
    parser.add_argument("--maps", type=int, default=500, help="MAPs a thread sends")
    parser.add_argument("--asks", type=int, default=500, help="FOLDs a thread asks")
    return parser.parse_args()


def program(source, output, threads):
    """Example 01's kernels; with `threads`, across.c on work cluster 1."""
    made = km.Program(
        "mapcollect",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={
            "reader": source.accessor_args(),
            "writer": output.accessor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=KERNELS / "mapcollect.h",
    )
    if threads:
        made.thread(
            cluster=1,
            sources=[HERE / "thread" / "across.c"],
            regions={"scratch": 64},
            handlers=THREAD_HANDLERS,
            defines={
                "kMap": f"{int(made.handler('map'))}u",
                "kHang": f"{int(made.handler('hang'))}u",
            },
        )
    return made


def timed_nops(run, group, jobs):
    """Microseconds a NOP of one round of `jobs` to `group`, pipelined; every event
    done."""
    started = time.perf_counter()
    handles = run.submit_many(0, np.full(jobs, group[0]), np.full(jobs, group[1]))
    events = run.wait(list(handles), timeout=30.0)
    spent = (time.perf_counter() - started) * 1e6 / jobs
    if any(event.kind != "done" for event in events):
        raise RuntimeError(f"NOP to {group}: {sorted({e.kind for e in events})}")
    return spent


def ask_thread(run, handler, args, timeout=20.0):
    """A command to island 0's thread (cluster 1), waited for: its word."""
    event = run.wait(run.submit(handler, cluster=1, args=args, result=True), timeout)
    if event.kind != "done":
        raise RuntimeError(f"thread command {handler}: {event.kind}")
    return event.result


def per_command_us(ticks, count):
    return None if ticks == BAD else round(ticks / TICKS_PER_US / count, 3)


def main():
    args = parse()
    steps = args.steps.split(",")
    cards = [int(card) for card in args.cards.split(",")]
    threads = any(step in steps for step in ("map", "ask", "hang"))
    report = {"cards": cards, "steps": steps, "links": args.links}
    generator = np.random.default_rng(25)
    with km.Session(cards=cards, links=args.links, log=log) as session:
        ranks = session.ranks
        graph = session.graph
        own = [(graph.first(i), graph.count(i)) for i in range(2)]
        log(f"open: {ranks} ranks, islands {own}")
        exact = generator.integers(-2, 3, (2 * ranks * TILE, TILE)).astype(np.float32)
        source = session.tensor(exact, "bfloat16", "tile")
        output = session.tensor(
            np.zeros((ranks * TILE, TILE), np.float32), "bfloat16", "tile"
        )
        made = [program(source, output, threads) for _ in range(2)]
        run = session.load(made, deadline=2.0)
        log("loaded on both islands")
        first = run.island(0)
        if "nop" in steps:
            rounds = range(args.rounds)
            report["nop_local_us"] = float(
                np.median([timed_nops(first, own[0], args.nop_jobs) for _ in rounds])
            )
            report["nop_across_us"] = float(
                np.median([timed_nops(first, own[1], args.nop_jobs) for _ in rounds])
            )
            log(
                f"NOP from island 0's ring: its own ranks {report['nop_local_us']:.3f}"
                f" us, island 1's {report['nop_across_us']:.3f} us"
            )
        wrong = None
        if "map" in steps:
            words = [source.address, output.address]
            local = ask_thread(first, "maps", [*own[0], args.maps, *words])
            whole = ask_thread(first, "maps", [0, ranks, args.maps, *words])
            report["thread_map_local_us"] = per_command_us(local, args.maps)
            report["thread_map_all_us"] = per_command_us(whole, args.maps)
            log(
                f"thread MAP: island 0's ranks {report['thread_map_local_us']} us, "
                f"all {ranks} {report['thread_map_all_us']} us"
            )
            wrong = "map"
        if "ask" in steps:
            number = 3 * 1 + 1  # KMX_THREAD(1, 1)
            word = ask_thread(first, "ask", [number, args.asks, 0x1234])
            report["thread_ask_us"] = per_command_us(word, args.asks)
            report["ask_word"] = word
            log(f"thread to thread: {report['thread_ask_us']} us a FOLD ({word:#x})")
        if "hang" in steps:
            group, hung = (100, 40), 30  # rank 130: one of island 1's
            word = ask_thread(first, "hang", [*group, hung, 300000, 2000000])
            report["hang_word"] = word
            report["hang_named_island"] = word == (
                EVENT_TIMEOUT << 8 | EVENT_ISLAND + 1
            )
            log(f"HANG on {group}: word {word:#x}")
            words = [source.address, output.address]
            after = ask_thread(first, "maps", [0, ranks, 1, *words])
            report["map_after_hang"] = after != BAD
            log(f"a MAP on all ranks after it: {report['map_after_hang']}")
            wrong = "map"
        log("stopping")
        results = run.stop()
        report["kernel_errors"] = [result["error_name"] for result in results]
        log(f"stopped: {report['kernel_errors']}")
        if wrong:
            (got,) = session.read_bf16(output)
            report.update(check(exact, got, ranks))
            log(f"output: {report['wrong_tiles']} wrong of {ranks}")
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        report["kernel_errors"] == ["none", "none"]
        and report.get("wrong_tiles", 0) == 0
        and report.get("thread_map_all_us", 0) is not None
        and report.get("thread_ask_us", 0) is not None
        and report.get("hang_named_island", True)
        and report.get("map_after_hang", True)
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


def check(exact, got, ranks):
    """Every rank's tile of the last MAP on all ranks against example 01's float64
    sums, bit-exact (integer inputs)."""
    sums = np.cumsum(exact.reshape(-1, TILE, TILE).astype(np.float64), 0)
    prefix = np.concatenate([np.zeros((1, TILE, TILE)), sums])
    want = rank_sums(prefix, 0, 2 * ranks, ranks)
    wrong = int((got.reshape(-1, TILE, TILE) != want).any(axis=(1, 2)).sum())
    return {"tiles_checked": ranks, "wrong_tiles": wrong}


if __name__ == "__main__":
    main()
