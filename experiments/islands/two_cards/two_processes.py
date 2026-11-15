"""Example 01's program in one process per card (child.py): each child sees only its
own card (TT_VISIBLE_DEVICES); when every child has loaded, all are told to go at
once, so their timed rounds overlap. One JSON line sums up each card. See README.md.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

CHILD = Path(__file__).resolve().parent / "child.py"
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def aiclk(card):
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    return int(path.read_text().split()[0])


def sample(cards, clocks, stop):
    """AICLK of every card every 50 ms until `stop` is set."""
    while not stop.wait(0.05):
        for card in cards:
            clocks[card].append(aiclk(card))


def relay(card, child, results, ready):
    """Passes the child's lines on, notes READY and keeps its last JSON line."""
    for line in child.stdout:
        line = line.rstrip("\n")
        print(f"[card {card}] {line}", flush=True)
        if line == "READY":
            ready[card].set()
        elif line.startswith("{"):
            results[card] = json.loads(line)


def median(values):
    ordered = sorted(values or [0.0])
    return ordered[len(ordered) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1", help="one child per card")
    parser.add_argument("--timeout", type=float, default=50.0, help="seconds")
    parser.add_argument("child_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.child_args[1:] if args.child_args[:1] == ["--"] else []
    cards = [int(card) for card in args.cards.split(",")]
    clocks = {card: [aiclk(card)] for card in cards}
    stop = threading.Event()
    sampler = threading.Thread(target=sample, args=(cards, clocks, stop))
    sampler.start()
    children, threads, results = {}, [], {}
    ready = {card: threading.Event() for card in cards}
    for card in cards:
        environ = dict(os.environ, TT_VISIBLE_DEVICES=str(card))
        children[card] = subprocess.Popen(
            [sys.executable, str(CHILD), *extra],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environ,
        )
        threads.append(
            threading.Thread(target=relay, args=(card, children[card], results, ready))
        )
        threads[-1].start()
        log(f"card {card}: started process {children[card].pid}")
    deadline = time.monotonic() + args.timeout
    for card in cards:
        if not ready[card].wait(timeout=max(deadline - time.monotonic(), 0.1)):
            log(f"card {card}: not ready")
    for card in cards:
        if ready[card].is_set():
            children[card].stdin.write("go\n")
            children[card].stdin.flush()
    log("go sent")
    codes = {}
    for card, child in children.items():
        try:
            codes[card] = child.wait(timeout=max(deadline - time.monotonic(), 0.1))
        except subprocess.TimeoutExpired:
            log(f"card {card}: no exit by the deadline, SIGTERM")
            child.terminate()
            codes[card] = child.wait(timeout=20.0)
        log(f"card {card}: exit {codes[card]}")
    for thread in threads:
        thread.join(timeout=5.0)
    stop.set()
    sampler.join(timeout=5.0)
    per_card = {}
    for card in cards:
        result = results.get(card, {})
        per_card[card] = {
            "nop_us_median": median(result.get("nop_us")),
            "map_us_median": median(result.get("map_us")),
            "map_wrong_tiles": result.get("map_wrong_tiles"),
            "kernel_error": result.get("kernel_error"),
            "window": result.get("window"),
        }
    windows = [entry["window"] for entry in per_card.values() if entry["window"]]
    overlap = None
    if len(windows) > 1:
        overlap = min(end for _, end in windows) - max(go for go, _ in windows)
        longest = max(end - go for go, end in windows)
        overlap = round(max(overlap, 0.0) / longest, 3)
    summary = {
        "cards": cards,
        "args": extra,
        "aiclk_mhz": {
            card: {"min": min(values), "max": max(values), "samples": len(values)}
            for card, values in clocks.items()
        },
        "overlap_of_longest": overlap,
        "loadavg": Path("/proc/loadavg").read_text().split()[:3],
        "exit": codes,
        "per_card": per_card,
    }
    print(json.dumps(summary), flush=True)
    sys.exit(0 if all(code == 0 for code in codes.values()) else 1)


if __name__ == "__main__":
    main()
