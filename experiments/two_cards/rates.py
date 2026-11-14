"""The link rates of link_packet runs: reads the JSON line of each log named on the
command line and prints, per run, each pair's rate at its receiver (bytes after the
first packet over the span from the first arrival to the last) and their sum by
direction. See README.md.
"""

import json
import sys
from pathlib import Path


def report(path):
    """One run's line of the table, from the last JSON line of its log."""
    lines = [line for line in Path(path).read_text().splitlines() if line[:1] == "{"]
    run = json.loads(lines[-1])
    options = run["options"]
    by_direction = {}
    for pair in run["pairs"]:
        receiver = pair["receiver"]
        mhz = run["aiclk_mhz"][pair["to"]]
        span_s = receiver["cycles"] / (mhz * 1e6)
        moved = (receiver["packets"] - 1) * options["payload"]
        rate = moved / span_s / 1e9 if span_s else 0.0
        key = f"{pair['from']}->{pair['to']}"
        by_direction.setdefault(key, []).append((pair["link"], rate, pair["ok"]))
    parts = []
    for key, rates in by_direction.items():
        each = " ".join(f"l{link} {rate:.2f}" for link, rate, _ in rates)
        total = sum(rate for _, rate, _ in rates)
        parts.append(f"{key}: {each} (sum {total:.2f} GB/s)")
    ok = run["ok"] and all(ok for rates in by_direction.values() for *_, ok in rates)
    return f"{Path(path).stem}: ok={ok} {'; '.join(parts)}"


def main():
    for path in sys.argv[1:]:
        print(report(path))


if __name__ == "__main__":
    main()
