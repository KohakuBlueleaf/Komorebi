"""The link rates of link_rate runs: prints, for each log named on the command line,
each pair's rate at the receiver (rx) and sender (tx) and their sums by direction.
See README.md for the two spans.
"""

import json
import sys
from pathlib import Path


def rx(rate):
    """A receiver rate for printing; none when no receiver kernel ran."""
    return "-" if rate is None else f"{rate:.2f}"


def report(path):
    """One run's line of the table, from the last JSON line of its log."""
    lines = [line for line in Path(path).read_text().splitlines() if line[:1] == "{"]
    run = json.loads(lines[-1])
    options = run["options"]
    by_direction = {}
    for pair in run["pairs"]:
        mhz_to = run["aiclk_mhz"][pair["to"]]
        mhz_from = run["aiclk_mhz"][pair["from"]]
        payload = options["payload"]
        packets = options["packets"]
        # The first counted packet is packet 0 when every packet counts, else the
        # last of the first batch.
        first = 0 if options["fused"] else min(options["batch"], packets) - 1
        receiver = pair["receiver"]
        span_s = receiver["cycles"] / (mhz_to * 1e6)
        into = (
            (packets - 1 - first) * payload / span_s / 1e9
            if span_s and receiver["magic"]
            else None
        )
        sender_s = pair["sender"]["cycles"] / (mhz_from * 1e6)
        out = packets * payload / sender_s / 1e9 if sender_s else 0.0
        key = f"{pair['from']}->{pair['to']}"
        by_direction.setdefault(key, []).append(
            (pair["link"], into, out, pair["ok"], tuple(pair["sender_core"]))
        )
    parts = []
    for key, rates in by_direction.items():
        each = " ".join(
            f"l{link}@{core[0]},{core[1]} rx {rx(into)} tx {out:.2f}"
            for link, into, out, _, core in rates
        )
        total_in = sum(rate[1] or 0.0 for rate in rates)
        total_out = sum(rate[2] for rate in rates)
        parts.append(f"{key}: {each} (sum rx {total_in:.2f} tx {total_out:.2f} GB/s)")
    ok = run["ok"] and all(rate[3] for rates in by_direction.values() for rate in rates)
    return f"{Path(path).stem}: ok={ok} {'; '.join(parts)}"


def main():
    for path in sys.argv[1:]:
        print(report(path))


if __name__ == "__main__":
    main()
