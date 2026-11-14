"""Samples a card's tt_aiclk file in a loop and writes (monotonic seconds, MHz) pairs
as JSON when it gets SIGTERM or after --seconds. Used beside spread.py.
Example: python sample_clock.py --card 0 --seconds 30 --out clock.json"""

import argparse
import json
import signal
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{args.card}/tt_aiclk")
    samples = []
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    end = time.monotonic() + args.seconds
    while not stop and time.monotonic() < end:
        now = time.monotonic()
        text = path.read_text().strip()
        samples.append((round(now, 5), int(text) if text.isdigit() else 0))
    Path(args.out).write_text(json.dumps(samples))


if __name__ == "__main__":
    main()
