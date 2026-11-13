"""Seconds to read one bfloat16 tile tensor back: bank by bank under one barrier
(Session.read_bf16) against the whole-buffer read (Tensor.read), `--reps` times in
turn; do they agree.
Example: python read_rate.py --tiles 64; python read_rate.py. One JSON line."""

import argparse
import json
import os
import time

import numpy as np

import komorebi as km


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiles", type=int, default=2048, help="32 x 32 tiles of 2 KB")
    parser.add_argument("--reps", type=int, default=3, help="reads each way")
    args = parser.parse_args()
    x = np.random.default_rng(1).uniform(-1, 1, (args.tiles * 32, 32)).astype("f4")
    with km.Session() as session:
        t = session.tensor(x, "bfloat16", "tile")
        banks_s, whole_s = [], []
        for _ in range(args.reps):  # alternating, so neither way gets only the warm-up
            t0 = time.monotonic()
            (banks,) = session.read_bf16(t)
            banks_s.append(round(time.monotonic() - t0, 3))
            t0 = time.monotonic()
            whole = t.read()
            whole_s.append(round(time.monotonic() - t0, 3))
    out = {
        "tiles": args.tiles,
        "mb": round(args.tiles * 2048 / 1e6, 2),
        "banks_s": banks_s,
        "whole_s": whole_s,
        "same": bool((banks == whole).all()),
        "load": round(os.getloadavg()[0], 1),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
