"""A scan's state in a tile function (README.md): h_t = a_t h_(t-1) + u_t over
TOKENS tiles in one command, and in two of TOKENS / 2 (the second started from the
state the first wrote), against float64 on the stored values. A JSON line a format.
argv: --cards N (one card; the first visible when not given) or 0,1 (two)."""

import argparse
import json
import time

import numpy as np

import komorebi as km
from komorebi import formats

STARTED = time.monotonic()
TOKENS = 64
HALF = TOKENS // 2
FORMATS = ("float32", "bfloat16")


def log(message):
    print(f"[{time.monotonic() - STARTED:6.2f}] {message}", flush=True)


@km.tile_function
def gated(a, h, u):
    return a * h + u


class Scan(km.Program):
    """run_all: TOKENS tiles from a zero state; run_half: HALF tiles from `first`,
    the state started from h0's tile and written to hs."""

    state: km.L1[km.fp32, 1, 32, 32]

    @km.handler
    def run_all(self, a: km.Tensor, u: km.Tensor, y: km.Tensor, hs: km.Tensor):
        h = km.tile.zeros(km.fp32)
        results = km.Stream(y.pages(0, TOKENS), write=True)
        for g, x in zip(km.Stream(a.pages(0, TOKENS)), km.Stream(u.pages(0, TOKENS))):
            h = gated(g, h, x)
            results.push(h)
        hs.pages(0, 1).write(h)

    @km.handler
    def run_half(
        self,
        a: km.Tensor,
        u: km.Tensor,
        y: km.Tensor,
        h0: km.Tensor,
        hs: km.Tensor,
        first: int,
    ):
        self.state.fill(km.Stream(h0.pages(0, 1)))
        h = self.state
        results = km.Stream(y.pages(first, HALF), write=True)
        for g, x in zip(
            km.Stream(a.pages(first, HALF)), km.Stream(u.pages(first, HALF))
        ):
            h = gated(g, h, x)
            results.push(h)
        hs.pages(0, 1).write(h)


def inputs(seed=17):
    """a: decays in [0.5, 1); u: normals. TOKENS tiles each, a tile a token."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 1.0, (TOKENS * 32, 32)).astype(np.float32)
    u = rng.standard_normal((TOKENS * 32, 32)).astype(np.float32)
    return a, u


def stored(values, dtype):
    """The values as the tensor holds them (the host's encode, decoded)."""
    raw, _ = formats.to_bytes(values, dtype, "tile")
    return np.asarray(formats.from_bytes(raw, values.shape, dtype, "tile"), np.float32)


def reference(a, u):
    """Each token's state in float64."""
    a, u = (v.astype(np.float64).reshape(TOKENS, 32, 32) for v in (a, u))
    h, out = np.zeros((32, 32)), []
    for t in range(TOKENS):
        h = a[t] * h + u[t]
        out.append(h)
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="")
    cards = [int(c) for c in parser.parse_args().cards.split(",") if c]
    a, u = inputs()
    opened = {"cards": cards} if len(cards) > 1 else {"card": next(iter(cards), None)}
    with km.Session(log=log, **opened) as session:
        for dtype in FORMATS:
            t = {
                "a": session.tensor(a, dtype, "tile"),
                "u": session.tensor(u, dtype, "tile"),
                "y": session.empty(a.shape, "float32", "tile"),
                "hs": session.empty((32, 32), "float32", "tile"),
            }
            y2 = session.empty(a.shape, "float32", "tile")
            h0 = session.tensor(np.zeros((32, 32), np.float32), "float32", "tile")
            s1 = session.empty((32, 32), "float32", "tile")
            s2 = session.empty((32, 32), "float32", "tile")
            program = Scan(log=log, session=session, h0=h0, **t)
            run = session.load(program)
            kinds = []
            try:
                sends = [
                    ("run_all", [t["a"], t["u"], t["y"], t["hs"]]),
                    ("run_half", [t["a"], t["u"], y2, h0, s1, 0]),
                    ("run_half", [t["a"], t["u"], y2, s1, s2, HALF]),
                ]
                for name, args in sends:
                    sent = run.submit(getattr(program, name), (0, 1), args=args)
                    kinds.append(run.wait(sent, timeout=30.0).kind)
                    log(f"{dtype} {name}: {kinds[-1]}")
            finally:
                run.stop()
            # read after the stop: a read waits for the cores, which a run holds
            got, halves = (np.asarray(v.read(), np.float32) for v in (t["y"], y2))
            last, last2 = (np.asarray(v.read(), np.float32) for v in (t["hs"], s2))
            want = reference(stored(a, dtype), stored(u, dtype))
            error = np.abs(got.astype(np.float64) - want)
            row = {
                "format": dtype,
                "kinds": kinds,
                "max_over_largest": float(error.max() / np.abs(want).max()),
                "last_token_max_over_largest": float(
                    error[-32:].max() / np.abs(want[-32:]).max()
                ),
                "two_commands_lanes_differ": int((halves != got).sum()),
                "final_state_equal_last": bool((last == got[-32:]).all()),
                "two_commands_final_lanes_differ": int((last2 != last).sum()),
            }
            print(json.dumps(row), flush=True)
            for tensor in [*t.values(), y2, h0, s1, s2]:
                tensor.free()
    log("done")


if __name__ == "__main__":
    main()
