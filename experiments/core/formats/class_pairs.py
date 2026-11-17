"""A class product's operand formats on the card: Y = X W on one core (X 1 x 4 tiles,
W 4 x 2, in step, sums in an fp32 field, HiFi4), each pair against float64 of the
stored values, or refused at build with its message. A JSON line a pair, the table
last (README.md, "A class product's operands"). argv: --card N."""

import argparse
import json
import time

import numpy as np

import komorebi as km
from komorebi import formats

STARTED = time.monotonic()
K, N = 4, 2  # tiles
# (X, W): every pair the card ran in a class; float32 on either side is refused.
PAIRS = (
    ("bfloat16", "bfp8_b"),
    ("bfp8_b", "bfloat16"),
    ("bfp4_b", "bfp8_b"),
    ("float32", "bfp8_b"),
    ("float32", "bfloat16"),
    ("bfloat16", "float32"),
    ("float32", "float32"),
)
WORKS = 1e-2  # rms relative error under which a pair's words are right


def log(message):
    print(f"[{time.monotonic() - STARTED:6.2f}] {message}", flush=True)


class Pair(km.Program):
    """Y = X W on one core: X's K tiles and W's K x N streamed in step."""

    WIDTH = km.constant(N)
    CHUNK = km.constant(2)
    BLOCK = CHUNK * WIDTH
    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]

    @km.tile_function(fidelity="HiFi4")
    def project(x, w):
        return x @ w

    @km.handler
    def product(self, x: km.Tensor, w: km.Tensor, y: km.Tensor):
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, 0:K], block=self.CHUNK),
            km.Stream(w.tiles[0:K, 0 : self.WIDTH], block=self.BLOCK),
        ):
            self.sums += self.project(xb, wb)
        y.tiles[0:1, 0 : self.WIDTH].write(self.sums)


def stored(values, dtype):
    """The values as the tensor holds them (the host's encode, decoded)."""
    raw, _ = formats.to_bytes(values, dtype, "tile")
    return np.asarray(formats.from_bytes(raw, values.shape, dtype, "tile"), np.float64)


def one(session, left, right, rng):
    """One pair: its words' rms relative error, or the build's refusal."""
    x = rng.standard_normal((32, K * 32)).astype(np.float32)
    w = (rng.standard_normal((K * 32, N * 32)) / np.sqrt(K * 32)).astype(np.float32)
    t = {
        "x": session.tensor(x, left, "tile"),
        "w": session.tensor(w, right, "tile"),
        "y": session.empty((32, N * 32), "float32", "tile"),
    }
    out = {"x": left, "w": right}
    try:
        program = Pair(log=log, session=session, **t)
        run = session.load(program)
    except km.ProgramError as error:
        why = str(error)
        out |= {
            "result": "refused",
            "why": why[why.find("a product's") :].split(":")[0],
        }
    else:
        try:
            event = run.wait(run.submit(program.product, (0, 1), args=list(t.values())))
        finally:
            run.stop()
        (raw,) = session.read(t["y"])
        got = np.asarray(
            formats.from_bytes(
                np.asarray(raw).reshape(-1), (32, N * 32), "float32", "tile"
            ),
            np.float64,
        )
        want = stored(x, left) @ stored(w, right)
        rms = float(np.sqrt(((got - want) ** 2).mean() / (want**2).mean()))
        works = event.kind == "done" and rms < WORKS
        out |= {"result": "works" if works else "WRONG", "kind": event.kind, "rms": rms}
    for tensor in t.values():
        tensor.free()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(3)
    table = []
    with km.Session(log=log, card=args.card) as session:
        for left, right in PAIRS:
            log(f"pair {left} x {right}")
            table.append(one(session, left, right, rng))
            print(json.dumps(table[-1]), flush=True)
    print(json.dumps({"pairs": table}), flush=True)


if __name__ == "__main__":
    main()
