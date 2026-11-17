"""A tile function's scalars from a tensor page (README.md): a 3 x 4 camera, a page a
row, a row's words read as float words for `clip(x, y, z, m0, m1, m2, m3)`, beside the
command with float words and `x * m0`, against numpy. A JSON line a command and format.
argv: --cards N (one card) or 0,1 (two: the group across the islands' edge)."""

import argparse
import json
import time

import numpy as np

import komorebi as km
from komorebi import formats

STARTED = time.monotonic()
CORES = 12
TILES = 4  # a core's
REPEAT = 5  # each command's runs, for its median microseconds
FORMATS = ("float32", "bfloat16")


def log(message):
    print(f"[{time.monotonic() - STARTED:6.2f}] {message}", flush=True)


@km.tile_function
def clip(x, y, z, m0: float, m1: float, m2: float, m3: float):
    return x * m0 + y * m1 + z * m2 + m3


class Camera(km.Program):
    """Each handler: a core's TILES tiles of x, y and z; the camera's row r read from
    its page (row) or given as command words (row_words)."""

    @km.handler
    def row(
        self,
        x: km.Tensor,
        y: km.Tensor,
        z: km.Tensor,
        o: km.Tensor,
        cam: km.Tensor,
        r: int,
    ):
        m0, m1, m2, m3 = cam.pages(r, 1).read(km.fp32)[0:4]
        first = self.rank * TILES
        results = km.Stream(o.pages(first, TILES), write=True)
        for a, b, c in zip(
            km.Stream(x.pages(first, TILES)),
            km.Stream(y.pages(first, TILES)),
            km.Stream(z.pages(first, TILES)),
        ):
            results.push(clip(a, b, c, m0, m1, m2, m3))

    @km.handler
    def row_words(
        self,
        x: km.Tensor,
        y: km.Tensor,
        z: km.Tensor,
        o: km.Tensor,
        m0: float,
        m1: float,
        m2: float,
        m3: float,
    ):
        first = self.rank * TILES
        results = km.Stream(o.pages(first, TILES), write=True)
        for a, b, c in zip(
            km.Stream(x.pages(first, TILES)),
            km.Stream(y.pages(first, TILES)),
            km.Stream(z.pages(first, TILES)),
        ):
            results.push(clip(a, b, c, m0, m1, m2, m3))

    @km.handler
    def scaled(self, x: km.Tensor, o: km.Tensor, cam: km.Tensor):
        (m0,) = cam.pages(0, 1).read(km.fp32)[0:1]
        first = self.rank * TILES
        results = km.Stream(o.pages(first, TILES), write=True)
        for a in km.Stream(x.pages(first, TILES)):
            results.push(a * m0)


def inputs(seed=23):
    """x, y, z: normals, CORES * TILES tiles; the camera: 3 rows of 4, each row a
    page of 1,024 words (the rest zero)."""
    rng = np.random.default_rng(seed)
    x, y, z = rng.standard_normal((3, CORES * TILES * 32, 32)).astype(np.float32)
    camera = rng.uniform(-2.0, 2.0, (3, 4)).astype(np.float32)
    pages = np.zeros((3, 1024), np.float32)
    pages[:, :4] = camera
    return x, y, z, camera, pages


def stored(values, dtype):
    """The values as the tensor holds them (the host's encode, decoded)."""
    raw, _ = formats.to_bytes(values, dtype, "tile")
    return np.asarray(formats.from_bytes(raw, values.shape, dtype, "tile"), np.float32)


def compared(got, want):
    """The largest error over the largest value, and lanes beyond 1e-6 of it."""
    error = np.abs(got.astype(np.float64) - want)
    scale = np.abs(want).max()
    return {
        "max_over_largest": float(error.max() / scale),
        "lanes_over_1e-6": int((error > 1e-6 * scale).sum()),
    }


def timed(run, handler, group, args):
    """REPEAT runs of one command: each run's microseconds; raises unless done."""
    times = []
    for _ in range(REPEAT):
        event = run.wait(run.submit(handler, group, args=args), timeout=30.0)
        if event.kind != "done":
            raise RuntimeError(f"{handler.name}: {event}")
        times.append(round(event.microseconds, 1))
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", default="")
    cards = [int(c) for c in parser.parse_args().cards.split(",") if c]
    x, y, z, camera, pages = inputs()
    opened = {"cards": cards} if len(cards) > 1 else {"card": next(iter(cards), None)}
    with km.Session(log=log, **opened) as session:
        group = (0, CORES)
        if len(cards) > 1:  # half the group on each island
            group = (session.graph.first(1) - CORES // 2, CORES)
        cam = session.tensor(pages, "float32", "row")
        for dtype in FORMATS:
            t = {n: session.tensor(v, dtype, "tile") for n, v in zip("xyz", (x, y, z))}
            t["o"] = session.empty(x.shape, "float32", "tile")
            program = Camera(log=log, session=session, cam=cam, **t)
            run = session.load(program)
            outs, rows = {}, []
            try:
                for r in range(3):
                    for name in ("row", "row_words"):
                        out = outs[(name, r)] = session.empty(
                            x.shape, "float32", "tile"
                        )
                        args = [t["x"], t["y"], t["z"], out]
                        args += [cam, r] if name == "row" else [*map(float, camera[r])]
                        times = timed(run, getattr(program, name), group, args)
                        rows.append({"handler": name, "row": r, "us": times})
                out = outs[("scaled", 0)] = session.empty(x.shape, "float32", "tile")
                times = timed(run, program.scaled, group, [t["x"], out, cam])
                rows.append({"handler": "scaled", "row": 0, "us": times})
            finally:
                run.stop()
            # read after the stop: a read waits for the cores, which a run holds
            sx, sy, sz = (stored(v, dtype).astype(np.float64) for v in (x, y, z))
            got = {k: np.asarray(v.read(), np.float32) for k, v in outs.items()}
            for row in rows:
                m = camera[row["row"]].astype(np.float64)
                want = (
                    sx * m[0]
                    if row["handler"] == "scaled"
                    else sx * m[0] + sy * m[1] + sz * m[2] + m[3]
                )
                mine = got[(row["handler"], row["row"])]
                row = {"format": dtype, "cards": cards or [0], **row}
                row["median_us"] = float(np.median(row["us"]))
                row |= compared(mine, want)
                if row["handler"] == "row":
                    words = got[("row_words", row["row"])]
                    row["lanes_differ_from_row_words"] = int((mine != words).sum())
                print(json.dumps(row), flush=True)
            for tensor in [*t.values(), *outs.values()]:
                tensor.free()
    log("done")


if __name__ == "__main__":
    main()
