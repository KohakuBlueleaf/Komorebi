"""Sizes given at each submit, km.UPTO among a handler's sets, words against numpy:
a stream's block 8, 4, 2, 1 beside the same class built for 8; --form width: a
tensor's width, 16 and 8 tiles; --form rect: a broadcast's rectangle, five of them
(one across the panels), beside the rectangle built in; --form all: the three."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats, power

WORDS = 1024  # float32 words a page: one tile's
CORES, TILES = 8, 20  # the cores and each core's tiles
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


@km.tile_function
def twice(a):
    return a + a


class Blocks(km.Program):
    """out = a + a, `tiles` tiles a core from rank x tiles, in blocks of `block`."""

    @km.handler
    def scale(self, a: km.Tensor, out: km.Tensor, tiles: int,
              block: km.constant[int] = 8):  # fmt: skip
        first = self.rank * tiles
        results = km.Stream(out.pages(first, tiles), write=True, block=block)
        for x in km.Stream(a.pages(first, tiles), block=block):
            results.push(twice(x))


class Wide(km.Program):
    """Each core's two columns of tiles (c = rank x 2) of x's two rows, doubled into
    its 4 pages of out; x's width (its rows' pitch) given at each submit."""

    @km.handler
    def stripe(self, out: km.Tensor, x: km.Tensor):
        c = self.rank * 2
        results = km.Stream(out.pages(self.rank * 4, 4), write=True, block=2)
        for tile in km.Stream(x.tiles[0:2, c : c + 2], block=2):
            results.push(twice(tile))


class Spread(km.Program):
    """Each row's sender reads x's tile of its row and broadcasts it along the row;
    each core of the rectangle writes it to its page y x 4 + x of out."""

    @km.handler
    def cast(self, x: km.Tensor, out: km.Tensor):
        row = km.Broadcast(self, along="row", window=2)
        if row.sender():  # noqa: SIM108 (the compiler's form)
            tiles = row.cast(x.pages(self.y, 1).read())
        else:
            tiles = row.arrive()
        total = km.tile.zeros(km.fp32)
        for tile in tiles:
            total += tile
        out.pages(self.y * 4 + self.x, 1).write(total)


class Placed(km.Program):
    """The same with its rectangle built in (rect= at the make)."""

    @km.handler
    def cast(self, x: km.Tensor, out: km.Tensor):
        row = km.Broadcast(self, along="row", window=2)
        if row.sender():  # noqa: SIM108 (the compiler's form)
            tiles = row.cast(x.pages(self.y, 1).read())
        else:
            tiles = row.arrive()
        total = km.tile.zeros(km.fp32)
        for tile in tiles:
            total += tile
        out.pages(self.y * 4 + self.x, 1).write(total)


def group_of(rect, columns):
    """The group of a rectangle's cores: its first and the span to its last."""
    x0, y0, w, h = rect
    first = y0 * columns + x0
    return first, (y0 + h - 1) * columns + x0 + w - first


def rects(session, result, dumped, runs):
    """Spread on five rectangles, one load; Placed at (0, 0, 4, 4) for the time."""
    columns, rows = session.grid
    x = np.random.default_rng(3).integers(-50, 51, (4, WORDS)).astype(np.float32)
    a = session.tensor(x, "float32")
    out = session.tensor(np.zeros((16, WORDS), np.float32), "float32")
    zero = np.zeros((16, WORDS), np.float32).view(np.uint8).reshape(-1, 4096)
    tried = [(0, 0, 4, 4), (2, 1, 4, 4), (columns // 2 - 2, 3, 4, 2),
             (1, rows - 4, 1, 4), (columns - 4, 0, 4, 1)]  # fmt: skip
    up_to = {"cast": [km.UPTO(rect=(4, 4))]}  # any rectangle up to 4 x 4
    run = session.load(Spread(x=a, out=out, handlers=up_to, session=session, log=log))
    for rect in tried:
        session.write(out, zero)
        group = group_of(rect, columns)
        run.wait(run.submit(Spread.cast, group, args=(a, out, rect)))
        got = session.read(out)[0].view(np.float32).reshape(-1, WORDS)
        want = np.zeros_like(got)
        for i in range(rect[2]):
            for j in range(rect[3]):
                want[j * 4 + i] = x[j]
        name = "_".join(map(str, rect))
        dumped[f"rect_{name}"] = got
        one = {"group": list(group), **differences(got, want)}
        if rect == (0, 0, 4, 4):
            times = [run.wait(run.submit(Spread.cast, group, args=(a, out, rect)))
                     .microseconds for _ in range(runs)]  # fmt: skip
            one["us"] = round(float(np.median(times)), 2)
        result["rects"][name] = one
    run.stop()
    result["kernel Spread"] = run.result()["error_name"]
    placed = Placed(x=a, out=out, rect=(0, 0, 4, 4), session=session, log=log)
    run = session.load(placed)
    group = group_of((0, 0, 4, 4), columns)
    session.write(out, zero)
    run.wait(run.submit(Placed.cast, group, args=(a, out)))
    got = session.read(out)[0].view(np.float32).reshape(-1, WORDS)
    dumped["placed"] = got
    times = [run.wait(run.submit(Placed.cast, group, args=(a, out))).microseconds
             for _ in range(runs)]  # fmt: skip
    want = np.stack([x[j] for j in range(4) for _ in range(4)])
    result["rects"]["placed 0_0_4_4"] = {
        **differences(got, want),
        "us": round(float(np.median(times)), 2),
    }
    run.stop()
    result["kernel Placed"] = run.result()["error_name"]


def widths(session, result, dumped):
    """Wide on tensors of 16 and then 8 tiles of columns, one load, 4 cores."""
    generator = np.random.default_rng(2)
    xs = {
        columns: generator.integers(-50, 51, (64, columns * 32)).astype(np.float32)
        for columns in (16, 8)
    }
    tensors = {c: session.tensor(x, "float32", "tile") for c, x in xs.items()}
    out = session.tensor(np.zeros((16, WORDS), np.float32), "float32")
    up_to = {"stripe": [km.UPTO(x=64)]}  # x's tile columns, up to 64
    wide = Wide(out=out, x=tensors[16], handlers=up_to, session=session, log=log)
    run = session.load(wide)
    for columns, x in xs.items():
        pages = formats.to_bytes(x, "float32", "tile")[0].view(np.float32)
        pages = pages.reshape(-1, WORDS)  # page r x columns + c
        want = np.stack([2 * pages[r * columns + c + j]
                         for core in range(4) for r in range(2)
                         for c in (core * 2,) for j in range(2)])  # fmt: skip
        run.wait(run.submit(Wide.stripe, (0, 4), args=(out, tensors[columns])))
        got = session.read(out)[0].view(np.float32).reshape(-1, WORDS)
        dumped[f"width_{columns}"] = got
        result["widths"][str(columns)] = differences(got, want)
    run.stop()
    result["kernel Wide"] = run.result()["error_name"]


def differences(got, want):
    """Wrong words: their count and the first eight (page, word)."""
    wrong = np.argwhere(got != want)
    return {"wrong": int(len(wrong)), "first": [[int(p), int(w)] for p, w in wrong[:8]]}


def timed(run, handler, args, runs):
    """The dispatcher's median microseconds of `runs` commands."""
    group = (0, CORES)
    times = [run.wait(run.submit(handler, group, args=args)).microseconds
             for _ in range(runs)]  # fmt: skip
    return round(float(np.median(times)), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--dump", default=None, help="an .npz of the outputs")
    forms = ("block", "width", "rect", "all")
    parser.add_argument("--form", default="block", choices=forms)
    args = parser.parse_args()
    x = np.random.default_rng(1).integers(-50, 51, (CORES * TILES, WORDS))
    x = x.astype(np.float32)
    result, dumped = {"cores": CORES, "tiles": TILES, "blocks": {}}, {"x": x}
    result["widths"], result["rects"] = {}, {}
    with km.Session(log=log) as session:
        if args.form in ("width", "all"):
            widths(session, result, dumped)
        if args.form in ("rect", "all"):
            rects(session, result, dumped, args.runs)
        a = session.tensor(x, "float32")
        out = session.tensor(np.zeros_like(x), "float32")
        makes = {"upto": {"scale": [km.UPTO(block=8)]}, "built": None}
        for name, handlers in makes.items() if args.form in ("block", "all") else ():
            made = Blocks(a=a, out=out, handlers=handlers, session=session, log=log)
            run = session.load(made)  # the same class: block a word, then built in
            for block in (8, 4, 2, 1) if handlers else (8,):
                session.write(out, np.zeros_like(x).view(np.uint8).reshape(-1, 4096))
                words = (a, out, TILES, block)
                run.wait(run.submit(made.scale, (0, CORES), args=words))
                got = session.read(out)[0].view(np.float32).reshape(-1, WORDS)
                dumped[f"{name}_{block}"] = got
                result["blocks"][f"{name} {block}"] = {
                    **differences(got, x + x),
                    "us": timed(run, made.scale, words, args.runs),
                }
            run.stop()
            result[f"kernel Blocks {name}"] = run.result()["error_name"]
    if args.dump:
        np.savez_compressed(args.dump, **dumped)
    result["ok"] = bool(
        all(v == "none" for k, v in result.items() if k.startswith("kernel"))
        and all(one["wrong"] == 0 for one in result["blocks"].values())
        and all(one["wrong"] == 0 for one in result["widths"].values())
        and all(one["wrong"] == 0 for one in result["rects"].values())
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
