"""The frame of frame.py driven by the class's X280 thread (thread/frame.c), against
numpy: each pass a command the thread submits, raster's group the bins in use, a word
tally returns that the thread reads from the command's event."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nn"))

import numpy as np
from forms import error, finish, log, read, tiles
from frame import BINS, Frame, reference

import komorebi as km


class FrameThread(Frame):
    """frame.py's passes, a tally and the thread."""

    scratch = km.region(4096)  # a thread has at least one region; frame.c uses none
    thread = km.x280_thread("thread/frame.c", cluster=1, deadline=10.0)

    @km.handler
    def tally(self, counts: km.Tensor, groups: int):
        (active,) = counts.pages(groups, 1).read()[0:1]
        if self.rank == self.leader:
            return active


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--tiles", type=int, default=2, help="tiles a core")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cores, n = args.cores, args.tiles
    count = cores * n
    shape = (count * 32, 32)
    x, y, z, albedo, light = (
        rng.uniform(-0.5, 2.0, shape).astype(np.float32) for _ in range(5)
    )
    used = BINS - 1  # the last bin is empty: raster runs on the bins in use
    ids = np.concatenate([np.arange(used), rng.integers(0, used, count - used)])
    ids = rng.permutation(ids).astype(np.uint32)
    page = np.zeros((1, 1024), np.uint32)
    page[0, :count] = ids
    report = {"cores": cores, "tiles": count, "bins": BINS}
    with km.Session(log=log) as session:
        tx, ty, tz, ta, tl = (
            tiles(session, v, "float32") for v in (x, y, z, albedo, light)
        )
        bins = session.tensor(page, "uint32")
        screen, lists, zbuf, color, image = (
            tiles(session, np.zeros(shape), "float32") for _ in range(5)
        )
        counts = session.tensor(np.zeros((BINS + 1, 1024), np.uint32), "uint32")
        program = FrameThread(
            x=tx, y=ty, z=tz, screen=screen, bins=bins, lists=lists, counts=counts,
            zbuf=zbuf, albedo=ta, light=tl, color=color, image=image,
            thread_words=[cores, n, count, BINS], session=session, log=log,
        )  # fmt: skip
        run = session.load(program, deadline=1.0, thread="card")
        try:
            result = run.thread_results(timeout=15.0)[1]
        finally:
            run.stop()
        got = read(session, image)
    report["thread_code"] = result.code
    report["thread_words"] = [int(w) for w in result.words]
    report["thread_us"] = round(result.microseconds, 1)
    report["active_wrong"] = {"count": int(list(result.words[:1]) != [used])}
    report["code"] = {"count": int(result.code != 0)}
    report["image"] = error(got, reference(x, y, z, ids, albedo, light))
    log(f"thread code {result.code}, words {report['thread_words']}: {report['image']}")
    finish(
        report,
        [("code", "count", 0), ("active_wrong", "count", 0), ("image", "rms", 1e-6)],
    )


if __name__ == "__main__":
    sys.exit(main())
