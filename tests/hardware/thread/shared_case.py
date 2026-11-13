"""Card case: one image on clusters 2 and 3 with two shared regions of bank 7; cluster 2
writes one and raises a flag in the other, cluster 3 waits for it and sums the first.
The sum and the region read back after the stop are exact. Prints one JSON line."""

import json
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()
WORDS = 4096 + 100  # a page and a part of one


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def main():
    prog = km.Program("shared", kernels=HERE / "kernels")
    prog.region("data", 4 * WORDS)
    prog.region("flag", 64)
    prog.thread(
        clusters=(2, 3),
        sources=[HERE / "shared.c"],
        regions={"own": 64},
        shared=["data", "flag"],
        deadline=5.0,
    )
    with km.Session(log=log) as session:
        run = session.load(prog)
        log("loaded")
        results = run.thread_results(timeout=5.0)
        log(f"thread results: {results}")
        run.stop()
        log("run stopped")
        got = run.threads[3].read("data").view(np.uint32)
    want = np.arange(1, WORDS + 1, dtype=np.uint32) * np.uint32(2654435761)
    out = {cluster: result._asdict() for cluster, result in results.items()}
    out["region_exact"] = bool(got.size == WORDS and (got == want).all())
    out["ok"] = (
        [(result.code, result.where) for result in results.values()] == [(0, 2), (0, 3)]
        and results[3].words == [int(want.sum(dtype=np.uint32))]
        and out["region_exact"]
    )
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
