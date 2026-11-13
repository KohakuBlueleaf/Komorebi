"""Card case: a thread still running at the host's timeout is asked to stop, returns,
and its result says so; the region it wrote reads back exact. Prints one JSON line
with `ok`."""

import json
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
STARTED = time.monotonic()
OUT_BYTES = 3 * 4096 + 100  # region pages 2 to 5 of the bank, the last one in part


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def main():
    prog = km.Program("spin", kernels=KERNELS)
    prog.thread(
        sources=[HERE / "spin.c"],
        regions={"unused": 64, "out": OUT_BYTES},
        deadline=5.0,
    )
    with km.Session(log=log) as session:
        run = session.load(prog)
        log("loaded")
        started = time.monotonic()
        result = run.thread_result(timeout=0.2)
        stop_s = time.monotonic() - started
        log(f"thread result: {result}")
        run.stop()
        log("run stopped")
        got = run.thread.read("out")
        log(f"region read: {got.size} bytes")
    want = (np.arange(1, OUT_BYTES // 4 + 1, dtype=np.uint32) * 2654435761).view(
        np.uint8
    )
    out = {"result": result._asdict(), "stop_s": round(stop_s, 3)}
    out["region_exact"] = bool(got.size == OUT_BYTES and (got == want).all())
    out["ok"] = (
        not result.ran
        and (result.state, result.error, result.code, result.where)
        == ("done", "none", 7, 2)
        and stop_s < 1.5
        and out["region_exact"]
    )
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
