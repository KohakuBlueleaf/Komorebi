"""Card case: a thread still running at the host's timeout is asked to stop, returns,
and its result says so. Prints one JSON line with `ok`."""

import json
import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def main():
    prog = km.Program("spin", kernels=KERNELS)
    prog.thread(sources=[HERE / "spin.c"], regions={"unused": 64}, deadline=5.0)
    with km.Session(log=log) as session:
        run = session.load(prog)
        log("loaded")
        t = time.monotonic()
        result = run.thread_result(timeout=0.2)
        stop_s = time.monotonic() - t
        log(f"thread result: {result}")
        run.stop()
        log("run stopped")
    out = {"result": result._asdict(), "stop_s": round(stop_s, 3)}
    out["ok"] = (
        not result.ran
        and (result.state, result.error, result.code, result.where)
        == ("done", "none", 7, 2)
        and stop_s < 1.5
    )
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
