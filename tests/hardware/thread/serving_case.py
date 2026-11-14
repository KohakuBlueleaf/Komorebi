"""Card case: a thread with no kmx_main serves a command that needs its four X280
cores (kmx_parallel) well past its deadline; all four mark their line. One JSON line."""

import json
import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()
DEADLINE_S = 0.5  # the thread's: it bounds kmx_main only


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def main():
    prog = km.Program("serving", kernels=HERE / "kernels")
    prog.thread(
        cluster=1,
        sources=[HERE / "serving.c"],
        regions={"r": 4096},
        handlers={"parallel": 1},
        deadline=DEADLINE_S,
    )
    with km.Session(log=log) as session:
        run = session.load(prog)
        log("loaded")
        time.sleep(3 * DEADLINE_S)
        event = run.wait(run.submit("parallel", cluster=1, result=True), timeout=5.0)
        log(f"after the thread's deadline: {event}")
        run.stop()
    out = {"kind": event.kind, "cores": event.result}
    out["ok"] = event.kind == "done" and event.result == 4
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
