"""Card case: a thread fills its command ring and stops taking events. Host commands
still run, the thread's timeout stops it, and the kernel finishes cleanly. Prints one
JSON line with `ok`."""

import json
import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def main():
    prog = km.Program("ring_stall", kernels=HERE.parent / "kernels")
    prog.thread(sources=[HERE / "ring_stall.c"], regions={"unused": 64}, deadline=5.0)
    with km.Session(log=log) as session:
        run = session.load(prog)
        log("loaded")
        time.sleep(0.2)  # the thread fills its ring meanwhile
        handles = [run.submit(1, (core % run.cores, 1)) for core in range(200)]
        events = run.wait(handles, timeout=5.0)
        log("host commands done")
        result = run.thread_result(timeout=0.3)
        log(f"thread result: {result}")
        stop = run.stop()
        log("run stopped")
    out = {
        "host_done": sum(event.kind == "done" for event in events),
        "thread": result._asdict(),
        "kernel_error": stop["error_name"],
    }
    out["ok"] = (
        out["host_done"] == 200
        and not result.ran
        and (result.state, result.code, result.where) == ("done", 7, 2)
        and len(result.words) == 1
        and stop["error"] == 0
    )
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
