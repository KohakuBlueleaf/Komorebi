"""Card case: kmx_spawn / kmx_join and their rejections, kmx_parallel beside a busy
core, kmx_parallel_all alone and a kmx_push into the thread's own bank, on cluster 1;
every return code and mark as expected. Prints one JSON line."""

import json
import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()
OK, FULL, NONE, BAD = 0, 1, 4, 5
WANT = [
    OK, FULL, FULL, BAD, BAD, OK, 101, OK, OK, OK, OK, 202, 303, OK, 1606, 2, 1,
    NONE, OK, 1023, BAD, BAD, OK, 0,
]  # fmt: skip


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def main():
    prog = km.Program("spawn", kernels=HERE / "kernels")
    prog.thread(cluster=1, sources=[HERE / "spawn.c"], regions={"r": 8192}, deadline=5)
    with km.Session(log=log) as session:
        run = session.load(prog)
        log("loaded")
        result = run.thread_result(timeout=5.0)
        log(f"thread result: {result}")
        run.stop()
    out = result._asdict()
    out["want"] = WANT
    out["ok"] = result.code == 0 and result.error == "none" and result.words == WANT
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
