"""Card case: kmx_spawn / kmx_join and their rejections, kmx_par beside a busy core,
kmx_par_all alone and a kmx_push into the thread's own bank, on cluster 1; every
return code and mark as expected. Prints one JSON line."""

import json
import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent
T0 = time.monotonic()
OK, FULL, NONE, BAD = 0, 1, 4, 5
WANT = [
    OK, FULL, FULL, BAD, BAD, OK, 101, OK, OK, OK, OK, 202, 303, OK, 1606, 2, 1,
    NONE, OK, 1023, BAD, BAD, OK, 0,
]  # fmt: skip


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def main():
    prog = km.Program("spawn", kernels=HERE / "kernels")
    prog.thread(cluster=1, sources=[HERE / "spawn.c"], regions={"r": 8192}, deadline=5)
    with km.Session(log=log) as session:
        run = session.load(prog)
        log("loaded")
        r = run.thread_result(timeout=5.0)
        log(f"thread result: {r}")
        run.stop()
    out = r._asdict()
    out["want"] = WANT
    out["ok"] = r.code == 0 and r.error == "none" and r.words == WANT
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
