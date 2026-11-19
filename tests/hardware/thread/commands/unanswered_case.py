"""Card case: a command the thread does not answer by its deadline comes back as its
TIMEOUT naming the cluster, and the same run then gets the thread's answer to the next
command and a core's to a grid command. Prints one JSON line; exits 1 unless ok."""

import json
import sys
import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent
LESSON = HERE.parents[3] / "tutorial" / "42_commands_to_a_thread"
DEADLINE_S = 0.2  # every command's (session.load(deadline=))
HOLD_MS = 600  # the held command's answer comes this late
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def program():
    made = km.Program(
        "unanswered",
        kernels=LESSON / "kernels",
        handlers=LESSON / "kernels" / "handlers.h",
    )
    made.thread(
        cluster=1,
        sources=[HERE / "unanswered.c"],
        regions={"unused": 64},
        handlers={"hold": 1, "sum": 2},
    )
    return made


def main():
    out = {"deadline_s": DEADLINE_S, "hold_ms": HOLD_MS}
    with km.Session(log=log) as session:
        run = session.load(program(), deadline=DEADLINE_S)
        held = run.submit("hold", cluster=1, args=(HOLD_MS,), result=True)
        try:
            event = run.wait(held, timeout=5.0)
            out["held"] = {"kind": event.kind, "cluster": int(event.auxiliary) & 0xFF}
        except km.CommandTimeout as error:
            out["held"] = {"kind": "raised", "error": str(error)[:300]}
        time.sleep(HOLD_MS / 1000)  # the late answer sends no event
        answered = run.wait(run.submit("sum", cluster=1, args=(20, 22), result=True))
        out["answered"] = {"kind": answered.kind, "result": int(answered.result)}
        grid = run.wait(run.submit("add", (0, 1), args=(40, 2), result=True))
        out["grid"] = {"kind": grid.kind, "result": int(grid.result)}
        run.stop()
    out["ok"] = bool(
        out["held"] == {"kind": "timeout", "cluster": 1}
        and out["answered"] == {"kind": "done", "result": 42}
        and out["grid"] == {"kind": "done", "result": 42}
    )
    print(json.dumps(out), flush=True)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
