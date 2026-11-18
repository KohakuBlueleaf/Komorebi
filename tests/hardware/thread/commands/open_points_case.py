"""Card case: three limits of commands to a thread on cluster 1. A held command and 7
more fill its 8 lines: a ninth from the host is rejected (`thread`), the run goes on;
kmx_parallel inside kmx_program; commands the thread submits and never waits for, kept
between commands, then taken with kmx_poll. Prints one JSON line."""

import json
import sys
import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent
EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "13_thread_commands"
sys.path.insert(0, str(EXAMPLE))
import thread_commands as example  # noqa: E402

STARTED = time.monotonic()
CLUSTER = 1
HANDLERS = {"hold": 1, "echo": 2, "sum": 3, "fire": 4, "poll": 5}
FIRES = 20  # fire commands of FIRE_EACH grid commands each, never waited for
FIRE_EACH = 10


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def program():
    made = km.Program(
        "open_points", kernels=example.KERNELS, handlers=example.KERNELS / "handlers.h"
    )
    made.thread(
        cluster=CLUSTER,
        sources=[HERE / "open_points.c"],
        regions={"unused": 4096},
        handlers=HANDLERS,
    )
    return made


def held(run):
    """hold (0.2 s) and 7 echoes take the 8 lines; a ninth echo is rejected; then all
    answer and one more echo is answered."""
    handles = [run.submit("hold", cluster=CLUSTER, args=(200_000, 41), result=True)]
    handles += [
        run.submit("echo", cluster=CLUSTER, args=(index,), result=True)
        for index in range(7)
    ]
    ninth = run.submit("echo", cluster=CLUSTER, args=(100,), result=True)
    try:
        run.wait(ninth, timeout=5.0)
        rejected = None
    except km.CommandRejected as error:
        rejected = error.reason_name
    words = [run.wait(handle, timeout=5.0).result for handle in handles]
    after = run.wait(run.submit("echo", cluster=CLUSTER, args=(7,), result=True))
    return {
        "rejected": rejected,
        "words_right": words == [42, *range(1, 8)],
        "after": after.result,
    }


def summed(run, count):
    event = run.wait(run.submit("sum", cluster=CLUSTER, args=(count,), result=True))
    want = count * (count - 1) // 2 & 0xFFFFFFFF
    return {"word": event.result, "right": event.result == want}


def fired(run):
    """FIRES commands that each submit FIRE_EACH grid commands and wait for none; an
    echo between them; then poll takes the events."""
    returns, echoes = [], []
    for index in range(FIRES):
        event = run.wait(
            run.submit("fire", cluster=CLUSTER, args=(FIRE_EACH,), result=True),
            timeout=5.0,
        )
        if event.kind != "done":
            return {"fire": index, "kind": event.kind, "returns": returns}
        word = event.result
        returns.append([word & 0xFFFF, word >> 16])
        echoes.append(
            run.wait(
                run.submit("echo", cluster=CLUSTER, args=(index,), result=True),
                timeout=5.0,
            ).result
            == index + 1
        )
    time.sleep(0.05)
    taken = run.wait(run.submit("poll", cluster=CLUSTER, result=True)).result
    again = run.wait(
        run.submit("fire", cluster=CLUSTER, args=(FIRE_EACH,), result=True)
    ).result
    return {
        "submitted": sum(first for first, _ in returns),
        "statuses": sorted({status for _, status in returns}),
        "echoes_right": all(echoes),
        "polled": {"done": taken & 0xFF, "rejected": taken >> 8 & 0xFF,
                   "other": taken >> 16},  # fmt: skip
        "fire_after_poll": [again & 0xFFFF, again >> 16],
    }


def main():
    out = {}
    with km.Session(log=log) as session:
        run = session.load(program())
        log("loaded")
        out["held"] = held(run)
        log(f"held: {out['held']}")
        out["sum"] = summed(run, 100_000)
        log(f"kmx_parallel in kmx_program: {out['sum']}")
        out["fired"] = fired(run)
        log(f"fired: {out['fired']}")
        result = run.stop()
        out["dispatcher_error"] = result["error_name"]
    out["ok"] = bool(
        out["held"]["rejected"] == "thread"
        and out["held"]["words_right"]
        and out["held"]["after"] == 8
        and out["sum"]["right"]
        and out["fired"].get("echoes_right")
        and out["fired"]["submitted"] == FIRES * FIRE_EACH
        and out["fired"]["polled"]["done"] == 32  # KMX_STASH: the newest kept
        and out["fired"]["fire_after_poll"] == [FIRE_EACH, 0]
        and out["dispatcher_error"] == "none"
    )
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
