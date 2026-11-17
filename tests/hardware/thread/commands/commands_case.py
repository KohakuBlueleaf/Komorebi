"""Card case: commands to an X280 thread from a stream (host-fed and on the X280), the
thread's word feeding a grid command, and a thread serving past its own deadline
(kmx_main's), on cluster 2; every word against numpy. Prints one JSON line."""

import json
import sys
import time
from pathlib import Path

import komorebi as km
from komorebi.engines.stream import Stream

EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "13_thread_commands"
sys.path.insert(0, str(EXAMPLE))
import thread_commands as example  # noqa: E402

STARTED = time.monotonic()
CLUSTER = 2
DEADLINE_S = 0.5  # the thread's: it bounds kmx_main only
ROUNDS = 50


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def replay(rounds, start):
    """x = AFFINE(FOLD(x), 5) on the group, `rounds` times."""
    value = start
    for _ in range(rounds):
        value = example.affine(example.fold(value), 5, example.GROUP[1])
    return value


def build_stream(made):
    """`ROUNDS` rounds: FOLD on the thread, its word into AFFINE on the group; HALT
    with the last AFFINE word."""
    stream = Stream()
    index, value = stream.registers(2)
    stream.set(value, 7)
    top = stream.label()
    stream.submit(
        made.threads[CLUSTER].handler("fold"), cluster=CLUSTER, args=[value],
        result=True,
    )  # fmt: skip
    stream.wait(0)
    stream.result(value)
    stream.submit(
        made.handler("affine"), group=example.GROUP, args=[value, 5], result=True
    )
    stream.wait(0)
    stream.result(value)
    stream.addi(index, index, 1)
    stream.jlt(index, stream.const(ROUNDS), top)
    stream.halt(value)
    return stream


def main():
    made = example.program(CLUSTER)
    made.threads[CLUSTER].deadline = DEADLINE_S
    want = replay(ROUNDS, 7)
    out = {"want": want}
    with km.Session(log=log) as session:
        run = session.load(made)
        log("loaded")
        stream = build_stream(made)
        hostfed = run.run_hostfed(stream, timeout=30.0)
        out["hostfed"] = {"status": hostfed.status, "value": hostfed.registers[1]}
        log(f"host-fed stream: {out['hostfed']}")
        upload = run.upload(stream)
        end = run.wait_engine(run.start(upload), timeout=30.0)
        run.release(upload)
        out["x280"] = {"kind": end.kind, "halt": end.value}
        log(f"stream on the X280: {out['x280']}")
        time.sleep(max(3 * DEADLINE_S - (time.monotonic() - STARTED), 0))
        late = time.monotonic() - STARTED
        value = run.wait(run.submit("fold", cluster=CLUSTER, args=(99,), result=True))
        out["after_deadline"] = {
            "seconds": round(late, 2),
            "kind": value.kind,
            "right": value.result == example.fold(99),
        }
        log(f"after the thread's deadline: {out['after_deadline']}")
        result = run.stop()
        out["dispatcher_error"] = result["error_name"]
    out["ok"] = bool(
        hostfed.status == "ended"
        and hostfed.registers[1] == want
        and end.kind == "end"
        and end.value == want
        and out["after_deadline"]["right"]
        and late > 2 * DEADLINE_S
        and result["error"] == 0
    )
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
