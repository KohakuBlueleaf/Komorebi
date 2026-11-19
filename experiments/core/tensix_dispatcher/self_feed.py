"""The dispatcher's own ceiling: the product kernel built with KM_TD_SELF_FEED, so one
host NOP becomes N commands the dispatcher submits itself, each on core j % spread.
One JSON line a run, e.g. `python self_feed.py --n 100000 --depth 16 --split B`."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.session.dispatch import tensix

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[2] / "tutorial" / "38_measuring" / "kernels"


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=100000, help="the dispatcher's own")
    parser.add_argument("--depth", type=int, default=16, help="ring slots in use")
    parser.add_argument("--spread", type=int, default=32, help="cores, one each")
    parser.add_argument("--split", choices=sorted(tensix.LOOPS), default="B")
    parser.add_argument("--runs", type=int, default=1, help="loads in one open")
    parser.add_argument("--parts", action="store_true", help="out's ticks by part")
    parser.add_argument(
        "--ack-count", action="store_true", help="the workers count their acks too"
    )
    return parser.parse_args(argv)


def one(session, program, args):
    """One load: the host's NOP on core 0 and the n commands before it."""
    run = session.load(program, depth=args.depth)
    started = time.perf_counter()
    event = run.wait(run.submit(0, (0, 1)), timeout=60.0)  # 0: NOP
    host_s = time.perf_counter() - started
    run.stop()
    result = run.result()
    last = result["last"]
    ok = (
        result["error_name"] == "none"
        and event.kind == "done"
        and result["commands"] == args.n + 2  # the host's NOP, then the EXIT
    )
    return {
        "ok": ok,
        "split": args.split,
        "ack_count": args.ack_count,
        "n": args.n,
        "depth": args.depth,
        "spread": args.spread,
        "us_per_command": round(last / run.tick_mhz / args.n, 4),
        "ticks_per_command": round(last / args.n, 1),
        "max_busy": result["max_busy"],
        "commands": result["commands"],
        "events": result["events"],
        "passes": result["passes"],
        "idle": result["idle"],
        "full": result["full"],
        "held": result["held"],
        "error": result["error_name"],
        "event": event.kind,
        "host_s": round(host_s, 3),
        "tick_mhz": run.tick_mhz,
        **run.port.tail,
        "parts": [round(p / args.n, 1) for p in run.port.parts] if args.parts else None,
    }


def main(argv=None):
    args = parse(argv)
    kept = tensix.SPLIT, tensix.DEFINES  # one process may run other scripts after
    try:
        measure(args)
    finally:
        tensix.SPLIT, tensix.DEFINES = kept


def measure(args):
    """The runs of `args`, the module's split and defines set for them."""
    tensix.SPLIT = args.split
    tensix.DEFINES = {"KM_TD_SELF_FEED": args.n, "KM_TD_SELF_SPREAD": args.spread}
    if args.parts:
        tensix.DEFINES["KM_TD_PARTS"] = 1
    counting = {}
    if args.ack_count:  # the workers' add lands on the count in reads
        tensix.DEFINES["KM_TD_ACK_COUNT"] = 1
        at = tensix.constants()["kTdAckCountAt"]
        counting = {"KM_TD_ACK_COUNT": 1, "KM_TD_ACK_COUNT_AT": at}
    with km.Session(dispatcher="tensix") as session:
        numbers = np.zeros((32, 32), np.uint32)  # tutorial 38's program, never run
        source = session.tensor(numbers, "uint32", "tile")
        destination = session.tensor(numbers, "uint32", "tile")
        program = km.Program(
            "self_feed",
            kernels=KERNELS,
            handlers=KERNELS / "handlers.h",
            buffers=[
                km.Buffer(
                    index=0, format="uint32", page_bytes=source.page_bytes, pages=8
                )
            ],
            compile_args={
                "reader": source.tensor_args(),
                "writer": destination.tensor_args(),
            },
            defines=counting,
        )
        for _ in range(args.runs):
            print(json.dumps(one(session, program, args)), flush=True)


if __name__ == "__main__":
    main()
