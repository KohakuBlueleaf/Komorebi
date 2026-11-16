"""Example 29's chain with its three handlers as overlays (steps_class.py's bodies):
the resident chain, then the overlays' chain at each slot count asked (default, a
slot a handler; 1; 2), every word of y, z, s and t against numpy after each, the
times beside the resident chain's. See ../README.md."""

import argparse
import json
import sys
import time

import numpy as np
from steps_class import Steps, bad_words, by_chain, expected, made, outputs

import komorebi as km
from komorebi.bench import power

STARTED = time.monotonic()


class OverlaySteps(Steps):
    """Steps with SCALE, ADD and SUM as overlays: the chain's walk loads each
    record's overlay, waits for the record's input, then calls it."""

    scale = km.handler(overlay=True)(Steps.scale.function)
    add = km.handler(overlay=True)(Steps.add.function)
    sum = km.handler(overlay=True)(Steps.sum.function)


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8, help="cores 0 .. N - 1")
    parser.add_argument("--width", type=int, default=2, help="tiles a core")
    parser.add_argument("--runs", type=int, default=200, help="timed runs a form")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--slots",
        default="default",
        help="slot counts in turn, comma separated: default (a slot a handler) or 1-4",
    )
    parser.add_argument(
        "--skip-resident", action="store_true", help="no resident chain first"
    )
    parser.add_argument(
        "--deadline", type=float, default=0.5, help="a command's deadline, s"
    )
    return parser.parse_args()


def chained(session, program, tensors, args, x):
    """The chain run at three factors (every word checked after each), then timed:
    (checks, host us a run, card us a run median, the run's error)."""
    run = session.load(program, wait=args.deadline, deadline=args.deadline)
    names = [tensors[name] for name in "xyzst"]
    tiles = args.cores * args.width
    cores = (0, args.cores)
    checks = {}
    for factor in (2, 5, 1):
        chain = run.chain(
            program.steps(*names, float(factor), args.width, tiles), cores=cores
        )
        by_chain(run, chain)
        checks[f"x{factor}"] = bad_words(outputs(session, tensors), expected(x, factor))
        if factor != 1:
            chain.free()
    start = time.perf_counter()
    times = [by_chain(run, chain) for _ in range(args.runs)]
    host = round((time.perf_counter() - start) * 1e6 / args.runs, 2)
    checks["after timing"] = bad_words(outputs(session, tensors), expected(x, 1))
    run.stop()
    chain.free()
    return checks, host, round(float(np.median(times)), 2), run.result()["error_name"]


def main():
    args = parse()
    generator = np.random.default_rng(args.seed)
    tiles = args.cores * args.width
    x = generator.integers(-50, 51, (tiles, 1024)).astype(np.float32)
    result = {"cores": args.cores, "width": args.width, "forms": {}}
    with km.Session(log=log) as session:
        tensors = {"x": session.tensor(x, "float32")}
        for name, rows in (("y", tiles), ("z", tiles), ("s", tiles), ("t", 1)):
            zeros = np.zeros((rows, 1024), np.float32)
            tensors[name] = session.tensor(zeros, "float32")
        forms = [] if args.skip_resident else [("resident", Steps, None)]
        for count in args.slots.split(","):
            slots = None if count == "default" else int(count)
            forms.append((f"overlays, slots {count}", OverlaySteps, slots))
        for name, cls, slots in forms:
            program = made(cls, tensors, session=session, log=log)
            if slots is not None:
                program.slots(count=slots)
            checks, host, card, error = chained(session, program, tensors, args, x)
            built = program.overlays_built
            form = {
                "checks": checks,
                "host_us": host,
                "card_us": card,
                "kernel": error,
                "slots": (
                    {part: list(value) for part, value in built.slots.items()}
                    if built is not None and program.overlays
                    else None
                ),
            }
            result["forms"][name] = form
            log(f"{name}: {json.dumps(form)}")
            wrong = any(any(bad.values()) for bad in checks.values())
            if error != "none" or wrong:
                break  # the forms after a wrong one are not run
    result["ok"] = len(result["forms"]) == len(forms) and all(
        form["kernel"] == "none"
        and not any(any(bad.values()) for bad in form["checks"].values())
        for form in result["forms"].values()
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
