"""The SwiGLU class's chain with up and down as overlays (swiglu_class.py's bodies):
the resident chain, then the overlays' chain at each slot count asked (default, a
slot a handler; 1), hidden and y word for word against the resident chain's and
against float64, each chain's time on the card. See README.md."""

import argparse
import json
import sys
import time

import numpy as np
from swiglu_class import SwiGLU, errors, inputs, tensors_of, wanted

import komorebi as km
from komorebi import power

STARTED = time.monotonic()


class OverlaySwiGLU(SwiGLU):
    """SwiGLU with up and down as overlays: the chain's walk loads each record's
    overlay, waits for the record's input whole, then calls it."""

    up = km.handler(overlay=True)(SwiGLU.up.function)
    down = km.handler(overlay=True)(SwiGLU.down.function)


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=50, help="timed chains a form")
    parser.add_argument(
        "--slots",
        default="default,1",
        help="slot counts in turn, comma separated: default (a slot a handler) or 1-4",
    )
    parser.add_argument(
        "--skip-resident", action="store_true", help="no resident chain first"
    )
    parser.add_argument(
        "--deadline", type=float, default=0.5, help="a command's deadline, s"
    )
    return parser.parse_args()


def chained(session, program, t, args):
    """The chain once (hidden and y read after the stop), then loaded again and timed:
    (hidden, y, card us a chain median, host us a chain, the runs' error)."""
    names = ("x", "w1", "w3", "w2", "hidden", "y")
    x, w1, w3, w2, hidden, y = (t[n] for n in names)
    hidden.write(np.zeros((32, 96 * 32), np.float32))
    y.write(np.zeros((32, 32 * 32), np.float32))
    run = session.load(program, wait=args.deadline, deadline=args.deadline)
    chain = run.chain(program.forward(x, w1, w3, w2, hidden, y))
    run.wait(run.submit(chain))
    run.stop()
    errors_seen = [run.result()["error_name"]]
    got = (hidden.read(), y.read())
    run = session.load(program, wait=args.deadline, deadline=args.deadline)
    chain = run.chain(program.forward(x, w1, w3, w2, hidden, y))
    run.wait(run.submit(chain))  # the first command loads the overlays
    start = time.perf_counter()
    times = [run.wait(run.submit(chain)).microseconds for _ in range(args.runs)]
    host = round((time.perf_counter() - start) * 1e6 / args.runs, 2)
    run.stop()
    errors_seen.append(run.result()["error_name"])
    error = next((e for e in errors_seen if e != "none"), "none")
    return got, round(float(np.median(times)), 2), host, error


def main():
    args = parse()
    data = inputs()
    result = {"shape": [32, 1024, 3072], "cores": {"up": 12, "down": 4}, "forms": {}}
    with km.Session(log=log) as session:
        t = tensors_of(session, data, empty=False)
        forms = [] if args.skip_resident else [("resident", SwiGLU, None)]
        for count in args.slots.split(","):
            slots = None if count == "default" else int(count)
            forms.append((f"overlays, slots {count}", OverlaySwiGLU, slots))
        first = None
        for name, cls, slots in forms:
            program = cls(log=log, session=session, **t)
            if slots is not None:
                program.slots(count=slots)
            (got_hidden, got_y), card, host, error = chained(session, program, t, args)
            want_hidden, _, want_y = wanted(data, got_hidden)
            first = first or (got_hidden, got_y)
            built = program.overlays_built
            form = {
                "kernel": error,
                "hidden_words_differ": int(np.sum(got_hidden != first[0])),
                "y_words_differ": int(np.sum(got_y != first[1])),
                "hidden": errors(got_hidden.astype(np.float64), want_hidden),
                "y": errors(got_y.astype(np.float64), want_y),
                "card_us": card,
                "host_us": host,
                "slots": (
                    {part: list(value) for part, value in built.slots.items()}
                    if built is not None and program.overlays
                    else None
                ),
            }
            result["forms"][name] = form
            log(f"{name}: {json.dumps(form)}")
            if error != "none" or form["y_words_differ"] or form["hidden_words_differ"]:
                break  # the forms after a wrong one are not run
    result["ok"] = len(result["forms"]) == len(forms) and all(
        form["kernel"] == "none"
        and not form["y_words_differ"]
        and not form["hidden_words_differ"]
        and form["y"]["rms_rel"] < 1e-2
        for form in result["forms"].values()
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
