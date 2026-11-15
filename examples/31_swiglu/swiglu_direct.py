"""The SwiGLU class's chain with the hidden sent direct: up's roots write their blocks
into down's cores' L1 (km.direct) in place of DRAM. One load, the DRAM chain then the
direct chain, y word for word against the DRAM chain's and float64, each chain's time,
and a host read of the hidden after the direct chain refused. See README.md."""

import argparse
import json
import sys
import time

import numpy as np
from swiglu_class import SwiGLU, errors, inputs, tensors_of, wanted

import komorebi as km

STARTED = time.monotonic()


class DirectSwiGLU(SwiGLU):
    """SwiGLU whose down may take the hidden direct, and the chain that sends it."""

    down = km.handler(direct=("hidden",))(SwiGLU.down.function)

    @km.chain
    def forward_direct(self, x, w1, w3, w2, hidden, y):
        self.up(x, w1, w3, hidden, cores=(0, 12))
        self.down(km.direct(hidden), w2, y, cores=(0, 4))  # up's blocks into L1


def aiclk():
    """The card's AI clock in MHz as the driver reports it (None where it is not)."""
    try:
        with open("/sys/class/tenstorrent/tenstorrent!0/tt_aiclk") as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=50, help="timed chains a form")
    parser.add_argument(
        "--deadline", type=float, default=0.5, help="a command's deadline, s"
    )
    return parser.parse_args()


def once(session, program, t, form, args):
    """A load, the chain `form` once, the stop: (y, the run's error, the hidden's host
    read refused or None, the chain's report)."""
    names = ("x", "w1", "w3", "w2", "hidden", "y")
    values = [t[n] for n in names]
    t["hidden"].write(np.zeros((32, 96 * 32), np.float32))
    t["y"].write(np.zeros((32, 32 * 32), np.float32))
    run = session.load(program, wait=args.deadline, deadline=args.deadline)
    method = program.forward if form == "dram" else program.forward_direct
    chain = run.chain(method(*values))
    try:
        event = run.wait(run.submit(chain))
    finally:
        run.stop()
    if event.kind != "done":
        raise RuntimeError(f"{form} chain: {event}")
    try:
        t["hidden"].read()
        refused = None
    except RuntimeError as refusal:
        refused = str(refusal)
    return t["y"].read(), run.result()["error_name"], refused, chain.report


def timed(session, program, t, args):
    """A load, both chains in turn `runs` times each: their card us, median."""
    names = ("x", "w1", "w3", "w2", "hidden", "y")
    values = [t[n] for n in names]
    run = session.load(program, wait=args.deadline, deadline=args.deadline)
    chains = {
        "dram": run.chain(program.forward(*values)),
        "direct": run.chain(program.forward_direct(*values)),
    }
    times = {form: [] for form in chains}
    try:
        for _ in range(1 + args.runs):  # the first of each not timed
            for form, chain in chains.items():
                event = run.wait(run.submit(chain))
                if event.kind != "done":
                    raise RuntimeError(f"{form} chain: {event}")
                times[form].append(event.microseconds)
    finally:
        run.stop()
    error = run.result()["error_name"]
    return {f: round(float(np.median(v[1:])), 2) for f, v in times.items()}, error


def main():
    args = parse()
    data = inputs()
    result = {"shape": [32, 1024, 3072], "cores": {"up": 12, "down": 4}}
    result["aiclk_mhz_before"] = aiclk()
    with km.Session(log=log) as session:
        t = tensors_of(session, data, empty=False)
        program = DirectSwiGLU(log=log, session=session, **t)
        y_dram, error_dram, _, _ = once(session, program, t, "dram", args)
        hidden_dram = t["hidden"].read()
        y_direct, error_direct, refused, report = once(
            session, program, t, "direct", args
        )
        kernels = [error_dram, error_direct]
        times = None
        if all(e == "none" for e in kernels) and np.array_equal(y_dram, y_direct):
            times, error_timed = timed(session, program, t, args)
            kernels.append(error_timed)
        result["aiclk_mhz_after"] = aiclk()
    _, _, want_y = wanted(data, hidden_dram)
    result.update(
        {
            "kernel": kernels,
            "y_words_differ": int(np.sum(y_dram != y_direct)),
            "y_dram": errors(y_dram.astype(np.float64), want_y),
            "y_direct": errors(y_direct.astype(np.float64), want_y),
            "hidden_read_refused": refused,
            "direct_report": {str(k): [list(r) for r in v] for k, v in report.items()},
            "card_us": times,
        }
    )
    result["ok"] = bool(
        all(e == "none" for e in kernels)
        and result["y_words_differ"] == 0
        and refused is not None
        and times is not None
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
