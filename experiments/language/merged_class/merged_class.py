"""Example 22's gate class and the dense experiment's Linear merged into ONE program
(km.merge): each handler run on its cores, checked against the host, its time beside
its own program's. README.md.
Example: python merged_class.py --beside"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "examples" / "22_gate" / "as_class"))
sys.path.insert(0, str(HERE.parent / "dense_class"))

import gate_class  # noqa: E402  (its module adds ../gate.py, the hand program)
import linear_class  # noqa: E402

GATE_TILES, GATE_CORES = 64, 12  # example 22's command: 12 cores, 64 tiles each
# The kernel buffer: the merged kernels take 98,656 B (Linear's alone 92,304 B, over
# its own script's 89 KiB); 97 KiB holds them, and the buffers still fit the L1 left.
KERNEL_BYTES = 97 * 1024
log = linear_class.log


def gate_made(session, args):
    """Gate made with bfloat16 tensors of example 22's shape, its inputs written;
    (the program, its tensors, the float64 reference)."""
    pages = GATE_CORES * GATE_TILES
    rng = np.random.default_rng(args.seed)
    a_raw, a_values = gate_class.values(rng, pages, "bfloat16")
    b_raw, b_values = gate_class.values(rng, pages, "bfloat16")
    a, b, out = (session.empty((pages, 1024), "bfloat16") for _ in range(3))
    for tensor, raw in ((a, a_raw), (b, b_raw)):
        session.write(tensor, raw.view(np.uint8).reshape(pages, -1))
    program = gate_class.Gate(a=a, b=b, out=out, BLOCK=args.gate_block, log=log)
    wanted = gate_class.hand.reference(a_values, b_values)
    return program, (a, b, out), wanted


def gate_run(session, run, handler, tensors, wanted, args):
    """Example 22's command: Gate's handler on 12 cores, its output against float64."""
    a, b, out = tensors
    session.write(out, np.zeros((GATE_CORES * GATE_TILES, 2048), np.uint8))
    words = (a, b, out, GATE_TILES, km.Pattern(args.pattern))
    times = []
    for _ in range(args.repeat):
        event = run.wait(run.submit(handler, (0, GATE_CORES), args=words), 30.0)
        if event.kind != "done":
            raise RuntimeError(f"gate: {event}")
        times.append(event.microseconds)
    record = gate_class.compared(session, out, wanted, "bfloat16")
    record["microseconds"] = times
    record["median_microseconds"] = float(np.median(times))
    return record


def linear_run(session, run, program, handler, args):
    """The K split (one row tile) as linear_class runs it: the exact check, then the
    timed random values and the SwiGLU gate (Linear's own gate handler). Returns what
    reads the outputs, called after the run stops (a tensor's read waits for every
    core to be idle, which a running grid never is)."""
    plan = (handler, (0, program.SPLIT_CORES))
    shape = argparse.Namespace(
        **{**vars(args), "rows": 1, "gate_cores": 64, "hints": {}}
    )
    exact = linear_class.check_exact(session, plan, shape)(run)
    swiglu = linear_class.check_random(session, plan, shape, True)(run)
    return lambda: {"exact": exact(), "swiglu": swiglu()}


def alone(session, program, body):
    """`body(run)` on a load of `program` alone; a callable it returns is called
    after the run stops."""
    run = session.load(program)
    try:
        got = body(run)
    finally:
        run.stop()
    return got() if callable(got) else got


def logged(report, name):
    """A record logged as soon as it is made."""
    log(f"RECORD {name} {json.dumps(report[name], default=float)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--beside", action="store_true", help="each program alone too")
    parser.add_argument("--repeat", type=int, default=3, help="timed commands")
    parser.add_argument("--pattern", default="AAAB")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--gate-block", type=int, default=2, help="Gate's BLOCK (8: over L1, merged)"
    )
    parser.add_argument(
        "--overlay", nargs="*", default=None, help="merged handler names to mark"
    )
    args = parser.parse_args()
    report = {}
    with km.Session(log=log, kernel_bytes=KERNEL_BYTES) as session:
        gate, tensors, wanted = gate_made(session, args)
        linear = linear_class.made(session, [1], linear_class.Linear.K_TILES.default)
        # The program of most L1 first: a load after one whose host transfer buffer
        # was larger clashed with tt-metal's check of the buffers' region (README.md).
        if args.beside:
            report["linear alone"] = alone(
                session,
                linear,
                lambda run: linear_run(session, run, linear, linear.split, args),
            )
            logged(report, "linear alone")
        merged = km.merge(
            gate, linear, session=session, overlay=args.overlay or (), log=log
        )
        run = session.load(merged)
        try:
            report["gate merged"] = gate_run(
                session, run, gate.gate, tensors, wanted, args
            )
            logged(report, "gate merged")
            finish = linear_run(session, run, linear, linear.split, args)
        finally:
            run.stop()
        report["linear merged"] = finish()
        logged(report, "linear merged")
        report["kernel_text"] = merged.sizes.text
        report["overlays"] = sorted(merged.overlays)
        if args.beside:
            report["gate alone"] = alone(
                session,
                gate,
                lambda run: gate_run(session, run, gate.gate, tensors, wanted, args),
            )
    ok = all(
        record["wrong"] == 0
        for name, record in report.items()
        if name.startswith("gate ")
    ) and all(
        record["exact"]["exact"] and record["swiglu"]["relative_rms"] < 1e-2
        for name, record in report.items()
        if name.startswith("linear ")
    )
    report["ok"] = bool(ok)
    print(json.dumps(report, default=float), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
