"""One trace pass on the card from prepare.py's inputs: the hand program (the project's
run_handler: LOAD, then the TRACE commands) or the class (load_tree once, then the
trace_rays commands); hit pages, profile pages, times to build/<program>.npz.
Example: python run_trace.py --program class --reps 5"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(HERE)]

from trace_class import TracePass, made  # noqa: E402

from projects.raytrace import card, packets  # noqa: E402

K = packets.K


def hand(session, inputs, reps):
    """The project's own driver of the pass: (hit rows, profile rows, Ran, times)."""
    rows = len(inputs["packets"])
    words = K["kPagePackets"] * packets.hit_vectors(False) * packets.VEC
    output, profile, ran, _, _ = card.run_handler(
        session,
        K["kHandlerTrace"],
        inputs["directory"],
        inputs["packets"],
        rows,
        words,
        tree=inputs["image"],
        args=[K["kModeNearest"]],
        repetitions=reps,
    )
    return output, profile, ran, card.command_microseconds(ran)[1:]


def of_class(session, inputs, reps):
    """The class: the same tensors, one load_tree, `reps` commands of trace_rays."""
    rows = len(inputs["packets"])
    words = K["kPagePackets"] * packets.hit_vectors(False) * packets.VEC
    workers = card.workers(session)
    tensors = {
        "directory": card.tensor(session, inputs["directory"]),
        "packets": card.tensor(session, inputs["packets"]),
        "hits": card.tensor(session, np.zeros((rows, words), np.uint32)),
        "profile": card.tensor(
            session, np.zeros((workers, K["kOutPageBytes"] // 4), np.uint32)
        ),
        "image": card.tensor(session, inputs["image"]),
    }
    program = made(session, len(inputs["image"]), tensors, card.log)
    args = tuple(tensors[name] for name in ("directory", "packets", "hits"))
    args += (K["kModeNearest"], tensors["profile"])
    run = session.load(program, deadline=5.0)
    try:
        handles = [
            run.submit(TracePass.load_tree, (0, workers), args=(tensors["image"],))
        ]
        handles += [
            run.submit(TracePass.trace_rays, (0, workers), args=args)
            for _ in range(reps)
        ]
        events = run.wait(handles, timeout=10.0 + 5.0 * reps)
    finally:
        run.stop()
    events = events if isinstance(events, list) else [events]
    ran = card.Ran(events, run.result(), run.state(), 0.0)
    output = card.read_rows(session, tensors["hits"], "hits")
    (profile,) = card.read(session, tensors["profile"])
    microseconds = card.command_microseconds(ran)
    print(json.dumps({"load_tree_microseconds": microseconds[:1].tolist()}))
    return output, profile, ran, microseconds[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--program", choices=("hand", "class"), required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--inputs", default=str(HERE / "build" / "inputs.npz"))
    arguments = parser.parse_args()
    inputs = dict(np.load(arguments.inputs))
    with card.open_session() as session:
        run = hand if arguments.program == "hand" else of_class
        output, profile, ran, microseconds = run(session, inputs, arguments.reps)
        workers = card.workers(session)
        mhz = card.aiclk_mhz()
    report = {
        "program": arguments.program,
        "run": card.check_run(ran, workers),
        "trace_microseconds": [float(value) for value in microseconds],
        "median_microseconds": float(np.median(microseconds)),
        "mhz": mhz,
    }
    report["ok"] = card.ran_cleanly(report["run"], workers)
    np.savez(
        HERE / "build" / f"{arguments.program}.npz",
        hits=output,
        profile=profile,
        microseconds=microseconds,
    )
    print(json.dumps(report, default=int), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
