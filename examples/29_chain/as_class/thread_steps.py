"""Example 29's chain run by the class's X280 thread, its records written from the
class's header (km.x280_thread(chains=...)), then by the host's run.chain with another
factor; every word against numpy. --merged: the class merged with another, the chain
on cores 8 to 15. See ../README.md."""

import argparse
import json
import sys
import time

import numpy as np
from steps_class import WORDS, Steps, expected, log, outputs

import komorebi as km
from komorebi import power

STARTED = time.monotonic()
CORES, WIDTH = 8, 2  # the chain's cores and each core's tiles, fixed in the class
TILES = CORES * WIDTH


class ThreadSteps(Steps):
    """Steps whose thread runs the chain STEPS: x is the make's a, y its out, z a
    tensor of the thread's chains (thread_tensors=), the factor 7 (a float word is
    no km.at_submit word: those are ints)."""

    records = km.region(4 * 64)  # the chain's records, 64 bytes a call
    thread = km.x280_thread(
        "thread/steps.c",
        cluster=1,
        deadline=10.0,
        chains={
            "steps": {
                "x": "a",
                "y": "out",
                "factor": 7.0,
                "width": WIDTH,
                "tiles": TILES,
                "cores": (0, CORES),
            }
        },
    )


class OffsetSteps(Steps):
    """Steps whose thread runs the chain STEPS on cores 8 to 15, merged below."""

    records = km.region(4 * 64)
    thread = km.x280_thread(
        "thread/merged_steps.c",
        cluster=1,
        deadline=10.0,
        chains={
            "steps": {
                "x": "a",
                "y": "out",
                "factor": 7.0,
                "width": WIDTH,
                "tiles": TILES,
                "cores": (CORES, CORES),
            }
        },
    )


class Touch(km.Program):
    """A second class of the merge: one handler, never called here."""

    @km.handler
    def touch(self, a: km.Tensor):
        km.Stream(a.pages(0, 1)).drain()


class MergedSteps(OffsetSteps, Touch):
    """The two merged: the thread OffsetSteps declares, made with the merged numbers."""


def differences(got, want):
    """Each output's wrong words: their count and the first eight (page, word)."""
    out = {}
    for name in want:
        wrong = np.argwhere(got[name] != want[name])
        out[name] = {
            "wrong": int(len(wrong)),
            "first": [[int(p), int(w)] for p, w in wrong[:8]],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dump", default=None, help="an .npz of the outputs")
    parser.add_argument("--merged", action="store_true", help="MergedSteps, cores 8+")
    args = parser.parse_args()
    generator = np.random.default_rng(args.seed)
    x = generator.integers(-50, 51, (TILES, WORDS)).astype(np.float32)
    cls = MergedSteps if args.merged else ThreadSteps
    cores = (CORES, CORES) if args.merged else (0, CORES)
    result = {"class": cls.__name__, "cores": cores, "width": WIDTH, "tiles": TILES}
    dumped = {"x": x}
    with km.Session(log=log) as session:
        tensors = {"x": session.tensor(x, "float32")}
        for name, rows in (("y", TILES), ("z", TILES), ("s", TILES), ("t", 1)):
            zeros = np.zeros((rows, WORDS), np.float32)
            tensors[name] = session.tensor(zeros, "float32")
        x_, y, z, s, t = (tensors[name] for name in "xyzst")
        program = cls(
            a=x_, b=x_, out=y, s=s, t=t, thread_tensors={"z": z},
            session=session, log=log,
        )  # fmt: skip
        run = session.load(program, wait=0.5, deadline=0.5, thread="card")
        thread = run.thread_results(timeout=15.0)[1]
        result["thread"] = {
            "state": thread.state,
            "error": thread.error,
            "code": thread.code,
            "words": [int(word) for word in thread.words],
            "microseconds": round(thread.microseconds, 2),
        }
        got = outputs(session, tensors)
        dumped.update({f"thread_{name}": got[name] for name in got})
        result["checks"] = {"thread x7": differences(got, expected(x, 7))}
        chain = run.chain(
            program.steps(x_, y, z, s, t, 5.0, WIDTH, TILES),
            cores=cores,
        )
        handle = run.submit(chain)
        result["host_microseconds"] = round(run.wait(handle).microseconds, 2)
        got = outputs(session, tensors)
        dumped.update({f"host_{name}": got[name] for name in got})
        result["checks"]["host x5"] = differences(got, expected(x, 5))
        run.stop()
        chain.free()
        result["kernel"] = run.result()["error_name"]
    if args.dump:
        np.savez_compressed(args.dump, **dumped)
    result["ok"] = bool(
        result["kernel"] == "none"
        and result["thread"]["code"] == 0
        and all(
            one["wrong"] == 0
            for check in result["checks"].values()
            for one in check.values()
        )
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
