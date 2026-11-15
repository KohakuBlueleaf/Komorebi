"""Work cluster written as a class: STEP and SUM of ../kernels as two handlers of a
class, an X280 thread submitting them by the class's generated C header (thread/
driver.c with ../decide.c), checked against ../work_cluster.py's numpy replay;
beside its own thread path with --beside. See README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import work_cluster as hand  # noqa: E402  (the hand program, its replay and paths)


class WorkCluster(km.Program):
    """step: each core of the group adds `round` and its page to its word. sum: the
    leader adds the words of `cores` pages from `first`, the command's result."""

    @km.handler
    def step(self, state: km.Tensor, round: int, first: int):
        page = first + self.rank
        word = state.pages(page, 1).read()
        word += round + page
        state.pages(page, 1).write(word)

    @km.handler
    def sum(self, state: km.Tensor, cores: int, first: int):
        if self.rank == self.leader:
            total = state.pages(first, 1).read()
            for page in range(first + 1, first + cores):
                word = state.pages(page, 1).read()
                total += word
            return total


def made(state, words):
    """The class made with the state tensor, its thread on cluster 1 with `words`."""
    program = WorkCluster(state=state, log=hand.log)
    program.thread(
        cluster=1,
        sources=[HERE / "thread" / "driver.c", hand.HERE / "decide.c"],
        includes=[hand.HERE, program.built.folder],
        regions={"groups": 4 * words[4]},
        words=words,
        deadline=10.0,
    )
    return program


def run_class(args, words, want):
    """The class's thread on the card: its result words and groups, the state, the
    loop's time on the thread, checked against the replay."""
    with km.Session(log=hand.log) as session:
        state = hand.state_tensor(session, args.cores)
        words = [state.address, *words[1:]]
        program = made(state, words)
        run = session.load(program, deadline=1.0, thread="card")
        results = run.thread_results(timeout=15.0)
        run.stop()
        result = results[1]
        groups = run.threads[1].read("groups").view(np.uint32)
        final = hand.read_state(state, args.cores)
    rounds = result.words[0] if result.words else 0
    got = [result.code, *(list(result.words) + [0, 0, 0])[:3]]
    got.append([int(first) for first in groups[:rounds]])
    return {
        "code": result.code,
        "rounds": rounds,
        "sum": got[2],
        "same_as_numpy": got == want[0],
        "state_wrong": int((final != want[1]).sum()),
        "ms": round(result.microseconds / 1e3, 2),
        "kernel_text": program.sizes.text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--beside", action="store_true", help="the hand thread first")
    parser.add_argument("--cores", type=int, default=119, help="the grid's cores used")
    parser.add_argument("--count", type=int, default=30, help="cores of each STEP")
    parser.add_argument("--limit", type=int, default=1_000_000, help="the ending sum")
    parser.add_argument("--max-rounds", type=int, default=1000)
    args = parser.parse_args()
    words = [0, args.cores, args.count, args.limit, args.max_rounds, 0]
    code, rounds, total, trail, groups, state = hand.replay(*words[1:5], base=0)
    want = ([code, rounds, total, trail, groups], state)
    report = {"replay": {"code": code, "rounds": rounds, "sum": total}, "runs": {}}
    if args.beside:
        got, milliseconds = hand.card_paths(args, {1: words}, ["thread"])
        loop = got["thread"][1]
        report["runs"]["hand"] = {
            "code": loop[0],
            "rounds": loop[1],
            "sum": loop[2],
            "same_as_numpy": list(loop[:5]) == want[0],
            "state_wrong": int((loop[6] != want[1]).sum()),
            "ms": milliseconds["thread"][1],
        }
    report["runs"]["class"] = run_class(args, words, want)
    report["ok"] = all(
        run["same_as_numpy"] and run["state_wrong"] == 0
        for run in report["runs"].values()
    )
    report["wall_s"] = round(time.monotonic() - hand.STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
