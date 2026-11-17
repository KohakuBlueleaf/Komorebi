"""Two-card sessions in turn in one process, each closed one and its tensor kept alive:
each loads a class, runs one command over both islands and checks its pages. --fail N
raises inside session N before the next opens. One JSON line; exits 1 when a check
fails. See README.md."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--fail", type=int, default=0, help="session raising (1 on)")
    return parser.parse_args()


class Put(km.Program):
    """put: rank r of the group adds `value` to word 0 of page r."""

    @km.handler
    def put(self, out: km.Tensor, value: int):
        word = out.pages(self.rank, 1).read()
        word += value
        out.pages(self.rank, 1).write(word)


class Planned(Exception):
    """The exception --fail raises inside a session."""


def one(cards, number, fail, kept):
    """Session `number`: its pages right (count of wrong pages) and its stop."""
    with km.Session(cards=cards, log=log) as session:
        ranks = session.ranks
        out = session.tensor(np.zeros((ranks, 16), np.uint32), "uint32")
        kept.append((session, out))
        run = session.load(Put(out=out, log=log), deadline=2.0)
        run.wait(run.submit(Put.put, (0, ranks), args=(out, number)), timeout=10.0)
        if number == fail:
            raise Planned(f"session {number}: raised inside the with block")
        stopped = run.stop()
        got = out.read().astype(np.int64)
    wrong = int((got[:, 0] != number).sum() + got[:, 1:].any(axis=1).sum())
    errors = [block["error_name"] for block in stopped.islands]
    return {"session": number, "ranks": ranks, "wrong": wrong, "kernels": errors}


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report, kept = {"cards": cards, "sessions": []}, []
    for number in range(1, args.sessions + 1):
        try:
            report["sessions"].append(one(cards, number, args.fail, kept))
        except Planned as error:
            report["sessions"].append({"session": number, "raised": str(error)})
        log(f"session {number}: {report['sessions'][-1]}")
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = len(report["sessions"]) == args.sessions and all(
        "raised" in s or (s["wrong"] == 0 and s["kernels"] == ["none"] * len(cards))
        for s in report["sessions"]
    )
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
