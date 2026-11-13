"""The same call submitted three ways, rounds back to back: one ATTEND command walking
the job table and one command per job index, each from the host and from a kept graph
that the dispatcher's engine runs (it decides each next command). Then the check. Prints
one JSON line; exits 1 when the check fails. Usage in README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi.engines.graph import Graph  # noqa: E402
from projects.attention import card, run  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

log = run.log


def calls(p, att, per_job):
    """The command words of one call: one command, or one per job index."""
    if not per_job:
        return [att.words()]
    jobs = max(len(j) for j in p.jobs)
    return [att.words()[:7] + [j, 1] for j in range(jobs)]


def graph(handler, group, words, rounds, req):
    """A kept graph: the call's commands one after the other, `rounds` times."""
    g = Graph(deadline_microseconds=2_000_000, max_runs=rounds, req=req, keep=True)
    it = g.register()
    last = []
    for w in words:
        last = [g.command(handler, group=group, args=w, after=last, exclusive=True)]
    step = g.cpu(after=last)
    step.addi(it, it, 1)
    step.lt(g.again, it, g.const(rounds))
    return g


def from_host(att, words, rounds):
    """Microseconds per call, the call's commands submitted from the host back to
    back."""
    rows = np.array(words * rounds, np.int64)
    t0 = time.perf_counter()
    handles = att.run.submit_many("attend", 0, att.group[1], rows, exclusive=True)
    events = att.run.wait(list(handles), timeout=60.0)
    if any(e.kind != "done" for e in events):
        raise km.KomorebiError("a command not done")
    return (time.perf_counter() - t0) * 1e6 / rounds


def from_graph(att, g, times):
    """Microseconds per call of a kept graph's rounds: uploaded and started once, then
    restarted `times` times; the median of the restarts."""
    up = att.run.upload(g)
    att.run.wait_engine(att.run.start(up, kept=True), timeout=60.0)
    us = []
    for _ in range(times):
        t0 = time.perf_counter()
        att.run.wait_engine(att.run.restart(up.req), timeout=60.0)
        us.append((time.perf_counter() - t0) * 1e6 / g.max_runs)
    att.run.release(up)
    return float(np.median(us))


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--shape", choices=pl.SHAPES, required=True)
    a.add_argument("--rounds", type=int, default=20, help="calls per run")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--json", default=None)
    args = a.parse_args()
    (b, h, hk, sq, d, causal), s = run.shape_dims(args.shape)
    q, k, v = sdpa.inputs(b, h, hk, sq, sq, d, d, args.seed)
    out = {"shape": args.shape, "rounds": args.rounds}
    with km.Session(log=log) as session:
        p = pl.make(s, session.nocs().grid)
        out["plan"], out["x280_mhz"] = p.summary(), session.pll_mhz
        tensors = {
            "q": card.bf16_tensor(session, q),
            "k": card.bf16_tensor(session, k),
            "v": card.bf16_tensor(session, v),
            "o": card.bf16_tensor(session, np.zeros((b, h, sq, d), np.float32)),
            "tables": card.table_tensor(session, pl.tables(p)),
        }
        att = card.Attention(session, p, tensors, 1.0 / np.sqrt(d))
        handler = att.run.program.handler("attend")
        group = (0, att.group[1])
        att.calls(2)
        us = {}
        for name, per_job in (("table", False), ("jobs", True)):
            words = calls(p, att, per_job)
            us[f"host_{name}"] = from_host(att, words, args.rounds)
            g = graph(handler, group, words, args.rounds, req=10 + per_job)
            us[f"graph_{name}"] = from_graph(att, g, 3)
            out[f"commands_per_call_{name}"] = len(words)
            log(f"{name}: {us}")
        out["microseconds_per_call"] = {n: round(v, 1) for n, v in us.items()}
        out["run"] = att.stop()
        out["check"] = run.check(session, tensors["o"], q, k, v, s, causal)
    out["ok"] = out["check"]["ok"] and out["run"]["ok"]
    print(json.dumps(out, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, default=float))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
