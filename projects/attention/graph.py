"""The same call issued three ways, rounds back to back: one ATTEND record walking the
job table and one record per job index, each from the host and from a kept graph that
the dispatch kernel's engine runs (it decides each next record). Then the check. Prints
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
    """The record words of one call: one record, or one per job index."""
    if not per_job:
        return [att.words()]
    jobs = max(len(j) for j in p.jobs)
    return [att.words()[:7] + [j, 1] for j in range(jobs)]


def graph(seg, team, words, rounds, req):
    """A kept graph: the call's records one after the other, `rounds` times."""
    g = Graph(deadline_us=2_000_000, max_runs=rounds, req=req, keep=True)
    it = g.reg()
    last = []
    for w in words:
        last = [g.record(seg, team=team, args=w, after=last, exclusive=True)]
    step = g.cpu(after=last)
    step.addi(it, it, 1)
    step.lt(g.again, it, g.const(rounds))
    return g


def from_host(att, words, rounds):
    """Microseconds per call, the call's records issued from the host back to back."""
    rows = np.array(words * rounds, np.int64)
    t0 = time.perf_counter()
    tickets = att.run.issue_many("attend", 0, att.team[1], rows, exclusive=True)
    events = att.run.wait(list(tickets), timeout=60.0)
    if any(e.kind != "done" for e in events):
        raise km.KomorebiError("a record not done")
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
        device = session.device
        tensors = {
            "q": card.bf16_tensor(device, q),
            "k": card.bf16_tensor(device, k),
            "v": card.bf16_tensor(device, v),
            "o": card.bf16_tensor(device, np.zeros((b, h, sq, d), np.float32)),
            "tables": card.table_tensor(device, pl.tables(p)),
        }
        att = card.Attention(session, p, tensors, 1.0 / np.sqrt(d))
        seg = att.run.program.seg("attend")
        team = (0, att.team[1])
        att.calls(2)
        us = {}
        for name, per_job in (("table", False), ("jobs", True)):
            words = calls(p, att, per_job)
            us[f"host_{name}"] = from_host(att, words, args.rounds)
            g = graph(seg, team, words, args.rounds, req=10 + per_job)
            us[f"graph_{name}"] = from_graph(att, g, 3)
            out[f"records_per_call_{name}"] = len(words)
            log(f"{name}: {us}")
        out["us_per_call"] = {n: round(v, 1) for n, v in us.items()}
        out["run"] = att.stop()
        out["check"] = run.check(session, tensors["o"], q, k, v, s, causal)
    out["ok"] = out["check"]["ok"] and out["run"]["ok"]
    print(json.dumps(out, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, default=float))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
