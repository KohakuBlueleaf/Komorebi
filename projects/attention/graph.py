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
from komorebi.bench import power  # noqa: E402
from komorebi.engines.graph import Graph  # noqa: E402
from projects.attention import card, run  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

log = run.log


def calls(plan, attention, per_job):
    """The command words of one call: one command, or one per job index."""
    if not per_job:
        return [attention.words()]
    jobs = max(len(core_jobs) for core_jobs in plan.jobs)
    return [attention.words()[:7] + [job_index, 1] for job_index in range(jobs)]


def graph(handler, group, words, rounds, request):
    """A kept graph: the call's commands one after the other, `rounds` times."""
    kept_graph = Graph(
        deadline_microseconds=2_000_000, max_runs=rounds, req=request, keep=True
    )
    iteration = kept_graph.register()
    last = []
    for command_words in words:
        last = [
            kept_graph.command(
                handler, group=group, args=command_words, after=last, exclusive=True
            )
        ]
    step = kept_graph.cpu(after=last)
    step.addi(iteration, iteration, 1)
    step.lt(kept_graph.again, iteration, kept_graph.const(rounds))
    return kept_graph


def from_host(attention, words, rounds):
    """Microseconds per call, the call's commands submitted from the host back to
    back."""
    rows = np.array(words * rounds, np.int64)
    started = time.perf_counter()
    handles = attention.run.submit_many(
        "attend", 0, attention.group[1], rows, exclusive=True
    )
    events = attention.run.wait(list(handles), timeout=60.0)
    if any(event.kind != "done" for event in events):
        raise km.KomorebiError("a command not done")
    return (time.perf_counter() - started) * 1e6 / rounds


def from_graph(attention, kept_graph, times):
    """Microseconds per call of a kept graph's rounds: uploaded and started once, then
    restarted `times` times; the median of the restarts."""
    upload = attention.run.upload(kept_graph)
    attention.run.wait_engine(attention.run.start(upload, kept=True), timeout=60.0)
    microseconds = []
    for _ in range(times):
        started = time.perf_counter()
        attention.run.wait_engine(attention.run.restart(upload.req), timeout=60.0)
        microseconds.append((time.perf_counter() - started) * 1e6 / kept_graph.max_runs)
    attention.run.release(upload)
    return float(np.median(microseconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=pl.SHAPES, required=True)
    parser.add_argument("--rounds", type=int, default=20, help="calls per run")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    (batches, heads, kv_heads, sequence_length, head_dim, causal), call = (
        run.shape_dims(arguments.shape)
    )
    queries, keys, values = sdpa.inputs(
        batches,
        heads,
        kv_heads,
        sequence_length,
        sequence_length,
        head_dim,
        head_dim,
        arguments.seed,
    )
    report = {"shape": arguments.shape, "rounds": arguments.rounds}
    with km.Session(log=log) as session:
        plan = pl.make(call, session.nocs().grid)
        report["plan"], report["x280_mhz"] = plan.summary(), session.pll_mhz
        tensors = {
            "q": card.bf16_tensor(session, queries),
            "k": card.bf16_tensor(session, keys),
            "v": card.bf16_tensor(session, values),
            "o": card.bf16_tensor(
                session,
                np.zeros((batches, heads, sequence_length, head_dim), np.float32),
            ),
            "tables": card.table_tensor(session, pl.tables(plan)),
        }
        attention = card.Attention(session, plan, tensors, 1.0 / np.sqrt(head_dim))
        handler = attention.run.program.handler("attend")
        group = (0, attention.group[1])
        attention.calls(2)
        microseconds = {}
        for name, per_job in (("table", False), ("jobs", True)):
            words = calls(plan, attention, per_job)
            microseconds[f"host_{name}"] = from_host(attention, words, arguments.rounds)
            kept_graph = graph(
                handler, group, words, arguments.rounds, request=10 + per_job
            )
            microseconds[f"graph_{name}"] = from_graph(attention, kept_graph, 3)
            report[f"commands_per_call_{name}"] = len(words)
            log(f"{name}: {microseconds}")
        report["microseconds_per_call"] = {
            name: round(value, 1) for name, value in microseconds.items()
        }
        report["run"] = attention.stop()
        report["check"] = run.check(
            session, tensors["o"], queries, keys, values, call, causal
        )
    report["ok"] = report["check"]["ok"] and report["run"]["ok"]
    print(json.dumps(power.clocked(report), default=float), flush=True)
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(report, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
