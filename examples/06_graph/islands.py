"""Example 06's graph over islands, run by island 0 alone: STEP and NORM on every rank,
LEFT on island 0's half, RIGHT on the other half (island 1's ranks with two islands of
equal ranks). Host-fed through island 0's run, then by island 0's dispatcher's engine;
each case's rounds, registers and every rank's state as the numpy replay. One JSON line.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as g

import komorebi as km
from komorebi.bench import power


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument(
        "--first", nargs=2, type=int, default=[0, 10_000_000], help="iteration, limit"
    )
    parser.add_argument(
        "--again", nargs=2, type=int, default=[3, 100_000_000], help="iteration, limit"
    )
    parser.add_argument("--max-runs", type=int, default=50)
    parser.add_argument("--deadline", type=float, default=1.0, help="per command, s")
    return parser.parse_args()


def state_tensor(session, ranks):
    rows = np.zeros((ranks, g.WORDS), np.uint32)
    rows[:, 0] = g.initial(ranks)
    return session.tensor(rows, "uint32")


def host_case(first, graph, registers, bindings):
    got = first.run_hostfed(graph, bindings, 30.0)
    return {
        "status": got.status,
        "rounds": got.runs,
        "it": got.registers[registers[0].index],
        "err": got.registers[registers[1].index],
    }


def x280_cases(first, graph, cases, states):
    upload, path = first.upload(graph), {}
    for index, (name, (iteration, limit)) in enumerate(cases.items()):
        bindings = {"state": states[name].address, "iteration": iteration}
        bindings["limit"] = limit
        handle = (
            first.start(upload, bindings, kept=True)
            if index == 0
            else first.restart(upload.req, bindings)
        )
        end = first.wait_engine(handle, timeout=30.0)
        rounds = 1 + sum(event.kind == "round" for event in handle.events)
        path[name] = {"status": "ended", "rounds": rounds}
        path[name]["it"], path[name]["err"] = end.registers
    first.release(upload)
    return path


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    cases = {"first": args.first, "again": args.again}
    report, ok = {"cards": cards}, True
    with km.Session(cards=cards, log=g.log) as session:
        ranks, graph_of = session.ranks, session.graph
        own = [
            (graph_of.first(i), graph_of.count(i)) for i in range(len(graph_of.islands))
        ]
        g.log(f"open: {ranks} ranks, islands {own}")
        states = {
            (mode, name): state_tensor(session, ranks)
            for mode in ("host", "x280")
            for name in cases
        }
        accessor = next(iter(states.values())).tensor_args()
        program = km.Program(
            "graph",
            kernels=g.KERNELS,
            buffers=[km.Buffer(index=16, format="uint32", page_bytes=64, pages=1)],
            compile_args={"writer": accessor},
            handlers=g.KERNELS / "graph.h",
        )
        graph, registers = g.build(ranks, args.max_runs, args.deadline)
        half = ranks // 2
        report |= {"ranks": ranks, "own": own, "left": [0, half]}
        report["right"] = [half, ranks - half]
        run = session.load(program, deadline=args.deadline)
        first = run.island(0)
        report["paths"] = {"host": {}}
        for name, (iteration, limit) in cases.items():
            bindings = {"state": states["host", name].address, "iteration": iteration}
            bindings["limit"] = limit
            report["paths"]["host"][name] = host_case(first, graph, registers, bindings)
            g.log(f"host {name}: {report['paths']['host'][name]}")
        report["paths"]["x280"] = x280_cases(
            first, graph, cases, {name: states["x280", name] for name in cases}
        )
        g.log(f"x280: {report['paths']['x280']}")
        report["kernel"] = [r["error_name"] for r in run.stop().islands]
        for mode, path in report["paths"].items():
            for name, (iteration, limit) in cases.items():
                rounds, want, errors, iteration_end = g.replay(
                    ranks, iteration, limit, args.max_runs
                )
                got = path[name]
                got["state_wrong"] = int(
                    (g.read_state(states[mode, name], ranks) != want).sum()
                )
                ok &= got["status"] == "ended" and got["state_wrong"] == 0
                ok &= (got["rounds"], got["it"], got["err"]) == (
                    rounds, iteration_end, errors[-1]
                )  # fmt: skip
    report["ok"] = bool(ok and all(error == "none" for error in report["kernel"]))
    report["wall_s"] = round(time.monotonic() - g.STARTED, 1)
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
