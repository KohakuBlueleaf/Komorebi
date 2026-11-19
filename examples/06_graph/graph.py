"""A graph on the card: a step on 60 cores, two branches on disjoint halves in flight
together, a reduce whose result word decides whether to go round again; the kept graph
is run again with other bindings. Run host-fed (the executor on the host) and on the
X280 (the executor in the dispatcher). Prints one JSON line; exits 1 unless ok.
"""

import argparse
import contextlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.engines import graph as graph_engine
from komorebi.engines import stream as stream_engine
from komorebi.engines.graph import Graph

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
PAGES, WORDS = 64, 16  # state tensor: one 64-byte page per core, word 0 used
MASK_32_BITS = 0xFFFFFFFF
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=60, help="group of STEP and NORM")
    parser.add_argument(
        "--first", nargs=2, type=int, default=[0, 1_000_000], help="iteration, limit"
    )
    parser.add_argument(
        "--again", nargs=2, type=int, default=[3, 10_000_000], help="iteration, limit"
    )
    parser.add_argument("--max-runs", type=int, default=50)
    parser.add_argument(
        "--deadline", type=float, default=1.0, help="per command, seconds"
    )
    parser.add_argument("--stall", action="store_true", help="a stalled graph first")
    parser.add_argument(
        "--decide",
        choices=["host", "x280", "both"],
        default="both",
        help="where the executor runs",
    )
    return parser.parse_args()


def initial(cores):
    return (np.arange(cores, dtype=np.uint64) * 7) & MASK_32_BITS


def replay(cores, iteration, limit, max_runs):
    """The graph in numpy: (rounds, final state, errors, iteration), or rounds None if
    `again` is still set after max_runs."""
    pages = np.arange(cores, dtype=np.uint64)
    state, half, errors = initial(cores), cores // 2, []
    for rounds in range(1, max_runs + 1):
        state = (state + np.uint64(iteration) + pages) & MASK_32_BITS
        state[:half] >>= np.uint64(1)
        state[half:] = (state[half:] * np.uint64(3)) & MASK_32_BITS
        errors.append(int(state.sum()) & MASK_32_BITS)
        iteration += 1
        if not errors[-1] < limit:
            return rounds, state, errors, iteration
    return None, state, errors, iteration


def build(cores, max_runs, deadline):
    """The graph; the state tensor's address is a binding, so the kept graph runs on
    another tensor without a new upload. `iteration` and `error` are registers 0 and 1,
    the two an X280 run's end event carries."""
    graph = Graph(max_runs=max_runs, deadline_microseconds=round(deadline * 1e6), req=6)
    iteration = graph.bindings("iteration")
    error = graph.register()
    state_address, limit = graph.bindings("state", "limit")
    half = cores // 2
    step = graph.command(1, group=(0, cores), args=[state_address, iteration])
    left = graph.command(2, group=(0, half), args=[state_address], after=[step])
    right = graph.command(
        3, group=(half, cores - half), args=[state_address], after=[step]
    )
    norm = graph.command(
        4,
        group=(0, cores),
        args=[state_address, cores],
        after=[left, right],
        result=error,
    )
    cpu = graph.cpu(after=[norm])
    cpu.addi(iteration, iteration, 1)
    cpu.lt(graph.again, error, limit)
    return graph, (iteration, error)


def main():
    args = parse()
    report = {"cores": args.cores}
    with km.Session(log=log) as session:
        log("device open")
        report.update(run(session, args))
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


def state_tensor(session, cores):
    rows = np.zeros((PAGES, WORDS), np.uint32)
    rows[:cores, 0] = initial(cores)
    return session.tensor(rows, "uint32")


def read_state(state, cores):
    rows = state.read()
    return rows[:cores, 0].astype(np.uint64)


def run(session, args):
    """The stalled graph first (its own run) when asked, then one run per path. The
    output's `ok`: every path as the numpy replay and as the simulator (commands
    submitted, per core), the paths alike, the branches in flight together."""
    group_cores = args.cores
    cases = {"first": args.first, "again": args.again}
    modes = ["host", "x280"] if args.decide == "both" else [args.decide]
    states = {
        (mode, name): state_tensor(session, group_cores)
        for mode in modes
        for name in cases
    }
    accessor = next(iter(states.values())).tensor_args()
    program = km.Program(
        "graph",
        kernels=KERNELS,
        buffers=[km.Buffer(index=16, format="uint32", page_bytes=64, pages=1)],
        compile_args={"writer": accessor},
        handlers=KERNELS / "graph.h",
    )
    graph, (iteration_register, error_register) = build(
        group_cores, args.max_runs, args.deadline
    )
    report, ok = {}, True
    if args.stall:
        load = session.load(program, deadline=args.deadline)
        report["stall"] = stall(load, args.deadline, modes[-1] == "x280")
        ok &= (
            report["stall"].get("error") == [8, 0]
            and report["stall"]["stop"] == "clean"
        )
    report["paths"] = {}
    for mode in modes:
        load = session.load(
            program, trace=8 * args.max_runs + 8, deadline=args.deadline
        )
        log(f"{mode}: loaded")
        cores = load.ranks  # a Tensix dispatcher's core takes no command
        if "bad_uploads" not in report:  # engines started before the traced commands
            report["bad_uploads"] = rejections(load)
        report["paths"][mode] = path = one_path(
            load, mode, graph, (iteration_register, error_register), states, cases
        )
        path.update(card_counts(load))
        if mode == "host" and "first" in path:
            path["branches_overlapping_rounds"] = overlap(load.trace(), path.pop("got"))
        path.pop("got", None)
    rejected = sum(got == want for got, want in report["bad_uploads"].values())
    report["bad_uploads_rejected"] = f"{rejected}/{len(report['bad_uploads'])}"
    ok &= rejected == len(report["bad_uploads"])
    want = simulated(graph, group_cores, cases, args.max_runs, cores)
    for mode, path in report["paths"].items():
        ok &= "failed" not in path and path.get("max_busy", 0) >= 2
        for name, (iteration, limit) in cases.items():
            rounds, expected_state, errors, iteration_end = replay(
                group_cores, iteration, limit, args.max_runs
            )
            got = path.get(name, {})
            got["state_wrong"] = int(
                (read_state(states[mode, name], group_cores) != expected_state).sum()
            )
            ok &= got.get("status") == "ended" and got["state_wrong"] == 0
            ok &= (got.get("rounds"), got.get("it"), got.get("err")) == (
                rounds, iteration_end, errors[-1]
            )  # fmt: skip
        ok &= path.get("kernel_commands") == want["kernel_commands"]
        ok &= path.get("core_commands") == want["core_commands"]
    paths = list(report["paths"].values())
    report["paths_agree"] = all(
        {key: path.get(key) for key in [*cases, "kernel_commands", "core_commands"]}
        == {
            key: paths[0].get(key)
            for key in [*cases, "kernel_commands", "core_commands"]
        }
        for path in paths
    )
    report["simulated"] = {"kernel_commands": want["kernel_commands"]}
    report["ok"] = bool(ok and report["paths_agree"])
    for path in paths:
        path["core_commands_ok"] = (
            path.pop("core_commands", None) == want["core_commands"]
        )
    return report


def one_path(load, mode, graph, registers, states, cases):
    """Both cases on one path, then the stop; a failure stops the run too. {case:
    {status, rounds, it, err}, "got": the first host-fed result (for its trace)}."""
    iteration_register, error_register = registers
    path = {}
    try:
        if mode == "host":
            for name, (iteration, limit) in cases.items():
                state_address = states["host", name].address
                log(f"host {name}: iteration {iteration}, limit {limit}")
                bindings = {
                    "state": state_address,
                    "iteration": iteration,
                    "limit": limit,
                }
                got = load.run_hostfed(graph, bindings, 30.0)
                log(f"host {name}: {got.status} after {got.runs} rounds")
                path.setdefault("got", got)
                path[name] = {
                    "status": got.status,
                    "rounds": got.runs,
                    "it": got.registers[iteration_register.index],
                    "err": got.registers[error_register.index],
                }
        else:
            path.update(on_x280(load, graph, states, cases))
        load.stop()
        log(f"{mode}: stopped")
    except km.KomorebiError as error:
        log(f"{mode}: failed: {error}")
        path["failed"] = str(error)
        with contextlib.suppress(km.KomorebiError):
            load.stop()  # stopped before the next load, its counts read if it can be
        if not load.stopped:
            with contextlib.suppress(km.KomorebiError):
                load.abort()
    return path


def card_counts(load):
    """What the card saw: the kernel's commands and most slots busy at once (its result
    block), every core's commands taken (its state words)."""
    if not load.stopped:
        return {}
    with contextlib.suppress(km.KomorebiError):
        result = load.result()
        cores = [state["runs"] for state in load.state()][: load.ranks]
        return {
            "max_busy": result["max_busy"],
            "kernel_commands": result["commands"],
            "core_commands": cores,
        }
    return {}


def simulated(graph, group_cores, cases, max_runs, cores):
    """The commands the simulator submits for both cases, each command's result word the
    replay's error of its round; plus the EXIT every run ends with (all cores)."""
    per_core, count = [1] * cores, 1
    for iteration, limit in cases.values():
        errors = replay(group_cores, iteration, limit, max_runs)[2]
        simulation = graph.simulate(
            {"state": 0, "iteration": iteration, "limit": limit},
            core_count=cores,
            result=lambda command, errors=errors: errors[command.run - 1],
        )
        count += len(simulation.commands)
        for command in simulation.commands:
            for core in range(command.first, command.first + command.count):
                per_core[core] += 1
    return {"kernel_commands": count, "core_commands": per_core}


def on_x280(load, graph, states, cases):
    """The graph uploaded once and run by the dispatcher's engine: started kept,
    then restarted by its req with the second case's bindings, then released. An
    error event raises EngineError. {case: {status, rounds, it, err}}."""
    upload = load.upload(graph)
    path = {}
    for index, (name, (iteration, limit)) in enumerate(cases.items()):
        bindings = {
            "state": states["x280", name].address,
            "iteration": iteration,
            "limit": limit,
        }
        log(f"x280 {name}: iteration {iteration}, limit {limit}")
        handle = (
            load.start(upload, bindings, kept=True)
            if index == 0
            else load.restart(upload.req, bindings)
        )
        end = load.wait_engine(handle, timeout=30.0)
        rounds = 1 + sum(event.kind == "round" for event in handle.events)
        log(f"x280 {name}: {end.kind} after {rounds} rounds")
        iteration_end, error = end.registers
        path[name] = {
            "status": "ended",
            "rounds": rounds,
            "it": iteration_end,
            "err": error,
        }
    load.release(upload)
    return path


def stall(load, deadline, x280):
    """A graph whose command never completes, in a run of its own: its deadline ends the
    graph with reason 8; the dispatcher gives the late command up and the run goes on,
    so the stop is clean ("stop": "clean", else the kernel's error). Stops the run."""
    graph = Graph(deadline_microseconds=round(deadline * 1e6), req=7)
    graph.command(5, group=(0, 4), args=[1])
    report = {"on": "x280" if x280 else "host"}
    try:
        if x280:
            load.wait_engine(load.start(load.upload(graph)), timeout=10.0)
            report["status"] = "ended"
        else:
            got = load.run_hostfed(graph, timeout=10.0)
            report.update(status=got.status, error=list(got.error or ()))
    except km.EngineError as error:
        report.update(status="error", error=[error.reason, error.node])
    try:
        load.stop()
        report["stop"] = "clean"
    except km.KernelFailed as error:
        report["stop"] = f"kernel error {(error.result or {}).get('error')}"
    if not load.stopped:
        with contextlib.suppress(km.KomorebiError):
            load.abort()
    return report


def bad_uploads():
    """One upload per class the kernel rejects, each a fault put into a good graph (a
    command, then a CPU node): {class: (bytes, [reason, node] of its error event)}."""
    graph = Graph(req=8)
    counter = graph.register()
    graph.cpu(after=[graph.command(1, group=(0, 4), args=[0])]).addi(
        counter, counter, 1
    )
    good = graph.encode()
    assert graph_engine.check(good) is None
    header = dict(
        zip(graph_engine.HEADER_FIELDS, graph_engine.HEADER.unpack_from(good))
    )
    edges = (
        graph_engine.HEADER.size
        + -(-header["node_count"] * graph_engine.NODE_BYTES // graph_engine.LINE)
        * graph_engine.LINE
    )
    ops = edges + -(-header["edge_count"] * 2 // graph_engine.LINE) * graph_engine.LINE
    faults = {
        "header_size": [
            stream_engine.ERROR_HEADER,
            None,
        ],  # one line more than the header says
        "edge_order": [graph_engine.ERROR_EDGE, 0],  # node 0's successor is node 0
        "index_range": [
            stream_engine.ERROR_TEMPLATE,
            0,
        ],  # node 0's template is past the last
        "cpu_opcode": [
            stream_engine.ERROR_OPCODE,
            1,
        ],  # the CPU node's micro-op is an SUBMIT
    }
    data = {name: bytearray(good) for name in faults}
    data["header_size"] += bytes(graph_engine.LINE)
    struct.pack_into("<H", data["edge_order"], edges, 0)
    struct.pack_into(
        "<H", data["index_range"], graph_engine.HEADER.size + 4, header["command_count"]
    )
    struct.pack_into("<B", data["cpu_opcode"], ops, stream_engine.OP_SUBMIT)
    return {name: (bytes(data[name]), want) for name, want in faults.items()}


def rejections(load):
    """Each bad upload run by the kernel: {class: [got, want]}, `got` the [reason,
    node] of its error event, else how it ended; the host check must agree too."""
    report = {}
    for name, (data, want) in bad_uploads().items():
        error = graph_engine.check(data)
        node = (
            None if error is None or error.node == graph_engine.NO_NODE else error.node
        )
        upload = load.upload(data, kind="graph")
        try:
            got = [load.wait_engine(load.start(upload), timeout=5.0).kind]
        except km.EngineError as engine_error:
            got = [engine_error.reason, engine_error.node]
        load.release(upload)
        if error is None or [error.reason, node] != want:
            got = ["host check", error]
        log(f"bad upload {name}: {got}, want {want}")
        report[name] = [got, want]
    return report


def overlap(trace, got):
    """Rounds in which the two branches were in flight at the same time."""
    handles = {(command.run, command.node): command.handle for command in got.commands}
    both = 0
    for round_number in range(1, got.runs + 1):
        left, right = handles.get((round_number, 1)), handles.get((round_number, 2))
        leads = [trace.lead(left, right), trace.lead(right, left)]
        both += any(lead is not None and lead > 0 for lead in leads)
    return f"{both}/{got.runs}"


if __name__ == "__main__":
    main()
