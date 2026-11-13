"""A graph on the card: a step on 60 cores, two branches on disjoint halves in flight
together, a reduce whose result word decides whether to go round again; the kept graph
is run again with other bindings. Run host-fed (the executor on the host) and on the
X280 (the executor in the dispatch kernel). Prints one JSON line; exits 1 unless ok.
"""

import argparse
import contextlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

import komorebi as km
from komorebi.engines import graph as gr
from komorebi.engines import stream as st
from komorebi.engines.graph import Graph

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
PAGES, WORDS = 64, 16  # state tensor: one 64-byte page per core, word 0 used
M32 = 0xFFFFFFFF
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cores", type=int, default=60, help="team of STEP and NORM")
    p.add_argument(
        "--first", nargs=2, type=int, default=[0, 1_000_000], help="it, limit"
    )
    p.add_argument(
        "--again", nargs=2, type=int, default=[3, 10_000_000], help="it, limit"
    )
    p.add_argument("--max-runs", type=int, default=50)
    p.add_argument("--deadline", type=float, default=1.0, help="per record, seconds")
    p.add_argument("--stall", action="store_true", help="a stalled graph first")
    p.add_argument(
        "--decide",
        choices=["host", "x280", "both"],
        default="both",
        help="where the executor runs",
    )
    return p.parse_args()


def initial(n):
    return (np.arange(n, dtype=np.uint64) * 7) & M32


def replay(n, it, limit, max_runs):
    """The graph in numpy: (rounds, final state, err, it), or rounds None if `again`
    is still set after max_runs."""
    p = np.arange(n, dtype=np.uint64)
    x, half, errs = initial(n), n // 2, []
    for rounds in range(1, max_runs + 1):
        x = (x + np.uint64(it) + p) & M32
        x[:half] >>= np.uint64(1)
        x[half:] = (x[half:] * np.uint64(3)) & M32
        errs.append(int(x.sum()) & M32)
        it += 1
        if not errs[-1] < limit:
            return rounds, x, errs, it
    return None, x, errs, it


def build(n, max_runs, deadline):
    """The graph; the state tensor's address is a binding, so the kept graph runs on
    another tensor without a new upload. `it` and `err` are registers 0 and 1, the two
    an X280 run's end event carries."""
    g = Graph(max_runs=max_runs, deadline_us=round(deadline * 1e6), req=6)
    it = g.bindings("it")
    err = g.reg()
    addr, limit = g.bindings("state", "limit")
    half = n // 2
    a = g.record(1, team=(0, n), args=[addr, it])
    b = g.record(2, team=(0, half), args=[addr], after=[a])
    c = g.record(3, team=(half, n - half), args=[addr], after=[a])
    d = g.record(4, team=(0, n), args=[addr, n], after=[b, c], result=err)
    e = g.cpu(after=[d])
    e.addi(it, it, 1)
    e.lt(g.again, err, limit)
    return g, (it, err)


def main():
    args = parse()
    out = {"cores": args.cores}
    with km.Session(log=log) as session:
        log("device open")
        out.update(run(session, args))
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["ok"] else 1)


def state_tensor(device, n):
    rows = np.zeros((PAGES, WORDS), np.uint32)
    rows[:n, 0] = initial(n)
    host = torch.from_numpy(rows.view(np.int32))
    return ttnn.from_torch(
        host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def read_state(state, n):
    rows = ttnn.to_torch(ttnn.from_device(state)).numpy().view(np.uint32)
    return rows[:n, 0].astype(np.uint64)


def run(session, args):
    """The stalled graph first (its own run) when asked, then one run per path. The
    output's `ok`: every path as the numpy replay and as the simulator (records
    issued, per core), the paths alike, the branches in flight together."""
    device, n = session.device, args.cores
    cases = {"first": args.first, "again": args.again}
    modes = ["host", "x280"] if args.decide == "both" else [args.decide]
    states = {(m, c): state_tensor(device, n) for m in modes for c in cases}
    accessor = ttnn.TensorAccessorArgs(
        next(iter(states.values()))
    ).get_compile_time_args()
    program = km.Program(
        "graph",
        kernels=KERNELS,
        buffers=[km.Buffer(index=16, format="uint32", page_bytes=64, pages=1)],
        compile_args={"writer": accessor},
        segments=KERNELS / "graph.h",
    )
    g, (it_reg, err_reg) = build(n, args.max_runs, args.deadline)
    out, ok = {}, True
    if args.stall:
        load = session.load(program, deadline=args.deadline)
        out["stall"] = stall(load, args.deadline, modes[-1] == "x280")
        ok &= out["stall"].get("error") == [8, 0]
    out["paths"] = {}
    for mode in modes:
        load = session.load(
            program, trace=8 * args.max_runs + 8, deadline=args.deadline
        )
        log(f"{mode}: loaded")
        cores = load.cores
        if "bad_uploads" not in out:  # engines started before the traced records
            out["bad_uploads"] = refusals(load)
        out["paths"][mode] = path = one_path(
            load, mode, g, (it_reg, err_reg), states, cases
        )
        path.update(card_counts(load))
        if mode == "host" and "first" in path:
            path["branches_overlapping_rounds"] = overlap(load.trace(), path.pop("got"))
        path.pop("got", None)
    refused = sum(got == want for got, want in out["bad_uploads"].values())
    out["bad_uploads_refused"] = f"{refused}/{len(out['bad_uploads'])}"
    ok &= refused == len(out["bad_uploads"])
    want = simulated(g, n, cases, args.max_runs, cores)
    for mode, path in out["paths"].items():
        ok &= "failed" not in path and path.get("max_busy", 0) >= 2
        for name, (it, limit) in cases.items():
            rounds, x, errs, it_end = replay(n, it, limit, args.max_runs)
            got = path.get(name, {})
            got["state_wrong"] = int((read_state(states[mode, name], n) != x).sum())
            ok &= got.get("status") == "ended" and got["state_wrong"] == 0
            ok &= (got.get("rounds"), got.get("it"), got.get("err")) == (
                rounds, it_end, errs[-1]
            )  # fmt: skip
        ok &= path.get("kernel_records") == want["kernel_records"]
        ok &= path.get("core_records") == want["core_records"]
    paths = list(out["paths"].values())
    out["paths_agree"] = all(
        {k: p.get(k) for k in [*cases, "kernel_records", "core_records"]}
        == {k: paths[0].get(k) for k in [*cases, "kernel_records", "core_records"]}
        for p in paths
    )
    out["simulated"] = {"kernel_records": want["kernel_records"]}
    out["ok"] = bool(ok and out["paths_agree"])
    for path in paths:
        path["core_records_ok"] = path.pop("core_records", None) == want["core_records"]
    return out


def one_path(load, mode, g, regs, states, cases):
    """Both cases on one path, then the stop; a failure stops the run too. {case:
    {status, rounds, it, err}, "got": the first host-fed result (for its trace)}."""
    it_reg, err_reg = regs
    path = {}
    try:
        if mode == "host":
            for name, (it, limit) in cases.items():
                addr = states["host", name].buffer_address()
                log(f"host {name}: it {it}, limit {limit}")
                b = {"state": addr, "it": it, "limit": limit}
                got = load.run_hostfed(g, b, 30.0)
                log(f"host {name}: {got.status} after {got.runs} rounds")
                path.setdefault("got", got)
                path[name] = {
                    "status": got.status,
                    "rounds": got.runs,
                    "it": got.regs[it_reg.index],
                    "err": got.regs[err_reg.index],
                }
        else:
            path.update(on_x280(load, g, states, cases))
        load.stop()
        log(f"{mode}: stopped")
    except km.KomorebiError as err:
        log(f"{mode}: failed: {err}")
        path["failed"] = str(err)
        with contextlib.suppress(km.KomorebiError):
            load.stop()  # stopped before the next load, its counts read if it can be
        if not load.stopped:
            with contextlib.suppress(km.KomorebiError):
                load.abort()
    return path


def card_counts(load):
    """What the card saw: the kernel's records and most slots busy at once (its result
    block), every core's records taken (its state words)."""
    if not load.stopped:
        return {}
    with contextlib.suppress(km.KomorebiError):
        r = load.result()
        cores = [c["runs"] for c in load.state()]
        return {
            "max_busy": r["max_busy"],
            "kernel_records": r["records"],
            "core_records": cores,
        }
    return {}


def simulated(g, n, cases, max_runs, cores):
    """The records the simulator issues for both cases, each record's result word the
    replay's err of its round; plus the EXIT every run ends with (all cores)."""
    per_core, count = [1] * cores, 1
    for it, limit in cases.values():
        errs = replay(n, it, limit, max_runs)[2]
        sim = g.simulate(
            {"state": 0, "it": it, "limit": limit},
            n_cores=cores,
            result=lambda r, errs=errs: errs[r.run - 1],
        )
        count += len(sim.records)
        for r in sim.records:
            for c in range(r.first, r.first + r.count):
                per_core[c] += 1
    return {"kernel_records": count, "core_records": per_core}


def on_x280(load, g, states, cases):
    """The graph uploaded once and run by the dispatch kernel's engine: started kept,
    then restarted by its req with the second case's bindings, then released. An
    error event raises EngineError. {case: {status, rounds, it, err}}."""
    up = load.upload(g)
    path = {}
    for k, (name, (it, limit)) in enumerate(cases.items()):
        b = {"state": states["x280", name].buffer_address(), "it": it, "limit": limit}
        log(f"x280 {name}: it {it}, limit {limit}")
        t = load.start(up, b, kept=True) if k == 0 else load.restart(up.req, b)
        end = load.wait_engine(t, timeout=30.0)
        rounds = 1 + sum(ev.kind == "round" for ev in t.events)
        log(f"x280 {name}: {end.kind} after {rounds} rounds")
        it_end, err = end.regs
        path[name] = {"status": "ended", "rounds": rounds, "it": it_end, "err": err}
    load.release(up)
    return path


def stall(load, deadline, x280):
    """A graph whose record never completes, in a run of its own: its deadline ends the
    graph with reason 8 and the kernel with its deadline error (4). Stops the run."""
    g = Graph(deadline_us=round(deadline * 1e6), req=7)
    g.record(5, team=(0, 4), args=[1])
    out = {"on": "x280" if x280 else "host"}
    try:
        if x280:
            load.wait_engine(load.start(load.upload(g)), timeout=10.0)
            out["status"] = "ended"
        else:
            got = load.run_hostfed(g, timeout=10.0)
            out.update(status=got.status, error=list(got.error or ()))
    except km.EngineError as err:
        out.update(status="error", error=[err.reason, err.node])
    try:
        load.stop()
    except km.KernelFailed as err:
        out["kernel_error"] = (err.result or {}).get("error")
    if not load.stopped:
        with contextlib.suppress(km.KomorebiError):
            load.abort()
    return out


def bad_uploads():
    """One upload per class the kernel refuses, each a fault put into a good graph (a
    record, then a CPU node): {class: (bytes, [reason, node] of its error event)}."""
    g = Graph(req=8)
    r = g.reg()
    g.cpu(after=[g.record(1, team=(0, 4), args=[0])]).addi(r, r, 1)
    good = g.encode()
    assert gr.check(good) is None
    h = dict(zip(gr.HEADER_FIELDS, gr.HEADER.unpack_from(good)))
    edges = gr.HEADER.size + -(-h["n_nodes"] * gr.NODE_BYTES // gr.LINE) * gr.LINE
    ops = edges + -(-h["n_edges"] * 2 // gr.LINE) * gr.LINE
    faults = {
        "header_size": [st.ERR_HEADER, None],  # one line more than the header says
        "edge_order": [gr.ERR_EDGE, 0],  # node 0's successor is node 0
        "index_range": [st.ERR_TEMPLATE, 0],  # node 0's template is past the last
        "cpu_opcode": [st.ERR_OPCODE, 1],  # the CPU node's micro-op is an ISSUE
    }
    data = {name: bytearray(good) for name in faults}
    data["header_size"] += bytes(gr.LINE)
    struct.pack_into("<H", data["edge_order"], edges, 0)
    struct.pack_into("<H", data["index_range"], gr.HEADER.size + 4, h["n_records"])
    struct.pack_into("<B", data["cpu_opcode"], ops, st.OP_ISSUE)
    return {name: (bytes(data[name]), want) for name, want in faults.items()}


def refusals(load):
    """Each bad upload run by the kernel: {class: [got, want]}, `got` the [reason,
    node] of its error event, else how it ended; the host check must agree too."""
    out = {}
    for name, (data, want) in bad_uploads().items():
        error = gr.check(data)
        node = None if error is None or error.node == gr.NO_NODE else error.node
        up = load.upload(data, kind="graph")
        try:
            got = [load.wait_engine(load.start(up), timeout=5.0).kind]
        except km.EngineError as err:
            got = [err.reason, err.node]
        load.release(up)
        if error is None or [error.reason, node] != want:
            got = ["host check", error]
        log(f"bad upload {name}: {got}, want {want}")
        out[name] = [got, want]
    return out


def overlap(trace, got):
    """Rounds in which the two branches were in flight at the same time."""
    tickets = {(r.run, r.node): r.ticket for r in got.records}
    both = 0
    for k in range(1, got.runs + 1):
        left, right = tickets.get((k, 1)), tickets.get((k, 2))
        leads = [trace.lead(left, right), trace.lead(right, left)]
        both += any(x is not None and x > 0 for x in leads)
    return f"{both}/{got.runs}"


if __name__ == "__main__":
    main()
