"""A graph run host-fed on the card: a step on 60 cores, two branches on disjoint halves
in flight together, a reduce whose result word decides on the card whether to go round
again; the kept graph is run again with other bindings. Prints one JSON line.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

import komorebi as km
from komorebi.engines import graph as gr
from komorebi.engines.graph import Graph

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
REFERENCES = HERE.parents[1] / "experiments" / "graph_checks" / "graphs"
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
    p.add_argument("--stall", action="store_true", help="end with a stalled record")
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


def build(n, max_runs):
    """The graph; the state tensor's address is a binding, so the kept graph runs on
    another tensor without a new upload."""
    g = Graph(max_runs=max_runs)
    addr, it, limit = g.bindings("state", "it", "limit")
    err = g.reg()
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
    bad = sorted(REFERENCES.glob("bad_*.bin"))
    refused = [gr.check(path.read_bytes()) is not None for path in bad]
    out["bad_uploads_refused"] = f"{sum(refused)}/{len(bad)}"
    with km.Session(log=log) as session:
        log("device open")
        out.update(run(session, args))
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out), flush=True)


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
    device, n = session.device, args.cores
    cases = {"first": args.first, "again": args.again}
    states = {name: state_tensor(device, n) for name in cases}
    accessor = ttnn.TensorAccessorArgs(states["first"]).get_compile_time_args()
    program = km.Program(
        "graph",
        kernels=KERNELS,
        buffers=[km.Buffer(index=16, format="uint32", page_bytes=64, pages=1)],
        compile_args={"writer": accessor},
        segments=KERNELS / "graph.h",
    )
    g, (it_reg, err_reg) = build(n, args.max_runs)
    run = session.load(program, trace=4 * 2 * args.max_runs + 8, deadline=args.deadline)
    log("loaded")
    out, done = {}, {}
    try:
        for name, (it, limit) in cases.items():
            addr = states[name].buffer_address()
            log(f"{name}: it {it}, limit {limit}")
            got = run.run_hostfed(g, {"state": addr, "it": it, "limit": limit}, 30.0)
            log(f"{name}: {got.status} after {got.runs} rounds")
            done[name] = got
        if args.stall:
            stall = Graph()
            stall.record(5, team=(0, 4), args=[1])
            got = run.run_hostfed(stall, timeout=10.0)
            out["stall"] = {"status": got.status, "error": list(got.error or ())}
        run.stop()
        log("stopped")
    except km.KomorebiError as err:
        log(f"failed: {err}")
        out["failed"] = str(err)
    for name, got in done.items():
        it, limit = cases[name]
        want_rounds, x, errs, want_it = replay(n, it, limit, args.max_runs)
        out[name] = {
            "status": got.status,
            "rounds": got.runs,
            "rounds_numpy": want_rounds,
            "err": got.regs[err_reg.index],
            "err_numpy": errs[-1],
            "it": got.regs[it_reg.index],
            "it_numpy": want_it,
            "state_wrong": int((read_state(states[name], n) != x).sum()),
            "records": len(got.records),
        }
    trace = run.trace()
    if trace is not None and "first" in done:
        out["branches_overlapping_rounds"] = overlap(trace, done["first"])
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
