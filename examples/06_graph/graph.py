"""A graph on the card: a step on 60 cores, two branches on disjoint halves in flight
together, a reduce whose result word decides whether to go round again; the kept graph
is run again with other bindings. Run host-fed (the executor on the host) and on the
X280 (the executor in the dispatch kernel). Prints one JSON line.
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
    out, done, hostfed = {}, {}, {}
    # One load per path: the kernel's most slots busy at once is the path's own.
    for mode in modes:
        run = session.load(program, trace=8 * args.max_runs + 8, deadline=args.deadline)
        log(f"{mode}: loaded")
        try:
            if mode == "host":
                for name, (it, limit) in cases.items():
                    addr = states["host", name].buffer_address()
                    log(f"host {name}: it {it}, limit {limit}")
                    b = {"state": addr, "it": it, "limit": limit}
                    got = run.run_hostfed(g, b, 30.0)
                    log(f"host {name}: {got.status} after {got.runs} rounds")
                    hostfed[name] = got
                    regs = (got.regs[it_reg.index], got.regs[err_reg.index])
                    done["host", name] = (got.status, got.runs, regs, len(got.records))
            else:
                done.update(on_x280(run, g, states, cases))
            if args.stall and mode == modes[-1]:
                out["stall"] = stall(run, args.deadline, mode == "x280")
            else:
                run.stop()
            log(f"{mode}: stopped")
        except km.KomorebiError as err:
            log(f"{mode}: failed: {err}")
            out[f"failed_{mode}"] = str(err)
        if not run.stopped:
            continue  # failed: no result block to read
        out[f"max_busy_{mode}"] = run.result()["max_busy"]
        if mode == "host" and run.trace() is not None and "first" in hostfed:
            out["branches_overlapping_rounds"] = overlap(run.trace(), hostfed["first"])
    if "max_busy_x280" in out:
        # LEFT and RIGHT in flight together on the X280: two slots busy at once.
        out["x280_branches_together"] = out["max_busy_x280"] >= 2
    for (mode, name), (status, rounds, (it_got, err_got), records) in done.items():
        it, limit = cases[name]
        want_rounds, x, errs, want_it = replay(n, it, limit, args.max_runs)
        key = name if mode == "host" else f"{name}_x280"
        out[key] = {
            "status": status,
            "rounds": rounds,
            "rounds_numpy": want_rounds,
            "err": err_got,
            "err_numpy": errs[-1],
            "it": it_got,
            "it_numpy": want_it,
            "state_wrong": int((read_state(states[mode, name], n) != x).sum()),
        }
        if records is not None:
            out[key]["records"] = records
    return out


def on_x280(run, g, states, cases):
    """The graph uploaded once and run by the dispatch kernel's engine: started kept,
    then restarted by its req with the second case's bindings, then released.
    An error event raises EngineError. {("x280", case): (status, rounds, (it, err),
    None)}: the records stay in the kernel."""
    up = run.upload(g)
    done = {}
    for k, (name, (it, limit)) in enumerate(cases.items()):
        b = {"state": states["x280", name].buffer_address(), "it": it, "limit": limit}
        log(f"x280 {name}: it {it}, limit {limit}")
        t = run.start(up, b, kept=True) if k == 0 else run.restart(up.req, b)
        end = run.wait_engine(t, timeout=30.0)
        rounds = 1 + sum(ev.kind == "round" for ev in t.events)
        log(f"x280 {name}: {end.kind} after {rounds} rounds")
        done["x280", name] = ("ended", rounds, tuple(end.regs), None)
    run.release(up)
    return done


def stall(run, deadline, x280):
    """A graph whose record never completes, last: its deadline ends the graph with
    reason 8 and the kernel with its deadline error. Stops the run."""
    g = Graph(deadline_us=round(deadline * 1e6), req=7)
    g.record(5, team=(0, 4), args=[1])
    if not x280:
        got = run.run_hostfed(g, timeout=10.0)
        run.stop()
        return {"status": got.status, "error": list(got.error or ())}
    out = {"on": "x280"}
    try:
        run.wait_engine(run.start(run.upload(g)), timeout=10.0)
        out["status"] = "ended"
    except km.EngineError as err:
        out.update(status="error", error=[err.reason, err.node])
    try:
        run.stop()
    except km.KernelFailed as err:
        out["kernel_error"] = (err.result or {}).get("error")
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
