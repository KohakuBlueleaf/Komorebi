"""MAP jobs with a random team size each, decided three ways: host-fed (one command per
job), by a micro-op stream on the X280 that issues each record and waits for it, and by
one that submits them to the dispatch kernel's job queue. Every way must give the same
tiles, bit-exact against numpy; prints one JSON line. See README.md.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

import komorebi as km
from komorebi.engines import stream as ks
from komorebi.layout import share

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
TILE = 32
STEP = 61  # the input window moves this many tiles per job
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", type=int, default=1000)
    p.add_argument("--tiles", type=int, default=64, help="input tiles T per job")
    p.add_argument("--in-tiles", type=int, default=512, help="tiles of the input")
    p.add_argument(
        "--ways",
        default="hostfed,issue,submit",
        help="comma-separated: hostfed, issue (X280, depth 1), submit (X280, queued)",
    )
    p.add_argument("--depth", type=int, default=16, help="ring slots used")
    p.add_argument("--deadline", type=float, default=1.0, help="per record, seconds")
    p.add_argument("--seed", type=int, default=1, help="the stream's xorshift seed")
    p.add_argument("--card", type=int, default=None)
    return p.parse_args(argv)


def job_stream(form, jobs, cores, window, addr, tiles, seed):
    """The stream of every way: per job a team size from RAND, the output page after
    the previous job's, the input window moved by STEP. `form` "issue": a computed
    first core, ISSUE and WAIT 0; "submit": SUBMIT, the queue picks the team."""
    s = ks.Stream(req=40 if form == "issue" else 41, seed=seed)
    j, x, count, first, room, base, page = s.regs(7)
    n_cores, n_jobs, wrap = s.const(cores), s.const(jobs), s.const(window)
    for r in (j, base, page):
        s.set(r, 0)
    top = s.label()
    s.rand(x)
    s.mod(count, x, n_cores)
    s.addi(count, count, 1)  # 1 to every core
    map_args = [addr["in"], tiles, addr["out"], base, page]
    if form == "issue":
        s.sub(room, n_cores, count)
        s.addi(room, room, 1)
        s.shr(first, x, 8)
        s.mod(first, first, room)
        s.issue(addr["map"], (first, count), args=map_args)
        s.wait(0)
    else:
        s.submit(addr["map"], count, args=map_args)
    s.add(base, base, count)
    s.addi(page, page, STEP)
    inside = s.forward()
    s.jlt(page, wrap, inside)
    s.sub(page, page, wrap)
    s.bind(inside)
    s.addi(j, j, 1)
    s.jlt(j, n_jobs, top)
    s.wait(0)
    s.halt(j)
    return s


def simulated(stream, cores):
    """The records the kernel issues for `stream`, from the package's simulator:
    [(first, count, output page, first input tile)]."""
    sim = stream.simulate(n_cores=cores, max_passes=1_000_000)
    if sim.status != "ended":
        raise SystemExit(f"the simulator ends the stream with {sim.status}")
    return [(r.first, r.count, r.args[3], r.args[4]) for r in sim.records]


def expected(x, jobs, tiles):
    """Every output tile: member m of a job adds its share of the job's input tiles
    (float64 over integer inputs, exact in bf16)."""
    sums = np.cumsum(x.reshape(-1, TILE, TILE).astype(np.float64), 0)
    prefix = np.concatenate([np.zeros((1, TILE, TILE)), sums])
    pages = sum(c for _, c, _, _ in jobs)
    out = np.zeros((pages, TILE, TILE))
    for _, count, base, start in jobs:
        for m in range(count):
            part = share(range(start, start + tiles), m, count)
            out[base + m] = prefix[part.stop] - prefix[part.start]
    return out


def membership(jobs, cores):
    """Records each core is a member of."""
    edge = np.zeros(cores + 1, np.int64)
    for first, count, _, _ in jobs:
        edge[first] += 1
        edge[first + count] -= 1
    return np.cumsum(edge)[:cores]


def main():
    args = parse()
    out = {"jobs": args.jobs, "tiles": args.tiles, "seed": args.seed}
    with km.Session(card=args.card, log=log) as session:
        log("device open")
        out.update(run_ways(session, args))
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out), flush=True)


def run_ways(session, args):
    device = session.device
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    ways = args.ways.split(",")
    tiles, in_tiles = args.tiles, max(args.in_tiles, args.tiles + STEP)
    window = in_tiles - tiles + 1
    rng = np.random.default_rng(args.seed)
    # Integers in [-2, 2]: every partial sum of up to 64 tiles is exact in bf16.
    x = rng.integers(-2, 3, (in_tiles * TILE, TILE)).astype(np.float32)

    def tile_tensor(a):
        return ttnn.from_torch(
            torch.from_numpy(a),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    # The jobs, from the simulator of the ISSUE stream: the same for every way.
    probe = {"in": 0, "out": 0, "map": 1}
    issue_jobs = simulated(
        job_stream("issue", args.jobs, cores, window, probe, tiles, args.seed), cores
    )
    submit_jobs = simulated(
        job_stream("submit", args.jobs, cores, window, probe, tiles, args.seed), cores
    )
    # The queue may issue a later, smaller job first: the same jobs, in another order.
    same = sorted(j[1:] for j in issue_jobs) == sorted(j[1:] for j in submit_jobs)
    want = expected(x, issue_jobs, tiles)
    pages = len(want)
    log(f"{len(issue_jobs)} jobs, {pages} output tiles, {cores} cores")

    # Tensors before any load: a tensor write waits for idle cores.
    inp = tile_tensor(x)
    outs = {w: tile_tensor(np.zeros((pages * TILE, TILE), np.float32)) for w in ways}
    program = km.Program(
        "uop_stream",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),  # kCbIn
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),  # kCbOut
        ],
        compile_args={
            "reader": ttnn.TensorAccessorArgs(inp).get_compile_time_args(),
            "writer": ttnn.TensorAccessorArgs(outs[ways[0]]).get_compile_time_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        depth=args.depth,
        segments=KERNELS / "uop_stream.h",
    )
    result = {"cores": cores, "same_jobs_issue_submit": same, "ways": {}}
    for way in ways:
        addr = {
            "in": inp.buffer_address(),
            "out": outs[way].buffer_address(),
            "map": program.seg("map"),
        }
        result["ways"][way] = one_way(
            session, program, args, way, addr, issue_jobs, cores, window
        )
        got = np.concatenate(session.read_bf16(outs[way])).reshape(-1, TILE, TILE)
        wrong = int(np.count_nonzero((got.astype(np.float64) != want).any((1, 2))))
        result["ways"][way]["tiles_wrong"] = wrong
        log(f"{way}: {wrong} tiles wrong of {pages}")
    return result


def one_way(session, program, args, way, addr, jobs, cores, window):
    """Loads the program, runs the jobs one way, stops; times and core counts."""
    run = session.load(program, deadline=args.deadline)
    log(f"{way}: loaded")
    n = len(jobs)
    out = {}
    t0 = time.perf_counter()
    if way == "hostfed":
        firsts = np.array([f for f, _, _, _ in jobs])
        counts = np.array([c for _, c, _, _ in jobs])
        words = np.array(
            [[addr["in"], args.tiles, addr["out"], b, p] for _, _, b, p in jobs],
            np.int64,
        )
        run.issue_many(np.full(n, addr["map"]), firsts, counts, words)
        run.drain(timeout=60.0)
    else:
        stream = job_stream(way, args.jobs, cores, window, addr, args.tiles, args.seed)
        up = run.upload(stream)
        ticket = run.start(up)
        end = run.wait_engine(ticket, timeout=60.0)
        out["halt"] = end.value
        run.release(up)
    out["us_per_job"] = round((time.perf_counter() - t0) * 1e6 / n, 3)
    run.stop()
    log(f"{way}: {out['us_per_job']} us per job")
    states = run.state()
    runs = np.array([c["runs"] for c in states])
    done = np.array([c["completed"] for c in states])
    out["records"] = run.result()["records"]
    out["member_records"] = int(done.sum())
    if way != "submit":  # the queue picks the teams: only the total is known
        member = membership(jobs, cores)
        out["cores_bad"] = int(
            np.count_nonzero((runs != member + 1) | (done != member))
        )
    else:
        out["cores_bad"] = int(done.sum() != sum(c for _, c, _, _ in jobs))
    return out


if __name__ == "__main__":
    main()
