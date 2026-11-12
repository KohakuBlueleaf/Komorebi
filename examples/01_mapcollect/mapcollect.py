"""Map/collect on resident workers under the dispatch kernel on X280 cluster 0: each
member of a MAP record's team adds its share of T input tiles into one tile; the host
checks every tile against numpy and prints one JSON line. See README.md.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

import komorebi as km
from komorebi.layout import share

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
TILE = 32
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--seg", choices=["map", "nop"], default="map")
    p.add_argument(
        "--teams", choices=["fixed", "random", "pair", "rotate"], default="fixed"
    )
    p.add_argument("--count", type=int, default=8, help="team size (fixed, rotate)")
    p.add_argument("--first", type=int, default=0, help="first core (fixed)")
    p.add_argument("--tiles", type=int, default=64, help="input tiles T")
    p.add_argument("--depth", type=int, default=4, help="ring slots used")
    p.add_argument("--out-pages", type=int, default=2048, help="pages per output")
    p.add_argument("--out-tensors", type=int, default=1, help="output tensors")
    p.add_argument("--stall", type=int, default=-1, help="job index made a STALL")
    p.add_argument("--stall-member", type=int, default=0)
    p.add_argument("--deadline", type=float, default=1.0, help="per record, seconds")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--batch", type=int, default=64, help="commands per host store")
    p.add_argument("--settle-us", type=float, default=1.0, help="pause after a fence")
    p.add_argument("--no-check", action="store_true", help="skip reading outputs")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def make_teams(args, cores, rng):
    """(first, count) per job. rotate: disjoint teams of --count cores in turn."""
    if args.teams == "fixed":
        return [(args.first, args.count)] * args.jobs
    if args.teams == "pair":
        half = cores // 2
        return [
            (0, half) if j % 2 == 0 else (half, cores - half) for j in range(args.jobs)
        ]
    if args.teams == "rotate":
        slots = max(1, cores // args.count)
        return [((j % slots) * args.count, args.count) for j in range(args.jobs)]
    teams = []
    for _ in range(args.jobs):
        count = int(rng.integers(1, cores + 1))
        teams.append((int(rng.integers(0, cores - count + 1)), count))
    return teams


def place_outputs(teams, pages, tensors):
    """Global output page of member 0 per job: a ring of `tensors` x `pages` pages in
    which no job crosses a tensor."""
    base, at = [], 0
    for _, count in teams:
        if at % pages + count > pages:
            at = (at // pages + 1) * pages
        if at >= tensors * pages:
            at = 0
        base.append(at)
        at += count
    return base


def checked_jobs(teams, base, pages):
    """Jobs whose output pages no later job overwrote."""
    owner = np.full(pages, -1)
    for j, ((_, count), b) in enumerate(zip(teams, base)):
        owner[b : b + count] = j
    return [
        j
        for j, ((_, c), b) in enumerate(zip(teams, base))
        if (owner[b : b + c] == j).all()
    ]


def member_sums(x, count):
    """Float64 sum of each member's share of the tiles of x: [count, 32, 32]."""
    tiles = x.reshape(-1, TILE, TILE)
    out = np.zeros((count, TILE, TILE))
    for m in range(count):
        part = share(range(len(tiles)), m, count)
        out[m] = tiles[part.start : part.stop].sum(0)
    return out


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    out = {"jobs": args.jobs, "seg": args.seg, "teams": args.teams}
    with km.Session(log=log) as session:
        log("device open")
        out.update(run(session, args, rng))
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out), flush=True)


def run(session, args, rng):
    device = session.device
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    teams = make_teams(args, cores, rng)
    n, tiles = args.jobs, args.tiles

    # Exact inputs: integers in [-2, 2], so every partial sum (|s| <= 2T) is exact in
    # bf16 for T <= 64. Random inputs: uniform in [-1, 1], rounded to bf16.
    x_exact = rng.integers(-2, 3, (tiles * TILE, TILE)).astype(np.float32)
    x_rand = torch.from_numpy(
        rng.uniform(-1, 1, (tiles * TILE, TILE)).astype(np.float32)
    )
    x_rand = x_rand.bfloat16().float().numpy()

    def tile_tensor(a):
        return ttnn.from_torch(
            torch.from_numpy(a),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    inputs = [tile_tensor(x_exact), tile_tensor(x_rand)]
    pages = args.out_pages
    outs = [tile_tensor(np.zeros((pages * TILE, TILE), np.float32))]
    outs += [tile_tensor(np.zeros((pages * TILE, TILE), np.float32))
             for _ in range(args.out_tensors - 1)]  # fmt: skip
    base = place_outputs(teams, pages, args.out_tensors)

    program = km.Program(
        "mapcollect",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),  # kCbIn
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),  # kCbOut
        ],
        compile_args={
            "reader": ttnn.TensorAccessorArgs(inputs[0]).get_compile_time_args(),
            "writer": ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        depth=args.depth,
        segments=KERNELS / "mapcollect.h",
    )
    run = session.load(
        program,
        trace=n + 1 if args.trace else 0,
        deadline=args.deadline,
        batch=args.batch,
        settle_us=args.settle_us,
    )
    log(f"loaded: {cores} cores, depth {args.depth}")

    tickets, timed_out = [], None
    t0 = time.perf_counter()
    try:
        for j, (first, count) in enumerate(teams):
            if j == args.stall:
                tickets.append(run.issue("stall", (first, count), [args.stall_member]))
            elif args.seg == "nop":
                tickets.append(run.issue(0, (first, count)))
            else:
                words = [
                    inputs[j % 2].buffer_address(),
                    tiles,
                    outs[base[j] // pages].buffer_address(),
                    base[j] % pages,
                ]
                tickets.append(run.issue("map", (first, count), words))
        waited = run.drain(timeout=60.0)
        host_s = time.perf_counter() - t0
        log(f"{waited} events")
        run.stop()
    except km.RecordTimeout as err:
        host_s = time.perf_counter() - t0
        timed_out = err
        log(f"timeout: {err}")
    result = run.result()
    out = {
        "cores": cores,
        "depth": args.depth,
        "count": args.count if args.teams in ("fixed", "rotate") else None,
        "tiles": tiles,
        "kernel": {
            k: result[k] for k in ("error_name", "records", "events", "max_busy")
        },
        "us_per_job_kernel": round(result["last"] / 50 / max(result["records"], 1), 3),
        "us_per_job_host": round(host_s * 1e6 / n, 3),
        "python_us_per_record": {
            "issue": round(run.issue_s * 1e6 / n, 3),
            "store": round(run.flush_s * 1e6 / max(run.stored, 1), 3),
        },
    }
    done = [
        run.event(t) for t in tickets if run.event(t) and run.event(t).kind == "done"
    ]
    if done:
        us = np.array([e.us for e in done])
        out["latency_us"] = {
            "mean": round(float(us.mean()), 3),
            "max": round(float(us.max()), 3),
        }
    states = run.state()
    out["cores_magic_ok"] = sum(c["magic_ok"] for c in states)
    out["cores_ended"] = {
        k: sum(c[k] != 0 for c in states)
        for k in ("reader_end", "writer_end", "compute_end")
    }
    out["compute_wrong"] = sum(sum(c["compute_wrong"]) for c in states)
    out["gave_up_cores"] = sum(1 for c in states if c["reader"] or c["writer"])
    if timed_out is not None:
        out["timed_out"] = {
            "ticket": timed_out.ticket,
            "slot": timed_out.slot,
            "seq": timed_out.seq,
            "team": list(timed_out.team),
            "not_reported": timed_out.cores,
        }
    trace = run.trace()
    if trace is not None:
        out.update(trace_summary(trace, tickets, teams))
    if args.seg == "map" and timed_out is None and not args.no_check:
        out.update(check(args, teams, base, outs, [x_exact, x_rand]))
    return out


def trace_summary(trace, tickets, teams):
    """Mean per-phase times, and how many consecutive disjoint pairs overlapped."""
    phases = [p for p in map(trace.phases, tickets) if p]
    keys = [k for k in phases[0] if k != "count"] if phases else []
    leads = []
    for j in range(len(tickets) - 1):
        (fa, ca), (fb, cb) = teams[j], teams[j + 1]
        if fa + ca <= fb or fb + cb <= fa:
            lead = trace.lead(tickets[j], tickets[j + 1])
            if lead is not None and lead > 0:
                leads.append(lead)
    return {
        "trace_us": {k: round(float(np.mean([p[k] for p in phases])), 3) for k in keys},
        "overlap": {
            "pairs": len(tickets) - 1,
            "overlapping": len(leads),
            "lead_us_mean": round(float(np.mean(leads)), 3) if leads else 0.0,
        },
    }


def check(args, teams, base, outs, xs):
    """Every member's tile of every job not overwritten: bit-exact on exact inputs,
    relative RMS of the total on random inputs."""
    got = []
    for i, t in enumerate(outs):
        got.append(ttnn.to_torch(t).float().numpy())
        log(f"output {i + 1} of {len(outs)} read back")
    got = np.concatenate(got).reshape(-1, TILE, TILE)
    jobs = [j for j in checked_jobs(teams, base, len(got)) if j != args.stall]
    refs, bad, rel_max, exact = {}, [], 0.0, 0
    for j in jobs:
        count = teams[j][1]
        key = (j % 2, count)
        if key not in refs:
            refs[key] = member_sums(xs[j % 2].astype(np.float64), count)
        ref, mine = refs[key], got[base[j] : base[j] + count].astype(np.float64)
        if j % 2 == 0:
            exact += 1
            if not np.array_equal(mine, ref):
                bad.append({"job": j, "team": teams[j]})
        else:
            total, ref_total = mine.sum(0), ref.sum(0)
            rel = float(
                np.sqrt(((total - ref_total) ** 2).mean() / (ref_total**2).mean())
            )
            rel_max = max(rel_max, rel)
            if rel > 1e-2:
                bad.append({"job": j, "team": teams[j], "rel_rms": rel})
    return {
        "jobs_checked": len(jobs),
        "exact_jobs_checked": exact,
        "jobs_bad": len(bad),
        "first_bad": bad[:5],
        "random_total_rel_rms_max": float(f"{rel_max:.3e}"),
    }


if __name__ == "__main__":
    main()
