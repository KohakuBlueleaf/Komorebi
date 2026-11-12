"""Map/collect on resident workers, fed by the test feeder (no X280).

Each MAP record names a team; every member adds its share of T input tiles into one
tile and stores it at out[job][member]; the host sums the members' tiles and checks
them against numpy. Jobs alternate exact-valued and random inputs.

    python mapcollect.py --jobs 20 --teams fixed --count 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests" / "hardware" / "feeder"))

import fed  # noqa: E402
import records as rec  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

K = rec.K
KERNELS = HERE / "kernels"
SEGMENTS = rec.constants(KERNELS / "mapcollect.h")
SEG_MAP, SEG_STALL = SEGMENTS["kSegMap"], SEGMENTS["kSegStall"]
TILE = 32


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--seg", choices=["map", "nop"], default="map")
    p.add_argument("--teams", choices=["fixed", "random", "pair"], default="fixed")
    p.add_argument("--count", type=int, default=8, help="team size (fixed)")
    p.add_argument("--first", type=int, default=0, help="first core (fixed)")
    p.add_argument("--tiles", type=int, default=64, help="input tiles T")
    p.add_argument("--depth", type=int, default=16, help="ring slots used")
    p.add_argument("--out-pages", type=int, default=2048, help="pages per output")
    p.add_argument("--out-tensors", type=int, default=1, help="output tensors")
    p.add_argument("--stall", type=int, default=-1, help="job index made a STALL")
    p.add_argument("--stall-member", type=int, default=0)
    p.add_argument("--deadline", type=float, default=0.05, help="per record, seconds")
    p.add_argument("--trace", action="store_true")
    p.add_argument(
        "--rows", action="store_true", help="complete through row leaders (two levels)"
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dump", type=Path, help="save records, trace and state (npz)")
    return p.parse_args()


def make_teams(args, workers, rng):
    if args.teams == "fixed":
        return [(args.first, args.count)] * args.jobs
    if args.teams == "pair":
        half = workers // 2
        return [
            (0, half) if j % 2 == 0 else (half, workers - half)
            for j in range(args.jobs)
        ]
    return rec.random_teams(rng, args.jobs, workers)


def place_outputs(teams, pages, tensors):
    """Global output page of member 0 per job: a ring of `tensors` x `pages` pages
    in which no job crosses a tensor."""
    base, at = [], 0
    for _, count in teams:
        if count > pages:
            raise ValueError(f"a team of {count} needs more than {pages} output pages")
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
        start, length = rec.share(len(tiles), m, count)
        out[m] = tiles[start : start + length].sum(0)
    return out


def trace_phases(res, teams, n):
    """Per job from the trace: microseconds of each phase (see the README)."""
    tr = res.trace.astype(np.int64)
    rows = res.records
    mhz = res.mhz
    out = []
    for j in range(n):
        first, count = teams[j]
        seq = int(rows["seq"][j])
        if seq == 0 or seq > tr.shape[1]:
            continue
        t = tr[first : first + count, seq - 1]
        issue = int(rows["issue"][j])
        rel = rec.wrap(t - issue)
        seen, done, sent = rel[:, 0], rel[:, 1], rel[:, 2]
        out.append(
            {
                "count": count,
                "first_seen": seen.min() / mhz,
                "last_seen": seen.max() / mhz,
                "last_done": done.max() / mhz,
                "last_member_sent": (sent[1:].max() if count > 1 else sent[0]) / mhz,
                "leader_sent": sent[0] / mhz,
                "push_after_last_done": (sent[0] - done.max()) / mhz,
                "acked": int(rec.wrap(rows["acked"][j] - issue)) / mhz,
                "passes_mean": float(t[:, 3].mean()),
            }
        )
    return out


def overlaps(res, teams, n):
    """Consecutive jobs on disjoint teams where a member of the second took its record
    before the first job's leader pushed its completion (wall clock, from the trace)."""
    tr = res.trace.astype(np.int64)
    seqs = res.records["seq"]
    found = []
    for j in range(n - 1):
        (fa, ca), (fb, cb) = teams[j], teams[j + 1]
        sa, sb = int(seqs[j]), int(seqs[j + 1])
        disjoint = fa + ca <= fb or fb + cb <= fa
        if not disjoint or 0 in (sa, sb) or max(sa, sb) > tr.shape[1]:
            continue
        pushed = tr[fa, sa - 1, 2]
        seen = tr[fb : fb + cb, sb - 1, 0].min()
        lead = int(rec.wrap(pushed - seen))
        if 0 < lead < 1 << 31:
            found.append(lead / res.mhz)
    return {
        "pairs": n - 1,
        "overlapping": len(found),
        "lead_us_mean": round(float(np.mean(found)) if found else 0.0, 3),
    }


def summarize_phases(phases):
    keys = [k for k in phases[0] if k != "count"] if phases else []
    return {k: round(float(np.mean([p[k] for p in phases])), 3) for k in keys}


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    device.disable_and_clear_program_cache()
    fed.log(f"device open {time.time() - t_start:.1f}s")
    try:
        out = run(device, args, rng)
    finally:
        ttnn.close_mesh_device(device)
    out["wall_s"] = round(time.time() - t_start, 1)
    print(json.dumps(out), flush=True)


def run(device, args, rng):
    runner = fed.Fed(device)
    workers, n, tiles = runner.workers, args.jobs, args.tiles
    teams = make_teams(args, workers, rng)
    fed.log(f"grid {runner.grid}, {workers} workers, aiclk {runner.mhz} MHz")

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
    zeros = np.zeros((pages * TILE, TILE), dtype=np.float32)
    outs = [tile_tensor(zeros) for _ in range(args.out_tensors)]
    base = place_outputs(teams, pages, args.out_tensors)

    records = []
    for j, (first, count) in enumerate(teams):
        if j == args.stall:
            records.append(rec.record(SEG_STALL, first, count, [args.stall_member]))
        elif args.seg == "nop":
            records.append(rec.record(K["KM_SEG_NOP"], first, count))
        else:
            words = [
                inputs[j % 2].buffer_address(),
                tiles,
                outs[base[j] // pages].buffer_address(),
                base[j] % pages,
            ]
            records.append(rec.record(SEG_MAP, first, count, words))

    accessor = ttnn.TensorAccessorArgs(inputs[0]).get_compile_time_args()
    kernels = fed.Kernels(
        reader=KERNELS / "reader.cpp",
        compute=KERNELS / "compute.cpp",
        writer=KERNELS / "writer.cpp",
        reader_ct=accessor,
        writer_ct=ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args(),
        compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        defines=[("KM_ROW_LEADERS", str(runner.grid[0]))] if args.rows else [],
        cbs=[
            (SEGMENTS["kCbIn"], ttnn.bfloat16, 2048, SEGMENTS["kInPages"]),
            (SEGMENTS["kCbOut"], ttnn.bfloat16, 2048, 2),
        ],
    )
    trace_cap = (n + 4) // 4 * 4 if args.trace else 0  # 64-byte pages
    res = runner.run(
        kernels,
        records,
        tensors=[*inputs, *outs],
        depth=args.depth,
        deadline_s=args.deadline,
        trace_cap=trace_cap,
    )
    if args.dump:
        np.savez(
            args.dump,
            teams=np.asarray(teams),
            base=np.asarray(base),
            seq=res.records["seq"],
            slot=res.records["slot"],
            issue=res.records["issue"],
            acked=res.records["acked"],
            state=res.state,
            trace=res.trace if res.trace is not None else np.zeros(0),
        )
    s = res.summary
    out = {
        "jobs": n,
        "seg": args.seg,
        "teams": args.teams,
        "count": args.count if args.teams == "fixed" else None,
        "tiles": tiles,
        "depth": args.depth,
        "completion": "row leaders" if args.rows else "first member",
        "aiclk_mhz": res.mhz,
        "feeder": s,
        "us_per_record": round(s["span_cycles"] / res.mhz / max(s["done"], 1), 3),
        "issue_us_mean": round(s["issue_cycles"] / res.mhz / max(s["issued"], 1), 3),
        "host_program_s": round(res.host_s, 2),
    }
    lat = rec.wrap(res.records["acked"] - res.records["issue"])[
        res.records["status"] == 1
    ]
    if len(lat):
        out["latency_us"] = {
            "mean": round(float(lat.mean()) / res.mhz, 3),
            "min": round(float(lat.min()) / res.mhz, 3),
            "max": round(float(lat.max()) / res.mhz, 3),
        }
    states = [rec.core_state(res.state, c) for c in range(workers)]
    out["cores_magic_ok"] = sum(c["magic_ok"] for c in states)
    out["cores_ended"] = {
        k: sum(c[k] != 0 for c in states)
        for k in ("reader_end", "writer_end", "compute_end")
    }
    out["compute_wrong"] = sum(sum(c["compute_wrong"]) for c in states)
    passes = sum(c["passes"] for c in states)
    out["next_cycles_per_pass"] = round(
        sum(c["next_cycles"] for c in states) / max(passes, 1), 1
    )
    gave = [
        {"core": c["core"], "reader": c["reader"], "writer": c["writer"]}
        for c in states
        if c["reader"] or c["writer"]
    ]
    out["gave_up_cores"] = len(gave)
    out["gave_up_where"] = {
        w: sum(
            1 for g in gave for k in ("reader", "writer") if g[k] and g[k]["where"] == w
        )
        for w in sorted(
            {g[k]["where"] for g in gave for k in ("reader", "writer") if g[k]}
        )
    }
    if s["status"] == "timeout":
        first, count = teams[s["index"]]
        team = range(first, first + count)
        out["timed_out"] = {"slot": s["slot"], "seq": s["seq"], "team": [first, count]}
        out["not_reported"] = [
            c for c in team if states[c]["reported"][s["slot"]] != s["seq"]
        ]
        out["named"] = [
            {"core": g["core"], **{k: g[k] for k in ("reader", "writer") if g[k]}}
            for g in gave
            if any(
                g[k] and g[k]["where"] in ("stall", "complete")
                for k in ("reader", "writer")
            )
        ]
    if args.trace and res.trace is not None:
        phases = trace_phases(res, teams, n)
        out["trace_us"] = summarize_phases(phases)
        out["overlap"] = overlaps(res, teams, n)

    if args.seg == "map" and s["status"] == "done":
        out.update(check(args, teams, base, outs, [x_exact, x_rand]))
    return out


def check(args, teams, base, outs, xs):
    pages = args.out_pages * len(outs)
    got = []
    for i, t in enumerate(outs):
        got.append(ttnn.to_torch(t).float().numpy())
        fed.log(f"output {i + 1} of {len(outs)} read back")
    got = np.concatenate(got)
    got = got.reshape(pages, TILE, TILE)
    fed.log("outputs read back")
    jobs = [j for j in checked_jobs(teams, base, pages) if j != args.stall]
    refs = {}
    bad, rel_max, exact_jobs = [], 0.0, 0
    for j in jobs:
        _, count = teams[j]
        key = (j % 2, count)
        if key not in refs:
            refs[key] = member_sums(xs[j % 2].astype(np.float64), count)
        ref = refs[key]
        mine = got[base[j] : base[j] + count].astype(np.float64)
        if j % 2 == 0:
            exact_jobs += 1
            if not np.array_equal(mine, ref):
                bad.append(
                    {
                        "job": j,
                        "team": teams[j],
                        "base": base[j],
                        "wrong_members": int((mine != ref).any((1, 2)).sum()),
                    }
                )
        else:
            total, ref_total = mine.sum(0), ref.sum(0)
            rel = float(
                np.sqrt(((total - ref_total) ** 2).mean() / (ref_total**2).mean())
            )
            rel_max = max(rel_max, rel)
            if rel > 1e-2:
                bad.append(
                    {"job": j, "team": teams[j], "base": base[j], "rel_rms": rel}
                )
    return {
        "jobs_checked": len(jobs),
        "exact_jobs_checked": exact_jobs,
        "jobs_bad": len(bad),
        "first_bad": bad[:5],
        "random_total_rel_rms_max": float(f"{rel_max:.3e}"),
    }


if __name__ == "__main__":
    main()
