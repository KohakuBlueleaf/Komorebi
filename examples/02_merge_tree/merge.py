"""Merge tree on a card: records go through the dispatch kernel on X280 cluster 0;
every job's total and its result word are checked against numpy. One JSON line.
See README.md (`merge_fed.py`: the same jobs under the test feeder).
"""

import argparse
import json
import sys
import time

import merge_fed as mf
import numpy as np

import komorebi as km

KERNELS = mf.KERNELS
M = km.abi.constants(KERNELS / "merge.h", known=km.abi.headers().constants)
TILE, NONE = mf.TILE, mf.NONE
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--teams", choices=["fixed", "random"], default="fixed")
    p.add_argument("--count", type=int, default=8, help="team size (fixed)")
    p.add_argument("--first", type=int, default=0, help="first core (fixed)")
    p.add_argument(
        "--team-list", help="first:count per job, comma-separated (overrides --teams)"
    )
    p.add_argument(
        "--form",
        choices=["owner", "tree", "both", "random"],
        default="both",
        help="both: owner and tree alternate in pairs (each team runs both)",
    )
    p.add_argument("--tiles", type=int, default=64, help="input tiles T")
    p.add_argument("--depth", type=int, default=4, help="ring slots used")
    p.add_argument("--out-pages", type=int, default=256, help="output tiles")
    p.add_argument("--part", type=int, default=1, help="tiles per part P (1 to 3)")
    p.add_argument("--wait", type=float, default=0.5, help="wait limit, seconds")
    p.add_argument("--deadline", type=float, default=0.05, help="per record, seconds")
    p.add_argument(
        "--stall-member",
        type=int,
        default=-1,
        help="then a load of four jobs whose job 1 has member M stall, and a clean one",
    )
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.part <= M["kPartMax"]:
        p.error(f"--part: 1 to {M['kPartMax']}")
    return args


def program(args, inp, out):
    page = M["kTileBytes"]
    return km.Program(
        "merge_tree",
        kernels=KERNELS,
        buffers=[
            km.Buffer(M["kCbIn"], "float32", page, M["kInPages"]),
            km.Buffer(M["kCbArrive"], "float32", page, M["kArrivePages"]),
            km.Buffer(M["kCbOut"], "float32", page, M["kOutPages"]),
        ],
        compile_args={
            "reader": inp.accessor_args(),
            "writer": out.accessor_args(),
        },
        # the compute kernel copies float32 tiles into DST unrounded
        compute={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": "UnpackToDestFp32",
        },
        depth=args.depth,
        segments=KERNELS / "merge.h",
    )


def gave_up(states):
    """Cores that gave up anywhere but waiting for a record (with the port a reader or
    writer waited on)."""
    out = []
    for c in states:
        g = {k: c[k] for k in ("reader", "writer", "compute") if c[k]}
        if any(v["where"] != "next" for v in g.values()):
            out.append({"core": c["core"], **g})
    return out


def launch(session, args, jobs, inputs, out, stall=-1, stall_member=NONE):
    """One load: the jobs (job j writes output part j % parts), each asking for its
    result word; job `stall` has member `stall_member` stall, under a deadline long
    enough for its receiver's wait to give up first. Then the stop and the state."""
    run = session.load(
        program(args, inputs[0], out),
        deadline=args.deadline if stall < 0 else 4 * args.wait,
        wait=args.wait,
    )
    log(f"loaded, {len(jobs)} jobs")
    tickets, timed_out = [], None
    t0 = time.perf_counter()
    try:
        for j, (first, count, form, src) in enumerate(jobs):
            words = [
                inputs[src % 2].address,
                args.tiles,
                out.address,
                j % (args.out_pages // args.part) * args.part,
                form,
                stall_member if j == stall else NONE,
                src,
                args.part,
            ]
            team = (first, count)
            tickets.append(run.issue("merge", team, words, exclusive=True, result=True))
        log(f"{run.drain(timeout=30.0)} events")
        host_s = time.perf_counter() - t0
        run.stop()
    except km.RecordTimeout as err:
        host_s = time.perf_counter() - t0
        timed_out = err
        log(f"timeout: {err}")
    events = [run.event(t) for t in tickets]
    states = run.state()
    summary = {
        "kernel": {k: run.result()[k] for k in ("error_name", "records", "events")},
        "done": sum(e is not None and e.kind == "done" for e in events),
        "us_per_job_host": round(host_s * 1e6 / len(jobs), 3),
        "cores_ended": {
            k: sum(c[k] != 0 for c in states)
            for k in ("reader_end", "writer_end", "compute_end")
        },
        "compute_wrong": sum(sum(c["compute_wrong"]) for c in states),
        "gave_up": gave_up(states)[:8],
        "waiting_ports": {
            c["core"]: {
                p: s["want"] - s["count"] for p, s in enumerate(c["ports"]) if s["want"]
            }
            for c in states
            if any(s["want"] for s in c["ports"])
        },
    }
    if timed_out is not None:
        j = tickets.index(timed_out.ticket)
        summary["timed_out"] = {
            "ticket": timed_out.ticket,
            "job": j,
            "team": list(timed_out.team),
            "form": jobs[j][2],
            "not_reported": [c["core"] for c in timed_out.cores],
        }
    return events, summary


def checksum(tiles):
    """The result word the kernel returns: the float32 words of the tiles, summed."""
    words = np.asarray(tiles, np.float32).view(np.uint32).astype(np.uint64)
    return int(words.sum() & 0xFFFFFFFF)


def check_results(args, jobs, events, out, xs):
    """Each finished job's result word against the replay (exact inputs: numpy's
    total; random inputs: the total read back, as long as no later job reused it)."""
    got = out.read().reshape(-1, TILE, TILE)
    parts = args.out_pages // args.part
    last = {j % parts: j for j in range(len(jobs))}
    bad, checked = [], 0
    for j, e in enumerate(events):
        if e is None or e.kind != "done":
            continue
        src, page = jobs[j][3], j % parts * args.part
        if src % 2 == 0:
            want = checksum(mf.reference(args, src, xs))
        elif last[j % parts] == j:
            want = checksum(got[page : page + args.part])
        else:
            continue
        checked += 1
        if e.result != want:
            bad.append({"job": j, "result": e.result, "want": want})
    return {"results_checked": checked, "results_bad": len(bad), "first": bad[:3]}


def checked(args, jobs, events, out, xs):
    done = np.array([e is not None and e.kind == "done" for e in events])
    return {
        **mf.check(args, jobs, done, out, xs),
        **check_results(args, jobs, events, out, xs),
    }


def run(session, args, rng):
    gx, gy = session.grid
    # random teams fit the arrival buffer by owner: one slot per other member
    jobs = mf.make_jobs(args, min(gx * gy, M["kArrivePages"] // args.part + 1), rng)
    log(f"grid {gx} x {gy}, {len(jobs)} jobs")

    # Exact: integers in [-2, 2] (every partial sum exact in float32). Random: uniform
    # in [-1, 1]. Sources differ, so the totals do.
    rows = (args.tiles + max(job[3] for job in jobs) + 1) * TILE
    xs = [
        rng.integers(-2, 3, (rows, TILE)).astype(np.float32),
        rng.uniform(-1, 1, (rows, TILE)).astype(np.float32),
    ]

    def tiles(a):
        return session.tensor(a, "float32", "tile")

    inputs = [tiles(x) for x in xs]
    out = tiles(np.zeros((args.out_pages * TILE, TILE), dtype=np.float32))
    events, summary = launch(session, args, jobs, inputs, out)
    result = {
        "jobs": len(jobs),
        "tiles": args.tiles,
        "part": args.part,
        "form": args.form,
        **summary,
        **checked(args, jobs, events, out, xs),
    }
    if args.stall_member >= 0:  # a member stalls in job 1; then a clean load
        few = jobs[:4]
        _, result["stall"] = launch(
            session, args, few, inputs, out, 1, args.stall_member
        )
        events, after = launch(session, args, few, inputs, out)
        result["after"] = {**after, **checked(args, few, events, out, xs)}
    result["ok"] = verdict(result, len(jobs), gx * gy)
    return result


def ended(load, cores):
    return set(load["cores_ended"].values()) == {cores} and load["compute_wrong"] == 0


def clean(load, jobs, cores):
    """Every job done and right (tiles and result words), no kernel error, no core gave
    up, no port left waiting, every core ended."""
    return (
        ended(load, cores)
        and load["kernel"]["error_name"] == "none"
        and load["done"] == jobs
        and load["jobs_bad"] == 0
        and load["results_bad"] == 0
        and not load["gave_up"]
        and not load["waiting_ports"]
    )


def verdict(result, jobs, cores):
    """The jobs clean; with --stall-member, the stalled load ended in the deadline on
    job 1 with every core ended, and the load after it clean."""
    ok = clean(result, jobs, cores)
    if "stall" in result:
        s = result["stall"]
        ok &= ended(s, cores) and s["kernel"]["error_name"] == "deadline"
        ok &= s.get("timed_out", {}).get("job") == 1
        ok &= clean(result["after"], min(4, jobs), cores)
    return bool(ok)


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    with km.Session(log=log) as session:
        log("device open")
        result = run(session, args, rng)
    result["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
