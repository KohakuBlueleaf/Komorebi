"""Merge tree on a card: records go through the dispatch kernel on X280 cluster 0;
every job's total and its result word are checked against numpy. One JSON line.
See README.md (`merge_fed.py`: the same jobs under the test feeder).
"""

import argparse
import contextlib
import json
import time

import merge_fed as mf
import numpy as np
import torch
import ttnn

import komorebi as km

KERNELS = mf.KERNELS
M, K = mf.M, mf.K
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


@contextlib.contextmanager
def fp32_unpack():
    """Workaround: Program.compute has no unpack_to_dest_mode, and the compute kernel
    copies float32 tiles into DST unrounded; set here on the session's descriptor."""
    plain = ttnn.ComputeConfigDescriptor

    def config(**kwargs):
        c = plain(**kwargs)
        c.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        return c

    ttnn.ComputeConfigDescriptor = config
    try:
        yield
    finally:
        ttnn.ComputeConfigDescriptor = plain


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
            "reader": ttnn.TensorAccessorArgs(inp).get_compile_time_args(),
            "writer": ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        depth=args.depth,
        segments=KERNELS / "merge.h",
    )


def port_words(run):
    """[cores, KM_PORTS, 2] (count, wanted) of every port, after the stop. The port
    words are not in run.state(): read from the control blocks."""
    rows = ttnn.to_torch(run.backend.ctl).numpy().astype(np.int64) & 0xFFFFFFFF
    start = run.backend.ctl_offset + K["KM_CTL_PORT"] // 4
    stride = K["KM_PORT_STRIDE"] // 4
    words = rows[:, start : start + K["KM_PORTS"] * stride]
    words = words.reshape(len(words), K["KM_PORTS"], stride)
    return words[:, :, [K["KM_PORT_COUNT"], K["KM_PORT_WANT"]]]


def gave_up(run):
    """Cores that gave up anywhere but waiting for a record, with the port a reader
    or writer waited on (the top byte of its word, not decoded by run.state())."""
    words = run.state_words()
    keys = {"reader": "kStReader", "writer": "kStWriter", "compute": "kStCompute"}
    out = []
    for c in run.state():
        g = {k: dict(c[k]) for k in keys if c[k]}
        if not any(v["where"] != "next" for v in g.values()):
            continue
        for k, v in g.items():
            if v["where"] in ("port", "emit"):
                v["port"] = int(words[c["core"], K[keys[k]]]) >> 24
        out.append({"core": c["core"], **g})
    return out


def launch(session, args, jobs, inputs, out, stall=-1, stall_member=NONE):
    """One load: the jobs (job j writes output part j % parts), each asking for its
    result word; job `stall` has member `stall_member` stall, under a deadline long
    enough for its receiver's wait to give up first. Then the stop and the state."""
    with fp32_unpack():
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
                inputs[src % 2].buffer_address(),
                args.tiles,
                out.buffer_address(),
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
    ports = port_words(run)
    summary = {
        "kernel": {k: run.result()[k] for k in ("error_name", "records", "events")},
        "done": sum(e is not None and e.kind == "done" for e in events),
        "us_per_job_host": round(host_s * 1e6 / len(jobs), 3),
        "cores_ended": {
            k: sum(c[k] != 0 for c in states)
            for k in ("reader_end", "writer_end", "compute_end")
        },
        "compute_wrong": sum(sum(c["compute_wrong"]) for c in states),
        "gave_up": gave_up(run)[:8],
        "waiting_ports": {
            c: {p: int(w - n) for p, (n, w) in enumerate(ports[c]) if w}
            for c in range(len(ports))
            if ports[c, :, 1].any()
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
    got = ttnn.to_torch(out).float().numpy().reshape(-1, TILE, TILE)
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
    device = session.device
    g = device.compute_with_storage_grid_size()
    # random teams fit the arrival buffer by owner: one slot per other member
    jobs = mf.make_jobs(args, min(g.x * g.y, M["kArrivePages"] // args.part + 1), rng)
    log(f"grid {g.x} x {g.y}, {len(jobs)} jobs")

    # Exact: integers in [-2, 2] (every partial sum exact in float32). Random: uniform
    # in [-1, 1]. Sources differ, so the totals do.
    rows = (args.tiles + max(job[3] for job in jobs) + 1) * TILE
    xs = [
        rng.integers(-2, 3, (rows, TILE)).astype(np.float32),
        rng.uniform(-1, 1, (rows, TILE)).astype(np.float32),
    ]

    def tiles(a):
        return ttnn.from_torch(
            torch.from_numpy(a),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

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
    return result


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    with km.Session(log=log) as session:
        log("device open")
        result = run(session, args, rng)
    result["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
