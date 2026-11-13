"""Merge tree on a card: commands go through the dispatcher on X280 cluster 0;
every job's total and its result word are checked against numpy. One JSON line.
See README.md (`merge_fed.py`: the same jobs under the test feeder).
"""

import argparse
import json
import sys
import time

import merge_fed
import numpy as np

import komorebi as km

KERNELS = merge_fed.KERNELS
MERGE_CONSTANTS = km.abi.constants(
    KERNELS / "merge.h", known=km.abi.headers().constants
)
TILE, NONE = merge_fed.TILE, merge_fed.NONE
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--groups", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--count", type=int, default=8, help="group size (fixed)")
    parser.add_argument("--first", type=int, default=0, help="first core (fixed)")
    parser.add_argument(
        "--group-list", help="first:count per job, comma-separated (overrides --groups)"
    )
    parser.add_argument(
        "--form",
        choices=["owner", "tree", "both", "random"],
        default="both",
        help="both: owner and tree alternate in pairs (each group runs both)",
    )
    parser.add_argument("--tiles", type=int, default=64, help="input tiles T")
    parser.add_argument("--depth", type=int, default=4, help="ring slots used")
    parser.add_argument("--out-pages", type=int, default=256, help="output tiles")
    parser.add_argument("--part", type=int, default=1, help="tiles per part P (1 to 3)")
    parser.add_argument("--wait", type=float, default=0.5, help="wait limit, seconds")
    parser.add_argument(
        "--deadline", type=float, default=0.05, help="per command, seconds"
    )
    parser.add_argument(
        "--stall-rank",
        type=int,
        default=-1,
        help="then a load of four jobs whose job 1 has rank M stall, and a clean one",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.part <= MERGE_CONSTANTS["kPartMax"]:
        parser.error(f"--part: 1 to {MERGE_CONSTANTS['kPartMax']}")
    return args


def program(args, input_tensor, output_tensor):
    page = MERGE_CONSTANTS["kTileBytes"]
    return km.Program(
        "merge_tree",
        kernels=KERNELS,
        buffers=[
            km.Buffer(
                MERGE_CONSTANTS["kCbIn"], "float32", page, MERGE_CONSTANTS["kInPages"]
            ),
            km.Buffer(
                MERGE_CONSTANTS["kCbArrive"],
                "float32",
                page,
                MERGE_CONSTANTS["kArrivePages"],
            ),
            km.Buffer(
                MERGE_CONSTANTS["kCbOut"], "float32", page, MERGE_CONSTANTS["kOutPages"]
            ),
        ],
        compile_args={
            "reader": input_tensor.accessor_args(),
            "writer": output_tensor.accessor_args(),
        },
        # the compute kernel copies float32 tiles into DST unrounded
        compute={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": "UnpackToDestFp32",
        },
        depth=args.depth,
        handlers=KERNELS / "merge.h",
    )


def gave_up(states):
    """Cores that gave up anywhere but waiting for a command (with the port a reader or
    writer waited on)."""
    cores_gave_up = []
    for state in states:
        gave_up_in = {
            key: state[key] for key in ("reader", "writer", "compute") if state[key]
        }
        if any(record["where"] != "next" for record in gave_up_in.values()):
            cores_gave_up.append({"core": state["core"], **gave_up_in})
    return cores_gave_up


def launch(session, args, jobs, inputs, output, stall=-1, stall_rank=NONE):
    """One load: the jobs (each job writes output part job % parts), each asking for its
    result word; job `stall` has rank `stall_rank` stall, under a deadline long
    enough for its receiver's wait to give up first. Then the stop and the state."""
    run = session.load(
        program(args, inputs[0], output),
        deadline=args.deadline if stall < 0 else 4 * args.wait,
        wait=args.wait,
    )
    log(f"loaded, {len(jobs)} jobs")
    handles, timed_out = [], None
    start = time.perf_counter()
    try:
        for job, (first, count, form, source) in enumerate(jobs):
            words = [
                inputs[source % 2].address,
                args.tiles,
                output.address,
                job % (args.out_pages // args.part) * args.part,
                form,
                stall_rank if job == stall else NONE,
                source,
                args.part,
            ]
            group = (first, count)
            handles.append(
                run.submit("merge", group, words, exclusive=True, result=True)
            )
        log(f"{run.drain(timeout=30.0)} events")
        host_seconds = time.perf_counter() - start
        run.stop()
    except km.CommandTimeout as error:
        host_seconds = time.perf_counter() - start
        timed_out = error
        log(f"timeout: {error}")
    events = [run.event(handle) for handle in handles]
    states = run.state()
    summary = {
        "kernel": {
            key: run.result()[key] for key in ("error_name", "commands", "events")
        },
        "done": sum(event is not None and event.kind == "done" for event in events),
        "microseconds_per_job_host": round(host_seconds * 1e6 / len(jobs), 3),
        "cores_ended": {
            key: sum(state[key] != 0 for state in states)
            for key in ("reader_end", "writer_end", "compute_end")
        },
        "compute_wrong": sum(sum(state["compute_wrong"]) for state in states),
        "gave_up": gave_up(states)[:8],
        "waiting_ports": {
            state["core"]: {
                port: port_state["want"] - port_state["count"]
                for port, port_state in enumerate(state["ports"])
                if port_state["want"]
            }
            for state in states
            if any(port_state["want"] for port_state in state["ports"])
        },
    }
    if timed_out is not None:
        job = handles.index(timed_out.handle)
        summary["timed_out"] = {
            "handle": timed_out.handle,
            "job": job,
            "group": list(timed_out.group),
            "form": jobs[job][2],
            "not_reported": [core["core"] for core in timed_out.cores],
        }
    return events, summary


def checksum(tiles):
    """The result word the kernel returns: the float32 words of the tiles, summed."""
    words = np.asarray(tiles, np.float32).view(np.uint32).astype(np.uint64)
    return int(words.sum() & 0xFFFFFFFF)


def check_results(args, jobs, events, output, input_arrays):
    """Each finished job's result word against the replay (exact inputs: numpy's
    total; random inputs: the total read back, as long as no later job reused it)."""
    got = output.read().reshape(-1, TILE, TILE)
    parts = args.out_pages // args.part
    last = {job % parts: job for job in range(len(jobs))}
    bad, checked = [], 0
    for job, event in enumerate(events):
        if event is None or event.kind != "done":
            continue
        source, page = jobs[job][3], job % parts * args.part
        if source % 2 == 0:
            want = checksum(merge_fed.reference(args, source, input_arrays))
        elif last[job % parts] == job:
            want = checksum(got[page : page + args.part])
        else:
            continue
        checked += 1
        if event.result != want:
            bad.append({"job": job, "result": event.result, "want": want})
    return {"results_checked": checked, "results_bad": len(bad), "first": bad[:3]}


def checked(args, jobs, events, output, input_arrays):
    done = np.array([event is not None and event.kind == "done" for event in events])
    return {
        **merge_fed.check(args, jobs, done, output, input_arrays),
        **check_results(args, jobs, events, output, input_arrays),
    }


def run(session, args, generator):
    grid_width, grid_height = session.grid
    # random groups fit the arrival buffer by owner: one slot per other rank
    jobs = merge_fed.make_jobs(
        args,
        min(grid_width * grid_height, MERGE_CONSTANTS["kArrivePages"] // args.part + 1),
        generator,
    )
    log(f"grid {grid_width} x {grid_height}, {len(jobs)} jobs")

    # Exact: integers in [-2, 2] (every partial sum exact in float32). Random: uniform
    # in [-1, 1]. Sources differ, so the totals do.
    rows = (args.tiles + max(job[3] for job in jobs) + 1) * TILE
    input_arrays = [
        generator.integers(-2, 3, (rows, TILE)).astype(np.float32),
        generator.uniform(-1, 1, (rows, TILE)).astype(np.float32),
    ]

    def tiles(array):
        return session.tensor(array, "float32", "tile")

    inputs = [tiles(array) for array in input_arrays]
    output = tiles(np.zeros((args.out_pages * TILE, TILE), dtype=np.float32))
    events, summary = launch(session, args, jobs, inputs, output)
    result = {
        "jobs": len(jobs),
        "tiles": args.tiles,
        "part": args.part,
        "form": args.form,
        **summary,
        **checked(args, jobs, events, output, input_arrays),
    }
    if args.stall_rank >= 0:  # a core stalls in job 1; then a clean load
        few = jobs[:4]
        _, result["stall"] = launch(
            session, args, few, inputs, output, 1, args.stall_rank
        )
        events, after = launch(session, args, few, inputs, output)
        result["after"] = {**after, **checked(args, few, events, output, input_arrays)}
    result["ok"] = verdict(result, len(jobs), grid_width * grid_height)
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
    """The jobs clean; with --stall-rank, the stalled load ended in the deadline on
    job 1 with every core ended, and the load after it clean."""
    ok = clean(result, jobs, cores)
    if "stall" in result:
        stalled = result["stall"]
        ok &= ended(stalled, cores) and stalled["kernel"]["error_name"] == "deadline"
        ok &= stalled.get("timed_out", {}).get("job") == 1
        ok &= clean(result["after"], min(4, jobs), cores)
    return bool(ok)


def main():
    args = parse()
    generator = np.random.default_rng(args.seed)
    with km.Session(log=log) as session:
        log("device open")
        result = run(session, args, generator)
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
