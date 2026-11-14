"""Map/collect on persistent kernels under the dispatcher on X280 cluster 0: each core
of a MAP command's group adds its partition of T input tiles into one tile; the host
checks every tile against numpy and prints one JSON line. See README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import dram, formats
from komorebi.layout import partition

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
TILE = 32
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def read_bf16_chunked(session, tensor, name):
    """A bfloat16 tile tensor as session.read_bf16 gives it, read in chunks of 2 MB
    with a line per chunk: a whole read of an output takes a minute or more."""
    total, parts = tensor.pages, []
    for first, pages in session.read_chunks(tensor):
        parts.append(pages)
        log(f"{name}: {first + len(pages)} of {total} pages read")
    rows, columns = tensor.shape
    return dram.untilize_bf16(np.concatenate(parts), rows, columns)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--handler", choices=["map", "nop"], default="map")
    parser.add_argument(
        "--groups", choices=["fixed", "random", "pair", "rotate"], default="fixed"
    )
    parser.add_argument(
        "--count", type=int, default=8, help="group size (fixed, rotate)"
    )
    parser.add_argument("--first", type=int, default=0, help="first core (fixed)")
    parser.add_argument("--tiles", type=int, default=64, help="input tiles T")
    parser.add_argument(
        "--in-tiles", type=int, default=512, help="tiles per input tensor"
    )
    parser.add_argument(
        "--submit",
        choices=["many", "single"],
        default="many",
        help="submit_many or not",
    )
    parser.add_argument("--depth", type=int, default=4, help="ring slots used")
    parser.add_argument("--out-pages", type=int, default=2048, help="pages per output")
    parser.add_argument("--out-tensors", type=int, default=1, help="output tensors")
    parser.add_argument("--stall", type=int, default=-1, help="job index made a STALL")
    parser.add_argument("--stall-rank", type=int, default=0)
    parser.add_argument(
        "--deadline", type=float, default=1.0, help="per command, seconds"
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--batch", type=int, default=64, help="commands per host store")
    parser.add_argument(
        "--settle-microseconds", type=float, default=1.0, help="pause after a fence"
    )
    parser.add_argument("--no-check", action="store_true", help="skip reading outputs")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def make_groups(args, cores, generator):
    """(first, count) per job. rotate: disjoint groups of --count cores in turn."""
    if args.groups == "fixed":
        return [(args.first, args.count)] * args.jobs
    if args.groups == "pair":
        half = cores // 2
        return [
            (0, half) if job % 2 == 0 else (half, cores - half)
            for job in range(args.jobs)
        ]
    if args.groups == "rotate":
        slots = max(1, cores // args.count)
        return [((job % slots) * args.count, args.count) for job in range(args.jobs)]
    groups = []
    for _ in range(args.jobs):
        count = int(generator.integers(1, cores + 1))
        groups.append((int(generator.integers(0, cores - count + 1)), count))
    return groups


def place_outputs(groups, pages, tensors):
    """Global output page of rank 0 per job: a ring of `tensors` x `pages` pages in
    which no job crosses a tensor."""
    first_pages, page_position = [], 0
    for _, count in groups:
        if page_position % pages + count > pages:
            page_position = (page_position // pages + 1) * pages
        if page_position >= tensors * pages:
            page_position = 0
        first_pages.append(page_position)
        page_position += count
    return first_pages


def checked_jobs(groups, first_pages, pages):
    """Jobs whose output pages no later job overwrote."""
    owner = np.full(pages, -1)
    for job, ((_, count), first_page) in enumerate(zip(groups, first_pages)):
        owner[first_page : first_page + count] = job
    return [
        job
        for job, ((_, count), first_page) in enumerate(zip(groups, first_pages))
        if (owner[first_page : first_page + count] == job).all()
    ]


def input_pages(jobs, tiles, input_tiles):
    """First input page per job: the window moves with the job number, so a tile left
    by an earlier job differs from the right one (the check counts where not)."""
    return (np.arange(jobs) * 61) % (input_tiles - tiles + 1)


def rank_sums(prefix, start, tiles, count):
    """Float64 sum of each core's partition of tiles [start, start + tiles) of an input,
    from its prefix sums over tiles: [count, 32, 32]."""
    sums = np.zeros((count, TILE, TILE))
    for rank in range(count):
        tile_range = partition(range(start, start + tiles), rank, count)
        sums[rank] = prefix[tile_range.stop] - prefix[tile_range.start]
    return sums


def ranks_of(groups, cores):
    """Commands each core is in."""
    edge = np.zeros(cores + 1, np.int64)
    first = np.array([start for start, _ in groups], np.int64)
    end = first + np.array([count for _, count in groups], np.int64)
    np.add.at(edge, first, 1)
    np.add.at(edge, end, -1)
    return np.cumsum(edge)[:cores]


def main():
    args = parse()
    generator = np.random.default_rng(args.seed)
    report = {"jobs": args.jobs, "handler": args.handler, "groups": args.groups}
    with km.Session(log=log) as session:
        log("device open")
        report.update(run(session, args, generator))
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


def run(session, args, generator):
    grid_width, grid_height = session.grid
    cores = grid_width * grid_height
    groups = make_groups(args, cores, generator)
    job_count, tiles = args.jobs, args.tiles

    # Exact inputs: integers in [-2, 2], every partial sum exact in bf16 for T <= 64.
    # Random inputs: uniform in [-1, 1]. Each job reads T tiles from first_tiles[job].
    input_tiles = max(args.in_tiles, tiles)
    exact_input = generator.integers(-2, 3, (input_tiles * TILE, TILE)).astype(
        np.float32
    )
    random_input = generator.uniform(-1, 1, (input_tiles * TILE, TILE)).astype(
        np.float32
    )
    random_input = formats.bf16_values(formats.bf16_bits(random_input))

    def tile_tensor(array):
        return session.tensor(array, "bfloat16", "tile")

    inputs = [tile_tensor(exact_input), tile_tensor(random_input)]
    pages = args.out_pages
    outputs = [tile_tensor(np.zeros((pages * TILE, TILE), np.float32))]
    outputs += [tile_tensor(np.zeros((pages * TILE, TILE), np.float32))
                for _ in range(args.out_tensors - 1)]  # fmt: skip
    first_pages = place_outputs(groups, pages, args.out_tensors)
    first_tiles = input_pages(job_count, tiles, input_tiles)

    program = km.Program(
        "mapcollect",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),  # kCbIn
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),  # kCbOut
        ],
        compile_args={
            "reader": inputs[0].accessor_args(),
            "writer": outputs[0].accessor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        depth=args.depth,
        handlers=KERNELS / "mapcollect.h",
    )
    run = session.load(
        program,
        trace=job_count + 1 if args.trace else 0,
        deadline=args.deadline,
        batch=args.batch,
        settle_microseconds=args.settle_microseconds,
    )
    log(f"loaded: {cores} cores, depth {args.depth}")

    # MAP arguments per job, a0 to a6 as the handler's header names them (no slow core:
    # slow_cycles 0, slow_rank none).
    input_addresses = np.array([tensor.address for tensor in inputs], np.int64)
    output_addresses = np.array([tensor.address for tensor in outputs], np.int64)
    jobs = np.arange(job_count)
    first_page_array = np.asarray(first_pages, np.int64)
    words = np.stack(
        [
            input_addresses[jobs % 2],  # input_address
            np.full(job_count, tiles),  # tiles
            output_addresses[first_page_array // pages],  # output_address
            first_page_array % pages,  # first_page
            np.zeros(job_count, np.int64),  # slow_cycles
            np.full(job_count, 0xFFFFFFFF),  # slow_rank
            first_tiles,  # first_tile
        ]
    ).T
    handlers = np.full(
        job_count, 0 if args.handler == "nop" else program.handler("map")
    )
    if 0 <= args.stall < job_count:
        handlers[args.stall] = program.handler("stall")
        words[args.stall] = [args.stall_rank, 0, 0, 0, 0, 0, 0]
    firsts = np.array([first for first, _ in groups])
    counts = np.array([count for _, count in groups])

    handles, timed_out = [], None
    start = time.perf_counter()
    try:
        if args.submit == "many":
            handles = list(run.submit_many(handlers, firsts, counts, words))
        else:
            rows = words.tolist()
            for job, (first, count) in enumerate(groups):
                arg = rows[job] if handlers[job] else []
                handles.append(run.submit(int(handlers[job]), (first, count), arg))
        waited = run.drain(timeout=60.0)
        host_seconds = time.perf_counter() - start
        log(f"{waited} events")
        run.stop()
    except km.CommandTimeout as error:
        host_seconds = time.perf_counter() - start
        timed_out = error
        log(f"timeout: {error}")
    result = run.result()
    report = {
        "cores": cores,
        "depth": args.depth,
        "count": args.count if args.groups in ("fixed", "rotate") else None,
        "tiles": tiles,
        "kernel": {
            key: result[key] for key in ("error_name", "commands", "events", "max_busy")
        },
        "microseconds_per_job_kernel": round(
            result["last"] / 50 / max(result["commands"], 1), 3
        ),
        "microseconds_per_job_host": round(host_seconds * 1e6 / job_count, 3),
        "submit": args.submit,
        "host_microseconds_per_command": {
            "submit": round(run.submit_s * 1e6 / job_count, 3),
            "wait_python": round((run.wait_s - run.native_s) * 1e6 / job_count, 3),
            "wait_native": round(run.native_s * 1e6 / job_count, 3),
        },
    }
    done = [
        run.event(handle)
        for handle in handles
        if run.event(handle) and run.event(handle).kind == "done"
    ]
    if done:
        latencies = np.array([event.microseconds for event in done])
        report["latency_microseconds"] = {
            "mean": round(float(latencies.mean()), 3),
            "max": round(float(latencies.max()), 3),
        }
    states = run.state()
    report["cores_magic_ok"] = sum(state["magic_ok"] for state in states)
    report["cores_ended"] = {
        key: sum(state[key] != 0 for state in states)
        for key in ("reader_end", "writer_end", "compute_end")
    }
    report["compute_wrong"] = sum(sum(state["compute_wrong"]) for state in states)
    report["gave_up_cores"] = sum(
        1 for state in states if state["reader"] or state["writer"]
    )
    if timed_out is None:
        report.update(core_counts(states, groups, cores))
    report["recovery_refused"] = run.recovery_refused
    report["recoveries"] = [
        {key: recovery[key] for key in ("handle", "cores", "text")}
        for recovery in run.recoveries
    ]
    if timed_out is not None:
        report["timed_out"] = {
            "handle": timed_out.handle,
            "slot": timed_out.slot,
            "sequence": timed_out.sequence,
            "group": list(timed_out.group),
            "not_reported": timed_out.cores,
            "message": str(timed_out),
        }
    trace = run.trace()
    if trace is not None:
        report.update(trace_summary(trace, handles, groups))
    if args.handler == "map" and timed_out is None and not args.no_check:
        report.update(
            check(
                session,
                args,
                groups,
                first_pages,
                first_tiles,
                outputs,
                [exact_input, random_input],
            )
        )
    stalled = None
    if 0 <= args.stall < job_count:
        stalled = groups[args.stall][0] + args.stall_rank
    report["ok"] = verdict(report, len(states), stalled)
    return report


def verdict(report, cores, stalled=None):
    """Every core ended, no compute check failed, no job wrong; without a stall, no
    kernel error, no core gave up and every core's counts right; with one, the
    stalled core (`stalled`) recovered, or named by the deadline among those that did
    not report (no reset lists, or other commands in flight held it)."""
    ended = set(report["cores_ended"].values()) == {cores}
    ok = ended and report["cores_magic_ok"] == cores and report["compute_wrong"] == 0
    ok &= report.get("jobs_bad", 0) == 0
    if stalled is None:
        ok &= "timed_out" not in report and report["kernel"]["error_name"] == "none"
        return bool(
            ok and report["gave_up_cores"] == 0 and report["core_counts_bad"] == 0
        )
    if "timed_out" not in report:
        recovered = [core for entry in report["recoveries"] for core in entry["cores"]]
        ok &= "timed_out" not in report and report["kernel"]["error_name"] == "none"
        return bool(ok and stalled in recovered)
    timed_out = report.get("timed_out")
    named = [core["core"] for core in timed_out["not_reported"]] if timed_out else []
    return bool(
        ok and report["kernel"]["error_name"] == "deadline" and stalled in named
    )


def core_counts(states, groups, cores):
    """Each core's commands taken (EXIT too) and finished (EXIT not) against the groups
    that hold it."""
    rank = ranks_of(groups, cores)
    runs = np.array([state["runs"] for state in states])
    done = np.array([state["completed"] for state in states])
    bad = np.flatnonzero((runs != rank + 1) | (done != rank))
    return {
        "rank_commands": int(rank.sum()),
        "core_counts_bad": len(bad),
        "first_bad_cores": [
            {"core": int(core), "rank": int(rank[core]), "runs": int(runs[core]),
             "completed": int(done[core])}
            for core in bad[:5]
        ],  # fmt: skip
    }


def trace_summary(trace, handles, groups):
    """Mean per-phase times, and how many consecutive disjoint pairs overlapped."""
    phases = [phase for phase in map(trace.phases, handles) if phase]
    keys = [key for key in phases[0] if key != "count"] if phases else []
    leads = []
    for pair in range(len(handles) - 1):
        (first, count), (next_first, next_count) = groups[pair], groups[pair + 1]
        if first + count <= next_first or next_first + next_count <= first:
            lead = trace.lead(handles[pair], handles[pair + 1])
            if lead is not None and lead > 0:
                leads.append(lead)
    return {
        "trace_microseconds": {
            key: round(float(np.mean([phase[key] for phase in phases])), 3)
            for key in keys
        },
        "overlap": {
            "pairs": len(handles) - 1,
            "overlapping": len(leads),
            "lead_microseconds_mean": round(float(np.mean(leads)), 3) if leads else 0.0,
        },
    }


def check(session, args, groups, first_pages, first_tiles, outputs, input_arrays):
    """Every core's tile of every job not overwritten: bit-exact on exact inputs,
    relative RMS of the total on random inputs. Counts the tiles that equal what the
    page held before (zeros or an earlier job's): there a stale tile would pass."""
    got = [
        read_bf16_chunked(session, tensor, f"output {index}")
        for index, tensor in enumerate(outputs)
    ]
    got = np.concatenate(got).reshape(-1, TILE, TILE)
    log(f"{len(outputs)} outputs read back")
    prefix = []
    for values in input_arrays:
        sums = np.cumsum(values.reshape(-1, TILE, TILE).astype(np.float64), 0)
        prefix.append(np.concatenate([np.zeros((1, TILE, TILE)), sums]))
    tiles = args.tiles

    def expected_sums(job):
        count = groups[job][1]
        if job == args.stall:
            return None
        return rank_sums(prefix[job % 2], int(first_tiles[job]), tiles, count)

    writers = [[] for _ in range(len(got))]  # jobs per page, in order
    for job, ((_, count), first_page) in enumerate(zip(groups, first_pages)):
        for page in range(first_page, first_page + count):
            writers[page].append(job)
    jobs = [
        job for job in checked_jobs(groups, first_pages, len(got)) if job != args.stall
    ]
    bad, relative_error_max, exact, same_as_before, empty = [], 0.0, 0, 0, 0
    for job in jobs:
        count = groups[job][1]
        want = expected_sums(job)
        got_tiles = got[first_pages[job] : first_pages[job] + count].astype(np.float64)
        for rank in range(count):
            page = first_pages[job] + rank
            earlier = writers[page][-2] if len(writers[page]) > 1 else None
            before = np.zeros((TILE, TILE))
            if earlier is not None and earlier != args.stall:
                before = expected_sums(earlier)[page - first_pages[earlier]]
            if not partition(range(tiles), rank, count):
                empty += 1  # no input tiles: zeros, as the page held
            elif np.array_equal(want[rank], before):
                same_as_before += 1
        if job % 2 == 0:
            exact += 1
            if not np.array_equal(got_tiles, want):
                bad.append({"job": job, "group": groups[job]})
        else:
            total, want_total = got_tiles.sum(0), want.sum(0)
            relative_error = float(
                np.sqrt(((total - want_total) ** 2).mean() / (want_total**2).mean())
            )
            relative_error_max = max(relative_error_max, relative_error)
            if relative_error > 1e-2:
                bad.append(
                    {"job": job, "group": groups[job], "rel_rms": relative_error}
                )
    return {
        "jobs_checked": len(jobs),
        "jobs_overwritten": len(groups) - len(jobs) - (0 <= args.stall < len(groups)),
        "exact_jobs_checked": exact,
        "jobs_bad": len(bad),
        "first_bad": bad[:5],
        "random_total_rel_rms_max": float(f"{relative_error_max:.3e}"),
        "tiles_same_as_before": same_as_before,
        "tiles_without_input": empty,
    }


if __name__ == "__main__":
    main()
