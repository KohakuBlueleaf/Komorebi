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
T0 = time.monotonic()


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def read_bf16_chunked(session, tensor, name):
    """A bfloat16 tile tensor as session.read_bf16 gives it, read in chunks of 2 MB
    with a line per chunk: a whole read of an output takes a minute or more."""
    total, parts = tensor.pages, []
    for first, pages in session.read_chunks(tensor):
        parts.append(pages)
        log(f"{name}: {first + len(pages)} of {total} pages read")
    rows, cols = tensor.shape
    return dram.untilize_bf16(np.concatenate(parts), rows, cols)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--handler", choices=["map", "nop"], default="map")
    p.add_argument(
        "--groups", choices=["fixed", "random", "pair", "rotate"], default="fixed"
    )
    p.add_argument("--count", type=int, default=8, help="group size (fixed, rotate)")
    p.add_argument("--first", type=int, default=0, help="first core (fixed)")
    p.add_argument("--tiles", type=int, default=64, help="input tiles T")
    p.add_argument("--in-tiles", type=int, default=512, help="tiles per input tensor")
    p.add_argument(
        "--submit",
        choices=["many", "single"],
        default="many",
        help="submit_many or not",
    )
    p.add_argument("--depth", type=int, default=4, help="ring slots used")
    p.add_argument("--out-pages", type=int, default=2048, help="pages per output")
    p.add_argument("--out-tensors", type=int, default=1, help="output tensors")
    p.add_argument("--stall", type=int, default=-1, help="job index made a STALL")
    p.add_argument("--stall-rank", type=int, default=0)
    p.add_argument("--deadline", type=float, default=1.0, help="per command, seconds")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--batch", type=int, default=64, help="commands per host store")
    p.add_argument(
        "--settle-microseconds", type=float, default=1.0, help="pause after a fence"
    )
    p.add_argument("--no-check", action="store_true", help="skip reading outputs")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args(argv)


def make_groups(args, cores, rng):
    """(first, count) per job. rotate: disjoint groups of --count cores in turn."""
    if args.groups == "fixed":
        return [(args.first, args.count)] * args.jobs
    if args.groups == "pair":
        half = cores // 2
        return [
            (0, half) if j % 2 == 0 else (half, cores - half) for j in range(args.jobs)
        ]
    if args.groups == "rotate":
        slots = max(1, cores // args.count)
        return [((j % slots) * args.count, args.count) for j in range(args.jobs)]
    groups = []
    for _ in range(args.jobs):
        count = int(rng.integers(1, cores + 1))
        groups.append((int(rng.integers(0, cores - count + 1)), count))
    return groups


def place_outputs(groups, pages, tensors):
    """Global output page of rank 0 per job: a ring of `tensors` x `pages` pages in
    which no job crosses a tensor."""
    base, at = [], 0
    for _, count in groups:
        if at % pages + count > pages:
            at = (at // pages + 1) * pages
        if at >= tensors * pages:
            at = 0
        base.append(at)
        at += count
    return base


def checked_jobs(groups, base, pages):
    """Jobs whose output pages no later job overwrote."""
    owner = np.full(pages, -1)
    for j, ((_, count), b) in enumerate(zip(groups, base)):
        owner[b : b + count] = j
    return [
        j
        for j, ((_, c), b) in enumerate(zip(groups, base))
        if (owner[b : b + c] == j).all()
    ]


def input_pages(jobs, tiles, in_tiles):
    """First input page per job: the window moves with the job number, so a tile left
    by an earlier job differs from the right one (the check counts where not)."""
    return (np.arange(jobs) * 61) % (in_tiles - tiles + 1)


def rank_sums(prefix, start, tiles, count):
    """Float64 sum of each core's partition of tiles [start, start + tiles) of an input,
    from its prefix sums over tiles: [count, 32, 32]."""
    out = np.zeros((count, TILE, TILE))
    for m in range(count):
        part = partition(range(start, start + tiles), m, count)
        out[m] = prefix[part.stop] - prefix[part.start]
    return out


def ranks_of(groups, cores):
    """Commands each core is in."""
    edge = np.zeros(cores + 1, np.int64)
    first = np.array([f for f, _ in groups], np.int64)
    end = first + np.array([c for _, c in groups], np.int64)
    np.add.at(edge, first, 1)
    np.add.at(edge, end, -1)
    return np.cumsum(edge)[:cores]


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    out = {"jobs": args.jobs, "handler": args.handler, "groups": args.groups}
    with km.Session(log=log) as session:
        log("device open")
        out.update(run(session, args, rng))
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["ok"] else 1)


def run(session, args, rng):
    gx, gy = session.grid
    cores = gx * gy
    groups = make_groups(args, cores, rng)
    n, tiles = args.jobs, args.tiles

    # Exact inputs: integers in [-2, 2], every partial sum exact in bf16 for T <= 64.
    # Random inputs: uniform in [-1, 1]. Job j reads T tiles from page in_page[j] on.
    in_tiles = max(args.in_tiles, tiles)
    x_exact = rng.integers(-2, 3, (in_tiles * TILE, TILE)).astype(np.float32)
    x_rand = rng.uniform(-1, 1, (in_tiles * TILE, TILE)).astype(np.float32)
    x_rand = formats.bf16_values(formats.bf16_bits(x_rand))

    def tile_tensor(a):
        return session.tensor(a, "bfloat16", "tile")

    inputs = [tile_tensor(x_exact), tile_tensor(x_rand)]
    pages = args.out_pages
    outs = [tile_tensor(np.zeros((pages * TILE, TILE), np.float32))]
    outs += [tile_tensor(np.zeros((pages * TILE, TILE), np.float32))
             for _ in range(args.out_tensors - 1)]  # fmt: skip
    base = place_outputs(groups, pages, args.out_tensors)
    in_page = input_pages(n, tiles, in_tiles)

    program = km.Program(
        "mapcollect",
        kernels=KERNELS,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),  # kCbIn
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),  # kCbOut
        ],
        compile_args={
            "reader": inputs[0].accessor_args(),
            "writer": outs[0].accessor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        depth=args.depth,
        handlers=KERNELS / "mapcollect.h",
    )
    run = session.load(
        program,
        trace=n + 1 if args.trace else 0,
        deadline=args.deadline,
        batch=args.batch,
        settle_microseconds=args.settle_microseconds,
    )
    log(f"loaded: {cores} cores, depth {args.depth}")

    # MAP arguments per job: input, T, output, output page of rank 0, no slow
    # core (a4 0 cycles, a5 none), first input page
    in_addr = np.array([t.address for t in inputs], np.int64)
    out_addr = np.array([t.address for t in outs], np.int64)
    jobs = np.arange(n)
    base_a = np.asarray(base, np.int64)
    words = np.stack(
        [in_addr[jobs % 2], np.full(n, tiles), out_addr[base_a // pages],
         base_a % pages, np.zeros(n, np.int64), np.full(n, 0xFFFFFFFF), in_page]
    ).T  # fmt: skip
    handlers = np.full(n, 0 if args.handler == "nop" else program.handler("map"))
    if 0 <= args.stall < n:
        handlers[args.stall] = program.handler("stall")
        words[args.stall] = [args.stall_rank, 0, 0, 0, 0, 0, 0]
    firsts = np.array([f for f, _ in groups])
    counts = np.array([c for _, c in groups])

    handles, timed_out = [], None
    t0 = time.perf_counter()
    try:
        if args.submit == "many":
            handles = list(run.submit_many(handlers, firsts, counts, words))
        else:
            rows = words.tolist()
            for j, (first, count) in enumerate(groups):
                arg = rows[j] if handlers[j] else []
                handles.append(run.submit(int(handlers[j]), (first, count), arg))
        waited = run.drain(timeout=60.0)
        host_s = time.perf_counter() - t0
        log(f"{waited} events")
        run.stop()
    except km.CommandTimeout as err:
        host_s = time.perf_counter() - t0
        timed_out = err
        log(f"timeout: {err}")
    result = run.result()
    out = {
        "cores": cores,
        "depth": args.depth,
        "count": args.count if args.groups in ("fixed", "rotate") else None,
        "tiles": tiles,
        "kernel": {
            k: result[k] for k in ("error_name", "commands", "events", "max_busy")
        },
        "microseconds_per_job_kernel": round(
            result["last"] / 50 / max(result["commands"], 1), 3
        ),
        "microseconds_per_job_host": round(host_s * 1e6 / n, 3),
        "submit": args.submit,
        "host_microseconds_per_command": {
            "submit": round(run.submit_s * 1e6 / n, 3),
            "wait_python": round((run.wait_s - run.native_s) * 1e6 / n, 3),
            "wait_native": round(run.native_s * 1e6 / n, 3),
        },
    }
    done = [
        run.event(t) for t in handles if run.event(t) and run.event(t).kind == "done"
    ]
    if done:
        us = np.array([e.microseconds for e in done])
        out["latency_microseconds"] = {
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
    if timed_out is None:
        out.update(core_counts(states, groups, cores))
    if timed_out is not None:
        out["timed_out"] = {
            "handle": timed_out.handle,
            "slot": timed_out.slot,
            "sequence": timed_out.sequence,
            "group": list(timed_out.group),
            "not_reported": timed_out.cores,
        }
    trace = run.trace()
    if trace is not None:
        out.update(trace_summary(trace, handles, groups))
    if args.handler == "map" and timed_out is None and not args.no_check:
        out.update(check(session, args, groups, base, in_page, outs, [x_exact, x_rand]))
    stalled = None
    if 0 <= args.stall < n:
        stalled = groups[args.stall][0] + args.stall_rank
    out["ok"] = verdict(out, len(states), stalled)
    return out


def verdict(out, cores, stalled=None):
    """Every core ended, no compute check failed, no job wrong; without a stall, no
    kernel error, no core gave up and every core's counts right; with one, the
    deadline named the stalled core (`stalled`) among those that did not report."""
    ended = set(out["cores_ended"].values()) == {cores}
    ok = ended and out["cores_magic_ok"] == cores and out["compute_wrong"] == 0
    ok &= out.get("jobs_bad", 0) == 0
    if stalled is None:
        ok &= "timed_out" not in out and out["kernel"]["error_name"] == "none"
        return bool(ok and out["gave_up_cores"] == 0 and out["core_counts_bad"] == 0)
    t = out.get("timed_out")
    named = [c["core"] for c in t["not_reported"]] if t else []
    return bool(ok and out["kernel"]["error_name"] == "deadline" and stalled in named)


def core_counts(states, groups, cores):
    """Each core's commands taken (EXIT too) and finished (EXIT not) against the groups
    that hold it."""
    rank = ranks_of(groups, cores)
    runs = np.array([c["runs"] for c in states])
    done = np.array([c["completed"] for c in states])
    bad = np.flatnonzero((runs != rank + 1) | (done != rank))
    return {
        "rank_commands": int(rank.sum()),
        "core_counts_bad": len(bad),
        "first_bad_cores": [
            {"core": int(c), "rank": int(rank[c]), "runs": int(runs[c]),
             "completed": int(done[c])}
            for c in bad[:5]
        ],  # fmt: skip
    }


def trace_summary(trace, handles, groups):
    """Mean per-phase times, and how many consecutive disjoint pairs overlapped."""
    phases = [p for p in map(trace.phases, handles) if p]
    keys = [k for k in phases[0] if k != "count"] if phases else []
    leads = []
    for j in range(len(handles) - 1):
        (fa, ca), (fb, cb) = groups[j], groups[j + 1]
        if fa + ca <= fb or fb + cb <= fa:
            lead = trace.lead(handles[j], handles[j + 1])
            if lead is not None and lead > 0:
                leads.append(lead)
    return {
        "trace_microseconds": {
            k: round(float(np.mean([p[k] for p in phases])), 3) for k in keys
        },
        "overlap": {
            "pairs": len(handles) - 1,
            "overlapping": len(leads),
            "lead_microseconds_mean": round(float(np.mean(leads)), 3) if leads else 0.0,
        },
    }


def check(session, args, groups, base, in_page, outs, xs):
    """Every core's tile of every job not overwritten: bit-exact on exact inputs,
    relative RMS of the total on random inputs. Counts the tiles that equal what the
    page held before (zeros or an earlier job's): there a stale tile would pass."""
    got = [read_bf16_chunked(session, t, f"output {i}") for i, t in enumerate(outs)]
    got = np.concatenate(got).reshape(-1, TILE, TILE)
    log(f"{len(outs)} outputs read back")
    prefix = []
    for x in xs:
        sums = np.cumsum(x.reshape(-1, TILE, TILE).astype(np.float64), 0)
        prefix.append(np.concatenate([np.zeros((1, TILE, TILE)), sums]))
    tiles = args.tiles

    def ref(j):
        count = groups[j][1]
        if j == args.stall:
            return None
        return rank_sums(prefix[j % 2], int(in_page[j]), tiles, count)

    writers = [[] for _ in range(len(got))]  # jobs per page, in order
    for j, ((_, count), b) in enumerate(zip(groups, base)):
        for p in range(b, b + count):
            writers[p].append(j)
    jobs = [j for j in checked_jobs(groups, base, len(got)) if j != args.stall]
    bad, rel_max, exact, same_as_before, empty = [], 0.0, 0, 0, 0
    for j in jobs:
        count = groups[j][1]
        want = ref(j)
        mine = got[base[j] : base[j] + count].astype(np.float64)
        for m in range(count):
            page = base[j] + m
            earlier = writers[page][-2] if len(writers[page]) > 1 else None
            before = np.zeros((TILE, TILE))
            if earlier is not None and earlier != args.stall:
                before = ref(earlier)[page - base[earlier]]
            if not partition(range(tiles), m, count):
                empty += 1  # no input tiles: zeros, as the page held
            elif np.array_equal(want[m], before):
                same_as_before += 1
        if j % 2 == 0:
            exact += 1
            if not np.array_equal(mine, want):
                bad.append({"job": j, "group": groups[j]})
        else:
            total, ref_total = mine.sum(0), want.sum(0)
            rel = float(
                np.sqrt(((total - ref_total) ** 2).mean() / (ref_total**2).mean())
            )
            rel_max = max(rel_max, rel)
            if rel > 1e-2:
                bad.append({"job": j, "group": groups[j], "rel_rms": rel})
    return {
        "jobs_checked": len(jobs),
        "jobs_overwritten": len(groups) - len(jobs) - (0 <= args.stall < len(groups)),
        "exact_jobs_checked": exact,
        "jobs_bad": len(bad),
        "first_bad": bad[:5],
        "random_total_rel_rms_max": float(f"{rel_max:.3e}"),
        "tiles_same_as_before": same_as_before,
        "tiles_without_input": empty,
    }


if __name__ == "__main__":
    main()
