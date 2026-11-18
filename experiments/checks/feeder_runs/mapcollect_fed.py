"""The map/collect example's kernels fed by the test feeder on the last core (no X280);
every MAP core's tile is checked against numpy. One JSON line.

Example: python mapcollect_fed.py --jobs 20 --groups fixed --count 8"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

import komorebi as km  # noqa: E402
from komorebi import formats  # noqa: E402
from komorebi.tools.feeder import commands as feeder_commands  # noqa: E402
from komorebi.tools.feeder import fed  # noqa: E402

CONSTANTS = feeder_commands.CONSTANTS
KERNELS = HERE.parents[2] / "examples" / "01_mapcollect" / "kernels"
HANDLERS = feeder_commands.constants(KERNELS / "mapcollect.h")
HANDLER_MAP, HANDLER_STALL = HANDLERS["kHandlerMap"], HANDLERS["kHandlerStall"]
HANDLER_ALIGN, HANDLER_LATE = HANDLERS["kHandlerAlign"], HANDLERS["kHandlerLate"]
HANDLER_BAD_ITEM, HANDLER_RESULT = (
    HANDLERS["kHandlerBadItem"],
    HANDLERS["kHandlerResult"],
)
TILE = 32


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--handler", choices=["map", "nop", "align"], default="map")
    parser.add_argument(
        "--release", type=int, default=8000, help="align: cycles after submit"
    )
    parser.add_argument(
        "--groups", choices=["fixed", "random", "pair", "overlap"], default="fixed"
    )
    parser.add_argument("--count", type=int, default=8, help="group size (fixed)")
    parser.add_argument(
        "--group-list", help="first:count per job, comma-separated (overrides --groups)"
    )
    parser.add_argument("--first", type=int, default=0, help="first core (fixed)")
    parser.add_argument("--tiles", type=int, default=64, help="input tiles T")
    parser.add_argument("--depth", type=int, default=4, help="ring slots used")
    parser.add_argument("--out-pages", type=int, default=2048, help="pages per output")
    parser.add_argument("--out-tensors", type=int, default=1, help="output tensors")
    parser.add_argument("--stall", type=int, default=-1, help="job index made a STALL")
    parser.add_argument("--stall-rank", type=int, default=0)
    parser.add_argument("--late", type=int, default=-1, help="job index made a LATE")
    parser.add_argument("--late-rank", type=int, default=0)
    parser.add_argument("--late-cycles", type=int, default=0)
    parser.add_argument(
        "--slow", type=int, default=-1, help="MAP job with slow compute"
    )
    parser.add_argument(
        "--bad-item", type=int, default=-1, help="job index made a BAD_ITEM"
    )
    parser.add_argument("--bad-side", choices=["writer", "reader"], default="writer")
    parser.add_argument(
        "--result", type=int, default=-1, help="job made a flagged RESULT"
    )
    parser.add_argument("--result-value", type=int, default=0x5EED, help="its a0")
    parser.add_argument("--slow-rank", type=int, default=0)
    parser.add_argument("--slow-cycles", type=int, default=0)
    parser.add_argument("--wait", type=float, default=2.0, help="wait limit, seconds")
    parser.add_argument(
        "--exit-at", type=int, default=-1, help="job index made an EXIT (group of 1)"
    )
    parser.add_argument(
        "--deadline", type=float, default=0.05, help="per command, seconds"
    )
    parser.add_argument("--raw", type=int, default=-1, help="job index submitted raw")
    parser.add_argument("--raw-slot", type=int, default=0, help="slot its header names")
    parser.add_argument("--exit-group", default="", help="first:count kept on the EXIT")
    parser.add_argument(
        "--row", type=int, help="cores per row for completion (0: one level)"
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-cap", type=int, help="commands traced (default: all)")
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="KM_FLAG_EXCLUSIVE on every MAP command",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--dump", type=Path, help="save commands, trace and state (npz)"
    )
    return parser.parse_args(argv)


def make_groups(args, workers, rng):
    if args.group_list:
        groups = [
            tuple(map(int, text.split(":"))) for text in args.group_list.split(",")
        ]
        args.jobs = len(groups)
        return groups
    if args.groups == "fixed":
        return [(args.first, args.count)] * args.jobs
    if args.groups == "pair":
        half = workers // 2
        return [
            (0, half) if j % 2 == 0 else (half, workers - half)
            for j in range(args.jobs)
        ]
    if args.groups == "overlap":  # two halves that share a quarter, alternating
        half, quarter = workers // 2, workers // 4
        return [(0, half) if j % 2 == 0 else (quarter, half) for j in range(args.jobs)]
    return feeder_commands.random_groups(rng, args.jobs, workers)


def place_outputs(groups, pages, tensors):
    """Global output page of rank 0 per job: a ring of `tensors` x `pages` pages
    in which no job crosses a tensor."""
    base, position = [], 0
    for _, count in groups:
        if count > pages:
            raise ValueError(f"a group of {count} needs more than {pages} output pages")
        if position % pages + count > pages:
            position = (position // pages + 1) * pages
        if position >= tensors * pages:
            position = 0
        base.append(position)
        position += count
    return base


def checked_jobs(groups, base, pages):
    """Jobs whose output pages no later job overwrote."""
    owner = np.full(pages, -1)
    for j, ((_, count), offset) in enumerate(zip(groups, base)):
        owner[offset : offset + count] = j
    return [
        j
        for j, ((_, size), offset) in enumerate(zip(groups, base))
        if (owner[offset : offset + size] == j).all()
    ]


def rank_errors(args, groups, result, states):
    """Cores whose runs, completions, slot rejections or last seen and reported sequence
    differ from what the groups of the submitted and acknowledged commands give."""
    workers, command_records, summary = len(states), result.commands, result.summary
    runs, done = np.zeros((2, workers), int), np.zeros((2, workers), int)  # low, high
    rejected = np.zeros(workers, int)
    last = {}  # slot: (sequence, cores reporting) of the last command submitted there
    for j, (first, count) in enumerate(groups):
        sequence, slot, acked = (
            int(command_records["sequence"][j]),
            int(command_records["slot"][j]),
            command_records["status"][j] == 1,
        )
        is_exit = j == args.exit_at
        group = slice(0, workers) if is_exit else slice(first, first + count)
        if sequence == 0:
            continue
        if j == args.raw and args.raw_slot != slot:  # not run by the cores that see it
            rejected[group] += 1
            continue
        runs[1, group] += 1
        runs[0, group] += acked
        done[1, group] += not is_exit
        done[0, group] += acked and not is_exit
        if acked:
            last[slot] = (sequence, group)
    if summary["status"] == "done" and not 0 <= args.exit_at < len(groups):
        runs[1] += 1  # the feeder's own EXIT, on slot 0
        runs[0] += bool(summary["exit_acked"])
        last[0] = (summary["submitted"] + 1, slice(0, workers))
    bad = []
    for core_state in states:
        core = core_state["core"]
        ok = runs[0, core] <= core_state["runs"] <= runs[1, core]
        ok &= done[0, core] <= core_state["completed"] <= done[1, core]
        ok &= core_state["slot_wrong"] == rejected[core]
        if summary["status"] == "done":
            for slot, (sequence, group) in last.items():
                included = group.start <= core < group.stop
                ok &= (
                    core_state["last"][slot] == sequence
                    and (core_state["reported"][slot] == sequence) == included
                )
        if not ok:
            bad.append(core)
    return bad


def rank_sums(x, count):
    """Float64 sum of each core's partition of the tiles of x: [count, 32, 32]."""
    tiles = x.reshape(-1, TILE, TILE)
    out = np.zeros((count, TILE, TILE))
    for rank in range(count):
        start, length = feeder_commands.partition(len(tiles), rank, count)
        out[rank] = tiles[start : start + length].sum(0)
    return out


def trace_phases(result, groups, job_count):
    """Per job from the trace: microseconds of each phase (see the README)."""
    trace = result.trace.astype(np.int64)
    rows = result.commands
    mhz = result.mhz
    out = []
    for j in range(job_count):
        first, count = groups[j]
        sequence = int(rows["sequence"][j])
        if sequence == 0 or sequence > trace.shape[1]:
            continue
        timestamps = trace[first : first + count, sequence - 1]
        submit = int(rows["submit"][j])
        rel = feeder_commands.wrap(timestamps - submit)
        seen, done, sent = rel[:, 0], rel[:, 1], rel[:, 2]
        out.append(
            {
                "count": count,
                "first_seen": seen.min() / mhz,
                "last_seen": seen.max() / mhz,
                "last_done": done.max() / mhz,
                "last_rank_sent": (sent[1:].max() if count > 1 else sent[0]) / mhz,
                "leader_sent": sent[0] / mhz,
                "push_after_last_done": (sent[0] - done.max()) / mhz,
                "acked": int(feeder_commands.wrap(rows["acked"][j] - submit)) / mhz,
                "passes_mean": float(timestamps[:, 3].mean()),
            }
        )
    return out


def overlaps(result, groups, job_count):
    """Consecutive jobs on disjoint groups where a core of the second took its command
    before the first job's leader pushed its completion (wall clock, from the trace)."""
    trace = result.trace.astype(np.int64)
    seqs = result.commands["sequence"]
    found = []
    for j in range(job_count - 1):
        (first_a, count_a), (first_b, cb) = groups[j], groups[j + 1]
        sequence_a, sequence_b = int(seqs[j]), int(seqs[j + 1])
        disjoint = first_a + count_a <= first_b or first_b + cb <= first_a
        if (
            not disjoint
            or 0 in (sequence_a, sequence_b)
            or max(sequence_a, sequence_b) > trace.shape[1]
        ):
            continue
        pushed = trace[first_a, sequence_a - 1, 2]
        seen = trace[first_b : first_b + cb, sequence_b - 1, 0].min()
        lead = int(feeder_commands.wrap(pushed - seen))
        if 0 < lead < 1 << 31:
            found.append(lead / result.mhz)
    return {
        "pairs": job_count - 1,
        "overlapping": len(found),
        "lead_microseconds_mean": round(float(np.mean(found)) if found else 0.0, 3),
    }


def in_flight_together(result, groups, job_count, window=CONSTANTS["KM_RING_SLOTS"]):
    """Pairs of jobs on overlapping groups (at most `window` apart) where a core of
    the later one took it before the earlier one's leader pushed (from the trace)."""
    trace = result.trace.astype(np.int64)
    seqs = result.commands["sequence"]
    pairs = together = 0
    for earlier in range(job_count):
        for later in range(earlier + 1, min(job_count, earlier + 1 + window)):
            (first_a, count_a), (first_b, cb) = groups[earlier], groups[later]
            sequence_a, sequence_b = int(seqs[earlier]), int(seqs[later])
            if (
                first_a + count_a <= first_b
                or first_b + cb <= first_a
                or 0 in (sequence_a, sequence_b)
            ):
                continue
            if max(sequence_a, sequence_b) > trace.shape[1]:
                continue
            pairs += 1
            seen = trace[first_b : first_b + cb, sequence_b - 1, 0].min()
            together += (
                0
                < int(feeder_commands.wrap(trace[first_a, sequence_a - 1, 2] - seen))
                < 1 << 31
            )
    return {"pairs": pairs, "together": together}


def summarize_phases(phases):
    keys = [key for key in phases[0] if key != "count"] if phases else []
    return {
        key: round(float(np.mean([phase[key] for phase in phases])), 3) for key in keys
    }


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    with km.Session(log=fed.log) as session:
        fed.log(f"device open {time.time() - t_start:.1f}s")
        out = run(session, args, rng)
    out["wall_s"] = round(time.time() - t_start, 1)
    out["ok"] = (
        out["feeder"]["status"] == "done"
        and not out["ranks_bad"]
        and out["compute_wrong"] == 0
        and out.get("jobs_bad", 0) == 0
    )
    print(json.dumps(out), flush=True)


def run(session, args, rng):
    runner = fed.Fed(session)
    groups = make_groups(args, runner.workers, rng)
    workers, job_count, tiles = runner.workers, args.jobs, args.tiles
    fed.log(f"grid {runner.grid}, {workers} workers, aiclk {runner.mhz} MHz")

    # Exact: integers in [-2, 2] (sums exact in bf16 for T <= 64). Random: uniform in
    # [-1, 1], rounded to bf16. Job j reads tiles j to j + T - 1 (outputs all differ).
    x_exact = rng.integers(-2, 3, ((tiles + job_count) * TILE, TILE)).astype(np.float32)
    x_rand = rng.uniform(-1, 1, ((tiles + job_count) * TILE, TILE)).astype(np.float32)
    x_rand = formats.bf16_values(formats.bf16_bits(x_rand))

    def tile_tensor(array):
        return session.tensor(array, "bfloat16", "tile")

    inputs = [tile_tensor(x_exact), tile_tensor(x_rand)]
    pages = args.out_pages
    zeros = np.zeros((pages * TILE, TILE), dtype=np.float32)
    outs = [tile_tensor(zeros) for _ in range(args.out_tensors)]
    base = place_outputs(groups, pages, args.out_tensors)

    commands = []
    flags = CONSTANTS["KM_FLAG_EXCLUSIVE"] if args.exclusive else 0
    for j, (first, count) in enumerate(groups):
        if j == args.stall:
            commands.append(
                feeder_commands.command(HANDLER_STALL, first, count, [args.stall_rank])
            )
        elif j == args.bad_item:
            side = [0, int(args.bad_side == "reader")]
            commands.append(
                feeder_commands.command(HANDLER_BAD_ITEM, first, count, side)
            )
        elif j == args.result:
            value, flag = [args.result_value], CONSTANTS["KM_FLAG_RESULT"]
            commands.append(
                feeder_commands.command(HANDLER_RESULT, first, count, value, flag)
            )
        elif j == args.late:
            late = [args.late_rank, args.late_cycles]
            commands.append(feeder_commands.command(HANDLER_LATE, first, count, late))
        elif j == args.exit_at:  # the feeder sets the group to every worker
            x = (
                tuple(map(int, args.exit_group.split(":")))
                if args.exit_group
                else (0, 1)
            )
            commands.append(feeder_commands.command(CONSTANTS["KM_HANDLER_EXIT"], *x))
        elif args.handler == "align":
            commands.append(feeder_commands.command(HANDLER_ALIGN, first, count))
        elif args.handler == "nop":
            commands.append(
                feeder_commands.command(CONSTANTS["KM_HANDLER_NOP"], first, count)
            )
        else:
            words = [
                inputs[j % 2].address,
                tiles,
                outs[base[j] // pages].address,
                base[j] % pages,
                args.slow_cycles if j == args.slow else 0,
                args.slow_rank if j == args.slow else 0xFFFFFFFF,
                j,  # first input tile
            ]
            commands.append(
                feeder_commands.command(HANDLER_MAP, first, count, words, flags)
            )
        if j == args.raw:  # submitted as built: the slot it names and its group kept
            commands[-1][feeder_commands.COMMAND_FIELDS.index("hdr")] |= (
                args.raw_slot << 16
            )

    kernels = fed.Kernels(
        reader=KERNELS / "reader.cpp",
        compute=KERNELS / "compute.cpp",
        writer=KERNELS / "writer.cpp",
        reader_ct=inputs[0].accessor_args(),
        writer_ct=outs[0].accessor_args(),
        compute_options={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        cbs=[
            (HANDLERS["kCbIn"], "bfloat16", 2048, HANDLERS["kInPages"]),
            (HANDLERS["kCbOut"], "bfloat16", 2048, 2),
        ],
    )
    trace_cap = (job_count + 4) // 4 * 4 if args.trace else 0  # 64-byte pages
    trace_cap = args.trace_cap if args.trace and args.trace_cap else trace_cap
    result = runner.run(
        kernels,
        commands,
        tensors=[*inputs, *outs],
        depth=args.depth,
        deadline_s=args.deadline,
        wait_s=args.wait,
        trace_cap=trace_cap,
        release=args.release if args.handler == "align" else 0,
        row=args.row,
        raw=args.raw + 1 if args.raw >= 0 else 0,
    )
    if args.dump:
        np.savez(
            args.dump,
            groups=np.asarray(groups),
            base=np.asarray(base),
            sequence=result.commands["sequence"],
            slot=result.commands["slot"],
            submit=result.commands["submit"],
            acked=result.commands["acked"],
            state=result.state,
            trace=result.trace if result.trace is not None else np.zeros(0),
        )
    summary = result.summary
    out = {
        "jobs": job_count,
        "handler": args.handler,
        "groups": args.groups,
        "count": args.count if args.groups == "fixed" else None,
        "tiles": tiles,
        "depth": args.depth,
        "excl": args.exclusive,
        "held": int(result.commands["held"].sum()),
        "result": (
            int(result.commands["result"][args.result]) if args.result >= 0 else None
        ),
        "aiclk_mhz": result.mhz,
        "feeder": summary,
        "microseconds_per_command": round(
            summary["span_cycles"] / result.mhz / max(summary["done"], 1), 3
        ),
        "submit_microseconds_mean": round(
            summary["submit_cycles"] / result.mhz / max(summary["submitted"], 1), 3
        ),
        "host_program_s": round(result.host_s, 2),
    }
    lat = feeder_commands.wrap(result.commands["acked"] - result.commands["submit"])[
        result.commands["status"] == 1
    ]
    if len(lat):
        out["latency_microseconds"] = {
            "mean": round(float(lat.mean()) / result.mhz, 3),
            "min": round(float(lat.min()) / result.mhz, 3),
            "max": round(float(lat.max()) / result.mhz, 3),
        }
    if (
        0 <= args.exit_at < job_count
    ):  # submitted only when every earlier command was acked
        commands, job = result.commands, args.exit_at
        gap = feeder_commands.wrap(commands["submit"][job] - commands["acked"][:job])
        out["exit"] = {
            "held": bool(commands["held"][job]),
            "status": int(commands["status"][job]),
            "after_every_ack": bool(((gap > 0) & (gap < 1 << 31)).all()),
        }
    states = [feeder_commands.core_state(result.state, core) for core in range(workers)]
    out["runs"] = sum(core_state["runs"] for core_state in states)
    if "exit" in out:  # cores whose report of the EXIT landed
        slot, sequence = int(result.commands["slot"][args.exit_at]), int(
            result.commands["sequence"][args.exit_at]
        )
        out["exit"]["reported_by"] = sum(
            core_state["reported"][slot] == sequence for core_state in states
        )
    out["slot_wrong"] = {
        core_state["core"]: core_state["slot_wrong"]
        for core_state in states
        if core_state["slot_wrong"]
    }
    out["trace_dropped"] = {
        core_state["core"]: core_state["trace_dropped"]
        for core_state in states
        if core_state["trace_dropped"]
    }
    out["ranks_bad"] = rank_errors(args, groups, result, states)[:16]
    out["cores_magic_ok"] = sum(core_state["magic_ok"] for core_state in states)
    out["cores_ended"] = {
        field: sum(core_state[field] != 0 for core_state in states)
        for field in ("reader_end", "writer_end", "compute_end")
    }
    out["compute_wrong"] = sum(
        sum(core_state["compute_wrong"]) for core_state in states
    )
    passes = sum(core_state["passes"] for core_state in states)
    out["next_cycles_per_pass"] = round(
        sum(core_state["next_cycles"] for core_state in states) / max(passes, 1), 1
    )
    parts = ("reader", "writer", "compute")
    gave = [
        {"core": core_state["core"], **{part: core_state[part] for part in parts}}
        for core_state in states
        if any(core_state[part] for part in parts)
    ]
    out["gave_up_cores"] = len(gave)
    out["gave_up_where"] = {
        where: sum(
            1
            for gave_entry in gave
            for part in parts
            if gave_entry[part] and gave_entry[part]["where"] == where
        )
        for where in sorted(
            {
                gave_entry[part]["where"]
                for gave_entry in gave
                for part in parts
                if gave_entry[part]
            }
        )
    }
    out["gave_up_busy"] = [  # cores that gave up anywhere but waiting for a command
        {
            "core": gave_entry["core"],
            **{part: gave_entry[part] for part in parts if gave_entry[part]},
        }
        for gave_entry in gave
        if any(
            gave_entry[part] and gave_entry[part]["where"] != "next" for part in parts
        )
    ][:16]
    if summary["status"] == "timeout":
        first, count = groups[summary["index"]]
        group = range(first, first + count)
        out["timed_out"] = {
            "slot": summary["slot"],
            "sequence": summary["sequence"],
            "group": [first, count],
        }
        out["not_reported"] = [
            core
            for core in group
            if states[core]["reported"][summary["slot"]] != summary["sequence"]
        ]
        out["named"] = [
            {
                "core": gave_entry["core"],
                **{part: gave_entry[part] for part in parts if gave_entry[part]},
            }
            for gave_entry in gave
            if any(
                gave_entry[part] and gave_entry[part]["where"] in ("stall", "complete")
                for part in parts
            )
        ]
    if args.trace and result.trace is not None:
        phases = trace_phases(result, groups, job_count)
        if args.handler == "align":  # release at submit + args.release cycles
            for phase in phases:
                phase["push_after_release"] = (
                    phase["leader_sent"] - args.release / result.mhz
                )
        out["trace_microseconds"] = summarize_phases(phases)
        out["overlap"] = overlaps(result, groups, job_count)
        out["overlapping_groups"] = in_flight_together(result, groups, job_count)

    if args.handler == "map" and summary["status"] == "done":
        out.update(check(args, groups, base, outs, [x_exact, x_rand]))
    return out


def check(args, groups, base, outs, inputs):
    pages = args.out_pages * len(outs)
    got = []
    for i, tensor in enumerate(outs):
        got.append(tensor.read())
        fed.log(f"output {i + 1} of {len(outs)} read back")
    got = np.concatenate(got)
    got = got.reshape(pages, TILE, TILE)
    fed.log("outputs read back")
    last = args.exit_at if args.exit_at >= 0 else len(groups)
    jobs = [
        j
        for j in checked_jobs(groups, base, pages)
        if j not in (args.stall, args.late, args.bad_item, args.result) and j < last
    ]
    bad, rel_max, exact_jobs = [], 0.0, 0
    for j in jobs:
        _, count = groups[j]
        x = inputs[j % 2][j * TILE : (j + args.tiles) * TILE]  # tiles j to j + T - 1
        ref = rank_sums(x.astype(np.float64), count)
        mine = got[base[j] : base[j] + count].astype(np.float64)
        if j % 2 == 0:
            exact_jobs += 1
            if not np.array_equal(mine, ref):
                bad.append(
                    {
                        "job": j,
                        "group": groups[j],
                        "base": base[j],
                        "wrong_ranks": int((mine != ref).any((1, 2)).sum()),
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
                    {"job": j, "group": groups[j], "base": base[j], "rel_rms": rel}
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
