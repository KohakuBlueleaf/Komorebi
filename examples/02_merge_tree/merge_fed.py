"""The merge-tree example's kernels fed by the test feeder on the last core (no X280);
every job's total is checked against numpy. One JSON line.

Example: python merge_fed.py --jobs 20 --count 8 --form both"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FEEDER = HERE.parents[1] / "tests" / "hardware" / "feeder"
sys.path.insert(0, str(FEEDER))

import commands as feeder_commands  # noqa: E402
import fed  # noqa: E402

import komorebi as km  # noqa: E402

RUNTIME_CONSTANTS = feeder_commands.K
KERNELS = HERE / "kernels"
MERGE_CONSTANTS = feeder_commands.constants(KERNELS / "merge.h")
TILE = 32
NONE = 0xFFFFFFFF
FORMS = {"owner": 0, "tree": 1}


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
        help="then a launch of one job whose rank M stalls, and a clean launch",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.part <= MERGE_CONSTANTS["kPartMax"]:
        parser.error(f"--part: 1 to {MERGE_CONSTANTS['kPartMax']}")
    return args


def make_jobs(args, workers, generator):
    """[(first, count, form, source)] per job: the job reads input tiles source to
    source + T - 1 of input source % 2 (0 exact, 1 random)."""
    if args.group_list:
        groups = [
            tuple(map(int, text.split(":"))) for text in args.group_list.split(",")
        ]
    elif args.groups == "fixed":
        groups = [(args.first, args.count)] * args.jobs
    else:
        groups = feeder_commands.random_groups(generator, args.jobs, workers)
    sources = list(range(len(groups)))
    if args.form == "both":  # each job twice in a row, on the same input: owner, tree
        groups = [group for group in groups for _ in range(2)]
        sources = [source for source in sources for _ in range(2)]
        forms = [job % 2 for job in range(len(groups))]
    elif args.form == "random":
        forms = [int(form) for form in generator.integers(0, 2, len(groups))]
    else:
        forms = [FORMS[args.form]] * len(groups)
    return [
        (first, count, form, source)
        for (first, count), form, source in zip(groups, forms, sources)
    ]


def kernels(input_tensor, output_tensor):
    """The three kernels, their buffers and a float32 compute path end to end."""
    compute_options = {
        "math_fidelity": "HiFi4",
        "fp32_dest_acc_en": True,
        "unpack_to_dest_mode": "UnpackToDestFp32",
    }
    page = MERGE_CONSTANTS["kTileBytes"]
    return fed.Kernels(
        reader=KERNELS / "reader.cpp",
        compute=KERNELS / "compute.cpp",
        writer=KERNELS / "writer.cpp",
        reader_ct=input_tensor.accessor_args(),
        writer_ct=output_tensor.accessor_args(),
        compute_options=compute_options,
        cbs=[
            (MERGE_CONSTANTS["kCbIn"], "float32", page, MERGE_CONSTANTS["kInPages"]),
            (
                MERGE_CONSTANTS["kCbArrive"],
                "float32",
                page,
                MERGE_CONSTANTS["kArrivePages"],
            ),
            (MERGE_CONSTANTS["kCbOut"], "float32", page, MERGE_CONSTANTS["kOutPages"]),
        ],
    )


def launch(runner, args, jobs, inputs, output, stall=-1, stall_rank=NONE):
    """Runs the jobs (each job writes output part job % parts); job `stall` has rank
    `stall_rank` stall, under a deadline long enough for its receiver's wait to
    give up first."""
    commands = []
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
        commands.append(
            feeder_commands.command(
                MERGE_CONSTANTS["kHandlerMerge"],
                first,
                count,
                words,
                RUNTIME_CONSTANTS["KM_FLAG_EXCLUSIVE"],
            )
        )
    return runner.run(
        kernels(inputs[0], output),
        commands,
        tensors=[*inputs, output],
        depth=args.depth,
        deadline_s=args.deadline if stall < 0 else 4 * args.wait,
        wait_s=args.wait,
    )


def summary(launched, jobs):
    """Feeder state, the cores that gave up and the ports still waiting."""
    feeder_summary = launched.summary
    states = [
        feeder_commands.core_state(launched.state, core)
        for core in range(len(launched.state))
    ]
    kernel_names = ("reader", "writer", "compute")
    gave_up_cores = []  # cores that gave up anywhere but waiting for a command
    for state in states:
        gave_up_in = {name: state[name] for name in kernel_names if state[name]}
        if any(record["where"] != "next" for record in gave_up_in.values()):
            gave_up_cores.append({"core": state["core"], **gave_up_in})
    report = {
        "status": feeder_summary["status"],
        "submitted": feeder_summary["submitted"],
        "done": feeder_summary["done"],
        "exit_acked": feeder_summary["exit_acked"],
        "cores_ended": {
            key: sum(state[key] != 0 for state in states)
            for key in ("reader_end", "writer_end", "compute_end")
        },
        "compute_wrong": sum(sum(state["compute_wrong"]) for state in states),
        "gave_up": gave_up_cores[:8],
        "waiting_ports": {
            core: waiting
            for core in range(len(states))
            if (waiting := feeder_commands.waiting_ports(launched.ports, core))
        },
    }
    if feeder_summary["status"] == "timeout":
        first, count, form, _ = jobs[feeder_summary["index"]]
        report["timed_out"] = {
            "job": feeder_summary["index"],
            "group": [first, count],
            "form": form,
        }
    return report


def reference(args, source, input_arrays):
    """The P total tiles of a job reading from input tile `source` (float64): total
    `sum_index` holds the input tiles whose number mod P is `sum_index`."""
    values = input_arrays[source % 2][
        source * TILE : (source + args.tiles) * TILE
    ].astype(np.float64)
    tiles = values.reshape(-1, TILE, TILE)
    return np.stack(
        [tiles[sum_index :: args.part].sum(0) for sum_index in range(args.part)]
    )


def check(args, jobs, done, output, input_arrays):
    """Exact jobs bit-equal to numpy, random jobs by relative RMS; with both forms,
    the two jobs of an exact pair bit-equal to each other. `done[job]`: that job ended.
    """
    got = output.read().reshape(-1, TILE, TILE)
    fed.log("outputs read back")
    job_count = len(jobs)
    part_tiles = args.part
    last = {
        job % (args.out_pages // part_tiles): job for job in range(job_count)
    }  # parts not reused
    bad, relative_error_max, exact, pairs = [], 0.0, {0: 0, 1: 0}, [0, 0]
    kept = {job: page for page, job in last.items() if done[job]}
    for job, page in sorted(kept.items()):
        first, count, form, source = jobs[job]
        expected = reference(args, source, input_arrays)
        got_part = got[page * part_tiles : page * part_tiles + part_tiles].astype(
            np.float64
        )
        if args.form == "both" and job % 2 == 1 and job - 1 in kept and source % 2 == 0:
            other = kept[job - 1] * part_tiles
            pairs[0] += 1
            pairs[1] += np.array_equal(
                got[page * part_tiles : page * part_tiles + part_tiles],
                got[other : other + part_tiles],
            )
        if source % 2 == 0:
            exact[form] += 1
            if not np.array_equal(got_part, expected):
                wrong = int((got_part != expected).sum())
                bad.append(
                    {"job": job, "group": [first, count], "form": form, "wrong": wrong}
                )
        else:
            relative_error = float(
                np.sqrt(((got_part - expected) ** 2).mean() / (expected**2).mean())
            )
            relative_error_max = max(relative_error_max, relative_error)
            if relative_error > 1e-5:
                bad.append(
                    {"job": job, "group": [first, count], "rel_rms": relative_error}
                )
    return {
        "jobs_checked": len(kept),
        "exact_checked": {"owner": exact[0], "tree": exact[1]},
        "pairs_checked": pairs[0],
        "pairs_bit_equal": int(pairs[1]),
        "jobs_bad": len(bad),
        "first_bad": bad[:5],
        "random_rel_rms_max": float(f"{relative_error_max:.3e}"),
    }


def main():
    args = parse()
    generator = np.random.default_rng(args.seed)
    started = time.time()
    with km.Session(log=fed.log) as session:
        fed.log(f"device open {time.time() - started:.1f}s")
        result = run(session, args, generator)
    result["wall_s"] = round(time.time() - started, 1)
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["ok"] else 1)


def run(session, args, generator):
    runner = fed.Fed(session)
    jobs = make_jobs(args, runner.workers, generator)
    job_count = len(jobs)
    fed.log(f"grid {runner.grid}, {runner.workers} workers, aiclk {runner.mhz} MHz")

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
    launched = launch(runner, args, jobs, inputs, output)
    result = {
        "jobs": job_count,
        "tiles": args.tiles,
        "part": args.part,
        "form": args.form,
        "aiclk_mhz": launched.mhz,
        "microseconds_per_job": round(
            launched.summary["span_cycles"]
            / launched.mhz
            / max(launched.summary["done"], 1),
            3,
        ),
        **summary(launched, jobs),
        **check(args, jobs, launched.commands["status"] == 1, output, input_arrays),
    }
    if args.stall_rank >= 0:  # a core stalls in job 1; then a clean launch
        few = jobs[:4]
        stalled = launch(runner, args, few, inputs, output, 1, args.stall_rank)
        result["stall"] = summary(stalled, few)
        after = launch(runner, args, few, inputs, output)
        done = after.commands["status"] == 1
        result["after"] = {
            **summary(after, few),
            **check(args, few, done, output, input_arrays),
        }
    result["ok"] = verdict(result, job_count, runner.workers)
    return result


def ended(load, cores):
    return set(load["cores_ended"].values()) == {cores} and not load["compute_wrong"]


def clean(load, jobs, cores):
    """Every job done and right, the exit acknowledged, no core gave up, no port left
    waiting, every core ended."""
    return (
        ended(load, cores)
        and load["status"] == "done"
        and load["done"] == jobs
        and load["exit_acked"]
        and load["jobs_bad"] == 0
        and not load["gave_up"]
        and not load["waiting_ports"]
    )


def verdict(result, jobs, cores):
    """The jobs clean; with --stall-rank, the stalled launch timed out on job 1 with
    every core ended, and the launch after it clean."""
    ok = clean(result, jobs, cores)
    if "stall" in result:
        stalled = result["stall"]
        ok &= ended(stalled, cores) and stalled["status"] == "timeout"
        ok &= stalled.get("timed_out", {}).get("job") == 1
        ok &= clean(result["after"], min(4, jobs), cores)
    return bool(ok)


if __name__ == "__main__":
    main()
