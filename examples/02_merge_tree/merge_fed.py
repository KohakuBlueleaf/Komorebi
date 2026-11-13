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

import commands as rec  # noqa: E402
import fed  # noqa: E402

import komorebi as km  # noqa: E402

K = rec.K
KERNELS = HERE / "kernels"
M = rec.constants(KERNELS / "merge.h")
TILE = 32
NONE = 0xFFFFFFFF
FORMS = {"owner": 0, "tree": 1}


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--groups", choices=["fixed", "random"], default="fixed")
    p.add_argument("--count", type=int, default=8, help="group size (fixed)")
    p.add_argument("--first", type=int, default=0, help="first core (fixed)")
    p.add_argument(
        "--group-list", help="first:count per job, comma-separated (overrides --groups)"
    )
    p.add_argument(
        "--form",
        choices=["owner", "tree", "both", "random"],
        default="both",
        help="both: owner and tree alternate in pairs (each group runs both)",
    )
    p.add_argument("--tiles", type=int, default=64, help="input tiles T")
    p.add_argument("--depth", type=int, default=4, help="ring slots used")
    p.add_argument("--out-pages", type=int, default=256, help="output tiles")
    p.add_argument("--part", type=int, default=1, help="tiles per part P (1 to 3)")
    p.add_argument("--wait", type=float, default=0.5, help="wait limit, seconds")
    p.add_argument("--deadline", type=float, default=0.05, help="per command, seconds")
    p.add_argument(
        "--stall-rank",
        type=int,
        default=-1,
        help="then a launch of one job whose rank M stalls, and a clean launch",
    )
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.part <= M["kPartMax"]:
        p.error(f"--part: 1 to {M['kPartMax']}")
    return args


def make_jobs(args, workers, rng):
    """[(first, count, form, source)] per job: the job reads input tiles source to
    source + T - 1 of input source % 2 (0 exact, 1 random)."""
    if args.group_list:
        groups = [tuple(map(int, t.split(":"))) for t in args.group_list.split(",")]
    elif args.groups == "fixed":
        groups = [(args.first, args.count)] * args.jobs
    else:
        groups = rec.random_groups(rng, args.jobs, workers)
    sources = list(range(len(groups)))
    if args.form == "both":  # each job twice in a row, on the same input: owner, tree
        groups = [t for t in groups for _ in range(2)]
        sources = [s for s in sources for _ in range(2)]
        forms = [j % 2 for j in range(len(groups))]
    elif args.form == "random":
        forms = [int(f) for f in rng.integers(0, 2, len(groups))]
    else:
        forms = [FORMS[args.form]] * len(groups)
    return [(f, c, form, s) for (f, c), form, s in zip(groups, forms, sources)]


def kernels(inp, out):
    """The three kernels, their buffers and a float32 compute path end to end."""
    config = {
        "math_fidelity": "HiFi4",
        "fp32_dest_acc_en": True,
        "unpack_to_dest_mode": "UnpackToDestFp32",
    }
    page = M["kTileBytes"]
    return fed.Kernels(
        reader=KERNELS / "reader.cpp",
        compute=KERNELS / "compute.cpp",
        writer=KERNELS / "writer.cpp",
        reader_ct=inp.accessor_args(),
        writer_ct=out.accessor_args(),
        compute_options=config,
        cbs=[
            (M["kCbIn"], "float32", page, M["kInPages"]),
            (M["kCbArrive"], "float32", page, M["kArrivePages"]),
            (M["kCbOut"], "float32", page, M["kOutPages"]),
        ],
    )


def launch(runner, args, jobs, inputs, out, stall=-1, stall_rank=NONE):
    """Runs the jobs (job j writes output part j % parts); job `stall` has rank
    `stall_rank` stall, under a deadline long enough for its receiver's wait to
    give up first."""
    commands = []
    for j, (first, count, form, src) in enumerate(jobs):
        words = [
            inputs[src % 2].address,
            args.tiles,
            out.address,
            j % (args.out_pages // args.part) * args.part,
            form,
            stall_rank if j == stall else NONE,
            src,
            args.part,
        ]
        commands.append(
            rec.command(M["kHandlerMerge"], first, count, words, K["KM_F_EXCL"])
        )
    return runner.run(
        kernels(inputs[0], out),
        commands,
        tensors=[*inputs, out],
        depth=args.depth,
        deadline_s=args.deadline if stall < 0 else 4 * args.wait,
        wait_s=args.wait,
    )


def summary(res, jobs):
    """Feeder state, the cores that gave up and the ports still waiting."""
    s = res.summary
    states = [rec.core_state(res.state, c) for c in range(len(res.state))]
    parts = ("reader", "writer", "compute")
    gave = []  # cores that gave up anywhere but waiting for a command
    for c in states:
        g = {k: c[k] for k in parts if c[k]}
        if any(v["where"] != "next" for v in g.values()):
            gave.append({"core": c["core"], **g})
    out = {
        "status": s["status"],
        "submitted": s["submitted"],
        "done": s["done"],
        "exit_acked": s["exit_acked"],
        "cores_ended": {
            k: sum(c[k] != 0 for c in states)
            for k in ("reader_end", "writer_end", "compute_end")
        },
        "compute_wrong": sum(sum(c["compute_wrong"]) for c in states),
        "gave_up": gave[:8],
        "waiting_ports": {
            c: w for c in range(len(states)) if (w := rec.waiting_ports(res.ports, c))
        },
    }
    if s["status"] == "timeout":
        first, count, form, _ = jobs[s["index"]]
        out["timed_out"] = {"job": s["index"], "group": [first, count], "form": form}
    return out


def reference(args, src, xs):
    """The P total tiles of a job reading from input tile `src` (float64): sum d
    holds the input tiles d mod P."""
    x = xs[src % 2][src * TILE : (src + args.tiles) * TILE].astype(np.float64)
    tiles = x.reshape(-1, TILE, TILE)
    return np.stack([tiles[d :: args.part].sum(0) for d in range(args.part)])


def check(args, jobs, done, out, xs):
    """Exact jobs bit-equal to numpy, random jobs by relative RMS; with both forms,
    the two jobs of an exact pair bit-equal to each other. `done[j]`: job j ended."""
    got = out.read().reshape(-1, TILE, TILE)
    fed.log("outputs read back")
    n = len(jobs)
    P = args.part
    last = {j % (args.out_pages // P): j for j in range(n)}  # parts not reused
    bad, rel_max, exact, pairs = [], 0.0, {0: 0, 1: 0}, [0, 0]
    kept = {j: page for page, j in last.items() if done[j]}
    for j, page in sorted(kept.items()):
        first, count, form, src = jobs[j]
        ref = reference(args, src, xs)
        mine = got[page * P : page * P + P].astype(np.float64)
        if args.form == "both" and j % 2 == 1 and j - 1 in kept and src % 2 == 0:
            other = kept[j - 1] * P
            pairs[0] += 1
            pairs[1] += np.array_equal(
                got[page * P : page * P + P], got[other : other + P]
            )
        if src % 2 == 0:
            exact[form] += 1
            if not np.array_equal(mine, ref):
                wrong = int((mine != ref).sum())
                bad.append(
                    {"job": j, "group": [first, count], "form": form, "wrong": wrong}
                )
        else:
            rel = float(np.sqrt(((mine - ref) ** 2).mean() / (ref**2).mean()))
            rel_max = max(rel_max, rel)
            if rel > 1e-5:
                bad.append({"job": j, "group": [first, count], "rel_rms": rel})
    return {
        "jobs_checked": len(kept),
        "exact_checked": {"owner": exact[0], "tree": exact[1]},
        "pairs_checked": pairs[0],
        "pairs_bit_equal": int(pairs[1]),
        "jobs_bad": len(bad),
        "first_bad": bad[:5],
        "random_rel_rms_max": float(f"{rel_max:.3e}"),
    }


def main():
    args = parse()
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    with km.Session(log=fed.log) as session:
        fed.log(f"device open {time.time() - t_start:.1f}s")
        result = run(session, args, rng)
    result["wall_s"] = round(time.time() - t_start, 1)
    print(json.dumps(result), flush=True)
    sys.exit(0 if result["ok"] else 1)


def run(session, args, rng):
    runner = fed.Fed(session)
    jobs = make_jobs(args, runner.workers, rng)
    n = len(jobs)
    fed.log(f"grid {runner.grid}, {runner.workers} workers, aiclk {runner.mhz} MHz")

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
    res = launch(runner, args, jobs, inputs, out)
    result = {
        "jobs": n,
        "tiles": args.tiles,
        "part": args.part,
        "form": args.form,
        "aiclk_mhz": res.mhz,
        "us_per_job": round(
            res.summary["span_cycles"] / res.mhz / max(res.summary["done"], 1), 3
        ),
        **summary(res, jobs),
        **check(args, jobs, res.commands["status"] == 1, out, xs),
    }
    if args.stall_rank >= 0:  # a core stalls in job 1; then a clean launch
        few = jobs[:4]
        stalled = launch(runner, args, few, inputs, out, 1, args.stall_rank)
        result["stall"] = summary(stalled, few)
        after = launch(runner, args, few, inputs, out)
        done = after.commands["status"] == 1
        result["after"] = {**summary(after, few), **check(args, few, done, out, xs)}
    result["ok"] = verdict(result, n, runner.workers)
    return result


def ended(launch, cores):
    return (
        set(launch["cores_ended"].values()) == {cores} and not launch["compute_wrong"]
    )


def clean(launch, jobs, cores):
    """Every job done and right, the exit acknowledged, no core gave up, no port left
    waiting, every core ended."""
    return (
        ended(launch, cores)
        and launch["status"] == "done"
        and launch["done"] == jobs
        and launch["exit_acked"]
        and launch["jobs_bad"] == 0
        and not launch["gave_up"]
        and not launch["waiting_ports"]
    )


def verdict(result, jobs, cores):
    """The jobs clean; with --stall-rank, the stalled launch timed out on job 1 with
    every core ended, and the launch after it clean."""
    ok = clean(result, jobs, cores)
    if "stall" in result:
        s = result["stall"]
        ok &= ended(s, cores) and s["status"] == "timeout"
        ok &= s.get("timed_out", {}).get("job") == 1
        ok &= clean(result["after"], min(4, jobs), cores)
    return bool(ok)


if __name__ == "__main__":
    main()
