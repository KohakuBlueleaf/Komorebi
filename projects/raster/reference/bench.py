"""Timing table of the C++ (1 thread, all threads) and CUDA rasterizers.

Each cell is the median of RUNS processes x REPS timed repetitions (after one untimed
warm-up per process); min and max are given too. Timed: triangle setup, binning
and fill (CPU: threads over triangles, then over tiles; GPU: kernels, with the list size
read back). Not timed: scene load, upload, output readback.

    python -m projects.raster.reference.bench [--scenes ...] [--runs 3] [--json FILE]
"""

import argparse
import json
import statistics

from projects.raster.reference import harness

STAGES = ("setup_ms", "bin_ms", "fill_ms", "total_ms")


def measure(scene, runs, reps):
    """List of (label, [timing records]) for the implementations available."""
    rows = []
    for label, runner in [
        ("C++ 1 thread", lambda: harness.run_cpu(scene, 1, reps, save=False)),
        (
            f"C++ {harness.threads_all()} threads",
            lambda: harness.run_cpu(scene, harness.threads_all(), reps, save=False),
        ),
        ("V100", lambda: harness.run_gpu(scene, reps, save=False)),
    ]:
        records = [runner() for _ in range(runs)]
        if records[0] is not None:
            rows.append((label, records))
    return rows


def summarize(records):
    """Median, min and max per stage over every repetition of every run."""
    out = {}
    for stage in STAGES:
        values = [rep[stage] for rec in records for rep in rec["reps"]]
        out[stage] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    out["load"] = sorted({f"{r['load_before']} / {r['load_after']}" for r in records})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = {}
    print("| scene | implementation | setup | binning | fill | total | n |")
    print("|---|---|---|---|---|---|---|")
    for scene in args.scenes:
        for label, records in measure(scene, args.runs, args.reps):
            s = summarize(records)
            report.setdefault(scene, {})[label] = s | {"raw": records}
            cells = " | ".join(
                f"{s[k]['median']:.3f} ({s[k]['min']:.3f}-{s[k]['max']:.3f})"
                for k in STAGES
            )
            print(f"| {scene} | {label} | {cells} | {s['total_ms']['n']} |")
            print(f"|  | host load: {'; '.join(s['load'])} | | | | | |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
