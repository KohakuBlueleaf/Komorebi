"""Timing and power table of the C++ (1 thread, all threads) and CUDA rasterizers:
setup, binning and fill, median of RUNS processes x repetitions; seconds per frame
and power from one more run of at least 3 s (the README's "Timings and power" says what
is timed). `python -m projects.raster.reference.bench [--scenes ...] [--json FILE]`"""

import argparse
import json
import math
import statistics

from projects.raster.reference import harness

STAGES = harness.STAGES
POWER_SECONDS = 3.0


def runners(scene):
    """(folder, label, function(repetitions, with_power)) per implementation."""
    threads = harness.threads_all()
    return [
        (
            "cpu1",
            "C++ 1 thread",
            lambda repetitions, with_power: harness.run_cpu(
                scene, 1, repetitions, False, with_power
            ),
        ),
        (
            "cpu_all",
            f"C++ {threads} threads",
            lambda repetitions, with_power: harness.run_cpu(
                scene, threads, repetitions, False, with_power
            ),
        ),
        (
            "v100",
            "V100",
            lambda repetitions, with_power: harness.run_gpu(
                scene, repetitions, False, with_power
            ),
        ),
    ]


def summarize(records):
    """Median, min and max per stage over every repetition of every run."""
    summary = {}
    for stage in STAGES:
        values = [
            repetition[stage] for record in records for repetition in record["reps"]
        ]
        summary[stage] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    summary["load"] = sorted(
        {f"{record['load_before']} / {record['load_after']}" for record in records}
    )
    return summary


def measure(scene, runs, repetitions):
    """{folder: {label, stages, power}} for the implementations available."""
    report = {}
    for folder, label, run in runners(scene):
        records = [run(repetitions, False) for _ in range(runs)]
        if records[0] is None:
            continue
        stages = summarize(records)
        frames = max(3, math.ceil(POWER_SECONDS * 1000 / stages["total_ms"]["median"]))
        sampled = run(frames, True)
        report[folder] = {
            "label": label,
            "stages": stages,
            "power": sampled["power"],
            "frames": frames,
            "load_power_run": f"{sampled['load_before']} / {sampled['load_after']}",
        }
    return report


def fmt(x, digits=1):
    return "n/a" if x is None else f"{x:.{digits}f}"


HEADER = (
    "| scene | implementation | setup ms | binning ms | fill ms | total ms | "
    "ms per frame | GPU W idle / mean / peak | GPU J per frame | "
    "CPU package J per frame | card W idle / mean / peak, J per frame | host load |\n"
    + "|---" * 12
    + "|"
)


def row(scene, label, stages, power, load=""):
    cells = " | ".join(
        f"{stages[stage]['median']:.2f} "
        f"({stages[stage]['min']:.2f}-{stages[stage]['max']:.2f})"
        for stage in STAGES
    )
    gpu = "n/a"
    if "gpu_mean_w" in power:
        gpu = f"{power['gpu_idle_w']:.0f} / {power['gpu_mean_w']:.0f} / "
        gpu += f"{power['gpu_peak_w']:.0f}"
    joules = power.get("gpu_joules_per_frame")
    cpu = power.get("cpu_package_joules_per_frame")
    card = "n/a"
    if "card_mean_w" in power:
        card = f"{power['card_idle_w']:.0f} / {power['card_mean_w']:.0f} / "
        card += f"{power['card_peak_w']:.0f}, {power['card_joules_per_frame']:.3f}"
    seconds_per_frame = power["seconds_per_frame"]
    return (
        f"| {scene} | {label} | {cells} | {seconds_per_frame * 1000:.2f} | {gpu} | "
        f"{fmt(joules, 3)} | {fmt(cpu, 3)} | {card} | {load} |"
    )


def external_rows(scene, external):
    """Rows for implementations in --external folders that carry timings."""
    rows = []
    for name, (_, info) in harness.collect(scene, external).items():
        stages = info.get("timings_ms")
        if name in ("numpy", "C++ 1 thread", "C++ all threads", "V100") or not stages:
            continue
        cells = {
            stage: {"median": value, "min": value, "max": value}
            for stage, value in stages.items()
        }
        if all(stage in cells for stage in STAGES):
            power = info.get("power", {"seconds_per_frame": info["seconds_per_frame"]})
            rows.append(row(scene, f"{name} (external)", cells, power))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--external", nargs="*", default=[])
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = {}
    print(HEADER)
    for scene in args.scenes:
        report[scene] = measure(scene, args.runs, args.reps)
        for folder, entry in report[scene].items():
            harness.attach_power(scene, folder, {"power": entry["power"]})
            load = "; ".join(entry["stages"]["load"])
            print(row(scene, entry["label"], entry["stages"], entry["power"], load))
        for line in external_rows(scene, args.external):
            print(line)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)


if __name__ == "__main__":
    main()
