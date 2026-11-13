"""Rays per second, time and power of the C++ (1 thread, all threads) and CUDA tracers.

A frame is the five stages in order; each cell is the median of RUNS processes x REPS
repetitions with min and max. What is timed: this folder's README.
"""

import argparse
import json
import math
import statistics

from projects.raytrace.reference import harness
from reference import runner

POWER_SECONDS = 3.0
RATES = ("primary", "shadow", "bounce", "together")


def runners(scene):
    """(folder, label, function(reps, with_power)) per implementation."""
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


def rates(record, repetition):
    """Mray/s of the stages of one repetition, and its total time."""
    shadow_rays = record["rays_shadow"] + record["rays_shadow2"]
    total_rays = record["rays_primary"] + shadow_rays + record["rays_bounce"]
    return {
        "primary": record["rays_primary"] / repetition["primary_ms"] / 1000,
        "shadow": shadow_rays
        / (repetition["shadow_ms"] + repetition["shadow2_ms"])
        / 1000,
        "bounce": record["rays_bounce"] / repetition["bounce_ms"] / 1000,
        "together": total_rays / repetition["total_ms"] / 1000,
        "total_ms": repetition["total_ms"],
    }


def summarize(records):
    """Median, min and max of each rate and the total time over all repetitions."""
    per_repetition = [
        rates(record, repetition) for record in records for repetition in record["reps"]
    ]
    summary = {}
    for key in (*RATES, "total_ms"):
        values = [repetition[key] for repetition in per_repetition]
        summary[key] = {"median": statistics.median(values), "min": min(values)}
        summary[key].update({"max": max(values), "n": len(values)})
    summary["load"] = sorted(
        {f"{record['load_before']} / {record['load_after']}" for record in records}
    )
    summary["rays"] = {
        name: value for name, value in records[0].items() if name.startswith("rays_")
    }
    return summary


def measure(scene, runs, reps):
    """{folder: {label, stages, power}} for the implementations available."""
    report = {}
    for folder, label, run in runners(scene):
        records = [run(reps, False) for _ in range(runs)]
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


HEADER = (
    "| scene | implementation | primary Mray/s | shadow Mray/s | bounce Mray/s | "
    "together Mray/s | total ms | ms per frame | GPU W idle / mean / peak | "
    "GPU J per frame | CPU package J per frame | host load |\n" + "|---" * 12 + "|"
)


def fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def cell(summary):
    if summary["median"] != summary["median"]:
        return "n/a"
    return f"{summary['median']:.2f} ({summary['min']:.2f}-{summary['max']:.2f})"


def row(scene, label, stages, power, load=""):
    cells = " | ".join(cell(stages[stage]) for stage in (*RATES, "total_ms"))
    gpu_watts = "n/a"
    if "gpu_mean_w" in power:
        gpu_watts = f"{power['gpu_idle_w']:.0f} / {power['gpu_mean_w']:.0f} / "
        gpu_watts += f"{power['gpu_peak_w']:.0f}"
    gpu_joules = power.get("gpu_joules_per_frame")
    cpu_joules = power.get("cpu_package_joules_per_frame")
    return (
        f"| {scene} | {label} | {cells} | {power['seconds_per_frame'] * 1000:.1f} | "
        f"{gpu_watts} | {fmt(gpu_joules, 2)} | {fmt(cpu_joules, 2)} | {load} |"
    )


def external_rows(scene, external, rays_per_frame):
    """Rows for implementations in --external folders that report a total time."""
    rows = []
    for name, (_, info) in harness.collect(scene, external).items():
        total = (info.get("timings_ms") or {}).get("total_ms")
        if name in ("numpy", "C++ 1 thread", "C++ all threads", "V100") or not total:
            continue
        rate = rays_per_frame / total / 1000
        stages = {
            stage: {"median": float("nan"), "min": 0, "max": 0} for stage in RATES
        }
        stages["together"] = {"median": rate, "min": rate, "max": rate}
        stages["total_ms"] = {"median": total, "min": total, "max": total}
        power = info.get("power", {"seconds_per_frame": info["seconds_per_frame"]})
        rows.append(row(scene, f"{name} (external)", stages, power))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=harness.SCENES)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--external", nargs="*", default=[])
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    report = {}
    print(HEADER)
    for scene in arguments.scenes:
        report[scene] = measure(scene, arguments.runs, arguments.reps)
        rays = 0
        for folder, entry in report[scene].items():
            runner.attach_power("raytrace", scene, folder, {"power": entry["power"]})
            load = "; ".join(entry["stages"]["load"])
            print(row(scene, entry["label"], entry["stages"], entry["power"], load))
            counts = entry["stages"]["rays"]
            rays = (
                counts["rays_primary"] + counts["rays_shadow"] + counts["rays_bounce"]
            )
            rays += counts["rays_shadow2"]
        for line in external_rows(scene, arguments.external, rays):
            print(line)
    if arguments.json:
        with open(arguments.json, "w") as file:
            json.dump(report, file, indent=1)


if __name__ == "__main__":
    main()
