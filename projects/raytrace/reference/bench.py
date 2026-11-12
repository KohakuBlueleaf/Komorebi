"""Rays per second, time and power of the C++ (1 thread, all threads) and CUDA tracers.

A frame is the five stages in order: primary rays, shadow rays from the primary hits,
bounce rays, shadow rays from the bounce hits, shading. Mray/s columns: rays of the
stage(s) over their time; "shadow" is both shadow stages together; "together" is all
rays of the frame over the time of all five stages. Each cell is the median of RUNS
processes x REPS repetitions (after one untimed warm-up per process) with min and max.
Timed: the tree walks and the shading arithmetic of the whole 1024 x 1024 image. Not
timed: loading the scene and tree, upload, readback. Seconds per frame and energy come
from one more run of at least 3 s (see projects/raster/reference/bench.py).

    python -m projects.raytrace.reference.bench [--scenes ...] [--runs 3] [--json FILE]
"""

import argparse
import json
import math
import statistics

from projects.raytrace.reference import harness
from ref import runner

POWER_SECONDS = 3.0
RATES = ("primary", "shadow", "bounce", "together")


def runners(scene):
    """(folder, label, function(reps, with_power)) per implementation."""
    n = harness.threads_all()
    return [
        ("cpu1", "C++ 1 thread", lambda r, w: harness.run_cpu(scene, 1, r, False, w)),
        (
            "cpu_all",
            f"C++ {n} threads",
            lambda r, w: harness.run_cpu(scene, n, r, False, w),
        ),
        ("v100", "V100", lambda r, w: harness.run_gpu(scene, r, False, w)),
    ]


def rates(record, rep):
    """Mray/s of the stages of one repetition, and its total time."""
    r = record
    shadow_rays = r["rays_shadow"] + r["rays_shadow2"]
    total_rays = r["rays_primary"] + shadow_rays + r["rays_bounce"]
    return {
        "primary": r["rays_primary"] / rep["primary_ms"] / 1000,
        "shadow": shadow_rays / (rep["shadow_ms"] + rep["shadow2_ms"]) / 1000,
        "bounce": r["rays_bounce"] / rep["bounce_ms"] / 1000,
        "together": total_rays / rep["total_ms"] / 1000,
        "total_ms": rep["total_ms"],
    }


def summarize(records):
    """Median, min and max of each rate and the total time over all repetitions."""
    per_rep = [rates(rec, rep) for rec in records for rep in rec["reps"]]
    out = {}
    for key in (*RATES, "total_ms"):
        values = [p[key] for p in per_rep]
        out[key] = {"median": statistics.median(values), "min": min(values)}
        out[key].update({"max": max(values), "n": len(values)})
    out["load"] = sorted({f"{r['load_before']} / {r['load_after']}" for r in records})
    out["rays"] = {k: v for k, v in records[0].items() if k.startswith("rays_")}
    return out


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


def fmt(x, digits=1):
    return "n/a" if x is None else f"{x:.{digits}f}"


def cell(s):
    if s["median"] != s["median"]:
        return "n/a"
    return f"{s['median']:.2f} ({s['min']:.2f}-{s['max']:.2f})"


def row(scene, label, stages, power, load=""):
    cells = " | ".join(cell(stages[k]) for k in (*RATES, "total_ms"))
    gpu = "n/a"
    if "gpu_mean_w" in power:
        gpu = f"{power['gpu_idle_w']:.0f} / {power['gpu_mean_w']:.0f} / "
        gpu += f"{power['gpu_peak_w']:.0f}"
    joules = power.get("gpu_joules_per_frame")
    cpu = power.get("cpu_package_joules_per_frame")
    return (
        f"| {scene} | {label} | {cells} | {power['seconds_per_frame'] * 1000:.1f} | "
        f"{gpu} | {fmt(joules, 2)} | {fmt(cpu, 2)} | {load} |"
    )


def external_rows(scene, external, rays_per_frame):
    """Rows for implementations in --external folders that report a total time."""
    rows = []
    for name, (_, info) in harness.collect(scene, external).items():
        total = (info.get("timings_ms") or {}).get("total_ms")
        if name in ("numpy", "C++ 1 thread", "C++ all threads", "V100") or not total:
            continue
        rate = rays_per_frame / total / 1000
        stages = {k: {"median": float("nan"), "min": 0, "max": 0} for k in RATES}
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
    args = parser.parse_args()
    report = {}
    print(HEADER)
    for scene in args.scenes:
        report[scene] = measure(scene, args.runs, args.reps)
        rays = 0
        for folder, entry in report[scene].items():
            runner.attach_power("raytrace", scene, folder, {"power": entry["power"]})
            load = "; ".join(entry["stages"]["load"])
            print(row(scene, entry["label"], entry["stages"], entry["power"], load))
            r = entry["stages"]["rays"]
            rays = r["rays_primary"] + r["rays_shadow"] + r["rays_bounce"]
            rays += r["rays_shadow2"]
        for line in external_rows(scene, args.external, rays):
            print(line)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
