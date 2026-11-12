"""Shared by the project harnesses: running a binary, storing a run as result files."""

import json
import os

import numpy as np

from ref import native, power, results


def threads_all():
    """The number of logical CPUs the process may use."""
    return len(os.sched_getaffinity(0))


def results_root(task):
    """Folder with one sub-folder of results per implementation."""
    return native.data_dir() / "results" / task


def out_dir(task, scene, impl):
    """Scratch folder for the raw .npy output of one binary on one scene."""
    path = native.data_dir() / "tmp" / task / scene / impl
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_binary(command, gpu, with_power):
    """Runs a binary; with power sampling when asked."""
    command = [str(c) for c in command]
    if with_power:
        return power.run_with_power(command, gpu=gpu)
    return native.run_json(command, gpu=gpu)


def stage_medians(record, stages):
    """Median of every timed stage over the repetitions listed in a binary's record."""
    reps = record.get("reps", [])
    return {k: float(np.median([r[k] for r in reps])) for k in stages} if reps else {}


def save_run(task, stages, scene, folder, name, arrays, record):
    """Stores an implementation's arrays and timing record in the results layout."""
    info = {"name": name, "scene": scene, "timings_ms": stage_medians(record, stages)}
    info["seconds_per_frame"] = record["loop_ms"] / record["reps_n"] / 1000
    rays = {k: v for k, v in record.items() if k.startswith("rays_")}
    if rays:
        info["rays"] = rays
    if "power" in record:
        info["power"] = record["power"]
    results.save_result(results_root(task) / folder, scene, task, arrays, info)


def attach_power(task, scene, folder, record):
    """Adds the power figures and frame time of a sampled run to a stored result."""
    path = results_root(task) / folder / f"{scene}.json"
    if path.exists():
        info = json.loads(path.read_text())
        info["seconds_per_frame"] = record["power"]["seconds_per_frame"]
        info["power"] = record["power"]
        path.write_text(json.dumps(info, indent=1))


def edge_mask(tri):
    """Pixels whose 3 x 3 neighbourhood holds more than one triangle index."""
    padded = np.pad(tri, 1, mode="edge")
    edge = np.zeros(tri.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            edge |= padded[dy : dy + tri.shape[0], dx : dx + tri.shape[1]] != tri
    return edge
