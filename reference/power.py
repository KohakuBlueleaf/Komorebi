"""Power beside a timing: GPU draw from nvidia-smi, CPU package energy if readable.
A binary prints LOOP_START and LOOP_END on stderr around its timed loop; the samples
inside give the mean and peak. No root needed; what the host does not expose is None.
"""

import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from reference import native

INTERVAL_MS = 100
RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")


def cpu_energy_uj():
    """Package energy counter in microjoules, or None when it cannot be read."""
    try:
        return int(RAPL.read_text())
    except (OSError, ValueError):
        return None


class GpuSampler:
    """Samples power.draw of one GPU with nvidia-smi in a background process."""

    def __init__(self, index):
        self.samples = []
        self.proc = None
        smi = shutil.which("nvidia-smi")
        if smi is None:
            return
        cmd = [smi, "-i", str(index), "--query-gpu=power.draw"]
        cmd += ["--format=csv,noheader,nounits", f"-lms={INTERVAL_MS}"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in self.proc.stdout:
            with contextlib.suppress(ValueError):
                self.samples.append((time.monotonic(), float(line)))

    def between(self, start, end):
        """Watts of the samples taken in [start, end] (monotonic seconds)."""
        return [w for t, w in self.samples if start <= t <= end]

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait()


def stderr_marks(stream, marks):
    """Records the monotonic time (and CPU energy) at LOOP_START and LOOP_END."""
    for line in stream:
        word = line.strip()
        if word in ("LOOP_START", "LOOP_END"):
            marks[word] = (time.monotonic(), cpu_energy_uj())


def run_with_power(command, gpu=None, idle_s=1.0):
    """Runs a binary that prints one JSON line; returns it with a "power" record."""
    env = dict(os.environ)
    sampler = None
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        sampler = GpuSampler(gpu)
        time.sleep(idle_s)
    idle_end = time.monotonic()
    load_before = native.loadavg()
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    marks = {}
    lines = []
    readers = [
        threading.Thread(target=stderr_marks, args=(proc.stderr, marks)),
        threading.Thread(target=lambda: lines.extend(proc.stdout)),
    ]
    for reader in readers:
        reader.start()
    proc.wait()
    for reader in readers:
        reader.join()
    out = "".join(lines)
    if sampler is not None:
        time.sleep(INTERVAL_MS / 1000)
        sampler.stop()
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command)
    result = json.loads(out.strip().splitlines()[-1])
    frames, loop_s = result["reps_n"], result["loop_ms"] / 1000
    power = {"seconds_per_frame": loop_s / frames}
    (t0, e0), (t1, e1) = marks["LOOP_START"], marks["LOOP_END"]
    if e0 is not None and e1 is not None:
        power["cpu_package_joules_per_frame"] = (e1 - e0) * 1e-6 / frames
    else:
        power["cpu_package_joules_per_frame"] = None
        power["cpu_package_note"] = "not available: the energy counter is not readable"
    if sampler is not None:
        idle = sampler.between(idle_end - idle_s, idle_end)
        inside = sampler.between(t0, t1)
        if idle and inside:
            power["gpu_idle_w"] = sum(idle) / len(idle)
            power["gpu_mean_w"] = sum(inside) / len(inside)
            power["gpu_peak_w"] = max(inside)
            power["gpu_samples"] = len(inside)
            power["gpu_joules_per_frame"] = power["gpu_mean_w"] * loop_s / frames
    result["power"] = power
    result["load_before"], result["load_after"] = load_before, native.loadavg()
    return result
