"""Running a card script as a child process for a hardware test: its last JSON line, and
an end with a timeout. A child still running at the timeout gets SIGTERM, so its session
closes the X280 runtime before it exits; SIGKILL follows only after a grace period."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

GRACE = 30.0  # seconds a script has to close its session after SIGTERM


def run_text(script, *args, timeout=120.0, grace=GRACE):
    """Runs `python script args...`; returns its stdout. Fails the test on a timeout or
    a non-zero exit, with the child's stderr."""
    child = subprocess.Popen(
        [sys.executable, str(script), *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        out, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            _, stderr = child.communicate(timeout=grace)
            how = f"SIGTERM, exit {child.returncode}"
        except subprocess.TimeoutExpired:
            child.kill()
            _, stderr = child.communicate()
            how = f"SIGKILL after {grace} s"
        pytest.fail(
            f"{Path(script).name} did not return in {timeout} s ({how})\n"
            f"{stderr[-2000:]}"
        )
    if child.returncode != 0:
        pytest.fail(f"{Path(script).name} exited {child.returncode}\n{stderr[-2000:]}")
    return out


def run_json(script, *args, timeout=120.0, grace=GRACE):
    """run_text, then the last stdout line that starts with "{", parsed."""
    out = run_text(script, *args, timeout=timeout, grace=grace)
    lines = [line for line in out.splitlines() if line.startswith("{")]
    return json.loads(lines[-1])
