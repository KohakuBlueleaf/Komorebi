"""What the kernel-size tests share: the child that builds a program and reads its
SIZES row, the paths it needs, the room to leave."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHILD = ROOT / "src" / "komorebi" / "tools" / "kernel_sizes.py"
DESCRIPTOR = Path("tt_metal/third_party/umd/tests/cluster_descriptor_examples")
HEADROOM = 1024  # bytes of the buffer left after the five kernels' text


def build_sizes(environment, arguments):
    """The child's `SIZES` row: program, text per thread, total, buffer, fits."""
    done = subprocess.run(
        [sys.executable, str(CHILD), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    rows = [line for line in done.stdout.splitlines() if line.startswith("SIZES ")]
    assert rows, f"no build reached (exit {done.returncode}):\n{done.stderr[-2000:]}"
    return json.loads(rows[-1].removeprefix("SIZES "))
