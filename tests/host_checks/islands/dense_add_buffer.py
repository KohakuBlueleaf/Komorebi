"""The dense addition's two-card kernels fit linear.py's own kernel buffer: its prebuild
on the mock builds both islands' loads, with no KernelSizeError."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SHAPE = ["--shapes", "32,4096,4096:in", "--cards", "0,1"]


def main():
    script = ROOT / "projects" / "dense" / "linear.py"
    command = [sys.executable, "-m", "komorebi.cli", "prebuild", "--cards", "0,1"]
    done = subprocess.run(
        [*command, str(script), *SHAPE], capture_output=True, text=True, check=False
    )
    out = done.stdout + done.stderr
    built = [line for line in out.splitlines() if "kernel buffer: " in line]
    stats = [line for line in out.splitlines() if "JIT cache stats" in line]
    print("\n".join(built + stats))  # the stats: whether it compiled (host_check)
    if "KernelSizeError" in out or not any("island 1" in line for line in built):
        print("FAIL: the addition's kernels do not fit, or island 1 built nothing")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
