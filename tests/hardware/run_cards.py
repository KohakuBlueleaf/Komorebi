"""Runs card test files one per pytest process, in turn, each under a timeout.
Usage: run_cards.py [--timeout S] [--wrap CMD] [FILE ...] (default: every test_*.py);
`--wrap` goes before each pytest, {name} being the file's stem (a watcher job per file).
Prints one line per file; exits non-zero if any file failed."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRACE = 30.0  # seconds after SIGTERM before SIGKILL: a session closes the X280 runtime


def command(path, wrap):
    """The command line that runs one file: [wrap...] python -m pytest FILE -v."""
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(path),
        "-v",
        "-p",
        "no:cacheprovider",
    ]
    prefix = shlex.split(wrap.format(name=Path(path).stem)) if wrap else []
    return prefix + pytest_cmd


def run_file(path, timeout, wrap):
    """Runs one file in its own process; its exit code, or "timeout"."""
    env = {**os.environ, "KMR_CARD": "1"}
    child = subprocess.Popen(command(path, wrap), env=env)
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=GRACE)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        return "timeout"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="per file, seconds"
    )
    parser.add_argument(
        "--wrap", default="", help="command before pytest; {name}: file"
    )
    args = parser.parse_args()
    files = [file if file.exists() else HERE / file for file in args.files]
    files = files or sorted(HERE.glob("test_*.py"))
    failed = []
    for path in files:
        code = run_file(path, args.timeout, args.wrap)
        print(f"{Path(path).name}: {'ok' if code == 0 else code}", flush=True)
        if code != 0:
            failed.append(path)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
