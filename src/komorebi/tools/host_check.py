"""The fast check off the card: every line of every checks.txt under a folder of checks
(tests/host_checks by default) run at once, each under its own limit; a line a check,
the total. Usage: python -m komorebi.tools.host_check [--root R] [-k S]
"""

import argparse
import concurrent.futures
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CHECKS = Path("tests") / "host_checks"
LIST = "checks.txt"
LOGS = Path("build") / "host_check"  # under the root: a folder a run, by its time
KEEP = 10  # runs whose logs are kept
RUN = re.compile(r"\d{8}-\d{6}(-\d+)?")


@dataclass(frozen=True)
class Check:
    """One line of a checks.txt: its limit in seconds, its folder, the words of its
    command (`python`, `kmr`, `{root}`, `{here}`, `{build}` filled at the run)."""

    limit: float
    here: Path
    words: tuple

    @property
    def line(self):
        return f"{self.here.name}: {shlex.join(self.words)}"


def read_checks(folder):
    """Every check of the checks.txt files under `folder`, in path order."""
    checks = []
    for path in sorted(Path(folder).rglob(LIST)):
        for number, text in enumerate(path.read_text().splitlines(), 1):
            words = shlex.split(text, comments=True)
            if not words:
                continue
            try:
                limit = float(words[0])
            except ValueError:
                raise SystemExit(f"{path}:{number}: a limit in seconds first") from None
            checks.append(Check(limit, path.parent, tuple(words[1:])))
    return checks


def command_of(check, root, build):
    """(environment, working folder, argv) of a check: leading NAME=value words set
    the environment, `cwd=P` the folder (the root by default)."""
    places = {"root": str(root), "here": str(check.here), "build": str(build)}
    words = [word.format(**places) for word in check.words]
    env = dict(os.environ)
    paths = [str(root / "src"), str(root), str(check.here), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    cwd = root
    while words and "=" in words[0] and not words[0].startswith("-"):
        name, value = words.pop(0).split("=", 1)
        if name == "cwd":
            cwd = Path(value)
        else:
            env[name] = value
    if words and words[0] == "python":
        words[0] = sys.executable
    elif words and words[0] == "kmr":
        words[:1] = [sys.executable, "-m", "komorebi.cli"]
    return env, cwd, words


def run_one(index, check, root, build):
    """Runs one check under its limit; (index, check, passed, why, seconds)."""
    env, cwd, argv = command_of(check, root, build)
    log = build / f"{index:03d}-{check.here.name}.log"
    started = time.perf_counter()
    try:
        with open(log, "w") as out:
            out.write(f"$ (cd {cwd}; {shlex.join(argv)})\n")
            out.flush()
            done = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=check.limit,
                check=False,
            )
        why = None if done.returncode == 0 else f"exit {done.returncode}"
    except subprocess.TimeoutExpired:
        why = f"over its limit of {check.limit:g} s"
    except OSError as error:
        why = f"{type(error).__name__}: {error}"
    seconds = time.perf_counter() - started
    return (
        index,
        check,
        why is None,
        why if why is None else f"{why}; log {log}",
        seconds,
    )


def run_folder(logs):
    """A new folder for this run's logs under `logs`, named by its time; the oldest
    runs' folders past the last KEEP are removed."""
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs = [p for p in logs.iterdir() if p.is_dir() and RUN.fullmatch(p.name)]
    runs.sort(key=lambda p: (p.name[:15], int(p.name[16:] or 0)))
    last = runs[-1].name if runs else ""
    folder = logs / stamp  # a second run within the same second takes a number
    if last[:15] == stamp:
        folder = logs / f"{stamp}-{int(last[16:] or 0) + 1}"
    folder.mkdir()
    runs.append(folder)
    for old in runs[:-KEEP]:
        shutil.rmtree(old)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=".", help="the repository's root")
    parser.add_argument("--checks", help="the folder of checks (default: the root's)")
    parser.add_argument("-k", dest="only", help="only the checks whose line has this")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    build = run_folder(root / LOGS)
    checks = read_checks(Path(args.checks) if args.checks else root / CHECKS)
    checks = [check for check in checks if not args.only or args.only in check.line]
    print(f"{len(checks)} checks, {args.jobs} at a time; logs: {build}", flush=True)
    started, failed = time.perf_counter(), 0
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [
            pool.submit(run_one, index, check, root, build)
            for index, check in enumerate(checks)
        ]
        for future in concurrent.futures.as_completed(futures):
            _, check, passed, why, seconds = future.result()
            failed += not passed
            said = "" if passed else f"  ({why})"
            print(
                f"{'PASS' if passed else 'FAIL'} {seconds:6.1f} s  {check.line}{said}",
                flush=True,
            )
    total = time.perf_counter() - started
    print(f"{len(checks) - failed} of {len(checks)} passed in {total:.1f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
