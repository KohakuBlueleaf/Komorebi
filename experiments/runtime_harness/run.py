"""Builds the runtime harness and runs its scenarios, by hand: python run.py [--tree
DIR] [--rows] [--tsan] [names...]; --build-only prints the program's path (reused while
the sources are unchanged). Exits non-zero if a scenario fails.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMS = {
    "one": [],
    "rows": ["-DKM_ROW_LEADERS=3", "-DKMSIM_TREE_ROW=3"],
    "tsan": ["-fsanitize=thread"],
}
ROWS = ["smoke", "exit_partial_team", "completion_every_team"]
PARTS = ("unpack", "math", "pack")


def units(tree):
    """(name, source, flags) of every translation unit of the program."""
    kernels = tree / "examples" / "01_mapcollect" / "kernels"
    trisc = ["-std=c++17", "-DCOMPILE_FOR_TRISC"]
    return [
        (
            "reader",
            kernels / "reader.cpp",
            ["-std=c++17", "-Dkernel_main=kmsim_reader_main"],
        ),
        (
            "writer",
            kernels / "writer.cpp",
            ["-std=c++17", "-Dkernel_main=kmsim_writer_main"],
        ),
        *[
            (
                p,
                kernels / "compute.cpp",
                [*trisc, f"-DTRISC_{p.upper()}", f"-Dkernel_main=kmsim_{p}_main"],
            )
            for p in PARTS
        ],
        *[
            (
                f"item_{p}",
                HERE / "item_kernels.cpp",
                [*trisc, f"-DTRISC_{p.upper()}", f"-DKMSIM_PART={p}"],
            )
            for p in PARTS
        ],
        ("sim", HERE / "sim.cpp", ["-std=c++20", "-Werror"]),
        ("scenarios", HERE / "scenarios.cpp", ["-std=c++20", "-Werror"]),
    ]


def build(out, tree=ROOT, form="one"):
    """Path of the scenario program for `form`, compiled into `out` unless current."""
    cxx = shutil.which("c++")
    include = [
        f"-I{HERE / 'stub'}",
        f"-I{HERE}",
        f"-I{tree / 'src' / 'komorebi_native' / 'include'}",
    ]
    common = [cxx, "-O1", "-g", "-Wall", *include, *FORMS[form]]
    todo = units(tree)
    sources = sorted(
        {
            *HERE.rglob("*.[ch]*"),
            *(tree / "src" / "komorebi_native" / "include").rglob("*.h*"),
        }
    )
    sources += sorted({src for _, src, _ in todo})
    key = hashlib.sha256(repr(common).encode())
    for src in sources:
        key.update(str(src).encode() + src.read_bytes())
    d = Path(out) / form
    program, stamp = d / "scenarios", d / "stamp"
    if program.exists() and stamp.exists() and stamp.read_text() == key.hexdigest():
        return program
    d.mkdir(parents=True, exist_ok=True)
    jobs = [
        [*common, *flags, "-c", str(src), "-o", str(d / f"{name}.o")]
        for name, src, flags in todo
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r in pool.map(
            lambda a: subprocess.run(a, capture_output=True, text=True), jobs
        ):
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-4000:])
    objs = [str(d / f"{name}.o") for name, _, _ in todo]
    subprocess.run([*common, *objs, "-pthread", "-o", str(program)], check=True)
    stamp.write_text(key.hexdigest())
    return program


def run(program, name, form="one"):
    """The JSON line of one scenario; thread sanitizer needs a fixed address layout."""
    fixed = ["setarch", platform.machine(), "-R"] if form == "tsan" else []
    r = subprocess.run(
        [*fixed, str(program), name], capture_output=True, text=True, timeout=600
    )
    if r.returncode != 0 or "ThreadSanitizer" in r.stderr:
        raise RuntimeError(f"{name} did not run cleanly:\n{r.stderr[-4000:]}")
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="scenarios (default: all)")
    ap.add_argument(
        "--tree", type=Path, default=ROOT, help="runtime headers and example"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "runtime_harness")
    ap.add_argument("--rows", action="store_true", help="also the row-leader tree")
    ap.add_argument("--tsan", action="store_true", help="also under thread sanitizer")
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    if a.build_only:
        print(build(a.out, a.tree))
        return 0
    every = subprocess.run(
        [build(a.out, a.tree), "list"], capture_output=True, text=True, check=True
    ).stdout.split()
    failed = 0
    for form in ["one", *(["rows"] if a.rows else []), *(["tsan"] if a.tsan else [])]:
        t0 = time.perf_counter()
        program = build(a.out, a.tree, form)
        print(f"{form}: built in {time.perf_counter() - t0:.1f} s")
        for name in a.names or (ROWS if form == "rows" else every):
            t0 = time.perf_counter()
            got = run(program, name, form)
            failed += not got["ok"]
            ms = (time.perf_counter() - t0) * 1000
            print(f"  {'ok  ' if got['ok'] else 'FAIL'} {name} ({ms:.0f} ms)")
            for line in got["fails"] + got["stuck"] + got["violations"]:
                print(f"       {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
