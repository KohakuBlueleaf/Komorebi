"""Builds the runtime harness and runs its scenarios, by hand: python run.py [--tree
DIR] [--rows] [--tsan] [names...]; --build-only [--pinned] prints a program's path
(reused while the sources are unchanged). Exits non-zero if a scenario fails.
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
# Builds: one-level tree, row leaders, thread sanitizer; pinned: the tests' quick build.
FORMS = {
    "one": ["-O1", "-g"],
    "rows": ["-O1", "-g", "-DKM_ROW_LEADERS=3", "-DKMSIM_TREE_ROW=3"],
    "tsan": ["-O1", "-g", "-fsanitize=thread"],
    "pinned": ["-O0"],
}
HARNESS = ["sim", "stub_calls", "dispatcher", "messages_dispatch", "checks", "pinned"]
SIM_FAMILY = ["sim", "stub_calls", "dispatcher", "messages_dispatch"]
ROWS = ["smoke", "exit_partial_group", "completion_every_group"]
PARTS = ("unpack", "math", "pack")


def units(tree, main):
    """(name, source, flags) of every translation unit of a program."""
    kernels = tree / "examples" / "01_mapcollect" / "kernels"
    # Each compute thread is its own program on the card: -Dkm=km_<thread> and static
    # stub functions keep one thread's inline code from replacing another's at link.
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
                [
                    *trisc,
                    f"-DTRISC_{p.upper()}",
                    f"-Dkm=km_{p}",
                    f"-Dkernel_main=kmsim_{p}_main",
                ],
            )
            for p in PARTS
        ],
        *[
            (
                f"item_{p}",
                HERE / "item_kernels.cpp",
                [*trisc, f"-DTRISC_{p.upper()}", f"-Dkm=km_{p}", f"-DKMSIM_PART={p}"],
            )
            for p in PARTS
        ],
        *[(u, HERE / f"{u}.cpp", ["-std=c++17", "-Werror"]) for u in [*HARNESS, main]],
        *messages_units(tree),
    ]


def messages_units(tree):
    """The dynamic handler's generic reader and writer, the forward example's compute
    (per thread), and the scenarios that run them."""
    kernels = tree / "src" / "komorebi_native" / "kernels"
    forward = tree / "examples" / "05_forward" / "kernels" / "compute.cpp"
    trisc = ["-std=c++17", "-DCOMPILE_FOR_TRISC"]
    return [
        (
            "messages_reader",
            kernels / "messages_reader.cpp",
            ["-std=c++17", "-Dkernel_main=kmsim_messages_reader_main"],
        ),
        (
            "messages_writer",
            kernels / "messages_writer.cpp",
            ["-std=c++17", "-Dkernel_main=kmsim_messages_writer_main"],
        ),
        *[
            (
                f"forward_{p}",
                forward,
                [
                    *trisc,
                    f"-DTRISC_{p.upper()}",
                    f"-Dkm=km_{p}",
                    f"-Dkernel_main=kmsim_forward_{p}_main",
                ],
            )
            for p in PARTS
        ],
        (
            "messages_scenarios",
            HERE / "messages_scenarios.cpp",
            ["-std=c++17", "-Werror"],
        ),
    ]


def two_units(todo, d):
    """The harness's own files (same flags) as two translation units in `d`, those on
    sim_impl.hpp and the rest: each header family is parsed once. Kernels stay apart."""
    plain = ["-std=c++17", "-Werror"]
    own = [(name, src) for name, src, flags in todo if flags == plain]
    units = []
    for part in ("sim", "checks"):
        unit = d / f"harness_{part}.cpp"
        mine = [src for name, src in own if (name in SIM_FAMILY) == (part == "sim")]
        unit.write_text("".join(f'#include "{src}"\n' for src in mine))
        units.append((f"harness_{part}", unit, plain))
    return [t for t in todo if t[2] != plain] + units


def build(out, tree=ROOT, form="one"):
    """Path of the scenario program for `form`, compiled into `out` unless current."""
    cxx = shutil.which("c++")
    include = [
        f"-I{HERE / 'stub'}",
        f"-I{HERE}",
        f"-I{tree / 'src' / 'komorebi_native' / 'include'}",
        f"-I{tree / 'examples' / '05_forward' / 'kernels'}",
    ]
    # KM_WIRE: a link's transport as NoC writes (the stub has no fabric headers)
    common = [cxx, "-Wall", "-DKM_WIRE", *include, *FORMS[form]]
    todo = units(tree, "pinned_main" if form == "pinned" else "scenarios")
    sources = sorted(
        {
            *HERE.rglob("*.[ch]*"),
            *(tree / "src" / "komorebi_native" / "include").rglob("*.h*"),
            *(tree / "examples" / "05_forward" / "kernels").glob("*.h"),
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
    if form == "pinned":
        todo = two_units(todo, d)
    jobs = [
        [*common, *flags, "-c", str(src), "-o", str(d / f"{name}.o")]
        for name, src, flags in todo
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
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
    ap.add_argument("--pinned", action="store_true", help="the tests' quick build")
    a = ap.parse_args()
    if a.build_only:
        print(build(a.out, a.tree, "pinned" if a.pinned else "one"))
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
