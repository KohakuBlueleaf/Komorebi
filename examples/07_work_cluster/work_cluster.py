"""Work cluster: a control loop submits commands to changing groups until a sum computed
on the card ends it; its decision (decide.c) runs in the X280 thread, host-fed through
ctypes, and in the thread on the CPU against a fake ring. Every path must match a numpy
replay. Prints one JSON line. See README.md."""

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import abi, native

HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()
MASK_32_BITS = 0xFFFFFFFF
FNV_BASIS, FNV_PRIME = 2166136261, 16777619
WC_AGAIN, WC_END, WC_NO_END = 0, 1, 2
PATHS = ("cpu", "hostfed", "thread")


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def initial(cores):
    """Every core's state word before the first round."""
    return (np.arange(cores, dtype=np.uint64) * 7) & MASK_32_BITS


def replay(cores, count, limit, max_rounds, base=0):
    """The program in numpy on cores [base, base + cores), apart from decide.c: (code,
    rounds, sum, trail, groups, final state of those cores)."""
    core_ids = np.arange(base, base + cores, dtype=np.uint64)
    state, first, trail, groups = initial(base + cores)[base:], 0, FNV_BASIS, []
    for round_index in range(max_rounds):
        groups.append(base + first)
        group = slice(first, first + count)
        state[group] = (
            state[group] + np.uint64(round_index + 1) + core_ids[group]
        ) & MASK_32_BITS
        total = int(state.sum()) & MASK_32_BITS
        trail = ((trail ^ first) * FNV_PRIME) & MASK_32_BITS
        if total >= limit:
            return 0, round_index + 1, total, trail, groups, state
        first = total % (cores - count + 1)
    return 2, max_rounds, total, trail, groups, state


# The decision through ctypes ----------------------------------------------------


class State(ctypes.Structure):
    _fields_ = [
        ("word", ctypes.c_uint32 * 6),
        ("round", ctypes.c_uint32),
        ("first", ctypes.c_uint32),
        ("sum", ctypes.c_uint32),
        ("trail", ctypes.c_uint32),
    ]


class Command(ctypes.Structure):
    NAMES = ("handler", "flags", "first", "count", "a0", "a1")
    _fields_ = [(name, ctypes.c_uint32) for name in NAMES]

    def row(self):
        return tuple(getattr(self, name) for name in self.NAMES)


def build(name, sources, shared):
    """Builds `sources` with the host's C compiler into the cache, once per content."""
    compiler = shutil.which("cc")
    if compiler is None:
        raise SystemExit("this path needs a C compiler (cc)")
    includes = [HERE, native.kmx_directory(), native.includes()[0]]
    key = native.source_key([*sources, *includes], "/")
    output = native.cache_directory() / "examples" / f"{name}-{key}" / name

    def make(path):
        flags = ["-shared", "-fPIC", "-ffreestanding"] if shared else []
        command = [compiler, "-std=c11", "-O2", "-Wall", "-Werror", *flags]
        command += [f"-I{directory}" for directory in includes]
        command += [str(source) for source in sources]
        completed = subprocess.run(
            [*command, "-o", str(path)], capture_output=True, text=True
        )
        if completed.returncode:
            raise SystemExit(f"{' '.join(command)}\n{completed.stderr}")

    if not output.is_file():
        native.build_whole(output, make)
    return output


def decision():
    """decide.c as a host library."""
    library = ctypes.CDLL(str(build("libdecide.so", [HERE / "decide.c"], shared=True)))
    library.wc_start.argtypes = [ctypes.POINTER(State), ctypes.POINTER(ctypes.c_uint32)]
    library.wc_round.argtypes = [ctypes.POINTER(State), ctypes.POINTER(Command)]
    library.wc_result.argtypes = [ctypes.POINTER(State), ctypes.c_uint32]
    for function in (library.wc_start, library.wc_result):
        function.restype = ctypes.c_uint32
    library.wc_round.restype = None
    return library


def hostfed(library, words, send):
    """decide.c's loop on the host; send(commands) submits a round's two commands and
    returns SUM's result word, or None if one did not finish. Returns (code, rounds,
    sum, trail, groups, commands), with the codes of thread/driver.c."""
    state, round_commands, commands, groups = State(), (Command * 2)(), [], []
    if library.wc_start(ctypes.byref(state), (ctypes.c_uint32 * 6)(*words)):
        return 1, 0, 0, 0, groups, commands
    while True:
        library.wc_round(ctypes.byref(state), round_commands)
        groups.append(round_commands[0].first)
        commands += [command.row() for command in round_commands]
        total = send(round_commands)
        if total is None:
            return 4, state.round, state.sum, state.trail, groups, commands
        outcome = library.wc_result(ctypes.byref(state), total)
        if outcome != WC_AGAIN:
            code = 0 if outcome == WC_END else 2
            return code, state.round, state.sum, state.trail, groups, commands


def model(cores, base=0):
    """A send() over the grid in numpy (kernels/writer.cpp's STEP and SUM)."""
    state = initial(base + cores).copy()

    def send(round_commands):
        step_command, sum_command = round_commands
        pages = np.arange(
            step_command.first, step_command.first + step_command.count, dtype=np.uint64
        )
        state[pages] = (
            state[pages] + np.uint64(step_command.a1) + pages
        ) & MASK_32_BITS
        return (
            int(state[sum_command.first : sum_command.first + sum_command.a1].sum())
            & MASK_32_BITS
        )

    return send


# The paths ----------------------------------------------------------------------


def cpu_path(words):
    """The thread's own C files on the CPU against the fake ring."""
    sources = [HERE / "cpu_ring.c", HERE / "thread" / "driver.c", HERE / "decide.c"]
    executable = build("cpu_ring", sources, shared=False)
    finished = subprocess.run(
        [executable, *map(str, words)], capture_output=True, text=True
    )
    lines = finished.stdout.split("\n")
    commands = [
        tuple(map(int, line.split()[1:]))
        for line in lines
        if line.startswith("command ")
    ]
    result_words = next(
        map(int, line.split()[1:]) for line in lines if line.startswith("result ")
    )
    groups = next(
        list(map(int, line.split()[1:])) for line in lines if line.startswith("groups")
    )
    code, rounds, total, trail = (list(result_words) + [0, 0, 0])[:4]
    return code, rounds, total, trail, groups[:rounds], commands


def state_tensor(session, cores):
    """One 64-byte page per core, word 0 its state; the address is a program word."""
    rows = np.zeros((abi.headers()["KM_CORES_MAX"], 16), np.uint32)
    rows[:cores, 0] = initial(cores)
    return session.tensor(rows, "uint32")


def read_state(state, cores):
    rows = state.read()
    return rows[:cores, 0].astype(np.uint64)


def program(state, loops, fallbacks):
    """The grid's kernels and one X280 thread per loop ({cluster: words}) that drives
    them, each with its host fallback."""
    work_program = km.Program(
        "work_cluster",
        kernels=HERE / "kernels",
        buffers=[km.Buffer(index=16, format="uint32", page_bytes=64, pages=1)],
        compile_args={"writer": state.accessor_args()},
    )
    for cluster, words in loops.items():
        work_program.thread(
            cluster=cluster,
            sources=[HERE / "thread" / "driver.c", HERE / "decide.c"],
            includes=[HERE],
            regions={"groups": 4 * words[4]},
            words=words,
            host_fallback=fallbacks[cluster],
            deadline=10.0,
        )
    return work_program


def host_fallback(library, commands):
    """The thread on the host: decide.c's loop submitting through the run, region 0 and
    the result words as the thread leaves them; the commands submitted go to `commands`.
    """

    def fallback(regions, words, run):
        code, rounds, total, trail, groups, submitted = hostfed(
            library, words, send_run(run)
        )
        regions["groups"][: len(groups)] = groups
        commands.extend(submitted)
        return code, [rounds, total, trail]

    return fallback


def card_paths(args, loops, which):
    """One program, its threads on the host (`hostfed`) or on the card (`thread`),
    each on its own state tensor, every loop at once. Returns {path: {cluster: (code,
    rounds, sum, trail, groups, commands or None, loop cores' state)}} and each path's
    time per loop in ms (the thread's own on the card, the fallbacks' on the host)."""
    library, got, milliseconds = decision(), {}, {}
    with km.Session(log=log) as session:
        log("device open")
        for path in which:
            state = state_tensor(session, args.cores)
            words_by_cluster = {
                cluster: [state.address, *words[1:]] for cluster, words in loops.items()
            }
            commands = {cluster: [] for cluster in loops}
            fallbacks = {
                cluster: host_fallback(library, commands[cluster]) for cluster in loops
            }
            work_program = program(state, words_by_cluster, fallbacks)
            where = "host" if path == "hostfed" else "card"
            run = session.load(work_program, deadline=1.0, thread=where)
            log(f"{path}: loaded, threads on the {where} on clusters {list(loops)}")
            started = time.monotonic()
            results = run.thread_results(timeout=15.0)
            took = time.monotonic() - started
            for cluster, result in results.items():
                loop_seconds = took if path == "hostfed" else result.microseconds / 1e6
                milliseconds.setdefault(path, {})[cluster] = round(
                    loop_seconds * 1e3, 2
                )
                log(f"{path} on cluster {cluster}: {result.state}, code {result.code}, "
                    f"{milliseconds[path][cluster]} ms")  # fmt: skip
            run.stop()
            log(f"{path}: stopped")
            final_state = read_state(state, args.cores)
            got[path] = {}
            for cluster, result in results.items():
                groups = run.threads[cluster].read("groups").view(np.uint32)
                rounds = result.words[0] if result.words else 0
                loop_result = (result.code, *(list(result.words) + [0, 0, 0])[:3])
                loop_result = (
                    *loop_result,
                    [int(first) for first in groups[:rounds]],
                    commands[cluster] or None,
                )
                base, cores = loops[cluster][5], loops[cluster][1]
                got[path][cluster] = (*loop_result, final_state[base : base + cores])
    return got, milliseconds


def send_run(run):
    """A send() through a Run: both commands at once, SUM exclusive, so it waits for
    STEP on the card; SUM's result word is returned with its event."""
    headers = abi.headers()

    def send(round_commands):
        handles = [
            run.submit(
                command.handler,
                group=(command.first, command.count),
                args=[command.a0, command.a1],
                exclusive=bool(command.flags & headers["KM_FLAG_EXCLUSIVE"]),
                result=bool(command.flags & headers["KM_FLAG_RESULT"]),
            )
            for command in round_commands
        ]
        try:
            events = run.wait(handles, timeout=5.0)
        except km.KomorebiError as error:
            log(f"a command did not finish: {error}")
            return None
        return (
            events[1].result if all(event.kind == "done" for event in events) else None
        )

    return send


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=119, help="the grid's cores used")
    parser.add_argument(
        "--count", type=int, default=30, help="cores of each STEP group"
    )
    parser.add_argument(
        "--limit", type=int, default=1_000_000, help="the sum that ends it"
    )
    parser.add_argument("--max-rounds", type=int, default=1000)
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=list(PATHS))
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[1],
        help="a thread on each, on its own partition of the cores, all at once",
    )
    args = parser.parse_args()
    cores_per_loop = args.cores // len(args.clusters)
    loops = {
        cluster: [
            0,
            cores_per_loop,
            args.count,
            args.limit,
            args.max_rounds,
            index * cores_per_loop,
        ]
        for index, cluster in enumerate(args.clusters)
    }
    report = {"cores": args.cores, "count": args.count, "limit": args.limit}
    report["clusters"] = args.clusters
    want, got, milliseconds, checks = {}, {}, {}, []
    for cluster, words in loops.items():
        code, rounds, total, trail, groups, state = replay(*words[1:5], base=words[5])
        want[cluster] = ([code, rounds, total, trail, groups], state)
        model_out = hostfed(decision(), words, model(words[1], words[5]))
        checks.append(list(model_out[:5]) == want[cluster][0])
        want[cluster] = (*want[cluster], model_out[5])
        report[f"replay_{cluster}"] = {"code": code, "rounds": rounds, "sum": total}
    report["decide_matches_numpy"] = all(checks)
    if "cpu" in args.paths:
        got["cpu"] = {
            cluster: (*cpu_path(words), None) for cluster, words in loops.items()
        }
    on_card = [path for path in ("hostfed", "thread") if path in args.paths]
    if on_card:
        card, milliseconds = card_paths(args, loops, on_card)
        got.update(card)
    for path, loops_of_path in got.items():
        report[path] = {}
        for cluster, (
            code,
            rounds,
            total,
            trail,
            groups,
            commands,
            state,
        ) in loops_of_path.items():
            loop = {"code": code, "rounds": rounds, "sum": total}
            loop["same_as_numpy"] = [code, rounds, total, trail, groups] == want[
                cluster
            ][0]
            if commands is not None:
                # the address word differs by run; the rest must be decide.c's commands
                loop["same_commands"] = [row[:4] + row[5:] for row in commands] == [
                    row[:4] + row[5:] for row in want[cluster][2]
                ]
            if state is not None:
                loop["state_wrong"] = int((state != want[cluster][1]).sum())
                loop["ms"] = milliseconds[path][cluster]
            report[path][cluster] = loop
            checks += [value for key, value in loop.items() if key.startswith("same")]
            checks.append(loop.get("state_wrong", 0) == 0)
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    checks.append(set(got) == set(args.paths))  # every path asked for ran
    report["ok"] = all(checks)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
