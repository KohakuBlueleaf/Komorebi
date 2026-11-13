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
T0 = time.monotonic()
M32 = 0xFFFFFFFF
FNV_BASIS, FNV_PRIME = 2166136261, 16777619
WC_AGAIN, WC_END, WC_NO_END = 0, 1, 2
PATHS = ("cpu", "hostfed", "thread")


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def initial(cores):
    """Every core's state word before the first round."""
    return (np.arange(cores, dtype=np.uint64) * 7) & M32


def replay(cores, count, limit, max_rounds, base=0):
    """The program in numpy on cores [base, base + cores), apart from decide.c: (code,
    rounds, sum, trail, groups, final state of those cores)."""
    p = np.arange(base, base + cores, dtype=np.uint64)
    x, first, trail, groups = initial(base + cores)[base:], 0, FNV_BASIS, []
    for k in range(max_rounds):
        groups.append(base + first)
        group = slice(first, first + count)
        x[group] = (x[group] + np.uint64(k + 1) + p[group]) & M32
        total = int(x.sum()) & M32
        trail = ((trail ^ first) * FNV_PRIME) & M32
        if total >= limit:
            return 0, k + 1, total, trail, groups, x
        first = total % (cores - count + 1)
    return 2, max_rounds, total, trail, groups, x


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
    cc = shutil.which("cc")
    if cc is None:
        raise SystemExit("this path needs a C compiler (cc)")
    includes = [HERE, native.kmx_dir(), native.includes()[0]]
    key = native.source_key([*sources, *includes], "/")
    out = native.cache_dir() / "examples" / f"{name}-{key}" / name

    def make(path):
        flags = ["-shared", "-fPIC", "-ffreestanding"] if shared else []
        cmd = [cc, "-std=c11", "-O2", "-Wall", "-Werror", *flags]
        cmd += [f"-I{d}" for d in includes] + [str(s) for s in sources]
        done = subprocess.run([*cmd, "-o", str(path)], capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f"{' '.join(cmd)}\n{done.stderr}")

    if not out.is_file():
        native.build_whole(out, make)
    return out


def decision():
    """decide.c as a host library."""
    lib = ctypes.CDLL(str(build("libdecide.so", [HERE / "decide.c"], shared=True)))
    lib.wc_start.argtypes = [ctypes.POINTER(State), ctypes.POINTER(ctypes.c_uint32)]
    lib.wc_round.argtypes = [ctypes.POINTER(State), ctypes.POINTER(Command)]
    lib.wc_result.argtypes = [ctypes.POINTER(State), ctypes.c_uint32]
    for f in (lib.wc_start, lib.wc_result):
        f.restype = ctypes.c_uint32
    lib.wc_round.restype = None
    return lib


def hostfed(lib, words, send):
    """decide.c's loop on the host; send(commands) submits a round's two commands and
    returns SUM's result word, or None if one did not finish. Returns (code, rounds,
    sum, trail, groups, commands), with the codes of thread/driver.c."""
    s, recs, commands, groups = State(), (Command * 2)(), [], []
    if lib.wc_start(ctypes.byref(s), (ctypes.c_uint32 * 6)(*words)):
        return 1, 0, 0, 0, groups, commands
    while True:
        lib.wc_round(ctypes.byref(s), recs)
        groups.append(recs[0].first)
        commands += [r.row() for r in recs]
        total = send(recs)
        if total is None:
            return 4, s.round, s.sum, s.trail, groups, commands
        step = lib.wc_result(ctypes.byref(s), total)
        if step != WC_AGAIN:
            code = 0 if step == WC_END else 2
            return code, s.round, s.sum, s.trail, groups, commands


def model(cores, base=0):
    """A send() over the grid in numpy (kernels/writer.cpp's STEP and SUM)."""
    x = initial(base + cores).copy()

    def send(recs):
        step, total = recs
        p = np.arange(step.first, step.first + step.count, dtype=np.uint64)
        x[p] = (x[p] + np.uint64(step.a1) + p) & M32
        return int(x[total.first : total.first + total.a1].sum()) & M32

    return send


# The paths ----------------------------------------------------------------------


def cpu_path(words):
    """The thread's own C files on the CPU against the fake ring."""
    sources = [HERE / "cpu_ring.c", HERE / "thread" / "driver.c", HERE / "decide.c"]
    exe = build("cpu_ring", sources, shared=False)
    done = subprocess.run([exe, *map(str, words)], capture_output=True, text=True)
    lines = done.stdout.split("\n")
    commands = [
        tuple(map(int, s.split()[1:])) for s in lines if s.startswith("command ")
    ]
    result = next(map(int, s.split()[1:]) for s in lines if s.startswith("result "))
    groups = next(
        list(map(int, s.split()[1:])) for s in lines if s.startswith("groups")
    )
    code, rounds, total, trail = (list(result) + [0, 0, 0])[:4]
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
    prog = km.Program(
        "work_cluster",
        kernels=HERE / "kernels",
        buffers=[km.Buffer(index=16, format="uint32", page_bytes=64, pages=1)],
        compile_args={"writer": state.accessor_args()},
    )
    for cluster, words in loops.items():
        prog.thread(
            cluster=cluster,
            sources=[HERE / "thread" / "driver.c", HERE / "decide.c"],
            includes=[HERE],
            regions={"groups": 4 * words[4]},
            words=words,
            host_fallback=fallbacks[cluster],
            deadline=10.0,
        )
    return prog


def host_fallback(lib, commands):
    """The thread on the host: decide.c's loop submitting through the run, region 0 and
    the result words as the thread leaves them; the commands submitted go to `commands`.
    """

    def fallback(regions, words, run):
        code, rounds, total, trail, groups, submitted = hostfed(
            lib, words, send_run(run)
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
    lib, got, ms = decision(), {}, {}
    with km.Session(log=log) as session:
        log("device open")
        for path in which:
            state = state_tensor(session, args.cores)
            mine = {c: [state.address, *w[1:]] for c, w in loops.items()}
            commands = {c: [] for c in loops}
            fallbacks = {c: host_fallback(lib, commands[c]) for c in loops}
            prog = program(state, mine, fallbacks)
            where = "host" if path == "hostfed" else "card"
            run = session.load(prog, deadline=1.0, thread=where)
            log(f"{path}: loaded, threads on the {where} on clusters {list(loops)}")
            t = time.monotonic()
            results = run.thread_results(timeout=15.0)
            took = time.monotonic() - t
            for c, r in results.items():
                loop_s = took if path == "hostfed" else r.us / 1e6
                ms.setdefault(path, {})[c] = round(loop_s * 1e3, 2)
                log(f"{path} on cluster {c}: {r.state}, code {r.code}, "
                    f"{ms[path][c]} ms")  # fmt: skip
            run.stop()
            log(f"{path}: stopped")
            x = read_state(state, args.cores)
            got[path] = {}
            for c, r in results.items():
                groups = run.threads[c].read("groups").view(np.uint32)
                rounds = r.words[0] if r.words else 0
                out = (r.code, *(list(r.words) + [0, 0, 0])[:3])
                out = (*out, [int(t) for t in groups[:rounds]], commands[c] or None)
                base, cores = loops[c][5], loops[c][1]
                got[path][c] = (*out, x[base : base + cores])
    return got, ms


def send_run(run):
    """A send() through a Run: both commands at once, SUM exclusive, so it waits for
    STEP on the card; SUM's result word comes back with its event."""
    k = abi.headers()

    def send(recs):
        handles = [
            run.submit(
                r.handler,
                group=(r.first, r.count),
                args=[r.a0, r.a1],
                exclusive=bool(r.flags & k["KM_F_EXCL"]),
                result=bool(r.flags & k["KM_F_RESULT"]),
            )
            for r in recs
        ]
        try:
            events = run.wait(handles, timeout=5.0)
        except km.KomorebiError as err:
            log(f"a command did not finish: {err}")
            return None
        return events[1].result if all(e.kind == "done" for e in events) else None

    return send


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cores", type=int, default=119, help="the grid's cores used")
    p.add_argument("--count", type=int, default=30, help="cores of each STEP group")
    p.add_argument("--limit", type=int, default=1_000_000, help="the sum that ends it")
    p.add_argument("--max-rounds", type=int, default=1000)
    p.add_argument("--paths", nargs="+", choices=PATHS, default=list(PATHS))
    p.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[1],
        help="a thread on each, on its own share of the cores, all at once",
    )
    args = p.parse_args()
    share = args.cores // len(args.clusters)
    loops = {
        c: [0, share, args.count, args.limit, args.max_rounds, i * share]
        for i, c in enumerate(args.clusters)
    }
    out = {"cores": args.cores, "count": args.count, "limit": args.limit}
    out["clusters"] = args.clusters
    want, got, ms, checks = {}, {}, {}, []
    for c, words in loops.items():
        code, rounds, total, trail, groups, x = replay(*words[1:5], base=words[5])
        want[c] = ([code, rounds, total, trail, groups], x)
        model_out = hostfed(decision(), words, model(words[1], words[5]))
        checks.append(list(model_out[:5]) == want[c][0])
        want[c] = (*want[c], model_out[5])
        out[f"replay_{c}"] = {"code": code, "rounds": rounds, "sum": total}
    out["decide_matches_numpy"] = all(checks)
    if "cpu" in args.paths:
        got["cpu"] = {c: (*cpu_path(w), None) for c, w in loops.items()}
    on_card = [path for path in ("hostfed", "thread") if path in args.paths]
    if on_card:
        card, ms = card_paths(args, loops, on_card)
        got.update(card)
    for path, per in got.items():
        out[path] = {}
        for c, (code, n, s, t, tm, commands, state) in per.items():
            loop = {"code": code, "rounds": n, "sum": s}
            loop["same_as_numpy"] = [code, n, s, t, tm] == want[c][0]
            if commands is not None:
                # the address word differs by run; the rest must be decide.c's commands
                loop["same_commands"] = [r[:4] + r[5:] for r in commands] == [
                    r[:4] + r[5:] for r in want[c][2]
                ]
            if state is not None:
                loop["state_wrong"] = int((state != want[c][1]).sum())
                loop["ms"] = ms[path][c]
            out[path][c] = loop
            checks += [v for k, v in loop.items() if k.startswith("same")]
            checks.append(loop.get("state_wrong", 0) == 0)
    out["wall_s"] = round(time.monotonic() - T0, 1)
    checks.append(set(got) == set(args.paths))  # every path asked for ran
    out["ok"] = all(checks)
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
