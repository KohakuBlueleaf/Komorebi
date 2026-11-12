"""Work cluster: a control loop issues records to changing teams until a sum computed
on the card ends it; its decision (decide.c) runs in the X280 thread, host-fed through
ctypes, and in the thread on the CPU against a fake ring. Every path must match a numpy
replay. Prints one JSON line. See README.md."""

import argparse
import ctypes
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

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


def replay(cores, count, limit, max_rounds):
    """The program in numpy, apart from decide.c: (code, rounds, sum, trail, teams,
    final state)."""
    p = np.arange(cores, dtype=np.uint64)
    x, first, trail, teams = initial(cores), 0, FNV_BASIS, []
    for k in range(max_rounds):
        teams.append(first)
        team = slice(first, first + count)
        x[team] = (x[team] + np.uint64(k + 1) + p[team]) & M32
        total = int(x.sum()) & M32
        trail = ((trail ^ first) * FNV_PRIME) & M32
        if total >= limit:
            return 0, k + 1, total, trail, teams, x
        first = total % (cores - count + 1)
    return 2, max_rounds, total, trail, teams, x


# The decision through ctypes ----------------------------------------------------


class State(ctypes.Structure):
    _fields_ = [
        ("word", ctypes.c_uint32 * 5),
        ("round", ctypes.c_uint32),
        ("first", ctypes.c_uint32),
        ("sum", ctypes.c_uint32),
        ("trail", ctypes.c_uint32),
    ]


class Record(ctypes.Structure):
    NAMES = ("seg", "flags", "first", "count", "a0", "a1")
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
    lib.wc_round.argtypes = [ctypes.POINTER(State), ctypes.POINTER(Record)]
    lib.wc_result.argtypes = [ctypes.POINTER(State), ctypes.c_uint32]
    for f in (lib.wc_start, lib.wc_result):
        f.restype = ctypes.c_uint32
    lib.wc_round.restype = None
    return lib


def hostfed(lib, words, send):
    """decide.c's loop on the host; send(records) issues a round's two records and
    returns SUM's result word, or None if one did not finish. Returns (code, rounds,
    sum, trail, teams, records), with the codes of thread/driver.c."""
    s, recs, records, teams = State(), (Record * 2)(), [], []
    if lib.wc_start(ctypes.byref(s), (ctypes.c_uint32 * 5)(*words)):
        return 1, 0, 0, 0, teams, records
    while True:
        lib.wc_round(ctypes.byref(s), recs)
        teams.append(recs[0].first)
        records += [r.row() for r in recs]
        total = send(recs)
        if total is None:
            return 4, s.round, s.sum, s.trail, teams, records
        step = lib.wc_result(ctypes.byref(s), total)
        if step != WC_AGAIN:
            code = 0 if step == WC_END else 2
            return code, s.round, s.sum, s.trail, teams, records


def model(cores):
    """A send() over the grid in numpy (kernels/writer.cpp's STEP and SUM)."""
    x = initial(cores).copy()

    def send(recs):
        step, _ = recs
        p = np.arange(step.first, step.first + step.count, dtype=np.uint64)
        x[p] = (x[p] + np.uint64(step.a1) + p) & M32
        return int(x.sum()) & M32

    return send


# The paths ----------------------------------------------------------------------


def cpu_path(words):
    """The thread's own C files on the CPU against the fake ring."""
    sources = [HERE / "cpu_ring.c", HERE / "thread" / "driver.c", HERE / "decide.c"]
    exe = build("cpu_ring", sources, shared=False)
    done = subprocess.run([exe, *map(str, words)], capture_output=True, text=True)
    lines = done.stdout.split("\n")
    records = [tuple(map(int, s.split()[1:])) for s in lines if s.startswith("record ")]
    result = next(map(int, s.split()[1:]) for s in lines if s.startswith("result "))
    teams = next(list(map(int, s.split()[1:])) for s in lines if s.startswith("teams"))
    code, rounds, total, trail = (list(result) + [0, 0, 0])[:4]
    return code, rounds, total, trail, teams[:rounds], records


def state_tensor(device, cores):
    """One 64-byte page per core, word 0 its state; the address is a program word."""
    rows = np.zeros((abi.headers()["KM_CORES_MAX"], 16), np.uint32)
    rows[:cores, 0] = initial(cores)
    host = torch.from_numpy(rows.view(np.int32))
    return ttnn.from_torch(
        host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def read_state(state, cores):
    rows = ttnn.to_torch(ttnn.from_device(state)).numpy().view(np.uint32)
    return rows[:cores, 0].astype(np.uint64)


def program(state, words, fallback):
    """The grid's kernels and the X280 thread that drives them, with `fallback` as
    its host fallback."""
    prog = km.Program(
        "work_cluster",
        kernels=HERE / "kernels",
        buffers=[km.Buffer(index=16, format="uint32", page_bytes=64, pages=1)],
        compile_args={"writer": ttnn.TensorAccessorArgs(state).get_compile_time_args()},
    )
    prog.thread(
        cluster=1,
        sources=[HERE / "thread" / "driver.c", HERE / "decide.c"],
        includes=[HERE],
        regions={"teams": 4 * words[4]},
        words=words,
        host_fallback=fallback,
        deadline=10.0,
    )
    return prog


def host_fallback(lib, records):
    """The thread on the host: decide.c's loop issuing through the run, region 0 and
    the result words as the thread leaves them; the records issued go to `records`."""

    def fallback(regions, words, run):
        code, rounds, total, trail, teams, issued = hostfed(lib, words, send_run(run))
        regions["teams"][: len(teams)] = teams
        records.extend(issued)
        return code, [rounds, total, trail]

    return fallback


def card_paths(args, words, which):
    """One program, its thread on the host (`hostfed`) or on the card (`thread`), each
    run on its own state tensor. Returns {path: (code, rounds, sum, trail, teams,
    records or None, state)}."""
    lib, got = decision(), {}
    with km.Session(log=log) as session:
        log("device open")
        for path in which:
            state = state_tensor(session.device, args.cores)
            w = [state.buffer_address(), *words[1:]]
            records = []
            prog = program(state, w, host_fallback(lib, records))
            where = "host" if path == "hostfed" else "card"
            run = session.load(prog, deadline=1.0, thread=where)
            log(f"{path}: loaded, thread on the {where}")
            t = time.monotonic()
            r = run.thread_result(timeout=15.0)
            log(f"{path}: {r.state}, code {r.code}, {time.monotonic() - t:.3f} s")
            run.stop()
            log(f"{path}: stopped")
            teams = run.thread.read("teams").view(np.uint32)
            rounds = r.words[0] if r.words else 0
            out = (r.code, *(list(r.words) + [0, 0, 0])[:3])
            out = (*out, [int(t) for t in teams[:rounds]], records or None)
            got[path] = (*out, read_state(state, args.cores))
    return got


def send_run(run):
    """A send() through a Run: both records at once, SUM exclusive, so it waits for
    STEP on the card; SUM's result word comes back with its event."""
    k = abi.headers()

    def send(recs):
        tickets = [
            run.issue(
                r.seg,
                team=(r.first, r.count),
                args=[r.a0, r.a1],
                exclusive=bool(r.flags & k["KM_F_EXCL"]),
                result=bool(r.flags & k["KM_F_RESULT"]),
            )
            for r in recs
        ]
        try:
            events = run.wait(tickets, timeout=5.0)
        except km.KomorebiError as err:
            log(f"a record did not finish: {err}")
            return None
        return events[1].result if all(e.kind == "done" for e in events) else None

    return send


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cores", type=int, default=119, help="the grid's cores used")
    p.add_argument("--count", type=int, default=30, help="cores of each STEP team")
    p.add_argument("--limit", type=int, default=1_000_000, help="the sum that ends it")
    p.add_argument("--max-rounds", type=int, default=1000)
    p.add_argument("--paths", nargs="+", choices=PATHS, default=list(PATHS))
    args = p.parse_args()
    words = [0, args.cores, args.count, args.limit, args.max_rounds]
    code, rounds, total, trail, teams, x = replay(*words[1:])
    want = {"code": code, "rounds": rounds, "sum": total, "trail": trail}
    out = {"cores": args.cores, "count": args.count, "limit": args.limit, **want}
    model_out = hostfed(decision(), words, model(args.cores))
    out["decide_matches_numpy"] = list(model_out[:5]) == [*want.values(), teams]
    got = {}
    if "cpu" in args.paths:
        got["cpu"] = (*cpu_path(words), None)
    on_card = [path for path in ("hostfed", "thread") if path in args.paths]
    if on_card:
        got.update(card_paths(args, words, on_card))
    for path, (c, n, s, t, tm, records, state) in got.items():
        out[path] = {
            "code": c,
            "rounds": n,
            "sum": s,
            "same_as_numpy": [c, n, s, t, tm] == [*want.values(), teams],
        }
        if records is not None:
            # the address word differs by run; the rest must be decide.c's records
            same = [r[:4] + r[5:] for r in records] == [
                r[:4] + r[5:] for r in model_out[5]
            ]
            out[path]["same_records"] = same
        if state is not None:
            out[path]["state_wrong"] = int((state != x).sum())
    out["wall_s"] = round(time.monotonic() - T0, 1)
    checks = [out["decide_matches_numpy"]]
    for path in got:
        checks += [v for k, v in out[path].items() if k.startswith("same")]
        checks.append(out[path].get("state_wrong", 0) == 0)
    out["ok"] = all(checks)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
