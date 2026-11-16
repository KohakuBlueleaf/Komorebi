"""Commands to an X280 thread: a command whose group names a work cluster goes to that
cluster's thread, which answers with the word of its kmx_program; every word checked,
the times beside a command to one grid core. Prints one JSON line. See README.md."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import abi
from komorebi.program import thread
from komorebi.session.process import aiclk_mhz

HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()
M32 = 0xFFFFFFFF
TICKS_PER_MICROSECOND = 50  # the X280 timer
CONSTANTS = abi.constants(HERE / "tc.h")
WINDOW = 8  # commands in flight to one thread: its command lines


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def echo(a0, a1):
    """tc_echo of tc.h."""
    return (((a0 * 0x9E3779B1) & M32) ^ a1) + 0x7F4A7C15 & M32


def grid_word(a0):
    """tc_grid of tc.h."""
    return a0 ^ 0xA5A5A5A5


def percentiles(values):
    values = np.asarray(values, float)
    if not values.size:
        return {}
    low, median, high = np.percentile(values, [1, 50, 99])
    return {
        "n": int(values.size),
        "mean": round(float(values.mean()), 3),
        "p1": round(float(low), 3),
        "median": round(float(median), 3),
        "p99": round(float(high), 3),
    }


def program(count, client, server, core):
    """The grid's kernels, the commanded thread on `server` and, if `client`, the
    commanding thread there."""
    made = km.Program("thread_commands", kernels=HERE / "kernels")
    made.thread(
        cluster=server,
        sources=[HERE / "x280" / "program.c"],
        includes=[HERE],
        regions={"unused": 64},
        deadline=60.0,
    )
    if client:
        made.thread(
            cluster=client,
            sources=[HERE / "x280" / "client.c"],
            includes=[HERE],
            regions={"ticks": max(8 * count, 64)},
            words=[count, server, core],
            deadline=30.0,
        )
    return made


def to_thread(run, handler, cluster, args, result=True):
    """A command whose group names `cluster`'s thread (KM_GROUP_CLUSTER); its handle."""
    h = run.h
    flags = h["KM_FLAG_RESULT"] if result else 0
    group = (h["KM_GROUP_CLUSTER_BASE"] + cluster, 1)
    return run._queue_command(handler, group, args, flags=flags)


def expected(index, core_count):
    """Host command `index` of the checked series: (handler, args, word)."""
    if index % 16 == 15:
        core = index % core_count
        return CONSTANTS["KM_TC_SUBMIT"], [index, core], (grid_word(index) + 1) & M32
    a1 = index ^ 0x5A5A5A5A
    return CONSTANTS["KM_TC_ECHO"], [index, a1], echo(index, a1)


def checked(run, server, count):
    """`count` commands to the thread in waves of WINDOW, every 16th a KM_TC_SUBMIT
    whose function submits a grid command and waits for it; wrong words and events."""
    wrong, not_done, first_wrong = 0, 0, None
    for start in range(0, count, WINDOW):
        wave = range(start, min(start + WINDOW, count))
        handles = []
        for index in wave:
            handler, args, _ = expected(index, run.cores)
            handles.append(to_thread(run, handler, server, args))
        for index, event in zip(wave, run.wait(handles, timeout=5.0), strict=True):
            if event.kind != "done":
                not_done += 1
            elif event.result != expected(index, run.cores)[2]:
                wrong += 1
                first_wrong = index if first_wrong is None else first_wrong
        if start % 2048 == 0:
            log(f"checked: {start + len(wave)} of {count}")
    return {"count": count, "wrong": wrong, "not_done": not_done, "first": first_wrong}


def mixed(run, server, waves):
    """Waves of four commands to the thread and four KM_TC_GRID_RESULT commands to one
    grid core each, interleaved, in flight together; wrong words of each kind."""
    wrong = {"thread": 0, "grid": 0}
    not_done = 0
    for wave in range(waves):
        handles, words = [], []
        for k in range(WINDOW):
            index = wave * WINDOW + k
            if k % 2 == 0:
                a1 = index ^ 0x5A5A5A5A
                handles.append(
                    to_thread(run, CONSTANTS["KM_TC_ECHO"], server, [index, a1])
                )
                words.append(("thread", echo(index, a1)))
            else:
                core = index % run.cores
                handle = run.submit(
                    CONSTANTS["KM_TC_GRID_RESULT"], group=(core, 1), args=[index],
                    result=True,
                )  # fmt: skip
                handles.append(handle)
                words.append(("grid", grid_word(index)))
        for (kind, word), event in zip(
            words, run.wait(handles, timeout=5.0), strict=True
        ):
            if event.kind != "done":
                not_done += 1
            elif event.result != word:
                wrong[kind] += 1
    return {"waves": waves, "wrong": wrong, "not_done": not_done}


def latency(run, server, count, core):
    """One command at a time, alternately to the thread (KM_TC_ECHO, no result flag: the
    event's ticks are the dispatcher's time) and a NOP to one grid core: host and
    dispatcher microseconds of each."""
    times = {"thread": ([], []), "grid": ([], [])}
    for index in range(2 * count):
        kind = "thread" if index % 2 == 0 else "grid"
        started = time.perf_counter()
        if kind == "thread":
            handle = to_thread(
                run, CONSTANTS["KM_TC_ECHO"], server, [index, 0], result=False
            )
        else:
            handle = run.submit(0, group=(core, 1))
        event = run.wait(handle, timeout=5.0)
        took = time.perf_counter() - started
        if event.kind != "done":
            raise SystemExit(f"latency: command {index} ended {event.kind}")
        times[kind][0].append(took * 1e6)
        times[kind][1].append(event.ticks / TICKS_PER_MICROSECOND)
    return {
        kind: {"host": percentiles(host), "dispatcher": percentiles(dispatcher)}
        for kind, (host, dispatcher) in times.items()
    }


def snapshot(run, cluster):
    """The thread's ring area as its LIM holds it: the kernel's consumed and event
    counts, and the first four words of each command line the kernel handed it."""
    thread_run = run.threads[cluster]
    area = thread.layout()["KMX_LIM_BYTES"]
    raw = thread_run.port.work_read(
        cluster, area, thread.ring_area_bytes(thread_run.ring)
    )
    words = np.frombuffer(bytes(raw), np.uint32)
    lines = run.h["KMD_WORK_EVENTS"] + 16 * thread_run.ring.event_depth
    in_lines = words[lines // 4 :].reshape(-1, 16)[:, :4]
    return {
        "consumed": int(words[0]),
        "told": int(words[16]),
        "lines": in_lines.tolist(),
    }


def case_run(session, args):
    """Every check and time but the deadline, in one load."""
    out = {}
    made = program(args.count, args.client, args.server, args.core)
    run = session.load(made, deadline=1.0)
    log(f"loaded: thread on {args.server}, commanded by the host and {args.client}")
    client = run.threads[args.client].wait(args.count * 0.002 + 10.0)
    log(f"client thread: {client.state} ({client.error}), code {client.code}, words "
        f"{client.words}, at {client.where}")  # fmt: skip
    log(f"thread {args.server}'s ring: {snapshot(run, args.server)}")
    out["client"] = {
        "state": client.state,
        "error": client.error,
        "code": client.code,
        "wrong": client.words[0] if client.words else None,
        "not_done": client.words[1] if client.words else None,
        "windowed_microseconds_per_command": (
            {
                "thread": round(
                    client.words[2] / TICKS_PER_MICROSECOND / args.count, 3
                ),
                "grid": round(client.words[3] / TICKS_PER_MICROSECOND / args.count, 3),
            }
            if len(client.words) >= 4
            else None
        ),
    }
    if client.state != "done":
        server = run.threads[args.server].collect(False)
        log(f"server thread: {server.state} ({server.error}), at {server.where}")
        raise SystemExit("the client thread failed")
    clocks = [aiclk_mhz()]
    out["host_checked"] = checked(run, args.server, args.count)
    clocks.append(aiclk_mhz())
    log(f"host checked: {out['host_checked']}")
    out["mixed"] = mixed(run, args.server, args.mix)
    log(f"mixed: {out['mixed']}")
    out["latency"] = latency(run, args.server, args.latency, args.core)
    clocks.append(aiclk_mhz())
    out["aiclk_mhz"] = clocks
    log(f"latency: {out['latency']}")
    started = time.perf_counter()
    for start in range(0, args.count, WINDOW):
        handles = [
            to_thread(
                run, CONSTANTS["KM_TC_ECHO"], args.server, [index, 0], result=False
            )
            for index in range(start, min(start + WINDOW, args.count))
        ]
        run.wait(handles, timeout=5.0)
    out["host_windowed_microseconds_per_command"] = round(
        (time.perf_counter() - started) * 1e6 / args.count, 3
    )
    server = run.threads[args.server].collect(False)
    log(f"server thread: {server.state}, error {server.error}, stuck {server.stuck}")
    out["server"] = {
        "state": server.state,
        "error": server.error,
        "stuck": server.stuck,
    }
    result = run.stop()
    log("stopped")
    ticks = run.threads[args.client].read("ticks").view(np.uint32)
    out["client_microseconds"] = {
        "thread": percentiles(ticks[: args.count] / TICKS_PER_MICROSECOND),
        "grid": percentiles(ticks[args.count : 2 * args.count] / TICKS_PER_MICROSECOND),
    }
    out["dispatcher"] = {
        "error": result["error_name"],
        "commands": result.get("commands"),
        "results": result.get("results"),
        "rejected": result.get("rejected"),
    }
    out["ok"] = bool(
        client.state == "done"
        and client.code == 0
        and client.words[:2] == [0, 0]
        and out["host_checked"]["wrong"] == 0
        and out["host_checked"]["not_done"] == 0
        and out["mixed"]["wrong"] == {"thread": 0, "grid": 0}
        and out["mixed"]["not_done"] == 0
        and server.error == "none"
        and not server.stuck
        and result["error"] == 0
    )
    return out


def case_deadline(session, args):
    """A command whose function does not return within the deadline: its timeout
    event, grid commands meanwhile, then the thread's commands again once it returned.
    """
    made = program(0, None, args.server, args.core)
    deadline_s = 0.02
    run = session.load(made, deadline=deadline_s)
    spin_s = 0.2
    log(f"loaded, deadline {deadline_s} s; the thread spins {spin_s} s")
    started = time.monotonic()
    spin = to_thread(
        run, CONSTANTS["KM_TC_SPIN"], args.server, [int(spin_s * 50e6), 7], result=True
    )
    event = run.wait(spin, timeout=5.0)
    out = {
        "spin": {
            "kind": event.kind,
            "auxiliary": event.auxiliary,
            "microseconds": event.ticks / TICKS_PER_MICROSECOND,
            "host_seconds": round(time.monotonic() - started, 4),
        }
    }
    log(f"spin's event: {out['spin']}")
    handles = [
        run.submit(CONSTANTS["KM_TC_GRID_RESULT"], group=(i % run.cores, 1), args=[i],
                   result=True)
        for i in range(100)
    ]  # fmt: skip
    events = run.wait(handles, timeout=5.0)
    out["grid_meanwhile"] = {
        "done_right": sum(
            event.kind == "done" and event.result == grid_word(i)
            for i, event in enumerate(events)
        ),
        "host_seconds": round(time.monotonic() - started, 4),
    }
    log(f"grid commands while the thread spins: {out['grid_meanwhile']}")
    time.sleep(max(spin_s + 0.05 - (time.monotonic() - started), 0))
    right = 0
    for start in range(0, 100, WINDOW):
        wave = range(start, min(start + WINDOW, 100))
        handles = [
            to_thread(run, CONSTANTS["KM_TC_ECHO"], args.server, [i, i]) for i in wave
        ]
        events = run.wait(handles, timeout=5.0)
        right += sum(
            event.kind == "done" and event.result == echo(i, i)
            for i, event in zip(wave, events, strict=True)
        )
    out["thread_after"] = {"done_right": right}
    log(f"thread commands after it returned: {out['thread_after']}")
    server = run.threads[args.server].collect(False)
    result = run.stop()
    log("stopped")
    out["server"] = {
        "state": server.state,
        "error": server.error,
        "stuck": server.stuck,
    }
    out["dispatcher"] = {"error": result["error_name"]}
    out["ok"] = bool(
        event.kind == "timeout"
        and out["grid_meanwhile"]["done_right"] == 100
        and right == 100
        and server.error == "none"
        and not server.stuck
        and result["error"] == 0
    )
    return out


CASES = {"run": case_run, "deadline": case_deadline}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--count", type=int, default=10000, help="commands of a kind")
    parser.add_argument("--mix", type=int, default=1000, help="mixed waves of eight")
    parser.add_argument("--latency", type=int, default=1000, help="timed of a kind")
    parser.add_argument("--server", type=int, default=2, help="the commanded cluster")
    parser.add_argument("--client", type=int, default=1, help="the commanding one")
    parser.add_argument("--core", type=int, default=0, help="grid core of the NOPs")
    args = parser.parse_args()
    out = {"case": args.case, "load": Path("/proc/loadavg").read_text().split()[:3]}
    with km.Session(log=log) as session:
        log("device open")
        out.update(CASES[args.case](session, args))
    print(json.dumps(out), flush=True)
    raise SystemExit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
