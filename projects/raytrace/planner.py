"""The frame driven by a thread on X280 work cluster 1 (`thread/plan.c`): it submits the
commands and, before each trace, cuts the trace's packets into chunk commands of equal
predicted cost on its four X280 cores (frame.py --plan MODE)."""

import re
import time
from pathlib import Path

import numpy as np

from komorebi.thread import BANK_OF_CLUSTER, TIMER_HZ
from projects.raytrace import packets

K = packets.K
THREAD = Path(__file__).resolve().parent / "thread"
PLAN_H = THREAD / "plan.h"
P = {
    name: int(value.rstrip("u"), 0)
    for name, value in re.findall(
        r"^#define (RP_\w+)\s+(\d+u?)\b", PLAN_H.read_text(), re.M
    )
}
P["RP_T_BASE"] = P["RP_COMMANDS_MAX"] * P["RP_REC_WORDS"]
P["RP_T_PROF"] = P["RP_T_BASE"] + 2 * P["RP_CORES_MAX"]
P["RP_TABLE_WORDS"] = P["RP_T_PROF"] + P["RP_TRACES"] * P["RP_CHUNKS"]
_BLOCKS = P["RP_TRACES"] * P["RP_CHUNKS"] * P["RP_CORES_MAX"]
P["RP_DIRS_BYTES"] = _BLOCKS * P["RP_DIR_BYTES"]
P["RP_COST_REGION_BYTES"] = _BLOCKS * P["RP_COST_BYTES"]
P["RP_LOG_BYTES"] = P["RP_FRAMES_MAX"] * P["RP_LOG_WORDS"] * 4
CLUSTER = 1
MODES = {"host": 0, "equal": 1, "counts": 2, "trace": 3, "frame": 4}
TRACES = ("primary", "shadow", "bounce", "shadow2")
CODES = {0: "ok", 1: "bad arguments", 2: "ring", 3: "command", 4: "plan"}
WORK_BYTES = 2 << 20  # plan.c work_t: cycles and hits per trace, prediction, prefix


def same_numbers():
    """The numbers plan.h takes from the kernels' header, as pairs that differ."""
    pairs = {
        "RP_HANDLER_TREELETS": K["kHandlerTreelets"],
        "RP_BANK_SHIFT": K["kBankShift"],
        "RP_DIR_BYTES": K["kDirBytes"],
        "RP_DIR_FIRST": K["kDirFirst"],
        "RP_DIR_COUNT": K["kDirCount"],
        "RP_DIR_COST": K["kDirCost"],
        "RP_COST_HEAD": K["kCostHead"],
        "RP_COST_MAX": K["kCostMax"],
        "RP_COST_BYTES": K["kCostBytes"],
    }
    return {k: (P[k], v) for k, v in pairs.items() if P[k] != v}


def table(entries, directory, prof_ts, workers):
    """The thread's table: entries (kind, handler, ten words) in submit order on every
    core, kind 1 + t for trace t; the owners' directory; the profile tensors per trace
    and chunk."""
    if len(entries) > P["RP_COMMANDS_MAX"]:
        raise ValueError(
            f"{len(entries)} commands; the table holds {P['RP_COMMANDS_MAX']}"
        )
    t = np.zeros(P["RP_TABLE_WORDS"], dtype=np.uint32)
    for i, (kind, handler, words) in enumerate(entries):
        at = i * P["RP_REC_WORDS"]
        t[at : at + 3] = [kind, handler, workers << 16]
        t[at + 3 : at + 3 + len(words)] = words
    base = directory[:, [K["kDirFirst"], K["kDirCount"]]].reshape(-1)
    t[P["RP_T_BASE"] : P["RP_T_BASE"] + len(base)] = base
    for tr, ts in enumerate(prof_ts):
        for k, pt in enumerate(ts):
            t[P["RP_T_PROF"] + tr * P["RP_CHUNKS"] + k] = pt.address
    return t


def program(prog, tab, mode, frames, workers, packets_, cap, commands):
    """prog with the planning thread on work cluster 1."""
    words = [0] * P["RP_WORDS"]
    words[P["RP_W_MODE"]] = MODES[mode]
    words[P["RP_W_FRAMES"]] = frames
    words[P["RP_W_WORKERS"]] = workers
    words[P["RP_W_BANK"]] = BANK_OF_CLUSTER[CLUSTER]
    words[P["RP_W_PACKETS"]] = packets_
    words[P["RP_W_CAP"]] = cap
    words[P["RP_W_COMMANDS"]] = commands
    prog.thread(
        cluster=CLUSTER,
        sources=[THREAD / "plan.c"],
        includes=[THREAD],
        regions={
            "table": tab,
            "dirs": P["RP_DIRS_BYTES"],
            "cost": P["RP_COST_REGION_BYTES"],
            "work": WORK_BYTES,
            "log": P["RP_LOG_BYTES"],
        },
        words=words,
        deadline=20.0,
    )
    return prog


def run(session, prog, frames, meter, log, deadline_s=5.0):
    """Loads prog under the power meter, waits for the thread; returns the log of each
    frame (wall, plan and chunks per trace, command ticks), its result, the state."""
    with meter:
        t0 = time.perf_counter()
        r = session.load(prog, deadline=deadline_s)
        res = r.thread_result(timeout=30.0)
        host_s = time.perf_counter() - t0
        log(f"thread: code {res.code} words {list(res.words)} ran {res.ran}")
        r.stop()
    rows = np.frombuffer(r.thread.read("log"), dtype=np.uint32)
    rows = rows[: frames * P["RP_LOG_WORDS"]].reshape(frames, P["RP_LOG_WORDS"])
    us = 1e6 / TIMER_HZ
    out = []
    for f in rows[: res.words[0] if res.words else 0]:
        n = int(f[P["RP_L_SUBMITTED"]])
        out.append(
            {
                "wall_microseconds": float(f[P["RP_L_FRAME"]]) * us,
                "plan_microseconds": [float(x) * us for x in f[1:5]],
                "chunks": [int(x) for x in f[5:9]],
                "plan_parts_microseconds": [float(x) * us for x in f[10:14]],
                "command_microseconds": (
                    f[P["RP_L_COMMAND"] : P["RP_L_COMMAND"] + n] * us
                ).tolist(),
            }
        )
    status = {
        "code": CODES.get(res.code, res.code),
        "ran": res.ran,
        "error": res.error,
        "words": list(res.words),
        "thread_microseconds": res.microseconds,
        "kernel": r.result(),
        "host_s": host_s,
    }
    return out, status, r.state(), meter.samples


def entries_of(recs, steps, per_step, treelets):
    """Table entries from frame.commands' (handler, words) list (LOAD first): a trace's
    chunk commands become one entry of kind 1 + its index; the rest stay as they are."""
    out = [(P["RP_KIND_FIRST"], *recs[0])]
    at = 1
    for step, k in zip(steps, per_step):
        name, handler = step[0], step[1]
        if handler == "kHandlerTrace" and treelets:
            s, words = recs[at]
            words = list(words)
            words[0] = words[4] = 0
            out.append((1 + TRACES.index(name), s, words))
        else:
            out.extend((0, *recs[at + j]) for j in range(k))
        at += k
    return out
