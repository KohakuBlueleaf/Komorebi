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
    return {name: (P[name], value) for name, value in pairs.items() if P[name] != value}


def table(entries, directory, profile_tensors, workers):
    """The thread's table: entries (kind, handler, ten words) in submit order on every
    core, kind 1 + the trace's index; the owners' directory; the profile tensors per
    trace and chunk."""
    if len(entries) > P["RP_COMMANDS_MAX"]:
        raise ValueError(
            f"{len(entries)} commands; the table holds {P['RP_COMMANDS_MAX']}"
        )
    table_words = np.zeros(P["RP_TABLE_WORDS"], dtype=np.uint32)
    for index, (kind, handler, words) in enumerate(entries):
        start = index * P["RP_REC_WORDS"]
        table_words[start : start + 3] = [kind, handler, workers << 16]
        table_words[start + 3 : start + 3 + len(words)] = words
    owners = directory[:, [K["kDirFirst"], K["kDirCount"]]].reshape(-1)
    table_words[P["RP_T_BASE"] : P["RP_T_BASE"] + len(owners)] = owners
    for trace, chunk_tensors in enumerate(profile_tensors):
        for chunk, tensor in enumerate(chunk_tensors):
            table_words[P["RP_T_PROF"] + trace * P["RP_CHUNKS"] + chunk] = (
                tensor.address
            )
    return table_words


def program(
    base, table_words, mode, frames, workers, packet_count, capacity, command_count
):
    """The program base with the planning thread on work cluster 1."""
    words = [0] * P["RP_WORDS"]
    words[P["RP_W_MODE"]] = MODES[mode]
    words[P["RP_W_FRAMES"]] = frames
    words[P["RP_W_WORKERS"]] = workers
    words[P["RP_W_BANK"]] = BANK_OF_CLUSTER[CLUSTER]
    words[P["RP_W_PACKETS"]] = packet_count
    words[P["RP_W_CAP"]] = capacity
    words[P["RP_W_COMMANDS"]] = command_count
    base.thread(
        cluster=CLUSTER,
        sources=[THREAD / "plan.c"],
        includes=[THREAD],
        regions={
            "table": table_words,
            "dirs": P["RP_DIRS_BYTES"],
            "cost": P["RP_COST_REGION_BYTES"],
            "work": WORK_BYTES,
            "log": P["RP_LOG_BYTES"],
        },
        words=words,
        deadline=20.0,
    )
    return base


def run(session, program_to_load, frames, meter, log, deadline_s=5.0):
    """Loads program_to_load under the power meter, waits for the thread; returns the
    log of each frame (wall, plan and chunks per trace, command ticks), its result, the
    state."""
    with meter:
        started = time.perf_counter()
        loaded = session.load(program_to_load, deadline=deadline_s)
        result = loaded.thread_result(timeout=30.0)
        host_s = time.perf_counter() - started
        log(f"thread: code {result.code} words {list(result.words)} ran {result.ran}")
        loaded.stop()
    rows = np.frombuffer(loaded.thread.read("log"), dtype=np.uint32)
    rows = rows[: frames * P["RP_LOG_WORDS"]].reshape(frames, P["RP_LOG_WORDS"])
    microseconds_per_tick = 1e6 / TIMER_HZ
    frame_logs = []
    for row in rows[: result.words[0] if result.words else 0]:
        submitted = int(row[P["RP_L_SUBMITTED"]])
        frame_logs.append(
            {
                "wall_microseconds": float(row[P["RP_L_FRAME"]])
                * microseconds_per_tick,
                "plan_microseconds": [
                    float(tick) * microseconds_per_tick for tick in row[1:5]
                ],
                "chunks": [int(count) for count in row[5:9]],
                "plan_parts_microseconds": [
                    float(tick) * microseconds_per_tick for tick in row[10:14]
                ],
                "command_microseconds": (
                    row[P["RP_L_COMMAND"] : P["RP_L_COMMAND"] + submitted]
                    * microseconds_per_tick
                ).tolist(),
            }
        )
    status = {
        "code": CODES.get(result.code, result.code),
        "ran": result.ran,
        "error": result.error,
        "words": list(result.words),
        "thread_microseconds": result.microseconds,
        "kernel": loaded.result(),
        "host_s": host_s,
    }
    return frame_logs, status, loaded.state(), meter.samples


def entries_of(records, steps, per_step, treelets):
    """Table entries from frame.commands' (handler, words) list (LOAD first): a trace's
    chunk commands become one entry of kind 1 + its index; the rest stay as they are."""
    entries = [(P["RP_KIND_FIRST"], *records[0])]
    position = 1
    for step, chunk_count in zip(steps, per_step):
        name, handler = step[0], step[1]
        if handler == "kHandlerTrace" and treelets:
            handler_word, words = records[position]
            words = list(words)
            words[0] = words[4] = 0
            entries.append((1 + TRACES.index(name), handler_word, words))
        else:
            entries.extend(
                (0, *records[position + chunk]) for chunk in range(chunk_count)
            )
        position += chunk_count
    return entries
