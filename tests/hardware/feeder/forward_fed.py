"""The forward example's dynamic run fed by the test feeder on the last core (no X280):
messages forward between the group's cores until their forwards are used up; every
core's totals are checked against the numpy replay. One JSON line. python forward_fed.py
--count 3"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXAMPLE = ROOT / "examples" / "05_forward"
sys.path[:0] = [str(HERE), str(EXAMPLE), str(ROOT / "src")]

import commands as rec  # noqa: E402
import fed  # noqa: E402
import replay  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi.layout import counter_tiles  # noqa: E402
from komorebi.program import Dynamic  # noqa: E402

K = rec.K
FORWARD = rec.constants(EXAMPLE / "kernels" / "forward.h")
KERNELS = ROOT / "src" / "komorebi_native" / "kernels"


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--count", type=int, default=3, help="group: cores 0 to count - 1")
    p.add_argument("--messages", type=int, default=1)
    p.add_argument("--forwards", type=int, default=1, help="forwards per message")
    p.add_argument(
        "--one-core", type=int, default=-1, help="every forward to this rank"
    )
    p.add_argument("--stall", type=int, default=-1, help="rank that stalls")
    p.add_argument("--probe", type=int, default=-1, help="rank that spends its quota")
    p.add_argument("--max-live", type=int, default=1 << 14)
    p.add_argument("--queue-messages", type=int, default=1 << 14)
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--idle-cycles", type=int, default=20000)
    p.add_argument("--counters", type=int, default=2, help="counter tiles of the group")
    p.add_argument("--deadline", type=float, default=5.0, help="the run, seconds")
    p.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    return p.parse_args()


def main():
    a = parse()
    try:
        messages = Dynamic(
            handler=FORWARD["kHandlerForward"],
            max_live=a.max_live,
            queue_messages=a.queue_messages,
            bins=a.bins,
            idle_cycles=a.idle_cycles,
            counters=a.counters,
        )
        messages.check(119, 4 << 30, 4)
        messages.check(a.count, 4 << 30, 4)
    except ValueError as err:  # the launcher's rejection, before any device opens
        print(json.dumps({"ok": False, "rejected": str(err)}), flush=True)
        sys.exit(1)
    kernels = fed.Kernels(
        reader=KERNELS / "messages_reader.cpp",
        compute=EXAMPLE / "kernels" / "compute.cpp",
        writer=KERNELS / "messages_writer.cpp",
        cbs=[(b.index, "uint32", b.page_bytes, b.pages) for b in messages.buffers()],
    )
    run = rec.command(
        FORWARD["kHandlerForward"],
        0,
        a.count,
        [
            a.messages,
            a.forwards,
            FORWARD["kModeOneCore"] if a.one_core >= 0 else FORWARD["kModeRandom"],
            max(a.one_core, 0),
            a.stall + 1,
            a.probe + 1,
        ],
        flags=K["KM_FLAG_EXCLUSIVE"],
    )
    with km.Session(log=fed.log) as session:
        fed.log("device open")
        f = fed.Fed(session)
        res_t = f.l1_blocks(FORWARD["kResWords"])
        result = f.run(
            kernels,
            [run],
            program_args=lambda core: ([], [], [res_t.address]),
            tensors=[res_t],
            depth=4,
            deadline_s=a.deadline,
            wait_s=a.wait,
            dynamic=messages,
        )
        res = rec.as_uint32(fed.read_int32(res_t))[: a.count]
        fed.log("results read back")
    print(json.dumps(report(a, messages, result, res)), flush=True)


def report(a, messages, result, res):
    """The checks and numbers of one run as a dict."""
    sequence = int(result.commands["sequence"][0])
    lines = result.lines[: a.count]
    gave_at = (K["KM_CONTROL_MESSAGES"] + K["KM_MESSAGES_GAVE"]) // 4
    gave = {
        c: [int(v) for v in result.control[c, gave_at : gave_at + 4]]
        for c in range(a.count)
        if result.control[c, gave_at]
    }
    m = len(counter_tiles(0, a.count, a.counters).workers)  # the rest: counter tiles
    one = a.one_core if a.one_core >= 0 else None
    consumed, total, visits = replay.replay(a.messages, a.forwards, m, one)
    if a.probe >= 0:  # the probe's messages visit their own core once
        visits[a.probe] += res[a.probe, FORWARD["kResSent"]]
    totals = res[:m, FORWARD["kResTotalLo"]].astype(np.uint64) | (
        res[:m, FORWARD["kResTotalHi"]].astype(np.uint64) << np.uint64(32)
    )
    out = {
        "feeder": result.summary,
        "run_status": int(result.commands["status"][0]),
        "run_sequence": sequence,
        "exact": bool((res[:, FORWARD["kResRun"]] == sequence).all())
        and bool((res[:m, FORWARD["kResConsumed"]] == consumed).all())
        and bool((totals == total).all())
        and bool((res[:m, FORWARD["kResVisits"]] == visits).all())
        and not res[m:, FORWARD["kResVisits"]].any(),
        "consumed_by_compute": int(res[:, FORWARD["kResConsumed"]].sum()),
        "expected_consumed": int(consumed.sum()),
        "forwards": int(res[:, FORWARD["kResVisits"]].sum()) - a.messages,
        "nulls": int(res[:, FORWARD["kResNull"]].sum()),
        "probe_sent": int(res[:, FORWARD["kResSent"]].sum()),
        "rejected": int(res[:, FORWARD["kResRejected"]].sum()),
        "quota_start": messages.quota(result.lines.shape[0]),
        "quota_left_min": int(res[:m, FORWARD["kResQuota"]].min()),
        "lines_not_idle": [
            c
            for c in range(a.count)
            if lines[c, K["KM_MESSAGES_LINE_RUN"]] != sequence
            or lines[c, K["KM_MESSAGES_LINE_STATE"]] != K["KM_MESSAGES_IDLE"]
        ],
        "sent": int(lines[:, K["KM_MESSAGES_LINE_SENT"]].sum()),
        "consumed": int(lines[:, K["KM_MESSAGES_LINE_CONSUMED"]].sum()),
        "messages_gave": gave,
        "state_gave": {
            c: {k: s[k] for k in ("reader", "writer", "compute") if s[k]}
            for c in range(a.count)
            for s in [rec.core_state(result.state, c)]
            if s["reader"] or s["writer"] or s["compute"]
        },
        "mhz": result.mhz,
        "host_s": round(result.host_s, 3),
    }
    out.update(timing(result, res))
    return out


def timing(result, res):
    """The submitter's checks; microseconds from the last consume to the quiet end."""
    run = result.rows[1]  # the RUN command is command 0
    submit = int(result.commands["submit"][0])
    out = {
        "checks": int(run[K["kFrRecMessagesChecks"]]),
        "submitter_sent": int(run[K["kFrRecMessagesSent"]]),
        "gave_core": int(run[K["kFrRecMessagesGave"]]) - 1,
    }
    quiet, ack = int(run[K["kFrRecMessagesEnd"]]), int(run[K["kFrRecMessagesEndAck"]])
    last = max(int(v) for v in res[:, FORWARD["kResLastConsume"]])
    if quiet:
        out["run_to_quiet_microseconds"] = float(rec.wrap(quiet - submit)) / result.mhz
        if last:
            out["last_consume_to_quiet_microseconds"] = (
                float(rec.wrap(quiet - last)) / result.mhz
            )
        if ack:
            out["quiet_to_end_ack_microseconds"] = (
                float(rec.wrap(ack - quiet)) / result.mhz
            )
    return out


if __name__ == "__main__":
    main()
