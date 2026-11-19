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
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src")]

import replay  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi.layout import counter_tiles  # noqa: E402
from komorebi.program.description import Dynamic  # noqa: E402
from komorebi.tools.feeder import commands as feeder_commands  # noqa: E402
from komorebi.tools.feeder import fed  # noqa: E402

CONSTANTS = feeder_commands.CONSTANTS
FORWARD = feeder_commands.constants(HERE / "kernels" / "forward.h")
KERNELS = ROOT / "src" / "komorebi_native" / "kernels"


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--count", type=int, default=3, help="group: cores 0 to count - 1"
    )
    parser.add_argument("--messages", type=int, default=1)
    parser.add_argument("--forwards", type=int, default=1, help="forwards per message")
    parser.add_argument(
        "--one-core", type=int, default=-1, help="every forward to this rank"
    )
    parser.add_argument("--stall", type=int, default=-1, help="rank that stalls")
    parser.add_argument(
        "--probe", type=int, default=-1, help="rank that spends its quota"
    )
    parser.add_argument("--max-live", type=int, default=1 << 14)
    parser.add_argument("--queue-messages", type=int, default=1 << 14)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--idle-cycles", type=int, default=20000)
    parser.add_argument(
        "--counters", type=int, default=2, help="counter tiles of the group"
    )
    parser.add_argument("--deadline", type=float, default=5.0, help="the run, seconds")
    parser.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    return parser.parse_args()


def main():
    arguments = parse()
    try:
        messages = Dynamic(
            handler=FORWARD["kHandlerForward"],
            max_live=arguments.max_live,
            queue_messages=arguments.queue_messages,
            bins=arguments.bins,
            idle_cycles=arguments.idle_cycles,
            counters=arguments.counters,
        )
        messages.check(119, 4 << 30, 4)
        messages.check(arguments.count, 4 << 30, 4)
    except ValueError as error:  # the launcher's rejection, before any device opens
        print(json.dumps({"ok": False, "rejected": str(error)}), flush=True)
        sys.exit(1)
    kernels = fed.Kernels(
        reader=KERNELS / "messages/reader.cpp",
        compute=HERE / "kernels" / "compute.cpp",
        writer=KERNELS / "messages/writer.cpp",
        cbs=[
            (buffer.index, "uint32", buffer.page_bytes, buffer.pages)
            for buffer in messages.buffers()
        ],
    )
    run = feeder_commands.command(
        FORWARD["kHandlerForward"],
        0,
        arguments.count,
        [
            arguments.messages,
            arguments.forwards,
            (
                FORWARD["kModeOneCore"]
                if arguments.one_core >= 0
                else FORWARD["kModeRandom"]
            ),
            max(arguments.one_core, 0),
            arguments.stall + 1,
            arguments.probe + 1,
        ],
        flags=CONSTANTS["KM_FLAG_EXCLUSIVE"],
    )
    with km.Session(log=fed.log) as session:
        fed.log("device open")
        runner = fed.Fed(session)
        res_t = runner.l1_blocks(FORWARD["kResWords"])
        result = runner.run(
            kernels,
            [run],
            program_args=lambda core: ([], [], [res_t.address]),
            tensors=[res_t],
            depth=4,
            deadline_s=arguments.deadline,
            wait_s=arguments.wait,
            dynamic=messages,
        )
        result_words = feeder_commands.as_uint32(fed.read_int32(res_t))[
            : arguments.count
        ]
        fed.log("results read back")
    out = report(arguments, messages, result, result_words)
    print(json.dumps(out), flush=True)
    sys.exit(0 if out["exact"] else 1)


def report(arguments, messages, result, result_words):
    """The checks and numbers of one run as a dict."""
    sequence = int(result.commands["sequence"][0])
    lines = result.lines[: arguments.count]
    gave_at = (CONSTANTS["KM_CONTROL_MESSAGES"] + CONSTANTS["KM_MESSAGES_GAVE"]) // 4
    gave = {
        core: [int(value) for value in result.control[core, gave_at : gave_at + 4]]
        for core in range(arguments.count)
        if result.control[core, gave_at]
    }
    worker_count = len(
        counter_tiles(0, arguments.count, arguments.counters).workers
    )  # the rest: counter tiles
    one = arguments.one_core if arguments.one_core >= 0 else None
    consumed, total, visits = replay.replay(
        arguments.messages, arguments.forwards, worker_count, one
    )
    if arguments.probe >= 0:  # the probe's messages visit their own core once
        visits[arguments.probe] += result_words[arguments.probe, FORWARD["kResSent"]]
    totals = result_words[:worker_count, FORWARD["kResTotalLo"]].astype(np.uint64) | (
        result_words[:worker_count, FORWARD["kResTotalHi"]].astype(np.uint64)
        << np.uint64(32)
    )
    out = {
        "feeder": result.summary,
        "run_status": int(result.commands["status"][0]),
        "run_sequence": sequence,
        "exact": bool((result_words[:, FORWARD["kResRun"]] == sequence).all())
        and bool(
            (result_words[:worker_count, FORWARD["kResConsumed"]] == consumed).all()
        )
        and bool((totals == total).all())
        and bool((result_words[:worker_count, FORWARD["kResVisits"]] == visits).all())
        and not result_words[worker_count:, FORWARD["kResVisits"]].any(),
        "consumed_by_compute": int(result_words[:, FORWARD["kResConsumed"]].sum()),
        "expected_consumed": int(consumed.sum()),
        "forwards": int(result_words[:, FORWARD["kResVisits"]].sum())
        - arguments.messages,
        "nulls": int(result_words[:, FORWARD["kResNull"]].sum()),
        "probe_sent": int(result_words[:, FORWARD["kResSent"]].sum()),
        "rejected": int(result_words[:, FORWARD["kResRejected"]].sum()),
        "quota_start": messages.quota(result.lines.shape[0]),
        "quota_left_min": int(result_words[:worker_count, FORWARD["kResQuota"]].min()),
        "lines_not_idle": [
            core
            for core in range(arguments.count)
            if lines[core, CONSTANTS["KM_MESSAGES_LINE_RUN"]] != sequence
            or lines[core, CONSTANTS["KM_MESSAGES_LINE_STATE"]]
            != CONSTANTS["KM_MESSAGES_IDLE"]
        ],
        "sent": int(lines[:, CONSTANTS["KM_MESSAGES_LINE_SENT"]].sum()),
        "consumed": int(lines[:, CONSTANTS["KM_MESSAGES_LINE_CONSUMED"]].sum()),
        "messages_gave": gave,
        "state_gave": {
            core: {
                name: core_state[name]
                for name in ("reader", "writer", "compute")
                if core_state[name]
            }
            for core in range(arguments.count)
            for core_state in [feeder_commands.core_state(result.state, core)]
            if core_state["reader"] or core_state["writer"] or core_state["compute"]
        },
        "mhz": result.mhz,
        "host_s": round(result.host_s, 3),
    }
    out.update(timing(result, result_words))
    return out


def timing(result, result_words):
    """The submitter's checks; microseconds from the last consume to the quiet end."""
    run = result.rows[1]  # the RUN command is command 0
    submit = int(result.commands["submit"][0])
    out = {
        "checks": int(run[CONSTANTS["kFrRecMessagesChecks"]]),
        "submitter_sent": int(run[CONSTANTS["kFrRecMessagesSent"]]),
        "gave_core": int(run[CONSTANTS["kFrRecMessagesGave"]]) - 1,
    }
    quiet, ack = int(run[CONSTANTS["kFrRecMessagesEnd"]]), int(
        run[CONSTANTS["kFrRecMessagesEndAck"]]
    )
    last = max(int(value) for value in result_words[:, FORWARD["kResLastConsume"]])
    if quiet:
        out["run_to_quiet_microseconds"] = (
            float(feeder_commands.wrap(quiet - submit)) / result.mhz
        )
        if last:
            out["last_consume_to_quiet_microseconds"] = (
                float(feeder_commands.wrap(quiet - last)) / result.mhz
            )
        if ack:
            out["quiet_to_end_ack_microseconds"] = (
                float(feeder_commands.wrap(ack - quiet)) / result.mhz
            )
    return out


if __name__ == "__main__":
    main()
