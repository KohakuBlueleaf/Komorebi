"""The hop example's dynamic run fed by the test feeder on the last core (no X280):
items hop between the team's cores until their hops are used up; every core's totals
are checked against the numpy replay. One JSON line. python hop_fed.py --count 3"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXAMPLE = ROOT / "examples" / "05_hop"
sys.path[:0] = [str(HERE), str(EXAMPLE), str(ROOT / "src")]

import fed  # noqa: E402
import records as rec  # noqa: E402
import replay  # noqa: E402
import ttnn  # noqa: E402

from komorebi.layout import counter_tiles  # noqa: E402
from komorebi.program import Dynamic  # noqa: E402

K = rec.K
HOP = rec.constants(EXAMPLE / "kernels" / "hop.h")
KERNELS = ROOT / "src" / "komorebi_native" / "kernels"


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--count", type=int, default=3, help="team: cores 0 to count - 1")
    p.add_argument("--items", type=int, default=1)
    p.add_argument("--hops", type=int, default=1, help="hops per item")
    p.add_argument("--one-core", type=int, default=-1, help="every hop to this member")
    p.add_argument("--stall", type=int, default=-1, help="member that stalls")
    p.add_argument("--probe", type=int, default=-1, help="member that spends its quota")
    p.add_argument("--max-live", type=int, default=1 << 14)
    p.add_argument("--queue-items", type=int, default=1 << 14)
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--idle-cycles", type=int, default=20000)
    p.add_argument("--counters", type=int, default=2, help="counter tiles of the team")
    p.add_argument("--deadline", type=float, default=5.0, help="the run, seconds")
    p.add_argument("--wait", type=float, default=1.0, help="wait limit, seconds")
    return p.parse_args()


def main():
    a = parse()
    try:
        dyn = Dynamic(
            seg=HOP["kSegHop"],
            max_live=a.max_live,
            queue_items=a.queue_items,
            bins=a.bins,
            idle_cycles=a.idle_cycles,
            counters=a.counters,
        )
        dyn.check(119, 4 << 30, 4)
        dyn.check(a.count, 4 << 30, 4)
    except ValueError as err:  # the launcher's refusal, before any device opens
        print(json.dumps({"ok": False, "refused": str(err)}), flush=True)
        sys.exit(1)
    kernels = fed.Kernels(
        reader=KERNELS / "dyn_reader.cpp",
        compute=EXAMPLE / "kernels" / "compute.cpp",
        writer=KERNELS / "dyn_writer.cpp",
        cbs=[(b.index, ttnn.uint32, b.page_bytes, b.pages) for b in dyn.buffers()],
    )
    run = rec.record(
        HOP["kSegHop"],
        0,
        a.count,
        [
            a.items,
            a.hops,
            HOP["kModeOneCore"] if a.one_core >= 0 else HOP["kModeRandom"],
            max(a.one_core, 0),
            a.stall + 1,
            a.probe + 1,
        ],
        flags=K["KM_F_EXCL"],
    )
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    fed.log("device open")
    try:
        f = fed.Fed(device)
        res_t = f.l1_blocks(HOP["kResWords"])
        result = f.run(
            kernels,
            [run],
            program_args=lambda core: ([], [], [res_t.buffer_address()]),
            tensors=[res_t],
            depth=4,
            deadline_s=a.deadline,
            wait_s=a.wait,
            dynamic=dyn,
        )
        res = rec.as_uint32(fed.read_int32(res_t))[: a.count]
        fed.log("results read back")
    finally:
        ttnn.close_mesh_device(device)
    print(json.dumps(report(a, dyn, result, res)), flush=True)


def report(a, dyn, result, res):
    """The checks and numbers of one run as a dict."""
    seq = int(result.records["seq"][0])
    lines = result.lines[: a.count]
    gave_at = (K["KM_CTL_DYN"] + K["KM_DYN_GAVE"]) // 4
    gave = {
        c: [int(v) for v in result.ctl[c, gave_at : gave_at + 4]]
        for c in range(a.count)
        if result.ctl[c, gave_at]
    }
    m = len(counter_tiles(0, a.count, a.counters).members)  # the rest: counter tiles
    one = a.one_core if a.one_core >= 0 else None
    retired, total, visits = replay.replay(a.items, a.hops, m, one)
    if a.probe >= 0:  # the probe's items visit their own core once
        visits[a.probe] += res[a.probe, HOP["kResSpawned"]]
    totals = res[:m, HOP["kResTotalLo"]].astype(np.uint64) | (
        res[:m, HOP["kResTotalHi"]].astype(np.uint64) << np.uint64(32)
    )
    out = {
        "feeder": result.summary,
        "run_status": int(result.records["status"][0]),
        "run_seq": seq,
        "exact": bool((res[:, HOP["kResRun"]] == seq).all())
        and bool((res[:m, HOP["kResRetired"]] == retired).all())
        and bool((totals == total).all())
        and bool((res[:m, HOP["kResVisits"]] == visits).all())
        and not res[m:, HOP["kResVisits"]].any(),
        "retired": int(res[:, HOP["kResRetired"]].sum()),
        "expected_retired": int(retired.sum()),
        "hops": int(res[:, HOP["kResVisits"]].sum()) - a.items,
        "nulls": int(res[:, HOP["kResNull"]].sum()),
        "probe_spawned": int(res[:, HOP["kResSpawned"]].sum()),
        "refused": int(res[:, HOP["kResRefused"]].sum()),
        "quota_start": dyn.quota(result.lines.shape[0]),
        "quota_left_min": int(res[:m, HOP["kResQuota"]].min()),
        "lines_not_idle": [
            c
            for c in range(a.count)
            if lines[c, K["KM_DYN_LINE_RUN"]] != seq
            or lines[c, K["KM_DYN_LINE_STATE"]] != K["KM_DYN_IDLE"]
        ],
        "sent": int(lines[:, K["KM_DYN_LINE_SENT"]].sum()),
        "consumed": int(lines[:, K["KM_DYN_LINE_CONSUMED"]].sum()),
        "dyn_gave": gave,
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
    """The issuer's checks; microseconds from the last retire to the quiet end."""
    run = result.rows[1]  # the RUN record is record 0
    issue = int(result.records["issue"][0])
    out = {
        "checks": int(run[K["kFrRecDynChecks"]]),
        "issuer_sent": int(run[K["kFrRecDynSent"]]),
        "gave_core": int(run[K["kFrRecDynGave"]]) - 1,
    }
    quiet, ack = int(run[K["kFrRecDynEnd"]]), int(run[K["kFrRecDynEndAck"]])
    last = max(int(v) for v in res[:, HOP["kResLastRetire"]])
    if quiet:
        out["run_to_quiet_us"] = float(rec.wrap(quiet - issue)) / result.mhz
        if last:
            out["last_retire_to_quiet_us"] = float(rec.wrap(quiet - last)) / result.mhz
        if ack:
            out["quiet_to_end_ack_us"] = float(rec.wrap(ack - quiet)) / result.mhz
    return out


if __name__ == "__main__":
    main()
