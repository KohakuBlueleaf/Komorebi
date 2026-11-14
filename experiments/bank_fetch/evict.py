"""Work cluster 1's thread reads the cluster cache's shape (`shape`), times kmx_evict
against the same reads of a pad in DRAM (`time`), or proves that it drops the lines
of a range the cores then copy into (`proof`). One JSON line. Example: python evict.py
proof --bytes 4096 --repeats 300"""

import argparse
import json
import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[1] / "examples" / "08_parallel_thread" / "kernels"  # NOPs
MODES = {"shape": 0, "time": 1, "proof": 2}
MASTERS = 38
TICKS = 50  # kmx_time() ticks a microsecond
PADS = ("zero_device", "dram")
STATES = ("after_read", "after_write", "after_evict")  # the cache before the call
VARIANTS = ("plain", "lead", "readers", "one_evict", "no_evict")  # evict.c's V_*
CONTROL = 4  # no_evict: the check must find bad lines
LEAD = 1 << 18  # bytes of region 0 before the range
T0 = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - T0:7.2f}] {message}", flush=True)


def shape_report(words):
    """evict.c's shape words as names: CCACHE0_CONFIG's four bytes, WayEnable, the
    masks (hex) by master, this core's L2 and the Zero Device read."""
    config = words[0]
    banks, ways, lg_sets, lg_line = (config >> s & 0xFF for s in (0, 8, 16, 24))
    masks = [f"{mask:#06x}" for mask in words[2 : 2 + MASTERS]]
    pl2 = words[42]
    return {
        "ccache_config": f"{config:#010x}",
        "banks": banks,
        "ways": ways,
        "sets": 1 << lg_sets,
        "line_bytes": 1 << lg_line,
        "way_bytes": banks * (1 << lg_sets) * (1 << lg_line),
        "wayenable": words[1],
        "cache_ways": (words[1] & 0xF) + 1,
        "masks": {"debug": masks[0], "noc": sorted(set(masks[1:33]))},
        "masks_trace": masks[33],
        "masks_cores": masks[34:38],
        "masks_high_bits": f"{words[40]:#x}",
        "core": words[41],
        "l2_config": f"{pl2:#010x}",
        "l2": {
            "banks": pl2 & 0xFF,
            "ways": pl2 >> 8 & 0xFF,
            "sets": 1 << (pl2 >> 16 & 0xFF),
            "line_bytes": 1 << (pl2 >> 24 & 0xFF),
        },
        "zero_device_bytes_read": words[43],
        "zero_device_nonzero_words": words[44],
    }


def time_report(words):
    """Per pad and state of the cache, the best and worst eviction in microseconds."""
    report = {}
    for p, pad in enumerate(PADS):
        for s, state in enumerate(STATES):
            best, worst = words[2 * (3 * p + s) : 2 * (3 * p + s) + 2]
            report[f"{pad}_{state}_us"] = [best / TICKS, worst / TICKS]
    return report


def proof_report(words):
    """Per variant its rounds, bad lines, first bad line and seconds; what the readers
    read and the first failed call."""
    report = {}
    for v, name in enumerate(VARIANTS):
        out = words[9 * v : 9 * v + 9]
        row = {"rounds": out[0], "bad_lines": out[1], "seconds": out[8] / TICKS / 1e6}
        if out[1]:
            row["first_bad"] = {
                "round": out[2],
                "line": out[3],
                "got": f"{out[4] | out[5] << 32:#018x}",
                "want": f"{out[6] | out[7] << 32:#018x}",
            }
        report[name] = row
    report["readers_mib"] = words[45]
    report["failed_call"] = f"{words[46]:#x}"
    return report


def regions(args):
    """The thread's regions for the mode."""
    if args.mode != "proof":
        return {"pad": 1 << 17, "fill": 1 << 20}
    return {"range": LEAD + args.bytes, "source": args.bytes, "noise": 1 << 22}


def merged(results, loads):
    """The proof words of loads of a variant or several: each variant's from its load,
    the readers' MiB summed, the failed calls together."""
    words = [0] * 48
    for result, load in zip(results, loads, strict=True):
        got = list(result.words) + [0] * 48
        for v in load:
            words[9 * v : 9 * v + 9] = got[9 * v : 9 * v + 9]
        words[45] += got[45]
        words[46] |= got[46]
    return words


def thread_run(session, args, variants, first):
    """Loads the thread with `variants` (proof) from round `first`; its result."""
    mask = sum(1 << v for v in variants)
    program = km.Program("bank_evict", kernels=KERNELS)
    program.thread(
        cluster=1,
        sources=[HERE / "thread" / "evict.c"],
        regions=regions(args),
        words=[MODES[args.mode], args.repeats, args.bytes, mask, first],
        deadline=args.deadline,
    )
    run = session.load(program)
    log(f"loaded, variants {variants}")
    while not run.thread.returned(5.0):
        log("thread running")
    result = run.thread_result(timeout=5.0)
    log(f"thread: {result.state}, code {result.code}, error {result.error}")
    run.stop()
    log("stopped")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bytes", type=int, default=4096, help="proof: the range")
    parser.add_argument("--variants", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--deadline", type=float, default=50.0)
    parser.add_argument("--load-each", action="store_true", help="a load a variant")
    args = parser.parse_args()
    variants = sorted(set(args.variants))
    loads = [[v] for v in variants] if args.load_each else [variants]
    with km.Session(log=log) as session:
        results = [
            thread_run(session, args, loads[i], 1 + 100000 * i)
            for i in range(len(loads))
        ]
    words = list(results[0].words) + [0] * 48
    if args.mode == "proof":
        words = merged(results, loads)
    failed = [r for r in results if not r.ran or r.code or r.state != "done"]
    result = (failed or results)[0]
    report = {"mode": args.mode, "thread": [result.state, result.code, result.error]}
    reports = {"shape": shape_report, "time": time_report, "proof": proof_report}
    report.update(reports[args.mode](words))
    if args.mode == "proof":
        report["bytes"] = args.bytes
        rows = [report[VARIANTS[v]] for v in set(args.variants) - {CONTROL}]
        report["exact"] = words[46] == 0 and all(
            row["rounds"] == args.repeats and row["bad_lines"] == 0 for row in rows
        )
    if args.mode == "proof" and CONTROL in args.variants:
        report["control_caught"] = report[VARIANTS[CONTROL]]["bad_lines"] > 0
    report["ok"] = result.ran and result.code == 0 and report.get("exact", True)
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
