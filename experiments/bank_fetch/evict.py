"""The cluster cache read from work cluster 1's thread (`shape`): its configuration,
the ways enabled, every master's way mask and the Zero Device past the runtime's
pinned span. One JSON line. Example: python evict.py shape"""

import argparse
import json
import time
from pathlib import Path

import komorebi as km

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[1] / "examples" / "08_parallel_thread" / "kernels"  # NOPs
MODES = {"shape": 0}
MASTERS = 38
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=sorted(MODES))
    args = parser.parse_args()
    with km.Session(log=log) as session:
        program = km.Program("bank_evict", kernels=KERNELS)
        program.thread(
            cluster=1,
            sources=[HERE / "thread" / "evict.c"],
            regions={"pad": 1 << 17},
            words=[MODES[args.mode]],
            deadline=10.0,
        )
        run = session.load(program)
        log("loaded")
        result = run.thread_result(timeout=15.0)
        log(f"thread: {result.state}, code {result.code}, error {result.error}")
        run.stop()
        log("stopped")
    report = {"mode": args.mode, "thread": [result.state, result.code, result.error]}
    report.update(shape_report(list(result.words) + [0] * 48))
    report["ok"] = result.ran and result.code == 0
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
