"""When each core of a row starts after one host launch: a kernel that only stamps the
card's wall clock at its start, `--reps` launches, one JSON line each with the start
offsets (cycles from the first core's start) and the host's seconds. Each core may then
spin or write 16 bytes or a block. Example: spread.py --cores 12 --reps 12"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km

KERNEL = Path(__file__).resolve().parent / "kernels" / "stamp.cpp"
ROW = 2048  # words in a core's row: the report, then the source bytes of a write
T0 = time.monotonic()


def holders(card):
    """Pids that hold the device file of `card` open (links in /proc/<pid>/fd)."""
    target = f"/dev/tenstorrent/{card}"
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            for fd in os.listdir(f"/proc/{entry}/fd"):
                if os.readlink(f"/proc/{entry}/fd/{fd}") == target:
                    pids.append(int(entry))
                    break
        except OSError:
            continue
    return pids


def run_ns(pids):
    """Seconds of CPU the threads of `pids` have run so far (from schedstat)."""
    total = 0
    for pid in pids:
        try:
            for task in os.listdir(f"/proc/{pid}/task"):
                with open(f"/proc/{pid}/task/{task}/schedstat") as stat:
                    total += int(stat.read().split()[0])
        except OSError:
            continue
    return total / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cores", type=int, default=12, help="cores 0..n-1 of row 0")
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument(
        "--what", choices=["none", "spin", "word", "write"], default="none"
    )
    parser.add_argument("--spin", type=int, default=50000, help="cycles to spin")
    parser.add_argument(
        "--reader",
        action="store_true",
        help="blocks of 10 launches alternate without and with a child process "
        "reading the card's tt_aiclk file in a loop",
    )
    parser.add_argument(
        "--clock-out",
        help="also run sample_clock.py beside the launches, into this file",
    )
    parser.add_argument(
        "--sleeps",
        help="milliseconds to wait before each launch, in blocks of --reps launches "
        "each (for example 0,10,100,500); the launch count is --reps a block",
    )
    parser.add_argument("--bytes", type=int, default=4096, help="bytes of a write")
    parser.add_argument("--processor", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()
    if os.environ.get("TT_VISIBLE_DEVICES") != str(args.card):
        sys.exit(f"stop: TT_VISIBLE_DEVICES must be {args.card}")
    spreads = []
    with km.Session() as session:
        print(f"[{time.monotonic() - T0:6.2f}] device open", flush=True)
        flags = session.tensor(
            np.zeros((args.cores, ROW), np.uint32),
            "uint32",
            shard_grid=(args.cores, 1),
        )
        what = ["none", "spin", "word", "write"].index(args.what)
        launch = session.launch()
        # Writes go to 40 MiB and up, 64 KiB a core: far from anything else.
        runtime = [
            (j, 0, [flags.address, what, args.spin, args.bytes, (40 << 20) + j * 65536])
            for j in range(args.cores)
        ]
        cores = [(0, 0, args.cores - 1, 0)]
        launch.kernel(str(KERNEL), cores, args.processor, [], [], [], runtime)
        reader = None
        sampler = None
        if args.clock_out:
            here = Path(__file__).resolve().parent / "sample_clock.py"
            sampler = subprocess.Popen(
                [sys.executable, str(here), "--card", str(args.card), "--out"]
                + [args.clock_out, "--seconds", "120"]
            )
            time.sleep(0.3)
        sleeps = [float(v) for v in args.sleeps.split(",")] if args.sleeps else [0.0]
        for rep in range(args.reps * len(sleeps)):
            pause = sleeps[rep // args.reps]
            if args.reader and rep % 10 == 0:
                if reader:
                    reader.terminate()
                    reader.wait()
                    reader = None
                if rep % 20 == 10:
                    clock = f"/sys/class/tenstorrent/tenstorrent!{args.card}/tt_aiclk"
                    loop = f"while :; do cat '{clock}' > /dev/null; done"
                    reader = subprocess.Popen(["sh", "-c", loop])
                    time.sleep(0.2)
            flags.write(np.zeros((args.cores, ROW), np.uint32))
            time.sleep(pause / 1000)
            others = holders(1 - args.card)
            others_before = run_ns(others)
            usage0 = resource.getrusage(resource.RUSAGE_THREAD)
            cpu0 = time.thread_time()
            t_start = time.monotonic()
            launch.run()
            t_run = time.monotonic()
            cpu_run = time.thread_time() - cpu0
            usage1 = resource.getrusage(resource.RUSAGE_THREAD)
            others_cpu = run_ns(others) - others_before
            session.device.finish()
            t_fin = time.monotonic()
            rows = flags.read().astype(np.int64)
            stamps = [int(r[1]) | int(r[2]) << 32 for r in rows]
            first = min(stamps)
            spreads.append((pause, max(stamps) - first))
            print(
                json.dumps(
                    {
                        "rep": rep,
                        "reader": reader is not None,
                        "sleep_ms": pause,
                        "t_start": round(t_start, 5),
                        "t_run": round(t_run, 5),
                        "other_card_processes": len(others),
                        "other_card_cpu_s": round(others_cpu, 4),
                        "done": [int(r[0]) for r in rows],
                        "start": [s - first for s in stamps],
                        "host_s_run": round(t_run - t_start, 5),
                        "cpu_s_run": round(cpu_run, 5),
                        "system_s_run": round(usage1.ru_stime - usage0.ru_stime, 5),
                        "involuntary_switches": usage1.ru_nivcsw - usage0.ru_nivcsw,
                        "voluntary_switches": usage1.ru_nvcsw - usage0.ru_nvcsw,
                        "host_s_finish": round(t_fin - t_start, 5),
                        "load": round(os.getloadavg()[0], 1),
                    }
                ),
                flush=True,
            )
    if reader:
        reader.terminate()
        reader.wait()
    if sampler:
        sampler.terminate()
        sampler.wait()
    for pause in sleeps:
        later = [spread for state, spread in spreads[1:] if state == pause]
        if later:
            print(
                json.dumps(
                    {
                        "summary": args.what,
                        "sleep_ms": pause,
                        "launches": len(later),
                        "over_1M": sum(spread > 1_000_000 for spread in later),
                        "over_100k_per_core": sum(
                            spread > 100_000 * (args.cores - 1) for spread in later
                        ),
                        "median_spread": sorted(later)[len(later) // 2],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
