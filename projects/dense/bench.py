"""Time, power, clock and profile of the matrix product on a card, one shape per call:
single calls, runs of calls back to back under the power sampler, the cut-down runs,
the check of Y, then one profiled call in a load of its own. Exits 1 when the check
fails. Usage in README.md."""

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi.bench import power  # noqa: E402
from komorebi.bench.power import aiclk_mhz, visible_card  # noqa: E402
from projects.dense import card, run  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from reference import dense as ref  # noqa: E402

K = pl.K
log = run.log


class Clock:
    """AICLK of the card (MHz) sampled every 20 ms while the with-block runs."""

    def __init__(self, index):
        self.index, self.samples, self.stop = index, [], threading.Event()

    def sample(self):
        while not self.stop.wait(0.02):
            self.samples.append(aiclk_mhz(self.index))

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exception):
        self.stop.set()
        self.thread.join()

    def summary(self):
        values = self.samples
        return (
            [min(values), round(float(np.mean(values)), 1), max(values)]
            if values
            else None
        )


SHARES = {  # profile words whose share of the slowest core's time is reported
    "read_A": "kPrRdRead",
    "read_W_chunks": "kPrRdWsRead",
    "reader_waits_for_space": "kPrRdWait",
    "unpack_waits_for_A": "kPrUnWaitA",
    "unpack_waits_for_W": "kPrUnWaitW",
    "matrix_unit": "kPrMathMm",
    "pack": "kPrPkPack",
    "pack_waits_for_math": "kPrPkWaitMath",
    "pack_waits_for_space": "kPrPkWaitOut",
    "write": "kPrWrWrite",
    "writer_waits_for_output": "kPrWrWait",
}
BODIES = ("kPrRdBody", "kPrUnBody", "kPrPkBody", "kPrWrBody")
GRID_PARTITIONS = {
    "read_A_dram": "kGpRdRead",
    "row_broadcast": "kGpRdBroadcast",
    "read_W_dram": "kGpWrRead",
    "column_broadcast": "kGpWrBroadcast",
    "writer_waits_for_output": "kGpWrWait",
    "write": "kGpWrWrite",
    "unpack_waits_for_A": "kGpUnWaitA",
    "unpack_waits_for_W": "kGpUnWaitW",
    "unpack_waits_for_partials": "kGpUnWaitPart",
    "matrix_unit": "kGpMathMm",
    "pack_waits_for_space": "kGpPkWaitOut",
    "pack": "kGpPkPack",
}
GRID_BODIES = ("kGpRdBody", "kGpUnBody", "kGpPkBody", "kGpWrBody")
CUTS = {  # cut-down runs: (name, mode) per form
    "plain": (
        ("reads_only", K["kModeReadOnly"]),
        ("compute_only", K["kModeComputeOnly"]),
    ),
    "grid": (
        ("broadcasts_only", pl.G["kGridBroadcasts"]),
        ("product_only", pl.G["kGridProduct"]),
    ),
    "split": (("parts_only", pl.S["kSplitParts"]),),
}


def profile_summary(profile, mhz, grid):
    """Shares of the slowest core's time per unit, and the balance across cores."""
    words, shares, bodies = (
        (pl.G, GRID_PARTITIONS, GRID_BODIES) if grid else (K, SHARES, BODIES)
    )
    cycles = profile.astype(np.float64)
    busy = np.maximum.reduce([cycles[:, words[key]] for key in bodies])
    slowest = int(np.argmax(busy))
    total = busy[slowest]
    return {
        "slowest_core": slowest,
        "slowest_microseconds": round(total / mhz, 3),
        "mean_busy_over_slowest": round(float(busy.mean() / total), 3),
        "partitions_of_slowest": {
            name: round(float(cycles[slowest, words[key]] / total), 3)
            for name, key in shares.items()
        },
    }


def loop(product, reps, index, one_microseconds, batch):
    """`reps` commands in runs of at most `batch` back to back (each waited for), with
    power and clock sampled; each wait's timeout 3 x their single-call time + 5 s."""
    meter = power.PowerSampler(index)
    host_s, command_microseconds, started = 0.0, [], time.perf_counter()
    with meter, Clock(index) as clock:
        meter.mark("open")
        time.sleep(1.0)
        meter.mark("loop")
        started = time.perf_counter()
        for first in range(0, reps, batch):
            count = min(batch, reps - first)
            seconds, microseconds = product.calls(
                count, timeout=3 * count * one_microseconds / 1e6 + 5
            )
            host_s, command_microseconds = (
                host_s + seconds,
                command_microseconds + list(microseconds),
            )
        wall_s = time.perf_counter() - started
        meter.mark("after")
    phase = meter.result.phases["loop"]
    return {
        "reps": reps,
        "batch": batch,
        "seconds": round(host_s, 3),
        "wall_seconds": round(wall_s, 3),
        "microseconds_per_call": round(host_s * 1e6 / reps, 3),
        "command_microseconds_median": float(np.median(command_microseconds)),
        "mean_w": phase.mean_w,
        "peak_w": phase.peak_w,
        "open_w": meter.result.phases["open"].mean_w,
        "joules_per_call": phase.mean_w * wall_s / reps if phase.mean_w else None,
        "aiclk_mhz": clock.summary(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=run.SHAPES, required=True)
    parser.add_argument("--form", choices=run.FORMS, default=None)
    parser.add_argument("--grid", default=None, help="as run.py's --grid")
    parser.add_argument("--split", default=None, help="as run.py's --split")
    parser.add_argument("--reduce", choices=list(pl.REDUCE_FORMS), default="owner")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--cut-seconds", type=float, default=1.0)
    parser.add_argument("--max-reps", type=int, default=1_000_000)
    parser.add_argument(
        "--batch", type=int, default=0, help="commands per run; 0: one run"
    )
    parser.add_argument(
        "--deadline", type=float, default=None, help="s; default: deadline()"
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    m_rows, k_depth, n_columns = run.SHAPES[args.shape]
    plan = run.make_plan(args.form, (m_rows, k_depth, n_columns), args.grid, args.split)
    form = form_of(plan)
    report = {
        "shape": args.shape,
        "m": m_rows,
        "k": k_depth,
        "n": n_columns,
        "plan": run.describe(plan),
    }
    log("building the inputs")
    a_matrix, w_matrix = ref.inputs(m_rows, k_depth, n_columns)
    log("inputs built")
    index = visible_card()
    with km.Session(log=log) as session:
        cores = len(session.nocs().words)
        started = time.perf_counter()
        tensors = {}
        for name, value in (
            ("a", a_matrix),
            ("w", w_matrix),
            ("y", np.zeros((m_rows, n_columns), np.float32)),
        ):
            tensors[name] = card.bf16_tensor(session, value)
            log(f"{name.upper()} uploaded")
        report["upload_s"] = round(time.perf_counter() - started, 3)
        log(f"operands uploaded in {report['upload_s']} s")
        profile_tensor = card.prof_tensor(session, cores)
        report["x280_mhz"] = session.pll_mhz
        report["deadline_s"] = args.deadline or deadline(plan)
        report["reduce"] = args.reduce
        load = dict(deadline=report["deadline_s"], reduce=args.reduce)
        product = run.load(session, plan, tensors, **load)
        mhz = report["profile_mhz"] = aiclk_mhz(index)
        report["load_microseconds"] = product.load_microseconds
        try:
            measure(product, plan, args, report)
        except km.KomorebiError as error:
            report["failed"] = product.failed(error)
            log(f"failed: {json.dumps(report['failed'], default=str)}")
            print(json.dumps(power.clocked(report), default=str), flush=True)
            if args.json:
                Path(args.json).write_text(json.dumps(report, indent=1, default=str))
            sys.exit(1)
        report["run"] = product.stop()
        started = time.perf_counter()
        rows = ref.sampled_rows(m_rows)
        got = card.read_rows(session, tensors["y"], rows, log)
        report["readback_s"] = round(time.perf_counter() - started, 3)
        log("Y's sampled rows read back")
        # the profile last, in a load of its own: one profiled call (none in the split)
        if form != "split":
            product = run.load(session, plan, tensors, **load)
            product.single(1)
            profile = product.profile(session, profile_tensor)
            report["profile"] = profile_summary(profile, mhz, form == "grid")
            product.stop()
            log("profile read")
    log("checking the sampled rows")
    report["check"] = run.check(got[:, :n_columns], rows, a_matrix, w_matrix, "uniform")
    report["ok"] = report["check"]["ok"] and run.clean(report["run"])
    report["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(power.clocked(report), default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


def deadline(plan):
    """A command's deadline (s): three times the call at 20 TFLOP/s, plus 50 ms."""
    return round(3 * pl.flops(plan) / 20e12 + 0.05, 3)


def form_of(plan):
    return {pl.Grid: "grid", pl.Split: "split"}.get(type(plan), "plain")


def measure(product, plan, args, report):
    """Single calls, the loop, the cut-down runs, one last full call."""
    index = visible_card()
    form = form_of(plan)
    single = product.single(21)
    report["single_microseconds"] = {
        "median": float(np.median(single)),
        "min": single.min(),
    }
    one_microseconds = float(np.median(single))
    log(f"single call {one_microseconds:.2f} us")
    reps = int(
        min(args.max_reps, max(2, math.ceil(args.seconds * 1e6 / one_microseconds)))
    )
    report["loop"] = loop(product, reps, index, one_microseconds, args.batch or reps)
    log(f"loop: {report['loop']['microseconds_per_call']} us per call")
    report["tflops"] = pl.flops(plan) / report["loop"]["microseconds_per_call"] / 1e6
    cut = {}
    for name, mode in CUTS[form]:
        cut_reps = int(
            min(args.max_reps, max(2, args.cut_seconds * 1e6 / one_microseconds))
        )
        host_s, _ = product.calls(cut_reps, mode=mode)
        cut[name] = round(host_s * 1e6 / cut_reps, 3)
    report["cut_microseconds_per_call"] = cut
    # the cut-down runs left Y as it was: one more full call before the check
    product.calls(1)


if __name__ == "__main__":
    main()
