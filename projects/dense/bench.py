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
from komorebi import power  # noqa: E402
from komorebi.session import aiclk_mhz, visible_card  # noqa: E402
from projects.dense import card, run  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from ref import dense as ref  # noqa: E402

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

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self):
        v = self.samples
        return [min(v), round(float(np.mean(v)), 1), max(v)] if v else None


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
GRID_SHARES = {
    "read_A_dram": "kGpRdRead",
    "row_lane": "kGpRdLane",
    "read_W_dram": "kGpWrRead",
    "column_lane": "kGpWrLane",
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
        ("lanes_only", pl.G["kGridLanes"]),
        ("product_only", pl.G["kGridProduct"]),
    ),
    "split": (("parts_only", pl.S["kSplitParts"]),),
}


def profile_summary(prof, mhz, grid):
    """Shares of the slowest core's time per unit, and the balance across cores."""
    words, shares, bodies = (
        (pl.G, GRID_SHARES, GRID_BODIES) if grid else (K, SHARES, BODIES)
    )
    pr = prof.astype(np.float64)
    busy = np.maximum.reduce([pr[:, words[k]] for k in bodies])
    slow = int(np.argmax(busy))
    total = busy[slow]
    return {
        "slowest_core": slow,
        "slowest_us": round(total / mhz, 3),
        "mean_busy_over_slowest": round(float(busy.mean() / total), 3),
        "shares_of_slowest": {
            name: round(float(pr[slow, words[key]] / total), 3)
            for name, key in shares.items()
        },
    }


def loop(product, reps, index, one_us, batch):
    """`reps` records in runs of at most `batch` back to back (each waited for), with
    power and clock sampled; each wait bounded by 3 x their single-call time + 5 s."""
    meter = power.PowerSampler(index)
    host_s, us, t0 = 0.0, [], time.perf_counter()
    with meter, Clock(index) as clock:
        meter.mark("open")
        time.sleep(1.0)
        meter.mark("loop")
        t0 = time.perf_counter()
        for first in range(0, reps, batch):
            n = min(batch, reps - first)
            s, u = product.calls(n, timeout=3 * n * one_us / 1e6 + 5)
            host_s, us = host_s + s, us + list(u)
        wall_s = time.perf_counter() - t0
        meter.mark("after")
    phase = meter.result.phases["loop"]
    return {
        "reps": reps,
        "batch": batch,
        "seconds": round(host_s, 3),
        "wall_seconds": round(wall_s, 3),
        "us_per_call": round(host_s * 1e6 / reps, 3),
        "record_us_median": float(np.median(us)),
        "mean_w": phase.mean_w,
        "peak_w": phase.peak_w,
        "open_w": meter.result.phases["open"].mean_w,
        "joules_per_call": phase.mean_w * wall_s / reps if phase.mean_w else None,
        "aiclk_mhz": clock.summary(),
    }


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--shape", choices=run.SHAPES, required=True)
    a.add_argument("--form", choices=run.FORMS, default=None)
    a.add_argument("--grid", default=None, help="as run.py's --grid")
    a.add_argument("--split", default=None, help="as run.py's --split")
    a.add_argument("--reduce", choices=list(pl.REDUCE_FORMS), default="owner")
    a.add_argument("--seconds", type=float, default=4.0)
    a.add_argument("--cut-seconds", type=float, default=1.0)
    a.add_argument("--max-reps", type=int, default=1_000_000)
    a.add_argument("--batch", type=int, default=0, help="records per run; 0: one run")
    a.add_argument(
        "--deadline", type=float, default=None, help="s; default: deadline()"
    )
    a.add_argument("--json", default=None)
    args = a.parse_args()
    m, k, n = run.SHAPES[args.shape]
    p = run.make_plan(args.form, (m, k, n), args.grid, args.split)
    form = form_of(p)
    out = {"shape": args.shape, "m": m, "k": k, "n": n, "plan": run.describe(p)}
    log("building the inputs")
    x, w = ref.inputs(m, k, n)
    log("inputs built")
    index = visible_card()
    with km.Session(log=log) as session:
        cores = len(session.nocs().words)
        t0 = time.perf_counter()
        tensors = {}
        for name, value in (("a", x), ("w", w), ("y", np.zeros((m, n), np.float32))):
            tensors[name] = card.bf16_tensor(session, value)
            log(f"{name.upper()} uploaded")
        out["upload_s"] = round(time.perf_counter() - t0, 3)
        log(f"operands uploaded in {out['upload_s']} s")
        prof_t = card.prof_tensor(session, cores)
        out["x280_mhz"] = session.pll_mhz
        out["deadline_s"] = args.deadline or deadline(p)
        out["reduce"] = args.reduce
        load = dict(deadline=out["deadline_s"], reduce=args.reduce)
        product = run.load(session, p, tensors, **load)
        mhz = out["profile_mhz"] = aiclk_mhz(index)
        out["load_us"] = product.load_us
        try:
            measure(product, p, args, out)
        except km.KomorebiError as err:
            out["failed"] = product.failed(err)
            log(f"failed: {json.dumps(out['failed'], default=str)}")
            print(json.dumps(out, default=str), flush=True)
            if args.json:
                Path(args.json).write_text(json.dumps(out, indent=1, default=str))
            sys.exit(1)
        out["run"] = product.stop()
        t0 = time.perf_counter()
        rows = ref.sampled_rows(m)
        got = card.read_rows(session, tensors["y"], rows, log)
        out["readback_s"] = round(time.perf_counter() - t0, 3)
        log("Y's sampled rows read back")
        # the profile last, in a load of its own: one profiled call (none in the split)
        if form != "split":
            product = run.load(session, p, tensors, **load)
            product.single(1)
            prof = product.profile(session, prof_t)
            out["profile"] = profile_summary(prof, mhz, form == "grid")
            product.stop()
            log("profile read")
    log("checking the sampled rows")
    out["check"] = run.check(got[:, :n], rows, x, w, "uniform")
    out["ok"] = out["check"]["ok"] and run.clean(out["run"])
    out["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(out, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
    sys.exit(0 if out["ok"] else 1)


def deadline(p):
    """A record's deadline (s): three times the call at 20 TFLOP/s, plus 50 ms."""
    return round(3 * pl.flops(p) / 20e12 + 0.05, 3)


def form_of(p):
    return {pl.Grid: "grid", pl.Split: "split"}.get(type(p), "plain")


def measure(product, p, args, out):
    """Single calls, the loop, the cut-down runs, one last full call."""
    index = visible_card()
    form = form_of(p)
    single = product.single(21)
    out["single_us"] = {"median": float(np.median(single)), "min": single.min()}
    one = float(np.median(single))
    log(f"single call {one:.2f} us")
    reps = int(min(args.max_reps, max(2, math.ceil(args.seconds * 1e6 / one))))
    out["loop"] = loop(product, reps, index, one, args.batch or reps)
    log(f"loop: {out['loop']['us_per_call']} us per call")
    out["tflops"] = pl.flops(p) / out["loop"]["us_per_call"] / 1e6
    cut = {}
    for name, mode in CUTS[form]:
        r2 = int(min(args.max_reps, max(2, args.cut_seconds * 1e6 / one)))
        host_s, _ = product.calls(r2, mode=mode)
        cut[name] = round(host_s * 1e6 / r2, 3)
    out["cut_us_per_call"] = cut
    # the cut-down runs left Y as it was: one more full call before the check
    product.calls(1)


if __name__ == "__main__":
    main()
