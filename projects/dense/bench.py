"""Time, power, clock and profile of the matrix product on a card, one shape per call:
a profiled call, single calls, calls back to back under the power sampler, the cut-down
runs, then the check of Y. Exits 1 when the check fails. Usage in README.md.
"""

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


def profile_summary(prof, mhz):
    """Shares of the slowest core's time per unit, and the balance across cores."""
    pr = prof.astype(np.float64)
    busy = np.maximum.reduce([pr[:, K[k]] for k in BODIES])
    slow = int(np.argmax(busy))
    total = busy[slow]
    return {
        "slowest_core": slow,
        "slowest_us": round(total / mhz, 3),
        "mean_busy_over_slowest": round(float(busy.mean() / total), 3),
        "shares_of_slowest": {
            name: round(float(pr[slow, K[key]] / total), 3)
            for name, key in SHARES.items()
        },
    }


def loop(product, reps, index):
    """`reps` PRODUCT records back to back with power and clock sampled."""
    meter = power.PowerSampler(index)
    with meter, Clock(index) as clock:
        meter.mark("open")
        time.sleep(1.0)
        meter.mark("loop")
        host_s, us = product.calls(reps, timeout=60.0)
        meter.mark("after")
    phase = meter.result.phases["loop"]
    return {
        "reps": reps,
        "seconds": round(host_s, 3),
        "us_per_call": round(host_s * 1e6 / reps, 3),
        "record_us_median": float(np.median(us)),
        "mean_w": phase.mean_w,
        "peak_w": phase.peak_w,
        "open_w": meter.result.phases["open"].mean_w,
        "joules_per_call": phase.mean_w * host_s / reps if phase.mean_w else None,
        "aiclk_mhz": clock.summary(),
    }


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--shape", choices=run.SHAPES, required=True)
    a.add_argument("--form", choices=["resident", "stream"], default=None)
    a.add_argument("--seconds", type=float, default=4.0)
    a.add_argument("--cut-seconds", type=float, default=1.0)
    a.add_argument("--max-reps", type=int, default=1_000_000)
    a.add_argument("--json", default=None)
    args = a.parse_args()
    m, k, n = run.SHAPES[args.shape]
    form = None if args.form is None else K[f"kForm{args.form.title()}"]
    p = pl.plan(m // 32, k // 32, n // 32, form=form)
    out = {"shape": args.shape, "m": m, "k": k, "n": n}
    out["plan"] = p.__dict__ | {"l1": p.l1_bytes()}
    x, w = ref.inputs(m, k, n)
    index = visible_card()
    with km.Session(log=log) as session:
        device = session.device
        cores = len(session.nocs().words)
        tensors = {
            "a": card.bf16_tensor(device, x),
            "w": card.bf16_tensor(device, w),
            "y": card.bf16_tensor(device, np.zeros((m, n), np.float32)),
        }
        prof_t = card.prof_tensor(device, cores)
        count = pl.team_size(p, cores)
        product = card.Product(session, p, tensors, count)
        mhz = aiclk_mhz(index)
        out["load_us"] = product.load_us
        single = product.single(1)
        out["profile"] = profile_summary(product.profile(session, prof_t), mhz)
        single = np.concatenate([single, product.single(20)])
        out["single_us"] = {"median": float(np.median(single)), "min": single.min()}
        one = float(np.median(single))
        log(f"single call {one:.2f} us; profile read")
        reps = int(min(args.max_reps, max(2, math.ceil(args.seconds * 1e6 / one))))
        out["loop"] = loop(product, reps, index)
        log(f"loop: {out['loop']['us_per_call']} us per call")
        out["tflops"] = pl.flops(p) / out["loop"]["us_per_call"] / 1e6
        cut = {}
        for name in ("reads_only", "compute_only"):
            mode = K["kModeReadOnly" if name == "reads_only" else "kModeComputeOnly"]
            r2 = int(min(args.max_reps, max(2, args.cut_seconds * 1e6 / one)))
            host_s, _ = product.calls(r2, mode=mode)
            cut[name] = round(host_s * 1e6 / r2, 3)
        out["cut_us_per_call"] = cut
        # the cut-down runs left Y as it was: one more full call before the check
        product.calls(1)
        out["run"] = product.stop()
        (y,) = session.read_bf16(tensors["y"])
    out["check"] = run.check(y[:m, :n], x, w, "uniform")
    out["ok"] = out["check"]["ok"] and out["run"]["kernel"] == "none"
    out["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(out, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
