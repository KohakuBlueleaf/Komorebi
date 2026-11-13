"""SwiGLU, y = (silu(x W1) * (x W3)) W2, on the card as LAYER commands through
`komorebi.Session`: W1, W3 and W2 loaded once and resident, a call's commands submitted
from the host back to back. Checked against float64 on sampled rows, optionally
timed. Prints one JSON line; exits 1 when the check fails. See SWIGLU.md.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import power  # noqa: E402
from komorebi.session import aiclk_mhz, visible_card  # noqa: E402
from projects.dense import bench, card, run  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from projects.dense import swiglu_plan as sp  # noqa: E402
from ref import dense as ref  # noqa: E402

K = pl.K
log = run.log
# Bound of the check on these inputs (SWIGLU.md, "The check's bound").
RMS_BOUND, MAX_BOUND = 2e-2, 3e-2
SHARES = {  # profile words reported per call on the slowest core
    "read_x": "kPrRdRead",
    "reader_waits_for_space": "kPrRdWait",
    "unpack_waits_for_a": "kPrUnWaitA",
    "products": "kPrMathMm",
    "epilogue": "kPrMathEp",
    "pack_waits_for_math": "kPrPkWaitMath",
    "pack_waits_for_space": "kPrPkWaitOut",
    "pack": "kPrPkPack",
    "write_y": "kPrWrWrite",
    "writer_waits_for_output": "kPrWrWait",
    "reader_body": "kPrRdBody",
    "unpack_body": "kPrUnBody",
    "math_body": "kPrMathBody",
    "pack_body": "kPrPkBody",
    "writer_body": "kPrWrBody",
}
BODIES = ("kPrRdBody", "kPrUnBody", "kPrMathBody", "kPrPkBody", "kPrWrBody")


class Layer:
    """Plan s loaded on a session's grid with tensors x, w1, w3, w2, y, prof: three
    LOAD commands at the start, then calls of the plan's LAYER commands."""

    def __init__(self, session, s, t, fidelity="HiFi2", deadline=2.0):
        self.s, self.t, self.group = s, t, (0, s.group)
        self.session = session
        prog = card.program(s, t["x"], fidelity)
        self.run = session.load(prog, deadline=deadline, wait=1.0)
        a = {k: v.address for k, v in t.items()}
        loads = sp.loads(s, a["w1"], a["w3"], a["w2"])
        handles = [self.run.submit("load", self.group, w) for w in loads]
        self.load_us = [e.us for e in self.run.wait(handles, timeout=10.0)]
        self.words = sp.commands(s, a["x"], a["y"])
        self.handler = self.run.program.handler("layer")

    def calls(self, n, exclusive=False, timeout=60.0):
        """n calls from the host, every command back to back: host seconds from the
        first submit to the last event, and each command's microseconds [n, commands].
        """
        rows = np.tile(np.array(self.words, np.int64), (n, 1))
        t0 = time.perf_counter()
        handles = self.run.submit_many(
            "layer", self.group[0], self.group[1], rows, exclusive=exclusive
        )
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - t0
        bad = [e for e in events if e.kind != "done"]
        if bad:
            raise km.KomorebiError(f"{len(bad)} commands not done: {bad[0]}")
        return host_s, np.array([e.us for e in events]).reshape(n, -1)

    def single(self, n):
        """n calls one at a time, each waited for: host microseconds each."""
        return np.array([self.calls(1)[0] * 1e6 for _ in range(n)])

    def profile(self, timeout=10.0):
        """Each core's profile page, summed over the commands so far."""
        prof = self.t["prof"]
        handle = self.run.submit("prof", self.group, [prof.address])
        self.run.wait(handle, timeout=timeout)
        (pages,) = self.session.read(prof)
        words = pages.reshape(-1).view(np.uint32).reshape(-1, K["kProfBytes"] // 4)
        return words[: self.group[1], : K["kProfWords"]].astype(np.int64)

    def stop(self):
        """Stops the run: the kernel's result, the cores that ended, the give-ups,
        and `ok` (no kernel error, no give-up, every core's three kernels ended)."""
        self.run.stop()
        states = self.run.state()
        out = {
            "kernel": self.run.result()["error_name"],
            "cores": len(states),
            "cores_ended": {
                k: sum(c[k] != 0 for c in states)
                for k in ("reader_end", "writer_end", "compute_end")
            },
            "gave_up": [
                {k: c[k] for k in ("core", "reader", "writer", "compute")}
                for c in states
                if c["reader"] or c["writer"] or c["compute"]
            ][:8],
        }
        ended = all(n == len(states) for n in out["cores_ended"].values())
        out["ok"] = out["kernel"] == "none" and not out["gave_up"] and ended
        return out


def profile_per_call(before, after, calls, mhz):
    """Microseconds per call of each profile share on the slowest core (the one
    with the longest body), from two profiles `calls` calls apart."""
    pr = (after - before).astype(np.float64) / calls
    busy = np.maximum.reduce([pr[:, K[k]] for k in BODIES])
    slow = int(np.argmax(busy))
    return {
        "slowest_rank": slow,
        "mean_busy_over_slowest": round(float(busy.mean() / busy[slow]), 3),
        "us": {n: round(pr[slow, K[k]] / mhz, 3) for n, k in SHARES.items()},
    }


def measured(index, fn):
    """fn() under the power sampler and the clock: its result and the loop's mean
    and peak power, the open card's power and AICLK (min, mean, max)."""
    meter = power.PowerSampler(index)
    with meter, bench.Clock(index) as clock:
        meter.mark("open")
        time.sleep(1.0)
        meter.mark("loop")
        got = fn()
        meter.mark("after")
    phase = meter.result.phases["loop"]
    return got, {
        "mean_w": phase.mean_w,
        "peak_w": phase.peak_w,
        "open_w": meter.result.phases["open"].mean_w,
        "aiclk_mhz": clock.summary(),
    }


def check(y_rows, rows, x, w1, w3, w2):
    """Sampled rows of Y against the float64 SwiGLU: error statistics and `ok`."""
    out = ref.errors(y_rows, ref.swiglu(x[rows], w1, w3, w2)) | {"rows": len(rows)}
    out["ok"] = out["rms_rel"] <= RMS_BOUND and out["max_rel"] <= MAX_BOUND
    return out


def parse():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--shape", choices=sp.SHAPES, required=True)
    a.add_argument(
        "--gate-pass", action="store_true", help="the gate a command of its own"
    )
    a.add_argument("--fidelity", choices=["HiFi2", "HiFi4"], default="HiFi2")
    a.add_argument("--calls", type=int, default=2, help="calls before the check")
    a.add_argument("--seconds", type=float, default=0.0, help="then timed loops")
    a.add_argument("--cut-seconds", type=float, default=1.0)
    a.add_argument("--max-calls", type=int, default=200_000)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--json", default=None)
    return a.parse_args()


def timed(layer, args, index, out):
    """Single calls with the profile around them, then the timed loops."""
    p0 = layer.profile()
    single = layer.single(20)
    mhz = aiclk_mhz(index)
    out["profile_single"] = profile_per_call(p0, layer.profile(), 20, mhz)
    one = float(np.median(single))
    out["single_us"] = {"median": one, "min": float(single.min())}
    log(f"single call {one:.2f} us")
    host_s, _ = layer.calls(200)  # back to back: the loop's length from this rate
    rate = host_s * 1e6 / 200
    n = int(min(args.max_calls, max(2, math.ceil(args.seconds * 1e6 / rate))))
    p0 = layer.profile()
    (host_s, us), out["host"] = measured(index, lambda: layer.calls(n))
    out["host"] |= {"calls": n, "seconds": round(host_s, 3)}
    out["host"]["us_per_call"] = round(host_s * 1e6 / n, 3)
    out["host"]["command_us_median"] = np.median(us, axis=0).round(2).tolist()
    out["profile_host"] = profile_per_call(p0, layer.profile(), n, mhz)
    log(f"host-fed: {out['host']['us_per_call']} us per call")
    n2 = int(min(args.max_calls, max(2, args.cut_seconds * 1e6 / rate)))
    host_s, _ = layer.calls(n2, exclusive=True)
    out["host_exclusive_us_per_call"] = round(host_s * 1e6 / n2, 3)
    log(f"host-fed, exclusive: {out['host_exclusive_us_per_call']} us per call")
    out["tflops"] = sp.flops(layer.s) / out["host"]["us_per_call"] / 1e6


def main():
    args = parse()
    m, d, h = sp.SHAPES[args.shape]
    out = {"shape": args.shape, "m": m, "d": d, "h": h, "gate_pass": args.gate_pass}
    out["fidelity"] = args.fidelity
    log("inputs")
    x, w1, w3, w2 = ref.swiglu_inputs(m, d, h, args.seed)
    index = visible_card()
    with km.Session(log=log) as session:
        cores = len(session.nocs().words)
        s = sp.make(m, d, h, args.gate_pass, cores)
        if not s.fits():
            raise SystemExit(f"does not fit L1: {s.l1_bytes()} bytes")
        out["plan"] = s.__dict__ | {"l1": s.l1_bytes(), "rows": s.rows()}
        out["x280_mhz"] = session.pll_mhz
        log("uploading")
        t = {
            n: card.bf16_tensor(session, v)
            for n, v in zip(("x", "w1", "w3", "w2"), (x, w1, w3, w2))
        }
        t["y"] = card.bf16_tensor(session, np.zeros((m, d), np.float32))
        t["prof"] = card.prof_tensor(session, cores)
        log("uploaded")
        layer = Layer(session, s, t, args.fidelity)
        out["load_us"] = layer.load_us
        log(f"loaded, group {s.group}, {len(layer.words)} commands per call")
        host_s, _ = layer.calls(args.calls)
        out["first_calls_us"] = round(host_s * 1e6 / args.calls, 3)
        log(f"{args.calls} calls in {host_s:.4f} s")
        if args.seconds:
            timed(layer, args, index, out)
        out["run"] = layer.stop()
        log("stopped")
        rows = ref.sampled_rows(m)
        y_rows = card.read_rows(session, t["y"], rows, log)
    out["check"] = check(y_rows, rows, x, w1, w3, w2)
    out["ok"] = out["check"]["ok"] and out["run"]["ok"]
    out["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
    print(json.dumps(out, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
