"""Time, power, clock and profile of attention on a card, one shape per call. One load:
single records, records back to back, the profile; a second: multi-pass records under
the power sampler, the cut-down runs (optional). Then the check (exit 1 when it fails);
at most 100 records per load. Usage in README.md."""

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
from projects.attention import card, run  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

K = pl.K
log = run.log
RECORDS_MAX = 100  # records one load may issue


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
    "q_reads": "kPrRdQRead",
    "kv_reads": "kPrRdKvRead",
    "sender_waits_for_slot": "kPrRdKvFree",
    "casts": "kPrRdCast",
    "receiver_waits": "kPrRdArrive",
    "unpack_waits_for_q": "kPrUnWaitQ",
    "unpack_waits_for_kv": "kPrUnWaitKv",
    "score": "kPrMaScore",
    "row_max": "kPrMaMax",
    "p_v": "kPrMaPv",
    "pack_waits_for_score": "kPrPkWait",
    "exp": "kPrPkExp",
    "pack_p": "kPrPkPack",
    "pack_output": "kPrPkOut",
    "normalize": "kPrPkFinal",
    "writer_waits": "kPrWrWait",
    "writes": "kPrWrWrite",
}
BODIES = ("kPrRdBody", "kPrUnBody", "kPrMaBody", "kPrPkBody", "kPrWrBody")


def profile_summary(prof, records):
    """Shares of the slowest core's time per unit, its busy time per record (cycles),
    and the balance across cores."""
    pr = prof.astype(np.float64)
    busy = np.maximum.reduce([pr[:, K[k]] for k in BODIES])
    slow = int(np.argmax(busy))
    total = busy[slow]
    return {
        "slowest_core": slow,
        "slowest_cycles_per_record": round(total / records),
        "mean_busy_over_slowest": round(float(busy.mean() / total), 3),
        "score_tiles_per_record": int(pr[slow, K["kPrPkTiles"]] / records),
        "slowest_page": [int(w) for w in prof[slow, : K["kProfWords"]]],
        "shares_of_slowest": {
            name: round(float(pr[slow, K[key]] / total), 3)
            for name, key in SHARES.items()
        },
    }


def back_to_back(att, n, mode=K["kModeFull"], passes=1):
    """n records back to back: microseconds per pass from the host's time."""
    host_s, us = att.calls(n, mode=mode, passes=passes)
    return round(host_s * 1e6 / (n * passes), 3), float(np.median(us)) / passes


def loop(att, n, passes, index):
    """n records of `passes` passes each back to back, power and clock sampled."""
    meter = power.PowerSampler(index)
    with meter, Clock(index) as clock:
        meter.mark("open")
        time.sleep(1.0)
        meter.mark("loop")
        host_s, us = att.calls(n, passes=passes, timeout=60.0)
        meter.mark("after")
    phase = meter.result.phases["loop"]
    calls = n * passes
    return {
        "records": n,
        "passes": passes,
        "seconds": round(host_s, 3),
        "us_per_call": round(host_s * 1e6 / calls, 3),
        "mean_w": phase.mean_w,
        "peak_w": phase.peak_w,
        "open_w": meter.result.phases["open"].mean_w,
        "temp_c": phase.mean_temp_c,
        "joules_per_call": phase.mean_w * host_s / calls if phase.mean_w else None,
        "aiclk_mhz": clock.summary(),
    }


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--shape", choices=pl.SHAPES, required=True)
    a.add_argument("--approx", action="store_true", help="approximate exponential")
    a.add_argument("--lane", type=int, default=None, help="lane length (default: plan)")
    a.add_argument("--chunks", type=int, default=None, help="causal chunks per head")
    a.add_argument("--seconds", type=float, default=4.0, help="power loop length")
    a.add_argument("--cuts", action="store_true", help="also the cut-down runs")
    a.add_argument("--noc", type=int, choices=[0, 1], default=0, help="lane's NoC")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--json", default=None)
    args = a.parse_args()
    (b, h, hk, sq, d, causal), s = run.shape_dims(args.shape)
    q, k, v = sdpa.inputs(b, h, hk, sq, sq, d, d, args.seed)
    scale = 1.0 / np.sqrt(d)
    out = {"shape": args.shape, "b": b, "h": h, "hk": hk, "s": sq, "d": d}
    out["causal"], out["approx"] = causal, args.approx
    index = visible_card()
    with km.Session(log=log) as session:
        nocs = session.nocs()
        p = pl.make(s, nocs.grid, length=args.lane, chunks=args.chunks)
        out["plan"] = p.summary()
        out["x280_mhz"], out["noc"] = session.pll_mhz, args.noc
        device = session.device
        tensors = {
            "q": card.bf16_tensor(device, q),
            "k": card.bf16_tensor(device, k),
            "v": card.bf16_tensor(device, v),
            "o": card.bf16_tensor(device, np.zeros((b, h, sq, d), np.float32)),
            "tables": card.table_tensor(device, pl.tables(p)),
            "prof": card.prof_tensor(device, len(nocs.words)),
        }
        # Load 1: single records, records back to back (5, then `n`: the second
        # batch's time less the first's is the per-record time), the profile.
        att = card.Attention(session, p, tensors, scale, args.approx, noc=args.noc)
        log(f"loaded, plan {out['plan']}")
        single = np.concatenate([att.calls(1)[1] for _ in range(6)])
        one = float(np.median(single[1:]))
        out["single_us"] = {"median": one, "min": float(single[1:].min())}
        att.profile(session)  # zeroes the pages
        n = int(min(60, max(10, math.ceil(0.2e6 / one))))
        first_s, _ = att.calls(5)
        host_s, us = att.calls(n)
        out["us_per_call"] = round((host_s - first_s) * 1e6 / (n - 5), 3)
        out["record_us_median"] = float(np.median(us))
        out["profile"] = profile_summary(att.profile(session), n + 5)
        out["records_issued"] = [6 + 1 + 5 + n + 1]
        out["runs"] = [att.stop()]
        log(f"single {one:.1f} us, back to back {out['us_per_call']} us per call")
        # Load 2: the power loop, then the cut-down runs.
        att = card.Attention(session, p, tensors, scale, args.approx, noc=args.noc)
        each = out["us_per_call"]
        passes = min(p.passes_max(), max(1, math.ceil(0.1e6 / each)))
        cuts = 11 if args.cuts else 0
        records = math.ceil(args.seconds * 1e6 / (each * passes))
        records = max(2, min(records, RECORDS_MAX - cuts))
        out["loop"] = loop(att, records, passes, index)
        log(f"loop: {out['loop']}")
        out["tflops"] = pl.flops(s) / out["loop"]["us_per_call"] / 1e6
        if args.cuts:
            att.profile(session)  # zeroes the loop's
            cut = {}
            for name, mode in (
                ("cast_only", K["kModeCastOnly"]),
                ("compute_only", K["kModeComputeOnly"]),
            ):
                ps = min(p.passes_max(), 4)
                cut[name], _ = back_to_back(att, 4, mode=mode, passes=ps)
                cut[name + "_profile"] = profile_summary(att.profile(session), 4)
            out["cuts"] = cut  # neither writes O
        out["records_issued"].append(records + cuts)
        assert max(out["records_issued"]) <= RECORDS_MAX, out["records_issued"]
        out["runs"].append(att.stop())
        out["check"] = run.check(session, tensors["o"], q, k, v, s, causal, args.approx)
    out["ok"] = out["check"]["ok"] and all(r["ok"] for r in out["runs"])
    out["aiclk_after_mhz"] = aiclk_mhz(index)
    out["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(out, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
