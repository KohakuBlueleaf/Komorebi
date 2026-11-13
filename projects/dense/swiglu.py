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
from reference import dense as ref  # noqa: E402

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
    """`plan` loaded on a session's grid with tensors x, w1, w3, w2, y, prof: three
    LOAD commands at the start, then calls of the plan's LAYER commands."""

    def __init__(self, session, plan, tensors, fidelity="HiFi2", deadline=2.0):
        self.plan, self.tensors, self.group = plan, tensors, (0, plan.group)
        self.session = session
        program = card.program(plan, tensors["x"], fidelity)
        self.run = session.load(program, deadline=deadline, wait=1.0)
        addresses = {name: tensor.address for name, tensor in tensors.items()}
        loads = sp.loads(plan, addresses["w1"], addresses["w3"], addresses["w2"])
        handles = [
            self.run.submit("load", self.group, load_words) for load_words in loads
        ]
        self.load_microseconds = [
            event.microseconds for event in self.run.wait(handles, timeout=10.0)
        ]
        self.words = sp.commands(plan, addresses["x"], addresses["y"])
        self.handler = self.run.program.handler("layer")

    def calls(self, count, exclusive=False, timeout=60.0):
        """`count` calls from the host, every command back to back: host seconds from
        the first submit to the last event, and each command's microseconds
        [count, commands]."""
        rows = np.tile(np.array(self.words, np.int64), (count, 1))
        started = time.perf_counter()
        handles = self.run.submit_many(
            "layer", self.group[0], self.group[1], rows, exclusive=exclusive
        )
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [event for event in events if event.kind != "done"]
        if unfinished:
            raise km.KomorebiError(
                f"{len(unfinished)} commands not done: {unfinished[0]}"
            )
        return host_s, np.array([event.microseconds for event in events]).reshape(
            count, -1
        )

    def single(self, count):
        """`count` calls one at a time, each waited for: host microseconds each."""
        return np.array([self.calls(1)[0] * 1e6 for _ in range(count)])

    def profile(self, timeout=10.0):
        """Each core's profile page, summed over the commands so far."""
        profile_tensor = self.tensors["prof"]
        handle = self.run.submit("prof", self.group, [profile_tensor.address])
        self.run.wait(handle, timeout=timeout)
        (pages,) = self.session.read(profile_tensor)
        words = pages.reshape(-1).view(np.uint32).reshape(-1, K["kProfBytes"] // 4)
        return words[: self.group[1], : K["kProfWords"]].astype(np.int64)

    def stop(self):
        """Stops the run: the kernel's result, the cores that ended, the give-ups,
        and `ok` (no kernel error, no give-up, every core's three kernels ended)."""
        self.run.stop()
        states = self.run.state()
        report = {
            "kernel": self.run.result()["error_name"],
            "cores": len(states),
            "cores_ended": {
                key: sum(state[key] != 0 for state in states)
                for key in ("reader_end", "writer_end", "compute_end")
            },
            "gave_up": [
                {key: state[key] for key in ("core", "reader", "writer", "compute")}
                for state in states
                if state["reader"] or state["writer"] or state["compute"]
            ][:8],
        }
        ended = all(
            ended_count == len(states) for ended_count in report["cores_ended"].values()
        )
        report["ok"] = report["kernel"] == "none" and not report["gave_up"] and ended
        return report


def profile_per_call(before, after, calls, mhz):
    """Microseconds per call of each profile share on the slowest core (the one
    with the longest body), from two profiles `calls` calls apart."""
    cycles = (after - before).astype(np.float64) / calls
    busy = np.maximum.reduce([cycles[:, K[key]] for key in BODIES])
    slowest = int(np.argmax(busy))
    return {
        "slowest_rank": slowest,
        "mean_busy_over_slowest": round(float(busy.mean() / busy[slowest]), 3),
        "microseconds": {
            name: round(cycles[slowest, K[key]] / mhz, 3)
            for name, key in SHARES.items()
        },
    }


def measured(index, function):
    """`function()` under the power sampler and the clock: its result and the loop's
    mean and peak power, the open card's power and AICLK (min, mean, max)."""
    meter = power.PowerSampler(index)
    with meter, bench.Clock(index) as clock:
        meter.mark("open")
        time.sleep(1.0)
        meter.mark("loop")
        got = function()
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
    report = ref.errors(y_rows, ref.swiglu(x[rows], w1, w3, w2)) | {"rows": len(rows)}
    report["ok"] = report["rms_rel"] <= RMS_BOUND and report["max_rel"] <= MAX_BOUND
    return report


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=sp.SHAPES, required=True)
    parser.add_argument(
        "--gate-pass", action="store_true", help="the gate a command of its own"
    )
    parser.add_argument("--fidelity", choices=["HiFi2", "HiFi4"], default="HiFi2")
    parser.add_argument("--calls", type=int, default=2, help="calls before the check")
    parser.add_argument("--seconds", type=float, default=0.0, help="then timed loops")
    parser.add_argument("--cut-seconds", type=float, default=1.0)
    parser.add_argument("--max-calls", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def timed(layer, args, index, report):
    """Single calls with the profile around them, then the timed loops."""
    profile_before = layer.profile()
    single = layer.single(20)
    mhz = aiclk_mhz(index)
    report["profile_single"] = profile_per_call(
        profile_before, layer.profile(), 20, mhz
    )
    one_microseconds = float(np.median(single))
    report["single_microseconds"] = {
        "median": one_microseconds,
        "min": float(single.min()),
    }
    log(f"single call {one_microseconds:.2f} us")
    host_s, _ = layer.calls(200)  # back to back: the loop's length from this rate
    rate = host_s * 1e6 / 200
    call_count = int(min(args.max_calls, max(2, math.ceil(args.seconds * 1e6 / rate))))
    profile_before = layer.profile()
    (host_s, command_microseconds), report["host"] = measured(
        index, lambda: layer.calls(call_count)
    )
    report["host"] |= {"calls": call_count, "seconds": round(host_s, 3)}
    report["host"]["microseconds_per_call"] = round(host_s * 1e6 / call_count, 3)
    report["host"]["command_microseconds_median"] = (
        np.median(command_microseconds, axis=0).round(2).tolist()
    )
    report["profile_host"] = profile_per_call(
        profile_before, layer.profile(), call_count, mhz
    )
    log(f"host-fed: {report['host']['microseconds_per_call']} us per call")
    exclusive_calls = int(min(args.max_calls, max(2, args.cut_seconds * 1e6 / rate)))
    host_s, _ = layer.calls(exclusive_calls, exclusive=True)
    report["host_exclusive_microseconds_per_call"] = round(
        host_s * 1e6 / exclusive_calls, 3
    )
    exclusive = report["host_exclusive_microseconds_per_call"]
    log(f"host-fed, exclusive: {exclusive} us per call")
    report["tflops"] = (
        sp.flops(layer.plan) / report["host"]["microseconds_per_call"] / 1e6
    )


def main():
    args = parse()
    m_rows, model_width, hidden_width = sp.SHAPES[args.shape]
    report = {
        "shape": args.shape,
        "m": m_rows,
        "d": model_width,
        "h": hidden_width,
        "gate_pass": args.gate_pass,
    }
    report["fidelity"] = args.fidelity
    log("inputs")
    x, w1, w3, w2 = ref.swiglu_inputs(m_rows, model_width, hidden_width, args.seed)
    index = visible_card()
    with km.Session(log=log) as session:
        cores = len(session.nocs().words)
        plan = sp.make(m_rows, model_width, hidden_width, args.gate_pass, cores)
        if not plan.fits():
            raise SystemExit(f"does not fit L1: {plan.l1_bytes()} bytes")
        report["plan"] = plan.__dict__ | {"l1": plan.l1_bytes(), "rows": plan.rows()}
        report["x280_mhz"] = session.pll_mhz
        log("uploading")
        tensors = {
            name: card.bf16_tensor(session, value)
            for name, value in zip(("x", "w1", "w3", "w2"), (x, w1, w3, w2))
        }
        tensors["y"] = card.bf16_tensor(
            session, np.zeros((m_rows, model_width), np.float32)
        )
        tensors["prof"] = card.prof_tensor(session, cores)
        log("uploaded")
        layer = Layer(session, plan, tensors, args.fidelity)
        report["load_microseconds"] = layer.load_microseconds
        log(f"loaded, group {plan.group}, {len(layer.words)} commands per call")
        host_s, _ = layer.calls(args.calls)
        report["first_calls_microseconds"] = round(host_s * 1e6 / args.calls, 3)
        log(f"{args.calls} calls in {host_s:.4f} s")
        if args.seconds:
            timed(layer, args, index, report)
        report["run"] = layer.stop()
        log("stopped")
        rows = ref.sampled_rows(m_rows)
        y_rows = card.read_rows(session, tensors["y"], rows, log)
    report["check"] = check(y_rows, rows, x, w1, w3, w2)
    report["ok"] = report["check"]["ok"] and report["run"]["ok"]
    report["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
    print(json.dumps(report, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
