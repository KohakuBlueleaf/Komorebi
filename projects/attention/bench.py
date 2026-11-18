"""Time, power, clock and profile of attention on a card, one shape per call. One load:
single commands, commands back to back, the profile; a second: multi-pass commands under
the power sampler, the cut-down runs (optional). Then the check (exit 1 when it fails);
at most 100 commands per load. Usage in README.md."""

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
from projects.attention import card, run  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

K = pl.K
log = run.log
COMMANDS_MAX = 100  # commands one load may submit


class Clock:
    """AICLK of the card (MHz) sampled every 20 ms while the with-block runs."""

    def __init__(self, card_index):
        self.card_index, self.samples, self.stop = card_index, [], threading.Event()

    def sample(self):
        while not self.stop.wait(0.02):
            self.samples.append(aiclk_mhz(self.card_index))

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self):
        samples = self.samples
        return (
            [min(samples), round(float(np.mean(samples)), 1), max(samples)]
            if samples
            else None
        )


SHARES = {  # profile words whose share of the slowest core's time is reported
    "q_reads": "kPrRdQRead",
    "kv_reads": "kPrRdKvRead",
    "sender_waits_for_slot": "kPrRdKvFree",
    "casts": "kPrRdBroadcast",
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


def profile_summary(profile, commands):
    """Shares of the slowest core's time per unit, its busy time per command (cycles),
    and the balance across cores."""
    cycles = profile.astype(np.float64)
    busy = np.maximum.reduce([cycles[:, K[body]] for body in BODIES])
    slowest = int(np.argmax(busy))
    total = busy[slowest]
    return {
        "slowest_core": slowest,
        "slowest_cycles_per_command": round(total / commands),
        "mean_busy_over_slowest": round(float(busy.mean() / total), 3),
        "score_tiles_per_command": int(cycles[slowest, K["kPrPkTiles"]] / commands),
        "slowest_page": [int(word) for word in profile[slowest, : K["kProfWords"]]],
        "partitions_of_slowest": {
            name: round(float(cycles[slowest, K[key]] / total), 3)
            for name, key in SHARES.items()
        },
    }


def back_to_back(attention, commands, mode=K["kModeFull"], passes=1):
    """n commands back to back: microseconds per pass from the host's time."""
    host_s, command_microseconds = attention.calls(commands, mode=mode, passes=passes)
    return (
        round(host_s * 1e6 / (commands * passes), 3),
        float(np.median(command_microseconds)) / passes,
    )


def loop(attention, commands, passes, card_index):
    """n commands of `passes` passes each back to back, power and clock sampled."""
    meter = power.PowerSampler(card_index)
    with meter, Clock(card_index) as clock:
        meter.mark("open")
        time.sleep(1.0)
        meter.mark("loop")
        host_s, command_microseconds = attention.calls(
            commands, passes=passes, timeout=60.0
        )
        meter.mark("after")
    phase = meter.result.phases["loop"]
    calls = commands * passes
    return {
        "commands": commands,
        "passes": passes,
        "seconds": round(host_s, 3),
        "microseconds_per_call": round(host_s * 1e6 / calls, 3),
        "mean_w": phase.mean_w,
        "peak_w": phase.peak_w,
        "open_w": meter.result.phases["open"].mean_w,
        "temp_c": phase.mean_temp_c,
        "joules_per_call": phase.mean_w * host_s / calls if phase.mean_w else None,
        "aiclk_mhz": clock.summary(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=pl.SHAPES, required=True)
    parser.add_argument("--approx", action="store_true", help="approximate exponential")
    parser.add_argument(
        "--broadcast", type=int, default=None, help="broadcast length (default: plan)"
    )
    parser.add_argument(
        "--chunks", type=int, default=None, help="causal chunks per head"
    )
    parser.add_argument("--seconds", type=float, default=4.0, help="power loop length")
    parser.add_argument("--cuts", action="store_true", help="also the cut-down runs")
    parser.add_argument(
        "--noc", type=int, choices=[0, 1], default=0, help="broadcast's NoC"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    (batches, heads, kv_heads, sequence_length, head_dim, causal), call = (
        run.shape_dims(arguments.shape)
    )
    queries, keys, values = sdpa.inputs(
        batches,
        heads,
        kv_heads,
        sequence_length,
        sequence_length,
        head_dim,
        head_dim,
        arguments.seed,
    )
    scale = 1.0 / np.sqrt(head_dim)
    report = {
        "shape": arguments.shape,
        "b": batches,
        "h": heads,
        "hk": kv_heads,
        "s": sequence_length,
        "d": head_dim,
    }
    report["causal"], report["approx"] = causal, arguments.approx
    card_index = visible_card()
    with km.Session(log=log) as session:
        nocs = session.nocs()
        plan = pl.make(
            call, nocs.grid, length=arguments.broadcast, chunks=arguments.chunks
        )
        report["plan"] = plan.summary()
        report["x280_mhz"], report["noc"] = session.pll_mhz, arguments.noc
        tensors = {
            "q": card.bf16_tensor(session, queries),
            "k": card.bf16_tensor(session, keys),
            "v": card.bf16_tensor(session, values),
            "o": card.bf16_tensor(
                session,
                np.zeros((batches, heads, sequence_length, head_dim), np.float32),
            ),
            "tables": card.table_tensor(session, pl.tables(plan)),
            "prof": card.prof_tensor(session, len(nocs.words)),
        }
        # Load 1: single commands, then back to back (5, `batch_commands`): the second
        # batch's time less the first's is the per-command time; the profile.
        attention = card.Attention(
            session, plan, tensors, scale, arguments.approx, noc=arguments.noc
        )
        log(f"loaded, plan {report['plan']}")
        single = np.concatenate([attention.calls(1)[1] for _ in range(6)])
        single_median = float(np.median(single[1:]))
        report["single_microseconds"] = {
            "median": single_median,
            "min": float(single[1:].min()),
        }
        attention.profile(session)  # zeroes the pages
        batch_commands = int(min(60, max(10, math.ceil(0.2e6 / single_median))))
        first_batch_seconds, _ = attention.calls(5)
        host_s, command_microseconds = attention.calls(batch_commands)
        report["microseconds_per_call"] = round(
            (host_s - first_batch_seconds) * 1e6 / (batch_commands - 5), 3
        )
        report["command_microseconds_median"] = float(np.median(command_microseconds))
        report["profile"] = profile_summary(
            attention.profile(session), batch_commands + 5
        )
        report["commands_submitted"] = [6 + 1 + 5 + batch_commands + 1]
        report["runs"] = [attention.stop()]
        batch_per_call = report["microseconds_per_call"]
        log(f"single {single_median:.1f} us, back to back {batch_per_call} us per call")
        # Load 2: the power loop, then the cut-down runs.
        attention = card.Attention(
            session, plan, tensors, scale, arguments.approx, noc=arguments.noc
        )
        per_call = report["microseconds_per_call"]
        passes = max(1, math.ceil(0.1e6 / per_call))
        cut_commands = 11 if arguments.cuts else 0
        commands = math.ceil(arguments.seconds * 1e6 / (per_call * passes))
        commands = max(2, min(commands, COMMANDS_MAX - cut_commands))
        report["loop"] = loop(attention, commands, passes, card_index)
        log(f"loop: {report['loop']}")
        loop_microseconds = report["loop"]["microseconds_per_call"]
        # a prebuild's calls return at once: no time, no rate
        report["tflops"] = (
            pl.flops(call) / loop_microseconds / 1e6 if loop_microseconds else None
        )
        if arguments.cuts:
            attention.profile(session)  # zeroes the loop's
            cut_runs = {}
            for name, mode in (
                ("broadcast_only", K["kModeBroadcastOnly"]),
                ("compute_only", K["kModeComputeOnly"]),
            ):
                cut_runs[name], _ = back_to_back(attention, 4, mode=mode, passes=4)
                cut_runs[name + "_profile"] = profile_summary(
                    attention.profile(session), 4
                )
            report["cuts"] = cut_runs  # neither writes O
        report["commands_submitted"].append(commands + cut_commands)
        assert max(report["commands_submitted"]) <= COMMANDS_MAX, report[
            "commands_submitted"
        ]
        report["runs"].append(attention.stop())
        report["check"] = run.check(
            session, tensors["o"], queries, keys, values, call, causal, arguments.approx
        )
    report["ok"] = report["check"]["ok"] and all(
        run_report["ok"] for run_report in report["runs"]
    )
    report["aiclk_after_mhz"] = aiclk_mhz(card_index)
    report["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(power.clocked(report), default=float), flush=True)
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(report, indent=1, default=float))
    if not report["ok"] and report["tflops"] is None:  # no time: a prebuild's calls
        raise SystemExit("bench.py: no time and no result words (reads were zeros)")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
